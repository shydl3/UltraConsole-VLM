"""PT0 基线测量与一致性回归（dev-guide 6.4.1 / 6.4.3）。

用法（在项目根目录，PYTHONPATH=Rules/src）：
  # 串行基线（默认开关）
  python Rules/scripts/bench_tracking.py run --input <视频> --name serial
  # 批量 + 流水线
  python Rules/scripts/bench_tracking.py run --input <视频> --name fast --batch 8 --pipeline
  # 逐帧一致性回归：trackId/框数/候选必须完全一致；bbox 硬门禁默认 1px
  # （批量前向的 cuDNN 数值微差，本轮实测上界 0.66px）；关键点为 advisory 量，
  # 默认容差 5px（本轮实测上界 3.8px），超限仅警告不判败。
  python Rules/scripts/bench_tracking.py diff --a serial --b fast

run 产物落在 --data-root（默认 tmp/bench-tracking）下的 <name>/：
observations.sqlite、candidates.jsonl、progress.json、observations_dump.jsonl。
--limit-sec 会先无损 remux 出前 N 秒片段再测（快速迭代用）。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Rules" / "src"))

from ultra_rules.vlm.contracts import EventDefinition, Poi, RunRequest  # noqa: E402
from ultra_rules.vlm.media import probe  # noqa: E402
from ultra_rules.vlm.storage import file_hash, read_json, write_json  # noqa: E402


def make_clip(source: Path, dest: Path, limit_sec: float) -> None:
    import av

    with av.open(str(source)) as inc, av.open(str(dest), "w") as out:
        video = inc.streams.video[0]
        added = out.add_stream_from_template(video)
        for packet in inc.demux(video):
            if packet.dts is None:
                continue
            if packet.pts is not None and float(packet.pts * video.time_base) > limit_sec:
                break
            packet.stream = added
            out.mux(packet)


def cmd_run(args) -> None:
    from ultra_rules.vlm import observations

    data_root = Path(args.data_root).resolve()
    run_dir = data_root / args.name
    if run_dir.exists():
        raise SystemExit(f"run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    source = Path(args.input).resolve()
    if args.limit_sec:
        clip = run_dir / f"clip-{args.limit_sec:g}s.mp4"
        make_clip(source, clip, args.limit_sec)
        source = clip
    meta = probe(source)
    meta.update(sha256=file_hash(source), fileName=source.name)

    request = RunRequest(
        videoId=meta["sha256"][:24],
        poseFps=args.pose_fps,
        pois=[Poi(id="gate", kind="line", points=[(0.15, 0.55), (0.85, 0.55)])],
        events=[EventDefinition(id="cross", description="person crosses the gate line",
                                triggerKinds=["line.cross"], poiIds=["gate"])],
    )
    write_json(run_dir / "request.json", {
        "request": request.model_dump(),
        "sourcePath": str(source),
        "video": meta,
        "poseModelPath": str(ROOT / "backend" / "models" / "yolo11m-pose.pt"),
        # --fresh：独立缓存目录，强制重算（归因测量用；默认共享缓存以复用观测）。
        "cacheRoot": str(run_dir / "cache-local" if args.fresh else data_root / "cache"),
    })
    os.environ["ULTRA_VLM_POSE_BATCH"] = str(args.batch)
    os.environ["ULTRA_VLM_PIPELINE"] = "1" if args.pipeline else "0"
    if args.fp32:
        # 数值一致性实验：禁 TF32 + cuDNN 确定性，检验批量前向的 ID 漂移是否收敛。
        import torch

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    observations.track(run_dir)

    with sqlite3.connect(run_dir / "observations.sqlite") as db:
        with (run_dir / "observations_dump.jsonl").open("w", encoding="utf-8") as dump:
            for (body,) in db.execute("SELECT body FROM observations ORDER BY frame_index"):
                dump.write(body + "\n")
    progress = read_json(run_dir / "progress.json")
    summary = {"name": args.name, "batch": args.batch, "pipeline": args.pipeline,
               "durationSec": meta["durationSec"], "sampledFrames": progress.get("sampledFrames"),
               "trackingSeconds": progress.get("trackingSeconds"),
               "actualPoseFps": progress.get("actualPoseFps"),
               "candidateCount": progress.get("candidateCount"), "cacheHit": progress.get("cacheHit")}
    write_json(run_dir / "bench_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_rows(run_dir: Path) -> dict[int, dict]:
    path = run_dir / "observations_dump.jsonl"
    return {row["frameIndex"]: row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def cmd_diff(args) -> None:
    a, b = _load_rows(Path(args.a)), _load_rows(Path(args.b))
    if a.keys() != b.keys():
        print(f"FAIL: frame sets differ (a={len(a)}, b={len(b)}, "
              f"only_a={len(a.keys() - b.keys())}, only_b={len(b.keys() - a.keys())})")
        raise SystemExit(1)
    bad_id = bad_count = 0
    max_bbox = max_kp = 0.0
    for idx in sorted(a):
        pa, pb = a[idx]["persons"], b[idx]["persons"]
        if a[idx]["mediaTimeSec"] != b[idx]["mediaTimeSec"]:
            print(f"FAIL: mediaTimeSec differs at frame {idx}")
            raise SystemExit(1)
        if len(pa) != len(pb):
            bad_count += 1
            continue
        for xa, xb in zip(pa, pb, strict=True):
            if xa["trackId"] != xb["trackId"] or xa["cls"] != xb["cls"]:
                bad_id += 1
            max_bbox = max(max_bbox, max(abs(u - v) for u, v in zip(xa["bbox"], xb["bbox"], strict=True)))
            ka, kb = xa["keypoints"], xb["keypoints"]
            if len(ka) != len(kb):
                bad_count += 1
                continue
            for ja, jb in zip(ka, kb, strict=True):
                max_kp = max(max_kp, abs(ja["x"] - jb["x"]), abs(ja["y"] - jb["y"]))
    ok = bad_id == 0 and bad_count == 0 and max_bbox <= args.bbox_tol
    kp_warn = max_kp > args.kp_tol
    print(json.dumps({"frames": len(a), "trackId_mismatches": bad_id, "person_count_mismatches": bad_count,
                      "max_bbox_delta_px": round(max_bbox, 4), "max_keypoint_delta_px": round(max_kp, 4),
                      "bbox_tol": args.bbox_tol, "kp_advisory_warn": kp_warn, "consistent": ok},
                     ensure_ascii=False, indent=2))
    ca = (Path(args.a) / "candidates.jsonl")
    cb = (Path(args.b) / "candidates.jsonl")
    if ca.exists() and cb.exists():
        na = sum(1 for line in ca.open() if line.strip())
        nb = sum(1 for line in cb.open() if line.strip())
        print(f"candidates: a={na} b={nb} equal={na == nb}")
        ok = ok and na == nb
    raise SystemExit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--name", required=True)
    run.add_argument("--batch", type=int, default=1)
    run.add_argument("--pipeline", action="store_true")
    run.add_argument("--pose-fps", type=float, default=10)
    run.add_argument("--limit-sec", type=float, default=0)
    run.add_argument("--fresh", action="store_true", help="独立缓存目录，强制重算")
    run.add_argument("--fp32", action="store_true", help="禁 TF32 + cuDNN 确定性（数值一致性实验）")
    run.add_argument("--data-root", default=str(ROOT / "tmp" / "bench-tracking"))
    diff = sub.add_parser("diff")
    diff.add_argument("--a", required=True, help="基线 run 目录")
    diff.add_argument("--b", required=True, help="待验证 run 目录")
    diff.add_argument("--bbox-tol", type=float, default=1.0)
    diff.add_argument("--kp-tol", type=float, default=5.0, help="关键点 advisory 容差，超限仅警告")
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    else:
        cmd_diff(args)


if __name__ == "__main__":
    main()