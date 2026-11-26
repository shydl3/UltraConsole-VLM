from __future__ import annotations

import gzip
import json
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path

from .contracts import RunRequest
from .media import TRUNCATION_KEYS, sampled, truncation_info
from .poi import PoiTracker
from .storage import VERSION, append_json, digest, file_hash, progress, read_json, write_json


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.environ.get(name, "") or default)))
    except ValueError:
        return default


def _chunks(items, size: int):
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# observations.sqlite is a transient artifact: the gzip cache is the completion
# marker and a partial DB is deleted before a retry. Trading durability for
# insert speed is therefore safe, and batching commits avoids thousands of fsyncs
# on multi-hour recordings.
OBSERVATION_COMMIT_ROWS = 200
OBSERVATION_PROGRESS_ROWS = 50


def open_observations_db(path: Path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=MEMORY")
    db.execute("PRAGMA synchronous=OFF")
    return db


SUPPORT_TRIGGERS = ("line.cross", "zone.enter", "zone.leave", "initial_inside")


def candidate_owners(candidate: dict, events) -> tuple[list[str], list[str]]:
    """Split rules into direct owners and signal-triggered support owners.

    Signal rules never fire on POI candidates, but they need those crossings and
    entries persisted to pick the target person and to detect a retreat. Support
    rows keep ``eventDefinitionIds`` empty so they are not judged twice.
    """
    applicable = [e.id for e in events if e.enabled and candidate["triggerKind"] in e.triggerKinds
                  and (not e.poiIds or candidate["poiId"] in e.poiIds)]
    support = [e.id for e in events if e.enabled and e.category == "forced_passage"
               and candidate["triggerKind"] in SUPPORT_TRIGGERS
               and (not e.poiIds or candidate["poiId"] in e.poiIds)]
    return applicable, support


def _observation_row(index, t, source_type, width, height, source_pts, persons):
    return {"frameIndex": index, "mediaTimeSec": t, "sourcePtsSec": source_pts,
            "timeSource": source_type, "width": width, "height": height,
            "persons": [p.model_dump(by_alias=True) for p in persons]}


class _FramePipeline:
    """PT0-2：解码线程 → 有界队列 → 推理消费者。

    生产线程做解码 + bgr24 转换，使 CPU 段与 GPU 推理重叠；有界队列
    提供背压并封顶内存（1080p 单帧约 6MB）。FIFO 保证帧序；生产端异常
    在消费端重新抛出。close() 后生产线程自行退出（daemon 兜底）。

    ``convert=False`` 时队列携带原始 VideoFrame，让需要多处消费同一帧的
    调用方只在实际取用时转换一次 BGR。
    """

    def __init__(self, frames, maxsize: int, convert: bool = True):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._error: list[BaseException] = []
        self._convert = convert
        self._thread = threading.Thread(target=self._produce, args=(frames,), daemon=True)
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self, frames):
        try:
            for index, t, source_type, frame in frames:
                item = (index, t, source_type, frame.width, frame.height,
                        float(frame.time) if frame.time is not None else None,
                        frame.to_ndarray(format="bgr24") if self._convert else frame)
                if not self._put(item):
                    return
        except BaseException as exc:
            self._error.append(exc)
        finally:
            self._put(None)

    def rows(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item
        if self._error:
            raise self._error[0]

    def close(self):
        self._stop.set()
        # 排空队列，让阻塞在 put 的生产线程观察到 stop 标志后退出。
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def tracker_config_for(request):
    """Tracker YAML shared by the tracking-only and combined detect phases."""
    import ultralytics
    import yaml

    config = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg/trackers/bytetrack.yaml").read_text())
    config["track_buffer"] = max(1, round(request.poseFps * request.maxGapSeconds))
    return config, ultralytics.__version__


def cache_fingerprint(request, meta, model, batch, tracker_config, ultralytics_version):
    """Observation cache identity; combined tracking must produce the same key."""
    return {"version": VERSION, "source": meta["sha256"], "model": file_hash(model),
            "poseFps": request.poseFps, "conf": request.conf, "iou": request.iou,
            "tracker": tracker_config, "ultralytics": ultralytics_version,
            "device": request.poseDevice, "timePolicy": "pyav-origin-scheduled-v1",
            # 批量前向可能引入数值微差，缓存按推理模式隔离；
            # pipeline 不改变结果，不入指纹（开关间共享缓存）。
            "poseBatch": batch, "poseInfer": "batch-bytetrack" if batch > 1 else "ultralytics-track",
            "posePrecision": "fp32-deterministic" if batch > 1 else "default"}


def track(run_dir: Path):
    # Imported only inside the YOLO process, never by the Qwen worker/API schema.
    import torch
    import yaml

    from ..pose_adapter import PoseAdapter

    config = read_json(run_dir / "request.json")
    request = RunRequest.model_validate(config["request"])
    source, meta = Path(config["sourcePath"]), config["video"]
    model = Path(config["poseModelPath"])
    # PT0 开关（默认保守串行）：批大小与解码/推理重叠独立可控。
    batch = _env_int("ULTRA_VLM_POSE_BATCH", 1, 1, 64)
    pipeline = os.environ.get("ULTRA_VLM_PIPELINE", "0") == "1"
    if batch > 1:
        # 批量前向在 TF32 下因 batch 形状不同产生 rounding 微差，长视频会被
        # Kalman/关联放大成 trackId 切换（本轮实测 15000 帧中 3623 帧漂移）；
        # 禁 TF32 + cuDNN 确定性后漂移收敛为 0（实测 bbox 差 ≤0.0005px）。
        # 因此批量模式强制 FP32 确定性，语义与同精度串行基线一致。
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    tracker_config, ultralytics_version = tracker_config_for(request)
    tracker_path = run_dir / "tracker.yaml"
    tracker_path.write_text(yaml.safe_dump(tracker_config))
    fingerprint = cache_fingerprint(request, meta, model, batch, tracker_config, ultralytics_version)
    key = digest(fingerprint)
    cache = Path(config["cacheRoot"]) / f"{key}.jsonl.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache_hit = cache.exists()
    if cache_hit:
        ref_path = run_dir / "observations.ref.json"
        if ref_path.exists():
            cached = read_json(ref_path)
            for field in TRUNCATION_KEYS:
                if field in cached:
                    meta[field] = cached[field]
    progress(run_dir, stage="tracking", sampledFrames=0, candidateCount=0, cacheHit=cache_hit,
             poseBatch=batch, pipeline=pipeline)
    tracker = PoiTracker(request)
    db = open_observations_db(run_dir / "observations.sqlite")
    db.execute("CREATE TABLE observations (frame_index INTEGER PRIMARY KEY, time REAL NOT NULL, body TEXT NOT NULL)")
    adapter = None

    def generate():
        nonlocal adapter
        torch.set_num_threads(4)
        adapter = PoseAdapter(str(model), request.poseDevice, request.conf, request.iou, str(tracker_path),
                              batch_size=batch)
        adapter.load()
        if adapter._model.task != "pose":
            raise ValueError("Semantic analysis requires a pose model")
        frames = sampled(source, meta, request.poseFps, truncate_on_error=True)
        pipe = _FramePipeline(frames, maxsize=max(8, batch * 2)) if pipeline else None
        if pipe is not None:
            items = pipe.rows()
        else:
            items = ((index, t, source_type, frame.width, frame.height,
                      float(frame.time) if frame.time is not None else None,
                      frame.to_ndarray(format="bgr24"))
                     for index, t, source_type, frame in frames)
        try:
            if batch > 1:
                for chunk in _chunks(items, batch):
                    persons_batch = adapter.infer_batch([item[6] for item in chunk])
                    for item, persons in zip(chunk, persons_batch, strict=True):
                        yield _observation_row(*item[:6], persons)
            else:
                for index, t, source_type, width, height, source_pts, bgr in items:
                    persons = adapter.infer(bgr)
                    yield _observation_row(index, t, source_type, width, height, source_pts, persons)
        finally:
            if pipe is not None:
                pipe.close()

    tmp = cache.with_suffix(f".{run_dir.name}.tmp")
    reader = gzip.open(cache, "rt", encoding="utf-8") if cache_hit else None
    writer = gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) if not cache_hit else None
    count, candidates, last_t = 0, 0, 0.0
    pending_db: list[tuple] = []
    started = time.perf_counter()
    try:
        rows = (json.loads(line) for line in reader) if reader else generate()
        for row in rows:
            # The cache stores the pre-tracker row; observations.sqlite stores the
            # post-update row (PoiTracker adds trackSegmentId/identityAmbiguous
            # in place, and build_sample relies on those fields).
            if writer:
                writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            for candidate in tracker.update(row):
                applicable, support = candidate_owners(candidate, request.events)
                if applicable or support:
                    candidate["eventDefinitionIds"] = applicable
                    if support:
                        candidate["supportRuleIds"] = support
                    append_json(run_dir / "candidates.jsonl", candidate)
                    candidates += 1
            body = json.dumps(row, ensure_ascii=False, allow_nan=False)
            pending_db.append((row["frameIndex"], row["mediaTimeSec"], body))
            count += 1
            last_t = row["mediaTimeSec"]
            if len(pending_db) >= OBSERVATION_COMMIT_ROWS:
                db.executemany("INSERT INTO observations VALUES (?, ?, ?)", pending_db)
                pending_db.clear()
                db.commit()
            if count % OBSERVATION_PROGRESS_ROWS == 0:
                progress(run_dir, sampledFrames=count, candidateCount=candidates, mediaTimeSec=last_t,
                         trackingFraction=min(1, last_t / meta["durationSec"]))
        if pending_db:
            db.executemany("INSERT INTO observations VALUES (?, ?, ?)", pending_db)
            pending_db.clear()
        db.execute("CREATE INDEX IF NOT EXISTS observation_time ON observations(time)")
        db.commit()
        if not count:
            raise ValueError("No observations decoded")
        if writer:
            writer.close()
            os.replace(tmp, cache)
        truncation = truncation_info(meta)
        write_json(run_dir / "observations.ref.json", {"path": str(cache), "fingerprint": fingerprint,
                                                       "cacheHit": cache_hit, **truncation})
        progress(run_dir, sampledFrames=count, candidateCount=candidates, mediaTimeSec=last_t,
                 trackingFraction=1, trackingSeconds=time.perf_counter() - started,
                 actualPoseFps=(count - 1) / last_t if last_t else 0, **truncation)
    finally:
        db.close()
        if reader:
            reader.close()
        if writer:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        if adapter:
            adapter.close()
