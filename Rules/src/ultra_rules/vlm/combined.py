"""One decode pass shared by the signal-cue and integrity consumers.

The original pipeline decoded the full recording once for tracking, once for
cues and once per integrity zone. This phase keeps the same per-consumer
sampling schedules and outputs, but decodes each frame only once.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from collections import deque
from pathlib import Path

from .contracts import RunRequest
from .cues import _Hysteresis, _red_ratio, _roi_mask
from .cues import cache_key as cue_cache_key
from .integrity import _ZoneRuntime
from .integrity import cache_key as integrity_cache_key
from .media import SampleSchedule, decoded
from .observations import (
    OBSERVATION_COMMIT_ROWS,
    OBSERVATION_PROGRESS_ROWS,
    _env_int,
    _FramePipeline,
    _observation_row,
    cache_fingerprint,
    candidate_owners,
    open_observations_db,
    tracker_config_for,
)
from .poi import PoiTracker
from .storage import append_json, digest, progress, read_json, write_json


def _observations_fingerprint(run_dir: Path, zones) -> dict | None:
    if not any(zone.ignoreDetections for zone in zones):
        return None
    path = run_dir / "observations.ref.json"
    return read_json(path).get("fingerprint") if path.exists() else None


def _load(path: Path | None):
    return json.loads(path.read_text(encoding="utf-8")) if path is not None and path.exists() else None


def _copy_sidecar(sidecar: Path | None, previews: Path) -> None:
    if sidecar is None or not sidecar.exists():
        return
    previews.mkdir(exist_ok=True)
    for image in sidecar.glob("*.jpg"):
        target = previews / image.name
        if not target.exists():
            shutil.copy2(image, target)


def detect_run(run_dir: Path) -> None:
    config = read_json(run_dir / "request.json")
    request = RunRequest.model_validate(config["request"])
    source = Path(config["sourcePath"])
    meta = config["video"]
    cache_root = Path(config["cacheRoot"])

    cues = request.cues
    zones = request.integrityZones
    cue_cache = cache_root / f"{cue_cache_key(meta, cues)}.json" if cues else None
    cue_cached = _load(cue_cache)
    cue_needed = bool(cues) and cue_cached is None

    obs_fingerprint = _observations_fingerprint(run_dir, zones) if zones else None
    integrity_cache = cache_root / f"{integrity_cache_key(meta, zones, obs_fingerprint)}.json" if zones else None
    integrity_sidecar = integrity_cache.with_suffix(".integrity") if integrity_cache else None
    integrity_cached = _load(integrity_cache)
    integrity_needed = bool(zones) and integrity_cached is None

    if not cue_needed and not integrity_needed:
        if cues:
            write_json(run_dir / "cues.json", {"cues": cue_cached})
            progress(run_dir, stage="cues", cueCount=len(cues))
        if zones:
            _copy_sidecar(integrity_sidecar, run_dir / "previews")
            write_json(run_dir / "integrity.json", {"zones": integrity_cached})
            progress(run_dir, stage="integrity", zoneCount=len(zones), integrityFraction=1)
        return

    cue_runtimes = {}
    cue_prepared = {}
    if cue_needed:
        for cue in cues:
            mask, bbox = _roi_mask(cue, meta["width"], meta["height"])
            cue_prepared[cue.id] = (mask, bbox)
            cue_runtimes[cue.id] = _Hysteresis(cue)
    cue_schedules = {cue.id: SampleSchedule(cue.sampleFps) for cue in cues if cue_needed}

    zone_runtimes = {}
    if integrity_needed:
        for zone in zones:
            zone_runtimes[zone.id] = _ZoneRuntime(source, zone, meta, run_dir / "observations.sqlite",
                                                  integrity_sidecar)
    zone_schedules = {zone.id: SampleSchedule(zone.sampleFps) for zone in zones if integrity_needed}

    progress(run_dir, stage="detect", cueCount=len(cues), zoneCount=len(zones))
    for _index, t, _source, frame in decoded(source, meta):
        bgr = None
        for cue in cues:
            if not cue_needed or not cue_schedules[cue.id].accept(t):
                continue
            if bgr is None:
                bgr = frame.to_ndarray(format="bgr24")
            mask, (x0, y0, x1, y1) = cue_prepared[cue.id]
            crop = bgr[y0:y1, x0:x1]
            cue_runtimes[cue.id].feed(t, crop.size > 0 and _red_ratio(crop, mask, cue) >= cue.minOnRatio)
        for zone in zones:
            if not integrity_needed or not zone_schedules[zone.id].accept(t):
                continue
            if bgr is None:
                bgr = frame.to_ndarray(format="bgr24")
            zone_runtimes[zone.id].feed(t, bgr)

    if cue_needed:
        cue_result = {cue.id: {"intervals": cue_runtimes[cue.id].finish()} for cue in cues}
        cue_cache.parent.mkdir(parents=True, exist_ok=True)
        cue_cache.write_text(json.dumps(cue_result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    else:
        cue_result = cue_cached

    if integrity_needed:
        integrity_result = {}
        try:
            for zone in zones:
                integrity_result[zone.id] = zone_runtimes[zone.id].finish()
                progress(run_dir, stage="detect", zoneCount=len(zones),
                         integrityFraction=len(integrity_result) / len(zones))
        finally:
            for runtime in zone_runtimes.values():
                runtime.close()
        integrity_cache.parent.mkdir(parents=True, exist_ok=True)
        integrity_cache.write_text(json.dumps(integrity_result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    else:
        integrity_result = integrity_cached

    if cues:
        write_json(run_dir / "cues.json", {"cues": cue_result})
        progress(run_dir, stage="cues", cueCount=len(cues))
    if zones:
        _copy_sidecar(integrity_sidecar, run_dir / "previews")
        write_json(run_dir / "integrity.json", {"zones": integrity_result})
        progress(run_dir, stage="integrity", zoneCount=len(zones), integrityFraction=1)


def _track_and_detect(run_dir: Path, config: dict, request: RunRequest) -> None:
    """Tracking + cues + integrity over one decoded stream.

    The pose consumer keeps the same ``poseFps`` sampling and batching as
    ``observations.track()``; cue/integrity consumers use their own schedules.
    Integrity is evaluated only after enough pose results exist to reproduce
    ``_persons_near`` (nearest observation within ±1s), so occlusion exclusion
    is identical to the phase-separated path.
    """
    import torch
    import yaml

    from ..pose_adapter import PoseAdapter

    source = Path(config["sourcePath"])
    meta = config["video"]
    model = Path(config["poseModelPath"])
    cache_root = Path(config["cacheRoot"])
    cues, zones = request.cues, request.integrityZones

    batch = _env_int("ULTRA_VLM_POSE_BATCH", 1, 1, 64)
    pipeline = os.environ.get("ULTRA_VLM_PIPELINE", "0") == "1"
    if batch > 1:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    tracker_config, ultralytics_version = tracker_config_for(request)
    (run_dir / "tracker.yaml").write_text(yaml.safe_dump(tracker_config))
    fingerprint = cache_fingerprint(request, meta, model, batch, tracker_config, ultralytics_version)
    cache = cache_root / f"{digest(fingerprint)}.jsonl.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        # A concurrent/previous run filled the cache after analyze_run's check.
        from .observations import track as replay_tracking
        replay_tracking(run_dir)
        detect_run(run_dir)
        return

    cue_cache = cache_root / f"{cue_cache_key(meta, cues)}.json" if cues else None
    cue_cached = _load(cue_cache)
    cue_needed = bool(cues) and cue_cached is None
    obs_fp = fingerprint if zones and any(zone.ignoreDetections for zone in zones) else None
    integrity_cache = cache_root / f"{integrity_cache_key(meta, zones, obs_fp)}.json" if zones else None
    integrity_sidecar = integrity_cache.with_suffix(".integrity") if integrity_cache else None
    integrity_cached = _load(integrity_cache)
    integrity_needed = bool(zones) and integrity_cached is None

    tracker = PoiTracker(request)
    db = open_observations_db(run_dir / "observations.sqlite")
    db.execute("CREATE TABLE observations (frame_index INTEGER PRIMARY KEY, time REAL NOT NULL, body TEXT NOT NULL)")
    pending_db: list[tuple] = []
    adapter = None
    pipe = None
    writer = None
    tmp = cache.with_suffix(f".{run_dir.name}.tmp")
    success = False
    cue_runtimes = {}
    cue_prepared = {}
    cue_schedules = {}
    zone_runtimes = {}
    zone_schedules = {}
    try:
        torch.set_num_threads(4)
        adapter = PoseAdapter(str(model), request.poseDevice, request.conf, request.iou, str(run_dir / "tracker.yaml"),
                              batch_size=batch)
        adapter.load()
        if adapter._model.task != "pose":
            raise ValueError("Semantic analysis requires a pose model")
        writer = gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1)

        if cue_needed:
            for cue in cues:
                mask, bbox = _roi_mask(cue, meta["width"], meta["height"])
                cue_prepared[cue.id] = (mask, bbox)
                cue_runtimes[cue.id] = _Hysteresis(cue)
                cue_schedules[cue.id] = SampleSchedule(cue.sampleFps)
        if integrity_needed:
            for zone in zones:
                zone_runtimes[zone.id] = _ZoneRuntime(source, zone, meta, run_dir / "observations.sqlite",
                                                      integrity_sidecar)
                zone_schedules[zone.id] = SampleSchedule(zone.sampleFps)

        progress(run_dir, stage="tracking", sampledFrames=0, candidateCount=0, cacheHit=False,
                 poseBatch=batch, pipeline=pipeline)
        frames = decoded(source, meta)
        if pipeline:
            pipe = _FramePipeline(frames, maxsize=max(8, batch * 2), convert=False)
            items = pipe.rows()
        else:
            items = ((index, t, source_type, frame.width, frame.height,
                      float(frame.time) if frame.time is not None else None, frame)
                     for index, t, source_type, frame in frames)

        pose_schedule = SampleSchedule(request.poseFps)
        pose_batch = []
        pose_history = deque()
        pending_zones = []
        count = candidates = 0
        last_t = 0.0
        started = time.perf_counter()

        def nearest_persons(t: float):
            best = None
            for pose_t, persons in pose_history:
                delta = abs(pose_t - t)
                if delta <= 1.0 and (best is None or delta < best[0]):
                    best = (delta, persons)
            return best[1] if best else []

        def flush_pose_batch():
            nonlocal count, candidates, last_t
            if not pose_batch:
                return
            if batch > 1:
                persons_batch = adapter.infer_batch([item[6] for item in pose_batch])
            else:
                persons_batch = [adapter.infer(item[6]) for item in pose_batch]
            for item, persons in zip(pose_batch, persons_batch, strict=True):
                row = _observation_row(*item[:6], persons)
                # Cache keeps the pre-tracker row; sqlite keeps the post-update
                # row because PoiTracker adds trackSegmentId/identityAmbiguous.
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
                # Integrity expects the same dict shape as observations.sqlite;
                # adapter outputs are pydantic objects until _observation_row dumps them.
                pose_history.append((row["mediaTimeSec"], row["persons"]))
                count += 1
                last_t = row["mediaTimeSec"]
                if len(pending_db) >= OBSERVATION_COMMIT_ROWS:
                    db.executemany("INSERT INTO observations VALUES (?, ?, ?)", pending_db)
                    pending_db.clear()
                    db.commit()
                if count % OBSERVATION_PROGRESS_ROWS == 0:
                    progress(run_dir, sampledFrames=count, candidateCount=candidates, mediaTimeSec=last_t,
                             trackingFraction=min(1, last_t / meta["durationSec"]))
            pose_batch.clear()
            latest = pose_history[-1][0] if pose_history else 0.0
            keep = []
            for runtime, t, bgr in pending_zones:
                if t <= latest - 1.0:
                    runtime.feed(t, bgr, nearest_persons(t))
                else:
                    keep.append((runtime, t, bgr))
            pending_zones[:] = keep
            cutoff = (min(t for _, t, _ in pending_zones) - 1.0) if pending_zones else latest - 3.0
            while pose_history and pose_history[0][0] < cutoff:
                pose_history.popleft()

        for index, t, source_type, width, height, source_pts, frame in items:
            bgr = None
            for cue in cues:
                if not cue_needed or not cue_schedules[cue.id].accept(t):
                    continue
                if bgr is None:
                    bgr = frame.to_ndarray(format="bgr24")
                mask, (x0, y0, x1, y1) = cue_prepared[cue.id]
                crop = bgr[y0:y1, x0:x1]
                cue_runtimes[cue.id].feed(t, crop.size > 0 and _red_ratio(crop, mask, cue) >= cue.minOnRatio)
            for zone in zones:
                if not integrity_needed or not zone_schedules[zone.id].accept(t):
                    continue
                if bgr is None:
                    bgr = frame.to_ndarray(format="bgr24")
                pending_zones.append((zone_runtimes[zone.id], t, bgr))
            if pose_schedule.accept(t):
                if bgr is None:
                    bgr = frame.to_ndarray(format="bgr24")
                pose_batch.append((index, t, source_type, width, height, source_pts, bgr))
                if len(pose_batch) >= batch:
                    flush_pose_batch()

        if pose_batch:
            flush_pose_batch()
        for runtime, t, bgr in pending_zones:
            runtime.feed(t, bgr, nearest_persons(t))
        if not count:
            raise ValueError("No observations decoded")
        if pending_db:
            db.executemany("INSERT INTO observations VALUES (?, ?, ?)", pending_db)
            pending_db.clear()
        db.execute("CREATE INDEX IF NOT EXISTS observation_time ON observations(time)")
        db.commit()
        writer.close()
        writer = None
        os.replace(tmp, cache)
        write_json(run_dir / "observations.ref.json",
                   {"path": str(cache), "fingerprint": fingerprint, "cacheHit": False})
        progress(run_dir, stage="tracking", sampledFrames=count, candidateCount=candidates, mediaTimeSec=last_t,
                 trackingFraction=1, trackingSeconds=time.perf_counter() - started,
                 actualPoseFps=(count - 1) / last_t if last_t else 0)

        if cue_needed:
            cue_result = {cue.id: {"intervals": cue_runtimes[cue.id].finish()} for cue in cues}
            cue_cache.parent.mkdir(parents=True, exist_ok=True)
            cue_cache.write_text(json.dumps(cue_result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        else:
            cue_result = cue_cached
        if integrity_needed:
            integrity_result = {}
            for zone in zones:
                integrity_result[zone.id] = zone_runtimes[zone.id].finish()
                progress(run_dir, stage="detect", zoneCount=len(zones),
                         integrityFraction=len(integrity_result) / len(zones))
            integrity_cache.parent.mkdir(parents=True, exist_ok=True)
            integrity_cache.write_text(json.dumps(integrity_result, ensure_ascii=False, allow_nan=False),
                                       encoding="utf-8")
        else:
            integrity_result = integrity_cached
        if cues:
            write_json(run_dir / "cues.json", {"cues": cue_result})
            progress(run_dir, stage="cues", cueCount=len(cues))
        if zones:
            _copy_sidecar(integrity_sidecar, run_dir / "previews")
            write_json(run_dir / "integrity.json", {"zones": integrity_result})
            progress(run_dir, stage="integrity", zoneCount=len(zones), integrityFraction=1)
        success = True
    finally:
        if pipe is not None:
            pipe.close()
        for runtime in zone_runtimes.values():
            runtime.close()
        db.close()
        if writer is not None:
            writer.close()
        if adapter is not None:
            adapter.close()
        if not success and tmp.exists():
            tmp.unlink()


def analyze_run(run_dir: Path) -> None:
    """Run tracking + cues + integrity with one decode when tracking is not cached."""
    config = read_json(run_dir / "request.json")
    request = RunRequest.model_validate(config["request"])
    meta = config["video"]
    model = Path(config["poseModelPath"])
    batch = _env_int("ULTRA_VLM_POSE_BATCH", 1, 1, 64)
    tracker_config, ultralytics_version = tracker_config_for(request)
    fingerprint = cache_fingerprint(request, meta, model, batch, tracker_config, ultralytics_version)
    cache = Path(config["cacheRoot"]) / f"{digest(fingerprint)}.jsonl.gz"
    observations = run_dir / "observations.sqlite"
    if cache.exists():
        # Replay only when this run has no observation database yet. A retry of
        # the same run already has complete tracking artifacts and must not
        # recreate the sqlite table.
        if not observations.exists():
            from .observations import track
            track(run_dir)
        detect_run(run_dir)
        return
    # A previous failed attempt may have left a partial observation database.
    # The tracking cache is the authoritative completion marker, so a fresh
    # recompute starts from clean per-run artifacts.
    observations.unlink(missing_ok=True)
    (run_dir / "candidates.jsonl").unlink(missing_ok=True)
    _track_and_detect(run_dir, config, request)
