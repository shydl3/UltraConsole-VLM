from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from .contracts import EventDefinition, RunRequest
from .media import decoded
from .storage import write_json


def windows(trigger: float, event: EventDefinition, duration: float):
    start, end = max(0, trigger - event.preSeconds), min(duration, trigger + event.postSeconds)
    if end <= start:
        raise ValueError("Candidate has no video context")
    result = []
    while start < end:
        stop = min(end, start + 16)
        result.append((start, stop))
        if stop == end:
            break
        start = stop - 2
    return result


def choose_rows(rows: list[dict], start: float, end: float, trigger: float, fps: float, maximum: int):
    if not rows:
        raise ValueError("No observations in candidate window")
    times = np.array([row["mediaTimeSec"] for row in rows])
    targets = np.arange(start, end, max(1 / fps, (end - start) / max(1, maximum - 4)))
    # Reserve source boundaries and trigger neighbours before uniform coverage.
    special = [times[0], times[-1], trigger, trigger - 0.15, trigger + 0.15]
    picked = set()
    for t in special:
        if len(picked) >= maximum:
            break
        if start <= t <= end:
            picked.add(int(np.argmin(abs(times - t))))
    for target in targets:
        if len(picked) >= maximum:
            break
        picked.add(int(np.argmin(abs(times - target))))
    return [rows[i] for i in sorted(picked)]


# A keyframe seek can land just after the requested timestamp (leading B-frames
# in an open GOP are dropped), so start a little earlier than the first wanted
# frame and retry with a wider window before giving up. Evidence rows are
# sampled at 0.5-1s intervals, so half a second of drift is still the same
# moment for review purposes.
SEEK_BACKOFFS_SEC = (2.5, 8.0, 30.0, 120.0)
MAX_FRAME_DRIFT_SEC = 0.5


def _row_seek_time(row: dict, meta: dict):
    """Absolute stream timestamp for a row, or None when it has no usable time."""
    if row.get("sourcePtsSec") is not None:
        return float(row["sourcePtsSec"])
    if row.get("mediaTimeSec") is not None and meta.get("timeOriginSec") is not None:
        return float(row["mediaTimeSec"]) + float(meta["timeOriginSec"])
    return None


def _scan_best_frames(container, stream, wanted, backoff: float):
    """Decode one window and keep the closest decoded frame per wanted time."""
    first = min(target for _, target in wanted)
    last = max(target for _, target in wanted)
    container.seek(int(max(0.0, first - backoff) / float(stream.time_base)), stream=stream, backward=True)
    best = {}
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        t = float(frame.time)
        if t > last + 0.001:
            break
        for index, target in wanted:
            diff = abs(t - target)
            current = best.get(index)
            if current is None or diff < current[0]:
                best[index] = (diff, frame)
    return best


def _images_by_time(path: Path, meta: dict, rows: list[dict]):
    """Seek near each wanted frame; None means no usable timestamps were found.

    This deliberately never falls back to decoding from frame 0 here: one
    evidence sample at the end of a multi-hour recording would otherwise decode
    the whole file, turning a per-episode cost into O(video position).
    """
    wanted = []
    for row in rows:
        target = _row_seek_time(row, meta)
        if target is None:
            return None
        wanted.append((row["frameIndex"], target))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for backoff in SEEK_BACKOFFS_SEC:
            best = _scan_best_frames(container, stream, wanted, backoff)
            if len(best) != len(wanted):
                continue
            if max(diff for diff, _ in best.values()) > MAX_FRAME_DRIFT_SEC:
                continue
            images = {}
            while best:
                index, (_diff, frame) = best.popitem()
                images[index] = frame.to_image()
            return images
    return None


def source_images(path: Path, meta: dict, rows: list[dict]):
    """Seek near the wanted timestamps; frame-index decode is the last resort."""
    images = _images_by_time(path, meta, rows)
    if images is not None:
        return images
    wanted = {row["frameIndex"] for row in rows}
    images = {}
    for index, _time, _source, frame in decoded(path, meta):
        if index in wanted:
            images[index] = frame.to_image()
        if index >= max(wanted):
            break
    if len(images) != len(rows):
        raise ValueError("unaligned_frame: source frames differ from observation cache")
    return images


def build_sample(run_dir: Path, config: dict, request: RunRequest, candidate: dict,
                 start: float, end: float, sample_id: str, render: bool | None = None):
    """Build a sample manifest and optionally render its contact sheet.

    ``render=None`` follows ``request.previewMode`` (lazy by default): the
    manifest is enough for the API to render the preview on first access, so
    deterministic evidence does not decode the source window at all. VLM and
    dry-run callers pass ``render=True`` because they need the frames now.
    """
    if render is None:
        render = getattr(request, "previewMode", "eager") == "eager"
    with sqlite3.connect(run_dir / "observations.sqlite") as db:
        rows = [json.loads(row[0]) for row in db.execute(
            "SELECT body FROM observations WHERE time >= ? AND time <= ? ORDER BY time, frame_index", (start, end))]
    selected = choose_rows(rows, start, end, candidate["triggerSec"], request.sampleFps, request.maxFrames)
    w, h = config["video"]["width"], config["video"]["height"]
    scale = min(1, math.sqrt(request.maxPixels / (w * h)))
    width, height = max(32, int(w * scale / 32) * 32), max(32, int(h * scale / 32) * 32)
    images = source_images(Path(config["sourcePath"]), config["video"], selected) if render else {}
    frames, times, summaries, visible, sizes = [], [], [], 0, []
    ambiguous, time_unreliable = candidate["identityAmbiguous"], False
    poi = next(p for p in request.pois if p.id == candidate["poiId"])
    for row in selected:
        im = None
        draw = None
        if render:
            im = images.pop(row["frameIndex"]).resize((width, height), Image.Resampling.BICUBIC)
            draw = ImageDraw.Draw(im)
            polygon = [(x * width, y * height) for x, y in poi.points]
            draw.line(polygon + (polygon[:1] if poi.kind == "zone" else []), fill="#ffdd44", width=1)
        target = next((p for p in row["persons"] if p.get("trackSegmentId") == candidate["trackSegmentId"]), None)
        neighbors = []
        for p in row["persons"]:
            x1, y1, x2, y2 = p["bbox"]
            is_target = p is target
            if render:
                bbox = (x1 * width / w, y1 * height / h, x2 * width / w, y2 * height / h)
                draw.rectangle(bbox, outline="#ff4466" if is_target else "#88bbaa", width=2 if is_target else 1)
                label = f"TARGET #{candidate['trackId']}" if is_target else f"#{p.get('trackId', '?')}"
                label_width = draw.textlength(label)
                draw.text((max(0, min(width - label_width - 2, bbox[0])), max(0, bbox[1] - 12)), label,
                          fill="#ff4466" if is_target else "#88bbaa")
            if not is_target:
                neighbors.append({"trackId": p.get("trackId"),
                                  "bbox": [round(v, 3) for v in (x1 / w, y1 / h, x2 / w, y2 / h)]})
        if target:
            visible += 1
            x1, y1, x2, y2 = target["bbox"]
            sizes.append((y2 - y1) * height / h)
            ambiguous = ambiguous or target.get("identityAmbiguous", False)
            valid = [k for k in target["keypoints"] if k.get("conf") is not None and k["conf"] >= 0.5]
            wrists = {name: {"x": round(target["keypoints"][i]["x"] / w, 3), "y": round(target["keypoints"][i]["y"] / h, 3)}
                      for name, i in (("leftWrist", 9), ("rightWrist", 10))
                      if len(target["keypoints"]) > i and (target["keypoints"][i].get("conf") or 0) >= 0.5}
            summary = {"bbox": [round(v, 3) for v in (x1 / w, y1 / h, x2 / w, y2 / h)],
                       "bboxConf": round(target["bboxConf"], 3), "validKeypoints": len(valid),
                       "trackId": target.get("trackId"), **wrists}
        else:
            summary = None
        times.append(row["mediaTimeSec"])
        time_unreliable = time_unreliable or row["timeSource"] != "pts"
        summaries.append({"timeSec": row["mediaTimeSec"], "target": summary, "otherPeople": neighbors})
        if render:
            frames.append(np.asarray(im))
    original_count = len(selected)
    encoded_count = original_count + (original_count % 2)
    if original_count % 2:
        if render:
            frames.append(frames[-1])
        times.append(times[-1])
    # This installed processor computes timestamps as indices/fps. A millisecond
    # timebase preserves true PTS for VFR too; these are explicitly not source indices.
    metadata = {"fps": 1000.0, "duration": config["video"]["durationSec"],
                "total_num_frames": math.ceil(config["video"]["durationSec"] * 1000),
                "frames_indices": [round(t * 1000) for t in times]}
    quality = {"targetVisibleFraction": visible / original_count, "identityAmbiguous": ambiguous,
               "timeUnreliable": time_unreliable, "contextTruncated": False,
               "medianTargetHeightPx": float(np.median(sizes)) if sizes else 0,
               "uniqueFrameCount": original_count,
               "observationContextMissing": (selected[0]["mediaTimeSec"] - start > max(.3, 2 / request.poseFps)
                                             or end - selected[-1]["mediaTimeSec"] > max(.3, 2 / request.poseFps))}
    manifest = {"sampleId": sample_id, "window": {"startSec": start, "endSec": end},
                "sourceFrameIndices": [r["frameIndex"] for r in selected],
                "sourcePtsSeconds": [r["sourcePtsSec"] for r in selected],
                "sampleTimesSeconds": times, "encodedFrames": encoded_count,
                "padding": encoded_count - original_count,
                "width": width, "height": height, "quality": quality, "observations": summaries,
                "videoMetadata": metadata, "timeMapping": "millisecond_timebase_v1", "poi": poi.model_dump()}
    write_json(run_dir / "samples" / f"{sample_id}.json", manifest)
    if not render:
        return None, metadata, manifest
    preview_path = run_dir / "previews" / f"{sample_id}.jpg"
    preview_path.parent.mkdir(exist_ok=True)
    contact_sheet(frames, times, preview_path)
    return np.stack(frames), metadata, manifest


def render_preview(run_dir: Path, sample_id: str) -> Path:
    """Render a sample contact sheet from its manifest on first request."""
    manifest_path = run_dir / "samples" / f"{sample_id}.json"
    preview_path = run_dir / "previews" / f"{sample_id}.jpg"
    if preview_path.is_file():
        return preview_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    rows = [{"frameIndex": index, "sourcePtsSec": pts, "mediaTimeSec": t}
            for index, pts, t in zip(manifest["sourceFrameIndices"], manifest["sourcePtsSeconds"],
                                     manifest["sampleTimesSeconds"], strict=False)]
    images = source_images(Path(config["sourcePath"]), config["video"], rows)
    width, height = manifest["width"], manifest["height"]
    poi = manifest["poi"]
    frames = []
    for row, observation in zip(rows, manifest["observations"], strict=True):
        im = images.pop(row["frameIndex"]).resize((width, height), Image.Resampling.BICUBIC)
        draw = ImageDraw.Draw(im)
        polygon = [(x * width, y * height) for x, y in poi["points"]]
        draw.line(polygon + (polygon[:1] if poi["kind"] == "zone" else []), fill="#ffdd44", width=1)
        target = observation.get("target")
        for neighbor in observation.get("otherPeople") or []:
            x1, y1, x2, y2 = neighbor["bbox"]
            bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            draw.rectangle(bbox, outline="#88bbaa", width=1)
            label = f"#{neighbor.get('trackId', '?')}"
            draw.text((max(0, min(width - draw.textlength(label) - 2, bbox[0])), max(0, bbox[1] - 12)),
                      label, fill="#88bbaa")
        if target:
            x1, y1, x2, y2 = target["bbox"]
            bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            draw.rectangle(bbox, outline="#ff4466", width=2)
            label = f"TARGET #{target.get('trackId', '?')}"
            draw.text((max(0, min(width - draw.textlength(label) - 2, bbox[0])), max(0, bbox[1] - 12)),
                      label, fill="#ff4466")
        frames.append(np.asarray(im))
    if len(frames) % 2:
        frames.append(frames[-1])
    times = manifest["sampleTimesSeconds"][:len(frames)]
    preview_path.parent.mkdir(exist_ok=True)
    tmp = preview_path.with_name(f"{preview_path.stem}.tmp{preview_path.suffix}")
    contact_sheet(frames, times, tmp)
    os.replace(tmp, preview_path)
    return preview_path


def contact_sheet(frames, times, path):
    indices = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
    sheet = Image.new("RGB", (4 * 320, math.ceil(len(indices) / 4) * 204), "#202020")
    draw = ImageDraw.Draw(sheet)
    for n, index in enumerate(indices):
        im = Image.fromarray(frames[index])
        im.thumbnail((320, 180))
        x, y = n % 4 * 320, n // 4 * 204
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + 184), f"{times[index]:.3f}s", fill="white")
    sheet.save(path)
