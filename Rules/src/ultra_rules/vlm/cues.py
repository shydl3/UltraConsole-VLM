"""Deterministic visual cue detection (e.g. a red LED strip) over a drawn ROI.

Runs in the pose/tracking environment (needs av + numpy + PIL), never in the
Qwen worker. Produces on/off intervals per cue so the deterministic decision
layer can gate the VLM instead of asking a small model to read a thin light.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .contracts import CueDefinition, RunRequest
from .media import sampled, truncation_info
from .storage import digest, progress, read_json, write_json

CUE_VERSION = "cues-v1"


def _roi_mask(cue: CueDefinition, width: int, height: int):
    """Return (mask, bbox) where mask is a boolean array inside the cue polygon."""
    pts = [(int(round(x * width)), int(round(y * height))) for x, y in cue.points]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = max(0, min(xs)), min(width, max(xs) + 1)
    y0, y1 = max(0, min(ys)), min(height, max(ys) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Cue ROI is outside the frame")
    mask_img = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask_img).polygon([(x - x0, y - y0) for x, y in pts], fill=1)
    mask = np.asarray(mask_img, dtype=bool)
    return mask, (x0, y0, x1, y1)


def _red_mask(bgr_crop: np.ndarray, cue: CueDefinition):
    """Boolean mask of pixels matching the red thresholds (full crop shape)."""
    r = bgr_crop[:, :, 2].astype(np.int16)
    g = bgr_crop[:, :, 1].astype(np.int16)
    b = bgr_crop[:, :, 0].astype(np.int16)
    return (r >= cue.minR) & (r - g >= cue.minRG) & (r - b >= cue.minRB)


def _red_ratio(bgr_crop: np.ndarray, mask: np.ndarray, cue: CueDefinition):
    active = _red_mask(bgr_crop, cue) & mask
    total = int(mask.sum())
    return float(active.sum()) / total if total else 0.0


class _Hysteresis:
    """Streaming on/off fold so a shared decode pass needs no sample buffers."""

    def __init__(self, cue: CueDefinition):
        self.cue = cue
        self.intervals: list[list[float]] = []
        self.on_start: float | None = None
        self.last_on: float | None = None
        self.off_start: float | None = None

    def feed(self, t: float, is_on: bool):
        if is_on:
            self.off_start = None
            if self.on_start is None:
                self.on_start = t
            self.last_on = t
        elif self.on_start is not None:
            if self.off_start is None:
                self.off_start = t
            if t - self.off_start >= self.cue.minOffSec:
                if self.last_on - self.on_start >= self.cue.minOnSec:
                    self.intervals.append([round(self.on_start, 3), round(self.last_on, 3)])
                self.on_start = self.last_on = self.off_start = None

    def finish(self):
        if self.on_start is not None and self.last_on - self.on_start >= self.cue.minOnSec:
            self.intervals.append([round(self.on_start, 3), round(self.last_on, 3)])
            self.on_start = self.last_on = self.off_start = None
        return self.intervals


def _hysteresis(samples, cue: CueDefinition):
    """Collapse a per-sample on/off signal into persistent on intervals."""
    fold = _Hysteresis(cue)
    for t, is_on in samples:
        fold.feed(t, is_on)
    return fold.finish()


def detect_timeline(source: Path, meta: dict, cue: CueDefinition):
    mask, bbox = _roi_mask(cue, meta["width"], meta["height"])
    x0, y0, x1, y1 = bbox

    def signal():
        # Stream samples into the hysteresis fold: a 24h pass at 10 fps would keep
        # ~hundreds of thousands of tuples alive if buffered as a list.
        for _index, t, _source, frame in sampled(source, meta, cue.sampleFps, truncate_on_error=True):
            bgr = frame.to_ndarray(format="bgr24")
            crop = bgr[y0:y1, x0:x1]
            if crop.size == 0:
                yield t, False
                continue
            yield t, _red_ratio(crop, mask, cue) >= cue.minOnRatio

    return {"intervals": _hysteresis(signal(), cue)}


def detect_all(source: Path, meta: dict, cues: list[CueDefinition]):
    """Scan once per distinct cue sample rate, not once per cue.

    Every signal zone in the same fps group shares one decode pass: the frame is
    converted to BGR once and each ROI is measured from that array. The default
    10 fps configuration therefore decodes a recording once for all zones.
    """
    if not cues:
        return {}
    prepared = {}
    for cue in cues:
        mask, bbox = _roi_mask(cue, meta["width"], meta["height"])
        prepared[cue.id] = (cue, mask, bbox, _Hysteresis(cue))
    groups: dict[float, list[CueDefinition]] = defaultdict(list)
    for cue in cues:
        groups[float(cue.sampleFps)].append(cue)
    for fps, group in groups.items():
        for _index, t, _source, frame in sampled(source, meta, fps, truncate_on_error=True):
            bgr = frame.to_ndarray(format="bgr24")
            for cue in group:
                _cue, mask, (x0, y0, x1, y1), fold = prepared[cue.id]
                crop = bgr[y0:y1, x0:x1]
                fold.feed(t, crop.size > 0 and _red_ratio(crop, mask, cue) >= cue.minOnRatio)
    return {cue.id: {"intervals": prepared[cue.id][3].finish()} for cue in cues}


def _otsu(values: np.ndarray):
    """Return a suggested red-ratio threshold separating off/on clusters, or None."""
    if values.size < 4:
        return None
    hist, edges = np.histogram(values, bins=64, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2.0
    sum_all = float((hist * centers).sum())
    w_b = 0.0
    sum_b = 0.0
    best = None
    max_var = -1.0
    for i in range(len(hist)):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += hist[i] * centers[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            best = float(centers[i])
    return best


def calibrate(source: Path, meta: dict, cue: CueDefinition, max_samples: int = 40000):
    """Measure red-ratio over time for one cue to help pick detection thresholds."""
    mask, bbox = _roi_mask(cue, meta["width"], meta["height"])
    x0, y0, x1, y1 = bbox
    roi_pixels = int(mask.sum())
    samples: list[list[float]] = []
    ratios: list[float] = []
    peak: dict | None = None
    for _index, t, _source, frame in sampled(source, meta, cue.sampleFps):
        bgr = frame.to_ndarray(format="bgr24")
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            ratio = 0.0
            red = None
        else:
            red = _red_mask(crop, cue) & mask
            ratio = float(red.sum()) / roi_pixels if roi_pixels else 0.0
        samples.append([round(t, 3), round(ratio, 6)])
        ratios.append(ratio)
        if peak is None or ratio > peak["ratio"]:
            peak = {"t": round(t, 3), "ratio": ratio, "crop": crop, "red": red}
        if len(samples) >= max_samples:
            break
    hist, edges = np.histogram(np.asarray(ratios, dtype=float), bins=64, range=(0.0, 1.0))
    result = {
        "roiPixelCount": roi_pixels,
        "sampleCount": len(samples),
        "minRatio": round(float(min(ratios)), 6) if ratios else 0.0,
        "maxRatio": round(float(max(ratios)), 6) if ratios else 0.0,
        "suggestedMinOnRatio": _otsu(np.asarray(ratios, dtype=float)),
        "histogram": {"edges": [round(float(e), 6) for e in edges], "counts": [int(c) for c in hist]},
        "samples": samples,
    }
    if peak is not None and peak.get("crop") is not None and peak.get("red") is not None:
        r = peak["crop"][:, :, 2].astype(np.int16)
        g = peak["crop"][:, :, 1].astype(np.int16)
        b = peak["crop"][:, :, 0].astype(np.int16)
        sel = peak["red"]
        rg = (r - g)[sel]
        rb = (r - b)[sel]
        result["peakFrame"] = {
            "t": peak["t"],
            "ratio": round(peak["ratio"], 6),
            "redPixelCount": int(sel.sum()),
            "rMinusG": {"min": int(rg.min()) if rg.size else None, "mean": round(float(rg.mean()), 1) if rg.size else None},
            "rMinusB": {"min": int(rb.min()) if rb.size else None, "mean": round(float(rb.mean()), 1) if rb.size else None},
        }
    return result


def cache_key(meta: dict, cues: list[CueDefinition]) -> str:
    """Stable cache key shared by the cues-only and combined detect phases."""
    return digest({"version": CUE_VERSION, "source": meta["sha256"],
                   "cues": [cue.model_dump() for cue in cues]})


def detect_run(run_dir: Path):
    config = read_json(run_dir / "request.json")
    request = RunRequest.model_validate(config["request"])
    if not request.cues:
        write_json(run_dir / "cues.json", {"cues": {}})
        progress(run_dir, stage="cues", cueCount=0)
        return
    source = Path(config["sourcePath"])
    meta = config["video"]
    cache = Path(config["cacheRoot"]) / f"{cache_key(meta, request.cues)}.json"
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
    else:
        result = detect_all(source, meta, request.cues)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_json(run_dir / "cues.json", {"cues": result})
    progress(run_dir, stage="cues", cueCount=len(request.cues), **truncation_info(meta))
