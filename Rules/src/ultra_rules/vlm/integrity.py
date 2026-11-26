"""Deterministic static-scene integrity monitoring over a drawn polygon ROI.

Runs in the pose/tracking environment (needs av + numpy + PIL), never in the
Qwen worker. A per-pixel median baseline is built from the first samples; every
later sample is aligned, illumination-normalized and compared against it on two
channels (gray intensity + edge structure). A change must persist for
``confirmSec`` before it becomes an ``integrity.change`` warning, so people
walking past, lighting drift and camera micro-shake do not alarm. Confirmed
intervals carry before/after/diff evidence crops for the review window.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from .contracts import IntegrityZone, RunRequest
from .cues import _otsu
from .media import sampled
from .storage import digest, progress, read_json, write_json

INTEGRITY_VERSION = "integrity-v3"
ANALYSIS_LONG_SIDE = 256
ALIGN_WIDTH = 320
SCENE_CUT_FRACTION = 0.6


def _gray(bgr: np.ndarray) -> np.ndarray:
    return (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]).astype(np.float32)


def _resize(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((width, height), Image.Resampling.BILINEAR))


def _small_gray(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    width = min(ALIGN_WIDTH, w)
    height = max(8, round(h * width / w))
    return _resize(_gray(bgr).astype(np.uint8), width, height).astype(np.float32)


def _phase_shift(ref: np.ndarray, cur: np.ndarray) -> tuple[int, int] | None:
    """Return (dy, dx) with cur == roll(ref, (dy, dx)), or None when ambiguous."""
    a = np.fft.rfft2(ref - ref.mean())
    b = np.fft.rfft2(cur - cur.mean())
    # conj(F(ref)) * F(cur) peaks at +d when cur == roll(ref, d) (numpy ifft sign).
    cross = np.conj(a) * b
    denom = np.abs(cross)
    denom[denom < 1e-9] = 1e-9
    corr = np.fft.irfft2(cross / denom, s=ref.shape)
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = ref.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    if abs(dy) >= h // 2 or abs(dx) >= w // 2:
        return None
    return int(dy), int(dx)


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = mask.copy()
    h, w = mask.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return out


def _largest_blob(mask: np.ndarray) -> int:
    """Run-length union-find connected components; O(rows x runs), no scipy."""
    parent: list[int] = []
    sizes: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra
        parent[rb] = ra

    prev_runs: list[tuple[int, int, int]] = []
    for row in mask:
        if not row.any():
            prev_runs = []
            continue
        diff = np.diff(np.concatenate(([False], row, [False])).astype(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        runs = []
        for s, e in zip(starts, ends, strict=True):
            idx = len(parent)
            parent.append(idx)
            sizes.append(int(e - s))
            for ps, pe, pid in prev_runs:
                if ps < e and s < pe:
                    union(idx, pid)
            runs.append((int(s), int(e), idx))
        prev_runs = runs
    if not sizes:
        return 0
    totals: dict[int, int] = {}
    for i, size in enumerate(sizes):
        root = find(i)
        totals[root] = totals.get(root, 0) + size
    return max(totals.values())


def _edge_mag(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)


def _polygon_mask(points, width: int, height: int, bbox, ana_w: int, ana_h: int) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    pts = [((x * width - x0) * ana_w / (x1 - x0), (y * height - y0) * ana_h / (y1 - y0)) for x, y in points]
    img = Image.new("L", (ana_w, ana_h), 0)
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


class _Context:
    """Baseline + geometry shared by every sample of one zone."""

    def __init__(self, zone: IntegrityZone, meta: dict, first_bgr: np.ndarray):
        self.zone = zone
        self.width, self.height = meta["width"], meta["height"]
        pts = [(int(round(x * self.width)), int(round(y * self.height))) for x, y in zone.points]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        self.bbox = (max(0, min(xs)), max(0, min(ys)), min(self.width, max(xs) + 1), min(self.height, max(ys) + 1))
        x0, y0, x1, y1 = self.bbox
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Integrity ROI is outside the frame")
        crop_w, crop_h = x1 - x0, y1 - y0
        scale = min(1.0, ANALYSIS_LONG_SIDE / max(crop_w, crop_h))
        self.ana_w = max(8, round(crop_w * scale))
        self.ana_h = max(8, round(crop_h * scale))
        self.mask = _polygon_mask(zone.points, self.width, self.height, self.bbox, self.ana_w, self.ana_h)
        self.roi_pixels = int(self.mask.sum())
        if not self.roi_pixels:
            raise ValueError("Integrity ROI covers no pixels")
        self.ref_small = _small_gray(first_bgr)
        self.first_crop = self.crop(first_bgr)
        self.baseline: np.ndarray | None = None
        self.base_edge: np.ndarray | None = None
        self.base_med = 0.0
        self.base_mad = 1.0

    def crop(self, bgr: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        return _resize(bgr[y0:y1, x0:x1], self.ana_w, self.ana_h)

    def set_baseline(self, crops: list[np.ndarray]) -> None:
        self.baseline = np.median(np.stack(crops), axis=0)
        self.base_edge = _edge_mag(self.baseline)
        values = self.baseline[self.mask]
        self.base_med = float(np.median(values))
        self.base_mad = max(float(np.median(np.abs(values - self.base_med))) * 1.4826, 1e-3)

    def exclusion(self, persons: list[dict]) -> np.ndarray:
        """Pixels covered by tracked persons, mapped into analysis-crop coords."""
        out = np.zeros((self.ana_h, self.ana_w), dtype=bool)
        x0, y0, x1, y1 = self.bbox
        sx, sy = self.ana_w / (x1 - x0), self.ana_h / (y1 - y0)
        for person in persons:
            bx1, by1, bx2, by2 = person["bbox"]
            ax1 = max(0, int((bx1 - x0) * sx))
            ay1 = max(0, int((by1 - y0) * sy))
            ax2 = min(self.ana_w, int(np.ceil((bx2 - x0) * sx)))
            ay2 = min(self.ana_h, int(np.ceil((by2 - y0) * sy)))
            if ax2 > ax1 and ay2 > ay1:
                out[ay1:ay2, ax1:ax2] = True
        return out

    def change_mask(self, gray: np.ndarray, exclude: np.ndarray) -> np.ndarray:
        # Two-pass robust illumination alignment. A raw diff first separates
        # candidate change pixels; illumination stats then come from the
        # still-clean subset so a large local change cannot drag the
        # normalization and hide itself. A near-full-frame raw diff is treated
        # as global lighting drift and normalized over the whole ROI instead.
        thr = self.zone.pixelThreshold
        raw = (np.abs(gray - self.baseline) > thr) & self.mask
        frac = float(raw.sum()) / self.roi_pixels
        stats_mask = self.mask
        if 0.2 < frac <= 0.9:
            clean = self.mask & ~raw
            if clean.sum() >= 0.1 * self.roi_pixels:
                stats_mask = clean
        values = gray[stats_mask]
        cur_med = float(np.median(values))
        cur_mad = max(float(np.median(np.abs(values - cur_med))) * 1.4826, 1e-3)
        scale = float(np.clip(self.base_mad / cur_mad, 0.5, 2.0))
        norm = (gray - cur_med) * scale + self.base_med
        intensity = np.abs(norm - self.baseline) > self.zone.pixelThreshold
        edge = np.abs(_edge_mag(norm) - self.base_edge) > self.zone.pixelThreshold
        w = self.zone.edgeWeight
        combined = ((1 - w) * intensity + w * edge) >= 0.5
        combined &= self.mask & ~exclude
        opened = _dilate(_erode(combined))
        return _erode(_dilate(opened))

    def evaluate(self, bgr: np.ndarray, persons: list[dict]):
        """Return (changed | None, ratio, crop_bgr, diff_mask, aligned_bgr)."""
        small = _small_gray(bgr)
        shift = _phase_shift(self.ref_small, small)
        if shift is None:
            return None, 0.0, None, None, None
        dy, dx = shift
        full_dy = round(dy * self.height / self.ref_small.shape[0])
        full_dx = round(dx * self.width / self.ref_small.shape[1])
        if max(abs(full_dy), abs(full_dx)) > self.zone.alignTolerance * self.width:
            return None, 0.0, None, None, None
        if dy or dx:
            bgr = np.roll(bgr, (-full_dy, -full_dx), axis=(0, 1))
            small = np.roll(small, (-dy, -dx), axis=(0, 1))
        if float((np.abs(small - self.ref_small) > self.zone.pixelThreshold * 1.5).mean()) > SCENE_CUT_FRACTION:
            return None, 0.0, None, None, None
        crop = self.crop(bgr)
        diff = self.change_mask(_gray(crop), self.exclusion(persons))
        ratio = float(diff.sum()) / self.roi_pixels
        blob = _largest_blob(diff) / self.roi_pixels
        changed = ratio >= self.zone.changeRatio and blob >= self.zone.minBlobRatio
        return changed, ratio, crop, diff, bgr


def _persons_near(db: sqlite3.Connection | None, t: float) -> list[dict]:
    if db is None:
        return []
    row = db.execute("SELECT body FROM observations WHERE time BETWEEN ? AND ? ORDER BY ABS(time - ?) LIMIT 1",
                     (t - 1.0, t + 1.0, t)).fetchone()
    return json.loads(row[0]).get("persons", []) if row else []


def _draw_polygon(img: Image.Image, zone: IntegrityZone, bbox, width: int, height: int) -> None:
    x0, y0, x1, y1 = bbox
    sx, sy = img.width / (x1 - x0), img.height / (y1 - y0)
    pts = [((x * width - x0) * sx, (y * height - y0) * sy) for x, y in zone.points]
    ImageDraw.Draw(img).line(pts + pts[:1], fill=(255, 193, 7), width=2)


def _write_evidence(evidence_dir: Path, ctx: _Context, index: int, crop: np.ndarray,
                    diff: np.ndarray, full_bgr: np.ndarray) -> dict:
    zone = ctx.zone
    names = {key: f"integrity-{zone.id}-{index}-{key}.jpg" for key in ("before", "after", "diff", "context")}
    before = Image.fromarray(ctx.first_crop[:, :, ::-1])
    _draw_polygon(before, zone, ctx.bbox, ctx.width, ctx.height)
    before.save(evidence_dir / names["before"], quality=88)
    after = Image.fromarray(crop[:, :, ::-1])
    _draw_polygon(after, zone, ctx.bbox, ctx.width, ctx.height)
    after.save(evidence_dir / names["after"], quality=88)
    overlay = crop[:, :, ::-1].copy()
    overlay[diff] = (0.45 * overlay[diff] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
    Image.fromarray(overlay).save(evidence_dir / names["diff"], quality=88)
    scale = min(1.0, 960 / full_bgr.shape[1])
    ctx_w, ctx_h = max(8, round(full_bgr.shape[1] * scale)), max(8, round(full_bgr.shape[0] * scale))
    context = Image.fromarray(_resize(full_bgr[:, :, ::-1], ctx_w, ctx_h))
    x0, y0, x1, y1 = ctx.bbox
    rx1, ry1 = round(x0 * scale), round(y0 * scale)
    rx2, ry2 = max(rx1 + 1, round(x1 * scale)), max(ry1 + 1, round(y1 * scale))
    region = _resize(overlay, rx2 - rx1, ry2 - ry1)
    context.paste(Image.fromarray(region), (rx1, ry1))
    draw = ImageDraw.Draw(context)
    pts = [(x * ctx_w, y * ctx_h) for x, y in zone.points]
    draw.line(pts + pts[:1], fill=(255, 193, 7), width=2)
    context.save(evidence_dir / names["context"], quality=88)
    return {key: f"previews/{name}" for key, name in names.items()}


class _Persistence:
    """Confirm a change only after it persists confirmSec; close after a clean spell."""

    def __init__(self, zone: IntegrityZone, on_confirm, on_close):
        self.zone = zone
        self.on_confirm = on_confirm
        self.on_close = on_close
        self.intervals: list[dict] = []
        self.run_start: float | None = None
        self.last_changed: float | None = None
        self.off_start: float | None = None
        self.prev_clean: float | None = None
        self._prev_clean: float | None = None
        self.peak = 0.0
        self.confirmed = False

    def push(self, t: float, changed: bool | None, ratio: float, payload) -> None:
        if changed is None:
            return
        if changed:
            self.off_start = None
            if self.run_start is None:
                self.run_start, self.peak, self.confirmed = t, ratio, False
                self._prev_clean = self.prev_clean
            self.peak = max(self.peak, ratio)
            self.last_changed = t
            if not self.confirmed and t - self.run_start >= self.zone.confirmSec:
                self.confirmed = True
                interval = {"onsetSec": round(self.run_start, 3), "confirmedSec": round(t, 3),
                            "offsetSec": None, "peakRatio": round(self.peak, 6),
                            "prevCleanSec": self._prev_clean}
                interval["evidence"] = self.on_confirm(len(self.intervals), payload)
                self.intervals.append(interval)
            elif self.confirmed:
                self.intervals[-1]["peakRatio"] = round(max(self.intervals[-1]["peakRatio"], ratio), 6)
        else:
            self.prev_clean = t
            if self.run_start is not None:
                if self.off_start is None:
                    self.off_start = t
                if t - self.off_start >= self.zone.confirmSec:
                    self._close()

    def _close(self) -> None:
        if self.confirmed and self.intervals:
            self.intervals[-1]["offsetSec"] = round(self.last_changed, 3)
            self.on_close(self.intervals[-1])
        self.run_start = self.last_changed = self.off_start = None
        self.confirmed = False
        self.peak = 0.0

    def finish(self) -> None:
        if self.run_start is not None:
            self._close()


def _frame_at(source: Path, meta: dict, t: float):
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        target = int((t + meta["timeOriginSec"]) / stream.time_base)
        container.seek(max(0, target), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and float(frame.time) - meta["timeOriginSec"] >= t - 1e-6:
                return frame
    return None


def _refine_onset(source: Path, meta: dict, ctx: _Context, db, interval: dict) -> None:
    """Binary-search the true change onset between the last clean sample and onset."""
    lo, hi = interval.get("prevCleanSec"), interval["onsetSec"]
    if lo is None or hi - lo <= 1.5 / ctx.zone.sampleFps:
        return
    for _ in range(5):
        mid = (lo + hi) / 2
        frame = _frame_at(source, meta, mid)
        if frame is None:
            return
        changed, _ratio, *_ = ctx.evaluate(frame.to_ndarray(format="bgr24"), _persons_near(db, mid))
        if changed is None:
            return
        if changed:
            hi = mid
        else:
            lo = mid
    interval["onsetSec"] = round(hi, 3)


def _open_db(observations: Path | None, zone: IntegrityZone):
    if observations is None or not zone.ignoreDetections or not observations.exists():
        return None
    return sqlite3.connect(observations)


class _ZoneRuntime:
    """Streaming integrity detector fed by a shared decode pass."""

    def __init__(self, source: Path, zone: IntegrityZone, meta: dict,
                 observations: Path | None = None, evidence_dir: Path | None = None):
        self.source = source
        self.zone = zone
        self.meta = meta
        self.db = _open_db(observations, zone)
        self.evidence_dir = evidence_dir
        self.ctx: _Context | None = None
        self.baseline_crops: list[np.ndarray] = []
        self.baseline_sec = 0.0
        self.sample_count = 0
        self.unreliable = 0
        if evidence_dir is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
        self.fold = _Persistence(zone, self._on_confirm, lambda interval: None)

    def _on_confirm(self, index: int, payload):
        if self.evidence_dir is None or payload is None:
            return {}
        crop, diff, full_bgr = payload
        return _write_evidence(self.evidence_dir, self.ctx, index, crop, diff, full_bgr)

    def feed(self, t: float, bgr: np.ndarray, persons: list[dict] | None = None):
        self.sample_count += 1
        if self.ctx is None:
            self.ctx = _Context(self.zone, self.meta, bgr)
        if len(self.baseline_crops) < self.zone.baselineFrames:
            small = _small_gray(bgr)
            shift = _phase_shift(self.ctx.ref_small, small)
            if shift is not None:
                dy, dx = shift
                if dy or dx:
                    bgr = np.roll(bgr, (-round(dy * self.ctx.height / small.shape[0]),
                                        -round(dx * self.ctx.width / small.shape[1])), axis=(0, 1))
                self.baseline_crops.append(_gray(self.ctx.crop(bgr)))
                self.baseline_sec = t
            if len(self.baseline_crops) < self.zone.baselineFrames:
                return
            self.ctx.set_baseline(self.baseline_crops)
        if persons is None:
            persons = _persons_near(self.db, t)
        changed, ratio, crop, diff, aligned = self.ctx.evaluate(bgr, persons)
        if changed is None:
            self.unreliable += 1
        self.fold.push(t, changed, ratio, (crop, diff, aligned) if changed else None)

    def finish(self) -> dict:
        self.fold.finish()
        for interval in self.fold.intervals:
            _refine_onset(self.source, self.meta, self.ctx, self.db, interval)
            interval.pop("prevCleanSec", None)
        return {"intervals": self.fold.intervals, "sampleCount": self.sample_count,
                "unreliableCount": self.unreliable, "baselineSec": round(self.baseline_sec, 3),
                "roiPixelCount": self.ctx.roi_pixels if self.ctx else 0}

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None


def detect_timeline(source: Path, meta: dict, zone: IntegrityZone,
                    observations: Path | None = None, evidence_dir: Path | None = None) -> dict:
    runtime = _ZoneRuntime(source, zone, meta, observations, evidence_dir)
    try:
        for _index, t, _source, frame in sampled(source, meta, zone.sampleFps):
            runtime.feed(t, frame.to_ndarray(format="bgr24"))
        return runtime.finish()
    finally:
        runtime.close()


def calibrate(source: Path, meta: dict, zone: IntegrityZone, max_samples: int = 600) -> dict:
    """Measure changeRatio over time to help pick thresholds for one zone."""
    ctx: _Context | None = None
    baseline_crops: list[np.ndarray] = []
    samples: list[list[float]] = []
    ratios: list[float] = []
    for _index, t, _source, frame in sampled(source, meta, zone.sampleFps):
        bgr = frame.to_ndarray(format="bgr24")
        if ctx is None:
            ctx = _Context(zone, meta, bgr)
        if len(baseline_crops) < zone.baselineFrames:
            baseline_crops.append(_gray(ctx.crop(bgr)))
            if len(baseline_crops) < zone.baselineFrames:
                continue
            ctx.set_baseline(baseline_crops)
        _changed, ratio, *_ = ctx.evaluate(bgr, [])
        samples.append([round(t, 3), round(ratio, 6)])
        ratios.append(ratio)
        if len(samples) >= max_samples:
            break
    hist, edges = np.histogram(np.asarray(ratios, dtype=float), bins=64, range=(0.0, 1.0))
    return {"roiPixelCount": ctx.roi_pixels if ctx else 0, "sampleCount": len(samples),
            "minRatio": round(float(min(ratios)), 6) if ratios else 0.0,
            "maxRatio": round(float(max(ratios)), 6) if ratios else 0.0,
            "suggestedChangeRatio": _otsu(np.asarray(ratios, dtype=float)),
            "histogram": {"edges": [round(float(e), 6) for e in edges], "counts": [int(c) for c in hist]},
            "samples": samples}


def cache_key(meta: dict, zones: list[IntegrityZone], obs_fingerprint: dict | None) -> str:
    """Stable cache key shared by the integrity-only and combined detect phases."""
    return digest({"version": INTEGRITY_VERSION, "source": meta["sha256"],
                   "zones": [zone.model_dump() for zone in zones],
                   "observations": obs_fingerprint})


def detect_run(run_dir: Path) -> None:
    config = read_json(run_dir / "request.json")
    request = RunRequest.model_validate(config["request"])
    if not request.integrityZones:
        write_json(run_dir / "integrity.json", {"zones": {}})
        progress(run_dir, stage="integrity", zoneCount=0, integrityFraction=1)
        return
    source = Path(config["sourcePath"])
    meta = config["video"]
    observations = run_dir / "observations.sqlite"
    obs_fingerprint = None
    if any(zone.ignoreDetections for zone in request.integrityZones) and (run_dir / "observations.ref.json").exists():
        obs_fingerprint = read_json(run_dir / "observations.ref.json").get("fingerprint")
    key = cache_key(meta, request.integrityZones, obs_fingerprint)
    cache_root = Path(config["cacheRoot"])
    cache = cache_root / f"{key}.json"
    sidecar = cache_root / f"{key}.integrity"
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
    else:
        sidecar.mkdir(parents=True, exist_ok=True)
        result = {}
        for zone in request.integrityZones:
            result[zone.id] = detect_timeline(source, meta, zone, observations, sidecar)
            progress(run_dir, stage="integrity", zoneCount=len(request.integrityZones),
                     integrityFraction=len(result) / len(request.integrityZones))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    previews = run_dir / "previews"
    previews.mkdir(exist_ok=True)
    if sidecar.exists():
        for image in sidecar.glob("*.jpg"):
            target = previews / image.name
            if not target.exists():
                shutil.copy2(image, target)
    write_json(run_dir / "integrity.json", {"zones": result})
    progress(run_dir, stage="integrity", zoneCount=len(request.integrityZones), integrityFraction=1)
