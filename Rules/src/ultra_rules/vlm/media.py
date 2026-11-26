"""One decoder/time base for tracking and model samples (including VFR)."""
from __future__ import annotations

import math
from pathlib import Path

import av


def probe(path: Path):
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        rotation = int(float(stream.metadata.get("rotate", "0"))) % 360
        if rotation:
            raise ValueError("Please normalize video rotation before upload")
        fps = float(stream.average_rate or stream.guessed_rate or 25)
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else float(container.duration or 0) / av.time_base
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Cannot determine video duration")
        first = next(container.decode(stream), None)
        if first is None:
            raise ValueError("Video has no decodable frames")
        if getattr(first, "rotation", 0):
            raise ValueError("Please normalize video rotation before upload")
        return {"width": first.width, "height": first.height, "fps": fps, "durationSec": duration,
                "frameCount": stream.frames, "timeOriginSec": float(first.time or 0), "timeBase": str(stream.time_base)}


TRUNCATION_KEYS = ("truncated", "truncatedAtSec", "truncationReason", "truncationError",
                   "timestampFallbackFrames")


def truncation_info(meta: dict) -> dict:
    return {key: meta[key] for key in TRUNCATION_KEYS if key in meta}


def _frame_timestamp(index: int, frame_time, fps: float, origin: float, last: float):
    """Resolve a frame time without ever going backwards on a missing PTS.

    MPEG-PS/HEVC tails sometimes contain a damaged frame whose decoder output
    has no timestamp. Falling back to ``index / fps`` can be a few milliseconds
    behind the previous PTS near the end of the file; use the previous timestamp
    plus one frame period instead.
    """
    if frame_time is None:
        return (last + 1.0 / fps if last >= 0 else index / fps), "frame_index"
    return float(frame_time) - origin, "pts"


def decoded(path: Path, meta: dict, *, truncate_on_error: bool = False):
    """Decode frames, optionally truncating at the first unusable timestamp.

    ``truncate_on_error=True`` is used by analysis phases: the prefix before a
    real PTS rewind / decoder failure is still analyzable and is marked in
    ``meta`` (``truncated``, ``truncatedAtSec``, ``truncationReason``). Missing
    PTS on an individual frame is repaired with the previous timestamp + one
    frame period and counted in ``timestampFallbackFrames``.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        last = -1.0
        try:
            for index, frame in enumerate(container.decode(stream)):
                if frame.time is None:
                    meta["timestampFallbackFrames"] = meta.get("timestampFallbackFrames", 0) + 1
                timestamp, source = _frame_timestamp(index, frame.time, meta["fps"], meta["timeOriginSec"], last)
                if timestamp < last - 1e-6:
                    if truncate_on_error:
                        meta.update(truncated=True, truncatedAtSec=max(0.0, last),
                                    truncationReason="non_monotonic_timestamp")
                        return
                    raise ValueError("Non-monotonic video timestamps; remux video before analysis")
                last = timestamp
                yield index, timestamp, source, frame
        except Exception as exc:
            if not truncate_on_error:
                raise
            meta.update(truncated=True, truncatedAtSec=max(0.0, last),
                        truncationReason="decode_error", truncationError=str(exc))


def sampled(path: Path, meta: dict, fps: float, *, truncate_on_error: bool = False):
    target = 0.0
    for index, timestamp, source, frame in decoded(path, meta, truncate_on_error=truncate_on_error):
        if timestamp + 1e-6 < target:
            continue
        yield index, timestamp, source, frame
        target = (math.floor(timestamp * fps + 1e-6) + 1) / fps


class SampleSchedule:
    """Re-use one decoded stream at a per-consumer sample rate.

    The acceptance rule is identical to ``sampled()`` so a shared decode pass
    selects exactly the frames each consumer would have decoded on its own.
    """

    __slots__ = ("fps", "target")

    def __init__(self, fps: float):
        self.fps = float(fps)
        self.target = 0.0

    def accept(self, timestamp: float) -> bool:
        if timestamp + 1e-6 < self.target:
            return False
        self.target = (math.floor(timestamp * self.fps + 1e-6) + 1) / self.fps
        return True
