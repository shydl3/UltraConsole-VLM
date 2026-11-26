"""Deterministic event verdicts for cue-bound rules.

Runs in the Qwen worker process but only reads precomputed cue timelines and
pose observations; it never decodes video or touches the model. When a rule
binds cues (event.cueIds), the decision is derived from measurable predicates
and the VLM is consulted only for uncertain cases.
"""
from __future__ import annotations

import json
import sqlite3
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

from .contracts import EventDefinition, RunRequest
from .storage import read_json

RETREAT_SECONDS = 5.0


def retreat_candidate_ids(candidates: list[dict]) -> set[str]:
    """The first opposite line.cross within RETREAT_SECONDS is a retreat, not a new entry."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in candidates:
        if c.get("triggerKind") == "line.cross":
            groups[(c.get("trackSegmentId"), c.get("poiId"))].append(c)
    skipped: set[str] = set()
    for group in groups.values():
        group.sort(key=lambda c: c["triggerSec"])
        pending = None
        for c in group:
            if pending is None:
                pending = c
                continue
            if c.get("direction") != pending.get("direction") and c["triggerSec"] - pending["triggerSec"] <= RETREAT_SECONDS:
                skipped.add(c["candidateId"])
                pending = None
            else:
                pending = c
    return skipped


def candidate_groups(candidates: list[dict]) -> dict[tuple, list[dict]]:
    """Index candidates by track segment and POI so retreat lookups stay O(1).

    Long recordings produce tens of thousands of candidates; scanning the whole
    list per verdict would make cue-bound analysis quadratic.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in candidates:
        groups[(c.get("trackSegmentId"), c.get("poiId"))].append(c)
    return groups


def load_cue_timelines(run_dir: Path) -> dict:
    path = run_dir / "cues.json"
    if not path.exists():
        return {}
    return read_json(path).get("cues", {})


def cue_state_in_window(timelines: dict, cue_ids: list[str], start: float, end: float, anchor: float | None = None):
    """Return (state, onset) for an explicit [start, end] window; 'on'/'off'/'unknown'."""
    if not cue_ids:
        return "off", None
    intervals: list[list[float]] = []
    for cid in cue_ids:
        if cid not in timelines:
            return "unknown", None
        intervals.extend(timelines[cid].get("intervals", []))
    hits = [iv for iv in intervals if iv[1] >= start and iv[0] <= end]
    if not hits:
        return "off", None
    pivot = start if anchor is None else anchor
    best = min(hits, key=lambda iv: abs(iv[0] - pivot))
    return "on", best[0]


def red_state(timelines: dict, cue_ids: list[str], entry_t: float, pre: float, post: float):
    """Return (state, onset) where state is 'on'/'off'/'unknown'."""
    return cue_state_in_window(timelines, cue_ids, entry_t - pre, entry_t + post, anchor=entry_t)


def retreat_event(candidate: dict, grouped: dict[tuple, list[dict]], start: float, end: float):
    """Return the retreating candidate inside [start, end], or None.

    A line entry retreats by crossing back in the opposite direction; a zone
    entry retreats by leaving the zone. Used by both POI-triggered rules and
    signal-triggered forced-passage rules (the latter starts the window at the
    target's entry time).
    """
    seg, poi = candidate["trackSegmentId"], candidate["poiId"]
    for other in grouped.get((seg, poi), []):
        if other["candidateId"] == candidate["candidateId"]:
            continue
        if other["trackSegmentId"] != seg or other["poiId"] != poi:
            continue
        if not (start <= other["triggerSec"] <= end):
            continue
        if candidate["triggerKind"] == "line.cross":
            if other["triggerKind"] == "line.cross" and other.get("direction") and other["direction"] != candidate.get("direction"):
                return other
        elif candidate["triggerKind"] in ("zone.enter", "initial_inside"):
            if other["triggerKind"] == "zone.leave":
                return other
    return None


def _retreated(candidate: dict, grouped: dict[tuple, list[dict]], start: float, end: float) -> bool:
    return retreat_event(candidate, grouped, start, end) is not None


def merge_cue_episodes(intervals: list[list[float]], dedupe_seconds: float):
    """Merge cue re-ignitions within ``dedupe_seconds`` of the episode start.

    The anchor is the first trigger of the episode, not a rolling window, so a
    flickering signal cannot chain forever. Returns one dict per episode with
    the merged interval and how many raw triggers it absorbed.
    """
    episodes: list[dict] = []
    for start, end in intervals:
        if episodes and start - episodes[-1]["interval"][0] <= dedupe_seconds:
            episodes[-1]["interval"][1] = max(episodes[-1]["interval"][1], end)
            episodes[-1]["rawTriggerCount"] += 1
        else:
            episodes.append({"interval": [start, end], "rawTriggerCount": 1})
    return episodes


TARGET_TRIGGERS = ("line.cross", "zone.enter", "initial_inside")


def build_target_index(candidates: list[dict]):
    """Index subject candidates by POI/kind with time-sorted lookup lists.

    Signal episodes are judged one by one; scanning every crossing in the run
    for each episode would make a dense signal quadratic on long recordings.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for candidate in candidates:
        kind = candidate.get("triggerKind")
        if kind in TARGET_TRIGGERS:
            grouped[(candidate.get("poiId"), kind)].append(candidate)
    index = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda candidate: candidate["triggerSec"])
        index[key] = ([candidate["triggerSec"] for candidate in rows], rows)
    return index


def _targets_in_window(index: dict, poi_id: str, kinds: tuple[str, ...], start: float, end: float):
    rows: list[dict] = []
    for kind in kinds:
        entry = index.get((poi_id, kind))
        if entry is None:
            continue
        times, values = entry
        rows.extend(values[bisect_left(times, start):bisect_right(times, end)])
    return rows


def select_cue_target(candidates: list[dict], line, zone, start: float, end: float, trigger: float, *,
                      index: dict | None = None):
    """Pick the person nearest the signal trigger for one channel.

    Prefer a crossing on the paired counting line in the entry direction; fall
    back to entering the optional auxiliary counting zone when no line crossing
    is present (e.g. tracking started while the person was already inside).
    """
    index = build_target_index(candidates) if index is None else index
    pool = _targets_in_window(index, line.id, ("line.cross",), start, end)
    if line.entryDirection:
        pool = [candidate for candidate in pool if candidate.get("direction") == line.entryDirection]
    if not pool and zone is not None:
        pool = _targets_in_window(index, zone.id, ("zone.enter", "initial_inside"), start, end)
    if not pool:
        return None
    return min(pool, key=lambda candidate: (abs(candidate["triggerSec"] - trigger), candidate["triggerSec"]))


def tracked_through(run_dir: Path, candidate: dict, start: float, end: float, max_gap: float) -> bool:
    """Public wrapper so signal-triggered verdicts can reuse the gap advisory."""
    return _tracked_through(run_dir, candidate, start, end, max_gap)


def _tracked_through(run_dir: Path, candidate: dict, start: float, end: float, max_gap: float) -> bool:
    db = sqlite3.connect(run_dir / "observations.sqlite")
    try:
        rows = db.execute(
            "SELECT time, body FROM observations WHERE time >= ? AND time <= ? ORDER BY time",
            (start - max_gap, end + max_gap)).fetchall()
    finally:
        db.close()
    seg = candidate["trackSegmentId"]
    times = [t for t, body in rows if any(p.get("trackSegmentId") == seg for p in json.loads(body).get("persons", []))]
    if not times:
        return False
    if times[0] > start + max_gap or times[-1] < end - max_gap:
        return False
    return all(b - a <= max_gap for a, b in zip(times, times[1:], strict=False))


def evaluate(candidate: dict, event: EventDefinition, request: RunRequest, run_dir: Path,
             candidates: list[dict] | None = None, *, grouped: dict[tuple, list[dict]] | None = None,
             timelines: dict | None = None):
    if not event.cueIds:
        return None
    # The understanding phase passes preloaded timelines/grouping so a multi-hour
    # run does not re-read cues.json or rescan candidates for every verdict.
    timelines = load_cue_timelines(run_dir) if timelines is None else timelines
    grouped = candidate_groups(candidates or []) if grouped is None else grouped
    entry_t = candidate["triggerSec"]
    state, onset = red_state(timelines, event.cueIds, entry_t, event.preSeconds, event.postSeconds)
    cue_evidence = {"cueIds": event.cueIds, "state": state, "onsetSec": onset}
    if state == "off":
        return {"decision": "no_match", "reason": "检测窗口内门框红灯未亮起",
                "cueEvidence": cue_evidence, "reviewRequired": False}
    if state == "unknown":
        return {"decision": "uncertain", "reason": "红灯检测不确定，转 VLM 复核",
                "cueEvidence": cue_evidence, "reviewRequired": True}
    window_end = onset + RETREAT_SECONDS
    if _retreated(candidate, grouped, entry_t, window_end):
        return {"decision": "no_match", "reason": "红灯亮起后 5 秒内目标退回原区域",
                "cueEvidence": cue_evidence, "reviewRequired": False}
    if not _tracked_through(run_dir, candidate, entry_t, window_end, request.maxGapSeconds):
        # Occlusion after entry is normal: a tracking gap never downgrades the
        # verdict to uncertain; it is recorded as an advisory warning only.
        return {"decision": "match", "reason": "门框红灯亮起且未观察到退回；目标因遮挡跟踪缺失，仅作辅助警告",
                "cueEvidence": cue_evidence, "reviewRequired": False, "trackingGap": True}
    return {"decision": "match", "reason": "门框红灯亮起且 5 秒内未退回",
            "cueEvidence": cue_evidence, "reviewRequired": False}
