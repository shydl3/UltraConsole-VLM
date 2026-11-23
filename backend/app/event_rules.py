"""Persistent semantic event templates with atomic, revision-checked transactions."""
from __future__ import annotations

import copy
import fcntl
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException
from pydantic import Field, field_validator, model_validator

from . import vlm_jobs
from .vlm_jobs import (
    AnalysisParameters,
    CueDefinition,
    EventDefinition,
    Identifier,
    IntegrityZone,
    Poi,
    RunRequest,
    StrictModel,
)

_lock = threading.RLock()

# System-managed integrity rule: auto-bound to every drawn integrity zone whenever
# a run contains zones; users never edit or bind it manually.
PRESET_INTEGRITY_ID = "preset_integrity"
PRESET_INTEGRITY_RULE = {
    "id": PRESET_INTEGRITY_ID,
    "name": "完整性检测（预设）",
    "description": "完整性检测区是否出现持续性画面改变（围墙、门体等静态设施被改动）",
    "enabled": True,
    "category": "integrity",
    "poiIds": [],
    "cueIds": [],
    "integrityIds": [],
    "triggerKinds": ["integrity.change"],
    "cueDedupeSec": 3.0,
    "preSeconds": 4,
    "postSeconds": 6,
    "criteria": [],
    "uncertainWhen": [],
}


class Rule(EventDefinition):
    name: str = Field(min_length=1, max_length=160)

    @field_validator("name", "description")
    @classmethod
    def text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Text must not be blank")
        return value

    @model_validator(mode="after")
    def template_only(self):
        if self.poiIds:
            raise ValueError("POI bindings belong to the current analysis, not the rule template")
        if self.cueIds:
            raise ValueError("Cue bindings belong to the current analysis, not the rule template")
        if self.integrityIds:
            raise ValueError("Integrity bindings belong to the current analysis, not the rule template")
        return self


class Batch(StrictModel):
    revision: int = Field(ge=0)
    upserts: list[Rule] = Field(default_factory=list, max_length=1000)
    deletes: list[Identifier] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def unique(self):
        ids = [r.id for r in self.upserts] + self.deletes
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate rule IDs in transaction")
        return self


class Selection(StrictModel):
    id: Identifier
    version: int = Field(ge=1)
    poiIds: list[Identifier] = Field(default_factory=list, max_length=50)
    cueIds: list[Identifier] = Field(default_factory=list, max_length=20)
    integrityIds: list[Identifier] = Field(default_factory=list, max_length=20)


class ManagedRun(AnalysisParameters):
    videoId: Identifier
    pois: list[Poi] = Field(default_factory=list, max_length=50)
    cues: list[CueDefinition] = Field(default_factory=list, max_length=20)
    integrityZones: list[IntegrityZone] = Field(default_factory=list, max_length=20)
    selections: list[Selection] = Field(min_length=1, max_length=30)
    revision: int = Field(ge=0)

    @model_validator(mode="after")
    def has_targets(self):
        if not self.pois and not self.integrityZones:
            raise ValueError("At least one POI or integrity zone is required")
        return self


@contextmanager
def transaction(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with _lock, (root / "event-rules.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load(root: Path):
    path = root / "event-rules.json"
    if not path.exists():
        default = Path(__file__).resolve().parents[2] / "Rules/configs/vlm_events.json"
        now = time.time()
        rules = []
        for event in vlm_jobs.read_json(default)["events"]:
            rule = Rule.model_validate({**event, "poiIds": [], "name": event["description"][:160]})
            rules.append({**rule.model_dump(), "version": 1, "createdAt": now, "updatedAt": now})
        vlm_jobs.write_json(path, {"revision": 1, "rules": rules})
    try:
        value = vlm_jobs.read_json(path)
        if not isinstance(value["revision"], int) or not isinstance(value["rules"], list):
            raise ValueError("Invalid rule library")
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(500, "Rule library is unreadable; existing file was preserved") from exc
    # Migrate existing non-empty libraries once; a deliberately empty library stays empty.
    changed = False
    if value["rules"] and not any(row["id"] == PRESET_INTEGRITY_ID for row in value["rules"]):
        now = time.time()
        value["rules"].append({**PRESET_INTEGRITY_RULE, "version": 1, "createdAt": now, "updatedAt": now})
        changed = True
    # Older libraries predate rule categories. Integrity rules are identifiable
    # by their trigger; everything else keeps the legacy POI-triggered behavior.
    for row in value["rules"]:
        if "category" not in row:
            row["category"] = "integrity" if row.get("id") == PRESET_INTEGRITY_ID \
                or "integrity.change" in (row.get("triggerKinds") or []) else "poi_event"
            changed = True
        if "cueDedupeSec" not in row:
            row["cueDedupeSec"] = 3.0
            changed = True
    # The bundled forced-passage template now means the signal-triggered rule.
    # Convert that one id in place (test libraries only) and bump its version so
    # a stale editor cannot submit the old POI bindings against the new meaning.
    for row in value["rules"]:
        if row.get("id") == "forced_passage_unopposed" and row.get("category") == "poi_event":
            row.update(category="forced_passage", triggerKinds=["cue.on"],
                       name="强行通过（未退回）", description="是否该人员强行通过安检门且未退回",
                       preSeconds=max(float(row.get("preSeconds") or 0), 8),
                       postSeconds=max(float(row.get("postSeconds") or 0), 6),
                       criteria=["目标实际通过指定通道", "信号区亮起后 5 秒内未观察到目标退回"],
                       uncertainWhen=["无法区分目标", "关键时段或目标所在区域不可见"],
                       version=int(row.get("version") or 0) + 1, updatedAt=time.time())
            value["revision"] = int(value.get("revision") or 0) + 1
            changed = True
    if changed:
        vlm_jobs.write_json(path, value)
    return value


def load(root: Path):
    with transaction(root):
        return _load(root)


def save(root: Path, request: Batch):
    with transaction(root):
        current = _load(root)
        if current["revision"] != request.revision:
            raise HTTPException(409, "规则库已在其他窗口修改，请导出草稿或重新加载后再保存 / Rule library changed; reload before saving")
        rows = {row["id"]: row for row in current["rules"]}
        for identifier in request.deletes:
            if identifier not in rows:
                raise HTTPException(404, f"Rule not found: {identifier}")
            del rows[identifier]
        now = time.time()
        for rule in request.upserts:
            previous = rows.get(rule.id, {})
            rows[rule.id] = {**rule.model_dump(), "version": previous.get("version", 0) + 1,
                             "createdAt": previous.get("createdAt", now), "updatedAt": now}
        if len(rows) > 1000:
            raise HTTPException(422, "Rule library supports up to 1000 definitions")
        result = {"revision": current["revision"] + 1, "rules": list(rows.values())}
        vlm_jobs.write_json(root / "event-rules.json", result)
        return result


def _bind_forced_passage(request: ManagedRun, selection: Selection, event: dict, name: str) -> None:
    """Resolve one forced-passage rule to paired (line, optional zone, signal) channels.

    A channel is a counting line plus the signal ROI with the same ``pairIndex``;
    a counting zone with that number may be bound as auxiliary presence evidence.
    The signal decides *when* to judge; the line decides who passed and whether
    they crossed back (retreat).
    """
    pois = {poi.id: poi for poi in request.pois}
    cues = {cue.id: cue for cue in request.cues}
    bound_pois = [pois[item] for item in selection.poiIds] if selection.poiIds else list(request.pois)
    bound_cues = [cues[item] for item in selection.cueIds] if selection.cueIds else list(request.cues)

    lines, zones = {}, {}
    for poi in bound_pois:
        if poi.pairIndex is None:
            if selection.poiIds:
                raise HTTPException(422, f"计数区/线缺少配对编号（编号必填）：{name} / {poi.label or poi.id}")
            continue
        target = lines if poi.kind == "line" else zones
        if poi.pairIndex in target:
            raise HTTPException(422, f"计数 POI 编号重复：{name} / #{poi.pairIndex}")
        target[poi.pairIndex] = poi
    if not lines:
        raise HTTPException(422, f"强行通过规则需要入口计数线：{name}")
    for index, poi in lines.items():
        if not poi.entryDirection:
            raise HTTPException(422, f"请为入口计数线设置进入方向：{name} / {poi.label or poi.id}（#{index}）")

    signals = {}
    for cue in bound_cues:
        if cue.pairIndex is None:
            raise HTTPException(422, f"信号区缺少配对编号（编号必填）：{name} / {cue.label or cue.id}")
        if cue.pairIndex in signals:
            raise HTTPException(422, f"信号区编号重复：{name} / #{cue.pairIndex}")
        signals[cue.pairIndex] = cue

    missing_cues = sorted(set(lines) - set(signals))
    missing_lines = sorted(set(signals) - set(lines))
    if missing_cues or missing_lines:
        detail = []
        if missing_cues:
            detail.append(f"缺少编号 {missing_cues} 的信号区")
        if missing_lines:
            detail.append(f"缺少编号 {missing_lines} 的入口计数线")
        raise HTTPException(422, f"强行通过规则的计数线与信号区必须按编号一一对应：{name}；" + "；".join(detail))

    if selection.poiIds:
        for index, zone in zones.items():
            if index not in lines:
                raise HTTPException(422, f"计数区 {zone.label or zone.id}（编号 {index}）没有对应的入口计数线/信号区：{name}")

    # Only channels that actually have a line participate; auxiliary zones must
    # share a channel number. "All" bindings ignore unrelated zones so multiple
    # rules can share one drawing board.
    aux = {index: zone for index, zone in zones.items() if index in lines}
    event["triggerKinds"] = ["cue.on"]
    event["poiIds"] = [lines[index].id for index in sorted(lines)] + [aux[index].id for index in sorted(aux)]
    event["cueIds"] = [signals[index].id for index in sorted(signals)]
    event["integrityIds"] = []


def resolve(root: Path, request: ManagedRun):
    with transaction(root):
        library = _load(root)
        if library["revision"] != request.revision:
            raise HTTPException(409, "Rule library changed; reload before starting analysis")
        rows = {row["id"]: row for row in library["rules"]}
        events, snapshots = [], []
        for selection in request.selections:
            if selection.id == PRESET_INTEGRITY_ID:
                continue  # auto-managed below, never taken from user bindings
            row = rows.get(selection.id)
            if not row or row["version"] != selection.version or not row["enabled"]:
                raise HTTPException(409, f"Rule changed, disabled or deleted: {selection.id}")
            event = {key: row[key] for key in EventDefinition.model_fields if key in row}
            event.setdefault("category", "poi_event")
            if set(selection.poiIds) - {poi.id for poi in request.pois}:
                raise HTTPException(422, f"Rule references an unknown POI: {row['name']}")
            if set(selection.cueIds) - {cue.id for cue in request.cues}:
                raise HTTPException(422, f"Rule references an unknown cue: {row['name']}")
            if set(selection.integrityIds) - {z.id for z in request.integrityZones}:
                raise HTTPException(422, f"Rule references an unknown integrity zone: {row['name']}")
            if not request.pois and request.integrityZones and event["category"] != "integrity":
                raise HTTPException(
                    422, f"当前仅绘制了完整性区，只能启用完整性规则：{row['name']} 需要计数线/计数区")
            if event["category"] == "forced_passage":
                _bind_forced_passage(request, selection, event, row["name"])
            elif event["category"] == "integrity":
                event["triggerKinds"] = ["integrity.change"]
                event["poiIds"], event["cueIds"] = [], []
                event["integrityIds"] = selection.integrityIds
                if not request.integrityZones:
                    raise HTTPException(422, f"Integrity rule needs at least one integrity zone: {row['name']}")
            else:
                event["poiIds"] = selection.poiIds
                # Empty cue binding means "all applicable signal ROIs" in the editor;
                # expand it so the deterministic cue gate actually runs.
                event["cueIds"] = selection.cueIds or [cue.id for cue in request.cues]
                event["integrityIds"] = selection.integrityIds
                applicable = [poi for poi in request.pois if not selection.poiIds or poi.id in selection.poiIds]
                poi_ok = any(set(event["triggerKinds"]) & ({"line.cross"} if poi.kind == "line" else {"zone.enter", "zone.leave", "initial_inside"}) for poi in applicable)
                integrity_applicable = [z for z in request.integrityZones if not selection.integrityIds or z.id in selection.integrityIds]
                integrity_ok = "integrity.change" in event["triggerKinds"] and bool(integrity_applicable)
                if not poi_ok and not integrity_ok:
                    raise HTTPException(422, f"Rule has no compatible POI/trigger: {row['name']}")
            events.append(event)
            snapshots.append({"id": row["id"], "name": row["name"], "version": row["version"], "event": copy.deepcopy(event)})
        if request.integrityZones:
            preset_row = rows.get(PRESET_INTEGRITY_ID)
            source = preset_row or PRESET_INTEGRITY_RULE
            preset = {key: source[key] for key in EventDefinition.model_fields if key in source}
            preset.update(enabled=True, poiIds=[], cueIds=[],
                          integrityIds=[zone.id for zone in request.integrityZones])
            events.append(preset)
            snapshots.append({"id": PRESET_INTEGRITY_ID, "name": source["name"],
                              "version": preset_row["version"] if preset_row else None,
                              "event": copy.deepcopy(preset)})
        values = request.model_dump(exclude={"revision", "selections"})
        try:
            run = RunRequest.model_validate({**values, "events": events})
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return run, {"libraryRevision": library["revision"], "rules": snapshots}


def snapshot(run_dir: Path):
    path = run_dir / "rules-snapshot.json"
    if path.exists():
        return vlm_jobs.read_json(path)
    # Older and CLI runs already contain the effective event definitions.
    path = run_dir / "request.json"
    if not path.exists():
        return {"libraryRevision": None, "rules": []}
    events = vlm_jobs.read_json(path).get("request", {}).get("events", [])
    return {"libraryRevision": None, "rules": [
        {"id": e["id"], "name": e["description"][:160], "version": None, "event": e} for e in events
    ]}
