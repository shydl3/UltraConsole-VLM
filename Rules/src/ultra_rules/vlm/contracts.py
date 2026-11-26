from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .models import default_model_path

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")]
Coordinate = Annotated[float, Field(ge=0, le=1)]
Point = tuple[Coordinate, Coordinate]
Trigger = Literal["line.cross", "zone.enter", "zone.leave", "initial_inside", "cue.on", "integrity.change"]
RuleCategory = Literal["poi_event", "forced_passage", "integrity"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Poi(StrictModel):
    id: Identifier
    label: str = Field(default="", max_length=200)
    kind: Literal["line", "zone"]
    points: list[Point] = Field(min_length=2, max_length=100)
    anchor: Literal["bbox_center", "bbox_bottom_center"] = "bbox_center"
    # Only meaningful for kind="line". When set, only crossings in this direction
    # are treated as "entries"; the opposite crossing is a retreat, not a new event.
    entryDirection: Literal["positive_to_negative", "negative_to_positive"] | None = None
    # Channel number for signal-triggered rules: a counting line and the signal
    # zone with the same pairIndex form one channel; an optional counting zone
    # with the same number is used as auxiliary presence evidence.
    pairIndex: int | None = Field(default=None, ge=1, le=1000)

    @model_validator(mode="after")
    def geometry(self):
        if self.kind == "line":
            if len(self.points) != 2 or self.points[0] == self.points[1]:
                raise ValueError("A line needs two distinct endpoints")
            if self.entryDirection is not None and self.entryDirection not in ("positive_to_negative", "negative_to_positive"):
                raise ValueError("Invalid entry direction")
        else:
            if self.points[0] == self.points[-1]:
                self.points = self.points[:-1]
            if len(self.points) < 3:
                raise ValueError("A zone needs at least three vertices")
            if any(a == b for a, b in zip(self.points, self.points[1:] + self.points[:1], strict=True)):
                raise ValueError("A zone cannot contain zero-length edges")
            area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(self.points, self.points[1:] + self.points[:1], strict=True))
            if abs(area) < 1e-8:
                raise ValueError("Zone area must be positive")
        return self


class CueDefinition(StrictModel):
    """A deterministic visual cue (e.g. a red LED strip) detected in a drawn ROI."""

    id: Identifier
    label: str = Field(default="", max_length=200)
    points: list[Point] = Field(min_length=3, max_length=100)
    # Channel number paired with the counting line of the same number.
    pairIndex: int | None = Field(default=None, ge=1, le=1000)
    method: Literal["red_mask"] = "red_mask"
    # Red pixel ratio above this fraction marks the frame as "on".
    minOnRatio: float = Field(default=0.06, gt=0, le=1)
    # RGB thresholds: red = (r >= minR) and (r - g >= minRG) and (r - b >= minRB).
    minR: int = Field(default=120, ge=0, le=255)
    minRG: int = Field(default=50, ge=0, le=255)
    minRB: int = Field(default=50, ge=0, le=255)
    # Temporal persistence: on for >= minOnSec to open an interval, off for >= minOffSec to close.
    minOnSec: float = Field(default=0.2, ge=0, le=5)
    minOffSec: float = Field(default=0.2, ge=0, le=5)
    sampleFps: float = Field(default=10, ge=1, le=30)

    @model_validator(mode="after")
    def geometry(self):
        if self.points[0] == self.points[-1]:
            self.points = self.points[:-1]
        if len(self.points) < 3:
            raise ValueError("A cue ROI needs at least three vertices")
        if any(a == b for a, b in zip(self.points, self.points[1:] + self.points[:1], strict=True)):
            raise ValueError("A cue ROI cannot contain zero-length edges")
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(self.points, self.points[1:] + self.points[:1], strict=True))
        if abs(area) < 1e-8:
            raise ValueError("Cue ROI area must be positive")
        return self


class IntegrityZone(StrictModel):
    """A static-scene region monitored for persistent visual change (tamper detection).

    The region is expected to stay unchanged for the whole video; a baseline is
    built from the first samples and every later sample is compared against it.
    A change must persist for confirmSec before it becomes an integrity.change
    warning, so transient occlusions and lighting drift do not alarm.
    """

    id: Identifier
    label: str = Field(default="", max_length=200)
    points: list[Point] = Field(min_length=3, max_length=100)
    # Independent low-rate sampling: integrity changes are minute-scale events.
    sampleFps: float = Field(default=0.5, ge=0.1, le=2)
    # Per-pixel median of the first N samples forms the baseline crop.
    baselineFrames: int = Field(default=5, ge=1, le=30)
    # Gray-level difference (0-255 scale) above which a pixel counts as changed.
    pixelThreshold: int = Field(default=25, ge=5, le=100)
    # Changed-pixel fraction inside the polygon that marks a sample as changed.
    changeRatio: float = Field(default=0.04, gt=0, le=1)
    # A change must persist this long before it is confirmed as an event.
    confirmSec: float = Field(default=10, ge=0, le=300)
    # Weight of the edge-structure channel vs the raw intensity channel (0..1).
    edgeWeight: float = Field(default=0.5, ge=0, le=1)
    # Largest connected changed blob must cover this fraction of the ROI.
    minBlobRatio: float = Field(default=0.005, ge=0, le=1)
    # Global camera shift up to this fraction of frame width is auto-aligned;
    # larger shifts mark the sample unreliable (camera moved / scene cut).
    alignTolerance: float = Field(default=0.02, ge=0, le=0.2)
    # Exclude pixels covered by tracked persons from the change mask.
    ignoreDetections: bool = True
    # Ask Qwen3-VL to review before/after crops once a change is confirmed.
    vlmReview: bool = False

    @model_validator(mode="after")
    def geometry(self):
        if self.points[0] == self.points[-1]:
            self.points = self.points[:-1]
        if len(self.points) < 3:
            raise ValueError("An integrity ROI needs at least three vertices")
        if any(a == b for a, b in zip(self.points, self.points[1:] + self.points[:1], strict=True)):
            raise ValueError("An integrity ROI cannot contain zero-length edges")
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(self.points, self.points[1:] + self.points[:1], strict=True))
        if abs(area) < 1e-8:
            raise ValueError("Integrity ROI area must be positive")
        return self


class EventDefinition(StrictModel):
    id: Identifier
    description: str = Field(min_length=1, max_length=1500)
    enabled: bool = True
    # Stable rule category (the user-facing name stays free text). It decides
    # required regions, triggering mode and the judgment path.
    category: RuleCategory = "poi_event"
    poiIds: list[Identifier] = Field(default_factory=list, max_length=50)
    cueIds: list[Identifier] = Field(default_factory=list, max_length=20)
    integrityIds: list[Identifier] = Field(default_factory=list, max_length=20)
    triggerKinds: list[Trigger] = Field(default_factory=lambda: ["line.cross", "zone.enter", "zone.leave"], min_length=1)
    # Signal-triggered rules merge re-ignitions within this window (anchored at
    # the first trigger) into one event.
    cueDedupeSec: float = Field(default=3.0, ge=0, le=60)
    preSeconds: float = Field(default=4, ge=0, le=60)
    postSeconds: float = Field(default=6, ge=0, le=60)
    criteria: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(default_factory=list, max_length=10)
    uncertainWhen: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def nonempty(self):
        if not self.description.strip() or self.preSeconds + self.postSeconds <= 0:
            raise ValueError("Description and time window must not be empty")
        if "cue.on" in self.triggerKinds and any(kind != "cue.on" for kind in self.triggerKinds):
            raise ValueError("cue.on cannot be combined with POI triggers")
        return self


class AnalysisParameters(StrictModel):
    modelPath: str = Field(default_factory=default_model_path, min_length=1, max_length=4096)
    poseDevice: Literal["cpu", "cuda"] = "cuda"
    poseFps: float = Field(default=10, ge=1, le=30)
    conf: float = Field(default=0.25, gt=0, lt=1)
    iou: float = Field(default=0.5, gt=0, lt=1)
    stableSeconds: float = Field(default=0.2, ge=0, le=2)
    hysteresisPx: float = Field(default=6, ge=0, le=50)
    maxGapSeconds: float = Field(default=1, ge=0.2, le=5)
    sampleFps: float = Field(default=2, ge=0.5, le=8)
    maxFrames: int = Field(default=32, ge=4, le=64, multiple_of=2)
    maxPixels: int = Field(default=196608, ge=65536, le=786432)
    maxNewTokens: int = Field(default=1024, ge=128, le=2048)
    # Deterministic evidence writes only the sample manifest in lazy mode; the
    # API renders the contact sheet on first access. VLM/dry-run samples always
    # render eagerly because they need the frames in memory.
    previewMode: Literal["eager", "lazy"] = "lazy"
    dryRun: bool = False


class RunRequest(AnalysisParameters):
    videoId: Identifier
    # Integrity-only runs are valid: no tracking POI is required when at least
    # one integrity zone is monitored.
    pois: list[Poi] = Field(default_factory=list, max_length=50)
    cues: list[CueDefinition] = Field(default_factory=list, max_length=20)
    integrityZones: list[IntegrityZone] = Field(default_factory=list, max_length=20)
    events: list[EventDefinition] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def references(self):
        if not self.pois and not self.integrityZones:
            raise ValueError("At least one POI or integrity zone is required")
        ids = [p.id for p in self.pois]
        cue_ids = [c.id for c in self.cues]
        integrity_ids = [z.id for z in self.integrityZones]
        event_ids = [e.id for e in self.events]
        if len(set(ids)) != len(ids) or len(set(event_ids)) != len(event_ids) or len(set(cue_ids)) != len(cue_ids) \
                or len(set(integrity_ids)) != len(integrity_ids):
            raise ValueError("POI, cue, integrity and event IDs must be unique")
        if not any(e.enabled for e in self.events):
            raise ValueError("At least one event must be enabled")
        for event in self.events:
            if set(event.poiIds) - set(ids):
                raise ValueError(f"Unknown POI referenced by {event.id}")
            if set(event.cueIds) - set(cue_ids):
                raise ValueError(f"Unknown cue referenced by {event.id}")
            if set(event.integrityIds) - set(integrity_ids):
                raise ValueError(f"Unknown integrity zone referenced by {event.id}")
            if event.category == "forced_passage":
                if "cue.on" not in event.triggerKinds or not event.cueIds:
                    raise ValueError(f"Forced-passage rule {event.id} needs cue.on and a bound signal zone")
                if not any(p.kind == "line" and p.id in event.poiIds for p in self.pois):
                    raise ValueError(f"Forced-passage rule {event.id} needs a bound counting line")
        return self


class Evidence(StrictModel):
    timeSec: float = Field(ge=0)
    observation: str = Field(min_length=1, max_length=500)


class ConditionCheck(StrictModel):
    index: int = Field(ge=0, le=9)
    value: Literal["yes", "no", "unknown"]
    # Source-video second of the frame that supports this judgement; null when unknown.
    evidenceTimeSec: float | None = Field(default=None, ge=0)


class ModelAnswer(StrictModel):
    decision: Literal["match", "no_match", "uncertain"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1500)
    # Optional for historical answers; new inference requires a visible-content description.
    videoDescription: str | None = Field(default=None, min_length=1, max_length=3000)
    evidence: list[Evidence] = Field(default_factory=list, max_length=8)
    # Structured per-condition verdicts: the primary output of new inference. The
    # final decision is derived from these by conjunction; empty for legacy answers,
    # which then fall back to trusting the model's own decision.
    conditions: list[ConditionCheck] = Field(default_factory=list, max_length=10)
    _parse_warnings: list[str] = PrivateAttr(default_factory=list)
