import json
import sqlite3
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from ultra_rules.vlm.contracts import IntegrityZone, RunRequest
from ultra_rules.vlm.integrity import (
    _largest_blob,
    _phase_shift,
    calibrate,
    detect_run,
    detect_timeline,
)
from ultra_rules.vlm.storage import read_json


def zone(**overrides):
    base = dict(id="wall", points=[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
                sampleFps=2, baselineFrames=3, confirmSec=0.6)
    base.update(overrides)
    return IntegrityZone(**base)


def meta(duration=4.0):
    return {"width": 64, "height": 64, "fps": 10, "durationSec": duration,
            "timeOriginSec": 0.0, "timeBase": "1/10", "sha256": "x"}


def _make_video(path: Path, frames=40, fps=10, patch_frames=(), brighten=None):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(frames):
            level = 128 + (brighten(i) if brighten else 0)
            arr = np.full((64, 64, 3), level, np.uint8)
            if i in patch_frames:
                arr[20:44, 20:44] = (255, 255, 255)
            frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def test_phase_shift_recovers_known_roll():
    rng = np.random.default_rng(7)
    ref = rng.integers(0, 255, (64, 96)).astype(np.float32)
    cur = np.roll(ref, (3, -2), axis=(0, 1))
    assert _phase_shift(ref, cur) == (3, -2)


def test_largest_blob_union_find():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True          # 4 px
    mask[5, 5:10] = True           # 5 px
    mask[8, 8] = True              # 1 px
    assert _largest_blob(mask) == 5
    assert _largest_blob(np.zeros((4, 4), dtype=bool)) == 0


def test_persistent_change_confirms_once(tmp_path):
    src = tmp_path / "v.mp4"
    _make_video(src, patch_frames=set(range(20, 40)))
    evidence = tmp_path / "ev"
    result = detect_timeline(src, meta(), zone(), evidence_dir=evidence)
    assert len(result["intervals"]) == 1
    interval = result["intervals"][0]
    assert 1.9 <= interval["onsetSec"] <= 2.1
    assert interval["confirmedSec"] >= 2.6
    assert interval["offsetSec"] == 3.5
    assert interval["peakRatio"] > 0.3
    for key in ("before", "after", "diff", "context"):
        assert interval["evidence"][key].startswith("previews/integrity-wall-0-")
        assert (evidence / Path(interval["evidence"][key]).name).is_file()


def test_transient_change_below_confirm_sec_is_ignored(tmp_path):
    src = tmp_path / "v.mp4"
    _make_video(src, patch_frames={20, 21, 22})
    result = detect_timeline(src, meta(), zone())
    assert result["intervals"] == []


def test_global_lighting_drift_is_normalized_away(tmp_path):
    src = tmp_path / "v.mp4"
    _make_video(src, brighten=lambda i: 40 if i >= 20 else 0)
    result = detect_timeline(src, meta(), zone())
    assert result["intervals"] == []


def _observations_db(path: Path, times):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE observations (frame_index INTEGER PRIMARY KEY, time REAL NOT NULL, body TEXT NOT NULL)")
    for i, t in enumerate(times):
        body = {"persons": [{"trackId": 1, "bbox": [16.0, 16.0, 49.0, 49.0]}]}
        db.execute("INSERT INTO observations VALUES (?, ?, ?)", (i, t, json.dumps(body)))
    db.commit()
    db.close()


def test_person_occlusion_is_excluded_when_enabled(tmp_path):
    src = tmp_path / "v.mp4"
    _make_video(src, patch_frames=set(range(20, 40)))
    db_path = tmp_path / "observations.sqlite"
    _observations_db(db_path, [2.0, 2.5, 3.0, 3.5])
    masked = detect_timeline(src, meta(), zone(ignoreDetections=True), observations=db_path)
    assert masked["intervals"] == []
    unmasked = detect_timeline(src, meta(), zone(ignoreDetections=False), observations=db_path)
    assert len(unmasked["intervals"]) == 1


def test_calibrate_reports_ratio_stats(tmp_path):
    src = tmp_path / "v.mp4"
    _make_video(src, patch_frames=set(range(20, 40)))
    result = calibrate(src, meta(), zone())
    assert result["roiPixelCount"] > 0
    assert result["sampleCount"] > 0
    assert result["maxRatio"] > 0.3
    assert result["minRatio"] < 0.01
    # Baseline consumes the first three samples (0 / 0.5 / 1.0s at 2fps).
    assert result["samples"][0][0] == 1.0


def test_detect_run_caches_and_copies_evidence(tmp_path):
    src = tmp_path / "v.mp4"
    _make_video(src, patch_frames=set(range(20, 40)))
    request = RunRequest(videoId="v1", integrityZones=[zone()],
                         events=[{"id": "e", "description": "围墙完整性", "triggerKinds": ["integrity.change"]}])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {"request": request.model_dump(), "sourcePath": str(src), "video": meta(),
              "cacheRoot": str(tmp_path / "cache")}
    (run_dir / "request.json").write_text(json.dumps(config), encoding="utf-8")
    detect_run(run_dir)
    first = read_json(run_dir / "integrity.json")
    assert len(first["zones"]["wall"]["intervals"]) == 1
    assert (run_dir / "previews" / "integrity-wall-0-before.jpg").is_file()
    detect_run(run_dir)
    assert read_json(run_dir / "integrity.json") == first
    assert read_json(run_dir / "progress.json")["integrityFraction"] == 1


def test_detect_run_without_zones(tmp_path):
    request = RunRequest(videoId="v1", pois=[{"id": "p", "kind": "line", "points": [[0, 0], [1, 1]]}],
                         events=[{"id": "e", "description": "x"}])
    (tmp_path / "request.json").write_text(json.dumps({"request": request.model_dump()}), encoding="utf-8")
    detect_run(tmp_path)
    assert read_json(tmp_path / "integrity.json") == {"zones": {}}


def test_run_request_integrity_only_and_references():
    request = RunRequest(videoId="v1", integrityZones=[zone()],
                         events=[{"id": "e", "description": "x", "triggerKinds": ["integrity.change"],
                                  "integrityIds": ["wall"]}])
    assert request.pois == []
    with pytest.raises(ValueError):
        RunRequest(videoId="v1", events=[{"id": "e", "description": "x"}])
    with pytest.raises(ValueError):
        RunRequest(videoId="v1", integrityZones=[zone()],
                   events=[{"id": "e", "description": "x", "integrityIds": ["missing"]}])


def test_understand_emits_positive_and_negative_records(tmp_path):
    from ultra_rules.vlm.qwen_worker import understand
    from ultra_rules.vlm.storage import iter_jsonl
    request = RunRequest(videoId="v1", integrityZones=[zone(), zone(id="gate")],
                         events=[{"id": "e", "description": "x", "triggerKinds": ["integrity.change"]}],
                         dryRun=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "request.json").write_text(json.dumps(
        {"request": request.model_dump(), "video": meta(), "sourcePath": str(tmp_path / "v.mp4"),
         "cacheRoot": str(tmp_path / "cache")}), encoding="utf-8")
    (run_dir / "integrity.json").write_text(json.dumps({"zones": {
        "wall": {"intervals": [{"onsetSec": 2.0, "confirmedSec": 3.0, "offsetSec": None,
                               "peakRatio": 0.4, "evidence": {}}],
                 "sampleCount": 8, "baselineSec": 1.0},
        "gate": {"intervals": [], "sampleCount": 8, "baselineSec": 1.0},
    }}), encoding="utf-8")
    understand(run_dir)
    rows = list(iter_jsonl(run_dir / "results.jsonl"))
    assert len(rows) == 2
    by_id = {row["resultId"]: row for row in rows}
    positive = by_id["integrity-wall-0-e"]
    negative = by_id["integrity-gate-none-e"]
    assert positive["decision"] == "match" and positive["integrityEvidence"]["changed"] is True
    assert negative["decision"] == "no_match" and negative["integrityEvidence"]["changed"] is False
    assert negative["integrityEvidence"]["onsetSec"] is None
    assert negative["integrityEvidence"]["sampleCount"] == 8
    assert "全程未检测到持续画面变化" in negative["reason"]
    assert read_json(run_dir / "summary.json")["resultCount"] == 2
