from __future__ import annotations

import sqlite3
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
from ultra_rules.vlm import combined, cues, integrity
from ultra_rules.vlm.contracts import RunRequest
from ultra_rules.vlm.media import probe, sampled
from ultra_rules.vlm.storage import read_json, write_json


def _make_video(path, fps=10, seconds=20, red_frames=None):
    red_frames = range(50, 100) if red_frames is None else red_frames
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(fps * seconds):
            arr = np.full((64, 64, 3), 128, np.uint8)
            if i in red_frames:
                arr[16:48, 16:48] = (0, 0, 255)
            frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _request():
    return RunRequest(
        videoId="v1",
        pois=[{"id": "line", "kind": "line", "points": [[.5, .1], [.5, .9]]}],
        cues=[{"id": "red", "points": [[.25, .25], [.75, .25], [.75, .75], [.25, .75]],
               "sampleFps": 10}],
        integrityZones=[{"id": "wall", "points": [[.0, .0], [.2, .0], [.2, .2], [.0, .2]],
                         "sampleFps": 0.5, "baselineFrames": 3, "confirmSec": 1.0,
                         "ignoreDetections": False}],
        events=[{"id": "e", "description": "test"}],
    )


def test_combined_detects_cues_and_integrity_with_one_decode(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    _make_video(source)
    meta = {**probe(source), "sha256": "x"}
    request = _request()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "request.json", {
        "request": request.model_dump(), "sourcePath": str(source), "video": meta,
        "cacheRoot": str(tmp_path / "cache"),
    })

    expected_cues = cues.detect_all(source, meta, request.cues)
    expected_integrity = {zone.id: integrity.detect_timeline(source, meta, zone)
                          for zone in request.integrityZones}

    real_decoded = combined.decoded
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        yield from real_decoded(*args, **kwargs)

    monkeypatch.setattr(combined, "decoded", counted)
    combined.detect_run(run_dir)

    assert calls == [1]
    assert read_json(run_dir / "cues.json") == {"cues": expected_cues}
    assert read_json(run_dir / "integrity.json") == {"zones": expected_integrity}

    # Both caches now exist: a second run must not decode the source again.
    calls.clear()
    monkeypatch.setattr(combined, "decoded",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decode")))
    combined.detect_run(run_dir)
    assert calls == []
    assert read_json(run_dir / "cues.json") == {"cues": expected_cues}
    assert read_json(run_dir / "integrity.json") == {"zones": expected_integrity}


class _FakeAdapter:
    instances = []

    def __init__(self, model_path, device, conf, iou, tracker_yaml, batch_size=1):
        self._model = SimpleNamespace(task="pose")
        self.batch_size = batch_size
        _FakeAdapter.instances.append(self)

    def load(self):
        pass

    def infer(self, _bgr):
        return []

    def infer_batch(self, images):
        return [[] for _ in images]

    def close(self):
        pass


def test_analyze_run_joins_tracking_cues_and_integrity(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    _make_video(source)
    meta = {**probe(source), "sha256": "x"}
    request = _request()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pose_model = tmp_path / "pose.pt"
    pose_model.write_bytes(b"fake pose model")
    write_json(run_dir / "request.json", {
        "request": request.model_dump(), "sourcePath": str(source), "video": meta,
        "poseModelPath": str(pose_model), "cacheRoot": str(tmp_path / "cache"),
    })
    monkeypatch.setattr("ultra_rules.pose_adapter.PoseAdapter", _FakeAdapter)
    monkeypatch.delenv("ULTRA_VLM_POSE_BATCH", raising=False)
    monkeypatch.delenv("ULTRA_VLM_PIPELINE", raising=False)

    combined.analyze_run(run_dir)

    expected_cues = cues.detect_all(source, meta, request.cues)
    expected_integrity = {zone.id: integrity.detect_timeline(source, meta, zone)
                          for zone in request.integrityZones}
    assert read_json(run_dir / "cues.json") == {"cues": expected_cues}
    assert read_json(run_dir / "integrity.json") == {"zones": expected_integrity}
    expected_pose_frames = len([1 for _ in sampled(source, meta, request.poseFps)])
    with sqlite3.connect(run_dir / "observations.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == expected_pose_frames
    assert read_json(run_dir / "observations.ref.json")["cacheHit"] is False

    # Retrying the same run must reuse its observation database and CV caches.
    monkeypatch.setattr(combined, "decoded",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decode")))
    monkeypatch.setattr("ultra_rules.pose_adapter.PoseAdapter",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pose")))
    combined.analyze_run(run_dir)
    assert read_json(run_dir / "cues.json") == {"cues": expected_cues}
    assert read_json(run_dir / "integrity.json") == {"zones": expected_integrity}

    # A fresh run with the same source/model must replay tracking and CV caches
    # without decoding or loading the pose model again.
    run2 = tmp_path / "run2"
    run2.mkdir()
    write_json(run2 / "request.json", read_json(run_dir / "request.json"))
    monkeypatch.setattr(combined, "decoded",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decode")))
    monkeypatch.setattr("ultra_rules.pose_adapter.PoseAdapter",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pose")))
    combined.analyze_run(run2)
    assert read_json(run2 / "cues.json") == {"cues": expected_cues}
    assert read_json(run2 / "integrity.json") == {"zones": expected_integrity}
    assert read_json(run2 / "observations.ref.json")["cacheHit"] is True
