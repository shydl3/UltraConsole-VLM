from __future__ import annotations

import json
import sqlite3
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
from ultra_rules.contracts import Keypoint, PersonObservation
from ultra_rules.vlm import observations
from ultra_rules.vlm.contracts import RunRequest
from ultra_rules.vlm.media import probe
from ultra_rules.vlm.storage import read_json, write_json


def _make_video(path, fps=5, seconds=2):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(fps * seconds):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 128, np.uint8), format="bgr24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _person():
    return PersonObservation(track_id=1, bbox=[16, 16, 48, 48], bbox_conf=.9,
                             keypoints=[Keypoint(x=32, y=32, conf=.9) for _ in range(17)])


class _FakeAdapter:
    def __init__(self, *args, **kwargs):
        self._model = SimpleNamespace(task="pose")

    def load(self):
        pass

    def infer(self, _bgr):
        return [_person()]

    def infer_batch(self, images):
        return [[_person()] for _ in images]

    def close(self):
        pass


def test_track_batch_sqlite_rows_keep_post_tracker_fields(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    _make_video(source)
    meta = {**probe(source), "sha256": "x"}
    model = tmp_path / "pose.pt"
    model.write_bytes(b"fake")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = RunRequest(videoId="v1", poseFps=5,
                         pois=[{"id": "line", "kind": "line", "points": [[.5, .1], [.5, .9]]}],
                         events=[{"id": "e", "description": "cross", "triggerKinds": ["line.cross"]}])
    write_json(run_dir / "request.json", {
        "request": request.model_dump(), "sourcePath": str(source), "video": meta,
        "poseModelPath": str(model), "cacheRoot": str(tmp_path / "cache"),
    })
    monkeypatch.setattr("ultra_rules.pose_adapter.PoseAdapter", _FakeAdapter)
    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "2")
    monkeypatch.setenv("ULTRA_VLM_PIPELINE", "0")

    observations.track(run_dir)

    with sqlite3.connect(run_dir / "observations.sqlite") as db:
        rows = list(db.execute("SELECT body FROM observations ORDER BY frame_index"))
    assert len(rows) == 10
    for (body,) in rows:
        person = json.loads(body)["persons"][0]
        assert person["trackSegmentId"] == "1-1"
        assert person["identityAmbiguous"] is False
    assert read_json(run_dir / "progress.json")["cacheHit"] is False
