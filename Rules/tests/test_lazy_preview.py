from __future__ import annotations

import json
import sqlite3
from fractions import Fraction

import av
import numpy as np
from PIL import Image
from ultra_rules.vlm.contracts import RunRequest
from ultra_rules.vlm.media import probe
from ultra_rules.vlm.sampling import build_sample, render_preview
from ultra_rules.vlm.storage import read_json, write_json


def _make_video(path, fps=10, seconds=2):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(fps * seconds):
            arr = np.full((64, 64, 3), 128, np.uint8)
            arr[16:48, 16:48] = (90, 90, 90)
            frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _request(preview_mode: str):
    return RunRequest(
        videoId="v1",
        pois=[{"id": "line", "kind": "line", "points": [[.5, .1], [.5, .9]]}],
        events=[{"id": "e", "description": "test", "preSeconds": .5, "postSeconds": .5}],
        previewMode=preview_mode,
    )


def _candidate():
    return {"candidateId": "c", "trackId": 1, "trackSegmentId": "1-1", "poiId": "line",
            "triggerKind": "line.cross", "direction": "negative_to_positive", "triggerSec": 1.0,
            "confirmedSec": 1.1, "eventDefinitionIds": ["e"], "identityAmbiguous": False}


def _prepare(tmp_path, name, preview_mode):
    source = tmp_path / f"{name}.mp4"
    _make_video(source)
    meta = {**probe(source), "sha256": "x"}
    run_dir = tmp_path / name
    run_dir.mkdir()
    request = _request(preview_mode)
    write_json(run_dir / "request.json", {"request": request.model_dump(), "sourcePath": str(source), "video": meta})
    person = {"trackId": 1, "trackSegmentId": "1-1", "bbox": [16, 16, 48, 48], "bboxConf": .9,
              "identityAmbiguous": False,
              "keypoints": [{"x": 32, "y": 32, "conf": .9} for _ in range(17)]}
    with sqlite3.connect(run_dir / "observations.sqlite") as db:
        db.execute("CREATE TABLE observations (frame_index INTEGER PRIMARY KEY, time REAL NOT NULL, body TEXT NOT NULL)")
        for i in range(20):
            t = i / 10
            body = {"frameIndex": i, "mediaTimeSec": t, "timeSource": "pts", "sourcePtsSec": t,
                    "persons": [person]}
            db.execute("INSERT INTO observations VALUES (?, ?, ?)", (i, t, json.dumps(body)))
        db.commit()
    return run_dir, source, meta, request


def test_lazy_sample_writes_manifest_only_then_renders_same_preview(tmp_path):
    lazy_dir, source, meta, request = _prepare(tmp_path, "lazy", "lazy")
    eager_dir, _, _, eager_request = _prepare(tmp_path, "eager", "eager")
    candidate = _candidate()
    config = {"sourcePath": str(source), "video": meta}

    lazy_frames, _, lazy_manifest = build_sample(lazy_dir, config, request, candidate, 0.0, 2.0, "s0")
    assert lazy_frames is None
    assert not (lazy_dir / "previews" / "s0.jpg").exists()
    assert read_json(lazy_dir / "samples" / "s0.json")["quality"]["targetVisibleFraction"] == 1

    preview = render_preview(lazy_dir, "s0")
    assert preview.is_file()
    with Image.open(preview) as image:
        assert image.width > 0 and image.height > 0

    eager_frames, _, eager_manifest = build_sample(eager_dir, config, eager_request, candidate, 0.0, 2.0, "s0")
    assert eager_frames is not None
    assert lazy_manifest == eager_manifest
    assert preview.read_bytes() == (eager_dir / "previews" / "s0.jpg").read_bytes()


def test_deterministic_evidence_lazy_skips_source_decode(tmp_path, monkeypatch):
    from ultra_rules.vlm import qwen_worker, sampling

    run_dir, source, meta, request = _prepare(tmp_path, "lazy-deterministic", "lazy")
    config = {"sourcePath": str(source), "video": meta}
    candidate = _candidate()
    monkeypatch.setattr(sampling, "source_images",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source decode")))

    segments = qwen_worker._build_deterministic_evidence(
        run_dir, config, request, candidate, request.events[0], "r0", {"decision": "match", "reason": "test"})

    assert segments[0]["previewPath"] == "previews/r0-0.jpg"
    assert not (run_dir / "previews" / "r0-0.jpg").exists()
    assert (run_dir / "samples" / "r0-0.json").is_file()
