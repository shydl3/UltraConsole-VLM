"""PT0-1/PT0-2：tracking 批量化与解码/推理流水线的语义保持测试。

CPU 部分用假帧/假适配器验证流水线 plumbing（帧序、背压、分块、异常传播、
缓存指纹隔离）；GPU 冒烟测试在 CUDA 可用时验证 infer_batch 路径可运行。
真实视频的串行/批量逐帧一致性回归属测量脚本（Rules/scripts/bench_tracking.py）。
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ultra_rules.vlm import observations
from ultra_rules.vlm.contracts import EventDefinition, Poi, RunRequest
from ultra_rules.vlm.storage import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
POSE_MODEL = ROOT / "backend" / "models" / "yolo11m-pose.pt"


class FakeFrame:
    def __init__(self, i: int):
        self.width = 64
        self.height = 64
        self.time = i / 25
        self._arr = np.full((64, 64, 3), i % 256, dtype=np.uint8)

    def to_ndarray(self, format=None):
        return self._arr


class FakeAdapter:
    instances: list[FakeAdapter] = []

    def __init__(self, model_path, device, conf, iou, tracker_yaml, batch_size=1):
        self.batch_size = batch_size
        self.calls: list[int] = []
        self.batch_calls: list[int] = []
        self._model = SimpleNamespace(task="pose")
        FakeAdapter.instances.append(self)

    def load(self):
        pass

    def infer(self, image_bgr, *, track=True):
        self.calls.append(int(image_bgr[0, 0, 0]))
        return []

    def infer_batch(self, images_bgr):
        self.batch_calls.append(len(images_bgr))
        self.calls.extend(int(im[0, 0, 0]) for im in images_bgr)
        return [[] for _ in images_bgr]

    def close(self):
        pass


@pytest.fixture
def bench(tmp_path, monkeypatch):
    FakeAdapter.instances = []
    # track() 在函数内延迟导入 PoseAdapter，须 patch 源模块属性。
    monkeypatch.setattr("ultra_rules.pose_adapter.PoseAdapter", FakeAdapter)

    def setup(n_frames: int):
        def fake_sampled(source, meta, fps, **kwargs):
            for i in range(n_frames):
                yield i, i / 10, "pts", FakeFrame(i)

        monkeypatch.setattr(observations, "sampled", fake_sampled)
        request = RunRequest(
            videoId="bench",
            pois=[Poi(id="gate", kind="line", points=[(0.2, 0.5), (0.8, 0.5)])],
            events=[EventDefinition(id="enter", description="person crosses the gate",
                                    triggerKinds=["line.cross"], poiIds=["gate"])],
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        write_json(run_dir / "request.json", {
            "request": request.model_dump(),
            "sourcePath": str(tmp_path / "source.mp4"),
            "video": {"sha256": "0" * 64, "durationSec": 10.0, "fps": 25,
                      "width": 64, "height": 64, "timeOriginSec": 0.0},
            "poseModelPath": str(POSE_MODEL),
            "cacheRoot": str(tmp_path / "cache"),
        })
        return run_dir

    return setup


def _frame_indices(run_dir: Path) -> list[int]:
    with sqlite3.connect(run_dir / "observations.sqlite") as db:
        return [row[0] for row in db.execute("SELECT frame_index FROM observations ORDER BY frame_index")]


def test_serial_default_is_unchanged(bench, monkeypatch):
    monkeypatch.delenv("ULTRA_VLM_POSE_BATCH", raising=False)
    monkeypatch.delenv("ULTRA_VLM_PIPELINE", raising=False)
    run_dir = bench(12)
    observations.track(run_dir)
    adapter = FakeAdapter.instances[-1]
    assert adapter.batch_size == 1
    assert adapter.calls == list(range(12))
    assert adapter.batch_calls == []
    assert _frame_indices(run_dir) == list(range(12))
    fingerprint = read_json(run_dir / "observations.ref.json")["fingerprint"]
    assert fingerprint["poseBatch"] == 1
    assert fingerprint["poseInfer"] == "ultralytics-track"
    assert fingerprint["posePrecision"] == "default"


def test_batch_chunks_and_order(bench, monkeypatch):
    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "4")
    monkeypatch.delenv("ULTRA_VLM_PIPELINE", raising=False)
    run_dir = bench(10)
    observations.track(run_dir)
    adapter = FakeAdapter.instances[-1]
    assert adapter.batch_calls == [4, 4, 2]
    assert adapter.calls == list(range(10))
    assert _frame_indices(run_dir) == list(range(10))
    fingerprint = read_json(run_dir / "observations.ref.json")["fingerprint"]
    assert fingerprint["poseBatch"] == 4
    assert fingerprint["poseInfer"] == "batch-bytetrack"
    assert fingerprint["posePrecision"] == "fp32-deterministic"


def test_batch_forces_fp32_determinism(bench, monkeypatch):
    torch = pytest.importorskip("torch")
    flags = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "2")
    run_dir = bench(4)
    try:
        observations.track(run_dir)
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    finally:
        (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags


def test_pipeline_preserves_order(bench, monkeypatch):
    monkeypatch.setenv("ULTRA_VLM_PIPELINE", "1")
    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "3")
    run_dir = bench(40)
    observations.track(run_dir)
    adapter = FakeAdapter.instances[-1]
    assert adapter.calls == list(range(40))
    assert adapter.batch_calls == [3] * 13 + [1]
    assert _frame_indices(run_dir) == list(range(40))
    assert read_json(run_dir / "progress.json")["pipeline"] is True


def test_pipeline_backpressure_and_drain():
    frames = ((i, i / 10, "pts", FakeFrame(i)) for i in range(200))
    pipe = observations._FramePipeline(frames, maxsize=4)
    try:
        time.sleep(0.1)
        assert pipe._queue.qsize() <= 4
        rows = list(pipe.rows())
        assert [row[0] for row in rows] == list(range(200))
        assert all(row[6].shape == (64, 64, 3) for row in rows)
    finally:
        pipe.close()
    pipe._thread.join(timeout=2)
    assert not pipe._thread.is_alive()


def test_pipeline_producer_error_propagates(bench, monkeypatch):
    def broken_sampled(source, meta, fps, **kwargs):
        for i in range(5):
            yield i, i / 10, "pts", FakeFrame(i)
        raise RuntimeError("decode broke")

    monkeypatch.setenv("ULTRA_VLM_PIPELINE", "1")
    run_dir = bench(10)
    monkeypatch.setattr(observations, "sampled", broken_sampled)
    with pytest.raises(RuntimeError, match="decode broke"):
        observations.track(run_dir)


def test_pipeline_close_stops_producer():
    started = []

    def slow_frames():
        for i in range(10000):
            started.append(i)
            yield i, i / 10, "pts", FakeFrame(i)

    pipe = observations._FramePipeline(slow_frames(), maxsize=2)
    time.sleep(0.05)
    pipe.close()
    pipe._thread.join(timeout=2)
    assert not pipe._thread.is_alive()
    assert len(started) < 10000


def test_batch_cache_is_isolated_from_serial(bench, monkeypatch):
    monkeypatch.delenv("ULTRA_VLM_PIPELINE", raising=False)
    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "1")
    run_dir = bench(6)
    observations.track(run_dir)
    serial_ref = read_json(run_dir / "observations.ref.json")
    assert serial_ref["cacheHit"] is False

    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "2")
    run_dir2 = run_dir.parent / "run2"
    run_dir2.mkdir()
    write_json(run_dir2 / "request.json", read_json(run_dir / "request.json"))
    observations.track(run_dir2)
    batch_ref = read_json(run_dir2 / "observations.ref.json")
    assert batch_ref["cacheHit"] is False
    assert batch_ref["path"] != serial_ref["path"]

    run_dir3 = run_dir.parent / "run3"
    run_dir3.mkdir()
    write_json(run_dir3 / "request.json", read_json(run_dir / "request.json"))
    observations.track(run_dir3)
    assert read_json(run_dir3 / "observations.ref.json")["cacheHit"] is True


def test_env_int_clamps_and_falls_back(monkeypatch):
    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "not-a-number")
    assert observations._env_int("ULTRA_VLM_POSE_BATCH", 1, 1, 64) == 1
    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "999")
    assert observations._env_int("ULTRA_VLM_POSE_BATCH", 1, 1, 64) == 64
    monkeypatch.setenv("ULTRA_VLM_POSE_BATCH", "0")
    assert observations._env_int("ULTRA_VLM_POSE_BATCH", 1, 1, 64) == 1


def test_gpu_infer_batch_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not POSE_MODEL.is_file():
        pytest.skip("CUDA or pose model unavailable")
    from ultra_rules.pose_adapter import PoseAdapter

    frames = []
    for i in range(16):
        im = np.zeros((320, 320, 3), dtype=np.uint8)
        im[:, :, :] = (40, 60, 80)
        x = 20 + i * 8
        im[80:280, x:x + 40] = (200, 200, 200)
        frames.append(im)
    serial = PoseAdapter(str(POSE_MODEL), "cuda", batch_size=1)
    batched = PoseAdapter(str(POSE_MODEL), "cuda", batch_size=8)
    serial_out = [serial.infer(im) for im in frames]
    batch_out = []
    for start in range(0, len(frames), 8):
        batch_out.extend(batched.infer_batch(frames[start:start + 8]))
    assert len(batch_out) == len(frames)
    assert [len(p) for p in batch_out] == [len(p) for p in serial_out]
    assert [[p.track_id for p in persons] for persons in batch_out] == \
           [[p.track_id for p in persons] for persons in serial_out]
    serial.close()
    batched.close()