from __future__ import annotations

from ultra_rules.vlm.runner import count_candidates, phase_timeout, phases_for


def config(duration: float):
    return {"video": {"durationSec": duration}}


def test_phase_timeouts_scale_for_a_24h_recording(tmp_path):
    day = 24 * 3600
    assert phase_timeout("segment", config(day), tmp_path) == day
    assert phase_timeout("tracking", config(day), tmp_path) == 4 * day
    assert phase_timeout("cues", config(day), tmp_path) == day
    assert phase_timeout("detect", config(day), tmp_path) == 4 * day
    assert phase_timeout("understanding", config(day), tmp_path) == 4 * day


def test_understanding_timeout_counts_candidates(tmp_path):
    (tmp_path / "candidates.jsonl").write_text("{}\n" * 20000)
    assert count_candidates(tmp_path) == 20000
    assert phase_timeout("understanding", config(24 * 3600), tmp_path) == 20000 * 30 + 3600


def test_short_clips_keep_the_generous_floor(tmp_path):
    assert phase_timeout("segment", config(90), tmp_path) == 3600
    assert phase_timeout("tracking", config(90), tmp_path) == 86400
    assert phase_timeout("detect", config(90), tmp_path) == 86400


def _run_config(cues: bool, integrity: bool):
    return {"posePython": "pose-python", "qwenPython": "qwen-python",
            "request": {"cues": [{"id": "c"}] if cues else [],
                        "integrityZones": [{"id": "z"}] if integrity else []}}


def test_phases_share_one_detect_pass_only_when_both_consumers_need_video():
    assert [name for name, _ in phases_for(_run_config(True, True))] == \
        ["segment", "detect", "understanding"]
    assert [name for name, _ in phases_for(_run_config(True, False))] == \
        ["segment", "tracking", "cues", "understanding"]
    assert [name for name, _ in phases_for(_run_config(False, True))] == \
        ["segment", "tracking", "integrity", "understanding"]
    assert [name for name, _ in phases_for(_run_config(False, False))] == \
        ["segment", "tracking", "understanding"]
