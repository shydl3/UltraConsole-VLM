"""Signal-triggered forced-passage rules: pairing, dedupe and deterministic verdicts."""
from __future__ import annotations

from ultra_rules.vlm import qwen_worker as worker
from ultra_rules.vlm.contracts import EventDefinition, RunRequest
from ultra_rules.vlm.decision import candidate_groups
from ultra_rules.vlm.observations import candidate_owners
from ultra_rules.vlm.storage import iter_jsonl


def line_poi(identifier="line", index=1, direction="negative_to_positive"):
    return {"id": identifier, "kind": "line", "points": [[.5, .1], [.5, .9]],
            "pairIndex": index, "entryDirection": direction}


def zone_poi(identifier="zone", index=1):
    return {"id": identifier, "kind": "zone", "points": [[.2, .2], [.8, .2], [.8, .8], [.2, .8]],
            "pairIndex": index}


def cue_roi(identifier="cue", index=1):
    return {"id": identifier, "points": [[.35, .35], [.65, .35], [.65, .65], [.35, .65]],
            "pairIndex": index}


def forced_event(identifier="e1", poi_ids=("line",), cue_ids=("cue",), **changes):
    values = {"id": identifier, "description": "强行通过且未退回", "category": "forced_passage",
              "triggerKinds": ["cue.on"], "poiIds": list(poi_ids), "cueIds": list(cue_ids),
              "preSeconds": 8, "postSeconds": 6}
    values.update(changes)
    return values


def make_request(pois=None, cues=None, events=None, **changes):
    return RunRequest(videoId="v1", pois=pois or [], cues=cues or [], events=events or [forced_event()],
                      **changes)


def make_candidate(identifier, trigger_sec, kind="line.cross", poi="line", direction="negative_to_positive"):
    return {"candidateId": identifier, "trackId": 1, "trackSegmentId": "1-1", "poiId": poi,
            "triggerKind": kind, "direction": direction, "triggerSec": trigger_sec,
            "identityAmbiguous": False, "eventDefinitionIds": []}


def test_cue_pairs_merge_episodes_and_isolate_channels():
    request = make_request(
        pois=[line_poi("line1", 1), line_poi("line2", 2), zone_poi("zone1", 1)],
        cues=[cue_roi("cue1", 1), cue_roi("cue2", 2)],
        events=[forced_event(poi_ids=("line1", "line2", "zone1"), cue_ids=("cue1", "cue2"))],
    )
    timelines = {
        "cue1": {"intervals": [[10.0, 10.2], [11.0, 11.2], [12.5, 13.0]]},
        "cue2": {"intervals": [[20.0, 20.2], [21.0, 21.2]]},
    }
    events = {event.id: event for event in request.events}
    pairs = worker._cue_pairs(request, events, timelines)
    assert [(pair[1].id, pair[3].id, pair[5], pair[6]) for pair in pairs] == [
        ("line1", "cue1", [10.0, 13.0], 3),
        ("line2", "cue2", [20.0, 21.2], 2),
    ]
    # The optional auxiliary zone is attached to its own channel only.
    assert pairs[0][2].id == "zone1"
    assert pairs[1][2] is None


def test_candidate_owners_persist_support_without_double_judging():
    support_event = EventDefinition.model_validate(forced_event(poi_ids=("line",)))
    judging_event = EventDefinition.model_validate({"id": "e2", "description": "计数", "poiIds": ["line"]})
    crossing = {"triggerKind": "line.cross", "poiId": "line"}
    applicable, support = candidate_owners(crossing, [support_event, judging_event])
    assert applicable == ["e2"]
    assert support == ["e1"]
    # An unrelated area is not persisted for either rule.
    assert candidate_owners({"triggerKind": "line.cross", "poiId": "other"},
                            [support_event, judging_event]) == ([], [])


def test_cue_verdict_match_retreat_and_no_subject(tmp_path, monkeypatch):
    request = make_request(pois=[line_poi(), zone_poi()], cues=[cue_roi()])
    event = request.events[0]
    line = request.pois[0]
    zone = request.pois[1]
    cue = request.cues[0]
    config = {"video": {"durationSec": 40.0, "width": 1920, "height": 1080, "sha256": "abc",
                        "sourcePath": str(tmp_path / "missing.mp4")}}
    monkeypatch.setattr(worker, "_build_deterministic_evidence",
                        lambda run_dir, config, request, candidate, event, result_id, deterministic: [
                            {"window": {"startSec": 2.0, "endSec": 21.0}, "quality": {},
                             "previewPath": "previews/x.jpg", "sampleManifest": "samples/x.json"}])
    monkeypatch.setattr(worker, "tracked_through", lambda *args: True)

    def run(candidates, episode_index=0, interval=(10.0, 10.2)):
        run_dir = tmp_path / f"run-{episode_index}-{len(candidates)}"
        run_dir.mkdir()
        state = {"model": None, "done": 0, "failed": 0,
                 "counts": {"match": 0, "no_match": 0, "uncertain": 0},
                 "completed": set(), "currentSegment": 0}
        worker._process_cue_candidate(run_dir, config, request, event, line, zone, cue,
                                      episode_index, interval, 1, candidates,
                                      candidate_groups(candidates), state)
        return list(iter_jsonl(run_dir / "results.jsonl"))[0]

    entry = make_candidate("entry", 9.8)
    matched = run([entry])
    assert matched["decision"] == "match" and matched["cueEvidence"]["retreated"] is False
    assert matched["cueEvidence"]["triggerCueId"] == "cue"
    assert matched["poiKind"] == "line" and matched["pairIndex"] == 1

    retreat = make_candidate("back", 12.0, direction="positive_to_negative")
    retreated = run([entry, retreat])
    assert retreated["decision"] == "no_match" and retreated["cueEvidence"]["retreated"] is True
    assert retreated["cueEvidence"]["retreatSec"] == 12.0

    empty = run([])
    assert empty["decision"] == "no_match"
    assert "no_subject" in empty["qualityWarnings"]
    assert empty["trackSegmentId"] is None


def test_no_subject_cue_episode_skips_evidence_decoding(tmp_path, monkeypatch):
    request = make_request(pois=[line_poi()], cues=[cue_roi()])
    event, line, cue = request.events[0], request.pois[0], request.cues[0]
    config = {"video": {"durationSec": 40.0, "width": 1920, "height": 1080, "sha256": "abc",
                        "sourcePath": str(tmp_path / "missing.mp4")}}
    monkeypatch.setattr(worker, "_build_deterministic_evidence",
                        lambda *args: (_ for _ in ()).throw(AssertionError("no-subject evidence")))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {"model": None, "done": 0, "failed": 0,
             "counts": {"match": 0, "no_match": 0, "uncertain": 0},
             "completed": set(), "currentSegment": 0}
    worker._process_cue_candidate(run_dir, config, request, event, line, None, cue, 0, [10.0, 10.2], 1,
                                  [], {}, state)
    result = list(iter_jsonl(run_dir / "results.jsonl"))[0]
    assert result["decision"] == "no_match"
    assert "no_subject" in result["qualityWarnings"]
    assert "cue_evidence_missing" not in result["qualityWarnings"]
    assert result["segments"] == []
    assert result["quality"] is None


def test_cue_verdict_uses_the_auxiliary_zone_when_no_line_crossing(tmp_path, monkeypatch):
    request = make_request(pois=[line_poi(), zone_poi()], cues=[cue_roi()])
    event, line, zone, cue = request.events[0], request.pois[0], request.pois[1], request.cues[0]
    config = {"video": {"durationSec": 40.0, "width": 1920, "height": 1080, "sha256": "abc",
                        "sourcePath": str(tmp_path / "missing.mp4")}}
    monkeypatch.setattr(worker, "_build_deterministic_evidence", lambda *args: [])
    monkeypatch.setattr(worker, "tracked_through", lambda *args: True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {"model": None, "done": 0, "failed": 0,
             "counts": {"match": 0, "no_match": 0, "uncertain": 0},
             "completed": set(), "currentSegment": 0}
    entry = make_candidate("zone-entry", 10.1, kind="zone.enter", poi="zone")
    leave = make_candidate("zone-leave", 12.0, kind="zone.leave", poi="zone")
    worker._process_cue_candidate(run_dir, config, request, event, line, zone, cue, 0, [10.0, 10.2], 1,
                                  [entry, leave], candidate_groups([entry, leave]), state)
    result = list(iter_jsonl(run_dir / "results.jsonl"))[0]
    assert result["decision"] == "no_match" and result["poiKind"] == "zone"
    assert result["cueEvidence"]["retreatSec"] == 12.0 and result["cueEvidence"]["retreatKind"] == "zone.leave"


def test_episode_dedup_is_recorded_in_the_result(tmp_path, monkeypatch):
    request = make_request(pois=[line_poi()], cues=[cue_roi()])
    event, line, cue = request.events[0], request.pois[0], request.cues[0]
    config = {"video": {"durationSec": 40.0, "width": 1920, "height": 1080, "sha256": "abc",
                        "sourcePath": str(tmp_path / "missing.mp4")}}
    monkeypatch.setattr(worker, "_build_deterministic_evidence", lambda *args: [])
    monkeypatch.setattr(worker, "tracked_through", lambda *args: True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {"model": None, "done": 0, "failed": 0,
             "counts": {"match": 0, "no_match": 0, "uncertain": 0},
             "completed": set(), "currentSegment": 0}
    entry = make_candidate("entry", 9.8)
    worker._process_cue_candidate(run_dir, config, request, event, line, None, cue, 0, [10.0, 13.0], 3,
                                  [entry], candidate_groups([entry]), state)
    result = list(iter_jsonl(run_dir / "results.jsonl"))[0]
    assert result["cueEvidence"]["rawTriggerCount"] == 3
    assert result["cueEvidence"]["interval"] == [10.0, 13.0]


def test_understand_processes_one_verdict_per_signal_episode(tmp_path, monkeypatch):
    from ultra_rules.vlm.qwen_worker import understand
    from ultra_rules.vlm.storage import append_json, read_json, write_json

    request = make_request(pois=[line_poi()], cues=[cue_roi()])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "request.json", {
        "modelPath": "/unused",
        "video": {"durationSec": 40.0, "width": 1920, "height": 1080, "sha256": "abc",
                  "sourcePath": str(tmp_path / "missing.mp4")},
        "request": request.model_dump(),
    })
    write_json(run_dir / "cues.json", {"cues": {"cue": {"intervals": [[10.0, 10.2], [11.0, 11.2]]}}})
    support = make_candidate("entry", 9.8)
    support.update(eventDefinitionIds=[], supportRuleIds=["e1"])
    append_json(run_dir / "candidates.jsonl", support)
    monkeypatch.setattr(worker, "_build_deterministic_evidence", lambda *args: [])
    monkeypatch.setattr(worker, "tracked_through", lambda *args: True)

    understand(run_dir)

    results = list(iter_jsonl(run_dir / "results.jsonl"))
    assert len(results) == 1
    assert results[0]["triggerKind"] == "cue.on"
    assert results[0]["decision"] == "match"
    assert results[0]["cueEvidence"]["interval"] == [10.0, 11.2]
    assert results[0]["cueEvidence"]["rawTriggerCount"] == 2
    assert read_json(run_dir / "summary.json")["resultCount"] == 1
    assert read_json(run_dir / "progress.json")["totalRequests"] == 1
