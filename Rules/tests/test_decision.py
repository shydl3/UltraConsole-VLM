import json
import sqlite3
from pathlib import Path

from ultra_rules.vlm.contracts import Poi, RunRequest
from ultra_rules.vlm.decision import (
    cue_state_in_window,
    evaluate,
    merge_cue_episodes,
    red_state,
    retreat_candidate_ids,
    retreat_event,
    select_cue_target,
)
from ultra_rules.vlm.storage import write_json


def request(**kwargs):
    return RunRequest(
        videoId="v1",
        pois=[{"id": "gate", "kind": "line", "points": [[0.5, 0.1], [0.5, 0.9]]}],
        cues=[{"id": "red", "points": [[0.4, 0.2], [0.6, 0.2], [0.6, 0.8], [0.4, 0.8]]}],
        events=[{"id": "e1", "description": "异常安检", "triggerKinds": ["line.cross"],
                 "cueIds": ["red"], "preSeconds": 2, "postSeconds": 4}],
        **kwargs,
    )


def candidate(cid, t, seg="1-1", direction="negative_to_positive", kind="line.cross", poi="gate"):
    return {"candidateId": cid, "trackId": 1, "trackSegmentId": seg, "poiId": poi,
            "triggerKind": kind, "direction": direction, "triggerSec": t, "confirmedSec": t + 0.2,
            "identityAmbiguous": False, "eventDefinitionIds": ["e1"]}


def write_observations(run_dir: Path, seg, times):
    db = sqlite3.connect(run_dir / "observations.sqlite")
    db.execute("CREATE TABLE observations (frame_index INTEGER PRIMARY KEY, time REAL NOT NULL, body TEXT NOT NULL)")
    for i, t in enumerate(times):
        body = json.dumps({"persons": [{"trackId": 1, "trackSegmentId": seg, "bbox": [0.4, 0.4, 0.6, 0.6]}]})
        db.execute("INSERT INTO observations VALUES (?, ?, ?)", (i, t, body))
    db.commit()
    db.close()


def test_red_state_on_and_off():
    timelines = {"red": {"intervals": [[10.0, 12.0]]}}
    assert red_state(timelines, ["red"], 11.0, 2, 4) == ("on", 10.0)
    assert red_state(timelines, ["red"], 20.0, 2, 4) == ("off", None)
    assert red_state(timelines, ["red"], 11.0, 2, 4)[0] == "on"


def test_retreat_candidate_ids_skips_only_the_retreat():
    rows = [candidate("c1", 1.0, direction="negative_to_positive"),
            candidate("c2", 2.0, direction="positive_to_negative"),
            candidate("c3", 10.0, direction="negative_to_positive")]
    skipped = retreat_candidate_ids(rows)
    assert skipped == {"c2"}


def test_retreat_candidate_ids_resets_after_pair():
    rows = [candidate("c1", 1.0, direction="negative_to_positive"),
            candidate("c2", 2.0, direction="positive_to_negative"),
            candidate("c3", 8.0, direction="negative_to_positive")]
    skipped = retreat_candidate_ids(rows)
    # c2 is the retreat of c1; c3 starts a new entry (7s later).
    assert skipped == {"c2"}


def test_evaluate_red_off(tmp_path):
    write_json(tmp_path / "cues.json", {"cues": {"red": {"intervals": []}}})
    write_observations(tmp_path, "1-1", [0.0, 0.5, 1.0, 1.5, 2.0])
    result = evaluate(candidate("c1", 1.0), request().events[0], request(), tmp_path, [candidate("c1", 1.0)])
    assert result["decision"] == "no_match"
    assert result["cueEvidence"]["state"] == "off"


def test_evaluate_red_on_retreated(tmp_path):
    write_json(tmp_path / "cues.json", {"cues": {"red": {"intervals": [[0.8, 6.0]]}}})
    write_observations(tmp_path, "1-1", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    entry = candidate("c1", 1.0, direction="negative_to_positive")
    retreat = candidate("c2", 2.0, direction="positive_to_negative")
    result = evaluate(entry, request().events[0], request(), tmp_path, [entry, retreat])
    assert result["decision"] == "no_match"
    assert result["cueEvidence"]["state"] == "on"


def test_evaluate_red_on_no_retreat_tracked(tmp_path):
    write_json(tmp_path / "cues.json", {"cues": {"red": {"intervals": [[0.8, 6.0]]}}})
    times = [0.0 + i * 0.5 for i in range(14)]  # 0..6.5s, continuous
    write_observations(tmp_path, "1-1", times)
    entry = candidate("c1", 1.0, direction="negative_to_positive")
    result = evaluate(entry, request().events[0], request(), tmp_path, [entry])
    assert result["decision"] == "match"


def test_evaluate_red_on_tracking_missing(tmp_path):
    write_json(tmp_path / "cues.json", {"cues": {"red": {"intervals": [[0.8, 6.0]]}}})
    write_observations(tmp_path, "1-1", [0.0, 0.5, 1.0, 1.5, 2.0])
    entry = candidate("c1", 1.0, direction="negative_to_positive")
    result = evaluate(entry, request().events[0], request(), tmp_path, [entry])
    # Occlusion after entry is normal: advisory warning, never uncertain.
    assert result["decision"] == "match"
    assert result["reviewRequired"] is False
    assert result["trackingGap"] is True


def test_cue_state_in_window_per_segment():
    timelines = {"red": {"intervals": [[30.0, 32.0]]}}
    assert cue_state_in_window(timelines, ["red"], 8.8, 24.8) == ("off", None)
    assert cue_state_in_window(timelines, ["red"], 22.8, 38.8) == ("on", 30.0)
    assert cue_state_in_window(timelines, ["red"], 36.8, 42.8) == ("off", None)
    assert cue_state_in_window({}, ["red"], 0, 10) == ("unknown", None)
    assert cue_state_in_window(timelines, [], 0, 10) == ("off", None)


def line_poi(**changes):
    values = {"id": "gate", "kind": "line", "points": [[0.5, 0.1], [0.5, 0.9]],
              "pairIndex": 1, "entryDirection": "negative_to_positive"}
    values.update(changes)
    return Poi.model_validate(values)


def zone_poi(**changes):
    values = {"id": "zone", "kind": "zone", "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
              "pairIndex": 1}
    values.update(changes)
    return Poi.model_validate(values)


def test_merge_cue_episodes_uses_the_first_trigger_as_anchor():
    assert merge_cue_episodes([[10.0, 10.2], [11.0, 11.2], [12.5, 13.0]], 3.0) == [
        {"interval": [10.0, 13.0], "rawTriggerCount": 3}]
    # 3.1s after the first trigger starts a new event.
    assert merge_cue_episodes([[10.0, 10.2], [13.1, 13.3]], 3.0) == [
        {"interval": [10.0, 10.2], "rawTriggerCount": 1},
        {"interval": [13.1, 13.3], "rawTriggerCount": 1}]
    # The anchor does not roll: 14.0 is 4s after the first trigger even though
    # it is 2s after the previous one.
    assert [e["rawTriggerCount"] for e in merge_cue_episodes([[10, 10.1], [12, 12.1], [14, 14.1]], 3.0)] == [2, 1]


def test_select_cue_target_prefers_the_entry_line_then_auxiliary_zone():
    line, zone = line_poi(), zone_poi()
    rows = [
        candidate("entry", 9.6, direction="negative_to_positive"),
        candidate("retreat", 10.4, direction="positive_to_negative"),
        candidate("entry2", 10.2, direction="negative_to_positive"),
        candidate("zone-entry", 10.1, kind="zone.enter", poi="zone"),
    ]
    # Opposite-direction crossings never become the entry target.
    assert select_cue_target(rows, line, zone, 9.0, 12.0, 10.0)["candidateId"] == "entry2"
    # Without a valid line crossing the auxiliary zone supplies the target.
    zone_rows = [candidate("zone-entry", 10.1, kind="zone.enter", poi="zone")]
    assert select_cue_target(zone_rows, line, zone, 9.0, 12.0, 10.0)["candidateId"] == "zone-entry"
    assert select_cue_target([], line, zone, 9.0, 12.0, 10.0) is None


def test_retreat_event_handles_line_and_zone():
    line_entry = candidate("entry", 10.0, direction="negative_to_positive")
    line_back = candidate("back", 12.0, direction="positive_to_negative")
    line_again = candidate("again", 11.0, direction="negative_to_positive")
    grouped = {("1-1", "gate"): [line_entry, line_back, line_again]}
    assert retreat_event(line_entry, grouped, 10.0, 15.0) is line_back
    assert retreat_event(line_entry, grouped, 10.0, 11.5) is None

    zone_entry = candidate("z-in", 10.0, kind="zone.enter", poi="zone")
    zone_out = candidate("z-out", 12.0, kind="zone.leave", poi="zone")
    grouped = {("1-1", "zone"): [zone_entry, zone_out]}
    assert retreat_event(zone_entry, grouped, 10.0, 15.0) is zone_out
    assert retreat_event(zone_entry, grouped, 10.0, 11.5) is None
