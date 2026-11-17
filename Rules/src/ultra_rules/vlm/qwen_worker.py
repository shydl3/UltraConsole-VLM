from __future__ import annotations

import time
from pathlib import Path

from .contracts import RunRequest
from .decision import candidate_groups, cue_state_in_window, evaluate, load_cue_timelines, retreat_candidate_ids
from .prompting import PROMPT_VERSION, integrity_prompt, prompt
from .results import aggregate_result, apply_answer, complete_json_answer, decision_from_conditions, parse_answer
from .sampling import build_sample, windows
from .segmenting import read_manifest
from .storage import VERSION, append_json, digest, iter_jsonl, progress, read_json, write_json

GENERATION_OPTIONS = {"do_sample": False, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "max_time": 120}


class JsonAnswerStop:
    """Stop the single-answer request once its JSON object is complete."""

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, _scores, **_kwargs):
        text = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
        return complete_json_answer(text)


class Qwen:
    def __init__(self, model_path: str):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        torch.set_num_threads(8)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Qwen requires the verified CUDA/BF16 environment (VLM_PYTHON)")
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", local_files_only=True).eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.top_k = 50
        self.runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
                        "gpu": torch.cuda.get_device_name(), "dtype": str(next(self.model.parameters()).dtype),
                        "loadSeconds": time.perf_counter() - started, "promptVersion": PROMPT_VERSION,
                        "generationOptions": GENERATION_OPTIONS, "stopOnCompleteJson": True}

    def infer(self, frames, metadata, question, max_tokens):
        from transformers import StoppingCriteriaList

        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "video", "video": "sampled-video"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[frames], video_metadata=[metadata],
                                do_sample_frames=False, do_resize=False, return_tensors="pt").to("cuda")
        stopping = StoppingCriteriaList([JsonAnswerStop(self.processor.tokenizer, inputs.input_ids.shape[1])])
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                            stopping_criteria=stopping, **GENERATION_OPTIONS)
        result = generated[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(result, skip_special_tokens=True)[0]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        json_complete = complete_json_answer(raw)
        finished = json_complete or (result.shape[1] > 0 and int(result[0, -1]) in eos)
        stats = {"inputTokens": inputs.input_ids.shape[1], "outputTokens": result.shape[1],
                 "inferenceSeconds": time.perf_counter() - started,
                 "peakAllocatedGiB": torch.cuda.max_memory_allocated() / 2**30,
                 "peakReservedGiB": torch.cuda.max_memory_reserved() / 2**30,
                 "generationTruncated": not finished,
                 "finishReason": "json_complete" if json_complete else "eos" if finished else "length_or_timeout"}
        del inputs, generated, result
        return raw, stats


def _is_entry(candidate, pois):
    poi = pois.get(candidate.get("poiId"))
    if poi is not None and poi.kind == "line" and poi.entryDirection and candidate.get("direction") != poi.entryDirection:
        return False
    return True


def _cue_veto(part, deterministic):
    if not deterministic:
        return
    if deterministic["cueEvidence"].get("state") == "off" and part.get("decision") in ("match", "uncertain"):
        part.update(decision="no_match", modelDecision=part.get("modelDecision", part.get("decision")))
        part["qualityWarnings"] = list(dict.fromkeys(part.get("qualityWarnings", []) + ["cue_override_red_off"]))
        part["reason"] = "确定性检测显示门框红灯未亮起，覆盖模型判断。"


def _build_deterministic_evidence(run_dir, config, request, candidate, event, result_id, deterministic):
    segments = []
    for index, (start, end) in enumerate(windows(candidate["triggerSec"], event, config["video"]["durationSec"])):
        sample_id = f"{result_id}-{index}"
        frames, metadata, manifest = build_sample(run_dir, config, request, candidate, start, end, sample_id)
        quality = manifest["quality"]
        quality.update(preContextClipped=candidate["triggerSec"] - event.preSeconds < 0,
                       postContextClipped=candidate["triggerSec"] + event.postSeconds > config["video"]["durationSec"])
        write_json(run_dir / "samples" / f"{sample_id}.json", manifest)
        part = {"window": manifest["window"], "quality": quality,
                "sampleManifest": f"samples/{sample_id}.json", "previewPath": f"previews/{sample_id}.jpg",
                "decision": deterministic["decision"], "reason": deterministic["reason"], "modelDecisionConfidence": None}
        segments.append(part)
        del frames
    return segments


def _process_candidate(run_dir, config, request, candidate, events, state, grouped, timelines):
    """Run deterministic + VLM verdicts for one candidate across its events.

    Mutates state (model/done/failed/counts) and appends global-time results to
    run_dir/results.jsonl incrementally so the API can stream partial output.
    """
    for event_id in candidate["eventDefinitionIds"]:
        event = events[event_id]
        result_id = f"{candidate['candidateId']}-{event_id}"
        if result_id in state["completed"]:
            continue
        output = {"schemaVersion": "1", "runId": run_dir.name, "videoId": request.videoId, **candidate,
                  "eventDefinitionId": event_id, "description": event.description, "resultId": result_id,
                  "eventCriteria": event.criteria or [event.description],
                  "confidenceType": "model_self_reported_uncalibrated", "modelDecisionConfidence": None,
                  "inferenceStatus": "succeeded", "decision": "uncertain", "segments": []}
        try:
            deterministic = evaluate(candidate, event, request, run_dir, grouped=grouped, timelines=timelines)
            if deterministic is not None and not deterministic["reviewRequired"] and not request.dryRun:
                output["segments"] = _build_deterministic_evidence(run_dir, config, request, candidate, event, result_id, deterministic)
                output.update(deterministicDecision=deterministic["decision"],
                              cueEvidence=deterministic["cueEvidence"],
                              decision=deterministic["decision"], reason=deterministic["reason"],
                              modelDecision=deterministic["decision"], inferenceStatus="deterministic")
                if deterministic.get("trackingGap"):
                    output["qualityWarnings"] = ["tracking_gap"]
                if output["segments"]:
                    sel = output["segments"][0]
                    for key in ("window", "quality", "previewPath", "sampleManifest"):
                        output[key] = sel[key]
            else:
                cue_evidence = deterministic["cueEvidence"] if deterministic else None
                for index, (start, end) in enumerate(windows(candidate["triggerSec"], event, config["video"]["durationSec"])):
                    sample_id = f"{result_id}-{index}"
                    frames, metadata, manifest = build_sample(run_dir, config, request, candidate, start, end, sample_id)
                    quality = manifest["quality"]
                    quality.update(preContextClipped=candidate["triggerSec"] - event.preSeconds < 0,
                                   postContextClipped=candidate["triggerSec"] + event.postSeconds > config["video"]["durationSec"])
                    write_json(run_dir / "samples" / f"{sample_id}.json", manifest)
                    part = {"window": manifest["window"], "quality": quality,
                            "sampleManifest": f"samples/{sample_id}.json", "previewPath": f"previews/{sample_id}.jpg"}
                    output["segments"].append(part)
                    # Per-window deterministic gate: a segment whose signal ROI never
                    # lit is no_match without asking the model, so occlusion-driven
                    # model uncertainty can never survive an all-off window.
                    if event.cueIds and cue_state_in_window(timelines, event.cueIds, start, end)[0] == "off":
                        part.update(decision="no_match", reason="该检测窗口内信号区未亮起，确定性门禁判不匹配",
                                    modelDecisionConfidence=None, qualityWarnings=[])
                        del frames
                        continue
                    if request.dryRun:
                        part.update(decision="uncertain", reason="仅抽帧预览，未调用模型", modelDecisionConfidence=None, downgradeReasons=[])
                    else:
                        if state["model"] is None:
                            state["model"] = Qwen(config["modelPath"])
                            write_json(run_dir / "runtime.json", state["model"].runtime)
                        model = state["model"]
                        question = prompt(candidate, event, manifest, cue_evidence=cue_evidence)
                        raw_dir = run_dir / "raw"
                        raw_dir.mkdir(exist_ok=True)
                        part["requestFingerprint"] = digest({"version": VERSION, "prompt": question, "manifest": manifest,
                                                             "model": config["modelPath"], "maxTokens": request.maxNewTokens,
                                                             "runtime": model.runtime, "video": config["video"]["sha256"]})
                        (raw_dir / f"{sample_id}.prompt.txt").write_text(question, encoding="utf-8")
                        last_error = None
                        attempts = []
                        for attempt in range(2):
                            (raw_dir / f"{sample_id}-{attempt}.prompt.txt").write_text(question, encoding="utf-8")
                            raw, stats = model.infer(frames, metadata, question, request.maxNewTokens)
                            raw_path = f"raw/{sample_id}-{attempt}.txt"
                            (run_dir / raw_path).write_text(raw, encoding="utf-8")
                            attempts.append({"rawOutputPath": raw_path, **stats})
                            try:
                                if stats["generationTruncated"] and not complete_json_answer(raw):
                                    raise ValueError("generation_truncated_or_timeout")
                                answer = parse_answer(raw, start, end, len(event.criteria or [event.description]),
                                                      trigger=candidate["triggerSec"], check_timestamps=False)
                                if not answer.videoDescription or not answer.videoDescription.strip():
                                    raise ValueError("missing_video_description")
                                apply_answer(part, answer, sample_times=manifest["sampleTimesSeconds"],
                                             source_indices=manifest["sourceFrameIndices"], trigger=candidate["triggerSec"])
                                _cue_veto(part, deterministic)
                                part["rawOutputPath"] = raw_path
                                last_error = None
                                break
                            except (ValueError, TypeError) as exc:
                                last_error = exc
                                attempts[-1]["parseError"] = str(exc)
                                question = prompt(candidate, event, manifest, retry=True, cue_evidence=cue_evidence)
                        part["attempts"] = attempts
                        if last_error:
                            write_json(run_dir / "samples" / f"{sample_id}-failure.json", {"attempts": attempts, "error": str(last_error)})
                            raise ValueError(f"invalid_output: {last_error}")
                    del frames
                if deterministic is not None:
                    output["deterministicDecision"] = deterministic["decision"]
                    output["cueEvidence"] = deterministic["cueEvidence"]
                aggregate_result(output)
                if request.dryRun:
                    output["inferenceStatus"] = "dry_run"
            state["counts"][output["decision"]] += 1
        except Exception as exc:
            state["failed"] += 1
            output.update(inferenceStatus="failed", decision="uncertain", error=f"{type(exc).__name__}: {exc}",
                          reason="推理或证据处理失败，不代表事件未发生。")
            append_json(run_dir / "errors.jsonl", {"resultId": result_id, "error": output["error"]})
            if state["model"]:
                state["model"].torch.cuda.empty_cache()
        append_json(run_dir / "results.jsonl", output)
        state["completed"].add(result_id)
        state["done"] += 1
        progress(run_dir, completedRequests=state["done"], failedRequests=state["failed"],
                 decisions=state["counts"], currentSegment=state.get("currentSegment", 0))


def _integrity_pairs(run_dir, request, events):
    """Zone-level integrity verdicts paired with their bound event definitions.

    Every zone yields either one pair per confirmed change interval (positive)
    or a single negative pair when the region stayed static for the whole video,
    so the preset rule is always visible in the records.
    """
    path = run_dir / "integrity.json"
    if not path.exists():
        return []
    zones = {z.id: z for z in request.integrityZones}
    pairs = []
    for zone_id, timeline in read_json(path).get("zones", {}).items():
        zone = zones.get(zone_id)
        if zone is None:
            continue
        intervals = timeline.get("intervals", [])
        for event in events.values():
            if "integrity.change" not in event.triggerKinds or (event.integrityIds and zone_id not in event.integrityIds):
                continue
            if intervals:
                for index, interval in enumerate(intervals):
                    pairs.append((zone, index, interval, event, timeline))
            else:
                pairs.append((zone, -1, None, event, timeline))
    return pairs


def _integrity_review_frames(run_dir, evidence, onset, confirmed, duration):
    """Before/after crops as a two-frame sample for the optional VLM review."""
    import math

    import numpy as np
    from PIL import Image

    frames = []
    for key in ("before", "after"):
        path = run_dir / evidence.get(key, "")
        if not path.is_file():
            return None, None
        frames.append(np.asarray(Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)))
    metadata = {"fps": 1000.0, "duration": duration, "total_num_frames": math.ceil(duration * 1000),
                "frames_indices": [round(onset * 1000), round(confirmed * 1000)]}
    return np.stack(frames), metadata


def _process_integrity(run_dir, config, request, zone, index, interval, event, state, timeline=None):
    """Emit one deterministic integrity.change verdict per zone interval.

    Positive intervals become match warnings; a zone without any confirmed
    interval becomes a single no_match record so negatives stay reviewable.
    Integrity verdicts come from measurable pixel evidence, not the model; the
    optional VLM review only re-checks structural vs lighting/occlusion changes.
    """
    timeline = timeline or {}
    negative = interval is None
    result_id = f"integrity-{zone.id}-{'none' if negative else index}-{event.id}"
    if result_id in state["completed"]:
        return
    duration = float(config["video"]["durationSec"])
    label = zone.label or zone.id
    if negative:
        onset = float(timeline.get("baselineSec") or 0.0)
        confirmed = onset
        offset, peak = None, 0.0
        evidence = {}
        reason = (f"完整性检测区「{label}」全程未检测到持续画面变化："
                  f"{timeline.get('sampleCount', 0)} 个抽样样本与基线比对（自 {onset:.1f}s 起监测）。")
        decision = "no_match"
    else:
        onset, confirmed = interval["onsetSec"], interval["confirmedSec"]
        offset, peak = interval.get("offsetSec"), interval["peakRatio"]
        evidence = interval.get("evidence") or {}
        reason = (f"完整性检测区「{label}」自 {onset:.1f}s 起出现持续画面变化："
                  f"峰值变化占比 {peak:.1%}，持续≥{zone.confirmSec:g}s 后确认")
        reason += f"，变化持续至 {offset:.1f}s。" if offset is not None else "，变化持续至视频结束。"
        decision = "match"
    start = max(0.0, onset - event.preSeconds)
    end = min(duration, (offset if offset is not None else confirmed) + event.postSeconds)
    if end <= start:
        end = min(duration, start + 1.0)
    quality = {"targetVisibleFraction": 1.0, "identityAmbiguous": False, "timeUnreliable": False,
               "contextTruncated": False, "medianTargetHeightPx": 256.0, "uniqueFrameCount": 4,
               "observationContextMissing": False, "evidenceTimeOutsideWindow": False}
    part = {"window": {"startSec": start, "endSec": end}, "quality": quality,
            "previewPath": evidence.get("context") or evidence.get("after"), "sampleManifest": None,
            "decision": decision, "reason": reason, "modelDecisionConfidence": None, "qualityWarnings": []}
    output = {"schemaVersion": "1", "runId": run_dir.name, "videoId": request.videoId,
              "resultId": result_id, "poiId": zone.id, "triggerKind": "integrity.change", "triggerSec": onset,
              "eventDefinitionId": event.id, "description": event.description,
              "eventCriteria": event.criteria or [event.description],
              "confidenceType": "deterministic_cv", "modelDecisionConfidence": None,
              "inferenceStatus": "deterministic", "decision": decision, "deterministicDecision": decision,
              "integrityEvidence": {"zoneId": zone.id, "label": label,
                                    "onsetSec": None if negative else onset,
                                    "offsetSec": offset, "confirmedSec": None if negative else confirmed,
                                    "peakRatio": peak, "sampleFps": zone.sampleFps,
                                    "confirmSec": zone.confirmSec, "changed": not negative,
                                    "sampleCount": timeline.get("sampleCount", 0),
                                    "baselineSec": timeline.get("baselineSec"), **evidence},
              "segments": [part], "window": part["window"], "quality": quality,
              "previewPath": part["previewPath"], "reason": reason,
              "qualityWarnings": [], "downgradeReasons": []}
    try:
        if not negative and zone.vlmReview and not request.dryRun:
            frames, metadata = _integrity_review_frames(run_dir, evidence, onset, confirmed, duration)
            if frames is None:
                part["qualityWarnings"].append("integrity_evidence_missing")
            else:
                if state["model"] is None:
                    state["model"] = Qwen(config["modelPath"])
                    write_json(run_dir / "runtime.json", state["model"].runtime)
                model = state["model"]
                question = integrity_prompt(zone, event, interval)
                raw_dir = run_dir / "raw"
                raw_dir.mkdir(exist_ok=True)
                criteria_count = len(event.criteria or [event.description])
                last_error = None
                for attempt in range(2):
                    (raw_dir / f"{result_id}-{attempt}.prompt.txt").write_text(question, encoding="utf-8")
                    raw, stats = model.infer(frames, metadata, question, request.maxNewTokens)
                    raw_path = f"raw/{result_id}-{attempt}.txt"
                    (run_dir / raw_path).write_text(raw, encoding="utf-8")
                    try:
                        if stats["generationTruncated"] and not complete_json_answer(raw):
                            raise ValueError("generation_truncated_or_timeout")
                        answer = parse_answer(raw, start, end, criteria_count, trigger=onset, check_timestamps=False)
                        conj, warnings = decision_from_conditions(answer.conditions, criteria_count)
                        part.update(decision=conj or answer.decision, modelDecision=answer.decision,
                                    modelDecisionConfidence=answer.confidence, reason=answer.reason,
                                    videoDescription=answer.videoDescription,
                                    conditions=[c.model_dump() for c in answer.conditions],
                                    qualityWarnings=list(dict.fromkeys(part["qualityWarnings"] + warnings)),
                                    rawOutputPath=raw_path)
                        last_error = None
                        break
                    except (ValueError, TypeError) as exc:
                        last_error = exc
                del frames
                if last_error:
                    # A failed review never cancels the measurable CV warning.
                    part["qualityWarnings"].append("integrity_vlm_review_failed")
                else:
                    output["inferenceStatus"] = "succeeded"
        output["decision"] = part["decision"]
        output["reason"] = part["reason"]
        output["modelDecision"] = part.get("modelDecision")
        output["modelDecisionConfidence"] = part.get("modelDecisionConfidence")
        output["videoDescription"] = part.get("videoDescription")
        output["conditions"] = part.get("conditions", [])
        output["qualityWarnings"] = part["qualityWarnings"]
        state["counts"][output["decision"]] += 1
    except Exception as exc:
        state["failed"] += 1
        output.update(inferenceStatus="failed", decision="uncertain", error=f"{type(exc).__name__}: {exc}",
                      reason="完整性证据处理失败，不代表区域未发生变化。")
        append_json(run_dir / "errors.jsonl", {"resultId": result_id, "error": output["error"]})
        if state["model"]:
            state["model"].torch.cuda.empty_cache()
    append_json(run_dir / "results.jsonl", output)
    state["completed"].add(result_id)
    state["done"] += 1
    progress(run_dir, completedRequests=state["done"], failedRequests=state["failed"],
             decisions=state["counts"], currentSegment=state.get("currentSegment", 0))


def understand(run_dir: Path):
    config = read_json(run_dir / "request.json")
    request = RunRequest.model_validate(config["request"])
    candidates = list(iter_jsonl(run_dir / "candidates.jsonl"))
    events = {e.id: e for e in request.events if e.enabled}
    pois = {p.id: p for p in request.pois}
    retreat_ids = retreat_candidate_ids(candidates)
    entry_candidates = [c for c in candidates if c["candidateId"] not in retreat_ids and _is_entry(c, pois)]
    integrity_pairs = _integrity_pairs(run_dir, request, events)
    total = sum(len(c["eventDefinitionIds"]) for c in entry_candidates) + len(integrity_pairs)
    # Logical time-range segments give progress granularity and crash recovery;
    # analysis stays in global time so results.jsonl needs no offset merge.
    segments = read_manifest(run_dir).get("segments") or [{"index": 0, "startSec": 0.0, "endSec": float("inf")}]
    completed = {row["resultId"] for row in iter_jsonl(run_dir / "results.jsonl") if "resultId" in row}
    progress(run_dir, stage="understanding", totalRequests=total, completedRequests=len(completed),
             failedRequests=0, segmentCount=len(segments))
    if not total:
        write_json(run_dir / "summary.json", {"resultCount": 0, "failedRequests": 0, "reason": "no_candidates", "dryRun": request.dryRun})
        return
    state = {"model": None, "done": len(completed), "failed": 0,
             "counts": {"match": 0, "no_match": 0, "uncertain": 0}, "completed": completed, "currentSegment": 0}
    grouped = candidate_groups(candidates)
    timelines = load_cue_timelines(run_dir) if any(event.cueIds for event in events.values()) else {}
    # Deterministic integrity warnings stream first: they need no model warm-up
    # and remain available even when the run has no tracking candidates at all.
    for zone, index, interval, event, timeline in integrity_pairs:
        _process_integrity(run_dir, config, request, zone, index, interval, event, state, timeline)
    for segment in segments:
        bucket = [c for c in entry_candidates if segment["startSec"] <= c["triggerSec"] < segment["endSec"]]
        if not bucket:
            continue
        state["currentSegment"] = segment["index"]
        progress(run_dir, currentSegment=segment["index"], segmentCount=len(segments))
        for candidate in bucket:
            _process_candidate(run_dir, config, request, candidate, events, state, grouped, timelines)
    write_json(run_dir / "summary.json", {"resultCount": state["done"], "failedRequests": state["failed"],
                                          "decisions": state["counts"], "dryRun": request.dryRun})
