from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

RULES_SRC = Path(__file__).resolve().parents[2] / "Rules/src"
if str(RULES_SRC) not in sys.path:
    sys.path.insert(0, str(RULES_SRC))

from ultra_rules.vlm.cli import prepare as prepare  # noqa: E402
from ultra_rules.vlm.contracts import AnalysisParameters as AnalysisParameters  # noqa: E402
from ultra_rules.vlm.contracts import CueDefinition as CueDefinition  # noqa: E402
from ultra_rules.vlm.contracts import EventDefinition as EventDefinition  # noqa: E402
from ultra_rules.vlm.contracts import Identifier as Identifier  # noqa: E402
from ultra_rules.vlm.contracts import IntegrityZone as IntegrityZone  # noqa: E402
from ultra_rules.vlm.contracts import Poi as Poi  # noqa: E402
from ultra_rules.vlm.contracts import RunRequest as RunRequest  # noqa: E402
from ultra_rules.vlm.contracts import StrictModel as StrictModel  # noqa: E402
from ultra_rules.vlm.runner import process_identity  # noqa: E402
from ultra_rules.vlm.storage import TERMINAL as TERMINAL  # noqa: E402
from ultra_rules.vlm.storage import iter_jsonl as iter_jsonl  # noqa: E402
from ultra_rules.vlm.storage import read_json as read_json  # noqa: E402
from ultra_rules.vlm.storage import write_json as write_json  # noqa: E402

# Re-entrant: the queue scheduler pumps while run creation may already hold it.
launch_lock = threading.RLock()


def owner_alive(run_dir: Path) -> bool:
    """True while the worker process recorded in owner.json is still running."""
    path = run_dir / "owner.json"
    if not path.exists():
        return False
    try:
        owner = read_json(path)
        return process_identity(owner["pid"]) == owner["identity"]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def status(run_dir: Path):
    result = read_json(run_dir / "status.json")
    # Historical runs predate videoDurationSec; request.json carries the probed
    # video metadata from prepare(), so backfill without touching status.json.
    if result.get("videoDurationSec") is None:
        request_path = run_dir / "request.json"
        if request_path.exists():
            try:
                result["videoDurationSec"] = read_json(request_path).get("video", {}).get("durationSec")
            except (OSError, ValueError):
                pass
    owner_path = run_dir / "owner.json"
    owner = read_json(owner_path) if owner_path.exists() else None
    alive = owner_alive(run_dir)
    cancelled = (run_dir / "cancel.requested").exists()
    if result["state"] not in TERMINAL:
        if owner and not alive:
            if cancelled:
                result.update(state="cancelled", finishedAt=time.time(), updatedAt=time.time())
            else:
                result.update(state="interrupted", error="Worker exited without a terminal status",
                              updatedAt=time.time())
            write_json(run_dir / "status.json", result)
        elif not owner and cancelled:
            # Cancelled while still queued: no worker will ever pick it up.
            result.update(state="cancelled", finishedAt=time.time(), updatedAt=time.time())
            write_json(run_dir / "status.json", result)
        elif (not owner and result["state"] != "queued"
              and time.time() - (run_dir / "status.json").stat().st_mtime > 30):
            result.update(state="interrupted", error="Worker never started")
            write_json(run_dir / "status.json", result)
    if (run_dir / "progress.json").exists():
        result["progress"] = read_json(run_dir / "progress.json")
    # User-facing task name: original video name + minute-precision timestamp.
    # The uuid run id stays internal (directory / API path), never displayed.
    library = run_dir / "library.json"
    if library.exists():
        result["displayName"] = read_json(library).get("displayName") or result.get("fileName")
    else:
        started = result.get("startedAt")
        base = result.get("fileName") or run_dir.name
        result["displayName"] = f"{base}_{time.strftime('%Y%m%d_%H%M', time.localtime(started))}" if started else base
    if cancelled and result["state"] not in TERMINAL:
        result["state"] = "cancelling"
    end = result.get("finishedAt") or time.time()
    running_at = result.get("startedRunningAt")
    if running_at is not None:
        # Future runs: queue wait is deliberately excluded from duration.
        result["durationSec"] = max(0.0, end - running_at)
    elif "queuedAt" in result:
        # New run still waiting for a worker: there is no compute duration yet.
        result["durationSec"] = None
    elif result.get("startedAt") is not None:
        # Historical runs without the split clock keep their old semantics.
        result["durationSec"] = max(0.0, end - result["startedAt"])
    return result


def launch(run_dir: Path, python: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RULES_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    with (run_dir / "coordinator.log").open("w") as log:
        proc = subprocess.Popen([python, "-m", "ultra_rules.vlm.runner", str(run_dir)],
                                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    write_json(run_dir / "owner.json", {"pid": proc.pid, "identity": process_identity(proc.pid)})

    def reap():
        proc.wait()
        status(run_dir)
    threading.Thread(target=reap, daemon=True).start()


def cancel(run_dir: Path):
    current = status(run_dir)
    if current["state"] in TERMINAL:
        return current
    (run_dir / "cancel.requested").touch()
    owner = read_json(run_dir / "owner.json") if (run_dir / "owner.json").exists() else None
    if current["state"] != "queued" and owner and process_identity(owner["pid"]) == owner["identity"]:
        # Coordinator handles TERM, terminates and reaps its active phase process.
        try:
            os.kill(owner["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    return status(run_dir)
