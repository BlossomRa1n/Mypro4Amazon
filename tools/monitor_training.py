"""Run a training command with periodic heartbeat and completion state."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.poll_seconds < 1:
        parser.error("a command and positive poll interval are required")

    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log).resolve() if args.log else run / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = run / "monitor_status.json"

    state = {
        "status": "starting",
        "command": command,
        "run_dir": str(run),
        "log": str(log_path),
        "monitor_pid": os.getpid(),
        "started_utc": utc_now(),
        "poll_seconds": args.poll_seconds,
    }

    def save():
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temp, status_path)

    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT)
        state.update(status="running", child_pid=process.pid, last_heartbeat_utc=utc_now())
        save()
        while process.poll() is None:
            state.update(last_heartbeat_utc=utc_now(), log_bytes=log_path.stat().st_size,
                         completed_marker=(run / "COMPLETED.json").exists())
            save()
            time.sleep(args.poll_seconds)
        code = process.returncode

    complete = code == 0 and (run / "COMPLETED.json").exists()
    state.update(status="complete" if complete else "failed", exit_code=code,
                 completed_marker=(run / "COMPLETED.json").exists(),
                 log_bytes=log_path.stat().st_size,
                 ended_utc=utc_now(), last_heartbeat_utc=utc_now())
    save()
    return 0 if complete else code or 1


if __name__ == "__main__":
    sys.exit(main())
