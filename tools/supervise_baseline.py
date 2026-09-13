"""Record the lifecycle of an isolated baseline run, without shutting down the host."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [sys.executable, "-u", str(source / "code/run_baseline.py"), "--run-dir", str(run), *arguments]
    state = {"command": command, "source": str(source), "supervisor_pid": os.getpid(),
             "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def save():
        temporary = run / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, run / "status.json")

    with (run / "train.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="running", child_pid=process.pid)
        save()
        code = process.wait()
    complete = code == 0 and (run / "COMPLETED.json").exists()
    state.update(status="complete" if complete else "failed", exit_code=code,
                 ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
    return 0 if complete else code or 1


if __name__ == "__main__":
    sys.exit(main())
