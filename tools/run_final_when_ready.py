"""Wait for the candidate queue, then acquire its GPU lock and run final evaluation."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--gpu-lock", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--wait-hours", type=float, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.wait_hours <= 0:
        parser.error("command and positive wait-hours required")
    status_path = Path(args.status)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    def status(action, **details):
        value = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "action": action, **details}
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temp.replace(status_path)
        print(json.dumps(value), flush=True)
    deadline = time.monotonic() + args.wait_hours * 3600
    status("waiting_for_candidate_completion")
    while not (Path(args.candidate_run) / "CANDIDATE_COMPLETED.json").exists():
        if time.monotonic() >= deadline:
            status("wait_timeout", candidate_run=args.candidate_run)
            return 2
        time.sleep(30)
    with Path(args.gpu_lock).open("a") as lock:
        status("waiting_for_gpu_lock")
        fcntl.flock(lock, fcntl.LOCK_EX)
        status("final_evaluation_start", command=command)
        result = subprocess.run(command, check=False)
        status("final_evaluation_process_exited", exit_code=result.returncode,
               note="No shutdown here; agent must verify local sync and final artifacts")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
