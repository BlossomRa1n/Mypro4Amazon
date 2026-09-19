"""Bundle completed token evidence before releasing a running shutdown wrapper."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import tarfile
import time


def build_bundle(run):
    results = json.loads((run / "results.json").read_text())
    completed = json.loads((run / "COMPLETED.json").read_text())
    manifest = json.loads((run / "suite_manifest.json").read_text())
    if completed.get("status") != "complete" or completed.get("results") != results:
        raise ValueError("Completion marker and results differ")
    if results["manifest"] != manifest or set(results["variants"]) != {"concat", "rankmixer"}:
        raise ValueError("Unexpected suite manifest or variant set")
    for variant in results["variants"].values():
        if len(variant["history"]) != manifest["epochs"]:
            raise ValueError("Incomplete training history")
        for split, count in (("screen", manifest["screen_users"]), ("test", manifest["test_users"])):
            if variant[split]["users"] != count:
                raise ValueError("Incomplete final evaluation")
    files = sorted(p for p in run.rglob("*") if p.is_file()
                   and p.suffix in {".json", ".npz", ".log", ".txt"}
                   and "cache" not in p.relative_to(run).parts
                   and p.name not in {"archive_manifest.json", "archive_finalize.log"})
    # Snapshot bytes once: the heartbeat can update while the bundle is built.
    snapshots = {p.relative_to(run).as_posix(): p.read_bytes() for p in files}
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in snapshots.items()}
    snapshots["archive_manifest.json"] = json.dumps(checksums, indent=2).encode()
    bundle = run / "evidence.tar.gz"
    import io
    with tarfile.open(bundle.with_suffix(".tmp"), "w:gz") as output:
        for name, content in snapshots.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            output.addfile(info, io.BytesIO(content))
    bundle.with_suffix(".tmp").replace(bundle)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (run / "evidence.sha256").write_text(digest + "\n")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--wrapper-pid", type=int, required=True)
    parser.add_argument("--archive-grace-seconds", type=int, default=900)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    training_command = Path(f"/proc/{args.training_pid}/cmdline").read_bytes()
    wrapper_command = Path(f"/proc/{args.wrapper_pid}/cmdline").read_bytes()
    if b"run_token_experiments.py" not in training_command or str(run).encode() not in training_command:
        raise ValueError("Training PID does not belong to this run")
    if b"auto_shutdown_token_suite.sh" not in wrapper_command:
        raise ValueError("Unexpected shutdown wrapper PID")
    os.kill(args.wrapper_pid, signal.SIGSTOP)
    print("Shutdown wrapper paused; training continues", flush=True)
    process_stat = Path(f"/proc/{args.training_pid}/stat")
    while process_stat.exists():
        if process_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
        time.sleep(900)
    digest = build_bundle(run)
    print("Evidence validated and bundled: " + digest, flush=True)
    deadline = time.monotonic() + args.archive_grace_seconds
    acknowledgement = run / "ARCHIVE_ACK.sha256"
    while time.monotonic() < deadline:
        if acknowledgement.exists() and acknowledgement.read_text().strip() == digest:
            break
        time.sleep(5)
    acknowledged = acknowledgement.exists() and acknowledgement.read_text().strip() == digest
    (run / "archive_shutdown_status.json").write_text(json.dumps({
        "evidence_sha256": digest, "local_archive_acknowledged": acknowledged,
        "released_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print("Releasing shutdown wrapper; local archive acknowledged=" + str(acknowledged), flush=True)
    os.sync()
    os.kill(args.wrapper_pid, signal.SIGCONT)


if __name__ == "__main__":
    main()
