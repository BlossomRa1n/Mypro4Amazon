"""Compare a validation-only fusion run and launch final evaluation if it wins."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def link_or_copy(source, target):
    target = Path(target)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    baseline = Path(args.baseline_dir)
    candidate = Path(args.candidate_dir)
    final = Path(args.final_dir)
    decision_path = candidate / "AUTO_SELECT.json"
    while not (candidate / "COMPLETED.json").exists():
        baseline_marker = baseline / "COMPLETED.json"
        chain_status = candidate / "chain_status"
        if baseline_marker.exists():
            marker = read_json(baseline_marker)
            if marker.get("status") not in ("complete", "validation-only"):
                decision_path.write_text(json.dumps({
                    "status": "rejected", "reason": "baseline failed",
                    "baseline_marker": marker,
                }, indent=2), encoding="utf-8")
                return 0
        if chain_status.exists():
            decision_path.write_text(json.dumps({
                "status": "rejected", "reason": chain_status.read_text(encoding="utf-8").strip(),
            }, indent=2), encoding="utf-8")
            return 0
        time.sleep(max(1, args.poll_seconds))

    candidate_marker = read_json(candidate / "COMPLETED.json")
    baseline_results = read_json(baseline / "results.json")
    candidate_results = read_json(candidate / "results.json")
    base = baseline_results["splits"].get("val")
    trial = candidate_results["splits"].get("val")
    if not base or not trial or candidate_marker.get("status") != "validation-only":
        decision = {"status": "rejected", "reason": "missing validation-only result"}
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        return 0

    # The business metric is DIN HR@5. Candidate coverage is the tie-breaker.
    din_improved = trial["din"]["hr"] > base["din"]["hr"]
    coverage_improved = trial["candidate_pool"]["hr"] > base["candidate_pool"]["hr"]
    din_equal = trial["din"]["hr"] == base["din"]["hr"]
    accepted = din_improved or (din_equal and coverage_improved and
                                trial["din"]["ndcg"] >= base["din"]["ndcg"])
    decision = {
        "status": "accepted" if accepted else "rejected",
        "rule": "DIN HR@5 primary; candidate-pool HR@100 tie-breaker; DIN NDCG non-regression",
        "baseline_val": base,
        "candidate_val": trial,
        "final_dir": str(final),
    }
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if not accepted:
        return 0

    final.mkdir(parents=True, exist_ok=True)
    for name in ("data.pkl", "svd.npy", "itemcf.pkl"):
        link_or_copy(baseline / name, final / name)
    command = [
        "/root/miniconda3/bin/python", "-B", "tools/monitor_training.py",
        "--run-dir", str(final), "--log", str(final / "train.log"),
        "--poll-seconds", "60", "--", "/root/miniconda3/bin/python", "-B",
        "code/run_baseline.py", "--data-dir", args.data_dir,
        "--run-dir", str(final), "--protocol", "future-window",
        "--future-test-users", "100000", "--eval-users", "10000",
        "--final-users", "100000", "--epochs", "1", "--dim", "256",
        "--v2-batch", "1024", "--din-batch", "256", "--workers", "4",
        "--fusion-mode", "rrf", "--itemcf-half-life-days", "90",
        "--eval-only-run", str(baseline),
    ]
    log = (final / "auto_launch.log").open("ab", buffering=0)
    process = subprocess.Popen(command, cwd=args.source_dir, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    decision["final_monitor_pid"] = process.pid
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
