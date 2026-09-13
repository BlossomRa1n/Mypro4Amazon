"""Verify a downloaded release and reproduce its two sample inference outputs."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run = Path(args.run).resolve()
    release = run / "release"
    audit = run / "local_verification"
    audit.mkdir(exist_ok=True)
    manifest = read_json(release / "bundle.json")
    server_gate = read_json(run / "BUNDLE_VERIFIED.json")
    require(digest(release / "bundle.json") == server_gate["bundle_manifest_sha256"],
            "Server bundle manifest hash differs")
    require(manifest["format_version"] == 1, "Unsupported bundle format")
    for relative, expected in manifest["sha256"].items():
        path = (release / relative).resolve()
        require(path.is_relative_to(release), "Unsafe manifest path")
        require(digest(path) == expected, "Artifact hash differs: " + relative)
    print("Bundle file hashes verified", flush=True)

    locked = read_json(run / "LOCKED_SCHEME.json")
    fresh = read_json(run / "fresh_acceptance.json")
    test = read_json(run / "final_test.json")
    require(fresh["accepted"], "Expected accepted frozen scheme")
    require(locked["scheme"] == fresh["final_scheme"] == test["scheme"] == manifest["choice"],
            "Final scheme differs across stages")
    require(manifest["test"] == test, "Exported test metadata differs")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    sequence = [event["action"] for event in events]
    stages = ["scheme_locked", "fresh_validation_complete", "final_test_complete", "final_evaluation_complete"]
    indices = [sequence.index(stage) for stage in stages]
    require(indices == sorted(indices), "Final evaluation events out of order")

    result = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "bundle_manifest_sha256": digest(release / "bundle.json"),
              "verified_files": len(manifest["sha256"]), "scheme_consistent": True,
              "event_order_verified": stages, "device": args.device, "inference": {}}
    import numpy as np
    development_users, fresh_users = set(), set()
    checked_evaluations = 0
    for directory in (run / "evaluations").iterdir():
        metrics = read_json(directory / "metrics.json")
        with np.load(directory / "users.npz") as arrays:
            hits = np.any(arrays["top5"] == arrays["target"][:, None], axis=1)
            ndcg = np.where(hits, 1. / np.log2(np.maximum(arrays["rank"], 1) + 1), 0.)
            require(np.array_equal(hits, arrays["hit5"]), "Saved hits differ from recommendations")
            require(np.allclose(ndcg, arrays["ndcg5"], rtol=0, atol=1e-12), "Saved NDCG differs")
            require(abs(float(hits.mean()) - metrics["hr5"]) < 1e-12, "HR metric differs")
            require(abs(float(ndcg.mean()) - metrics["ndcg5"]) < 1e-12, "NDCG metric differs")
            require(abs(float(arrays["pool_hit"].mean()) - metrics["coverage"]) < 1e-12,
                    "Coverage metric differs")
            if directory.name.endswith(("_screen", "_confirm")):
                development_users.update(arrays["uid"].tolist())
            if "fresh_validation" in directory.name:
                fresh_users.update(arrays["uid"].tolist())
        checked_evaluations += 1
    require(len(development_users) == locked["fresh_offset"], "Development user count differs")
    require(len(fresh_users) == locked["fresh_users"], "Reserved user count differs")
    require(not development_users.intersection(fresh_users), "Reserved users overlap development users")
    result["metric_recomputation"] = {"evaluations": checked_evaluations,
                                       "development_users": len(development_users),
                                       "reserved_users": len(fresh_users), "overlap": 0}
    print("Metrics and reserved-user separation verified", flush=True)
    for split, reference in (("test", "bundle_test_verification.csv"),
                             ("latest", "latest_recommendations_sample.csv")):
        expected = read_csv(run / reference)
        output = audit / (split + ".csv")
        command = [sys.executable, str(release / "code" / "optimized_inference.py"),
                   "--bundle", str(release), "--output", str(output), "--split", split,
                   "--max-users", str(len(expected)), "--device", args.device]
        with (audit / (split + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=release, stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = read_csv(output)
        require(actual == expected, "Local " + split + " predictions differ from server")
        require(all(len(set(row["article_" + str(i)] for i in range(1, 6))) == 5
                    and all(row["article_" + str(i)] for i in range(1, 6)) for row in actual),
                "Invalid Top-5 output")
        if split == "test":
            complete = {row["user_id"]: row for row in read_csv(run / "test_recommendations.csv")}
            require(len(complete) == test["proposal"]["users"], "Test output user count differs")
            require(all(complete[row["user_id"]] == row for row in actual), "Full test CSV differs")
        result["inference"][split] = {"users": len(actual), "predictions_equal": True,
                                        "sha256": digest(output)}
        print(split + " independent inference matches server", flush=True)
    import torch
    import numpy
    result["environment"] = {"python": sys.version, "torch": torch.__version__,
                             "numpy": numpy.__version__}
    result["status"] = "passed"
    (audit / "LOCAL_VERIFIED.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
