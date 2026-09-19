"""Select on paired development predictions, then supervise one final evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np


DEFAULT_FUSION_WEIGHTS = [1.5, 1.0, 0.7, 0.05]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_failures(baseline, candidate):
    for run in (baseline, candidate):
        status_path = run / "monitor_status.json"
        if status_path.exists():
            state = read_json(status_path)
            require(state.get("status") in ("starting", "running", "complete", "validation-only"),
                    f"{run.name} monitor failed or has unknown status: {state.get('status')}")
            require(state.get("exit_code") in (None, 0), f"{run.name} monitor exited unsuccessfully")
        marker = run / "COMPLETED.json"
        if marker.exists():
            require(read_json(marker).get("status") in ("complete", "validation-only"),
                    f"{run.name} has unsuccessful completion marker")
    for path in (candidate / "chain_status", candidate / ".chain_status",
                 candidate.with_name(candidate.name + ".chain_status")):
        if path.exists():
            raise ValueError(f"Chain stopped ({path}): {path.read_text(encoding='utf-8').strip()}")


def wait_for_results(baseline, candidate, poll_seconds, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while True:
        check_failures(baseline, candidate)
        if all((run / "COMPLETED.json").exists() for run in (baseline, candidate)):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for baseline and candidate completion")
        time.sleep(min(poll_seconds, remaining, 60.))


def load_result(run, candidate=False):
    result = read_json(run / "results.json")
    marker = read_json(run / "COMPLETED.json")
    require(result.get("protocol") == "future-window", f"{run.name}: wrong protocol")
    require(isinstance(result.get("validation_only"), bool), f"{run.name}: missing result scope")
    validation_only = result["validation_only"]
    expected_splits = {"val"} if validation_only else {"val", "test"}
    require(not candidate or validation_only, "Candidate must contain validation-only results")
    require(set(result.get("splits", {})) == expected_splits, f"{run.name}: result split mismatch")
    require(marker.get("status") == ("validation-only" if validation_only else "complete"),
            f"{run.name}: completion scope mismatch")
    require(marker.get("validation_only") is validation_only and
            set(marker.get("splits", [])) == expected_splits,
            f"{run.name}: completion metadata mismatch")
    require(marker.get("results") == result, f"{run.name}: stale or inconsistent result marker")
    require(not validation_only or not (run / "test_predictions.json").exists(),
            f"{run.name}: validation-only run contains test predictions")
    for key in ("data_id", "run_identity"):
        require(isinstance(result.get(key), str) and result[key], f"{run.name}: missing {key}")
    spec = read_json(run / "run_manifest.json")
    require(fingerprint(spec) == result["run_identity"], f"{run.name}: run identity mismatch")
    require(spec["args"].get("protocol") == result["protocol"], f"{run.name}: manifest protocol mismatch")
    require(bool(spec["args"].get("validation_only", False)) == validation_only,
            f"{run.name}: manifest scope mismatch")
    require(spec["args"].get("fusion_mode", "quota") == result.get("fusion_mode") and
            spec["args"].get("itemcf_half_life_days", 0.) == result.get("itemcf_half_life_days"),
            f"{run.name}: fusion recipe mismatch")
    expected_weights = spec["args"].get("fusion_weights", DEFAULT_FUSION_WEIGHTS)
    require(result.get("fusion_weights", DEFAULT_FUSION_WEIGHTS) == expected_weights,
            f"{run.name}: fusion weights mismatch")
    return result, spec


def validate_sources(baseline, base, trial, base_spec, trial_spec):
    require(base["data_id"] == trial["data_id"], "Baseline/candidate data_id mismatch")
    for key in ("protocol", "future_test_users", "sample_users", "seed", "hist_len", "dim", "candidates", "cf_neighbors"):
        require(base_spec["args"].get(key) == trial_spec["args"].get(key), f"Source argument mismatch: {key}")
    for result in (base, trial):
        source = result.get("checkpoint_source", {})
        require(isinstance(source.get("run_dir"), str) and
                Path(source["run_dir"]).resolve() == baseline.resolve(), "Wrong checkpoint source directory")
        require(isinstance(source.get("run_identity"), str) and source["run_identity"],
                "Missing checkpoint source run identity")
    require(Path(trial_spec["args"].get("eval_only_run", "")).resolve() == baseline.resolve(),
            "Candidate did not evaluate the baseline checkpoints")
    require(not base_spec["args"].get("eval_only_run"), "Baseline must own its training checkpoints")

    # Read only checkpoint metadata; mmap avoids loading all tensor storage into RAM.
    import torch
    identities, metadata = [], {}
    for kind in ("v2", "din"):
        path = baseline / (kind + "_best.pth")
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        manifest = saved.get("manifest", {})
        require(manifest.get("kind") == kind and manifest.get("data_id") == base["data_id"],
                f"{kind} checkpoint data/kind mismatch")
        identity = manifest.get("run_identity")
        require(isinstance(identity, str) and identity, f"{kind} checkpoint lacks identity")
        identities.append(identity)
        stat = path.stat()
        metadata[kind] = {"manifest": manifest, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        del saved
    require(trial["checkpoint_source"].get("checkpoint_run_identities") == identities,
            "Candidate checkpoint identities differ from source files")
    require(trial["checkpoint_source"]["run_identity"] in (base["run_identity"], identities[0]),
            "Candidate source run identity mismatch")
    require(base["checkpoint_source"]["run_identity"] == base["run_identity"],
            "Baseline source identity mismatch")
    if "checkpoint_run_identities" in base["checkpoint_source"]:
        require(base["checkpoint_source"]["checkpoint_run_identities"] == identities,
                "Baseline checkpoint identities differ from source files")
    return metadata


def load_predictions(path):
    rows = read_json(path)
    require(isinstance(rows, list) and rows, f"{path}: empty/malformed predictions")
    indexed = {}
    for row in rows:
        uid, targets, items = row.get("user_id"), row.get("targets"), row.get("items")
        require(isinstance(uid, str) and uid and uid not in indexed, "Missing or duplicate user_id")
        require(isinstance(targets, list) and targets and all(isinstance(x, str) for x in targets),
                f"{uid}: complete targets required")
        require(len(set(targets)) == len(targets) == row.get("target_count"), f"{uid}: invalid target_count")
        require(isinstance(items, list) and len(items) <= 5 and all(isinstance(x, str) for x in items)
                and len(set(items)) == len(items), f"{uid}: malformed Top-5 items")
        require(row.get("candidate_hit") in (0, 1), f"{uid}: missing candidate_hit")
        require(not set(items).intersection(targets) or row["candidate_hit"] == 1,
                f"{uid}: Top-5 hit absent from candidate pool")
        indexed[uid] = row
    return indexed


def metrics(rows):
    hr, ndcg, recall, cover = [], [], [], []
    for row in rows:
        targets = set(row["targets"])
        gains = [int(item in targets) for item in row["items"]]
        found = sum(gains)
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = sum(1. / math.log2(rank + 2) for rank in range(min(5, len(targets))))
        hr.append(int(found > 0))
        ndcg.append(dcg / ideal)
        recall.append(found / len(targets))
        cover.append(row["candidate_hit"])
    n = len(rows)
    return {"din": {"hr": sum(hr) / n, "ndcg": sum(ndcg) / n, "recall": sum(recall) / n,
                    "users": n, "k": 5},
            "candidate_pool": {"hr": sum(cover) / n, "users": n}}, hr


def validate_reported_metrics(result, rows):
    recomputed, _ = metrics(list(rows.values()))
    reported = result["splits"]["val"]
    for section, names in (("din", ("hr", "ndcg", "recall")), ("candidate_pool", ("hr",))):
        require(reported[section]["users"] == len(rows), "Prediction count does not match reported users")
        for name in names:
            require(math.isclose(reported[section][name], recomputed[section][name], abs_tol=1e-10),
                    f"Predictions disagree with reported {section}.{name}")
    require(reported["din"]["k"] == 5, "DIN evaluation is not Top-5")


def compare_predictions(base_rows, trial_rows):
    users = sorted(trial_rows)
    require(set(users).issubset(base_rows), "Candidate includes users absent from baseline validation")
    for uid in users:
        require(set(base_rows[uid]["targets"]) == set(trial_rows[uid]["targets"]),
                f"{uid}: full targets differ")
    baseline_metrics, base_hits = metrics([base_rows[uid] for uid in users])
    trial_metrics, trial_hits = metrics([trial_rows[uid] for uid in users])
    gained = sum(b == 0 and t == 1 for b, t in zip(base_hits, trial_hits))
    lost = sum(b == 1 and t == 0 for b, t in zip(base_hits, trial_hits))
    both = sum(b == 1 and t == 1 for b, t in zip(base_hits, trial_hits))
    n = len(users)
    counts = [gained, lost, both, n - gained - lost - both]
    draws = np.random.default_rng(42).multinomial(n, np.asarray(counts) / n, size=10000)
    interval = np.quantile((draws[:, 0] - draws[:, 1]) / n, [.025, .975]).tolist()
    accepted = (trial_metrics["din"]["hr"] > baseline_metrics["din"]["hr"] and
                trial_metrics["din"]["ndcg"] >= baseline_metrics["din"]["ndcg"])
    return {"status": "accepted" if accepted else "rejected",
            "rule": "Strict paired DIN HR@5 improvement AND DIN NDCG@5 non-regression",
            "aligned_users": n, "baseline_available_users": len(base_rows),
            "alignment_sha256": fingerprint([[uid, sorted(trial_rows[uid]["targets"])] for uid in users]),
            "baseline_val": baseline_metrics, "candidate_val": trial_metrics,
            "paired_hr5": {"gained": gained, "lost": lost, "both_hit": both,
                           "neither_hit": counts[3], "difference": (gained - lost) / n,
                           "ci95": interval, "method": "paired-user percentile bootstrap; 10000 draws; seed 42",
                           "interpretation": "Descriptive uncertainty only; no significance claim or selection gate"}}


def select(baseline, candidate):
    base, base_spec = load_result(baseline)
    trial, trial_spec = load_result(candidate, candidate=True)
    source = validate_sources(baseline, base, trial, base_spec, trial_spec)
    base_rows = load_predictions(baseline / "val_predictions.json")
    trial_rows = load_predictions(candidate / "val_predictions.json")
    validate_reported_metrics(base, base_rows)
    validate_reported_metrics(trial, trial_rows)
    require(base["splits"]["val"]["candidate_pool"]["k"] ==
            trial["splits"]["val"]["candidate_pool"]["k"], "Candidate evaluation budgets differ")
    decision = compare_predictions(base_rows, trial_rows)
    decision.update(protocol="future-window", scope="paired-validation-only", data_id=base["data_id"],
                    baseline_run_identity=base["run_identity"], candidate_run_identity=trial["run_identity"],
                    checkpoint_source=source)
    return decision, trial_spec["args"]


def link_or_copy(source, target):
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def launch_final(args, decision, recipe):
    require(decision["status"] == "accepted", "Only accepted selections may launch final evaluation")
    baseline, final = Path(args.baseline_dir).resolve(), Path(args.final_dir).resolve()
    require(final not in (baseline, Path(args.candidate_dir).resolve()), "Final directory must be separate")
    source = Path(args.source_dir).resolve()
    require((source / "tools/monitor_training.py").is_file() and (source / "code/run_baseline.py").is_file(),
            "Final evaluator source is missing")
    require(Path(recipe["data_dir"]).resolve() == Path(args.data_dir).resolve(), "Final data directory mismatch")
    selection_id = fingerprint(decision)
    lock = final / "AUTO_LAUNCH.json"
    if lock.exists():
        prior = read_json(lock)
        require(prior.get("selection_id") == selection_id, "Final directory belongs to another selection")
        require(prior.get("status") == "started", "Previous final launch is unresolved; inspect it before retrying")
        return {"final_launch_status": "already-started", "final_monitor_pid": prior["monitor_pid"]}
    require(not final.exists() or not any(final.iterdir()), "Refusing to reuse a nonempty final directory")
    final.mkdir(parents=True, exist_ok=True)
    claim = {"selection_id": selection_id, "status": "claimed", "selector_pid": os.getpid()}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for name in ("data.pkl", "svd.npy", "itemcf.pkl"):
            require((baseline / name).is_file(), f"Missing baseline asset: {name}")
            link_or_copy(baseline / name, final / name)
        runner = [sys.executable, "-B", str(source / "code/run_baseline.py")]
        final_args = dict(recipe, run_dir=str(final), data_dir=str(Path(args.data_dir).resolve()),
                          eval_only_run=str(baseline), final_users=recipe["future_test_users"],
                          fusion_weights=recipe.get("fusion_weights", DEFAULT_FUSION_WEIGHTS))
        for key in ("data_dir", "run_dir", "protocol", "sample_users", "eval_users", "final_users",
                    "future_test_users", "epochs", "dim", "hist_len", "negatives", "candidates",
                    "cf_neighbors", "fusion_mode", "itemcf_half_life_days", "v2_batch", "din_batch",
                    "workers", "seed", "eval_only_run"):
            runner.extend(["--" + key.replace("_", "-"), str(final_args[key])])
        runner.extend(["--fusion-weights", *(str(value) for value in final_args["fusion_weights"])])
        command = [sys.executable, "-B", str(source / "tools/monitor_training.py"), "--run-dir", str(final),
                   "--log", str(final / "train.log"), "--poll-seconds", "60", "--", *runner]
        with (final / "auto_launch.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=source, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        claim.update(status="started", monitor_pid=process.pid, command=command)
        write_json(lock, claim)
        return {"final_launch_status": "started", "final_monitor_pid": process.pid}
    except Exception as error:
        claim.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(lock, claim)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-dir", "candidate-dir", "final-dir", "data-dir", "source-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.)
    parser.add_argument("--timeout-seconds", type=float, default=43200.)
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.poll_seconds, args.timeout_seconds)):
        parser.error("poll-seconds and timeout-seconds must be finite and positive")
    baseline, candidate = Path(args.baseline_dir).resolve(), Path(args.candidate_dir).resolve()
    decision_path = candidate / "AUTO_SELECT.json"
    try:
        require(baseline != candidate, "Baseline and candidate directories must differ")
        wait_for_results(baseline, candidate, args.poll_seconds, args.timeout_seconds)
        decision, recipe = select(baseline, candidate)
        decision["final_dir"] = str(Path(args.final_dir).resolve())
        if decision["status"] == "accepted":
            decision.update(launch_final(args, decision, recipe))
        write_json(decision_path, decision)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        write_json(decision_path, {"status": "failed", "reason": f"{type(error).__name__}: {error}",
                                   "final_launch_status": "not-confirmed",
                                   "policy": "fail closed; inspect any AUTO_LAUNCH.json before retrying"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
