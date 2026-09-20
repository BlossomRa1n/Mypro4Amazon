"""Validate and compare the completed four-way controlled token experiment.

Uses NumPy only; no checkpoint/model or production dataset is loaded. Rate
differences are candidate minus reference. Validation failure exits nonzero
without writing a successful report.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compare_token_results import BOOTSTRAP_SAMPLES, MEAN_ATOL, SEED, load_metrics, paired


DATA_ID = "754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158"
SUITE = "controlled_din_token_v2"
VARIANTS = ("din_random", "din_svd", "semantic_concat_random", "semantic_concat_svd")
PAIRS = (("semantic_concat_random", "din_random"),
         ("semantic_concat_svd", "din_svd"),
         ("din_svd", "din_random"),
         ("semantic_concat_svd", "semantic_concat_random"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rate(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and 0 <= value <= 1, label + ": invalid rate")
    return value


def canonical(key):
    for prefix in ("mlp.", "head."):
        if key.startswith(prefix):
            return "fusion." + key[len(prefix):]
    return key


def canonical_state(audit, variant):
    state = audit["state_sha256"]
    require(isinstance(state, dict) and state, variant + ": empty initial state hashes")
    result = {}
    for key, value in state.items():
        require(isinstance(value, str) and len(value) == 64 and
                all(char in "0123456789abcdef" for char in value), variant + ": invalid tensor hash " + key)
        mapped = canonical(key)
        require(mapped not in result, variant + ": duplicate canonical tensor " + mapped)
        result[mapped] = value
    return result


def required_shared_keys():
    keys = {"brand_embedding.weight", "category_embedding.weight"}
    for index in (0, 2, 4):
        keys.update({f"attention_layer.mlp.{index}.weight", f"attention_layer.mlp.{index}.bias"})
    keys.update({"attention_layer.mlp.1.weight", "attention_layer.mlp.3.weight"})
    for index in (0, 4, 8):
        keys.add(f"fusion.{index}.bias")
        if index:
            keys.add(f"fusion.{index}.weight")
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            keys.add(f"fusion.{index + 1}.{suffix}")
        keys.add(f"fusion.{index + 2}.weight")
    keys.update({"fusion.12.weight", "fusion.12.bias"})
    return keys


def validate_initialization(run_dir):
    audits = {variant: read_json(run_dir / variant / "initial_state_audit.json") for variant in VARIANTS}
    states = {variant: canonical_state(audit, variant) for variant, audit in audits.items()}
    required = required_shared_keys()
    for variant, audit in audits.items():
        kind = "concat" if variant.startswith("semantic_concat_") else "din"
        mode = variant.rsplit("_", 1)[1]
        require(audit["kind"] == kind and audit["user_init"] == mode,
                variant + ": initial audit kind/user mode mismatch")
        require(audit["training_seed"] == 42, variant + ": training RNG seed is not 42")
        require(isinstance(audit["parameter_count"], int) and audit["parameter_count"] > 0,
                variant + ": invalid parameter count")
        copied = {canonical(key) for key in audit["copied_parameters_verified"]}
        require(required <= copied, variant + ": missing explicitly copied common tensors: " +
                ", ".join(sorted(required - copied)))
        require(required | {"item_embedding.weight", "user_embedding.weight", "fusion.0.weight"} <= states[variant].keys(),
                variant + ": initial state audit is incomplete")
        require(isinstance(audit["cpu_rng_sha256"], str) and len(audit["cpu_rng_sha256"]) == 64,
                variant + ": missing CPU RNG hash")
        cuda = audit["cuda_rng_sha256"]
        require(isinstance(cuda, list) and len(cuda) > 0 and
                all(isinstance(value, str) and len(value) == 64 for value in cuda),
                variant + ": formal server run requires CUDA RNG hashes")

    common = set.intersection(*(set(state) for state in states.values()))
    common -= {"user_embedding.weight", "fusion.0.weight"}
    for key in common:
        require(len({state[key] for state in states.values()}) == 1,
                "Initial common tensor differs across structures: " + key)
    for left, right in (("din_random", "semantic_concat_random"),
                         ("din_svd", "semantic_concat_svd")):
        require(states[left]["user_embedding.weight"] == states[right]["user_embedding.weight"],
                left + "/" + right + ": initial user embeddings differ")
    require(states["din_random"]["user_embedding.weight"] != states["din_svd"]["user_embedding.weight"],
            "Random and SVD user initializations unexpectedly coincide")
    for left, right in (("din_random", "din_svd"), ("semantic_concat_random", "semantic_concat_svd")):
        require(states[left].keys() == states[right].keys(), left + "/" + right + ": state keys differ")
        for key in states[left]:
            if key != "user_embedding.weight":
                require(states[left][key] == states[right][key],
                        left + "/" + right + ": extra initialization difference in " + key)
    first = audits[VARIANTS[0]]
    for variant, audit in audits.items():
        require(audit["cpu_rng_sha256"] == first["cpu_rng_sha256"], variant + ": initial CPU RNG differs")
        require(audit["cuda_rng_sha256"] == first["cuda_rng_sha256"], variant + ": initial CUDA RNG differs")
    return {"canonical_common_tensors": sorted(common), "first_layer_bias_explicitly_shared": True,
            "cpu_rng_sha256": first["cpu_rng_sha256"], "cuda_rng_sha256": first["cuda_rng_sha256"],
            "parameter_counts": {variant: audit["parameter_count"] for variant, audit in audits.items()},
            "audit_file_sha256": {variant: sha256(run_dir / variant / "initial_state_audit.json")
                                  for variant in VARIANTS}}


def identity_arrays(path, expected_users):
    with np.load(path, allow_pickle=False) as archive:
        values = {}
        for key in ("uid", "position"):
            require(key in archive, path.name + ": missing " + key)
            array = archive[key]
            require(array.ndim == 1 and len(array) == expected_users and array.dtype.kind in "iu",
                    path.name + ": invalid identity array " + key)
            values[key] = array.copy()
    require(len(np.unique(values["uid"])) == expected_users, path.name + ": duplicate users")
    return values


def compare_run(run_dir, expected_users=100000):
    run_dir = Path(run_dir).resolve()
    require(expected_users >= 2, "expected-users must be at least 2")
    require(not (run_dir / "SUPERSEDED.json").exists(), "Run is marked SUPERSEDED")
    monitor = read_json(run_dir / "monitor_status.json")
    require(monitor["status"] == "complete" and type(monitor["exit_code"]) is int and monitor["exit_code"] == 0,
            "Monitor must report complete with exit_code=0")
    completion = read_json(run_dir / "COMPLETED.json")
    require(completion["status"] == "complete" and completion["suite"] == SUITE and
            completion["variants"] == list(VARIANTS), "COMPLETED.json does not identify the four completed v2 groups")
    require(completion["results"] == "results.json", "Unexpected completed results path")
    manifest = read_json(run_dir / "suite_manifest.json")
    expected = {"suite": SUITE, "data_id": DATA_ID, "variants": list(VARIANTS),
                "candidate_budget": 75, "fusion_mode": "rrf", "fusion_weights": [2.0, 1.0, 0.7, 0.05],
                "negative_count": 4, "epochs": 3, "screen_users": expected_users, "test_users": expected_users,
                "seed": 42, "init_seed": 424242, "token_dim": 256, "itemcf_half_life_days": 180.}
    for key, value in expected.items():
        require(manifest.get(key) == value, f"suite_manifest.{key}: expected {value!r}, got {manifest.get(key)!r}")
    require(manifest["source"]["data_id"] == DATA_ID, "Source data_id differs")
    results = read_json(run_dir / "results.json")
    require(results["manifest"] == manifest, "results.json manifest differs from suite_manifest.json")
    require(set(results["variants"]) == set(VARIANTS), "results.json does not contain exactly the four variants")

    histories, selection, configs = {}, {}, {}
    for variant in VARIANTS:
        marker = read_json(run_dir / variant / "COMPLETED.json")
        require(marker["status"] == "complete" and marker["variant"] == variant, variant + ": incomplete marker")
        spec = marker["manifest"]
        for key, value in {"variant": variant, "seed": 42, "init_seed": 424242, "epochs": 3,
                           "negative_count": 4, "source": manifest["source"]}.items():
            require(spec[key] == value, variant + ": variant manifest differs in " + key)
        require(spec["kind"] == ("concat" if variant.startswith("semantic_concat_") else "din") and
                spec["user_init"] == variant.rsplit("_", 1)[1], variant + ": wrong structure/initialization label")
        config = spec["model_config"]
        require(config["embed_dim"] == 256 and config["brand_embed_dim"] == 64 and
                config["hidden_dims"] == [256, 128, 64], variant + ": nonformal model topology")
        if variant.startswith("semantic_concat_"):
            require(config["token_dim"] == 256 and config["fusion"] == "concat", variant + ": wrong token topology")
        configs[variant] = {key: value for key, value in config.items() if key not in ("token_dim", "fusion")}
        history = read_json(run_dir / variant / "history.json")
        require(len(history) == 3 and [entry["epoch"] for entry in history] == [1, 2, 3],
                variant + ": history must contain epochs 1,2,3")
        require(history == results["variants"][variant]["history"], variant + ": results/history mismatch")
        for entry in history:
            require(type(entry["steps"]) is int and entry["steps"] > 0, variant + ": invalid steps")
            require(isinstance(entry["loss"], (int, float)) and math.isfinite(entry["loss"]), variant + ": invalid loss")
            rate(entry["metrics"]["din"]["hr"], variant + ": epoch HR")
            require(entry["metrics"]["users"] == expected_users and entry["metrics"]["din"]["users"] == expected_users,
                    variant + ": epoch development user count differs")
        require(len({entry["steps"] for entry in history}) == 1, variant + ": steps change across epochs")
        histories[variant] = history
        # Python max retains the first occurrence, matching the runner's strict >.
        best = max(history, key=lambda entry: entry["metrics"]["din"]["hr"])
        require(marker["best_hr5"] == best["metrics"]["din"]["hr"], variant + ": completed best HR differs")
        selection[variant] = {"epoch": best["epoch"], "hr5": best["metrics"]["din"]["hr"]}
    require(all(config == configs[VARIANTS[0]] for config in configs.values()), "Common model configuration differs")
    require(len({entry["steps"] for history in histories.values() for entry in history}) == 1,
            "Training step count differs across groups")

    output = {"status": "validated", "run_dir": str(run_dir), "data_id": DATA_ID,
              "suite": SUITE, "expected_users": expected_users, "difference_direction": "candidate minus reference",
              "metric_units": "fractions, not percentage points", "bootstrap_samples": BOOTSTRAP_SAMPLES,
              "bootstrap_seed": SEED, "numpy_version": np.__version__,
              "validation": {"protocol": "passed", "monitor_exit": "complete/0", "all_four_groups": "complete",
                             "epochs": 3, "steps_per_epoch": histories[VARIANTS[0]][0]["steps"],
                             "selection_rule": "earliest epoch attaining maximum development HR@5",
                             "selected_epochs": selection, "initialization": validate_initialization(run_dir),
                             "metric_mean_absolute_tolerance": MEAN_ATOL,
                             "suite_manifest_sha256": sha256(run_dir / "suite_manifest.json")},
              "limitations": [
                  "Single training seed: intervals condition on fixed models and omit training-seed variability.",
                  "No multiple-comparison correction; user-level intervals assume independent users.",
                  "Development users selected the best epoch, so screen intervals are descriptive, not independent validation.",
                  "Test is held out from epoch selection in this suite; prior historical reuse is not ruled out.",
                  "Identity and pool_hit arrays are verified. Without candidate IDs, identical candidate membership cannot be independently proven from these metric archives.",
                  "Initial RNG states agree; architecture-specific dropout calls do not produce identical masks across structures.",
              ], "splits": {}}
    for split in ("screen", "test"):
        loaded, identities = {}, {}
        for variant in VARIANTS:
            arrays, summary = load_metrics(run_dir, variant, split)
            require(summary["users"] == expected_users, variant + "/" + split + ": final user count differs")
            label = f"{variant}_{split}_final"
            metrics = read_json(run_dir / f"{label}_metrics.json")
            require(metrics["candidate_pool"]["k"] == 75, label + ": candidate budget differs")
            require(metrics == results["variants"][variant][split], label + ": metrics/results mismatch")
            identities[variant] = identity_arrays(run_dir / f"{label}_users.npz", expected_users)
            if split == "screen":
                best = histories[variant][selection[variant]["epoch"] - 1]
                require(metrics["din"] == best["metrics"]["din"], variant + ": final screen metrics do not match selected epoch")
            loaded[variant] = arrays, summary
        first_ids = identities[VARIANTS[0]]
        for variant, values in identities.items():
            for key in ("uid", "position"):
                require(np.array_equal(values[key], first_ids[key]), split + "/" + variant + ": paired " + key + " differs")
        output["splits"][split] = {
            "role": "epoch_selection" if split == "screen" else "held_out_from_epoch_selection_in_this_suite",
            "row_identity_verified": True,
            "models": {variant: summary for variant, (_, summary) in loaded.items()},
            "comparisons": {f"{candidate}_vs_{reference}": paired(loaded[candidate][0], loaded[reference][0])
                            for candidate, reference in PAIRS}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-users", type=int, default=100000,
                        help="Expected users in each split; override only for synthetic acceptance fixtures")
    args = parser.parse_args()
    try:
        result = compare_run(args.run_dir, expected_users=args.expected_users)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Controlled comparison failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Validated four groups and saved eight paired comparisons to {args.output.resolve()}")


if __name__ == "__main__":
    main()
