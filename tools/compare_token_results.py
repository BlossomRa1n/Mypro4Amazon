"""Compare archived token-suite user metrics without loading models or PyTorch.

Requires baseline_{screen,test}_{users.npz,metrics.json} and
{concat,rankmixer}_{screen,test}_final_{users.npz,metrics.json}.
All differences are candidate minus reference, in rate units (not percent).
"""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


BOOTSTRAP_SAMPLES = 10000
SEED = 42
# User NDCG arrays are stored as float32, while JSON metrics use float64.
MEAN_ATOL = 1e-7
MODELS = ("baseline", "concat", "rankmixer")
PAIRS = (("concat", "baseline"), ("rankmixer", "baseline"),
         ("rankmixer", "concat"))


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    return value


def load_metrics(run_dir, model, split):
    label = f"{model}_{split}" + ("" if model == "baseline" else "_final")
    array_path = run_dir / f"{label}_users.npz"
    metric_path = run_dir / f"{label}_metrics.json"
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {}
        for key in ("hit5", "ndcg5", "pool_hit"):
            if key not in archive:
                raise ValueError(f"{array_path.name}: missing {key}")
            values = archive[key]
            if values.ndim != 1 or values.dtype.kind not in "biuf":
                raise ValueError(f"{array_path.name}: {key} must be a numeric 1-D array")
            if not np.isfinite(values).all():
                raise ValueError(f"{array_path.name}: {key} contains nonfinite values")
            if key == "ndcg5":
                if np.any((values < 0) | (values > 1)):
                    raise ValueError(f"{array_path.name}: ndcg5 must be in [0, 1]")
            elif not np.isin(values, (0, 1)).all():
                raise ValueError(f"{array_path.name}: {key} must contain only 0 or 1")
            arrays[key] = values.astype(np.float64 if key == "ndcg5" else np.int8)
    size = len(arrays["hit5"])
    if size < 2 or any(len(values) != size for values in arrays.values()):
        raise ValueError(f"{array_path.name}: arrays must have equal lengths of at least 2")
    if np.any(arrays["hit5"] > arrays["pool_hit"]):
        raise ValueError(f"{array_path.name}: a Top-5 hit is absent from the candidate pool")
    if not np.array_equal(arrays["ndcg5"] > 0, arrays["hit5"] == 1):
        raise ValueError(f"{array_path.name}: hit5 and positive ndcg5 disagree")

    metrics = json.loads(metric_path.read_text(encoding="utf-8-sig"))
    try:
        for location, count in (("users", metrics["users"]),
                                ("din.users", metrics["din"]["users"]),
                                ("candidate_pool.users", metrics["candidate_pool"]["users"])):
            if _number(count, f"{metric_path.name}: {location}") != size:
                raise ValueError(f"{metric_path.name}: {location} does not match array length {size}")
        if metrics["din"]["k"] != 5:
            raise ValueError(f"{metric_path.name}: din.k must be 5")
        checks = (("hit5", metrics["din"]["hr"]),
                  ("ndcg5", metrics["din"]["ndcg"]),
                  ("pool_hit", metrics["candidate_pool"]["hr"]))
        for key, expected in checks:
            expected = _number(expected, f"{metric_path.name}: {key} mean")
            actual = float(arrays[key].mean())
            if not math.isclose(actual, expected, rel_tol=0., abs_tol=MEAN_ATOL):
                raise ValueError(f"{metric_path.name}: {key} array mean {actual} "
                                 f"does not match metric {expected}")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{metric_path.name}: invalid metric structure ({exc})") from exc
    summary = {"users": size, "hr5": float(arrays["hit5"].mean()),
               "ndcg5": float(arrays["ndcg5"].mean()),
               "pool_hr": float(arrays["pool_hit"].mean()),
               "arrays_file": array_path.name, "metrics_file": metric_path.name}
    return arrays, summary


def paired(candidate, reference):
    """Summarize aligned, validated per-user arrays without retaining them."""
    size = len(candidate["hit5"])
    if size != len(reference["hit5"]):
        raise ValueError("Compared models have different user counts")
    if not np.array_equal(candidate["pool_hit"], reference["pool_hit"]):
        raise ValueError("Compared models have different per-user pool_hit arrays")
    delta = candidate["hit5"] - reference["hit5"]
    counts = np.array([(delta == value).sum() for value in (-1, 0, 1)])
    draws = np.random.default_rng(SEED).multinomial(
        size, counts / size, size=BOOTSTRAP_SAMPLES)
    hr_interval = np.quantile((draws[:, 2] - draws[:, 0]) / size, [0.025, 0.975])
    ndcg_delta = candidate["ndcg5"] - reference["ndcg5"]
    ndcg_mean = float(ndcg_delta.mean())
    standard_error = float(ndcg_delta.std(ddof=1) / math.sqrt(size))
    margin = NormalDist().inv_cdf(0.975) * standard_error
    return {
        "users": size, "gained": int(counts[2]), "lost": int(counts[0]),
        "unchanged": int(counts[1]), "hr5_delta": float(delta.mean()),
        "hr5_delta_ci95": hr_interval.tolist(),
        "hr5_ci_method": "paired multinomial bootstrap percentile (10000 draws, seed 42)",
        "ndcg5_delta": ndcg_mean,
        "ndcg5_delta_ci95": [ndcg_mean - margin, ndcg_mean + margin],
        "ndcg5_delta_standard_error": standard_error,
        "ndcg5_ci_method": "paired normal approximation; mean +/- 1.959964 * sample_sd / sqrt(n)",
    }


def compare_run(run_dir):
    run_dir = Path(run_dir).resolve()
    result = {
        "run_dir": str(run_dir), "difference_direction": "candidate minus reference",
        "metric_units": "fraction, not percentage points",
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": SEED,
        "validation": {"metric_mean_absolute_tolerance": MEAN_ATOL,
                       "checks": ["array lengths", "array values", "metric means and user counts",
                                  "identical per-user pool_hit within each split"]},
        "limitations": [
            "Single training seed: intervals condition on fixed models and omit training-seed variability.",
            "No multiple-comparison correction is applied.",
            "Screen users selected the best epoch; screen is not independent validation.",
            "Test is held out from epoch selection in this suite; this does not establish it was never used previously.",
            "Intervals assume independent users and aligned row order. Archives contain no user IDs, so "
            "equal pool_hit arrays and matching means cannot prove user identity or identical candidate membership.",
        ],
        "splits": {},
    }
    for split in ("screen", "test"):
        loaded = {model: load_metrics(run_dir, model, split) for model in MODELS}
        comparisons = {}
        for candidate, reference in PAIRS:
            try:
                comparison = paired(loaded[candidate][0], loaded[reference][0])
            except ValueError as exc:
                raise ValueError(f"{split} {candidate}_vs_{reference}: {exc}") from exc
            comparisons[f"{candidate}_vs_{reference}"] = comparison
        result["splits"][split] = {
            "role": "epoch_selection_not_independent_validation" if split == "screen"
                    else "held_out_from_epoch_selection_in_this_suite",
            "models": {model: summary for model, (_, summary) in loaded.items()},
            "comparisons": comparisons,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Write aggregate comparisons to this JSON file")
    args = parser.parse_args()
    try:
        result = compare_run(args.run_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Comparison failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Saved six paired comparisons to {args.output.resolve()}")


if __name__ == "__main__":
    main()
