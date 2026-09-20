"""Exercise the actual controlled runner on tiny synthetic data.

Run from any directory with the repository's PyTorch environment::

    python tools/check_controlled_token.py --device both

No production dataset or checkpoint is read.  Temporary checkpoints are
removed after the check.  Evaluation metrics below are smoke placeholders,
not ranking-quality measurements.
"""
import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from baseline_data import BenchmarkData, ITEM_KEYS, USER_KEYS, PrefixDataset, collate
from baseline_runtime import seed_all
from run_controlled_token_experiments import ControlledSuite, user_svd_factors


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact(left, right, message):
    require(torch.equal(left.detach().cpu(), right.detach().cpu()), message)


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_state(device):
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def check_rng(actual, expected, variant):
    exact(actual["torch_cpu"], expected["torch_cpu"],
          variant + ": first training forward CPU RNG is not seed=42; reset after _new_model")
    if expected["torch_cuda"] is not None:
        exact(actual["torch_cuda"], expected["torch_cuda"],
              variant + ": first training forward CUDA RNG is not seed=42")
    require(actual["python"] == expected["python"], variant + ": Python RNG differs")
    left, right = actual["numpy"], expected["numpy"]
    require(left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:],
            variant + ": NumPy RNG differs")


def synthetic_data():
    rows = []
    for user in range(12):
        for step in range(10):
            item = (user * 5 + step) % 64
            rows.append({"user_id": "u%02d" % user, "parent_asin": "i%02d" % item,
                         "rating": float(4 + step % 2),
                         "timestamp": 1600000000000 + step * 86400000,
                         "category": "c%d" % (item % 3)})
    frame = pd.DataFrame(rows).sort_values(["user_id", "timestamp", "parent_asin"])
    brands = {item: "b%d" % (index % 4)
              for index, item in enumerate(sorted(frame.parent_asin.unique()))}
    return BenchmarkData(frame.reset_index(drop=True), brands=brands, hist_len=4, seed=42)


def make_suite(directory, device, dim=256):
    suite = ControlledSuite.__new__(ControlledSuite)
    suite.args = SimpleNamespace(
        dim=dim, token_dim=dim, hist_len=4, seed=42, init_seed=424242,
        negatives=4, epochs=1, din_batch=4, workers=0, max_train_steps=2,
        microbatch=16, retain_checkpoints=True, candidates=75,
        screen_users=4, test_users=4, fusion_mode="rrf",
        fusion_weights=[2.0, 1.0, 0.7, 0.05], itemcf_half_life_days=180.,
    )
    suite.data = synthetic_data()
    suite.data_id = suite.data.manifest["data_id"]
    suite.device = device
    suite.run_dir = Path(directory)
    suite.source = {"data_id": suite.data_id, "smoke": "synthetic-only"}
    suite.manifest = {"suite": "synthetic-controlled-smoke", "data_id": suite.data_id}
    factors = np.random.default_rng(7).normal(size=(len(suite.data.items), suite.args.dim)).astype(np.float32)
    factors /= np.maximum(np.linalg.norm(factors, axis=1, keepdims=True), 1e-8)
    factors[:2] = 0.
    suite.factors = factors
    suite.user_factors = user_svd_factors(suite.data, factors)
    (suite._shared_state, suite._fusion_tail_state,
     suite._common_head_keys, suite.random_user_state) = suite._make_templates()

    # Requiring the key catches accidentally excluding the whole first layer.
    # A nonzero sentinel also proves _new_model copies it, rather than merely
    # leaving the two constructors' equal zero biases in place.
    require("0.bias" in suite._fusion_tail_state, "compatible first-layer bias is not explicitly shared")
    require("head.0.bias" in suite._common_head_keys, "first-layer bias missing from the copied-key audit")
    bias = suite._fusion_tail_state["0.bias"]
    suite._fusion_tail_state["0.bias"] = torch.linspace(0.01, 0.02, bias.numel()).reshape(bias.shape)
    return suite


def variant_parts(variant):
    if variant.startswith("semantic_concat_"):
        return "concat", variant[len("semantic_concat_"):]
    return "din", variant[len("din_"):]


def check_initial_states(suite):
    models = {variant: suite._new_model(*variant_parts(variant)) for variant in suite.VARIANTS}
    states = {variant: model.state_dict() for variant, model in models.items()}
    expected_random = suite.random_user_state.clone()
    expected_random[:2] = 0.

    # This independently derives means directly from each user's train prefix.
    expected_svd = np.zeros_like(suite.user_factors)
    for user in range(2, len(suite.data.users)):
        mask = (suite.data.uid == user) & suite.data.train_mask & (suite.data.iid >= 2)
        ids = suite.data.iid[mask]
        if len(ids):
            expected_svd[user] = suite.factors[ids].sum(axis=0, dtype=np.float32) / len(ids)
    exact(torch.as_tensor(suite.user_factors), torch.as_tensor(expected_svd), "SVD user means differ from train-prefix means")
    require(not torch.equal(expected_random, torch.as_tensor(expected_svd)), "synthetic user modes accidentally coincide")

    comparisons = 0
    for variant, state in states.items():
        kind, mode = variant_parts(variant)
        exact(state["item_embedding.weight"], torch.as_tensor(suite.factors), variant + ": item SVD differs")
        expected_user = expected_random if mode == "random" else torch.as_tensor(expected_svd)
        exact(state["user_embedding.weight"], expected_user, variant + ": wrong user initialization")
        for key, expected in suite._shared_state.items():
            exact(state[key], expected, variant + ": shared tensor differs: " + key)
            comparisons += 1
        prefix = "mlp." if kind == "din" else "head."
        for suffix, expected in suite._fusion_tail_state.items():
            exact(state[prefix + suffix], expected, variant + ": shared fusion tensor differs: " + suffix)
            comparisons += 1
    for random_variant, svd_variant in (("din_random", "din_svd"),
                                        ("semantic_concat_random", "semantic_concat_svd")):
        left, right = states[random_variant], states[svd_variant]
        require(left.keys() == right.keys(), "same-architecture state keys differ")
        for key in left:
            if key != "user_embedding.weight":
                exact(left[key], right[key], random_variant + "/" + svd_variant + ": extra change in " + key)
                comparisons += 1
    require(states["din_random"]["mlp.0.weight"].shape !=
            states["semantic_concat_random"]["head.0.weight"].shape,
            "smoke must exercise incompatible first-layer weight shapes")
    if suite.args.dim == 256:
        for variant, state in states.items():
            kind, _ = variant_parts(variant)
            prefix = "mlp." if kind == "din" else "head."
            expected_width = 1358 if kind == "din" else 1536
            require(tuple(state[prefix + "0.weight"].shape) == (256, expected_width),
                    variant + ": formal first-layer topology differs")
            require(tuple(state[prefix + "4.weight"].shape) == (128, 256) and
                    tuple(state[prefix + "8.weight"].shape) == (64, 128),
                    variant + ": formal hidden topology differs from [256,128,64]")
    del models, states
    return comparisons


def check_training(suite):
    expected = None
    active = {"variant": None}
    traces = {variant: [] for variant in suite.VARIANTS}
    first_rng_hashes = {}
    real_new_model = suite._new_model

    def observed_new_model(kind, mode):
        model = real_new_model(kind, mode)
        original_forward = model.forward_bpr

        def observed_forward(user, positive, negative):
            if model.training:
                variant = active["variant"]
                if not traces[variant]:
                    actual = rng_state(suite.device)
                    check_rng(actual, expected, variant)
                    first_rng_hashes[variant] = {
                        "cpu": digest_tensor(actual["torch_cpu"]),
                        "cuda": digest_tensor(actual["torch_cuda"]) if actual["torch_cuda"] is not None else None,
                    }
                # Identity/order and every negative are observed from the actual
                # _train_one batches, not generated again by a parallel test loop.
                traces[variant].append({
                    "users": user["user_id"].detach().cpu().tolist(),
                    "positive": positive["item_id"].detach().cpu().tolist(),
                    "negative": negative["item_id"].detach().cpu().tolist(),
                })
            return original_forward(user, positive, negative)

        model.forward_bpr = observed_forward
        return model

    # Replace expensive ranking evaluation only; the real dataset, DataLoader,
    # BPR forward/backward, optimizer, scheduler and checkpoint path all execute.
    evaluation_batch = collate([PrefixDataset(suite.data, suite.args.negatives)[i] for i in range(4)])

    def smoke_evaluate(model, records, pools, label):
        batch = {key: value.to(suite.device) for key, value in evaluation_batch.items()}
        with torch.no_grad():
            positive, negative = model.forward_bpr(
                {key: batch[key] for key in USER_KEYS},
                {key: batch["pos_" + key] for key in ITEM_KEYS},
                {key: batch["neg_" + key] for key in ITEM_KEYS})
        require(torch.isfinite(positive).all().item() and torch.isfinite(negative).all().item(),
                label + ": nonfinite evaluation scores")
        return {"din": {"hr": 0.25}, "label": label, "smoke_placeholder": True}, {}

    result = {}
    with patch.object(suite, "_new_model", side_effect=observed_new_model), \
            patch.object(suite, "evaluate", side_effect=smoke_evaluate):
        for variant in suite.VARIANTS:
            active["variant"] = variant
            seed_all(suite.args.seed)
            expected = rng_state(suite.device)
            model, history = suite._train_one(variant, records=[], pools=[])
            require(len(history) == 1 and history[0]["steps"] == 2,
                    variant + ": actual _train_one did not execute exactly two steps")
            require(np.isfinite(history[0]["loss"]), variant + ": loss is not finite")
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            require(gradients, variant + ": backward produced no gradients")
            require(all(torch.isfinite(gradient).all().item() for gradient in gradients),
                    variant + ": nonfinite gradients")
            require(any(torch.count_nonzero(gradient).item() for gradient in gradients),
                    variant + ": all gradients are zero")
            result[variant] = {"steps": history[0]["steps"], "loss": history[0]["loss"],
                               "finite_gradient_tensors": len(gradients),
                               "first_training_rng_sha256": first_rng_hashes[variant]}
            del model
    baseline = traces[suite.VARIANTS[0]]
    require(len(baseline) == 2, "expected exactly two observed training batches")
    for variant in suite.VARIANTS:
        require(traces[variant] == baseline, variant + ": training order or negative samples differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--dim", type=int, default=256,
                        help="Embedding/token width; 256 exercises the formal 1358/1536-input topology")
    args = parser.parse_args()
    if args.dim < 1:
        parser.error("dim must be positive")
    devices = ("cpu", "cuda") if args.device == "both" else (args.device,)
    if "cuda" in devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    torch.set_num_threads(2)
    report = {"status": "passed", "data": "synthetic-only", "training_seed": 42,
              "embed_dim": args.dim, "formal_topology": args.dim == 256,
              "scope": "initialization, actual training RNG, batch order, negatives, BPR backward", "devices": {}}
    for name in devices:
        device = torch.device(name)
        with tempfile.TemporaryDirectory(prefix="controlled-token-smoke-") as directory:
            suite = make_suite(directory, device, dim=args.dim)
            comparisons = check_initial_states(suite)
            report["devices"][name] = {"exact_tensor_comparisons": comparisons,
                                       "first_layer_bias_explicit_copy": True,
                                       "variants": check_training(suite)}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
