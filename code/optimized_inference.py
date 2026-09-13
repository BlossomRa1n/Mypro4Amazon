"""Run the exported, versioned recommendation pipeline on observed histories."""
import argparse
import csv
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch

from baseline_data import BenchmarkData
from optimization_models import score_din_cached
from run_baseline import make_model, model_config, recall, score_din
from run_fusion_experiments import Channels
from run_optimization import file_hash
from structural_models import StructuralDIN, variant_for_experiment


class LatestHistoryView:
    """Expose an end-exclusive cutoff without mutating the benchmark's timestamps."""
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)

    def history_end(self, uid, position):
        if position == self.data.ends[uid]:
            return int(position)
        return self.data.history_end(uid, position)

    def user_features(self, uid, position):
        return BenchmarkData.user_features(self, uid, position)


def load_model(path, data, args, kind, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["manifest"]["data_id"] != data.manifest["data_id"]:
        raise ValueError("Checkpoint and feature data IDs differ")
    if kind == "v2":
        model = make_model(data, args, "v2", torch.device("cpu"))
        model.load_state_dict(saved["model"])
    else:
        variant = saved["manifest"].get("candidate_variant") or variant_for_experiment(saved["manifest"].get("experiment", ""))
        model = StructuralDIN(variant=variant or "control", **model_config(data, args, "din"))
        if variant is None:
            model.load_baseline(saved["model"])
        else:
            model.load_state_dict(saved["model"])
    return model.to(device).eval()


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported bundle version")
    for relative, digest in manifest["sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError("Bundle verification failed: " + relative)
    return manifest


def predict(root, split="latest", max_users=0, device=None, use_cache=True):
    root = Path(root).resolve()
    manifest = verify_bundle(root)
    with (root / "data.pkl").open("rb") as stream:
        data = pickle.load(stream)["data"]
    if data.manifest["data_id"] != manifest["data_id"]:
        raise ValueError("Bundle feature identity differs")
    with (root / "itemcf.pkl").open("rb") as stream:
        cf = pickle.load(stream)
    args = SimpleNamespace(**manifest["baseline_args"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if split == "latest":
        data = LatestHistoryView(data)
        records = [(uid, int(data.ends[uid])) for uid in range(2, len(data.users))
                   if data.ends[uid] > data.starts[uid]]
        if max_users:
            records = records[:max_users]
    elif split in ("val", "test"):
        records = data.evaluation(split, max_users)
    else:
        raise ValueError(split)
    v2 = load_model(root / "v2.pth", data, args, "v2", device)
    _, scores = recall(v2, data, records, device, manifest["candidate_budget"])
    del v2
    torch.cuda.empty_cache()
    channels = Channels(data, cf)
    pools = []
    for record, values in zip(records, scores):
        parts, seen = channels.build(record, values, manifest["candidate_budget"], manifest["recipe"])
        pools.append(channels.merge(parts, seen, manifest["candidate_budget"], manifest["recipe"]))
    del scores
    din = load_model(root / "din.pth", data, args, "din", device)
    scorer = score_din_cached if use_cache and manifest["cached_inference_verified"] else score_din
    rankings = scorer(din, data, records, pools, device)
    return [{"user_id": data.users[uid], **{"article_" + str(i + 1): data.items[ranked[i]] if i < len(ranked) else ""
             for i in range(5)}} for (uid, _), ranked in zip(records, rankings)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("latest", "val", "test"), default="latest")
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--reference-scoring", action="store_true")
    args = parser.parse_args()
    if args.max_users < 0:
        parser.error("max-users must be nonnegative")
    torch.set_num_threads(4)
    rows = predict(args.bundle, args.split, args.max_users, args.device, not args.reference_scoring)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["user_id"] + ["article_" + str(i) for i in range(1, 6)])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output.resolve()), "users": len(rows), "split": args.split,
                      "sha256": file_hash(output)}))


if __name__ == "__main__":
    main()
