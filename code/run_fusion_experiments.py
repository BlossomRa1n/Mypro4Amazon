"""Validation-only fixed-budget fusion and inexpensive channel experiments."""
import argparse
import gc
import json
from itertools import islice
from pathlib import Path
import time

import numpy as np
import torch

from baseline_data import DAY_MS, fingerprint
from baseline_runtime import quota_merge
from run_baseline import recall, score_din, write_json
from run_optimization import Optimization, file_hash, paired_comparison


RECIPES = {
    "quota": {"fusion": "quota", "half_life": 0, "brand": False},
    "rrf": {"fusion": "rrf", "half_life": 0, "brand": False},
    "cf30": {"fusion": "quota", "half_life": 30, "brand": False},
    "cf90": {"fusion": "quota", "half_life": 90, "brand": False},
    "brand": {"fusion": "quota", "half_life": 0, "brand": True},
}


class Channels:
    def __init__(self, data, cf):
        self.data, self.cf = data, cf
        self.hot = [int(i) for i in np.argsort(-data.train_counts, kind="stable")
                    if i >= 2 and data.train_counts[i] > 0]
        self.category, self.brand = {}, {}
        for item in self.hot:
            self.category.setdefault(int(data.item_category[item]), []).append(item)
            if data.item_brand[item] >= 2:
                self.brand.setdefault(int(data.item_brand[item]), []).append(item)

    def build(self, record, v2, budget, recipe):
        data, (neighbors, similarity) = self.data, self.cf
        uid, pos = record
        end = data.history_end(uid, pos)
        seen = set(data.iid[data.starts[uid]:end].tolist())
        start = max(data.starts[uid], end - data.hist_len)
        hist = data.iid[start:end]
        cf_score = {}
        ages = (data.ts[end - 1] - data.ts[start:end]) / DAY_MS if len(hist) else []
        for iid, age in zip(hist, ages):
            decay = 2. ** (-age / recipe["half_life"]) if recipe["half_life"] else 1.
            for candidate, score in zip(neighbors[iid], similarity[iid]):
                if candidate >= 2 and candidate not in seen and score > 0:
                    cf_score[int(candidate)] = cf_score.get(int(candidate), 0.) + float(score) * decay
        cf_score = dict(sorted(cf_score.items(), key=lambda x: (-x[1], x[0]))[:budget])
        for item in self.hot:
            if len(cf_score) >= budget:
                break
            if item not in seen and item not in cf_score:
                cf_score[item] = -1.
        cats, counts = np.unique(data.item_category[hist], return_counts=True)
        cat_score = {}
        for cat, count in sorted(zip(cats, counts), key=lambda x: -x[1])[:3]:
            eligible = list(islice((i for i in self.category.get(cat, ()) if i not in seen), 20))
            cat_score.update({i: float(count / max(len(hist), 1)) for i in eligible})
        hot_score = {i: 1 / (rank + 1) for rank, i in
                     enumerate(islice((i for i in self.hot if i not in seen), 5))}
        channels = [cf_score, v2, cat_score, hot_score]
        if recipe["brand"]:
            brands, counts = np.unique(data.item_brand[hist], return_counts=True)
            brand_score = {}
            known = [(brand, count) for brand, count in zip(brands, counts) if brand >= 2]
            for brand, count in sorted(known, key=lambda x: (-x[1], x[0]))[:3]:
                for rank, item in enumerate(islice((i for i in self.brand.get(brand, ()) if i not in seen), budget)):
                    brand_score[item] = float(count / max(len(hist), 1) / (rank + 1))
            channels.append(dict(sorted(brand_score.items(), key=lambda x: (-x[1], x[0]))[:budget]))
        return channels, seen

    def merge(self, channels, seen, budget, recipe):
        weights = [1.5, 1., .7, .05] + ([.2] if recipe["brand"] else [])
        if recipe["fusion"] == "quota":
            result = quota_merge(channels, weights, budget, seen)
        else:
            scores = {}
            for channel, weight in zip(channels, weights):
                for rank, (item, _) in enumerate(sorted(channel.items(), key=lambda x: (-x[1], x[0])), 1):
                    if item >= 2 and item not in seen:
                        scores[item] = scores.get(item, 0.) + weight / (60 + rank)
            result = dict(sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:budget])
        for item in self.hot:
            if len(result) >= budget:
                break
            if item not in seen and item not in result:
                result[item] = -1.
        return result


class FusionExperiments(Optimization):
    def __init__(self, args):
        followup = Path(args.followup_run).resolve()
        if not (followup / "FOLLOWUP_COMPLETED.json").exists():
            raise ValueError("Follow-up queue must finish first")
        previous = json.loads((followup / "manifest.json").read_text())
        for name, digest in previous["source"].items():
            if file_hash(Path(__file__).parent / name) != digest:
                raise ValueError(f"Follow-up dependency changed: {name}")
        for key in ("screen_users", "confirm_users", "max_train_steps"):
            if getattr(args, key) != previous[key]:
                raise ValueError(f"Follow-up sampling differs: {key}")
        super().__init__(args)
        if previous["baseline"] != self.manifest["baseline"]:
            raise ValueError("Baseline identity changed")
        selected = json.loads((followup / "combination_decision.json").read_text())["selected"]
        self.din, self.v2, self.budget = Path(selected["din"]), Path(selected["v2"]), selected["budget"]
        inputs = {"selected": selected, "din_sha256": self.identity(self.din), "v2_sha256": self.identity(self.v2),
                  "recipes": RECIPES, "rrf_offset": 60, "brand_weight": .2,
                  "half_life_anchor": "last observed event; no target time used",
                  "policy": "fixed models/budget; screen-only recipe choice, then development confirmation"}
        path = self.run / "fusion_inputs.json"
        if path.exists() and json.loads(path.read_text()) != inputs:
            raise ValueError("Fusion inputs changed")
        write_json(path, inputs)
        self.channels = Channels(self.data, self.cf)

    def v2_scores(self, records):
        model = self.model("v2", self.v2)
        _, scores = recall(model, self.data, records, self.device, self.budget)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return scores

    def evaluate_recipe(self, name, records, recipe_name, v2_scores):
        directory = self.run / "evaluations" / name
        directory.mkdir(exist_ok=True)
        spec = {"records": fingerprint(records), "recipe": RECIPES[recipe_name], "budget": self.budget,
                "din": self.identity(self.din), "v2": self.identity(self.v2)}
        path = directory / "metrics.json"
        if path.exists():
            metrics = json.loads(path.read_text())
            if metrics["spec"] != spec:
                raise ValueError("Fusion evaluation cache differs")
            with np.load(directory / "users.npz") as saved:
                return metrics, {k: saved[k] for k in saved.files}
        self.event("fusion_evaluate", name=name, users=len(records))
        recipe = RECIPES[recipe_name]
        started = time.monotonic()
        pools, channel_hits, union_hits, brand_unique = [], [], [], []
        for record, v2 in zip(records, v2_scores):
            channels, seen = self.channels.build(record, v2, self.budget, recipe)
            target = int(self.data.iid[record[1]])
            hits = [target in channel for channel in channels]
            channel_hits.append(hits)
            union_hits.append(any(hits))
            brand_unique.append(bool(recipe["brand"] and hits[-1] and not any(hits[:-1])))
            pools.append(self.channels.merge(channels, seen, self.budget, recipe))
        build_seconds = time.monotonic() - started
        model = self.model("din", self.din)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        rankings = score_din(model, self.data, records, pools, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.monotonic() - started
        targets = np.asarray([self.data.iid[p] for _, p in records], dtype=np.int32)
        rank = np.asarray([r.index(int(t)) + 1 if t in r else 0 for r, t in zip(rankings, targets)])
        hit = (rank > 0) & (rank <= 5)
        arrays = {"uid": np.asarray([u for u, _ in records], dtype=np.int32), "target": targets,
                  "hit5": hit, "ndcg5": np.where(hit, 1 / np.log2(np.maximum(rank, 1) + 1), 0),
                  "pool_hit": np.asarray([t in p for t, p in zip(targets, pools)]),
                  "channel_hits": np.asarray(channel_hits), "union_hit": np.asarray(union_hits),
                  "brand_unique_hit": np.asarray(brand_unique), "rank": rank,
                  "top5": np.asarray([r[:5] + [0] * (5 - len(r[:5])) for r in rankings], dtype=np.int32)}
        metrics = {"spec": spec, "users": len(records), "hr5": float(hit.mean()),
                   "ndcg5": float(arrays["ndcg5"].mean()), "coverage": float(arrays["pool_hit"].mean()),
                   "raw_union_coverage": float(np.mean(union_hits)),
                   "channel_coverage": np.mean(channel_hits, axis=0).tolist(),
                   "brand_unique_coverage": float(np.mean(brand_unique)),
                   "pool_mean": float(np.mean([len(p) for p in pools])),
                   "fusion_seconds": build_seconds, "rerank_seconds": seconds}
        temp = directory / "users.tmp"
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(directory / "users.npz")
        write_json(path, metrics)
        del model, pools, rankings
        gc.collect()
        torch.cuda.empty_cache()
        self.event("fusion_evaluated", name=name, hr5=metrics["hr5"], coverage=metrics["coverage"])
        return metrics, arrays

    def execute(self):
        if (self.run / "FUSION_COMPLETED.json").exists():
            return
        screens = {}
        scores = self.v2_scores(self.screen)
        for name in RECIPES:
            if not self.available():
                return
            screens[name], _ = self.evaluate_recipe(name + "_screen", self.screen, name, scores)
        del scores
        chosen = max(screens, key=lambda k: (screens[k]["hr5"], screens[k]["ndcg5"], k == "quota"))
        write_json(self.run / "screen_decision.json", {"screen": screens, "chosen": chosen})
        if not self.available():
            return
        scores = self.v2_scores(self.confirm)
        baseline, reference = self.evaluate_recipe("quota_confirm", self.confirm, "quota", scores)
        metrics, candidate = self.evaluate_recipe(chosen + "_confirm", self.confirm, chosen, scores)
        paired = paired_comparison(candidate, reference)
        selected = chosen if paired["confirmed_gain"] else "quota"
        write_json(self.run / "FUSION_COMPLETED.json", {"screen": screens, "screen_winner": chosen,
                   "selected_recipe": selected, "comparison": paired, "baseline": baseline, "candidate": metrics,
                   "overall_complete": False,
                   "still_pending": ["structure_ablations", "trained_cuda_equivalence", "candidate_training",
                                     "fresh_validation_acceptance", "final_test", "sync", "shutdown"]})
        self.event("fusion_complete", selected=selected, paired=paired, overall_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--followup-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=10000)
    parser.add_argument("--confirm-users", type=int, default=90000)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--max-train-steps", type=int, default=0)
    args = parser.parse_args()
    if min(args.screen_users, args.confirm_users, args.hours) <= 0:
        parser.error("counts and hours must be positive")
    runner = FusionExperiments(args)
    try:
        runner.execute()
    except Exception as error:
        runner.event("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
