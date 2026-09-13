"""Bounded, resumable validation-only experiments against a frozen baseline."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import datetime
import gc
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_data import PrefixDataset, USER_KEYS, ITEM_KEYS, collate, fingerprint
from baseline_runtime import atomic_save, seed_all, restore_checkpoint, save_checkpoint
from run_baseline import candidate_pools, infonce, make_model, recall, score_din, to_device, write_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_comparison(candidate, reference, seed=42):
    if not np.array_equal(candidate["uid"], reference["uid"]):
        raise ValueError("Paired evaluation users/order differ")
    if not np.array_equal(candidate["target"], reference["target"]):
        raise ValueError("Paired evaluation targets differ")
    delta = candidate["hit5"].astype(np.int8) - reference["hit5"].astype(np.int8)
    counts = np.array([(delta == v).sum() for v in (-1, 0, 1)])
    boot = np.random.default_rng(seed).multinomial(len(delta), counts / len(delta), size=10000)
    interval = np.quantile((boot[:, 2] - boot[:, 0]) / len(delta), [.025, .975])
    ndcg_delta = candidate["ndcg5"] - reference["ndcg5"]
    return {"hr5_delta": float(delta.mean()), "hr5_delta_ci95": interval.tolist(),
            "ndcg5_delta": float(ndcg_delta.mean()), "users": len(delta),
            "improved_users": int(counts[2]), "regressed_users": int(counts[0]),
            "confirmed_gain": bool(interval[0] > 0 and ndcg_delta.mean() >= 0)}


class MixedNegativeDataset(PrefixDataset):
    def __init__(self, data, cf, seed=42, hard_count=2):
        super().__init__(data, 4)
        self.cf, self.seed, self.hard_count = cf, seed, hard_count

    def __getitem__(self, idx):
        data = self.data
        position = data.train_positions[idx]
        uid, target = int(data.uid[position]), int(data.iid[position])
        batch = data.user_features(uid, position)
        batch.update({"pos_" + key: value for key, value in data.item_features(target).items()})
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(idx)]))
        excluded = data.train_sets[uid] | {target}
        hard = []
        if self.hard_count:
            # Only the observed prefix supplies anchors, never the training target.
            history = batch["hist_items"][batch["hist_items"] >= 2][-10:]
            if len(history):
                neighbors, scores = self.cf
                candidates = neighbors[history, :30].ravel()
                valid = scores[history, :30].ravel() > 0
                eligible = sorted({int(i) for i in candidates[valid]
                                   if i >= 2 and i not in excluded and i in data.active_set})
                if eligible:
                    hard = rng.choice(eligible, min(self.hard_count, len(eligible)), replace=False).tolist()
        random = data.negatives(uid, target, 4 - len(hard), rng, excluded=excluded | set(hard))
        negatives = np.asarray(hard + random.tolist(), dtype=np.int64)
        batch.update({"neg_" + key: value for key, value in data.item_features(negatives).items()})
        batch["hard_count"] = np.int64(len(hard))
        return batch


class Optimization:
    def __init__(self, args):
        self.options = args
        self.base, self.run = Path(args.baseline).resolve(), Path(args.run_dir).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        for subdir in ("cache", "evaluations", "experiments"):
            (self.run / subdir).mkdir(exist_ok=True)
        if not (self.base / "COMPLETED.json").exists():
            raise ValueError("Baseline has not completed")
        baseline_manifest = json.loads((self.base / "run_manifest.json").read_text())
        self.args = SimpleNamespace(**baseline_manifest["args"])
        with (self.base / "data.pkl").open("rb") as stream:
            self.data = pickle.load(stream)["data"]
        with (self.base / "itemcf.pkl").open("rb") as stream:
            self.cf = pickle.load(stream)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(4)
        self.hashes = {}
        source = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
        self.manifest = {"baseline": baseline_manifest, "data_id": self.data.manifest["data_id"],
                         "source": source, "screen_users": args.screen_users,
                         "confirm_users": args.confirm_users, "max_train_steps": args.max_train_steps,
                         "policy": "validation-only selection; baseline test metrics not used",
                         "seeds": [42, 43], "din_learning_rate": .0001, "v2_learning_rate": .00005}
        previous = self.run / "manifest.json"
        if previous.exists() and json.loads(previous.read_text()) != self.manifest:
            raise ValueError("Experiment manifest changed; use a new output directory")
        write_json(previous, self.manifest)
        records = self.data.evaluation("val", args.screen_users + args.confirm_users)
        self.screen = records[:args.screen_users]
        self.confirm = records[args.screen_users:]
        if not self.screen or not self.confirm:
            raise ValueError("Need disjoint screening and confirmation validation users")
        self.deadline = time.monotonic() + args.hours * 3600
        self.base_v2, self.base_din = self.base / "v2_best.pth", self.base / "din_best.pth"

    def event(self, action, **details):
        event = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "action": action, **details}
        with (self.run / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
        write_json(self.run / "status.json", event)

    def available(self):
        if time.monotonic() >= self.deadline:
            self.event("paused", reason="stage_time_budget_reached")
            return False
        free = shutil.disk_usage(self.run).free / 1024 ** 3
        if free < 5:
            self.event("paused", reason="less_than_5_GiB_free", free_gib=free)
            return False
        return True

    def identity(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.hashes:
            self.hashes[key] = file_hash(path)
        return self.hashes[key]

    def model(self, kind, path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["manifest"]["data_id"] != self.data.manifest["data_id"]:
            raise ValueError("Checkpoint data ID differs")
        model = make_model(self.data, self.args, kind, self.device)
        model.load_state_dict(saved["model"])
        return model.eval()

    def pools(self, records, budget, v2_path=None):
        v2_path = v2_path or self.base_v2
        identity = fingerprint([self.data.manifest["data_id"], records, self.identity(v2_path), budget])
        path = self.run / "cache" / (identity + ".pkl")
        if path.exists():
            with path.open("rb") as stream:
                saved = pickle.load(stream)
            if saved["identity"] != identity:
                raise ValueError("Pool cache identity differs")
            return saved["pools"]
        self.event("build_pools", users=len(records), budget=budget, v2=str(v2_path))
        model = self.model("v2", v2_path)
        _, scores = recall(model, self.data, records, self.device, budget)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        pools, _ = candidate_pools(self.data, records, scores, self.cf, budget)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump({"identity": identity, "pools": pools}, stream, protocol=5)
        os.replace(temporary, path)
        return pools

    def evaluate(self, name, records, budget=100, din_path=None, v2_path=None):
        din_path, v2_path = din_path or self.base_din, v2_path or self.base_v2
        directory = self.run / "evaluations" / name
        directory.mkdir(exist_ok=True)
        spec = {"records_hash": fingerprint(records), "data_id": self.data.manifest["data_id"],
                "budget": budget, "din": self.identity(din_path), "v2": self.identity(v2_path)}
        if (directory / "metrics.json").exists():
            metrics = json.loads((directory / "metrics.json").read_text())
            if metrics["spec"] != spec:
                raise ValueError("Evaluation cache identity differs")
            with np.load(directory / "users.npz") as arrays:
                return metrics, {k: arrays[k] for k in arrays.files}
        self.event("evaluate", name=name, users=len(records), budget=budget)
        pools = self.pools(records, budget, v2_path)
        model = self.model("din", din_path)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        rankings = score_din(model, self.data, records, pools, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.monotonic() - started
        targets = np.array([self.data.iid[p] for _, p in records], dtype=np.int32)
        positions = np.array([ranked.index(int(target)) + 1 if target in ranked else 0
                              for ranked, target in zip(rankings, targets)], dtype=np.int32)
        hit = (positions > 0) & (positions <= 5)
        arrays = {"uid": np.array([u for u, _ in records], dtype=np.int32), "target": targets,
                  "rank": positions, "pool_hit": np.array([t in p for t, p in zip(targets, pools)]),
                  "hit5": hit, "ndcg5": np.where(hit, 1 / np.log2(np.maximum(positions, 1) + 1), 0),
                  "category": self.data.item_category[targets], "train_popularity": self.data.train_counts[targets],
                  "history_length": np.array([self.data.history_end(u, p) - self.data.starts[u] for u, p in records]),
                  "top5": np.array([r[:5] + [0] * (5 - len(r[:5])) for r in rankings], dtype=np.int32)}
        coverage = float(arrays["pool_hit"].mean())
        metrics = {"spec": spec, "users": len(records), "hr5": float(hit.mean()),
                   "ndcg5": float(arrays["ndcg5"].mean()), "coverage": coverage,
                   "conditional_hr5": float(hit.mean()) / coverage if coverage else 0.,
                   "rerank_seconds": seconds, "rerank_ms_per_user": seconds / len(records) * 1000,
                   "pool_size_mean": float(np.mean([len(p) for p in pools]))}
        groups = {}
        for column in ("category", "history_length", "train_popularity"):
            values = arrays[column]
            if column == "history_length":
                labels = np.select([values <= 5, values <= 20], ["1-5", "6-20"], default="21+")
            elif column == "train_popularity":
                labels = np.select([values == 0, values <= 5, values <= 50], ["unseen", "1-5", "6-50"], default="51+")
            else:
                labels = np.array([self.data.categories[v] for v in values])
            groups[column] = {str(label): {"users": int((labels == label).sum()),
                              "hr5": float(hit[labels == label].mean()),
                              "coverage": float(arrays["pool_hit"][labels == label].mean())}
                              for label in np.unique(labels)}
        metrics["groups"] = groups
        temporary = directory / "users.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, directory / "users.npz")
        write_json(directory / "metrics.json", metrics)
        del model, rankings, pools
        gc.collect()
        torch.cuda.empty_cache()
        self.event("evaluation_complete", name=name, **{k: metrics[k] for k in ("hr5", "ndcg5", "coverage", "rerank_seconds")})
        return metrics, arrays

    def train(self, name, kind="din", hard_count=0, epochs=1, seed=42, objective="bpr"):
        directory = self.run / "experiments" / name
        directory.mkdir(exist_ok=True)
        initial = self.base_din if kind == "din" else self.base_v2
        manifest = {"data_id": self.data.manifest["data_id"], "experiment": name,
                    "initial_sha256": self.identity(initial), "kind": kind, "epochs": epochs,
                    "seed": seed, "hard_count": hard_count, "objective": objective,
                    "learning_rate": .0001 if kind == "din" else .00005,
                    "optimizer_policy": "fresh AdamW; identical control/treatment initialization",
                    "source": self.manifest["source"], "max_train_steps": self.options.max_train_steps}
        completed = directory / "COMPLETED.json"
        if completed.exists():
            if json.loads(completed.read_text())["manifest"] != manifest:
                raise ValueError("Training experiment identity changed")
            return directory / "best.pth"
        if not self.available():
            return None
        self.event("training_start", name=name, manifest=manifest)
        pools = self.pools(self.screen, 100) if kind == "din" else None
        seed_all(seed)
        model = self.model(kind, initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=manifest["learning_rate"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        start, best, history = 0, -1., []
        latest = directory / "latest.pth"
        if latest.exists():
            start, best, history = restore_checkpoint(latest, model, optimizer, scheduler, manifest)
        dataset = MixedNegativeDataset(self.data, self.cf, seed=seed, hard_count=hard_count)
        targets = [int(self.data.iid[p]) for _, p in self.screen]
        for epoch in range(start, epochs):
            dataset.epoch = self.args.epochs + epoch
            generator = torch.Generator().manual_seed(seed + dataset.epoch)
            loader = DataLoader(dataset, batch_size=self.args.din_batch if kind == "din" else self.args.v2_batch,
                                shuffle=True, generator=generator, num_workers=self.args.workers,
                                collate_fn=collate, pin_memory=self.device.type == "cuda", drop_last=True)
            model.train()
            total, steps, actual_hard, count = 0., 0, 0, 0
            started = time.monotonic()
            for batch in tqdm(loader, desc=f"{name} {epoch + 1}/{epochs}", mininterval=20):
                actual_hard += int(batch.pop("hard_count").sum())
                count += len(batch["user_id"])
                batch = to_device(batch, self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    if kind == "v2":
                        loss = infonce(model, batch)
                    else:
                        pos, neg = model.forward_bpr({k: batch[k] for k in USER_KEYS},
                                   {k: batch["pos_" + k] for k in ITEM_KEYS},
                                   {k: batch["neg_" + k] for k in ITEM_KEYS})
                        if objective == "bpr":
                            loss = F.softplus(neg.float() - pos.float()[:, None]).mean()
                        elif objective == "softmax":
                            scores = torch.cat([pos.float()[:, None], neg.float()], dim=1)
                            loss = F.cross_entropy(scores, torch.zeros(len(pos), device=self.device, dtype=torch.long))
                        else:
                            raise ValueError(objective)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())
                steps += 1
                if self.options.max_train_steps and steps >= self.options.max_train_steps:
                    break
            if not steps:
                raise ValueError("No full training batches")
            scheduler.step()
            model.eval()
            if kind == "din":
                rankings = score_din(model, self.data, self.screen, pools, self.device)
                metric = float(np.mean([t in r[:5] for t, r in zip(targets, rankings)]))
            else:
                rankings, _ = recall(model, self.data, self.screen, self.device, 100)
                metric = float(np.mean([t in r for t, r in zip(targets, rankings)]))
            entry = {"epoch": epoch + 1, "loss": total / steps, "steps": steps,
                     "screen_metric": metric, "actual_hard_per_sample": actual_hard / count,
                     "seconds": time.monotonic() - started}
            history.append(entry)
            if metric > best:
                best = metric
                atomic_save({"model": model.state_dict(), "manifest": manifest, "epoch": epoch + 1}, directory / "best.pth")
            save_checkpoint(latest, model, optimizer, scheduler, epoch + 1, best, manifest, history)
            write_json(directory / "history.json", history)
            self.event("training_epoch_complete", name=name, **entry)
            if epoch + 1 < epochs and not self.available():
                return None
        # Completed experiments keep every distinct best/final weight; only disposable optimizer state is compacted.
        best_epoch = max(range(len(history)), key=lambda i: history[i]["screen_metric"]) + 1
        if best_epoch == epochs:
            temporary = directory / "final.link"
            if temporary.exists():
                temporary.unlink()
            os.link(directory / "best.pth", temporary)
            os.replace(temporary, directory / "final.pth")
        else:
            atomic_save({"model": model.state_dict(), "manifest": manifest, "epoch": epochs,
                         "resume_policy": "completed experiment: weights-only continuation"}, directory / "final.pth")
        write_json(completed, {"manifest": manifest, "history": history, "best": best})
        latest.unlink()
        self.event("training_complete", name=name, compacted_optimizer_checkpoint=str(latest))
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        return directory / "best.pth"

    def core(self):
        path = self.run / "pool_decision.json"
        if not path.exists():
            screens = {}
            for budget in (100, 200, 300):
                if not self.available():
                    return
                screens[budget], _ = self.evaluate(f"pool_{budget}_screen", self.screen, budget)
            chosen = max(screens, key=lambda b: (screens[b]["hr5"], screens[b]["ndcg5"], -b))
            if not self.available():
                return
            baseline, reference = self.evaluate("pool_100_confirm", self.confirm, 100)
            preferred, candidate = self.evaluate(f"pool_{chosen}_confirm", self.confirm, chosen)
            comparison = paired_comparison(candidate, reference)
            budget = chosen if comparison["confirmed_gain"] else 100
            write_json(path, {"screen": screens, "screen_winner": chosen, "comparison": comparison,
                              "confirmed_budget": budget, "baseline_confirm": baseline,
                              "preferred_confirm": preferred})
            self.event("pool_decision", screen_winner=chosen, confirmed_budget=budget, comparison=comparison)
        for name, hard in (("din_random_seed42", 0), ("din_mixed_seed42", 2)):
            if self.train(name, hard_count=hard) is None:
                return
        path = self.run / "negative_decision.json"
        if not path.exists():
            screens, checkpoints = {}, {"baseline": self.base_din,
                "random": self.run / "experiments/din_random_seed42/best.pth",
                "mixed": self.run / "experiments/din_mixed_seed42/best.pth"}
            for name, checkpoint in checkpoints.items():
                screens[name], _ = self.evaluate(f"negative_{name}_screen", self.screen, din_path=checkpoint)
            chosen = max(screens, key=lambda n: (screens[n]["hr5"], screens[n]["ndcg5"], n == "baseline"))
            confirmed = {}
            for name, checkpoint in checkpoints.items():
                if not self.available():
                    return
                metrics, values = self.evaluate(f"negative_{name}_confirm", self.confirm, din_path=checkpoint)
                confirmed[name] = (metrics, values)
            improvement = paired_comparison(confirmed[chosen][1], confirmed["baseline"][1])
            effect = paired_comparison(confirmed["mixed"][1], confirmed["random"][1])
            winner = chosen if improvement["confirmed_gain"] else "baseline"
            write_json(path, {"screen": screens, "screen_winner": chosen, "confirmed_winner": winner,
                              "winner_checkpoint": str(checkpoints[winner]), "against_baseline": improvement,
                              "mixed_against_random": effect,
                              "confirmation": {k: v[0] for k, v in confirmed.items()}})
            self.event("negative_decision", winner=winner, improvement=improvement, treatment_effect=effect)
        if self.train("v2_extra2_seed42", kind="v2", epochs=2) is None:
            return
        path = self.run / "v2_decision.json"
        if not path.exists():
            checkpoint = self.run / "experiments/v2_extra2_seed42/best.pth"
            for label, records in (("screen", self.screen), ("confirm", self.confirm)):
                if not self.available():
                    return
                self.evaluate("v2_extra2_" + label, records, v2_path=checkpoint)
            reference_metrics, reference = self.evaluate("pool_100_confirm", self.confirm)
            metrics, candidate = self.evaluate("v2_extra2_confirm", self.confirm, v2_path=checkpoint)
            comparison = paired_comparison(candidate, reference)
            write_json(path, {"fixed_din_comparison": comparison, "baseline": reference_metrics,
                              "continued_v2": metrics, "checkpoint": str(checkpoint)})
        write_json(self.run / "CORE_COMPLETED.json", {"status": "core_complete", "followup_stages_pending": True})
        self.event("core_complete", next="replication_and_combination; do not shut down yet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=10000)
    parser.add_argument("--confirm-users", type=int, default=90000)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--max-train-steps", type=int, default=0)
    args = parser.parse_args()
    if min(args.screen_users, args.confirm_users, args.hours) <= 0:
        parser.error("user counts and time budget must be positive")
    runner = Optimization(args)
    try:
        runner.core()
    except Exception as error:
        runner.event("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
