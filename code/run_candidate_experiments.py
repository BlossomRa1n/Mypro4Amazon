"""Matched fine-tuning on internal heldout targets with recall-mined negatives."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from baseline_data import USER_KEYS, ITEM_KEYS, collate
from baseline_runtime import seed_all, atomic_save, restore_checkpoint, save_checkpoint
from run_baseline import model_config, score_din, to_device, write_json
from run_optimization import file_hash, paired_comparison
from run_structural_experiments import StructuralExperiments
from structural_models import StructuralDIN, variant_for_experiment


class CandidateDataset(Dataset):
    def __init__(self, data, records, candidates, seed=42, recall_count=1):
        self.data, self.records, self.candidates = data, records, candidates
        self.seed, self.recall_count, self.epoch = seed, recall_count, 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        uid, pos = map(int, self.records[index])
        target = int(self.data.iid[pos])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        eligible = self.candidates[index]
        eligible = sorted({int(i) for i in eligible if i >= 2 and i in self.data.active_set
                           and i not in self.data.train_sets[uid] and i != target})
        hard = rng.choice(eligible, min(self.recall_count, len(eligible)), replace=False).tolist() if eligible else []
        random = self.data.negatives(uid, target, 4 - len(hard), rng,
                                     excluded=self.data.train_sets[uid] | set(hard))
        negatives = np.asarray(hard + random.tolist(), dtype=np.int64)
        row = self.data.user_features(uid, pos)
        row.update({"pos_" + k: v for k, v in self.data.item_features(target).items()})
        row.update({"neg_" + k: v for k, v in self.data.item_features(negatives).items()})
        row["hard_count"] = np.int64(len(hard))
        return row


class CandidateExperiments(StructuralExperiments):
    def __init__(self, args):
        structural = Path(args.structural_run).resolve()
        result = json.loads((structural / "STRUCTURAL_COMPLETED.json").read_text())
        super().__init__(args)
        structural_manifest = json.loads((structural / "manifest.json").read_text())
        if structural_manifest["baseline"] != self.manifest["baseline"]:
            raise ValueError("Structural data differs")
        self.base_din = Path(result["din_checkpoint"])
        candidate_dir = Path(args.candidate_dir).resolve()
        prepared = json.loads((candidate_dir / "PREPARED.json").read_text())
        if prepared["baseline_data_id"] != self.data.manifest["data_id"]:
            raise ValueError("Candidate data ID differs")
        for name, digest in prepared["sha256"].items():
            if file_hash(candidate_dir / name) != digest:
                raise ValueError("Candidate artifact corrupted: " + name)
        self.training_records = np.load(candidate_dir / "records.npy", mmap_mode="r")
        self.candidates = np.load(candidate_dir / "negatives.npy", mmap_mode="r")
        if len(self.training_records) != len(self.candidates) or len(self.training_records) != prepared["users"]:
            raise ValueError("Candidate record alignment differs")
        if not np.all(self.data.train_mask[self.training_records[:, 1]]):
            raise ValueError("Candidate labels include original heldout data")
        if not np.array_equal(self.data.uid[self.training_records[:, 1]], self.training_records[:, 0]):
            raise ValueError("Candidate user/position alignment differs")
        saved = torch.load(self.base_din, map_location="cpu", weights_only=False)
        self.candidate_variant = variant_for_experiment(saved["manifest"].get("experiment", "")) or "control"
        del saved
        inputs = {"structural_result_sha256": file_hash(structural / "STRUCTURAL_COMPLETED.json"),
                  "initial_din_sha256": self.identity(self.base_din), "prepared": prepared,
                  "candidate_variant": self.candidate_variant, "epochs": 1, "learning_rate": .0001,
                  "negatives": 4, "treatment_recall_count": 1,
                  "policy": "matched one-pass continuation on identical internal targets; BPR; train-only labels"}
        path = self.run / "candidate_inputs.json"
        if path.exists() and json.loads(path.read_text()) != inputs:
            raise ValueError("Candidate experiment inputs changed")
        write_json(path, inputs)
        self.prepared = prepared

    def model(self, kind, path):
        if kind == "v2":
            return super().model(kind, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["manifest"]["data_id"] != self.data.manifest["data_id"]:
            raise ValueError("Checkpoint data ID differs")
        variant = saved["manifest"].get("candidate_variant") or variant_for_experiment(saved["manifest"].get("experiment", ""))
        model = StructuralDIN(variant=variant or "control", **model_config(self.data, self.args, "din"))
        if variant is None:
            model.load_baseline(saved["model"])
        else:
            model.load_state_dict(saved["model"])
        return model.to(self.device).eval()

    def train_candidate(self, recall_count, seed=42):
        name = f"candidate_{'recall' if recall_count else 'random'}_seed{seed}"
        directory = self.run / "experiments" / name
        directory.mkdir(exist_ok=True)
        manifest = {"data_id": self.data.manifest["data_id"], "experiment": name,
                    "candidate_variant": self.candidate_variant, "initial_sha256": self.identity(self.base_din),
                    "candidate_sha256": self.prepared["sha256"], "recall_count": recall_count,
                    "seed": seed, "epochs": 1, "learning_rate": .0001, "source": self.manifest["source"],
                    "max_train_steps": self.options.max_train_steps}
        completed = directory / "COMPLETED.json"
        if completed.exists():
            if json.loads(completed.read_text())["manifest"] != manifest:
                raise ValueError("Candidate checkpoint identity differs")
            return directory / "best.pth"
        if not self.available():
            return None
        seed_all(seed)
        model = self.model("din", self.base_din)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        latest = directory / "latest.pth"
        start, best, history = 0, -1., []
        if latest.exists():
            start, best, history = restore_checkpoint(latest, model, optimizer, scheduler, manifest)
        self.event("candidate_training_start", name=name, manifest=manifest, users=len(self.training_records))
        if start == 0:
            dataset = CandidateDataset(self.data, self.training_records, self.candidates, seed, recall_count)
            loader = DataLoader(dataset, batch_size=self.args.din_batch, shuffle=True,
                                generator=torch.Generator().manual_seed(seed), num_workers=self.args.workers,
                                collate_fn=collate, pin_memory=self.device.type == "cuda", drop_last=True)
            model.train()
            total, steps, hard, samples = 0., 0, 0, 0
            started = time.monotonic()
            for batch in tqdm(loader, desc=name, mininterval=20):
                hard += int(batch.pop("hard_count").sum())
                samples += len(batch["user_id"])
                batch = to_device(batch, self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    positive, negative = model.forward_bpr({k: batch[k] for k in USER_KEYS},
                         {k: batch["pos_" + k] for k in ITEM_KEYS}, {k: batch["neg_" + k] for k in ITEM_KEYS})
                    loss = F.softplus(negative.float() - positive.float()[:, None]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite candidate loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())
                steps += 1
                if self.options.max_train_steps and steps >= self.options.max_train_steps:
                    break
            if not steps:
                raise ValueError("No complete candidate training batch")
            scheduler.step()
            model.eval()
            pools = self.pools(self.screen, 100)
            rankings = score_din(model, self.data, self.screen, pools, self.device)
            best = float(np.mean([self.data.iid[p] in ranked[:5] for (_, p), ranked in zip(self.screen, rankings)]))
            history = [{"epoch": 1, "loss": total / steps, "steps": steps, "samples": samples,
                        "actual_recall_per_sample": hard / samples, "screen_hr5": best,
                        "seconds": time.monotonic() - started}]
            atomic_save({"model": model.state_dict(), "manifest": manifest, "epoch": 1}, directory / "best.pth")
            save_checkpoint(latest, model, optimizer, scheduler, 1, best, manifest, history)
            write_json(directory / "history.json", history)
        final = directory / "final.pth"
        if not final.exists():
            os.link(directory / "best.pth", final)
        write_json(completed, {"manifest": manifest, "history": history})
        latest.unlink()
        self.event("candidate_training_complete", name=name, history=history)
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        return directory / "best.pth"

    def execute(self):
        if (self.run / "CANDIDATE_COMPLETED.json").exists():
            return
        checkpoints = {"accepted": self.base_din}
        for name, count in (("random", 0), ("recall", 1)):
            checkpoint = self.train_candidate(count)
            if checkpoint is None:
                return
            checkpoints[name] = checkpoint
        screens = {}
        for name, checkpoint in checkpoints.items():
            screens[name], _ = self.evaluate(name + "_screen", self.screen, din_path=checkpoint)
        winner = max(screens, key=lambda k: (screens[k]["hr5"], screens[k]["ndcg5"], k == "accepted"))
        confirmed = {}
        for name, checkpoint in checkpoints.items():
            if not self.available():
                return
            confirmed[name] = self.evaluate(name + "_confirm", self.confirm, din_path=checkpoint)
        effect = paired_comparison(confirmed["recall"][1], confirmed["random"][1])
        comparison = paired_comparison(confirmed[winner][1], confirmed["accepted"][1])
        selected, replication = "accepted", None
        if comparison["confirmed_gain"] and (winner == "random" or effect["confirmed_gain"]):
            checkpoint = self.train_candidate(int(winner == "recall"), seed=43)
            if checkpoint is None:
                return
            _, values = self.evaluate("winner_seed43_confirm", self.confirm, din_path=checkpoint)
            replication = {"against_accepted": paired_comparison(values, confirmed["accepted"][1])}
            stable = replication["against_accepted"]["confirmed_gain"]
            if winner == "recall":
                control = self.train_candidate(0, seed=43)
                if control is None:
                    return
                _, reference = self.evaluate("control_seed43_confirm", self.confirm, din_path=control)
                replication["treatment_effect"] = paired_comparison(values, reference)
                stable &= replication["treatment_effect"]["confirmed_gain"]
            if stable:
                selected = winner
        write_json(self.run / "CANDIDATE_COMPLETED.json", {"screen": screens, "screen_winner": winner,
                   "confirmation": {k: v[0] for k, v in confirmed.items()}, "recall_vs_random": effect,
                   "against_accepted": comparison, "replication": replication, "selected": selected,
                   "din_checkpoint": str(checkpoints[selected]), "fusion_recipe": self.recipe_name,
                   "overall_complete": False})
        self.event("candidate_complete", selected=selected, overall_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fusion-run", required=True)
    parser.add_argument("--structural-run", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=10000)
    parser.add_argument("--confirm-users", type=int, default=90000)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--max-train-steps", type=int, default=0)
    args = parser.parse_args()
    if min(args.screen_users, args.confirm_users, args.hours) <= 0:
        parser.error("counts and hours must be positive")
    runner = CandidateExperiments(args)
    try:
        runner.execute()
    except Exception as error:
        runner.event("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
