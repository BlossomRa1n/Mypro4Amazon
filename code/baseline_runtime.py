"""Shared checkpoint, ranking metric and candidate contracts for the baseline."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, manifest, history):
    atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
                 "manifest": manifest, "history": history,
                 "random": random.getstate(), "numpy_random": np.random.get_state(),
                 "torch_random": torch.get_rng_state(),
                 "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def restore_checkpoint(path, model, optimizer, scheduler, manifest):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["manifest"] != manifest:
        raise ValueError("Checkpoint data/config/code identity mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    if state["cuda_random"]:
        torch.cuda.set_rng_state_all(state["cuda_random"])
    return state["epoch"], state["best"], state["history"]


def rebind_optimizer(optimizer, model):
    parameters = list(model.parameters())
    if sum(len(g["params"]) for g in optimizer.param_groups) != len(parameters):
        raise ValueError("Cannot resume: optimizer/model parameter structure differs")
    offset = 0
    optimizer.state.clear()
    for group in optimizer.param_groups:
        count = len(group["params"])
        group["params"] = parameters[offset:offset + count]
        offset += count


def pair_auc(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if negative.ndim != 2 or negative.shape[0] != len(positive):
        raise ValueError("Negative scores must have shape [users, negatives]")
    return float(((positive[:, None] > negative) + .5 * (positive[:, None] == negative)).mean())


def ranking_metrics(rankings, targets, k):
    if len(rankings) != len(targets):
        raise ValueError("Each evaluation user must have one ranking")
    hit, ndcg, recall = [], [], []
    for ranked, target in zip(rankings, targets):
        gt = set(target) if isinstance(target, (set, list, tuple, np.ndarray)) else {int(target)}
        ranked = list(dict.fromkeys(ranked))[:k]
        gains = [int(i in gt) for i in ranked]
        found = sum(gains)
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt))))
        hit.append(float(found > 0))
        recall.append(found / max(len(gt), 1))
        ndcg.append(dcg / max(ideal, 1e-12))
    return {"hr": float(np.mean(hit)) if hit else 0.,
            "ndcg": float(np.mean(ndcg)) if ndcg else 0.,
            "recall": float(np.mean(recall)) if recall else 0., "users": len(targets), "k": k}


def quota_merge(channels, weights, budget, excluded=()):
    """Select channel quotas, then refill duplicates from remaining ranked items."""
    excluded = set(excluded)
    ordered = [sorted(ch.items(), key=lambda pair: (-pair[1], str(pair[0])))
               if isinstance(ch, dict) else sorted(ch, key=lambda pair: (-pair[1], str(pair[0])))
               for ch in channels]
    total = sum(max(w, 0) for w in weights)
    if total <= 0:
        return {}
    quotas = [max(1, int(budget * w / total)) if w > 0 else 0 for w in weights]
    scores, chosen = {}, set()
    for entries, w, q in zip(ordered, weights, quotas):
        if w <= 0:
            continue
        valid = [(iid, score) for iid, score in entries if iid not in excluded]
        chosen.update(iid for iid, _ in valid[:q])
        for iid, score in valid:
            scores[iid] = scores.get(iid, 0.) + w * score
    ranked = sorted(scores, key=lambda iid: (-scores[iid], str(iid)))
    for iid in ranked:
        if len(chosen) >= budget:
            break
        chosen.add(iid)
    return {iid: scores[iid] for iid in ranked if iid in chosen and iid not in excluded} if len(chosen) <= budget else {
        iid: scores[iid] for iid in [i for i in ranked if i in chosen][:budget]}
