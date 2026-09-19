"""Attribute future-window candidate coverage to the four baseline channels.

The script reuses ``run_baseline.candidate_pools`` and the exact prepared
``data.pkl``/ItemCF assets from a completed run.  It is validation-only: it
loads the V2 checkpoint, evaluates development users, and writes JSON stats.
"""
import argparse
import itertools
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from run_baseline import make_model, recall, targets_for_protocol, validate_fusion_weights


CHANNELS = ("itemcf", "v2", "category", "hot")
DAY_MS = 86400000.0


def _compact_itemcf_topk(candidate_ids, candidate_scores, valid, budget,
                         item_count, device):
    """Aggregate ItemCF scores without materialising a catalog-sized matrix.

    ``candidate_ids`` is shaped ``[batch, history, neighbors]``.  The old
    implementation used ``scatter_add`` into ``[batch, item_count]`` and then
    called ``topk`` over the complete catalog.  Most catalog entries can never
    receive a score in this operation, so sort the valid ``(row, item)`` keys,
    segment-sum duplicate neighbors, and rank only the resulting sparse rows.
    The input flattening order is retained by the stable key sort, preserving
    the accumulation order for duplicate entries.
    """
    batch = candidate_ids.shape[0]
    flat_valid = torch.as_tensor(valid, dtype=torch.bool, device=device).reshape(batch, -1)
    if not bool(flat_valid.any()):
        return (np.empty((batch, 0), dtype=np.int64),
                np.empty((batch, 0), dtype=np.float32))

    # ``segment_reduce`` is available in the supported PyTorch releases.  Keep
    # the former implementation as a compatibility path for older checkpoints
    # that may still be evaluated with an older runtime.
    if not hasattr(torch, "segment_reduce"):
        ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=device)
        scores = torch.as_tensor(np.where(valid, candidate_scores, 0.),
                                 dtype=torch.float32, device=device)
        totals = torch.zeros((batch, item_count), dtype=torch.float32, device=device)
        totals.scatter_add_(1, ids.reshape(batch, -1), scores.reshape(batch, -1))
        values, ids = totals.topk(min(int(budget), max(item_count - 2, 1)), dim=1)
        return ids.detach().cpu().numpy(), values.detach().cpu().numpy()

    ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=device).reshape(batch, -1)
    scores = torch.as_tensor(candidate_scores, dtype=torch.float32,
                             device=device).reshape(batch, -1)
    row_grid = torch.arange(batch, dtype=torch.long, device=device)[:, None]
    row_ids = row_grid.expand_as(ids)[flat_valid]
    ids = ids[flat_valid]
    scores = scores[flat_valid]

    # A row/item composite key gives one contiguous segment per duplicate item.
    keys = row_ids * int(item_count) + ids
    order = torch.argsort(keys, stable=True)
    keys = keys[order]
    scores = scores[order]
    unique_keys, lengths = torch.unique_consecutive(keys, return_counts=True)
    sums = torch.segment_reduce(scores, reduce="sum", lengths=lengths)
    grouped_rows = torch.div(unique_keys, int(item_count), rounding_mode="floor")
    grouped_ids = unique_keys.remainder(int(item_count))

    # Pack variable-length sparse rows into a compact dense matrix so one
    # stable argsort ranks every row in a single operation.
    row_counts = torch.bincount(grouped_rows, minlength=batch)
    max_count = int(row_counts.max().item())
    offsets = torch.cumsum(row_counts, 0) - row_counts
    positions = torch.arange(len(grouped_rows), device=device) - offsets[grouped_rows]
    packed_scores = torch.zeros((batch, max_count), dtype=torch.float32, device=device)
    packed_ids = torch.zeros((batch, max_count), dtype=torch.long, device=device)
    packed_scores[grouped_rows, positions] = sums
    packed_ids[grouped_rows, positions] = grouped_ids
    rank = torch.argsort(packed_scores, dim=1, descending=True, stable=True)
    take = min(int(budget), max_count)
    rank = rank[:, :take]
    values = torch.gather(packed_scores, 1, rank).detach().cpu().numpy()
    ids = torch.gather(packed_ids, 1, rank).detach().cpu().numpy()
    return ids, values


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _target_stats(records, targets, details, all_v2_items):
    """Return user/target hit sets and actual-vs-rank-outside classification."""
    actual_hits = {name: set() for name in CHANNELS}
    target_hits = {name: 0 for name in CHANNELS}
    reachable = {name: 0 for name in CHANNELS}
    rank_outside = {name: 0 for name in CHANNELS}
    unreachable = {name: 0 for name in CHANNELS}
    actual_sets = {name: [] for name in CHANNELS}
    full_sets = {name: [] for name in CHANNELS}
    target_count = 0

    for row, (target_set, detail) in enumerate(zip(targets, details)):
        target_set = set(map(int, target_set))
        target_count += len(target_set)
        channels = detail["channels"]
        full_channels = list(detail["full_channels"])
        # V2 scores the complete catalog before taking top-k; therefore every
        # unseen catalog item is reachable and an absent target is rank-outside.
        seen = detail["seen"]
        full_channels[1] = set(all_v2_items).difference(seen)
        for index, name in enumerate(CHANNELS):
            actual = set(map(int, channels[index]))
            full = set(map(int, full_channels[index]))
            actual_sets[name].append(actual)
            full_sets[name].append(full)
            hits = target_set.intersection(actual)
            possible = target_set.intersection(full)
            actual_hits[name].update([row] if hits else [])
            target_hits[name] += len(hits)
            reachable[name] += len(possible)
            rank_outside[name] += len(possible.difference(actual))
            unreachable[name] += len(target_set.difference(full))

    actual_union_users = set()
    actual_union_targets = 0
    full_union_targets = 0
    rank_outside_union = 0
    unreachable_union = 0
    for row, target_set in enumerate(targets):
        target_set = set(map(int, target_set))
        actual_union = set().union(*(actual_sets[name][row] for name in CHANNELS))
        full_union = set().union(*(full_sets[name][row] for name in CHANNELS))
        actual = target_set.intersection(actual_union)
        possible = target_set.intersection(full_union)
        if actual:
            actual_union_users.add(row)
        actual_union_targets += len(actual)
        full_union_targets += len(possible)
        rank_outside_union += len(possible.difference(actual))
        unreachable_union += len(target_set.difference(full_union))

    n_users = len(records)
    channel_result = {}
    for name in CHANNELS:
        actual_sizes = [len(items) for items in actual_sets[name]]
        channel_result[name] = {
            "users_hit": len(actual_hits[name]),
            "hr_at_100": len(actual_hits[name]) / max(n_users, 1),
            "target_hits": target_hits[name],
            "target_hit_rate": target_hits[name] / max(target_count, 1),
            "target_reachable": reachable[name],
            "target_rank_outside_k": rank_outside[name],
            "target_unreachable": unreachable[name],
            "candidate_count_mean": float(np.mean(actual_sizes)) if actual_sizes else 0.,
            "candidate_count_min": min(actual_sizes) if actual_sizes else 0,
            "candidate_count_max": max(actual_sizes) if actual_sizes else 0,
        }

    pairwise = {}
    pairwise_candidate = {}
    for left, right in itertools.combinations(CHANNELS, 2):
        key = left + "_and_" + right
        pairwise[key] = len(actual_hits[left].intersection(actual_hits[right]))
        overlaps = []
        for lset, rset in zip(actual_sets[left], actual_sets[right]):
            union = lset | rset
            overlaps.append(len(lset & rset) / len(union) if union else 0.)
        pairwise_candidate[key] = {
            "mean_jaccard": float(np.mean(overlaps)) if overlaps else 0.,
            "mean_shared_items": float(np.mean([
                len(lset & rset) for lset, rset in zip(actual_sets[left], actual_sets[right])
            ])) if overlaps else 0.,
        }

    unique_hits = {}
    for name in CHANNELS:
        others = set().union(*(actual_hits[o] for o in CHANNELS if o != name))
        unique_hits[name] = len(actual_hits[name] - others)

    return {
        "users": n_users,
        "target_pairs": target_count,
        "channels": channel_result,
        "unique_hit_users": unique_hits,
        "pairwise_hit_users": pairwise,
        "pairwise_candidate_overlap": pairwise_candidate,
        "union": {
            "users_hit": len(set().union(*actual_hits.values())) if actual_hits else 0,
            "hr_at_100": len(set().union(*actual_hits.values())) / max(n_users, 1),
            "target_hits": actual_union_targets,
            "target_reachable": full_union_targets,
            "target_rank_outside_k": rank_outside_union,
            "target_unreachable": unreachable_union,
        },
    }


def _iter_itemcf_channels(data, records, cf, budget, half_life_days, device,
                          batch_size=128):
    """Yield actual and uncapped ItemCF sets using batched torch scatter-add.

    The per-record formula is the same as ``candidate_pools``.  Histories and
    neighbor rows are gathered as one tensor batch; only the compact top-k
    scores and candidate IDs are copied back to Python.  Full reachability is
    recovered from the valid neighbor IDs, so no dense score matrix is kept.
    """
    neighbors, similarity = cf
    item_count = len(data.items)
    k_neighbors = neighbors.shape[1]
    hot = np.asarray([int(i) for i in np.argsort(-data.train_counts, kind="stable")
                      if i >= 2 and data.train_counts[i] > 0], dtype=np.int64)
    for offset in range(0, len(records), batch_size):
        chunk = records[offset:offset + batch_size]
        size = len(chunk)
        max_hist = max(data.hist_len, 1)
        hist_ids = np.zeros((size, max_hist), dtype=np.int64)
        event_pos = np.zeros((size, max_hist), dtype=np.int64)
        hist_valid = np.zeros((size, max_hist), dtype=bool)
        seen_sets = []
        for row, (uid, pos) in enumerate(chunk):
            end = data.history_end(uid, pos)
            start = max(data.starts[uid], end - data.hist_len)
            positions = np.arange(start, end, dtype=np.int64)
            length = min(len(positions), max_hist)
            if length:
                hist_ids[row, :length] = data.iid[positions[-length:]]
                event_pos[row, :length] = positions[-length:]
                hist_valid[row, :length] = True
            seen_sets.append(set(data.iid[data.starts[uid]:end].tolist()))

        # Gather all neighbor rows and apply query-time recency decay.
        candidate_ids = neighbors[hist_ids].reshape(size, max_hist, k_neighbors)
        candidate_scores = similarity[hist_ids].astype(np.float32).reshape(
            size, max_hist, k_neighbors)
        if half_life_days > 0:
            ends = np.asarray([data.history_end(uid, pos) for uid, pos in chunk], dtype=np.int64)
            last_ts = data.ts[ends - 1, None, None]
            ages = np.maximum((last_ts - data.ts[event_pos][:, :, None]) / DAY_MS, 0.)
            decay = np.exp(-math.log(2.) * ages / half_life_days).astype(np.float32)
            candidate_scores *= decay
        valid = hist_valid[:, :, None] & (candidate_ids >= 2) & (candidate_scores > 0)
        for row, seen in enumerate(seen_sets):
            if seen:
                valid[row] &= ~np.isin(candidate_ids[row], np.fromiter(seen, dtype=np.int64))

        # Aggregate only valid neighbor entries.  This avoids a dense
        # ``[batch, catalog_size]`` allocation, which dominates attribution
        # runtime and GPU memory for the large Amazon catalog.
        top_ids, top_values = _compact_itemcf_topk(
            candidate_ids, candidate_scores, valid, budget, item_count, device)

        actual, full = [], []
        for row, seen in enumerate(seen_sets):
            # topk is only used to reduce the candidate list.  Re-sort with
            # the baseline's deterministic (-score, item-id) order.
            pairs = [(int(iid), float(score)) for iid, score in zip(top_ids[row], top_values[row])
                     if int(iid) >= 2 and float(score) > 0.]
            pairs.sort(key=lambda pair: (-pair[1], pair[0]))
            chosen = {iid for iid, _ in pairs[:budget]}
            for iid in hot:
                if len(chosen) >= budget:
                    break
                if int(iid) not in seen and int(iid) not in chosen:
                    chosen.add(int(iid))
            raw_ids = np.unique(candidate_ids[row][valid[row]])
            raw_ids = {int(iid) for iid in raw_ids if int(iid) >= 2}
            actual.append(chosen)
            # Future targets are unseen by construction; hot reachability can
            # be checked against the global hot set during aggregation without
            # copying it for every user.
            full.append(raw_ids)
        yield actual, full


def _stream_stats(data, records, targets, v2_scores, cf, budget, half_life_days,
                  device, batch_size=128):
    """Compute attribution metrics without retaining 100k channel details."""
    n_users = len(records)
    actual_hits = {name: set() for name in CHANNELS}
    target_hits = {name: 0 for name in CHANNELS}
    reachable = {name: 0 for name in CHANNELS}
    rank_outside = {name: 0 for name in CHANNELS}
    unreachable = {name: 0 for name in CHANNELS}
    size_sum = {name: 0 for name in CHANNELS}
    size_min = {name: None for name in CHANNELS}
    size_max = {name: 0 for name in CHANNELS}
    pairwise = {left + "_and_" + right: 0
                for left, right in itertools.combinations(CHANNELS, 2)}
    pair_shared = {key: 0. for key in pairwise}
    pair_jaccard = {key: 0. for key in pairwise}
    target_count = 0
    actual_union_targets = full_union_targets = 0
    rank_outside_union = unreachable_union = 0
    actual_union_users = set()
    hot = np.asarray([int(i) for i in np.argsort(-data.train_counts, kind="stable")
                      if i >= 2 and data.train_counts[i] > 0], dtype=np.int64)
    hot_set = set(map(int, hot))
    category_hot = {cat: [int(i) for i in hot if data.item_category[i] == cat]
                    for cat in range(2, len(data.categories))}
    category_hot_sets = {cat: set(items) for cat, items in category_hot.items()}
    row_offset = 0
    for cf_actual, cf_full in _iter_itemcf_channels(
            data, records, cf, budget, half_life_days, device, batch_size):
        for local, ((uid, pos), target_set) in enumerate(
                zip(records[row_offset:row_offset + len(cf_actual)],
                    targets[row_offset:row_offset + len(cf_actual)])):
            row = row_offset + local
            target_set = set(map(int, target_set))
            target_count += len(target_set)
            end = data.history_end(uid, pos)
            seen = set(data.iid[data.starts[uid]:end].tolist())
            hist_start = max(data.starts[uid], end - data.hist_len)
            hist = data.iid[hist_start:end]
            cats, counts = np.unique(data.item_category[hist], return_counts=True)
            cat_actual = set()
            top_categories = []
            for cat, count in sorted(zip(cats, counts), key=lambda x: -x[1])[:3]:
                cat = int(cat)
                top_categories.append(cat)
                added = 0
                for item in category_hot.get(cat, ()):
                    if item not in seen:
                        cat_actual.add(item)
                        added += 1
                        if added >= 20:
                            break
            top_category_set = set(top_categories)
            # For future targets, history exclusion is already guaranteed. The
            # uncapped category reachability can therefore be tested directly
            # against the target items' category and global hot membership.
            cat_full = {item for item in target_set
                        if int(data.item_category[item]) in top_category_set
                        and item in category_hot_sets.get(int(data.item_category[item]), set())}
            hot_actual = set(int(i) for i in [i for i in hot if int(i) not in seen][:5])
            channels = {"itemcf": cf_actual[local],
                        "v2": set(map(int, v2_scores[row])),
                        "category": cat_actual, "hot": hot_actual}
            full_channels = {"itemcf": cf_full[local] | hot_set, "v2": None,
                             "category": cat_full,
                             "hot": hot_set}
            for name in CHANNELS:
                actual = channels[name]
                full = full_channels[name]
                size = len(actual)
                size_sum[name] += size
                size_min[name] = size if size_min[name] is None else min(size_min[name], size)
                size_max[name] = max(size_max[name], size)
                hits = target_set.intersection(actual)
                possible = (target_set if name == "v2" else target_set.intersection(full))
                if hits:
                    actual_hits[name].add(row)
                target_hits[name] += len(hits)
                reachable[name] += len(possible)
                rank_outside[name] += len(possible.difference(actual))
                unreachable[name] += len(target_set.difference(possible))
            actual_union = set().union(*(channels[name] for name in CHANNELS))
            actual = target_set.intersection(actual_union)
            # V2 can rank every unseen catalog item before the top-k cap.
            possible = target_set
            if actual:
                actual_union_users.add(row)
            actual_union_targets += len(actual)
            full_union_targets += len(possible)
            rank_outside_union += len(possible.difference(actual))
            unreachable_union += 0
            for left, right in itertools.combinations(CHANNELS, 2):
                key = left + "_and_" + right
                left_set, right_set = channels[left], channels[right]
                if target_set.intersection(left_set) and target_set.intersection(right_set):
                    pairwise[key] += 1
                shared = len(left_set & right_set)
                pair_shared[key] += shared
                union = len(left_set | right_set)
                pair_jaccard[key] += shared / union if union else 0.
        row_offset += len(cf_actual)

    channel_result = {}
    for name in CHANNELS:
        channel_result[name] = {
            "users_hit": len(actual_hits[name]),
            "hr_at_100": len(actual_hits[name]) / max(n_users, 1),
            "target_hits": target_hits[name],
            "target_hit_rate": target_hits[name] / max(target_count, 1),
            "target_reachable": reachable[name],
            "target_rank_outside_k": rank_outside[name],
            "target_unreachable": unreachable[name],
            "candidate_count_mean": size_sum[name] / max(n_users, 1),
            "candidate_count_min": size_min[name] or 0,
            "candidate_count_max": size_max[name],
        }
    pairwise_candidate = {
        key: {"mean_jaccard": pair_jaccard[key] / max(n_users, 1),
              "mean_shared_items": pair_shared[key] / max(n_users, 1)}
        for key in pairwise
    }
    unique_hits = {}
    for name in CHANNELS:
        others = set().union(*(actual_hits[o] for o in CHANNELS if o != name))
        unique_hits[name] = len(actual_hits[name] - others)
    return {
        "users": n_users,
        "target_pairs": target_count,
        "channels": channel_result,
        "unique_hit_users": unique_hits,
        "pairwise_hit_users": pairwise,
        "pairwise_candidate_overlap": pairwise_candidate,
        "union": {"users_hit": len(actual_union_users),
                   "hr_at_100": len(actual_union_users) / max(n_users, 1),
                   "target_hits": actual_union_targets,
                   "target_reachable": full_union_targets,
                   "target_rank_outside_k": rank_outside_union,
                   "target_unreachable": unreachable_union},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Completed future-window run containing data.pkl, svd.npy and itemcf.pkl")
    parser.add_argument("--checkpoint-dir", default="",
                        help="Directory containing v2_best.pth; defaults to --run-dir")
    parser.add_argument("--users", type=int, default=100000,
                        help="Number of deterministic development users to evaluate")
    parser.add_argument("--budget", type=int, default=0,
                        help="Channel budget; defaults to the run manifest candidates value")
    parser.add_argument("--output", default="",
                        help="Output JSON; defaults to <run-dir>/future_window_attribution.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="ItemCF GPU aggregation batch size")
    args = parser.parse_args()
    if args.users < 1 or args.batch_size < 1:
        parser.error("--users and --batch-size must be positive")

    run = Path(args.run_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else run
    spec_path = run / "run_manifest.json"
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)
    spec = _read_json(spec_path)
    run_args = spec.get("args", {})
    if run_args.get("protocol") != "future-window":
        raise ValueError("attribution requires a future-window run manifest")
    budget = args.budget or int(run_args.get("candidates", 100))
    fusion_mode = run_args.get("fusion_mode", "quota")
    fusion_weights = validate_fusion_weights(run_args.get("fusion_weights"))
    half_life = float(run_args.get("itemcf_half_life_days", 0.))

    with (run / "data.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    data = saved["data"]
    with (run / "itemcf.pkl").open("rb") as stream:
        cf = pickle.load(stream)
    checkpoint = checkpoint_dir / "v2_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) \
        if args.device == "auto" else torch.device(args.device)
    model_args = argparse.Namespace(**run_args)
    model = make_model(data, model_args, "v2", device)
    saved_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved_checkpoint["model"])
    model.eval()

    records = data.evaluation("val", args.users)
    if not records:
        raise ValueError("no development records available")
    targets = targets_for_protocol(data, records, "future-window")
    print(f"Evaluating {len(records):,} future-window development users", flush=True)
    _, scores = recall(model, data, records, device, budget)
    result = _stream_stats(data, records, targets, scores, cf, budget, half_life,
                           device, batch_size=args.batch_size)
    result.update({
        "protocol": "future-window",
        "run_dir": str(run),
        "checkpoint_dir": str(checkpoint_dir),
        "data_id": data.manifest.get("data_id"),
        "users_requested": args.users,
        "budget": budget,
        "fusion_mode": fusion_mode,
        "fusion_weights": fusion_weights,
        "itemcf_half_life_days": half_life,
        "interpretation": {
            "actual": "target appears in the channel's current candidate set",
            "rank_outside_k": "target is in an uncapped train-only channel reachability set but omitted by the current cap",
            "unreachable": "target is absent from every uncapped channel reachability set",
        },
    })
    output = Path(args.output).resolve() if args.output else run / "future_window_attribution.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
