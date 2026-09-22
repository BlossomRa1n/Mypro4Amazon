"""Versioned, positive next-item benchmark data and strict prefix samples."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROTOCOL = "positive-first-interaction-leave-two-out-v1"
PAD, UNK = 0, 1
DAY_MS = 86400000.0
USER_KEYS = ("user_id", "hist_items", "hist_brands", "hist_ratings",
             "hist_time_deltas", "hist_verified", "hist_len", "click_count",
             "time_span", "user_avg_rating", "user_std_rating",
             "user_verified_ratio", "user_avg_helpful")
ITEM_KEYS = ("item_id", "category_id", "brand_id", "item_click_count",
             "created_at_ts", "item_avg_rating", "item_rating_number")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def encode(values):
    classes = ["__PAD__", "__UNKNOWN__"] + sorted(set(map(str, values)))
    if len(set(classes)) != len(classes):
        raise ValueError("Reserved ID occurs in input")
    return classes, {value: idx for idx, value in enumerate(classes)}


def load_data(data_dir, categories, sample_users=0, seed=42):
    frames = []
    for category in categories:
        frame = pd.read_csv(Path(data_dir) / (category + ".csv"),
                            dtype={"user_id": str, "parent_asin": str})
        frame = frame.loc[frame.rating >= 4, ["user_id", "parent_asin", "rating", "timestamp"]]
        frame["category"] = category
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True).dropna()
    frame = frame.sort_values(["user_id", "timestamp", "parent_asin"], kind="stable")
    frame = frame.drop_duplicates(["user_id", "parent_asin"], keep="first")
    if sample_users:
        users = np.sort(frame.user_id.unique())
        selected = np.random.default_rng(seed).choice(users, min(sample_users, len(users)), replace=False)
        frame = frame.loc[frame.user_id.isin(selected)]
    return frame.reset_index(drop=True)


def load_brands(data_dir, categories, items):
    """Only static brand IDs are used; no future rating/helpful aggregates."""
    wanted = set(items)
    brands = {}
    sources = []
    for category in categories:
        candidates = [Path(data_dir) / "raw/meta_categories" / ("meta_" + category + ".jsonl"),
                      Path(data_dir) / (category + "_meta.jsonl")]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        sources.append(str(path))
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                obj = json.loads(line)
                iid = obj.get("parent_asin")
                if iid not in wanted or iid in brands:
                    continue
                details = obj.get("details") or {}
                brand = obj.get("store") or (details.get("Brand") if isinstance(details, dict) else None)
                if brand:
                    brands[iid] = str(brand).strip()
    return brands, sources


class BenchmarkData:
    protocol = PROTOCOL

    def __init__(self, frame, brands=None, hist_len=50, seed=42):
        self.hist_len, self.seed = hist_len, seed
        self.users, user_map = encode(frame.user_id)
        self.items, item_map = encode(frame.parent_asin)
        self.categories, category_map = encode(frame.category)
        brands = brands or {}
        self.brands, brand_map = encode(brands.values())
        self.uid = frame.user_id.map(user_map).to_numpy(np.int32)
        self.iid = frame.parent_asin.map(item_map).to_numpy(np.int32)
        self.ts = frame.timestamp.to_numpy(np.int64)
        self.rating = frame.rating.to_numpy(np.float32)
        n_users, n_items = len(self.users), len(self.items)
        counts = np.bincount(self.uid, minlength=n_users)
        self.ends = counts.cumsum()
        self.starts = self.ends - counts
        self.split()
        self.train_mask = np.arange(len(frame)) < self.train_ends[self.uid]
        self.train_positions = np.flatnonzero(self.train_mask & (self.ts > self.ts[self.starts[self.uid]]))
        self.train_counts = np.bincount(self.iid[self.train_mask], minlength=n_items)
        self.active_items = np.flatnonzero(self.train_counts > 0)
        self.active_set = set(self.active_items.tolist())
        self.train_sets = [set(self.iid[self.starts[u]:self.train_ends[u]].tolist()) for u in range(n_users)]
        self.item_category = np.zeros(n_items, dtype=np.int64)
        self.item_brand = np.zeros(n_items, dtype=np.int64)
        item_rows = frame.drop_duplicates("parent_asin")
        item_idx = item_rows.parent_asin.map(item_map).to_numpy()
        self.item_category[item_idx] = item_rows.category.map(category_map).to_numpy()
        self.item_brand[2:] = [brand_map.get(brands.get(raw), UNK) for raw in self.items[2:]]
        self.item_pop = (np.log1p(self.train_counts) / max(np.log1p(self.train_counts.max()), 1)).astype(np.float32)
        self.item_created = np.zeros(n_items, dtype=np.float32)
        self.item_avg = np.zeros(n_items, dtype=np.float32)
        train = frame.loc[self.train_mask].assign(item_idx=self.iid[self.train_mask])
        aggregate = train.groupby("item_idx").agg(ts=("timestamp", "min"), rating=("rating", "mean"))
        self.time_origin = int(self.ts.min())
        self.time_scale = max(int(self.ts[self.train_mask].max()) - self.time_origin, 1)
        self.item_created[aggregate.index] = (aggregate.ts.to_numpy() - self.time_origin) / self.time_scale
        self.item_avg[aggregate.index] = aggregate.rating.to_numpy() / 5.0
        self.count_scale = max(float(np.log1p((self.train_ends - self.starts).max())), 1)
        self.span_scale = max(float(np.log1p(self.time_scale / DAY_MS)), 1)
        self.eval_selection = np.random.default_rng(seed).permutation(len(self.val_users))
        hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in
                  [("uid", self.uid), ("iid", self.iid), ("ts", self.ts),
                   ("rating", self.rating), ("train_mask", self.train_mask),
                   ("brand", self.item_brand), ("category", self.item_category)]}
        self.manifest = {"protocol": self.protocol, "hist_len": hist_len, "seed": seed,
                         "hashes": hashes, "encoders_hash": fingerprint([self.users, self.items, self.brands, self.categories]),
                         "interactions": len(frame), "train_interactions": int(self.train_mask.sum()),
                         "train_samples": len(self.train_positions), "eval_users": len(self.val_users),
                         "num_users": n_users, "num_items": n_items,
                         "candidate_policy": "all catalog IDs >=2, exclude complete observed positive history",
                         "static_features": "CSV category and optional raw-meta brand; historical availability assumed",
                         "statistics": "fit on train only; no meta rating/helpful/verified snapshot",
                         "split_note": "user-wise next-positive benchmark, not a global deployment-time backtest"}
        self.manifest["data_id"] = fingerprint(self.manifest)

    def split(self):
        self.train_ends = self.ends.copy()
        self.val_users, self.val_positions, self.test_positions = [], [], []
        # A tied timestamp is one event boundary: it cannot straddle history/target.
        for uid in range(2, len(self.users)):
            start, end = self.starts[uid], self.ends[uid]
            if end - start < 3:
                continue
            val_pos, test_pos = end - 2, end - 1
            if self.ts[val_pos] <= self.ts[start] or self.ts[test_pos] <= self.ts[val_pos]:
                continue
            train_end = start + np.searchsorted(self.ts[start:end], self.ts[val_pos], side="left")
            self.train_ends[uid] = train_end
            self.val_users.append(uid)
            self.val_positions.append(val_pos)
            self.test_positions.append(test_pos)
        self.val_users = np.asarray(self.val_users, dtype=np.int32)
        self.val_positions = np.asarray(self.val_positions, dtype=np.int64)
        self.test_positions = np.asarray(self.test_positions, dtype=np.int64)

    def history_end(self, uid, position):
        start = self.starts[uid]
        return start + np.searchsorted(self.ts[start:position + 1], self.ts[position], side="left")

    def user_features(self, uid, position):
        end = self.history_end(uid, position)
        start = max(self.starts[uid], end - self.hist_len)
        hist = self.iid[start:end]
        times = self.ts[start:end]
        ratings = self.rating[start:end]
        length = len(hist)
        pad = lambda values: np.pad(values, (0, self.hist_len - length))
        gaps = np.concatenate(([0.0], np.diff(times) / DAY_MS)) if length else np.empty(0)
        # Fixed unit: one interval unit = 30 days, shared by train and evaluation.
        gaps = np.minimum(gaps / 30.0, 365.0 / 30.0)
        count = end - self.starts[uid]
        span = (times[-1] - times[0]) / DAY_MS if length > 1 else 0
        return {"user_id": np.int64(uid), "hist_items": pad(hist).astype(np.int64),
                "hist_brands": pad(self.item_brand[hist]).astype(np.int64),
                "hist_ratings": pad(ratings).astype(np.float32),
                "hist_time_deltas": pad(gaps).astype(np.float32),
                "hist_verified": np.zeros(self.hist_len, dtype=np.float32), "hist_len": np.int64(length),
                "click_count": np.float32(np.log1p(count) / self.count_scale),
                "time_span": np.float32(np.log1p(span) / self.span_scale),
                "user_avg_rating": np.float32(ratings.mean() / 5 if length else 0),
                "user_std_rating": np.float32(ratings.std() / 2 if length else 0),
                "user_verified_ratio": np.float32(0), "user_avg_helpful": np.float32(0)}

    def item_features(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        return {"item_id": ids, "category_id": self.item_category[ids], "brand_id": self.item_brand[ids],
                "item_click_count": self.item_pop[ids], "created_at_ts": self.item_created[ids],
                "item_avg_rating": self.item_avg[ids], "item_rating_number": self.item_pop[ids]}

    def evaluation(self, split="val", max_users=0):
        selection = self.eval_selection[:max_users or len(self.eval_selection)]
        positions = self.val_positions if split == "val" else self.test_positions
        return [(int(self.val_users[i]), int(positions[i])) for i in selection]

    def negatives(self, uid, target, k, rng, excluded=None):
        excluded = self.train_sets[uid] if excluded is None else excluded
        if len(self.active_items) - len(excluded & self.active_set) < k + 1:
            pool = np.asarray([i for i in self.active_items if i != target and i not in excluded])
            if len(pool) < k:
                raise ValueError("Too few eligible negative items")
            return rng.choice(pool, k, replace=False)
        selected = set()
        while len(selected) < k:
            for iid in rng.choice(self.active_items, max(2 * k, 16)):
                if iid != target and iid not in excluded:
                    selected.add(int(iid))
                if len(selected) == k:
                    break
        return np.asarray(sorted(selected), dtype=np.int64)


class PrefixDataset(Dataset):
    """Prefix samples with deterministic random or mixed candidate negatives.

    ``candidate_pools`` must be in the same order as ``data.train_positions``
    and contain ranked item IDs for each training position.  The mixed policy
    takes two candidates from ranks 11--25 and 26--50 (one-based), then fills
    the remaining slots with random eligible negatives.  Candidate IDs are
    always filtered against the user's complete known training positives.
    """
    def __init__(self, data, negatives=4, negative_policy="random",
                 candidate_pools=None, candidate_ranges=((11, 25, 2), (26, 50, 2)),
                 fixed_random_pools=None):
        self.data, self.k, self.epoch = data, int(negatives), 0
        self.negative_policy = str(negative_policy)
        self.candidate_ranges = tuple(tuple(int(value) for value in row)
                                      for row in candidate_ranges)
        if self.k < 1:
            raise ValueError("negatives must be positive")
        if self.negative_policy not in ("random", "mixed_rrf"):
            raise ValueError("negative_policy must be random or mixed_rrf")
        if self.negative_policy == "mixed_rrf":
            candidate_pools = candidate_pools if candidate_pools is not None else ()
            if len(candidate_pools) != len(self):
                raise ValueError("candidate_pools must match train sample count")
            if sum(row[2] for row in self.candidate_ranges) >= self.k:
                raise ValueError("candidate quotas must be smaller than negatives")
        elif candidate_pools is not None:
            raise ValueError("candidate_pools only applies to mixed_rrf policy")
        self.candidate_pools = candidate_pools
        self.fixed_random_pools = fixed_random_pools
        if fixed_random_pools is not None:
            if negative_policy != "random":
                raise ValueError("fixed_random_pools only applies to random policy")
            if len(fixed_random_pools) != len(self):
                raise ValueError("fixed_random_pools must match train sample count")
            if fixed_random_pools.ndim != 2 or fixed_random_pools.shape[1] < self.k:
                raise ValueError("fixed_random_pools must have at least k columns")

    def _ranked_pool(self, idx):
        pool = self.candidate_pools[idx]
        if isinstance(pool, dict):
            # Candidate-pool dicts preserve the RRF insertion order.
            pool = list(pool)
        return np.asarray(pool, dtype=np.int64).reshape(-1)

    def _mixed_negatives(self, idx, uid, target, rng):
        excluded = set(self.data.train_sets[uid])
        excluded.add(int(target))
        selected = []
        seen = set(excluded)
        pool = self._ranked_pool(idx)
        for start, stop, quota in self.candidate_ranges:
            if start < 1 or stop < start or quota < 0:
                raise ValueError("invalid candidate rank range")
            eligible = [int(item) for item in pool[start - 1:stop]
                        if int(item) not in seen and int(item) >= 2]
            take = min(quota, len(eligible))
            if take:
                chosen = rng.choice(np.asarray(eligible, dtype=np.int64), take, replace=False)
            else:
                chosen = np.asarray([], dtype=np.int64)
            selected.extend(int(item) for item in chosen)
            seen.update(int(item) for item in chosen)
        # A user's known-positive history can remove items from a rank band.
        # Fill missing candidate quotas from the remaining ranked pool before
        # falling back to catalog-random negatives, and leave the fallback
        # count auditable in the sampler manifest.
        missing = sum(row[2] for row in self.candidate_ranges) - len(selected)
        if missing:
            remaining = [int(item) for item in pool if int(item) >= 2 and int(item) not in seen]
            take = min(missing, len(remaining))
            if take:
                chosen = rng.choice(np.asarray(remaining, dtype=np.int64), take, replace=False)
                selected.extend(int(item) for item in chosen)
                seen.update(int(item) for item in chosen)
        random_count = self.k - len(selected)
        if random_count:
            random = self.data.negatives(uid, target, random_count, rng, excluded=seen)
            selected.extend(int(item) for item in random)
        if len(selected) != self.k or len(set(selected)) != self.k:
            raise AssertionError("negative sampler produced duplicate or missing items")
        return np.asarray(selected, dtype=np.int64)

    def __len__(self):
        return len(self.data.train_positions)

    def __getitem__(self, idx):
        data = self.data
        position = data.train_positions[idx]
        uid, target = int(data.uid[position]), int(data.iid[position])
        batch = data.user_features(uid, position)
        batch.update({"pos_" + key: value for key, value in data.item_features(target).items()})
        rng = np.random.default_rng(np.random.SeedSequence([data.seed, self.epoch, int(idx)]))
        if self.negative_policy == "random":
            if self.fixed_random_pools is None:
                negatives = data.negatives(uid, target, self.k, rng)
            else:
                negatives = np.asarray(self.fixed_random_pools[idx, :self.k], dtype=np.int64)
                if len(set(negatives.tolist())) != self.k or any(
                        int(item) < 2 or int(item) in data.train_sets[uid] or int(item) == target
                        for item in negatives):
                    raise AssertionError("fixed random pool contains invalid negative")
        else:
            negatives = self._mixed_negatives(idx, uid, target, rng)
        batch.update({"neg_" + key: value for key, value in data.item_features(negatives).items()})
        return batch


def collate(rows):
    return {key: torch.as_tensor(np.stack([row[key] for row in rows])) for key in rows[0]}
