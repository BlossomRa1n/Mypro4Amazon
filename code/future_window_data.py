"""Predict the whole positive suffix from a fixed, strictly earlier 80% prefix."""
import numpy as np

from baseline_data import BenchmarkData, fingerprint


class FutureWindowData(BenchmarkData):
    protocol = "positive-first-interaction-user-prefix80-future20-v1"

    def __init__(self, frame, brands=None, hist_len=50, seed=42, test_users=100000,
                 prior_evaluation_users=()):
        super().__init__(frame, brands, hist_len, seed)
        # Replace leave-two-out positions from BenchmarkData with a strict 80/20 prefix.
        self.split()
        excluded = set(prior_evaluation_users)
        eligible = [i for i in self.eval_selection if self.users[self.val_users[i]] not in excluded]
        if test_users < 1 or len(eligible) < test_users:
            raise ValueError("Insufficient eligible test users outside previous evaluation users")
        self.test_selection = np.asarray(eligible[:test_users], dtype=np.int64)
        selected = set(self.test_selection.tolist())
        self.dev_selection = np.asarray([i for i in self.eval_selection if i not in selected], dtype=np.int64)
        if not len(self.dev_selection):
            raise ValueError("No development users remain")
        self.manifest.update({"split_note": "80% user-prefix to whole future positive suffix; disjoint dev/test users",
                              "test_users": test_users, "development_users": len(self.dev_selection),
                              "test_users_hash": fingerprint(self.val_users[self.test_selection].tolist()),
                              "prior_evaluation_exclusion_hash": fingerprint(sorted(excluded)),
                              "target_policy": "all unique positive products in held-out suffix; no sequential reveal"})
        self.manifest.pop("data_id")
        self.manifest["data_id"] = fingerprint(self.manifest)

    def split(self):
        self.train_ends = self.ends.copy()
        users, positions = [], []
        for uid in range(2, len(self.users)):
            start, end = int(self.starts[uid]), int(self.ends[uid])
            if end - start < 2:
                continue
            position = start + min(max((end - start) * 4 // 5, 1), end - start - 1)
            position = start + int(np.searchsorted(self.ts[start:end], self.ts[position], side="left"))
            if position == start:
                continue
            self.train_ends[uid] = position
            users.append(uid)
            positions.append(position)
        self.val_users = np.asarray(users, dtype=np.int32)
        self.val_positions = np.asarray(positions, dtype=np.int64)
        self.test_positions = self.val_positions.copy()

    def evaluation(self, split="val", max_users=0):
        if split not in ("val", "test"):
            raise ValueError(split)
        indices = self.dev_selection if split == "val" else self.test_selection
        indices = indices[:max_users or len(indices)]
        return [(int(self.val_users[i]), int(self.val_positions[i])) for i in indices]

    def targets(self, records):
        return [set(map(int, self.iid[position:self.ends[uid]])) for uid, position in records]
