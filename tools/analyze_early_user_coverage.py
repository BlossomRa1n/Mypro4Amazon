"""Descriptive coverage analysis of already archived development predictions.

No model scoring, target lookup, bootstrap, selection changes, or final test reads.
This post-hoc stratification cannot establish a causal effect of user exposure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    base = root / 'server_snapshot/next_cross_20260924/base_archive/data.pkl'
    old = root / 'server_snapshot/next_cross_20260924/protocol_complete'
    run = root / 'server_snapshot/training_diagnostics_20260924/run'
    legacy = root / 'server_snapshot/training_diagnostics_20260924/remote_root/src_v2/code'
    require(sha(base) == 'e71413ee4eead36954650c1ddb99c8334ead9edecee75ab6b9c01012efd9bb06', 'base drift')
    sys.path.insert(0, str(legacy))
    with base.open('rb') as stream:
        value = pickle.load(stream)
    data = value.get('data') if isinstance(value, dict) and 'data' in value else value
    positions = np.load(old / 'ranker_train_positions.npy', allow_pickle=False)
    records = read(old / 'confirm_records.json')
    uid = np.asarray([row[0] for row in records], dtype=np.int64)
    position = np.asarray([row[1] for row in records], dtype=np.int64)
    total_counts = np.bincount(np.asarray(data.uid)[positions], minlength=len(data.users))
    completed = read(run / 'COMPLETED.json')
    require(completed['status'] == 'complete' and completed['test_future_labels_read'] is False, 'raw diagnostic incomplete')
    results = {}
    sources = {str(p.relative_to(root)): sha(p) for p in (base, old/'ranker_train_positions.npy', old/'confirm_records.json', run/'COMPLETED.json')}
    for seed in (42, 43, 44):
        group = run / f'seed_{seed}/raw'
        files = ['epoch_1_trace.json', 'confirm_best_users.npz', 'confirm_last_users.npz']
        for name in files:
            relative = f'seed_{seed}/raw/{name}'
            require(sha(group/name) == completed['evidence_hashes'][relative], 'raw evidence drift: '+relative)
            sources[str((group/name).relative_to(root))] = sha(group/name)
        trace = read(group/'epoch_1_trace.json')
        order = np.random.default_rng(seed).permutation(len(positions)).astype(np.int64)
        require(hashlib.sha256(np.ascontiguousarray(order).tobytes()).hexdigest() == trace['order_sha256'], 'permutation trace mismatch')
        step = completed['seeds'][str(seed)]['best']['step']
        consumed_rows = sum(trace['batch_sizes'][:step])
        exposures = np.bincount(np.asarray(data.uid)[positions[order[:consumed_rows]]], minlength=len(data.users))[uid]
        with np.load(group/'confirm_best_users.npz', allow_pickle=False) as b, np.load(group/'confirm_last_users.npz', allow_pickle=False) as l:
            best = {k:b[k].copy() for k in b.files}
            last = {k:l[k].copy() for k in l.files}
        for arrays in (best, last):
            require(np.array_equal(arrays['uid'], uid) and np.array_equal(arrays['position'], position), 'confirm identities differ')
        for key in ('candidate_ids', 'lengths', 'labels', 'pool_hit', 'auc_valid'):
            require(np.array_equal(best[key], last[key]), 'paired input differs: '+key)
        masks = {'no_user_target_consumed_at_best': exposures == 0, 'at_least_one_user_target_consumed_at_best': exposures > 0}
        rows = {}
        for label, mask in masks.items():
            n = int(mask.sum())
            require(n > 0, 'empty descriptive subgroup')
            values = {'users':n, 'mean_targets_consumed_at_best':float(exposures[mask].mean()), 'mean_total_legal_training_rows':float(total_counts[uid[mask]].mean()), 'median_total_legal_training_rows':float(np.median(total_counts[uid[mask]])), 'pool_hit':float(best['pool_hit'][mask].mean())}
            for metric in ('hit5', 'ndcg5'):
                values[metric] = {'best':float(best[metric][mask].mean()), 'last':float(last[metric][mask].mean()), 'best_minus_last':float((best[metric][mask].astype(float)-last[metric][mask]).mean())}
            valid = mask & best['auc_valid']
            values['candidate_gauc_valid_users'] = int(valid.sum())
            values['candidate_gauc_coverage'] = float(valid.sum()/n)
            values['candidate_gauc'] = {'best':float(best['auc'][valid].mean()) if valid.any() else None, 'last':float(last['auc'][valid].mean()) if valid.any() else None}
            rows[label] = values
        require(sum(row['users'] for row in rows.values()) == len(uid), 'subgroup coverage mismatch')
        for metric in ('hit5', 'ndcg5'):
            weighted = sum(row['users']*row[metric]['best_minus_last'] for row in rows.values())/len(uid)
            expected = float((best[metric].astype(float)-last[metric]).mean())
            require(abs(weighted-expected)<1e-12, 'subgroup decomposition mismatch')
        results[str(seed)] = {'best_step':step, 'consumed_rows_at_best':consumed_rows, 'groups':rows}
    report = {'status':'complete', 'analysis':'post-hoc descriptive exposure stratification; not an acceptance test', 'test100k_accessed':False, 'new_model_scoring':False, 'sources':sources, 'per_seed':results, 'limitations':['Exposure strata depend on randomized training order and user history length; differences are not randomized treatment effects.', 'No user-specific positive target consumed does not imply unchanged embedding: shared gradients and AdamW effects remain possible.', 'Already reused development confirm; no fresh holdout claim.', 'No new bootstrap or change to preregistered model-selection criteria.']}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':'complete','per_seed':results}, ensure_ascii=False))


if __name__ == '__main__':
    main()
