"""Reanalyse archived gated versus raw cross; no training or server access."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compare_token_results import load_metrics


def analyse(run_dir):
    reference, candidate = 'cross_raw', 'cross_gated_normalized'
    expected = dict(line.strip().split('  ', 1)[::-1]
                    for line in (run_dir / 'files.sha256').read_text().splitlines())
    hashes = {}

    def verify(name):
        digest = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        if digest != expected['./' + name]:
            raise ValueError('Archived SHA-256 mismatch: ' + name)
        hashes[name] = digest

    output = dict(reference=reference, candidate=candidate,
                  direction='candidate minus reference', units='rate, not percentage points',
                  bootstrap_replicates=10000, bootstrap_seed=20260922,
                  method='paired user bootstrap percentile, two-sided 95%; no multiplicity adjustment',
                  scope='single trained seed; exploratory reused screen/test, not independent confirmation',
                  splits={})
    audits = []
    for variant in (reference, candidate):
        name = variant + '/initial_state_audit.json'
        verify(name)
        audits.append(json.loads((run_dir / name).read_text()))
    changed = [k for k in audits[0]['state_sha256']
               if audits[0]['state_sha256'][k] != audits[1]['state_sha256'][k]]
    if changed != ['cross_gate_logit']:
        raise ValueError('Unexpected initial parameter differences: ' + str(changed))
    output['initial_parameter_differences'] = changed
    for split in ('screen', 'test'):
        values, summaries, identities = [], [], []
        for variant in (reference, candidate):
            stem = f'{variant}_{split}_final'
            for suffix in ('_users.npz', '_metrics.json'):
                verify(stem + suffix)
            arrays, summary = load_metrics(run_dir, variant, split)
            values.append(arrays)
            summaries.append(summary)
            with np.load(run_dir / (stem + '_users.npz'), allow_pickle=False) as z:
                identities.append({k: z[k] for k in ('uid', 'position')})
        n = len(values[0]['hit5'])
        for key in ('uid', 'position'):
            if not np.array_equal(identities[0][key], identities[1][key]):
                raise ValueError('Pair identity mismatch: ' + key)
        if len(np.unique(identities[0]['uid'])) != n:
            raise ValueError('Repeated users require cluster bootstrap')
        if not np.array_equal(values[0]['pool_hit'], values[1]['pool_hit']):
            raise ValueError('Candidate pool hit mismatch')
        deltas = np.stack([values[1][k].astype(float) - values[0][k]
                           for k in ('hit5', 'ndcg5')], axis=1)
        rng = np.random.default_rng(20260922)
        draws = np.empty((10000, 2))
        for start in range(0, 10000, 100):
            indices = rng.integers(0, n, size=(100, n))
            draws[start:start + 100] = deltas[indices].mean(axis=1)
        result = dict(users=n, checks='archive hashes, unique aligned uid/record position, pool_hit, finite metrics and JSON means passed',
                      reference=summaries[0], candidate=summaries[1], metrics={})
        for j, key in enumerate(('hit5', 'ndcg5')):
            d = deltas[:, j]
            result['metrics'][key] = dict(delta=float(d.mean()),
                ci95=np.quantile(draws[:, j], [.025, .975]).tolist(),
                improved_users=int((d > 0).sum()), worsened_users=int((d < 0).sum()),
                unchanged_users=int((d == 0).sum()))
        wins, losses = int((deltas[:, 0] > 0).sum()), int((deltas[:, 0] < 0).sum())
        discordant = wins + losses
        result['hr5_mcnemar_exact_two_sided_p'] = min(1., 2 * sum(
            math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2 ** discordant)
        output['splits'][split] = result
    output['verified_source_sha256'] = hashes
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.run_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['splits'], indent=2))
