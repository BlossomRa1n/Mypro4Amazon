"""Read-only verification of a local diagnostic archive before a separate shutdown gate.

No SSH, deletion, shutdown, training, or bootstrap reruns. Checkpoint files must be
trusted local artifacts: torch.load is used on CPU. A receipt is written only after
all verification succeeds. --allow-smoke is for tiny validation fixtures only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


def strict_json(text):
    def reject(value): raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(text, parse_constant=reject)
    def finite(obj):
        if isinstance(obj, float): require(math.isfinite(obj), 'nonfinite JSON number')
        elif isinstance(obj, dict):
            for child in obj.values(): finite(child)
        elif isinstance(obj, list):
            for child in obj: finite(child)
    finite(value)
    return value


def read_json(path):
    return strict_json(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-8), 'metric mismatch: ' + label)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as handle: arrays = {key: handle[key].copy() for key in handle.files}
    fields = ('uid', 'position', 'candidate_ids', 'scores', 'labels', 'lengths', 'hit5', 'ndcg5', 'pool_hit', 'auc', 'auc_valid')
    require(all(key in arrays for key in fields), 'missing array fields: ' + str(path))
    for key in ('uid', 'position', 'candidate_ids', 'lengths'):
        require(np.issubdtype(arrays[key].dtype, np.integer), 'integer dtype required: ' + key)
    require(np.all(arrays['uid'] >= 2) and np.all(arrays['position'] >= 0), 'invalid user/position IDs')
    n = len(arrays['uid']); shape = arrays['candidate_ids'].shape
    require(n > 0 and len(set(arrays['uid'].tolist())) == n, 'empty/duplicate array users')
    require(len(shape) == 2 and shape[0] == n, 'candidate array shape mismatch')
    require(arrays['scores'].shape == arrays['labels'].shape == shape, 'candidate scores/labels shape mismatch')
    for key in fields:
        require(np.isfinite(arrays[key]).all(), 'nonfinite array: ' + key)
        if key not in ('candidate_ids', 'scores', 'labels'): require(arrays[key].shape == (n,), 'user array shape mismatch: ' + key)
    require(arrays['labels'].dtype == np.bool_ and arrays['auc_valid'].dtype == np.bool_, 'boolean validity/label mask required')
    for index, length in enumerate(arrays['lengths']):
        length = int(length); require(0 <= length <= shape[1], 'invalid candidate length')
        ids = arrays['candidate_ids'][index, :length]; labels = arrays['labels'][index, :length]; scores = arrays['scores'][index, :length]
        require(len(set(ids.tolist())) == length and np.all(ids >= 2), 'invalid candidate IDs')
        require(np.all(arrays['candidate_ids'][index, length:] == 0) and not arrays['labels'][index, length:].any() and np.all(arrays['scores'][index, length:] == 0), 'nonzero candidate padding')
        positive, negative = scores[labels], scores[~labels]; valid = bool(len(positive) and len(negative))
        require(bool(arrays['auc_valid'][index]) == valid, 'AUC validity mismatch')
        delta = positive[:, None] - negative[None, :]
        expected_auc = float(((delta > 0) + .5 * (delta == 0)).mean()) if valid else 0.
        close(arrays['auc'][index], expected_auc, 'per-user AUC')
        rank = sorted(range(length), key=lambda i: (-float(scores[i]), int(ids[i])))[:5]
        require(arrays['hit5'][index] == int(any(labels[i] for i in rank)), 'per-user hit mismatch')
        require(arrays['pool_hit'][index] == int(labels.any()), 'pool hit mismatch')
    require(np.all((arrays['ndcg5'] >= 0) & (arrays['ndcg5'] <= 1)), 'NDCG outside range')
    return arrays


def verify_metrics(arrays, metrics):
    for key, source in [('hr5', 'hit5'), ('ndcg5', 'ndcg5'), ('pool_hit', 'pool_hit')]: close(metrics[key], arrays[source].mean(), key)
    valid = arrays['auc_valid']
    require(metrics['users'] == len(valid) and metrics['auc_valid_users'] == int(valid.sum()), 'metric user count mismatch')
    close(metrics['auc_coverage'], valid.mean(), 'AUC coverage')
    close(metrics['gauc'], arrays['auc'][valid].mean() if valid.any() else 0., 'GAUC')


def verify(args):
    supplied_run = Path(args.run_dir)
    require(not supplied_run.is_symlink(), 'symlink archive root')
    run = supplied_run.resolve(); protocol_path = Path(args.protocol_manifest).resolve(); receipt_path = Path(args.receipt).resolve()
    for entry in run.rglob('*'):
        require(not entry.is_symlink() and (entry.is_file() or entry.is_dir()), 'symlink or special archive entry: ' + str(entry))
    require(not receipt_path.exists(), 'refusing to overwrite receipt')
    require(not receipt_path.is_relative_to(run), 'receipt must be outside immutable archive')
    protocol = read_json(protocol_path); complete = read_json(run/'COMPLETED.json')
    require(protocol['manifest_hash'] == canonical_hash({k:v for k,v in protocol.items() if k != 'manifest_hash'}), 'protocol manifest hash mismatch')
    require(complete.get('status') == 'complete' and complete.get('test_future_labels_read') is False, 'incomplete or test-exposed run')
    smoke = complete.get('smoke')
    require(type(smoke) is bool and smoke == protocol.get('smoke'), 'smoke identity mismatch')
    require(protocol.get('protocol') == 'raw-deterioration-20260924-v1' and protocol.get('test_evaluation_allowed') is False, 'protocol identity/test policy mismatch')
    test_identity = protocol['test_identity_receipt']
    require(test_identity.get('future_labels_read') is False and (smoke or test_identity.get('count') == 100000), 'sealed test identity mismatch')
    require(not smoke or bool(getattr(args, 'allow_smoke', False)), 'smoke archive cannot pass formal gate')
    require(complete['protocol_manifest_hash'] == protocol['manifest_hash'], 'completion protocol mismatch')
    require(not (run/'FAILED.json').exists(), 'failed marker present')
    hashes = complete.get('evidence_hashes'); require(isinstance(hashes, dict) and hashes, 'unsealed completion')
    files = {}
    for relative, expected in hashes.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'invalid sealed relative path')
        path = (run/relative).resolve()
        require(path.is_relative_to(run) and path.is_file(), 'missing or escaping evidence: ' + relative)
        expected_hash = expected['sha256'] if isinstance(expected, dict) else expected
        require(sha(path) == expected_hash, 'evidence hash mismatch: ' + relative)
        size = path.stat().st_size
        if isinstance(expected, dict) and 'size_bytes' in expected: require(size == expected['size_bytes'], 'evidence size mismatch: ' + relative)
        files[relative] = {'sha256': expected_hash, 'size_bytes': size}
        if path.suffix == '.json': read_json(path)
        elif path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                strict_json(line)
    require({str(path.relative_to(run)) for path in run.rglob('*') if path.is_file()} == set(hashes) | {'COMPLETED.json'}, 'archive contains unsealed files')
    def evidence(relative):
        require(relative in hashes, 'required evidence not sealed: ' + relative)
        return run/relative
    locked = read_json(evidence('CHECKPOINTS_LOCKED.json'))
    config = read_json(evidence('run_config.json')); panels = read_json(evidence('panels.json'))
    settings = protocol['settings']
    require(config['protocol_manifest_hash'] == protocol['manifest_hash'] and config['smoke'] == smoke and config['epochs'] == 3, 'run configuration binding mismatch')
    require(config['seeds'] == [42, 43, 44] and config['screen_checkpoints'] == [.25, .5, 1., 2., 3.], 'run schedule mismatch')
    for key in ('dim', 'token_dim', 'hist_len', 'batch_size'): require(config[key] == settings[key], 'run setting mismatch: ' + key)
    if not smoke: require(settings == dict(dim=256, token_dim=256, hist_len=50, batch_size=256, candidates=75), 'formal settings mismatch')
    require(panels['identity_sha256'] == canonical_hash({key:value for key,value in panels.items() if key != 'identity_sha256'}), 'panel identity hash mismatch')
    require(panels['seed'] == 20260924 and panels['smoke'] == smoke, 'panel seed/smoke mismatch')
    for split, negatives in [('train', 16), ('screen', 50)]:
        rows = panels[split]; require(bool(rows) and (smoke or len(rows) == 512), 'panel size mismatch')
        require(panels['split_hashes'][split]['identities_sha256'] == canonical_hash([row[:3] for row in rows]) and panels['split_hashes'][split]['negatives_sha256'] == canonical_hash([row[3] for row in rows]), 'panel split hash mismatch')
        require(len({(row[0], row[1]) for row in rows}) == len(rows), 'duplicate panel identity')
        for uid, position, positive, negative in rows:
            require(all(type(value) is int for value in [uid, position, positive, *negative]) and uid >= 2 and position >= 0 and positive >= 2, 'invalid panel IDs')
            require(len(negative) == negatives if not smoke or split == 'train' else 0 < len(negative) <= negatives, 'panel negative count mismatch')
            require(len(set(negative)) == len(negative) and positive not in negative and all(value >= 2 for value in negative), 'invalid panel negatives')
    screen_reference = None
    model_reference = None
    differences = {'hit5': [], 'ndcg5': []}; confirm_reference = None; counts = dict(checkpoints=0, screen_arrays=0, confirm_arrays=0)
    for seed in ('42', '43', '44'):
        prefix = f'seed_{seed}/raw/'; history = read_json(evidence(prefix+'history.json'))
        completed_seed = read_json(evidence(prefix+'COMPLETED.json'))
        require(completed_seed['seed'] == int(seed) and completed_seed['init_seed'] == 424242 + int(seed) - 42 and completed_seed['precision'] == 'FP32' and completed_seed['variant'] == 'raw', 'seed training identity mismatch')
        require(completed_seed['protocol_manifest_hash'] == protocol['manifest_hash'], 'seed protocol mismatch')
        require(completed_seed == complete['seeds'][seed], 'seed completion mismatch')
        points = [point for entry in history for point in entry['requested_points']]
        require(points == [.25, .5, 1., 2., 3.], 'five requested screening points missing')
        steps_per_epoch = completed_seed['steps_per_epoch']; require(type(steps_per_epoch) is int and steps_per_epoch > 0, 'invalid epoch size')
        if not smoke: require(steps_per_epoch == 2362, 'formal steps per epoch mismatch')
        model_config = completed_seed['model_config']
        expected_config = dict(embed_dim=settings['dim'], brand_embed_dim=min(64, settings['dim']), hidden_dims=[256, 128, 64] if settings['dim'] >= 128 else [64, 32], hist_len=settings['hist_len'], dropout=.1, token_dim=settings['token_dim'], fusion='concat', cross_mode='raw')
        require(all(model_config.get(key) == value for key, value in expected_config.items()), 'model configuration mismatch')
        require(all(type(model_config.get(key)) is int and model_config[key] >= 2 for key in ('num_users', 'num_items', 'num_brands', 'num_categories')), 'model cardinality mismatch')
        if model_reference is None: model_reference = model_config
        else: require(model_config == model_reference, 'cross-seed model configuration mismatch')
        require([entry['step'] for entry in history] == sorted(set(math.ceil(point * steps_per_epoch) for point in points)), 'screen step schedule mismatch')
        if not smoke: require(len(history) == 5, 'formal run requires five distinct screen arrays')
        best_entry = max(history, key=lambda entry: (entry['screen']['hr5'], entry['screen']['ndcg5'], -entry['step']))
        audit = read_json(evidence(prefix+'initial_state_audit.json'))
        require(audit['gate_difference_whitelist'] == ['cross_gate_logit'] and audit['gate_value'] == 10., 'initial gate audit mismatch')
        require(set(audit['copied_tensor_keys']) == set(audit['state_sha256']) == set(audit['template_sha256']), 'initial audit keys mismatch')
        for key in audit['copied_tensor_keys']:
            if key != 'cross_gate_logit': require(audit['state_sha256'][key] == audit['template_sha256'][key], 'initial copied tensor mismatch')
        steps_path = evidence(prefix+'steps.jsonl'); step_rows = [strict_json(line) for line in steps_path.read_text().splitlines()]
        require([row['step'] for row in step_rows] == list(range(1, 3*steps_per_epoch+1)), 'training steps missing')
        for epoch in (1, 2, 3):
            trace = read_json(evidence(prefix+f'epoch_{epoch}_trace.json'))
            require(trace == completed_seed['traces'][str(epoch)], 'trace completion mismatch')
            require(all(type(size) is int and size > 0 for size in trace['batch_sizes']), 'invalid trace batch sizes')
            epoch_steps = step_rows[(epoch-1)*steps_per_epoch:epoch*steps_per_epoch]
            require([row['batch_rows'] for row in epoch_steps] == trace['batch_sizes'] and all(row['epoch'] == epoch for row in epoch_steps), 'step trace batch/epoch mismatch')
            if not smoke: require(trace['rows'] == 604511 and trace['covered_users'] == 100000, 'formal trace coverage mismatch')
            require(len(trace['batch_sizes']) == steps_per_epoch and sum(trace['batch_sizes']) == trace['rows'], 'trace row/batch mismatch')
            require(sum(trace['negative_sources'].values()) == trace['rows']*16, 'negative stream count mismatch')
            for key in ('sha256', 'order_sha256'): require(len(trace[key]) == 64 and all(char in '0123456789abcdef' for char in trace[key]), 'invalid stream digest')
        for entry in history:
            close(entry['epoch_fraction'], entry['step']/steps_per_epoch, 'history epoch fraction')
            diagnostics = entry['diagnostics']; invariance = diagnostics['candidate_batch_invariance']
            require(invariance['passed'] is True and invariance['atol'] == 1e-5 and invariance['rtol'] == 1e-5 and invariance['max_abs_difference'] >= 0, 'candidate batch invariance missing/failed')
            require(all(key in diagnostics for key in ('train', 'screen', 'user_embedding_norm', 'item_embedding_norm', 'batchnorm', 'fixed_train_positive_norms', 'cross_type_offset_norm', 'parameter_l2', 'gradient_l2_after_clip')), 'missing diagnostic summaries')
            require(all(key in diagnostics['fixed_train_positive_norms'] for key in ('user', 'item', 'raw_product', 'cross_token')) and bool(diagnostics['batchnorm']), 'missing token/BN summaries')
            for split in ('train', 'screen'): require(all(key in diagnostics[split] for key in ('pair_auc', 'bpr_loss', 'valid_users', 'users', 'margin_quantiles')), 'missing panel summary')
            arrays = load_arrays(evidence(prefix+f"screen_step{entry['step']}_users.npz")); verify_metrics(arrays, entry['screen']); counts['screen_arrays'] += 1
            require(arrays['candidate_ids'].shape[1] <= settings['candidates'] and (smoke or len(arrays['uid']) == 20000), 'screen population/budget mismatch')
            require(np.all(arrays['candidate_ids'] < model_config['num_items']) and np.all(arrays['uid'] < model_config['num_users']), 'screen catalog/user range mismatch')
            stable = ('uid', 'position', 'candidate_ids', 'lengths', 'labels', 'pool_hit')
            if screen_reference is None: screen_reference = {key:arrays[key].copy() for key in stable}
            else: require(all(np.array_equal(arrays[key], screen_reference[key]) for key in stable), 'unstable screen identity/candidate pool')
        confirm = {}
        for kind in ('best', 'last'):
            checkpoint = evidence(prefix+kind+'.pth'); metadata = locked[seed][kind]
            require(metadata['init_seed'] == completed_seed['init_seed'] and metadata['precision'] == 'FP32' and metadata['model_config'] == model_config, 'checkpoint training configuration mismatch')
            selected_entry = best_entry if kind == 'best' else history[-1]
            require(metadata['screen'] == selected_entry['screen'] and metadata['epoch_fraction'] == selected_entry['epoch_fraction'], 'checkpoint screening binding mismatch')
            require(metadata == completed_seed[kind], 'locked checkpoint metadata mismatch')
            require(metadata['protocol_manifest_hash'] == protocol['manifest_hash'] and metadata['seed'] == int(seed) and metadata['variant'] == 'raw', 'checkpoint protocol/seed/variant mismatch')
            require(sha(checkpoint) == metadata['checkpoint_sha256'], 'locked checkpoint hash mismatch')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(payload['manifest'] == {k:v for k,v in metadata.items() if k != 'checkpoint_sha256'}, 'checkpoint manifest mismatch')
            require(all(torch.is_tensor(tensor) and torch.isfinite(tensor).all().item() for tensor in payload['model'].values()), 'nonfinite checkpoint')
            require(set(payload['model']) == set(audit['copied_tensor_keys']), 'checkpoint parameter set mismatch')
            require(metadata['step'] == (best_entry['step'] if kind == 'best' else 3*steps_per_epoch), 'best/last selection mismatch')
            del payload; counts['checkpoints'] += 1
            arrays = load_arrays(evidence(prefix+f'confirm_{kind}_users.npz')); verify_metrics(arrays, read_json(evidence(prefix+f'confirm_{kind}_metrics.json')))
            require(arrays['candidate_ids'].shape[1] <= settings['candidates'] and (smoke or len(arrays['uid']) == 80000), 'confirm population/budget mismatch')
            require(np.all(arrays['candidate_ids'] < model_config['num_items']) and np.all(arrays['uid'] < model_config['num_users']), 'confirm catalog/user range mismatch')
            counts['confirm_arrays'] += 1; confirm[kind] = arrays
            stable = ('uid', 'position', 'candidate_ids', 'lengths', 'labels', 'pool_hit')
            if confirm_reference is None: confirm_reference = {key: arrays[key].copy() for key in stable}
            else: require(all(np.array_equal(arrays[key], confirm_reference[key]) for key in stable), 'unpaired confirm identities/candidates')
        for metric in differences: differences[metric].append(confirm['best'][metric].astype(np.float64)-confirm['last'][metric].astype(np.float64))
    paired = read_json(evidence('confirm_best_last.json')); require(paired == complete['confirm'], 'confirm summary mismatch')
    require(paired['bootstrap_seed'] == 20260924 and paired['bootstrap_replicates'] == 10000, 'bootstrap registration mismatch')
    for metric, rows in differences.items():
        average = np.mean(rows, axis=0); reported = paired[metric]
        close(reported['mean_delta'], average.mean(), metric+' seed-average delta')
        require(reported['users'] == len(average) and reported['gained'] == int((average > 0).sum()) and reported['lost'] == int((average < 0).sum()), 'aggregate gained/lost mismatch')
        ci = reported['ci95']; require(len(ci) == 2 and all(math.isfinite(x) for x in ci) and -1 <= ci[0] <= ci[1] <= 1, 'invalid recorded CI')
        for seed, delta in zip(('42','43','44'), rows):
            row = paired['per_seed'][seed]; close(row[metric+'_delta'], delta.mean(), metric+' per-seed delta')
            if metric == 'hit5': require(row['gained'] == int((delta > 0).sum()) and row['lost'] == int((delta < 0).sum()), 'per-seed gained/lost mismatch')
    criteria = dict(all_three_seed_hr_positive=all(float(row.mean())>0 for row in differences['hit5']), mean_hr_at_least_0_0005=paired['hit5']['mean_delta']>=.0005, hr_ci95_lower_positive=paired['hit5']['ci95'][0]>0, mean_ndcg_nonnegative=paired['ndcg5']['mean_delta']>=0)
    require(paired['followup_criteria'] == criteria and paired['supports_early_stopping_followup'] == all(criteria.values()), 'followup report mismatch')
    require(counts['checkpoints'] == 6 and counts['confirm_arrays'] == 6 and (smoke or counts['screen_arrays'] == 15), 'archive counts mismatch')
    receipt = dict(status='verified', formal_archive_verified=not smoke, smoke=smoke, run_dir=str(run), protocol_manifest_sha256=sha(protocol_path), protocol_manifest_hash=protocol['manifest_hash'], completion_sha256=sha(run/'COMPLETED.json'), counts=counts, evidence=files,
                   limitations=['Source completion seals SHA-256; sizes independently recorded here (checked against source sizes only if provided).', 'No bootstrap rerun: CI values are finite/range checked and matched to sealed report; paired deltas independently recomputed.', 'NDCG per-user values are finite/range checked and aggregates recomputed; full future target denominators are not reopened.', 'Stream hashes and initial audits are checked for consistency, not replayed; external protocol/code archive binding and original cohort UID/order authentication remain separate gates.', 'No remote transfer, shutdown, deletion, or test evaluation is performed.'])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('x') as stream: json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--protocol-manifest', required=True); parser.add_argument('--receipt', required=True); parser.add_argument('--allow-smoke', action='store_true')
    result = verify(parser.parse_args(argv)); print(json.dumps({'status': result['status'], 'formal_archive_verified': result['formal_archive_verified'], 'counts': result['counts']}))

if __name__ == '__main__': main()
