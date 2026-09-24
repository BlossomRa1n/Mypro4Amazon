"""Raw-only three-seed development diagnostics. No optimizer resume or final test.

The optimization block and stream tracing mirror run_cross_multiseed._train_model.
Screen selects best; confirm is already-development evidence and never selects.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import math
import shutil
import platform
import sys
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import run_cross_multiseed as r
from baseline_data import collate
from baseline_runtime import seed_all
from diagnostic_metrics import preserve_training_state, evaluate_candidates, score_candidates, pair_auc, parameter_summary, quantiles

PANEL_SEED = 20260924
POINTS = (.25, .5, 1., 2., 3.)
USER_KEYS = ('user_id', 'hist_items', 'hist_brands', 'hist_ratings', 'hist_time_deltas', 'hist_verified', 'hist_len', 'click_count', 'time_span', 'user_avg_rating', 'user_std_rating', 'user_verified_ratio', 'user_avg_helpful')
ITEM_KEYS = ('item_id', 'category_id', 'brand_id', 'item_click_count', 'created_at_ts', 'item_avg_rating', 'item_rating_number')


def selection_key(metrics, step):
    return float(metrics['hr5']), float(metrics['ndcg5']), -int(step)


def checked_save(model, manifest, path):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    for key, value in state.items():
        if not torch.isfinite(value).all(): raise FloatingPointError(f'nonfinite checkpoint tensor {key}')
    digest = r.atomic_torch_save({'model': state, 'manifest': manifest}, path)
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    if any(not torch.equal(value, loaded['model'][key]) for key, value in state.items()): raise ValueError('checkpoint reload mismatch')
    return digest


def guard_targets(data, records):
    """Whitelist exact development records, never infer an allowed test cohort."""
    allowed = {(int(row[0]), int(row[1])) for split in ('train', 'screen', 'confirm') for row in records[split]}
    original = data.targets
    def guarded(rows):
        rows = list(rows)
        if any((int(row[0]), int(row[1])) not in allowed for row in rows): raise PermissionError('target outside train/screen/confirm whitelist')
        return original(rows)
    data.targets = guarded
    return original


def make_panels(data, positions, train_pools, screen_records, smoke):
    rng = np.random.default_rng(PANEL_SEED)
    indexes = np.sort(rng.choice(len(positions), min(512, len(positions)), replace=False))
    view = copy.copy(data); view.train_positions = np.asarray(positions); view.seed = PANEL_SEED
    dataset = r.TracedPrefixDataset(view, negatives=16, negative_policy='mixed_rrf', candidate_pools=train_pools, candidate_ranges=((11, 25, 2), (26, 50, 2)))
    train = []
    for index in indexes:
        row = dataset[int(index)]; position = int(positions[index]); uid = int(data.uid[position])
        train.append((uid, position, int(data.iid[position]), list(map(int, row['neg_item_id']))))
    screen = []
    indexes = np.sort(rng.choice(len(screen_records), min(512, len(screen_records)), replace=False))
    for index in indexes:
        uid, position, _ = screen_records[int(index)]; uid, position = int(uid), int(position)
        targets = set(map(int, data.targets([(uid, position)])[0]))
        if not targets: raise ValueError('screen panel record without a future positive')
        observed = set(map(int, data.iid[int(data.starts[uid]):position]))
        eligible = np.asarray(sorted(set(map(int, data.active_items)) - observed - targets - {0, 1}), dtype=np.int64)
        if len(eligible) < 50 and not smoke: raise ValueError('screen panel needs 50 eligible negatives')
        negatives = list(map(int, rng.choice(eligible, min(50, len(eligible)), replace=False)))
        positive = int(data.iid[position])
        if positive not in targets: raise ValueError("first future item is not in authorized targets")
        screen.append((uid, position, positive, negatives))
    return {'train': train, 'screen': screen, 'seed': PANEL_SEED, 'screen_positive_policy': 'first interaction at authorized future-window boundary', 'screen_negative_policy': '50 train-active random items excluding complete observed prefix and all known future targets; evaluation only', 'train_negative_policy': 'fixed epoch0 mixed16 sampler with independent seed; no future exclusion', 'smoke': bool(smoke)}


def panel_diagnostics(data, model, panels, screen_records, screen_pools):
    result = {}
    with preserve_training_state(model), torch.inference_mode():
        for split in ('train', 'screen'):
            by_user = {}; margins = []
            for uid, position, positive, negatives in panels[split]:
                scores = score_candidates(data, model, uid, position, [positive] + negatives)
                delta = scores[0] - scores[1:]
                auc, valid = pair_auc(scores, [True] + [False] * len(negatives))
                loss = float(np.logaddexp(0, -delta).mean()) if len(delta) else 0.0
                margins.extend(delta.tolist()); by_user.setdefault(uid, []).append((auc, loss, valid))
            valid_users = [np.mean([(a, l) for a, l, v in rows if v], axis=0) for rows in by_user.values() if any(v for _, _, v in rows)]
            avg = np.mean(valid_users, axis=0) if valid_users else [0., 0.]
            result[split] = {'pair_auc': float(avg[0]), 'bpr_loss': float(avg[1]), 'valid_users': len(valid_users), 'users': len(by_user), 'margin_quantiles': quantiles(margins), 'aggregation': 'row means within user, then equal user means'}
        differences = []
        for (uid, position, _), pool in list(zip(screen_records[:8], screen_pools[:8])):
            ids = list(map(int, pool))
            if ids:
                alone = score_candidates(data, model, uid, position, ids[:1])[0]
                together = score_candidates(data, model, uid, position, ids)[0]
                difference = float(abs(alone - together))
                if difference > 1e-5 + 1e-5 * abs(float(together)): raise AssertionError('candidate batch invariance failed')
                differences.append(difference)
        result['candidate_batch_invariance'] = {'max_abs_difference': max(differences, default=0.), 'users': len(differences), 'comparison': 'same first candidate alone versus existing full candidate pool', 'atol': 1e-5, 'rtol': 1e-5, 'passed': True}
        norms = {key: [] for key in ('user', 'item', 'raw_product', 'cross_token')}
        device = next(model.parameters()).device
        for uid, position, positive, _ in panels['train']:
            user = {key: value.to(device) for key, value in collate([data.user_features(uid, position)]).items()}
            encoded = model._encode_user(user)
            items = {key: torch.as_tensor(value, device=device) for key, value in data.item_features([positive]).items()}
            u = encoded['user_emb']; v = model.item_embedding(items['item_id'])
            for key, value in [('user', u), ('item', v), ('raw_product', u * v), ('cross_token', model._tokenize(encoded, items)[:, 4])]:
                norms[key].append(float(value.norm(dim=-1).mean()))
        result['fixed_train_positive_norms'] = {key: quantiles(values) for key, values in norms.items()}
        result['cross_type_offset_norm'] = float(model.token_type[4].norm())
        result.update(parameter_summary(model))
    return result


def train_seed(data, args, directory, seed, state, positions, pools, records, panels, manifest):
    directory.mkdir(parents=True)
    model = r._new_variant(data, args, state, 'raw')
    audit = {'state_sha256': r._state_hashes(model), 'template_sha256': {key: r.sha256_array(value.numpy()) for key, value in state.items()}, 'copied_tensor_keys': sorted(state), 'gate_difference_whitelist': ['cross_gate_logit'], 'gate_value': float(model.cross_gate_logit.detach())}
    for key, value in model.state_dict().items():
        if key != 'cross_gate_logit' and not torch.equal(value.cpu(), state[key]): raise AssertionError('initial tensor copy mismatch: ' + key)
    r.write_json(directory / 'initial_state_audit.json', audit)
    device = torch.device(args.device); model.to(device)
    view = copy.copy(data); view.train_positions = np.asarray(positions, dtype=np.int64); view.seed = seed
    dataset = r.TracedPrefixDataset(view, negatives=16, negative_policy='mixed_rrf', candidate_pools=pools['train'], candidate_ranges=((11, 25, 2), (26, 50, 2)))
    seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
    steps_per_epoch = sum(1 for _ in r._batch_chunks(np.arange(len(dataset)), args.batch_size))
    if not steps_per_epoch: raise ValueError('empty training stream')
    checks = {}
    for point in POINTS: checks.setdefault(math.ceil(point * steps_per_epoch), []).append(point)
    history = []; step = 0; best_key = None; best_meta = None; train_seconds = 0.; eval_seconds = 0.; traces = {}
    base_meta = {'variant': 'raw', 'seed': seed, 'init_seed': r.INIT_SEEDS[seed], 'model_config': r._config(data, args, 'raw'), 'protocol_manifest_hash': manifest['manifest_hash'], 'automatic_resume': False, 'optimizer_saved': False, 'precision': 'FP32'}
    for epoch in range(3):
        model.train(); dataset.epoch = epoch
        order = np.random.default_rng(seed + epoch).permutation(len(dataset)).astype(np.int64)
        digest = hashlib.sha256(); source_counts = dict(band11_25=0, band26_50=0, candidate_fallback=0, random=0); consumed_uids = set(); sizes = []; losses = []
        for indexes in r._batch_chunks(order, args.batch_size):
            started = time.monotonic(); step += 1; sizes.append(len(indexes)); rows = [dataset[int(index)] for index in indexes]
            for index, row in zip(indexes, rows):
                position = int(view.train_positions[int(index)]); uid = int(view.uid[position]); consumed_uids.add(uid)
                digest.update(np.asarray([int(index), position, uid], dtype=np.int64).tobytes()); digest.update(np.asarray(row['neg_item_id'], dtype=np.int64).tobytes())
                for key, count in dataset.last_sources.pop(int(index)).items(): source_counts[key] += count
            batch = {key: value.to(device) for key, value in collate(rows).items()}
            optimizer.zero_grad(set_to_none=True)
            users = {key: batch[key] for key in USER_KEYS}; positive = {key: batch['pos_' + key] for key in ITEM_KEYS}; negative = {key: batch['neg_' + key] for key in positive}
            pos_score, neg_score = model.forward_bpr(users, positive, negative)
            loss = F.softplus(neg_score.float() - pos_score.float()[:, None]).mean()
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite raw loss seed {seed}')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach())); train_seconds += time.monotonic() - started
            with (directory / 'steps.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps({'step': step, 'epoch': epoch + 1, 'loss': losses[-1], 'lr': optimizer.param_groups[0]['lr'], 'gradient_l2_before_clip': float(grad), 'batch_rows': len(indexes)}) + '\n')
            if step in checks:
                started = time.monotonic()
                with preserve_training_state(model):
                    metrics, arrays = evaluate_candidates(data, model, records['screen'], pools['screen'])
                    np.savez_compressed(directory / f'screen_step{step}_users.npz', **arrays)
                    diagnostics = panel_diagnostics(data, model, panels, records['screen'], pools['screen'])
                    entry = {'step': step, 'epoch_fraction': step / steps_per_epoch, 'requested_points': checks[step], 'screen': metrics, 'diagnostics': diagnostics, 'gradient_l2_before_clip': float(grad), 'train_seconds_cumulative': train_seconds}
                    meta = dict(base_meta, step=step, epoch_fraction=step / steps_per_epoch, screen=metrics)
                    if best_key is None or selection_key(metrics, step) > best_key:
                        best_key = selection_key(metrics, step); best_meta = dict(meta)
                        best_meta['checkpoint_sha256'] = checked_save(model, meta, directory / 'best.pth')
                    if step == 3 * steps_per_epoch:
                        last_meta = dict(meta); last_meta['checkpoint_sha256'] = checked_save(model, meta, directory / 'last.pth')
                eval_seconds += time.monotonic() - started; entry['eval_seconds_cumulative'] = eval_seconds; history.append(entry); r.write_json(directory / 'history.json', history)
        expected = set(np.unique(np.asarray(data.uid)[positions]).astype(int).tolist())
        if consumed_uids != expected: raise AssertionError('training user coverage mismatch')
        trace = dict(sha256=digest.hexdigest(), rows=len(order), order_sha256=r.sha256_array(order), batch_sizes=sizes, negative_sources=source_counts, covered_users=len(consumed_uids))
        traces[str(epoch + 1)] = trace; r.write_json(directory / f'epoch_{epoch + 1}_trace.json', trace)
        scheduler.step()
    result = dict(base_meta, best=best_meta, last=last_meta, steps_per_epoch=steps_per_epoch, traces=traces, train_seconds=train_seconds, eval_seconds=eval_seconds)
    r.write_json(directory / 'COMPLETED.json', result)
    return result


def paired_confirmation(run, records, results, smoke):
    deltas = {'hit5': [], 'ndcg5': []}; per_seed = {}
    expected_uid = np.asarray([row[0] for row in records['confirm']], dtype=np.int64)
    expected_position = np.asarray([row[1] for row in records['confirm']], dtype=np.int64)
    if not len(expected_uid) or len(set(expected_uid.tolist())) != len(expected_uid): raise ValueError('confirm requires unique nonempty users')
    reference = None
    for seed in r.SEEDS:
        base = run / f'seed_{seed}' / 'raw'
        with np.load(base / 'confirm_best_users.npz') as best, np.load(base / 'confirm_last_users.npz') as last:
            if not np.array_equal(best['uid'], last['uid']) or not np.array_equal(best['position'], last['position']): raise AssertionError('unpaired confirm identities')
            for arrays in (best, last):
                if not np.array_equal(arrays['uid'], expected_uid) or not np.array_equal(arrays['position'], expected_position): raise ValueError('confirm differs from locked record order')
            stable = ('candidate_ids', 'lengths', 'labels', 'pool_hit')
            if any(not np.array_equal(best[key], last[key]) for key in stable): raise ValueError('best/last candidate identity changed')
            if reference is None: reference = {key: best[key].copy() for key in stable}
            elif any(not np.array_equal(best[key], reference[key]) for key in stable): raise ValueError('candidate identity differs across seeds')
            row = {}
            for metric in deltas:
                delta = best[metric].astype(np.float64) - last[metric].astype(np.float64); deltas[metric].append(delta); row[metric + '_delta'] = float(delta.mean())
                if metric == 'hit5': row.update(gained=int((delta > 0).sum()), lost=int((delta < 0).sum()))
            per_seed[str(seed)] = row
    output = {'label': 'already-development confirm; best minus last, paired user bootstrap after averaging seeds', 'per_seed': per_seed, 'bootstrap_seed': PANEL_SEED, 'bootstrap_replicates': 10000}
    for metric, rows in deltas.items():
        delta = np.mean(rows, axis=0); rng = np.random.default_rng(PANEL_SEED); samples = np.empty(10000)
        # Bounded memory, exact paired nonparametric user resampling.
        for index in range(10000): samples[index] = delta[rng.integers(0, len(delta), size=len(delta))].mean()
        output[metric] = {'mean_delta': float(delta.mean()), 'ci95': list(map(float, np.quantile(samples, [.025, .975]))), 'users': len(delta), 'gained': int((delta > 0).sum()), 'lost': int((delta < 0).sum())}
    criteria = {'all_three_seed_hr_positive': all(row['hit5_delta'] > 0 for row in per_seed.values()),
                'mean_hr_at_least_0_0005': output['hit5']['mean_delta'] >= .0005,
                'hr_ci95_lower_positive': output['hit5']['ci95'][0] > 0,
                'mean_ndcg_nonnegative': output['ndcg5']['mean_delta'] >= 0}
    output['supports_early_stopping_followup'] = all(criteria.values())
    output['followup_criteria'] = criteria
    output['failed_followup_criteria'] = [key for key, passed in criteria.items() if not passed]
    output['decision_scope'] = 'diagnostic report only; does not authorize full training, extra experiments, cross winner selection, or final test'
    return output


def run(args):
    import diagnostic_protocol
    path = Path(args.protocol_manifest).resolve(); directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()): raise FileExistsError('refusing to overwrite diagnostic run; no automatic resume')
    try:
        manifest, data, records, positions, pools = diagnostic_protocol.load_inputs(path)
        smoke = bool(manifest.get('smoke'))
        if bool(getattr(args, 'smoke', False)) != smoke: raise ValueError('smoke flag must match protocol')
        settings = manifest.get('settings', {})
        config = SimpleNamespace(dim=int(settings.get('dim', 16 if smoke else 256)), token_dim=int(settings.get('token_dim', 16 if smoke else 256)), hist_len=int(settings.get('hist_len', getattr(data, 'hist_len', 50) if smoke else 50)), batch_size=int(settings.get('batch_size', 8 if smoke else 256)), epochs=3, smoke=smoke)
        if not smoke and (config.dim, config.token_dim, config.hist_len, config.batch_size) != (256, 256, 50, 256): raise ValueError('formal settings differ from registered configuration')
        requested_device = getattr(args, 'device', 'auto')
        config.device = ('cpu' if smoke else 'cuda') if requested_device == 'auto' else requested_device
        if config.device not in ('cpu', 'cuda'): raise ValueError('device must be auto, cpu or cuda')
        if not smoke and config.device != 'cuda': raise ValueError('formal diagnostic training requires CUDA')
        if config.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA required but unavailable')
        torch.set_num_threads(4)
        original_targets = guard_targets(data, records)
        try:
            with preserve_training_state(): panels = make_panels(data, positions, pools['train'], records['screen'], smoke)
            panels['split_hashes'] = {split: {'identities_sha256': r.sha256_json([row[:3] for row in panels[split]]), 'negatives_sha256': r.sha256_json([row[3] for row in panels[split]])} for split in ('train', 'screen')}
            panels['identity_sha256'] = r.sha256_json(panels); r.write_json(directory / 'panels.json', panels)
            factors, factor_receipt = r._load_factors(Path(manifest['base_run']), data, config.dim, smoke)
            environment = {'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
                           'numpy': np.__version__, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                           'cudnn': torch.backends.cudnn.version(), 'torch_threads': torch.get_num_threads(),
                           'torch_interop_threads': torch.get_num_interop_threads(),
                           'gpu': [{'index': index, 'name': torch.cuda.get_device_name(index), 'total_memory': torch.cuda.get_device_properties(index).total_memory} for index in range(torch.cuda.device_count())]}
            r.write_json(directory / 'run_config.json', dict(vars(config), environment=environment, protocol_manifest_hash=manifest['manifest_hash'], seeds=list(r.SEEDS), screen_checkpoints=list(POINTS), factors=factor_receipt, train_positions_sha256=r.sha256_array(np.asarray(positions)), automatic_resume=False))
            results = {}
            for seed in r.SEEDS:
                template, state = r._template(data, config, r.INIT_SEEDS[seed], factors); del template
                if seed == r.SEEDS[0]:
                    tensor_bytes = sum(value.numel() * value.element_size() for value in state.values())
                    candidates = int(settings.get('candidates', 75))
                    score_bytes = (15 * len(records['screen']) + 6 * len(records['confirm'])) * (candidates * 9 + 64)
                    required = 7 * tensor_bytes + score_bytes + (0 if smoke else 1024 ** 3)
                    free = shutil.disk_usage(directory).free
                    r.write_json(directory / 'disk_preflight.json', {'checkpoint_tensor_bytes': tensor_bytes, 'retained_checkpoints': 6, 'atomic_peak_checkpoints': 7, 'score_arrays_estimate_bytes': score_bytes, 'required_bytes': required, 'free_bytes': free})
                    if free < required: raise OSError('insufficient diagnostic checkpoint and score-array disk space')
                results[str(seed)] = train_seed(data, config, directory / f'seed_{seed}' / 'raw', seed, state, positions, pools, records, panels, manifest)
                del state
            locked = {str(seed): {kind: results[str(seed)][kind] for kind in ('best', 'last')} for seed in r.SEEDS}
            r.write_json(directory / 'CHECKPOINTS_LOCKED.json', locked)
            for seed in r.SEEDS:
                base = directory / f'seed_{seed}' / 'raw'
                for kind in ('best', 'last'):
                    if r.sha256_file(base / f'{kind}.pth') != locked[str(seed)][kind]['checkpoint_sha256']: raise ValueError('locked checkpoint changed')
                    payload = torch.load(base / f'{kind}.pth', map_location='cpu', weights_only=False)
                    with preserve_training_state():
                        model = r.SemanticTokenDIN(**payload['manifest']['model_config']); model.load_state_dict(payload['model']); del payload
                        model.to(torch.device(config.device))
                        metrics, arrays = evaluate_candidates(data, model, records['confirm'], pools['confirm'])
                        np.savez_compressed(base / f'confirm_{kind}_users.npz', **arrays); r.write_json(base / f'confirm_{kind}_metrics.json', metrics); del model
            paired = paired_confirmation(directory, records, results, smoke); r.write_json(directory / 'confirm_best_last.json', paired)
            complete = {'status': 'complete', 'smoke': smoke, 'protocol_manifest_hash': manifest['manifest_hash'], 'test_future_labels_read': False, 'seeds': results, 'confirm': paired}
            complete['evidence_hashes'] = {str(p.relative_to(directory)): r.sha256_file(p) for p in sorted(directory.rglob('*')) if p.is_file()}
            r.write_json(directory / 'COMPLETED.json', complete)
            return complete
        finally: data.targets = original_targets
    except BaseException as error:
        r.write_json(directory / 'FAILED.json', {'status': 'failed', 'type': type(error).__name__, 'message': str(error), 'automatic_resume': False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-manifest', required=True); parser.add_argument('--run-dir', required=True); parser.add_argument('--smoke', action='store_true'); parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return run(parser.parse_args(argv))

if __name__ == '__main__': main()
