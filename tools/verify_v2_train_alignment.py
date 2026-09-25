#!/usr/bin/env python3
"""Independent read-only V2 training-pool and early-stop archive verification.

Never trains, regenerates candidates, or loads a future-target dataset. Local
paths explicitly map SHA-bound remote references. A receipt is created only
after all checks pass and must remain outside all immutable input trees.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

VERSION = 'afternoon-v2-train-pool-20260925-v1'
DOC_SHA = 'e0b5b95aaf2d9b39f55849c41c1733906af1eb8e4ba3de14973f0859c150ac05'
FORMAL_PROTOCOL = 'c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6'
FORMAL_OLD_SHA = '4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1'
FORMAL_ROWS = 604511
METRIC_HELPER_SHA = '6c3b9ecaec8d7494d74451020899c1a2ee6fefede539f59c6b33b488be323b79'
SEEDS = ('42', '43', '44')
STABLE = ('uid', 'position', 'candidate_ids', 'lengths', 'labels', 'pool_hit', 'auc_valid')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'nonfinite JSON number')
        elif isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
    finite(value)
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-8),
            'metric mismatch: ' + label)


def safe_path(path):
    path = Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe/symlink path')
    return path


def safe_root(path):
    root = safe_path(path)
    require(root.is_dir(), 'missing archive root')
    for entry in root.rglob('*'):
        require(not entry.is_symlink() and (entry.is_file() or entry.is_dir()), 'symlink/special archive entry')
        require(entry.name != 'FAILED.json', 'failed marker present')
    return root


def safe_file(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts and str(rel) == relative, 'unsafe evidence path')
    path = safe_path(Path(root) / rel)
    require(path.is_file() and path.is_relative_to(Path(root)), 'missing/escaping evidence: ' + relative)
    return path


def bound_file(reference, local_path, *, expected_name=None):
    path = safe_path(local_path)
    require(path.is_file() and isinstance(reference, dict), 'missing referenced file')
    remote = Path(reference.get('path', ''))
    require(remote.is_absolute() and '..' not in remote.parts, 'unsafe source reference')
    if expected_name is not None:
        require(remote.name == expected_name and path.name == expected_name, 'source basename mismatch')
    require(digest_ok(reference.get('sha256')) and sha(path) == reference['sha256'], 'source SHA mismatch: ' + path.name)
    require(reference.get('bytes', reference.get('size_bytes')) == path.stat().st_size, 'source byte count mismatch: ' + path.name)
    return path


def sealed_tree(root, complete):
    hashes = complete.get('evidence_hashes')
    require(isinstance(hashes, dict) and hashes, 'empty evidence seal')
    files = {}
    for rel, bound in hashes.items():
        path = safe_file(root, rel)
        expected = bound['sha256'] if isinstance(bound, dict) else bound
        require(digest_ok(expected) and sha(path) == expected, 'evidence hash mismatch: ' + rel)
        files[rel] = {'sha256': expected, 'bytes': path.stat().st_size}
        if isinstance(bound, dict) and ('bytes' in bound or 'size_bytes' in bound):
            require(bound.get('bytes', bound.get('size_bytes')) == path.stat().st_size, 'evidence size mismatch')
        if path.suffix == '.json': read_json(path)
        elif path.suffix == '.jsonl':
            # The same strict parser is used on individual lines below where consumed.
            for line in path.read_text().splitlines():
                require(isinstance(json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSONL'))), dict), 'invalid JSONL row')
    require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == set(hashes) | {'COMPLETED.json'},
            'unsealed or missing archive files')
    return files


def disk_check(value, reserve, *, smoke=False):
    required = 0 if smoke else reserve
    require(type(value.get('remaining_bytes')) is int and value['remaining_bytes'] >= 0, 'invalid remaining disk budget')
    require(value.get('reserve_bytes', -1) >= required, 'disk reserve below registration')
    require(value.get('free_bytes', -1) >= value['remaining_bytes'] + value['reserve_bytes'], 'disk reserve/remaining budget failed')
    require(value.get('free_inodes', 0) >= 16, 'disk inode gate failed')
    require(type(value.get('filesystem_device')) is int and Path(value.get('path', '')).is_absolute(), 'disk filesystem identity missing')


def verify_protocol(args):
    protocol_path = safe_path(args.protocol_manifest)
    old_path = safe_path(args.old_protocol_manifest)
    protocol, old = read_json(protocol_path), read_json(old_path)
    for value in (protocol, old):
        require(value.get('manifest_hash') == canonical({k: v for k, v in value.items() if k != 'manifest_hash'}), 'protocol canonical hash mismatch')
    smoke = protocol.get('smoke')
    require(type(smoke) is bool and old.get('smoke') == smoke, 'protocol smoke mismatch')
    require(not smoke or args.allow_smoke, 'smoke cannot pass formal gate')
    require(protocol.get('protocol') == 'raw-deterioration-20260924-v1' and protocol.get('test_evaluation_allowed') is False, 'legacy protocol/test policy mismatch')
    require(sha(old_path) == protocol['old_protocol']['sha256'], 'old protocol binding mismatch')
    if not smoke:
        require(protocol['manifest_hash'] == FORMAL_PROTOCOL and sha(old_path) == FORMAL_OLD_SHA, 'unregistered formal source protocol')
        require(protocol['settings'] == dict(dim=256, token_dim=256, hist_len=50, batch_size=256, candidates=75), 'formal settings drift')
    document = safe_path(args.protocol_doc)
    require(sha(document) == DOC_SHA, 'registered V2 protocol document changed')
    legacy = safe_root(args.legacy_source_dir)
    actual = {p.name: sha(p) for p in legacy.glob('*.py')}
    require(actual == protocol['code_hashes'], 'legacy source set/hash mismatch')
    positions_path = safe_file(old_path.parent, old['ranker_train_positions'])
    positions = np.load(positions_path, allow_pickle=False, mmap_mode='r')
    require(positions.ndim == 1 and positions.dtype == np.dtype('int64') and len(positions) > 0
            and np.all(positions >= 0) and np.all(positions[1:] > positions[:-1]), 'invalid original train positions')
    require(array_sha(positions) == old['ranker_train_positions_sha256'], 'original positions hash mismatch')
    require(smoke or len(positions) == FORMAL_ROWS, 'formal original row count mismatch')
    old_cache_path = safe_path(args.old_train_cache)
    require(sha(old_cache_path) == protocol['candidate_caches']['train']['sha256'], 'original train cache binding mismatch')
    old_cache = read_json(old_cache_path)
    require(old_cache['source']['rrf_weights'] == [2., 0., .7, .05] and old_cache['shape'] == [len(positions), 75], 'original train cache policy mismatch')
    return dict(protocol=protocol, old=old, smoke=smoke, positions=positions, old_cache=old_cache,
                protocol_path=protocol_path, old_path=old_path, document=document, legacy=legacy)


def verify_cache_payload(root, sidecar, files):
    meta = read_json(sidecar)
    require(meta.get('schema') == 'cross-pools-int32-v2' and meta.get('status') == 'complete', 'unsupported/incomplete cache')
    require(meta.get('dtype') == meta.get('lengths_dtype') == '<i4' and meta.get('budget') == 75 and meta.get('block_rows') == 1024, 'cache dtype/budget/block mismatch')
    arrays = {}
    for name in ('items', 'lengths'):
        reference = meta['files'][name]
        path = safe_file(root, reference['name'])
        require(reference['name'] in files and sha(path) == reference['sha256'] == files[reference['name']]['sha256'], 'cache payload seal mismatch')
        arrays[name] = np.load(path, allow_pickle=False, mmap_mode='r')
        require(arrays[name].dtype.str == '<i4', 'cache payload dtype mismatch')
    items, lengths = arrays['items'], arrays['lengths']
    require(items.ndim == 2 and list(items.shape) == meta['shape'] and items.shape[1] == 75 and lengths.shape == (len(items),), 'cache payload shape mismatch')
    source = meta['source']
    require(meta['records_hash'] == meta['position_uid_row_hash'] == source['records_hash'] and len(items) == source['records_count'], 'cache row identity mismatch')
    digest = hashlib.sha256(); digest.update(b'[')
    for index, length in enumerate(lengths):
        length = int(length)
        require(0 <= length <= 75, 'invalid cache row length')
        row = items[index]; ids = row[:length]
        require(len(np.unique(ids)) == len(ids) and np.all(ids >= 2) and np.all(ids < source['catalog_size']) and np.all(row[length:] == 0), 'invalid cache IDs/duplicates/padding')
        if index: digest.update(b',')
        digest.update(json.dumps(ids.tolist(), separators=(',', ':')).encode())
    digest.update(b']')
    require(digest.hexdigest() == meta['pool_hash'], 'cache logical pool hash mismatch')
    return meta


def verify_pool(args, context, root_path, mode, pilot=None):
    root = safe_root(root_path)
    complete = read_json(root / 'COMPLETED.json')
    require(complete.get('status') == 'complete' and complete.get('protocol') == VERSION and complete.get('mode') == mode
            and complete.get('smoke') == context['smoke'], 'pool completion identity mismatch')
    for key in ('target_access_performed', 'scoring_performed', 'final_test_allowed', 'full_training_allowed'):
        require(complete.get(key) is False, 'pool target/test/full policy mismatch: ' + key)
    files = sealed_tree(root, complete)
    started = read_json(safe_file(root, 'CONSTRUCTION_STARTED.json'))
    source = complete['source']
    require(started['source'] == source and started['protocol'] == VERSION and started['mode'] == mode, 'pool start/completion source mismatch')
    original = context['positions']; expected_positions = original[:min(4096, len(original))] if mode == 'pilot' else original
    position_path = bound_file(complete['positions'], root / 'ranker_train_positions.npy', expected_name='ranker_train_positions.npy')
    require(position_path.name in files, 'positions not sealed')
    positions = np.load(position_path, allow_pickle=False)
    require(positions.dtype == original.dtype and np.array_equal(positions, expected_positions), 'pool training positions differ from registered slice')
    require(complete['rows'] == len(positions) and source['positions_sha256'] == array_sha(positions), 'pool position count/hash mismatch')
    protocol, old = context['protocol'], context['old']
    expected = dict(protocol=VERSION, base_data_id=old['base_data_id'], encoders_hash=old['encoders_hash'],
                    source_hashes=old['base_assets'], code_hashes=old['code_hashes'], legacy_source_hashes=protocol['code_hashes'],
                    budget=75, records_count=len(positions), rrf_weights=[2., 1., .7, .05], cf_neighbors=300,
                    itemcf_half_life_days=180., smoke_fallback=context['smoke'], mode=mode,
                    training_targets_read=False, final_test_allowed=False, full_training_allowed=False)
    require(all(source.get(key) == value for key, value in expected.items()), 'pool source contract mismatch')
    require(source['catalog_size'] == context['old_cache']['source']['catalog_size'], 'catalog cardinality changed')
    require(digest_ok(source.get('records_hash')), 'invalid record hash')
    if mode == 'full':
        require(source['records_hash'] == context['old_cache']['source']['records_hash'], 'full pool records differ from original train pool')
    bound_file(source['diagnostic_manifest'], context['protocol_path'])
    bound_file(source['protocol_document'], context['document'])
    bound_file(source['builder'], args.builder_source, expected_name='build_v2_train_pool.py')
    sidecar_name = 'candidate_pilot.json' if mode == 'pilot' else 'candidate_train_v2.json'
    sidecar = bound_file(complete['cache'], root / sidecar_name, expected_name=sidecar_name)
    require(sidecar_name in files, 'cache sidecar not sealed')
    meta = verify_cache_payload(root, sidecar, files)
    require(meta['source'] == source and meta['pool_hash'] == complete['pool_hash'], 'pool sidecar/completion mismatch')
    disk_check(started['disk'], 2 * 1024**3, smoke=context['smoke'])
    disk_check(complete['final_disk'], 2 * 1024**3, smoke=context['smoke'])
    require(started['disk']['filesystem_device'] == complete['final_disk']['filesystem_device'], 'pool output filesystem changed')
    blocks = complete['blocks']
    require(isinstance(blocks, list) and blocks and sum(row['rows'] for row in blocks) == len(positions), 'pool block coverage mismatch')
    require([row['rows'] for row in blocks] == [min(1024, len(positions) - start) for start in range(0, len(positions), 1024)], 'pool block boundaries mismatch')
    log = [json.loads(line) for line in safe_file(root, 'blocks.jsonl').read_text().splitlines()]
    require(log == [dict(block=index+1, **row) for index, row in enumerate(blocks)], 'pool block log mismatch')
    for row in blocks:
        require(row['seconds'] > 0 and math.isfinite(row['seconds']), 'invalid pool timing')
        close(row['rows_per_second'], row['rows'] / row['seconds'], 'pool block throughput')
    require(complete['initialization_seconds'] >= 0 and complete['build_and_validation_seconds'] >= sum(row['seconds'] for row in blocks), 'pool timing inconsistent')
    estimate = complete['initialization_seconds'] + max(row['seconds']/row['rows'] for row in blocks[-3:]) * len(original) * 1.5
    close(complete['conservative_full_estimate_seconds'], estimate, 'conservative pool estimate')
    require(complete['required_followup_reserve_seconds'] == 3600, 'followup time reserve changed')
    env = complete['environment']
    if not context['smoke']:
        require(env.get('cuda_available') is True and env.get('torch_threads') == 4, 'formal GPU environment missing')
        require(complete['peak_gpu_allocated_bytes'] > 0 and complete['peak_gpu_reserved_bytes'] >= complete['peak_gpu_allocated_bytes'], 'formal GPU peak evidence missing')
    require(complete['peak_host_rss_bytes'] > 0, 'host peak memory evidence missing')
    if mode == 'pilot':
        require(complete['pilot'] is None and started['pilot'] is None, 'pilot falsely references another pilot')
    else:
        require(pilot is not None, 'full pool requires independently verified pilot')
        bound_file(complete['pilot'], pilot['root'] / 'COMPLETED.json', expected_name='COMPLETED.json')
        require(started['pilot'] == complete['pilot'], 'full pilot binding differs')
        for key in ('builder', 'diagnostic_manifest', 'protocol_document', 'rrf_weights', 'legacy_source_hashes', 'source_hashes'):
            require(source[key] == pilot['complete']['source'][key], 'pilot/full source mismatch: ' + key)
        for key in ('python', 'torch', 'numpy', 'cuda_runtime', 'cudnn', 'torch_threads', 'matmul_allow_tf32', 'cudnn_allow_tf32', 'float32_matmul_precision'):
            require(env[key] == pilot['complete']['environment'][key], 'pilot/full environment mismatch: ' + key)
    return dict(root=root, complete=complete, files=files, meta=meta,
                receipt=dict(status='verified', mode=mode, rows=len(positions), completion_sha256=sha(root/'COMPLETED.json'),
                             pool_hash=meta['pool_hash'], cache_sha256=sha(sidecar), source_sha256=canonical(source),
                             evidence=files, target_data_opened=False,
                             limitation='Pilot identity hash is source-bound and positions matched; no raw dataset is loaded to independently re-derive pilot UID/name records.'))


def paired_equal(left, right):
    require(all(np.array_equal(left[key], right[key]) for key in STABLE), 'paired identities/candidates/labels/masks mismatch')


def recompute_paired(differences, paired, sources):
    require(paired.get('protocol') == VERSION and paired.get('final_test_allowed') is False
            and paired.get('full_training_allowed') is False, 'paired protocol/test/full policy mismatch')
    require(set(paired['per_seed']) == set(SEEDS), 'paired seed coverage mismatch')
    require(paired.get('bootstrap_seed') == 20260925 and paired.get('bootstrap_replicates') == 10000
            and paired.get('ci_level') == .975 and paired.get('quantiles') == [.0125, .9875], 'bootstrap registration mismatch')
    require(paired.get('array_sources') == sources, 'paired array source seal mismatch')
    recomputed = {}
    for metric, values in differences.items():
        require(len(values) == 3 and all(np.isfinite(row).all() for row in values), 'invalid paired differences')
        average = np.mean(values, axis=0); expected = paired[metric]
        close(expected['mean_delta'], average.mean(), metric+' mean delta')
        require(expected['users'] == len(average) and expected['gained'] == int((average > 0).sum())
                and expected['lost'] == int((average < 0).sum()), 'paired aggregate counts mismatch')
        rng = np.random.default_rng(20260925); samples = np.empty(10000)
        for index in range(10000):
            samples[index] = average[rng.integers(0, len(average), size=len(average))].mean()
        ci = np.quantile(samples, [.0125, .9875])
        require(len(expected['ci975']) == 2, 'CI shape mismatch')
        for actual, value in zip(expected['ci975'], ci):
            close(actual, value, metric+' independently recomputed CI975')
        recomputed[metric] = dict(mean=float(average.mean()), lower=float(ci[0]))
        for seed, delta in zip(SEEDS, values):
            item = paired['per_seed'][seed]
            close(item[metric+'_delta'], delta.mean(), metric+' per-seed difference')
            if metric == 'hit5':
                require(item['gained'] == int((delta > 0).sum()) and item['lost'] == int((delta < 0).sum()), 'paired per-seed counts mismatch')
    criteria = dict(all_seed_hr_positive=all(float(row.mean()) > 0 for row in differences['hit5']),
                    mean_hr_at_least_0_0005=recomputed['hit5']['mean'] >= .0005,
                    hr_ci975_lower_positive=recomputed['hit5']['lower'] > 0,
                    mean_ndcg_nonnegative=recomputed['ndcg5']['mean'] >= 0)
    require(paired['criteria'] == criteria, 'selection criteria mismatch')
    require(paired['supports_v2_train_pool_followup'] == all(criteria.values()), 'followup decision mismatch')
    require(paired['selection'] == ('v2_train_pool' if all(criteria.values()) else 'raw_retained'), 'selection result mismatch')


def system_disk_check(value, smoke):
    reserve = 0 if smoke else 1536 * 1024**2
    require(value['reserve_bytes'] >= reserve and value['atomic_extra_models'] == 1, 'system reserve/atomic budget changed')
    require(type(value['remaining_models']) is int and 0 <= value['remaining_models'] <= 3, 'invalid remaining model count')
    require(value['checkpoint_bound_bytes'] > 0 and value['remaining_arrays_logs_bytes'] >= 0, 'negative system budget')
    required = ((value['remaining_models'] + 1) * value['checkpoint_bound_bytes']
                + value['remaining_arrays_logs_bytes'] + value['reserve_bytes'])
    require(value['required_free_bytes'] == required and value['free_bytes'] >= required and value['free_inodes'] >= 64,
            'system free/remaining/inode gate failed')
    require(type(value['filesystem_device']) is int and Path(value['path']).is_absolute(), 'system filesystem identity missing')
    if 'data_disk' in value:
        disk_check(value['data_disk'], 2*1024**3, smoke=smoke)
        require(smoke or value['data_disk']['filesystem_device'] != value['filesystem_device'], 'data/system filesystem must differ')


def verify_run(args, context, pool):
    # Metric/tensor readers are independent of the new builder and training code.
    helper = safe_path(Path(__file__).with_name('verify_training_diagnostics_archive.py'))
    require(sha(helper) == METRIC_HELPER_SHA, 'independent metric helper source changed')
    spec = importlib.util.spec_from_file_location('v2_independent_metric_reader', helper)
    metrics_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics_module)
    load_arrays, verify_metrics = metrics_module.load_arrays, metrics_module.verify_metrics
    import torch

    require(pool is not None, 'full pool required for run verification')
    run = safe_root(args.run_dir); raw = safe_root(args.raw_run)
    complete = read_json(run/'COMPLETED.json')
    smoke = context['smoke']; protocol = context['protocol']; settings = protocol['settings']
    require(complete.get('status') == 'complete' and complete.get('protocol') == VERSION and complete.get('smoke') == smoke, 'run completion identity mismatch')
    for key in ('test_future_labels_read', 'final_test_allowed', 'full_training_allowed'):
        require(complete.get(key) is False, 'run test/full policy mismatch')
    files = sealed_tree(run, complete)
    def evidence(relative):
        require(relative in files, 'required run evidence not sealed: ' + relative)
        return safe_file(run, relative)

    raw_complete = read_json(raw/'COMPLETED.json')
    require(raw_complete.get('status') == 'complete' and raw_complete.get('test_future_labels_read') is False
            and raw_complete.get('protocol_manifest_hash') == protocol['manifest_hash'], 'raw completion protocol mismatch')
    raw_gate = read_json(args.raw_archive_receipt) if args.raw_archive_receipt else None
    if not smoke:
        require(raw_gate is not None and raw_gate.get('formal_archive_verified') is True and raw_gate.get('status') == 'verified'
                and raw_gate.get('completion_sha256') == sha(raw/'COMPLETED.json')
                and raw_gate.get('protocol_manifest_hash') == protocol['manifest_hash']
                and raw_gate.get('protocol_manifest_sha256') == sha(context['protocol_path']), 'raw archive receipt mismatch')
    def raw_evidence(relative):
        path = safe_file(raw, relative)
        require(relative in raw_complete['evidence_hashes'] and sha(path) == raw_complete['evidence_hashes'][relative], 'raw control evidence changed')
        if raw_gate:
            require(relative in raw_gate['evidence'] and sha(path) == raw_gate['evidence'][relative]['sha256'], 'raw archive evidence changed')
        return path

    manifest = read_json(evidence('run_manifest.json'))
    require(manifest['protocol'] == VERSION and manifest['smoke'] == smoke and manifest['legacy_protocol_manifest_hash'] == protocol['manifest_hash']
            and manifest['legacy_source_hashes'] == protocol['code_hashes'], 'run source identity mismatch')
    for key in ('test_future_labels_read', 'final_test_allowed', 'full_training_allowed'):
        require(manifest.get(key) is False, 'manifest test/full policy mismatch')
    bound_file(manifest['runner'], args.runner_source, expected_name='run_v2_train_alignment.py')
    bound_file(manifest['builder'], args.builder_source, expected_name='build_v2_train_pool.py')
    bound_file(manifest['protocol_manifest'], context['protocol_path'])
    bound_file(manifest['protocol_document'], context['document'])
    train = manifest['train_cache']
    bound_file(train['receipt'], pool['root']/'COMPLETED.json', expected_name='COMPLETED.json')
    require(train['cache'] == pool['complete']['cache'] and train['positions'] == pool['complete']['positions']
            and train['source'] == pool['complete']['source'] and train['pool_hash'] == pool['meta']['pool_hash'], 'run/new pool binding mismatch')
    require(manifest['eval_cache_refs'] == protocol['candidate_caches'], 'evaluation cache references changed')
    require(manifest['data_root'] == str(Path(pool['complete']['cache']['path']).parent)
            and Path(manifest['system_root']).is_absolute(), 'run output roots mismatch')

    controls = manifest['controls']
    bound_file(controls['raw_completion'], raw/'COMPLETED.json', expected_name='COMPLETED.json')
    bound_file(controls['raw_locked'], raw_evidence('CHECKPOINTS_LOCKED.json'), expected_name='CHECKPOINTS_LOCKED.json')
    require(controls.get('reuse_status') == 'registered_conditions_verified' and controls.get('no_raw_retraining') is True, 'raw reuse status mismatch')
    required_raw = {'run_config.json', 'panels.json', 'CHECKPOINTS_LOCKED.json'}
    for seed in SEEDS:
        prefix = f'seed_{seed}/raw/'
        required_raw.update(prefix+name for name in ('initial_state_audit.json','epoch_1_trace.json','history.json','best.pth','confirm_best_users.npz','confirm_best_metrics.json'))
        steps = raw_complete['seeds'][seed]['steps_per_epoch']
        required_raw.update(prefix+f'screen_step{step}_users.npz' for step in sorted(set(math.ceil(p*steps) for p in (.25,.5,1.))))
    require(set(controls['used_raw_files']) == required_raw, 'raw control file coverage mismatch')
    for relative, reference in controls['used_raw_files'].items():
        bound_file(reference, raw_evidence(relative), expected_name=Path(relative).name)
    if not smoke:
        bound_file(controls['archive_receipt'], args.raw_archive_receipt)
        require(args.backup_receipt, 'backup receipt path required')
        bound_file(controls['backup_receipt'], args.backup_receipt)
        backup = read_json(args.backup_receipt)
        require(backup.get('status') == 'verified' and backup['raw_completion_sha256'] == sha(raw/'COMPLETED.json')
                and backup['archive_receipt_sha256'] == sha(args.raw_archive_receipt), 'recoverable backup binding mismatch')

    config = manifest['config']; raw_config = read_json(raw_evidence('run_config.json'))
    require(config['epochs'] == 1 and config['smoke'] == smoke and manifest['scheduler_t_max'] == 3, 'run epoch/scheduler changed')
    for key in ('dim','token_dim','hist_len','batch_size','candidates'):
        require(config[key] == settings[key], 'run common setting changed: ' + key)
    require(raw_config['epochs'] == 3 and raw_config['seeds'] == [42,43,44] and raw_config['screen_checkpoints'] == [.25,.5,1.,2.,3.], 'raw reference training schedule changed')
    require(manifest['factors'] == raw_config['factors'], 'item SVD initialization changed')
    require(config['train_pool_hash'] == train['pool_hash'], 'model training pool hash mismatch')
    if not smoke:
        require(config['device'] == 'cuda' and manifest['factors']['synthetic'] is False, 'formal device/factors mismatch')
        require(config['checkpoint_bound'] >= 1_260_800_000 and config['array_allowance'] >= 250_000_000, 'formal resource budget reduced')
        for key in ('torch','numpy','cuda_runtime','cudnn','torch_threads','torch_interop_threads','gpu'):
            require(manifest['environment'][key] == raw_config['environment'][key], 'raw/new environment mismatch: ' + key)
        require(manifest['environment']['python'].split()[0] == raw_config['environment']['python'].split()[0], 'raw/new Python mismatch')
        require(manifest['environment']['matmul_allow_tf32'] is False and manifest['environment']['cudnn_allow_tf32'] is True
                and manifest['environment']['float32_matmul_precision'] == 'highest', 'formal precision defaults changed')
    old_has_tf32 = all(key in raw_config['environment'] for key in ('matmul_allow_tf32','cudnn_allow_tf32','float32_matmul_precision'))
    require(manifest['environment']['historical_tf32_recorded'] == old_has_tf32
            and manifest['environment']['historical_tf32_equality_verified'] == old_has_tf32, 'historical precision evidence overstated')
    for key in ('torch','numpy','cuda_runtime','cudnn','torch_threads','matmul_allow_tf32','cudnn_allow_tf32','float32_matmul_precision'):
        require(manifest['environment'][key] == pool['complete']['environment'][key], 'pool/run environment mismatch: ' + key)
    panels = read_json(evidence('panels.json'))
    require(panels == read_json(raw_evidence('panels.json')) and panels['identity_sha256'] == canonical({k:v for k,v in panels.items() if k!='identity_sha256'}), 'fixed old diagnostic panels changed')
    resources = read_json(evidence('resource_snapshots.json'))
    require(resources and resources[0] == manifest['resource_budget'] and resources[0]['remaining_models'] == 3, 'initial resource receipt mismatch')
    for item in resources:
        system_disk_check(item, smoke)
        require(item['checkpoint_bound_bytes'] == config['checkpoint_bound'], 'checkpoint budget changed within run')
    # Initial snapshot plus seed start, three screens and one save per seed, plus confirm.
    require(len(resources) == 19, 'stage resource snapshot coverage mismatch')
    require([item['remaining_models'] for item in resources] == [3]+[3]*5+[2]*5+[1]*5+[0]*3, 'stage resource model budget sequence mismatch')
    for item in resources[1:]:
        require('data_disk' in item, 'stage data-disk snapshot missing')

    locked = read_json(evidence('CHECKPOINTS_LOCKED.json')); raw_locked = read_json(raw_evidence('CHECKPOINTS_LOCKED.json'))
    require(set(locked) == set(complete['seeds']) == set(SEEDS), 'run seed coverage mismatch')
    differences = {'hit5':[], 'ndcg5':[]}; array_sources = {}; screen_reference = confirm_reference = None
    counts = dict(checkpoints=0,screen_arrays=0,confirm_arrays=0)
    expected_files = {'run_manifest.json','panels.json','CHECKPOINTS_LOCKED.json','paired_confirm.json','resource_snapshots.json'}
    for seed in SEEDS:
        prefix = f'seed_{seed}/raw_v2_train/'; rp = f'seed_{seed}/raw/'
        expected_files.update(prefix+name for name in ('COMPLETED.json','initial_state_audit.json','epoch_1_trace.json','history.json','steps.jsonl','best.pth','confirm_best_users.npz','confirm_best_metrics.json'))
        result = read_json(evidence(prefix+'COMPLETED.json'))
        require(result == complete['seeds'][seed] and result['protocol'] == VERSION and result['variant'] == 'raw_v2_train'
                and result['seed'] == int(seed) and result['init_seed'] == 424242+int(seed)-42
                and result['precision'] == 'FP32' and result['automatic_resume'] is False and result['optimizer_saved'] is False,
                'seed training identity mismatch')
        require(result['protocol_manifest_hash'] == protocol['manifest_hash'] and result['train_pool_hash'] == train['pool_hash'], 'seed source identity mismatch')
        steps = result['steps_per_epoch']; raw_result = raw_complete['seeds'][seed]
        require(steps == raw_result['steps_per_epoch'] and (smoke or steps == 2362), 'seed update count mismatch')
        require(result['model_config'] == raw_result['model_config'] and result['model_config']['cross_mode'] == 'raw'
                and result['model_config']['fusion'] == 'concat', 'model configuration changed beyond pool')
        audit = read_json(evidence(prefix+'initial_state_audit.json')); old_audit = read_json(raw_evidence(rp+'initial_state_audit.json'))
        for key in ('state_sha256','template_sha256','copied_tensor_keys','gate_difference_whitelist','gate_value'):
            require(audit[key] == old_audit[key], 'initial parameter/buffer audit differs: ' + key)
        require(audit['all_parameters_and_buffers_equal_to_raw'] is True and audit['gate_value'] == 10., 'initial raw equality missing')
        trace = read_json(evidence(prefix+'epoch_1_trace.json')); raw_trace = read_json(raw_evidence(rp+'epoch_1_trace.json'))
        require(result['traces'] == {'1':trace}, 'seed trace seal mismatch')
        for key in ('rows','order_sha256','batch_sizes','covered_users'):
            require(trace[key] == raw_trace[key], 'row/order/batch/coverage invariant mismatch: ' + key)
        require(trace['rows'] == len(context['positions']) and len(trace['batch_sizes']) == steps and sum(trace['batch_sizes']) == trace['rows'], 'trace row/batch size mismatch')
        require(smoke or (trace['covered_users'] == 100000 and trace['batch_sizes'][-1] == 95), 'formal user/last-batch coverage mismatch')
        require(trace.get('negative_stream_expected_equal') is False and digest_ok(trace.get('negative_digest')) and digest_ok(trace.get('sha256')), 'independent negative stream evidence missing')
        require(set(trace['negative_sources']) == {'band11_25','band26_50','candidate_fallback','random'}
                and all(type(value) is int and value >= 0 for value in trace['negative_sources'].values())
                and sum(trace['negative_sources'].values()) == trace['rows']*16, 'negative source accounting mismatch')
        step_rows = [json.loads(line) for line in evidence(prefix+'steps.jsonl').read_text().splitlines()]
        require([row['step'] for row in step_rows] == list(range(1,steps+1)) and [row['batch_rows'] for row in step_rows] == trace['batch_sizes'], 'training update sequence mismatch')
        require(all(row['epoch'] == 1 and row['lr'] == .001 and math.isfinite(row['loss']) and row['loss'] >= 0
                    and math.isfinite(row['gradient_l2_before_clip']) and row['gradient_l2_before_clip'] >= 0 for row in step_rows), 'training numeric/LR contract mismatch')
        history = read_json(evidence(prefix+'history.json')); old_history = read_json(raw_evidence(rp+'history.json'))
        checks = sorted(set(math.ceil(point*steps) for point in (.25,.5,1.)))
        require([row['step'] for row in history] == checks, 'screen schedule mismatch')
        early = [row for row in old_history if row['step'] <= steps]
        require([row['step'] for row in early] == checks, 'old screen schedule mismatch')
        raw_best = max(early,key=lambda row:(row['screen']['hr5'],row['screen']['ndcg5'],-row['step']))
        require(raw_best['step'] == raw_locked[seed]['best']['step'] and (smoke or raw_best['step'] == {'42':591,'43':1181,'44':1181}[seed]), 'old selected early-best changed')
        best = None
        for entry in history:
            close(entry['epoch_fraction'],entry['step']/steps,'screen epoch fraction')
            current_key = (entry['screen']['hr5'],entry['screen']['ndcg5'],-entry['step'])
            if best is None or current_key > (best['screen']['hr5'],best['screen']['ndcg5'],-best['step']): best = entry
            require(entry['selected_best_step'] == best['step'] and entry['selected'] == (entry['step'] == best['step'])
                    and entry['selected_state_sha256'] == best['selected_state_sha256'], 'sequential CPU best selection mismatch')
            require(set(entry['selected_state_sha256']) == set(audit['state_sha256']) and all(digest_ok(item) for item in entry['selected_state_sha256'].values()), 'selected state hash set mismatch')
            name = f"screen_step{entry['step']}_users.npz"; expected_files.add(prefix+name)
            observed = load_arrays(evidence(prefix+name)); control = load_arrays(raw_evidence(rp+name))
            verify_metrics(observed,entry['screen']); verify_metrics(control,next(row['screen'] for row in early if row['step']==entry['step'])); paired_equal(observed,control)
            require(smoke or len(observed['uid']) == 20000, 'screen user count mismatch')
            require(observed['candidate_ids'].shape[1] <= settings['candidates'] and np.all(observed['candidate_ids'] < result['model_config']['num_items']), 'screen candidate range mismatch')
            if screen_reference is None: screen_reference = {key:observed[key] for key in STABLE}
            else: paired_equal(observed,screen_reference)
            invariant = entry['diagnostics']['candidate_batch_invariance']
            require(invariant['passed'] is True and invariant['atol'] == invariant['rtol'] == 1e-5, 'candidate batch invariance failed')
            for bn in entry['diagnostics']['batchnorm'].values():
                require(bn['batches'] == entry['step'], 'diagnostic observation changed BN count')
            counts['screen_arrays'] += 1
        metadata = locked[seed]
        require(metadata == result['best'] and metadata['step'] == best['step'] and metadata['screen'] == best['screen']
                and metadata['selected_state_sha256'] == best['selected_state_sha256'] and metadata['cpu_clone_includes_buffers'] is True, 'locked best/clone mismatch')
        for key in ('protocol','variant','seed','init_seed','model_config','protocol_manifest_hash','train_pool_hash','automatic_resume','optimizer_saved','precision'):
            require(metadata[key] == result[key], 'checkpoint training metadata mismatch: ' + key)
        checkpoint = evidence(prefix+'best.pth')
        require(sha(checkpoint) == metadata['checkpoint_sha256'] and checkpoint.stat().st_size == metadata['checkpoint_bytes']
                and checkpoint.stat().st_size <= config['checkpoint_bound'], 'checkpoint lock/byte bound mismatch')
        payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
        require(payload['manifest'] == {key:value for key,value in metadata.items() if key not in ('checkpoint_sha256','checkpoint_bytes')}, 'checkpoint embedded manifest mismatch')
        require(set(payload['model']) == set(audit['copied_tensor_keys'])
                and all(torch.is_tensor(value) and torch.isfinite(value).all().item() for value in payload['model'].values()), 'checkpoint keys/nonfinite mismatch')
        state_hashes = {key:array_sha(value.detach().cpu().numpy()) for key,value in payload['model'].items()}
        require(state_hashes == metadata['selected_state_sha256'], 'checkpoint differs from selected CPU state')
        model_specs = {key:(tuple(value.shape),value.dtype) for key,value in payload['model'].items()}
        del payload
        old_checkpoint = raw_evidence(rp+'best.pth')
        require(sha(old_checkpoint) == raw_locked[seed]['best']['checkpoint_sha256'], 'old best checkpoint lock mismatch')
        old_payload = torch.load(old_checkpoint,map_location='cpu',weights_only=False)
        require(model_specs == {key:(tuple(value.shape),value.dtype) for key,value in old_payload['model'].items()}, 'checkpoint tensor shape/dtype differs from old raw')
        del old_payload; counts['checkpoints'] += 1
        new_path = evidence(prefix+'confirm_best_users.npz'); old_path = raw_evidence(rp+'confirm_best_users.npz')
        observed, control = load_arrays(new_path), load_arrays(old_path)
        paired_equal(observed,control); verify_metrics(observed,read_json(evidence(prefix+'confirm_best_metrics.json')))
        verify_metrics(control,read_json(raw_evidence(rp+'confirm_best_metrics.json')))
        require(smoke or len(observed['uid']) == 80000, 'confirm user count mismatch')
        require(observed['candidate_ids'].shape[1] <= settings['candidates'] and np.all(observed['candidate_ids'] < result['model_config']['num_items']), 'confirm candidate range mismatch')
        if confirm_reference is None: confirm_reference = {key:observed[key] for key in STABLE}
        else: paired_equal(observed,confirm_reference)
        for metric in differences: differences[metric].append(observed[metric].astype(np.float64)-control[metric].astype(np.float64))
        sources = complete['paired']['array_sources'][seed]
        bound_file(sources['new_arrays'],new_path,expected_name='confirm_best_users.npz')
        bound_file(sources['raw_arrays'],old_path,expected_name='confirm_best_users.npz')
        array_sources[seed] = sources; counts['confirm_arrays'] += 1
    require(set(files) == expected_files and counts == dict(checkpoints=3,screen_arrays=9,confirm_arrays=3), 'run evidence coverage mismatch')
    require(not set(screen_reference['uid']) & set(confirm_reference['uid']), 'screen/confirm overlap')
    paired = read_json(evidence('paired_confirm.json'))
    require(paired == complete['paired'] and paired['comparison'] == 'v2-train-pool early-best minus old raw early-best; reused development confirm', 'paired completion/estimand mismatch')
    recompute_paired(differences,paired,array_sources)
    return dict(status='verified',formal_archive_verified=not smoke,smoke=smoke,counts=counts,
                completion_sha256=sha(run/'COMPLETED.json'),evidence=files,selection=paired['selection'],
                metric_helper_sha256=METRIC_HELPER_SHA,
                supports_v2_train_pool_followup=paired['supports_v2_train_pool_followup'],
                bootstrap_independently_recomputed=True,raw_archive_receipt_sha256=sha(args.raw_archive_receipt) if args.raw_archive_receipt else None,
                limitations=['Training is not replayed; initialization and trace records are authenticated against sealed raw controls.',
                             'NDCG denominators are not reopened; stored per-user bounds, paired identities and aggregate statistics are verified.',
                             'Old raw archive authentication and transfer inventory remain explicit separate prerequisites.',
                             'Confirm remains reused development data; no new test or full training is authorized.'])


def verify(args):
    context = verify_protocol(args)
    pilot = verify_pool(args, context, args.pilot_dir, 'pilot')
    pool = verify_pool(args, context, args.pool_dir, 'full', pilot) if args.pool_dir else None
    require(args.pool_only or args.run_dir, 'run directory required unless pool-only')
    result = dict(status='verified', formal_archive_verified=False, smoke=context['smoke'], protocol=VERSION,
                  protocol_doc_sha256=DOC_SHA, legacy_protocol_sha256=sha(context['protocol_path']),
                  pilot=pilot['receipt'], pool=pool['receipt'] if pool else None,
                  test_future_labels_read=False, full_training_allowed=False, final_test_allowed=False,
                  verifier_sha256=sha(__file__))
    roots = [pilot['root'], context['legacy'], context['old_path'].parent]
    if pool is not None: roots.append(pool['root'])
    if not args.pool_only:
        result['run'] = verify_run(args,context,pool)
        result['formal_archive_verified'] = result['run']['formal_archive_verified']
        roots.extend([safe_path(args.run_dir),safe_path(args.raw_run)])
    receipt = safe_path(args.receipt)
    require(not receipt.exists() and all(not receipt.is_relative_to(root) for root in roots), 'receipt overwrites or is inside immutable input')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir', 'protocol-manifest', 'old-protocol-manifest', 'old-train-cache', 'protocol-doc', 'legacy-source-dir', 'builder-source', 'receipt'):
        parser.add_argument('--'+name, required=True)
    for name in ('pool-dir', 'run-dir', 'raw-run', 'raw-archive-receipt', 'backup-receipt', 'runner-source'):
        parser.add_argument('--'+name)
    parser.add_argument('--pool-only', action='store_true')
    parser.add_argument('--allow-smoke', action='store_true')
    result = verify(parser.parse_args(argv))
    print(json.dumps(dict(status=result['status'], formal_archive_verified=result['formal_archive_verified'],
                          pilot_rows=result['pilot']['rows'], pool_rows=result['pool']['rows'] if result['pool'] else None)))


if __name__ == '__main__':
    main()
