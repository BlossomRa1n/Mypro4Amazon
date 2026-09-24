#!/usr/bin/env python3
"""Build exact frozen train-prefix recall pools without opening future labels.

Run in a fresh CPU-only process. A full run publishes candidate_final_train.json;
pilots publish candidate_pilot.json and cannot masquerade as full training pools.
Existing output and interrupted construction are never overwritten or deleted.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

# Set before importing numpy, torch, or any frozen source module.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

FORMAL_DATA_ID = '754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158'
FORMAL_ROWS = 4731777
FORMAL_DIAGNOSTIC_HASH = 'c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6'
FORMAL_OLD_PROTOCOL_SHA = '4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1'
DISK_RESERVE_BYTES = 1024 ** 3
WEIGHTS = [2.0, 0.0, 0.7, 0.05]
BUDGET = 75
_CONTEXT = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def checked(path, expected):
    path = Path(path)
    if sha(path) != expected: raise ValueError('input SHA-256 drift: ' + str(path))
    return path


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    payload = {k: v for k, v in value.items() if k != 'manifest_hash'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()
    if value.get('manifest_hash') != digest: raise ValueError('manifest hash mismatch')
    return value


def code_hashes(directory):
    return {p.name: sha(p) for p in sorted(Path(directory).glob('*.py'))}


def validate_formal_pins(diagnostic, smoke):
    if not smoke:
        if diagnostic.get('manifest_hash') != FORMAL_DIAGNOSTIC_HASH:
            raise ValueError('unregistered formal diagnostic manifest')
        if diagnostic.get('old_protocol', {}).get('sha256') != FORMAL_OLD_PROTOCOL_SHA:
            raise ValueError('unregistered original protocol SHA-256')


def disk_preflight(output, rows):
    """Check the output filesystem before any position or cache allocation."""
    output = Path(output)
    # Same-filesystem rename publishes each payload without a second allocation.
    payload_bytes = int(rows) * (BUDGET * 4 + 4 + 8)
    # Three small NPY headers plus JSON/lock receipts; deliberately conservative.
    metadata_allowance = 1024 ** 2
    required = payload_bytes + metadata_allowance + DISK_RESERVE_BYTES
    usage = shutil.disk_usage(output)
    receipt = dict(schema='full-prefix-disk-preflight-v1',
                   status='passed' if usage.free >= required else 'insufficient_space',
                   output_directory=str(output.resolve()), filesystem_device=os.stat(output).st_dev,
                   rows=int(rows), payload_bytes=payload_bytes,
                   metadata_allowance_bytes=metadata_allowance,
                   reserve_bytes=DISK_RESERVE_BYTES, required_free_bytes=required,
                   observed_free_bytes=usage.free,
                   payload_publication='same-filesystem rename; no duplicate payload allocation')
    exclusive_json(output / 'DISK_PREFLIGHT.json', receipt)
    if usage.free < required:
        raise OSError('insufficient output filesystem space: need ' + str(required) +
                      ' free bytes, observed ' + str(usage.free))
    return receipt


def exclusive_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.incomplete')
    with temporary.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    if path.exists(): raise FileExistsError(path)
    os.rename(temporary, path)


def validate_workers(workers):
    if type(workers) is not int or workers < 1 or workers > 4:
        raise ValueError('workers must be an integer in [1, 4]')


def validate_positions(data, formal=False):
    import numpy as np
    positions = np.asarray(data.train_positions)
    if positions.ndim != 1 or positions.dtype.kind not in 'iu' or not len(positions):
        raise ValueError('invalid train positions')
    if np.any(positions < 0) or np.any(positions >= len(data.uid)):
        raise ValueError('train position out of bounds')
    if np.any(positions[1:] <= positions[:-1]): raise ValueError('train row reordering/duplicates')
    mask = np.arange(len(data.uid)) < data.train_ends[data.uid]
    if not np.array_equal(mask, data.train_mask): raise ValueError('legal train mask drift')
    expected = np.flatnonzero(mask & (data.ts > data.ts[data.starts[data.uid]]))
    if not np.array_equal(positions, expected): raise ValueError('full legal train position coverage mismatch')
    counts = np.bincount(data.iid[mask], minlength=len(data.items))
    if not np.array_equal(counts, data.train_counts): raise ValueError('train-only item counts drift')
    if formal and (len(positions) != FORMAL_ROWS or data.manifest['data_id'] != FORMAL_DATA_ID):
        raise ValueError('formal full-prefix identity/count mismatch')
    return np.asarray(positions, dtype=np.int64).copy()


class TrainOnlyArray:
    """Permit only reads whose indices are within the legal training mask."""
    def __init__(self, array, mask): self._array, self._mask = array, mask
    def __getitem__(self, index):
        import numpy as np
        if not np.all(self._mask[index]): raise PermissionError('future event read forbidden')
        return self._array[index]
    def __array__(self, *args, **kwargs): raise PermissionError('unguarded full event array read forbidden')
    def __len__(self): return len(self._array)


def forbidden_targets(*args, **kwargs):
    raise PermissionError('all future target access is forbidden during candidate construction')


def guard_data(data):
    data.targets = forbidden_targets
    data.evaluation = forbidden_targets
    for name in ('iid', 'ts', 'rating'):
        if hasattr(data, name):
            array = getattr(data, name)
            array.flags.writeable = False
            setattr(data, name, TrainOnlyArray(array, data.train_mask))
    return data


def _work(task):
    sequence, records = task
    return sequence, [tuple(row) for row in records], _CONTEXT(records)


class OrderedBuilder:
    """At most one 1024-row block and four subchunks are in flight."""
    def __init__(self, builder, workers):
        validate_workers(workers)
        self.builder, self.workers, self.pool = builder, workers, None
    def __enter__(self):
        global _CONTEXT
        import torch
        if torch.cuda.is_initialized(): raise RuntimeError('CUDA initialized before CPU fork')
        torch.set_num_threads(1)
        if self.workers > 1:
            if 'fork' not in mp.get_all_start_methods(): raise RuntimeError('safe CPU fork unavailable')
            _CONTEXT = self.builder
            self.pool = mp.get_context('fork').Pool(self.workers)
        return self
    def __call__(self, records):
        rows = list(records)
        if not self.pool: return self.builder(rows)
        chunk = max(1, (len(rows) + self.workers - 1) // self.workers)
        tasks = [(index, rows[start:start + chunk])
                 for index, start in enumerate(range(0, len(rows), chunk))]
        result = []
        for task, value in zip(tasks, self.pool.map(_work, tasks, chunksize=1)):
            index, identity, candidates = value
            if index != task[0] or identity != task[1] or len(candidates) != len(identity):
                raise ValueError('parallel candidate row identity/order mismatch')
            result.extend(candidates)
        if len(result) != len(rows): raise ValueError('parallel row count mismatch')
        return result
    def __exit__(self, exc_type, exc, tb):
        if self.pool:
            if exc_type: self.pool.terminate()
            else: self.pool.close()
            self.pool.join()
        global _CONTEXT
        _CONTEXT = None


def source_for(old, data, records, cache, smoke):
    return dict(base_data_id=old['base_data_id'], source_hashes=old['base_assets'],
                records_hash=cache.rows_hash(records), records_count=len(records), budget=BUDGET,
                candidate_policy='four-way RRF from train corpus; no future target filtering',
                rrf_weights=WEIGHTS, itemcf_half_life_days=180.0, smoke_fallback=smoke,
                encoders_hash=old['encoders_hash'], code_hashes=old['code_hashes'],
                cf_neighbors=300, catalog_size=len(data.items),
                padding='zero tail, lengths define valid ordered IDs')


def uniform_indices(count, requested):
    import numpy as np
    if requested < 1 or requested > count: raise ValueError('sample row count outside legal range')
    return np.linspace(0, count - 1, requested, dtype=np.int64)


def run(args):
    validate_workers(args.workers)
    if args.pilot_rows is not None and args.pilot_rows < 1: raise ValueError('pilot rows must be positive')
    source_dir = Path(args.legacy_source_dir).resolve()
    diagnostic_path = Path(args.protocol_manifest).resolve()
    diagnostic = read_manifest(diagnostic_path)
    diagnostic_hash = sha(diagnostic_path)
    if diagnostic.get('protocol') != 'raw-deterioration-20260924-v1': raise ValueError('wrong diagnostic protocol')
    if diagnostic.get('smoke') is not args.smoke: raise ValueError('smoke flag must match diagnostic manifest')
    if diagnostic.get('test_evaluation_allowed') is not False: raise ValueError('test must remain sealed')
    validate_formal_pins(diagnostic, args.smoke)
    frozen_hashes = code_hashes(source_dir)
    if not frozen_hashes or frozen_hashes != diagnostic['code_hashes']: raise ValueError('frozen source code drift')
    for name in ('run_cross_multiseed', 'cross_pool_cache', 'baseline_data'):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != source_dir:
            raise RuntimeError('run in fresh process: incompatible imported frozen module ' + name)
    sys.path.insert(0, str(source_dir))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized(): raise RuntimeError('CUDA initialized before CPU-only construction')
    r = importlib.import_module('run_cross_multiseed')
    cache = importlib.import_module('cross_pool_cache')
    old_ref = diagnostic['old_protocol']
    old_path = checked(old_ref['path'], old_ref['sha256'])
    if not args.smoke: checked(old_path, FORMAL_OLD_PROTOCOL_SHA)
    old = read_manifest(old_path)
    if old.get('smoke') is not args.smoke: raise ValueError('old protocol smoke flag drift')
    base = Path(diagnostic['base_run']).resolve()
    if base != Path(old['base_run']).resolve(): raise ValueError('base directory mismatch')
    assets = old['base_assets']
    for required in ('data.pkl', 'itemcf.pkl', 'v2_best.pth', 'run_manifest.json'):
        if required not in assets and not args.smoke: raise ValueError('unbound base asset ' + required)
    def verify_inputs():
        checked(diagnostic_path, diagnostic_hash)
        checked(old_path, old_ref['sha256'])
        if code_hashes(source_dir) != frozen_hashes: raise ValueError('frozen source drift during construction')
        for name, digest in assets.items():
            if Path(name).name != name: raise ValueError('unsafe base asset name')
            checked(base / name, digest)
        checked(base / 'data.pkl', old['base_data_hash'])
    verify_inputs()
    data = r.load_data(base)
    if data.manifest['data_id'] != old['base_data_id'] or data.manifest['encoders_hash'] != old['encoders_hash']:
        raise ValueError('base identity mismatch')
    positions = validate_positions(data, formal=not args.smoke)
    full_count = len(positions)
    full_hash = cache.rows_hash(cache.PositionRecords(data, positions))
    if args.pilot_rows is not None: positions = positions[uniform_indices(len(positions), args.pilot_rows)]
    positions.flags.writeable = False
    records = cache.PositionRecords(data, positions)
    source = source_for(old, data, records, cache, args.smoke)
    data = guard_data(data)
    builder = r.RecallPoolBuilder(data, base, BUDGET, WEIGHTS, args.smoke)
    for value in vars(builder).values():
        if isinstance(value, np.ndarray): value.flags.writeable = False
    for value in vars(data).values():
        if isinstance(value, np.ndarray): value.flags.writeable = False
    sample = [records[int(i)] for i in uniform_indices(len(records), min(128, len(records)))]
    started = time.monotonic()
    serial = builder(sample)
    serial_seconds = time.monotonic() - started
    output = Path(args.output_dir).resolve()
    if output.exists(): raise FileExistsError('refuse to overwrite output: ' + str(output))
    output.mkdir(parents=True, exist_ok=False)
    disk_receipt = disk_preflight(output, len(records))
    receipt = dict(schema='full-prefix-pools-v1', status='building', mode='pilot' if args.pilot_rows else 'full',
                   full_legal_rows=full_count, full_records_hash=full_hash,
                   selected_rows=len(records), source=source, workers=args.workers,
                   frozen_source_dir=str(source_dir), frozen_code_hashes=frozen_hashes,
                   builder_sha256=sha(__file__), protocol_manifest_sha256=diagnostic_hash,
                   old_protocol_sha256=old_ref['sha256'], cpu_only=True,
                   disk_preflight=disk_receipt,
                   candidate_algorithm='unchanged frozen RecallPoolBuilder; ordered CPU fork subchunks',
                   test_future_labels_read=False, test_pool_constructed=False)
    exclusive_json(output / 'CONSTRUCTION_STARTED.json', receipt)
    with OrderedBuilder(builder, args.workers) as parallel:
        started = time.monotonic(); actual = parallel(sample); parallel_seconds = time.monotonic() - started
        if actual != serial: raise ValueError('parallel and serial ordered candidates differ')
        overlap = None
        if not args.smoke:
            original_positions = np.load(old_path.parent / old['ranker_train_positions'], allow_pickle=False, mmap_mode='r')
            if r.sha256_array(original_positions) != old['ranker_train_positions_sha256']:
                raise ValueError('original cache training positions drift')
            if np.any(original_positions[1:] <= original_positions[:-1]): raise ValueError('original positions reordered')
            located = np.searchsorted(np.asarray(data.train_positions), original_positions)
            if np.any(located >= len(data.train_positions)) or not np.array_equal(np.asarray(data.train_positions)[located], original_positions):
                raise ValueError('original cache positions not legal full-prefix rows')
            old_records = cache.PositionRecords(data, original_positions)
            old_source = source_for(old, data, old_records, cache, False)
            old_cache_ref = diagnostic['candidate_caches']['train']
            old_cache_path = checked(old_cache_ref['path'], old_cache_ref['sha256'])
            old_pools, old_meta = cache.load_cache(old_cache_path, old_source)
            selected = uniform_indices(len(old_records), min(128, len(old_records)))
            repeated = parallel([old_records[int(i)] for i in selected])
            if repeated != [old_pools[int(i)].tolist() for i in selected]:
                raise ValueError('frozen builder differs from original candidate cache on overlap')
            overlap = dict(rows=len(selected), existing_rows=len(old_records),
                           existing_pool_hash=old_meta['pool_hash'], exact_ordered_equality=True)
        position_path = output / 'ranker_train_positions.npy'
        with position_path.with_suffix('.npy.incomplete').open('xb') as f:
            np.save(f, positions, allow_pickle=False); f.flush(); os.fsync(f.fileno())
        os.rename(position_path.with_suffix('.npy.incomplete'), position_path)
        cache_path = output / ('candidate_pilot.json' if args.pilot_rows else 'candidate_final_train.json')
        completed = 0
        candidate_seconds = 0.0
        started = time.monotonic()
        def block(rows):
            nonlocal completed, candidate_seconds
            block_started = time.monotonic()
            result = parallel(rows)
            candidate_seconds += time.monotonic() - block_started
            completed += len(rows)
            if completed == len(records): verify_inputs()
            if completed % (1024 * 64) == 0 or completed == len(records):
                print(json.dumps(dict(completed=completed, total=len(records), elapsed_seconds=time.monotonic()-started)), flush=True)
            return result
        pools, metadata = cache.build_cache(cache_path, records, BUDGET, source, block)
        elapsed = time.monotonic() - started
    if len(pools) != len(records) or metadata['records_hash'] != source['records_hash']:
        raise ValueError('completed cache row identity mismatch')
    receipt.update(status='complete', cache_path=cache_path.name, cache_sha256=sha(cache_path),
                   positions_path=position_path.name, positions_sha256=sha(position_path),
                   pool_hash=metadata['pool_hash'], records_hash=metadata['records_hash'],
                   build_and_validation_seconds=elapsed, rows_per_second=len(records)/max(elapsed,1e-9),
                   candidate_build_seconds=candidate_seconds,
                   candidate_rows_per_second=len(records)/max(candidate_seconds,1e-9),
                   serial_parallel_check=dict(rows=len(sample), exact_ordered_equality=True,
                                              serial_seconds=serial_seconds, parallel_seconds=parallel_seconds),
                   original_cache_overlap=overlap)
    exclusive_json(output / 'COMPLETED.json', receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-source-dir', required=True)
    p.add_argument('--protocol-manifest', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--pilot-rows', type=int)
    p.add_argument('--smoke', action='store_true', help='tiny fixture only; must match manifest')
    run(p.parse_args())


if __name__ == '__main__': main()
