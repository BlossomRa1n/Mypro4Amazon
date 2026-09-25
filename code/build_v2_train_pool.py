"""Independent bounded GPU builder for the registered V2-aligned train pool.

No evaluation or training entry point; legacy source and caches remain read-only.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import numpy as np
import torch

sys.dont_write_bytecode = True
VERSION = 'afternoon-v2-train-pool-20260925-v1'
DOC_SHA = 'e0b5b95aaf2d9b39f55849c41c1733906af1eb8e4ba3de14973f0859c150ac05'
WEIGHTS = [2., 1., .7, .05]
FORMAL_ROWS = 604511
DATA_RESERVE = 2 * 1024**3
MAX_META = 16 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def safe(value):
    p = Path(value).absolute()
    need('..' not in p.parts and not any(q.is_symlink() for q in (p, *p.parents)), 'unsafe path: ' + str(p))
    return p


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def ref(path):
    p = safe(path)
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def read(path):
    def pairs(rows):
        result = {}
        for k, v in rows:
            need(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(safe(path).read_text(), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    with safe(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def legacy_inputs(args):
    directory = safe(args.legacy_source_dir)
    manifest = read(args.protocol_manifest)
    need(bool(args.smoke) is manifest['smoke'], 'smoke mode differs')
    need(sha(args.protocol_doc) == DOC_SHA, 'new registered protocol changed')
    actual = {p.name: sha(p) for p in directory.glob('*.py')}
    need(actual == manifest['code_hashes'], 'legacy source set/hash differs')
    for name in ('run_cross_multiseed', 'diagnostic_protocol', 'cross_pool_cache', 'run_training_diagnostics', 'diagnostic_metrics', 'baseline_data', 'token_models'):
        if name in sys.modules:
            need(Path(sys.modules[name].__file__).resolve().parent == directory, 'conflicting legacy import: ' + name)
    sys.path.insert(0, str(directory))
    r = importlib.import_module('run_cross_multiseed')
    cache = importlib.import_module('cross_pool_cache')
    protocol = importlib.import_module('diagnostic_protocol')
    manifest, data, records, positions, pools = protocol.load_inputs(safe(args.protocol_manifest))
    need(r.code_hashes() == actual, 'imported legacy code differs')
    need(args.smoke or (len(positions) == FORMAL_ROWS and manifest['settings'] == dict(dim=256, token_dim=256, hist_len=50, batch_size=256, candidates=75)), 'registered training contract differs')
    return r, cache, manifest, data, records, positions, pools


class TrainOnlyArray:
    def __init__(self, array, mask):
        self.array, self.mask = array, mask
    def __getitem__(self, index):
        if not np.all(self.mask[index]):
            raise PermissionError('future event array access forbidden')
        return self.array[index]
    def __array__(self, *args, **kwargs):
        raise PermissionError('unguarded full event array access forbidden')
    def __len__(self):
        return len(self.array)


def denied(*args, **kwargs):
    raise PermissionError('all target/evaluation access forbidden during pool construction')


def guard_training_data(data, positions):
    guarded = copy.copy(data)
    guarded.targets = denied
    guarded.evaluation = denied
    positions = np.asarray(positions, dtype=np.int64)
    need(len(positions) > 0 and np.all(positions[1:] > positions[:-1]) and np.all(data.train_mask[positions]), 'nonlegal or reordered training positions')
    membership = np.zeros(len(data.uid), dtype=bool)
    membership[positions] = True
    for name in ('iid', 'ts', 'rating'):
        setattr(guarded, name, TrainOnlyArray(getattr(data, name), data.train_mask))
    # Bind the original class method to the guarded copy, never the unguarded data.
    user_features = type(data).user_features.__get__(guarded, type(data))
    history_end = type(data).history_end.__get__(guarded, type(data))
    def checked_features(uid, pos):
        uid, pos = int(uid), int(pos)
        need(0 <= pos < len(membership) and membership[pos] and int(data.uid[pos]) == uid, 'V2 user features outside exact training records')
        end = history_end(uid, pos)
        need(int(data.starts[uid]) <= end <= pos < int(data.train_ends[uid]), 'training history boundary crossed')
        return user_features(uid, pos)
    guarded.user_features = checked_features
    return guarded


def disk_snapshot(path, remaining_bytes, smoke=False):
    p = safe(path)
    usage = shutil.disk_usage(p)
    fs = os.statvfs(p)
    reserve = 0 if smoke else DATA_RESERVE
    need(usage.free >= remaining_bytes + reserve, 'data disk reserve/remaining budget not met')
    need(fs.f_favail >= 16, 'insufficient free data inodes')
    return dict(path=str(p), filesystem_device=p.stat().st_dev, free_bytes=usage.free, free_inodes=fs.f_favail, remaining_bytes=remaining_bytes, reserve_bytes=reserve)


def environment():
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(), torch_threads=torch.get_num_threads(), cuda_available=torch.cuda.is_available(), matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, float32_matmul_precision=torch.get_float32_matmul_precision())


def source_for(args, cache, manifest, data, positions, records, mode):
    old = read(manifest['old_protocol']['path'])
    source = dict(base_data_id=old['base_data_id'], source_hashes=old['base_assets'], records_hash=cache.rows_hash(records), records_count=len(records), budget=75, candidate_policy='four-way RRF from train corpus; no future target filtering', rrf_weights=WEIGHTS, itemcf_half_life_days=180., smoke_fallback=bool(args.smoke), encoders_hash=old['encoders_hash'], code_hashes=old['code_hashes'], cf_neighbors=300, catalog_size=len(data.items), padding='zero tail, lengths define valid ordered IDs', protocol=VERSION, protocol_document=ref(args.protocol_doc), diagnostic_manifest=ref(args.protocol_manifest), legacy_source_hashes=manifest['code_hashes'], builder=ref(__file__), positions_sha256=hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest(), mode=mode, training_targets_read=False, final_test_allowed=False, full_training_allowed=False)
    return source


def build(args):
    output = safe(args.output_dir)
    need(not os.path.lexists(output), 'new output required; preserve earlier attempts')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    r, cache, manifest, data, _, positions, _ = legacy_inputs(args)
    need(args.smoke or torch.cuda.is_available(), 'formal V2 builder requires CUDA')
    all_positions = np.asarray(positions, dtype=np.int64)
    count = min(4096, len(all_positions)) if args.mode == 'pilot' else len(all_positions)
    selected = all_positions[:count].copy()
    records = cache.PositionRecords(data, selected)
    source = source_for(args, cache, manifest, data, selected, records, args.mode)
    pilot_ref = None
    if args.mode == 'full':
        need(args.pilot_receipt, 'verified resource pilot required')
        pilot = read(args.pilot_receipt)
        need(pilot['status'] == 'complete' and pilot['mode'] == 'pilot' and pilot['source']['builder'] == source['builder'] and pilot['source']['diagnostic_manifest'] == source['diagnostic_manifest'] and pilot['source']['protocol_document'] == source['protocol_document'] and pilot['source']['rrf_weights'] == WEIGHTS and pilot['rows'] == min(4096, len(all_positions)), 'pilot source/contract differs')
        pilot_tree = safe(args.pilot_receipt).parent
        for relative, expected in pilot['evidence_hashes'].items():
            need(sha(safe(pilot_tree / relative)) == expected, 'pilot evidence drift')
        pilot_ref = ref(args.pilot_receipt)
    expected_payload = count * (75 * 4 + 4 + 8) + 1024**2
    preflight = disk_snapshot(output.parent, expected_payload, args.smoke)
    output.mkdir()
    write(output / 'CONSTRUCTION_STARTED.json', dict(protocol=VERSION, source=source, mode=args.mode, pilot=pilot_ref, output_root=str(output), disk=preflight, environment=environment(), started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        guarded = guard_training_data(data, selected)
        if not args.smoke:
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        builder = r.RecallPoolBuilder(guarded, safe(manifest['base_run']), 75, WEIGHTS, args.smoke)
        initialization = time.monotonic() - started
        blocks = []
        position_path = output / 'ranker_train_positions.npy'
        with position_path.open('xb') as stream:
            np.save(stream, selected, allow_pickle=False)
        sidecar = output / ('candidate_pilot.json' if args.mode == 'pilot' else 'candidate_train_v2.json')
        def block(rows):
            began = time.monotonic()
            result = builder(rows)
            if not args.smoke:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - began
            blocks.append(dict(rows=len(rows), seconds=elapsed, rows_per_second=len(rows) / max(elapsed, 1e-12)))
            disk_snapshot(output, MAX_META, args.smoke)
            with (output / 'blocks.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(block=len(blocks), **blocks[-1])) + '\n')
            print(json.dumps(dict(mode=args.mode, completed_rows=sum(x['rows'] for x in blocks), total_rows=count, seconds=elapsed)), flush=True)
            return result
        started = time.monotonic()
        pools, meta = cache.build_cache(sidecar, records, 75, source, block, block_rows=1024)
        elapsed = time.monotonic() - started
        need(len(pools) == count and meta['source'] == source and meta['records_hash'] == source['records_hash'], 'completed cache differs')
        need({p.name: sha(p) for p in safe(args.legacy_source_dir).glob('*.py')} == manifest['code_hashes'] and ref(args.protocol_manifest) == source['diagnostic_manifest'] and ref(args.protocol_doc) == source['protocol_document'], 'source drift during construction')
        estimate = initialization + max(x['seconds'] / x['rows'] for x in blocks[-3:]) * len(all_positions) * 1.5
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(status='complete', protocol=VERSION, mode=args.mode, smoke=bool(args.smoke), rows=count, source=source, cache=ref(sidecar), positions=ref(position_path), pool_hash=meta['pool_hash'], pilot=pilot_ref, initialization_seconds=initialization, build_and_validation_seconds=elapsed, blocks=blocks, conservative_full_estimate_seconds=estimate, required_followup_reserve_seconds=3600, peak_host_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024), peak_gpu_allocated_bytes=0 if args.smoke else torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=0 if args.smoke else torch.cuda.max_memory_reserved(), environment=environment(), final_disk=disk_snapshot(output, 0, args.smoke), target_access_performed=False, scoring_performed=False, final_test_allowed=False, full_training_allowed=False, evidence_hashes={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        write(output / 'COMPLETED.json', result)
        return result
    except BaseException as error:
        write(output / 'FAILED.json', dict(status='failed', type=type(error).__name__, message=str(error), automatic_retry=False))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('pilot', 'full'))
    for name in ('legacy-source-dir', 'protocol-manifest', 'protocol-doc', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--pilot-receipt')
    p.add_argument('--smoke', action='store_true')
    return build(p.parse_args(argv))


if __name__ == '__main__':
    main()
