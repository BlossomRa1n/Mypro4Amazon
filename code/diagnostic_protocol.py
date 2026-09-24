"""Independent raw-training diagnosis; an identity-only 100k test reservation.

Old protocol files and candidate provenance are preserved. This module exposes
no test evaluation command and never reads future targets while preparing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import run_cross_multiseed as r
from cross_pool_cache import PositionRecords, load_cache, rows_hash

VERSION = 'raw-deterioration-20260924-v1'
LOCK_REL = 'server_snapshot/test100k_feasibility_20260924/locked_test100k/test100k_manifest.json'
MONTH_REL = 'server_snapshot/next_cross_20260924/history_audit/monthly_20260924'
LOCK_SHA = '9416c80d3c2bda822239e19a1ffd5cc8b0497c1aca252225dee26329a9cf4303'
OLD_SHA = '4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1'


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, digest):
    path = Path(path)
    if r.sha256_file(path) != digest:
        raise ValueError('file SHA-256 drift: ' + str(path))
    return path


def evidence_refs(root, value):
    """Verify exact referenced files, including nested audit/proof references."""
    root = Path(root).resolve()
    found = {}
    def visit(obj):
        if isinstance(obj, dict):
            if 'path' in obj and 'sha256' in obj:
                rel = obj['path']
                path = root / (rel if rel.startswith('server_snapshot/') else MONTH_REL + '/' + rel)
                if not path.resolve().is_relative_to(root):
                    raise ValueError('evidence reference escapes root')
                checked(path, obj['sha256'])
                found[str(path.relative_to(root))] = obj['sha256']
            for child in obj.values(): visit(child)
        elif isinstance(obj, list):
            for child in obj: visit(child)
    visit(value)
    return found


def verify_lock(root, data, records, smoke=False, smoke_lock=None):
    root = Path(root)
    path = Path(smoke_lock) if smoke else root / LOCK_REL
    if not smoke: checked(path, LOCK_SHA)
    lock = read(path)
    if lock.get('formal_evaluation_allowed') is not False:
        raise ValueError('test must remain sealed')
    if lock.get('base_data_id') != data.manifest['data_id']:
        raise ValueError('test base data mismatch')
    refs = evidence_refs(root, lock)
    registry = read(root / lock['source_registry']['path'])
    refs.update(evidence_refs(root, registry))
    audit = read(root / lock['independent_stageab_audit']['path'])
    refs.update(evidence_refs(root, audit))
    if registry.get('unresolved_sources_within_reviewed_scope') != []:
        raise ValueError('unresolved test history sources')
    if not smoke:
        if lock.get('status') != 'identity_frozen_for_next_protocol' or lock['count'] != 100000:
            raise ValueError('fixed 100k test required')
        if len(registry['sources']) != registry.get('source_count') or registry['source_count'] != 298:
            raise ValueError('test source registry incomplete')
        old_registry_path = root / MONTH_REL / 'source_registry_month_20260924.json'
        refs.update(evidence_refs(root, read(old_registry_path)))
        if audit.get('status') != 'deterministic_reconstruction_supported':
            raise ValueError('StageAB reconstruction audit missing')
    excluded = read(root / lock['exclusions']['path'])
    excluded_ids = excluded['raw_user_ids']
    if len(excluded_ids) != excluded['count'] or len(set(excluded_ids)) != len(excluded_ids):
        raise ValueError('exclusion identity count mismatch')
    if r.sha256_json(sorted(excluded_ids)) != excluded['raw_user_ids_sha256']:
        raise ValueError('exclusion identity hash mismatch')
    test = [tuple(row) for row in read(root / lock['records']['path'])]
    if len(test) != lock['count'] or rows_hash(test) != lock['cohort_records_hash']:
        raise ValueError('locked test records mismatch')
    raw = [row[2] for row in test]; uids = [row[0] for row in test]
    if len(set(raw)) != len(test) or len(set(uids)) != len(test):
        raise ValueError('duplicate locked test identity')
    if r.sha256_json(sorted(raw)) != lock['cohort_raw_hash'] or set(raw) & set(excluded_ids):
        raise ValueError('test history exclusion mismatch')
    eligible, _ = r._eligible_records(data, set())
    mapping = {uid: (position, name) for uid, position, name in eligible}
    for label, values in {**records, 'locked_test': test}.items():
        if any(mapping.get(uid) != (position, name) for uid, position, name in values):
            raise ValueError('identity/eligibility mismatch: ' + label)
    train_raw = {row[2] for row in records['train']}
    if set(raw) & train_raw or set(uids) & {row[0] for row in records['train']}:
        raise ValueError('locked test overlaps development')
    available = sorted({row[2] for row in eligible} - set(excluded_ids))
    if len(available) != lock['eligible_available_population']:
        raise ValueError('test available population count drift')
    if r.sha256_json(available) != lock['population_sorted_raw_ids_sha256']:
        raise ValueError('test eligible population drift')
    expected = sorted(np.random.default_rng(lock['cohort_seed']).choice(np.asarray(available), len(test), replace=False).tolist())
    if raw != expected: raise ValueError('locked test sampling drift')
    return test, {'lock_sha256': r.sha256_file(path), 'verified_evidence': refs,
                  'count': len(test), 'future_labels_read': False}


def temporal_audit(data, positions, records):
    """Recompute fitted statistics using only the immutable legal train mask."""
    mask = np.arange(len(data.uid)) < data.train_ends[data.uid]
    if not np.array_equal(mask, data.train_mask): raise ValueError('train mask/window drift')
    if np.any(positions >= data.train_ends[data.uid[positions]]):
        raise ValueError('held-out training target detected')
    ids = data.iid[mask]
    counts = np.bincount(ids, minlength=len(data.items))
    if not np.array_equal(counts, data.train_counts): raise ValueError('item counts not fitted on legal train prefix')
    expected_pop = (np.log1p(counts) / max(np.log1p(counts.max()),1)).astype(np.float32)
    if not np.array_equal(expected_pop, data.item_pop): raise ValueError('item popularity drift')
    expected_avg = np.divide(np.bincount(ids, weights=data.rating[mask], minlength=len(data.items)),
                             5 * counts, out=np.zeros(len(data.items)), where=counts>0).astype(np.float32)
    if not np.allclose(expected_avg, data.item_avg, rtol=1e-6, atol=1e-7):
        raise ValueError('item average not fitted on legal train prefix')
    created = np.full(len(data.items), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(created,ids,data.ts[mask])
    expected_created = np.zeros(len(data.items),dtype=np.float32)
    active = counts > 0
    expected_created[active] = (created[active]-data.time_origin)/data.time_scale
    if not np.array_equal(expected_created,data.item_created): raise ValueError('item creation statistic drift')
    if not np.array_equal(np.flatnonzero(active),data.active_items): raise ValueError('active catalog drift')
    # Fixed spacing avoids consuming any training RNG. Timestamp-only probe;
    # targets() and held-out item identities are not read.
    probe_positions = positions[np.linspace(0,len(positions)-1,min(1024,len(positions)),dtype=int)]
    probes = [(int(data.uid[p]),int(p)) for p in probe_positions]
    probes += [(int(u),int(p)) for label in ('screen','confirm') for u,p,_ in records[label][:512]]
    for uid,pos in probes:
        end = int(data.history_end(uid,pos)); start = int(data.starts[uid])
        if not start <= end <= pos or np.any(data.ts[start:end] >= data.ts[pos]):
            raise ValueError('history includes non-earlier timestamps')
    return dict(status='passed_with_registered_transductive_limits', train_rows=len(positions),
                history_boundary_probes=len(probes), train_mask_sha256=r.sha256_array(mask),
                item_statistics_recomputed_from_train_only=True,
                strict_per_row_causal=False, strict_global_calendar_causal=False,
                training_negative_exclusions='full legal user train_sets, including later training positives',
                recall_svd_v2='original full legal-prefix assets SHA-bound, not re-fitted per training row',
                test_targets_read=False)


def load_inputs(path):
    path = Path(path); m = read(path)
    if m.get('protocol') != VERSION or m.get('manifest_hash') != r.sha256_json({k:v for k,v in m.items() if k != 'manifest_hash'}):
        raise ValueError('diagnostic manifest mismatch')
    if type(m.get('smoke')) is not bool: raise ValueError('invalid smoke flag')
    if m.get('test_evaluation_allowed') is not False: raise ValueError('test scoring must remain forbidden')
    if m['code_hashes'] != r.code_hashes(): raise ValueError('diagnostic code drift')
    checked(m['protocol_document']['path'], m['protocol_document']['sha256'])
    old_path = checked(m['old_protocol']['path'], m['old_protocol']['sha256'])
    if not m['smoke'] and r.sha256_file(old_path) != OLD_SHA: raise ValueError('unregistered old protocol')
    old = read(old_path)
    if old['manifest_hash'] != r.sha256_json({k:v for k,v in old.items() if k != 'manifest_hash'}):
        raise ValueError('old protocol hash mismatch')
    for name, digest in old['base_assets'].items(): checked(Path(m['base_run']) / name, digest)
    data = r.load_data(Path(m['base_run']))
    if data.manifest['data_id'] != old['base_data_id'] or data.manifest['encoders_hash'] != old['encoders_hash']:
        raise ValueError('base identity drift')
    records = {label: r._records(old_path.parent, old, label) for label in ('train','screen','confirm')}
    sets = {label: {row[0] for row in rows} for label, rows in records.items()}
    if sets['train'] != sets['screen'] | sets['confirm'] or sets['screen'] & sets['confirm']:
        raise ValueError('development cohort overlap/union mismatch')
    if not m['smoke'] and {k:len(v) for k,v in records.items()} != r.FORMAL_COUNTS:
        raise ValueError('formal development counts changed')
    positions = np.load(old_path.parent / old['ranker_train_positions'], mmap_mode='r', allow_pickle=False)
    expected = np.asarray(data.train_positions)[np.isin(np.asarray(data.uid)[data.train_positions], list(sets['train']))]
    if not np.array_equal(positions, expected) or r.sha256_array(positions) != old['ranker_train_positions_sha256']:
        raise ValueError('training positions drift')
    if temporal_audit(data,positions,records) != m['temporal_audit']:
        raise ValueError('temporal audit drift')
    test, receipt = verify_lock(m['evidence_root'], data, records, m['smoke'], m.get('smoke_lock'))
    if receipt != m['test_identity_receipt']: raise ValueError('test evidence receipt drift')
    if not m['smoke'] and m['settings'] != dict(dim=256,token_dim=256,hist_len=50,batch_size=256,candidates=75):
        raise ValueError('formal settings drift')
    pools = {}
    for label in records:
        cache_path = checked(m['candidate_caches'][label]['path'], m['candidate_caches'][label]['sha256'])
        pool, metadata = load_cache(cache_path)
        source = metadata['source']
        rows = PositionRecords(data, positions) if label == 'train' else records[label]
        expected_weights = [2.,0.,.7,.05] if label == 'train' else [2.,1.,.7,.05]
        checks = {'base_data_id':old['base_data_id'], 'encoders_hash':old['encoders_hash'],
                  'source_hashes':old['base_assets'], 'code_hashes':old['code_hashes'],
                  'records_hash':rows_hash(rows), 'records_count':len(rows), 'budget':m['settings']['candidates'],
                  'rrf_weights':expected_weights, 'cf_neighbors':300, 'itemcf_half_life_days':180.,
                  'smoke_fallback':m['smoke'], 'catalog_size':len(data.items)}
        if any(source.get(key) != value for key,value in checks.items()):
            raise ValueError('reused candidate provenance mismatch: ' + label)
        pools[label] = pool
    locked_uids = {int(row[0]) for row in test}; original = data.targets
    def guarded(rows):
        rows = list(rows)
        if any(int(row[0]) in locked_uids for row in rows): raise PermissionError('sealed test targets cannot be read')
        return original(rows)
    data.targets = guarded
    return m, data, records, positions, pools


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists(): raise FileExistsError('preserve existing protocol output')
    old_path = Path(args.old_protocol).resolve(); old = read(old_path)
    if not args.smoke: checked(old_path, OLD_SHA)
    data = r.load_data(Path(args.base_run))
    records = {label:r._records(old_path.parent,old,label) for label in ('train','screen','confirm')}
    positions = np.load(old_path.parent / old['ranker_train_positions'],mmap_mode='r',allow_pickle=False)
    _, receipt = verify_lock(args.evidence_root, data, records, args.smoke, getattr(args,'smoke_lock',None))
    settings = dict(dim=16,token_dim=16,hist_len=data.hist_len,batch_size=8,candidates=75) if args.smoke else dict(dim=256,token_dim=256,hist_len=50,batch_size=256,candidates=75)
    m = dict(protocol=VERSION, smoke=bool(args.smoke), base_run=str(Path(args.base_run).resolve()),
             old_protocol=dict(path=str(old_path),sha256=r.sha256_file(old_path)),
             evidence_root=str(Path(args.evidence_root).resolve()), test_identity_receipt=receipt,
             temporal_audit=temporal_audit(data,positions,records),
             settings=settings, code_hashes=r.code_hashes(),
             candidate_caches={label:dict(path=str(old_path.parent/f'candidate_{label}.json'),sha256=r.sha256_file(old_path.parent/f'candidate_{label}.json')) for label in records},
             test_evaluation_allowed=False, training_temporal_policy='full legal train-prefix assets; not strict per-row or global-calendar causal',
             reused_pool_provenance='original cache source hashes retained; old code and new diagnostic code separately bound')
    document = Path(__file__).resolve().parents[1] / 'docs/TRAINING_DIAGNOSTICS_PROTOCOL_20260924.md'
    m['protocol_document'] = dict(path=str(document), sha256=r.sha256_file(document))
    if args.smoke: m['smoke_lock'] = str(Path(args.smoke_lock).resolve())
    m['manifest_hash'] = r.sha256_json(m)
    output.parent.mkdir(parents=True,exist_ok=True)
    r.write_json(output,m)
    load_inputs(output)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-protocol',required=True);p.add_argument('--base-run',required=True)
    p.add_argument('--evidence-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-lock')
    m=prepare(p.parse_args());print(json.dumps({'status':'validated','protocol':m['protocol'],'manifest_hash':m['manifest_hash']}))


if __name__ == '__main__': main()
