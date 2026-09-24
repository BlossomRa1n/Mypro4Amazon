#!/usr/bin/env python3
"""Read-only independent early-stop cross archive verification; never opens test.

Raw archive verification and external cohort/transfer authentication are separate
prerequisites. CPU checkpoint loading is restricted to trusted local artifacts.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import numpy as np
import torch
from verify_training_diagnostics_archive import (read_json, strict_json, sha,
    canonical_hash, require, close, load_arrays, verify_metrics)

VERSION = 'early-stop-zero-cross-20260925-v1'
FORMAL_PROTOCOL = 'c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6'
OLD_HASH = '9e2fbaf4e8ccb10e62cc3ec356d1f74f299517f898c7dbb2c6037348f616627d'
OLD_FILE = '1f437772b1e5418d5f09da60b97e437fc5079587f0c55296148ef43a6d9bb5a1'
DOC_HASH = '8d6dc4a421c84f09ed4dd626c2fbb144557b00a561cbf1de1be62baa87f6d710'
SEEDS = ('42', '43', '44')
STABLE = ('uid', 'position', 'candidate_ids', 'labels', 'lengths', 'pool_hit')


def safe_root(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_dir(), 'missing/symlink archive root')
    root = path.resolve()
    for entry in root.rglob('*'):
        require(not entry.is_symlink() and (entry.is_file() or entry.is_dir()), 'symlink or special archive entry')
        require(entry.name != 'FAILED.json', 'failed marker present')
    return root


def safe_file(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts and str(rel) == relative, 'unsafe evidence path')
    path = root / rel
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'missing or escaping evidence: ' + relative)
    return path


def sealed_files(root, hashes, exact=False):
    require(isinstance(hashes, dict) and hashes, 'empty evidence seal')
    files = {}
    for rel, bound in hashes.items():
        path = safe_file(root, rel)
        expected = bound['sha256'] if isinstance(bound, dict) else bound
        require(sha(path) == expected, 'evidence hash mismatch: ' + rel)
        files[rel] = dict(sha256=expected, size_bytes=path.stat().st_size)
        if isinstance(bound, dict) and 'size_bytes' in bound:
            require(bound['size_bytes'] == path.stat().st_size, 'evidence size mismatch')
        if path.suffix == '.json': read_json(path)
        if path.suffix == '.jsonl':
            for line in path.read_text().splitlines(): strict_json(line)
    if exact:
        require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == set(hashes)|{'COMPLETED.json'}, 'unsealed archive files')
    return files


def paired_equal(left, right):
    require(all(np.array_equal(left[k], right[k]) for k in STABLE), 'raw/zero paired identities/candidates/labels mismatch')


def recompute_paired(differences, paired, sources):
    require(paired.get('protocol') == VERSION and paired.get('final_test_allowed') is False, 'paired protocol/test policy mismatch')
    require(paired.get('comparison') == 'early_best_zero minus early_best_raw; already-development confirm'
            and set(paired['per_seed']) == set(SEEDS), 'paired comparison/seed mismatch')
    require(paired.get('bootstrap_seed') == 20260925 and paired.get('bootstrap_replicates') == 10000
            and paired.get('ci_level') == .975 and paired.get('quantiles') == [.0125, .9875], 'bootstrap registration mismatch')
    require(paired.get('array_sources') == sources, 'paired array source seal mismatch')
    recomputed = {}
    for metric, values in differences.items():
        average = np.mean(values, axis=0); expected = paired[metric]
        close(expected['mean_delta'], average.mean(), metric+' mean delta')
        require(expected['users'] == len(average) and expected['gained'] == int((average>0).sum())
                and expected['lost'] == int((average<0).sum()), 'paired aggregate gained/lost mismatch')
        rng = np.random.default_rng(20260925)
        samples = np.empty(10000)
        for index in range(10000):
            samples[index] = average[rng.integers(0, len(average), size=len(average))].mean()
        ci = np.quantile(samples, [.0125, .9875])
        recomputed[metric] = dict(mean=float(average.mean()), lower=float(ci[0]))
        require(len(expected['ci975']) == 2, 'CI shape mismatch')
        for actual, value in zip(expected['ci975'], ci): close(actual, value, metric+' independently recomputed CI975')
        for seed, delta in zip(SEEDS, values):
            entry = paired['per_seed'][seed]
            close(entry[metric+'_delta'], delta.mean(), metric+' seed delta')
            if metric == 'hit5':
                require(entry['gained'] == int((delta>0).sum()) and entry['lost'] == int((delta<0).sum()), 'seed gained/lost mismatch')
    criteria = dict(all_seed_hr_positive=all(float(row.mean())>0 for row in differences['hit5']),
                    mean_hr_at_least_0_0005=recomputed['hit5']['mean']>=.0005,
                    hr_ci975_lower_positive=recomputed['hit5']['lower']>0,
                    mean_ndcg_nonnegative=recomputed['ndcg5']['mean']>=0)
    require(paired['criteria'] == criteria, 'selection criteria mismatch')
    require(paired['selection'] == ('zero_cross' if all(criteria.values()) else 'raw_retained'), 'selection result mismatch')


def verify(args):
    run = safe_root(args.run_dir); raw = safe_root(args.raw_run); old = safe_root(args.old_zero_run)
    receipt_path = Path(args.receipt).resolve()
    require(not receipt_path.exists(), 'refuse to overwrite receipt')
    require(all(not receipt_path.is_relative_to(root) for root in (run, raw, old)), 'receipt inside immutable input')
    protocol_path = Path(args.protocol_manifest); protocol = read_json(protocol_path)
    require(protocol['manifest_hash'] == canonical_hash({k:v for k,v in protocol.items() if k!='manifest_hash'}), 'protocol seal mismatch')
    require(protocol.get('protocol') == 'raw-deterioration-20260924-v1' and protocol.get('test_evaluation_allowed') is False, 'protocol identity/test policy mismatch')
    complete = read_json(run/'COMPLETED.json'); smoke = complete.get('smoke')
    require(type(smoke) is bool and smoke == protocol.get('smoke'), 'smoke identity mismatch')
    require(not smoke or args.allow_smoke, 'smoke cannot pass formal gate')
    require(smoke or protocol['manifest_hash'] == FORMAL_PROTOCOL, 'unregistered formal protocol')
    require(complete.get('status') == 'complete' and complete.get('protocol') == VERSION and complete.get('test_future_labels_read') is False, 'incomplete/test-exposed archive')
    identity = protocol['test_identity_receipt']
    require(identity.get('future_labels_read') is False and (smoke or identity.get('count') == 100000), 'sealed test identity mismatch')
    files = sealed_files(run, complete['evidence_hashes'], exact=True)
    def evidence(rel):
        require(rel in files, 'required evidence not sealed: '+rel)
        return safe_file(run, rel)
    raw_complete = read_json(raw/'COMPLETED.json'); raw_gate = read_json(args.raw_archive_receipt)
    require(raw_gate.get('status') == 'verified' and raw_gate.get('smoke') == smoke
            and raw_gate.get('formal_archive_verified') == (not smoke), 'raw archive receipt status mismatch')
    require(raw_gate.get('completion_sha256') == sha(raw/'COMPLETED.json')
            and raw_gate.get('protocol_manifest_hash') == protocol['manifest_hash']
            and raw_gate.get('protocol_manifest_sha256') == sha(protocol_path), 'raw archive receipt binding mismatch')
    require(raw_complete.get('status') == 'complete' and raw_complete.get('test_future_labels_read') is False
            and raw_complete.get('protocol_manifest_hash') == protocol['manifest_hash'], 'raw control protocol mismatch')
    def raw_evidence(rel):
        path = safe_file(raw, rel)
        require(rel in raw_complete['evidence_hashes'] and rel in raw_gate['evidence'], 'raw evidence unsealed')
        digest = sha(path)
        require(digest == raw_complete['evidence_hashes'][rel] == raw_gate['evidence'][rel]['sha256'], 'raw evidence binding mismatch')
        return path
    source_path = Path(args.source_manifest).resolve(); sources = read_json(source_path)
    source_dir = safe_root(source_path.parent/'src')
    require({str(p.relative_to(source_dir)) for p in source_dir.rglob('*') if p.is_file()} == set(sources), 'source coverage mismatch')
    sealed_files(source_dir, sources)
    manifest = read_json(evidence('run_manifest.json'))
    require(manifest.get('protocol') == VERSION and manifest.get('smoke') == smoke
            and manifest.get('test_future_labels_read') is False, 'run manifest identity mismatch')
    require(manifest['legacy_protocol_manifest_hash'] == protocol['manifest_hash']
            and manifest['legacy_source_hashes'] == protocol['code_hashes'], 'legacy protocol/source mismatch')
    require(manifest['runner_sha256'] == sources['run_early_stop_cross.py']
            and manifest['protocol_doc_sha256'] == sources['OVERNIGHT_CROSS_PROTOCOL_20260925.md'] == DOC_HASH, 'runner/document source binding mismatch')
    if not smoke:
        require(manifest['deadline_utc'] == '2026-09-25T00:04:49+00:00'
                and datetime.fromisoformat(manifest['started_at_utc']) < datetime.fromisoformat(manifest['deadline_utc']), 'run began past registered deadline')
        require(sources['archive_verified.json'] == sha(args.raw_archive_receipt), 'source raw receipt binding mismatch')
    controls = manifest['controls']
    require(controls['raw_completion'] == sha(raw/'COMPLETED.json') and controls['raw_locked'] == sha(raw_evidence('CHECKPOINTS_LOCKED.json')), 'raw control hash mismatch')
    for rel, digest in controls['used_raw_files'].items(): require(sha(raw_evidence(rel)) == digest, 'used raw file mismatch')
    if not smoke:
        require(controls['archive_receipt_sha256'] == sha(args.raw_archive_receipt), 'raw receipt control mismatch')
        backup = read_json(source_dir/'RECOVERY_BACKUP_BINDING.json')
        require(backup.get('status') == 'verified' and backup['raw_completion_sha256'] == controls['raw_completion']
                and backup['archive_receipt_sha256'] == controls['archive_receipt_sha256']
                and sha(source_dir/'RECOVERY_BACKUP_BINDING.json') == controls['backup_receipt_sha256'], 'recoverable backup binding mismatch')
    old_manifest = read_json(old/'dev_manifest.json')
    require(old_manifest['manifest_hash'] == canonical_hash({k:v for k,v in old_manifest.items() if k!='manifest_hash'}), 'old dev canonical hash mismatch')
    require(sha(old/'dev_manifest.json') == controls['old_dev_manifest_sha256'], 'old dev control binding mismatch')
    if not smoke:
        require(old_manifest['manifest_hash'] == OLD_HASH and sha(old/'dev_manifest.json') == OLD_FILE
                and old_manifest.get('protocol_manifest_hash') == 'dcf65b10c266ba02b00d0493815a14b8af9aa4f47551d7c562a2010104409667'
                and old_manifest.get('test_accessed') is False, 'unregistered old zero control')
    config = manifest['config']; settings = protocol['settings']; raw_config = read_json(raw_evidence('run_config.json'))
    require(config['epochs'] == 1 and config['smoke'] == smoke and manifest['scheduler_t_max'] == 3, 'early epoch/scheduler mismatch')
    for key in ('dim', 'token_dim', 'hist_len', 'batch_size', 'candidates'): require(config[key] == settings[key], 'run configuration mismatch: '+key)
    if not smoke:
        require(settings == dict(dim=256, token_dim=256, hist_len=50, batch_size=256, candidates=75) and config['device']=='cuda', 'formal run configuration mismatch')
        for key in ('torch', 'numpy', 'cuda_runtime', 'cudnn', 'torch_threads'):
            require(manifest['environment'][key] == raw_config['environment'][key], 'environment mismatch: '+key)
        require(manifest['environment']['python'].split()[0] == raw_config['environment']['python'].split()[0], 'Python mismatch')
    panels = read_json(evidence('panels.json'))
    require(panels == read_json(raw_evidence('panels.json')), 'raw/zero panel mismatch')
    require(panels['identity_sha256'] == canonical_hash({k:v for k,v in panels.items() if k!='identity_sha256'}), 'panels seal mismatch')
    disk = read_json(evidence('disk_preflight.json'))
    require(disk['free_bytes']>=disk['required_bytes'] and disk['retained_models']==3 and disk['atomic_peak_models']==4, 'disk preflight failed')
    locked = read_json(evidence('CHECKPOINTS_LOCKED.json')); raw_locked = read_json(raw_evidence('CHECKPOINTS_LOCKED.json'))
    require(set(complete['seeds']) == set(locked) == set(SEEDS), 'seed coverage mismatch')
    counts = dict(checkpoints=0, screen_arrays=0, confirm_arrays=0)
    differences = {'hit5':[], 'ndcg5':[]}; array_sources={}; confirm_reference=None; screen_reference=None
    expected_files={'run_manifest.json','panels.json','disk_preflight.json','CHECKPOINTS_LOCKED.json','paired_confirm.json'}
    for seed in SEEDS:
        prefix=f'seed_{seed}/zero_cross/'; rp=f'seed_{seed}/raw/'
        fixed=('COMPLETED.json','initial_state_audit.json','epoch_1_trace.json','history.json','steps.jsonl','best.pth','confirm_best_users.npz','confirm_best_metrics.json')
        expected_files.update(prefix+name for name in fixed)
        result=read_json(evidence(prefix+'COMPLETED.json'))
        require(result == complete['seeds'][seed], 'seed completion mismatch')
        require(result['variant']=='zero_cross' and result['seed']==int(seed) and result['init_seed']==424242+int(seed)-42
                and result['precision']=='FP32' and result['protocol']==VERSION and result['automatic_resume'] is False and result['optimizer_saved'] is False, 'seed training identity mismatch')
        require(result['protocol_manifest_hash']==result['legacy_protocol_manifest_hash']==protocol['manifest_hash'], 'seed protocol mismatch')
        raw_result=raw_complete['seeds'][seed]; steps=result['steps_per_epoch']
        require(type(steps) is int and steps>0 and steps==raw_result['steps_per_epoch'] and (smoke or steps==2362), 'epoch step count mismatch')
        model_config=result['model_config']; reference=dict(raw_result['model_config'],cross_mode='zero_cross')
        require(model_config==reference, 'raw/zero model configuration mismatch')
        audit=read_json(evidence(prefix+'initial_state_audit.json')); raw_audit=read_json(raw_evidence(rp+'initial_state_audit.json'))
        require(audit == raw_audit and audit['gate_value']==10. and audit['gate_difference_whitelist']==['cross_gate_logit'], 'initial state differs from raw')
        trace=read_json(evidence(prefix+'epoch_1_trace.json')); raw_trace=read_json(raw_evidence(rp+'epoch_1_trace.json'))
        old_rel=prefix+'epoch_1_trace.json'; old_path=safe_file(old,old_rel)
        require(sha(old_path)==old_manifest['evidence_hashes'][old_rel]==controls['old_zero_traces'][old_rel], 'old trace seal mismatch')
        require(trace==raw_trace==read_json(old_path)==result['traces']['1'] and set(result['traces'])=={'1'}, 'epoch-one trace mismatch')
        require(len(trace['batch_sizes'])==steps and sum(trace['batch_sizes'])==trace['rows'], 'trace batch/rows mismatch')
        require(sum(trace['negative_sources'].values())==trace['rows']*16, 'negative stream count mismatch')
        require(smoke or (trace['rows']==604511 and trace['covered_users']==100000), 'formal training coverage mismatch')
        step_rows=[strict_json(line) for line in evidence(prefix+'steps.jsonl').read_text().splitlines()]
        require([entry['step'] for entry in step_rows]==list(range(1,steps+1))
                and [entry['batch_rows'] for entry in step_rows]==trace['batch_sizes']
                and all(entry['epoch']==1 and entry['loss']>=0 and entry['gradient_l2_before_clip']>=0 and entry['lr']==.001 for entry in step_rows), 'training step sequence mismatch')
        history=read_json(evidence(prefix+'history.json')); raw_history=read_json(raw_evidence(rp+'history.json'))
        early=[entry for entry in raw_history if entry['step']<=steps]
        require([point for entry in history for point in entry['requested_points']]==[.25,.5,1.]
                and [entry['step'] for entry in history]==sorted(set(math.ceil(point*steps) for point in (.25,.5,1.))), 'screen schedule mismatch')
        best=max(history,key=lambda entry:(entry['screen']['hr5'],entry['screen']['ndcg5'],-entry['step']))
        raw_best=max(early,key=lambda entry:(entry['screen']['hr5'],entry['screen']['ndcg5'],-entry['step']))
        require(raw_best['step']==raw_locked[seed]['best']['step'], 'raw archived best differs from early-best')
        if not smoke: require(raw_best['step']=={'42':591,'43':1181,'44':1181}[seed], 'raw registered best mismatch')
        for entry in history:
            close(entry['epoch_fraction'],entry['step']/steps,'epoch fraction')
            name=f"screen_step{entry['step']}_users.npz";expected_files.add(prefix+name)
            arrays=load_arrays(evidence(prefix+name));verify_metrics(arrays,entry['screen'])
            control=load_arrays(raw_evidence(rp+name));paired_equal(arrays,control)
            raw_entry=next(item for item in early if item['step']==entry['step']);verify_metrics(control,raw_entry['screen'])
            require(arrays['candidate_ids'].shape[1]<=settings['candidates'] and (smoke or len(arrays['uid'])==20000), 'screen count/budget mismatch')
            require(np.all(arrays['candidate_ids']<model_config['num_items']) and np.all(arrays['uid']<model_config['num_users']), 'screen ID range mismatch')
            if screen_reference is None: screen_reference={key:arrays[key] for key in STABLE}
            else: paired_equal(arrays,screen_reference)
            invariance=entry['diagnostics']['candidate_batch_invariance']
            require(invariance['passed'] is True and invariance['atol']==1e-5 and invariance['rtol']==1e-5, 'candidate batch invariance failed')
            counts['screen_arrays']+=1
        metadata=locked[seed];require(metadata==result['best'], 'locked best mismatch')
        require(metadata['step']==best['step'] and metadata['screen']==best['screen'] and metadata['epoch_fraction']==best['epoch_fraction'], 'checkpoint selection mismatch')
        for key in ('protocol','legacy_protocol_manifest_hash','variant','seed','init_seed','model_config','protocol_manifest_hash','automatic_resume','optimizer_saved','precision'):
            require(metadata[key]==result[key], 'checkpoint identity mismatch: '+key)
        raw_checkpoint=raw_evidence(rp+'best.pth')
        raw_metadata=raw_locked[seed]['best']
        require(sha(raw_checkpoint)==raw_metadata['checkpoint_sha256'], 'raw locked checkpoint SHA mismatch')
        raw_payload=torch.load(raw_checkpoint,map_location='cpu',weights_only=False)
        require(raw_payload['manifest']=={key:value for key,value in raw_metadata.items() if key!='checkpoint_sha256'}, 'raw checkpoint embedded manifest mismatch')
        require(all(torch.is_tensor(tensor) for tensor in raw_payload['model'].values()), 'raw checkpoint non-tensor value')
        raw_specs={key:(tuple(tensor.shape),tensor.dtype) for key,tensor in raw_payload['model'].items()}
        del raw_payload
        checkpoint=evidence(prefix+'best.pth');require(sha(checkpoint)==metadata['checkpoint_sha256'], 'locked checkpoint SHA mismatch')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        require(payload['manifest']=={key:value for key,value in metadata.items() if key!='checkpoint_sha256'}, 'checkpoint embedded manifest mismatch')
        require(set(payload['model'])==set(audit['copied_tensor_keys']) and all(torch.is_tensor(t) and torch.isfinite(t).all().item() for t in payload['model'].values()), 'checkpoint tensor keys/nonfinite mismatch')
        require({key:(tuple(tensor.shape),tensor.dtype) for key,tensor in payload['model'].items()}==raw_specs, 'checkpoint tensor shape/dtype differs from sealed raw best')
        del payload;counts['checkpoints']+=1
        zero_path=evidence(prefix+'confirm_best_users.npz');raw_path=raw_evidence(rp+'confirm_best_users.npz')
        zero=load_arrays(zero_path);control=load_arrays(raw_path);paired_equal(zero,control)
        verify_metrics(zero,read_json(evidence(prefix+'confirm_best_metrics.json')))
        verify_metrics(control,read_json(raw_evidence(rp+'confirm_best_metrics.json')))
        require(zero['candidate_ids'].shape[1]<=settings['candidates'] and (smoke or len(zero['uid'])==80000), 'confirm count/budget mismatch')
        require(np.all(zero['candidate_ids']<model_config['num_items']) and np.all(zero['uid']<model_config['num_users']), 'confirm ID range mismatch')
        if confirm_reference is None:confirm_reference={key:zero[key] for key in STABLE}
        else:paired_equal(zero,confirm_reference)
        counts['confirm_arrays']+=1
        for metric in differences:differences[metric].append(zero[metric].astype(np.float64)-control[metric].astype(np.float64))
        array_sources[seed]=dict(raw_arrays_sha256=sha(raw_path),zero_arrays_sha256=sha(zero_path))
    require(set(files)==expected_files, 'unexpected/missing early-cross evidence coverage')
    require(counts['checkpoints']==3 and counts['confirm_arrays']==3 and (smoke or counts['screen_arrays']==9), 'archive count mismatch')
    paired=read_json(evidence('paired_confirm.json'));require(paired==complete['paired'], 'paired completion mismatch')
    recompute_paired(differences,paired,array_sources)
    receipt=dict(status='verified',formal_archive_verified=not smoke,smoke=smoke,counts=counts,
                 run_dir=str(run),completion_sha256=sha(run/'COMPLETED.json'),cross_completion_sha256=sha(run/'COMPLETED.json'),protocol_manifest_hash=protocol['manifest_hash'],
                 protocol_manifest_sha256=sha(protocol_path),source_manifest_sha256=sha(source_path),
                 raw_archive_receipt_sha256=sha(args.raw_archive_receipt),old_dev_manifest_sha256=sha(old/'dev_manifest.json'),
                 evidence=files,bootstrap_independently_recomputed=True,selection=paired['selection'],test_future_labels_read=False,
                 limitations=['Original cohort UID/order authentication and remote transfer verification remain separate gates.',
                              'Per-user NDCG denominators are not reopened; finite bounds, pairing, aggregate differences and bootstrap are verified.',
                              'Initialization/stream audits are matched to sealed controls; training is not replayed.'])
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    with receipt_path.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','raw-run','old-zero-run','protocol-manifest','raw-archive-receipt','source-manifest','receipt'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--allow-smoke',action='store_true')
    result=verify(parser.parse_args(argv));print(json.dumps(dict(status=result['status'],counts=result['counts'],selection=result['selection'])))


if __name__=='__main__':main()
