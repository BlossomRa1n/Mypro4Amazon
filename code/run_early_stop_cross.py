"""Frozen early-stop zero-cross development comparison; no final test entry point.

Imports original dependencies from a separately sealed legacy source directory.
Only this new runner is added; original protocol/code hashes remain immutable.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

VERSION = 'early-stop-zero-cross-20260925-v1'

DEADLINE = datetime(2026, 9, 25, 0, 4, 49, tzinfo=timezone.utc)

def read(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    def finite(obj):
        if isinstance(obj, float) and not math.isfinite(obj): raise ValueError('nonfinite JSON number')
        if isinstance(obj, dict):
            for child in obj.values(): finite(child)
        elif isinstance(obj, list):
            for child in obj: finite(child)
    finite(value)
    return value

def check_deadline(now, smoke=False):
    if now.tzinfo is None: raise ValueError('timezone-aware start required')
    if not smoke and now >= DEADLINE: raise ValueError('registered overnight start deadline expired')

def check_environment(expected, actual):
    for key in ('torch','numpy','cuda_runtime','cudnn','torch_threads'):
        if actual.get(key) != expected.get(key): raise ValueError('raw/zero environment mismatch: '+key)
    if actual['python'].split()[0] != expected['python'].split()[0]: raise ValueError('raw/zero Python mismatch')

def validate_old_manifest(value, file_hash, smoke):
    if value.get('manifest_hash') != r.sha256_json({key:item for key,item in value.items() if key!='manifest_hash'}): raise ValueError('old dev manifest seal mismatch')
    if not smoke and (value['manifest_hash']!='9e2fbaf4e8ccb10e62cc3ec356d1f74f299517f898c7dbb2c6037348f616627d' or file_hash!='1f437772b1e5418d5f09da60b97e437fc5079587f0c55296148ef43a6d9bb5a1' or value.get('protocol_manifest_hash')!='dcf65b10c266ba02b00d0493815a14b8af9aa4f47551d7c562a2010104409667' or value.get('test_accessed') is not False): raise ValueError('unregistered old dev control')

def validate_backup_binding(backup, raw_hash, archive_hash):
    if backup.get('status')!='verified' or backup.get('raw_completion_sha256')!=raw_hash or backup.get('archive_receipt_sha256')!=archive_hash: raise ValueError('recoverable backup gate binding failed')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def legacy(args):
    directory = Path(args.legacy_source_dir).resolve()
    manifest = read(args.protocol_manifest)
    actual = {path.name:sha(path) for path in directory.glob('*.py')}
    if actual != manifest['code_hashes']: raise ValueError('legacy source directory is not exact sealed code')
    for name in ('run_cross_multiseed', 'run_training_diagnostics', 'diagnostic_protocol', 'diagnostic_metrics'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent != directory: raise ValueError('conflicting dependency already imported: '+name)
    sys.path.insert(0, str(directory))
    global r, d, protocol, collate, seed_all, preserve_training_state, evaluate_candidates, panel_diagnostics, checked_save, selection_key, USER_KEYS, ITEM_KEYS
    r = importlib.import_module('run_cross_multiseed'); d = importlib.import_module('run_training_diagnostics'); protocol = importlib.import_module('diagnostic_protocol')
    collate=d.collate; seed_all=d.seed_all; preserve_training_state=d.preserve_training_state; evaluate_candidates=d.evaluate_candidates
    panel_diagnostics=d.panel_diagnostics; checked_save=d.checked_save; selection_key=d.selection_key; USER_KEYS=d.USER_KEYS; ITEM_KEYS=d.ITEM_KEYS
    return manifest

def train_seed(data, args, directory, seed, state, positions, pools, records, panels, manifest):
    directory.mkdir(parents=True)
    model = r._new_variant(data, args, state, 'zero_cross')
    audit = {'state_sha256': r._state_hashes(model), 'template_sha256': {key: r.sha256_array(value.numpy()) for key, value in state.items()}, 'copied_tensor_keys': sorted(state), 'gate_difference_whitelist': ['cross_gate_logit'], 'gate_value': float(model.cross_gate_logit.detach())}
    for key, value in model.state_dict().items():
        if key != 'cross_gate_logit' and not torch.equal(value.cpu(), state[key]): raise AssertionError('initial tensor copy mismatch: ' + key)
    expected_audit = read(args.raw_run / f'seed_{seed}/raw/initial_state_audit.json')
    if audit['state_sha256'] != expected_audit['state_sha256']: raise ValueError('raw/zero initialization tensor mismatch')
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
    for point in (.25, .5, 1.): checks.setdefault(math.ceil(point * steps_per_epoch), []).append(point)
    history = []; step = 0; best_key = None; best_meta = None; train_seconds = 0.; eval_seconds = 0.; traces = {}
    base_meta = {'protocol': VERSION, 'legacy_protocol_manifest_hash': manifest['manifest_hash'], 'variant': 'zero_cross', 'seed': seed, 'init_seed': r.INIT_SEEDS[seed], 'model_config': r._config(data, args, 'zero_cross'), 'protocol_manifest_hash': manifest['manifest_hash'], 'automatic_resume': False, 'optimizer_saved': False, 'precision': 'FP32'}
    for epoch in range(1):
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
                eval_seconds += time.monotonic() - started; entry['eval_seconds_cumulative'] = eval_seconds; history.append(entry); r.write_json(directory / 'history.json', history)
        expected = set(np.unique(np.asarray(data.uid)[positions]).astype(int).tolist())
        if consumed_uids != expected: raise AssertionError('training user coverage mismatch')
        trace = dict(sha256=digest.hexdigest(), rows=len(order), order_sha256=r.sha256_array(order), batch_sizes=sizes, negative_sources=source_counts, covered_users=len(consumed_uids))
        for expected_path in (args.raw_run / f'seed_{seed}/raw/epoch_1_trace.json', args.old_zero_run / f'seed_{seed}/zero_cross/epoch_1_trace.json'):
            if read(expected_path) != trace: raise ValueError('zero consumed trace differs from sealed control')
        traces[str(epoch + 1)] = trace; r.write_json(directory / f'epoch_{epoch + 1}_trace.json', trace)
        scheduler.step()
    result = dict(base_meta, best=best_meta, steps_per_epoch=steps_per_epoch, traces=traces, train_seconds=train_seconds, eval_seconds=eval_seconds)
    r.write_json(directory / 'COMPLETED.json', result)
    return result


def control_inputs(args, manifest):
    raw=Path(args.raw_run); complete=read(raw/'COMPLETED.json'); locked=read(raw/'CHECKPOINTS_LOCKED.json')
    if complete.get('status')!='complete' or complete.get('test_future_labels_read') is not False or complete['protocol_manifest_hash']!=manifest['manifest_hash']: raise ValueError('raw control protocol mismatch')
    bindings={'raw_completion':sha(raw/'COMPLETED.json'),'raw_locked':sha(raw/'CHECKPOINTS_LOCKED.json'),'used_raw_files':{}}
    required=['panels.json','run_config.json','CHECKPOINTS_LOCKED.json']
    for seed in r.SEEDS:
        prefix=f'seed_{seed}/raw/'
        required += [prefix+name for name in ('initial_state_audit.json','epoch_1_trace.json','history.json','best.pth','confirm_best_users.npz','confirm_best_metrics.json')]
        history=read(raw/(prefix+'history.json')); early=[entry for entry in history if entry['step']<=complete['seeds'][str(seed)]['steps_per_epoch']]
        best=max(early,key=lambda entry:selection_key(entry['screen'],entry['step']))
        if best['step']!=locked[str(seed)]['best']['step']:raise ValueError('raw early selection differs from archived best')
        if not manifest['smoke'] and best['step']!={42:591,43:1181,44:1181}[seed]:raise ValueError('raw registered best step differs')
        required += [prefix+f"screen_step{entry['step']}_users.npz" for entry in early]
        if locked[str(seed)]['best']['checkpoint_sha256']!=complete['evidence_hashes'][prefix+'best.pth']:raise ValueError('raw best lock mismatch')
    for rel in required:
        actual=sha(raw/rel)
        if actual!=complete['evidence_hashes'][rel]:raise ValueError('raw control evidence mismatch: '+rel)
        bindings['used_raw_files'][rel]=actual
    if not manifest['smoke']:
        gate=read(args.archive_receipt)
        if gate.get('formal_archive_verified') is not True or gate['completion_sha256']!=bindings['raw_completion']:raise ValueError('raw local archive gate mismatch')
        backup=read(args.backup_receipt)
        validate_backup_binding(backup, bindings['raw_completion'], sha(args.archive_receipt))
        bindings['archive_receipt_sha256']=sha(args.archive_receipt);bindings['backup_receipt_sha256']=sha(args.backup_receipt)
    old=Path(args.old_zero_run); old_manifest=read(old/'dev_manifest.json')
    validate_old_manifest(old_manifest, sha(old/'dev_manifest.json'), manifest['smoke'])
    # Old dev's manifest seals all actual epoch-one trace files, independently
    # of the newer raw control. No historical candidate/test scores are opened.
    old_hashes=old_manifest.get('evidence_hashes',{})
    bindings['old_dev_manifest_sha256']=sha(old/'dev_manifest.json');bindings['old_zero_traces']={}
    for seed in r.SEEDS:
        rel=f'seed_{seed}/zero_cross/epoch_1_trace.json'; actual=sha(old/rel)
        if actual!=old_hashes.get(rel):raise ValueError('old zero trace not sealed in dev manifest')
        if read(old/rel)!=read(raw/f'seed_{seed}/raw/epoch_1_trace.json'):raise ValueError('old/new raw reference trace differs')
        bindings['old_zero_traces'][rel]=actual
    return bindings


def paired(args, records, run):
    expected_uid=np.asarray([row[0] for row in records['confirm']],dtype=np.int64);expected_position=np.asarray([row[1] for row in records['confirm']],dtype=np.int64)
    deltas={'hit5':[],'ndcg5':[]}; rows={}; sources={}
    for seed in r.SEEDS:
        zp=run/f'seed_{seed}/zero_cross/confirm_best_users.npz';rp=Path(args.raw_run)/f'seed_{seed}/raw/confirm_best_users.npz'
        with np.load(zp,allow_pickle=False) as zero,np.load(rp,allow_pickle=False) as raw:
            if not np.array_equal(zero['uid'],expected_uid) or not np.array_equal(zero['position'],expected_position):raise ValueError('confirm record order mismatch')
            for key in ('uid','position','candidate_ids','lengths','labels','pool_hit'):
                if not np.array_equal(zero[key],raw[key]):raise ValueError('unpaired raw/zero '+key)
            entry={}
            for metric in deltas:
                delta=zero[metric].astype(np.float64)-raw[metric].astype(np.float64);deltas[metric].append(delta);entry[metric+'_delta']=float(delta.mean())
                if metric=='hit5':entry.update(gained=int((delta>0).sum()),lost=int((delta<0).sum()))
            rows[str(seed)]=entry;sources[str(seed)]={'raw_arrays_sha256':sha(rp),'zero_arrays_sha256':sha(zp)}
    result={'protocol':VERSION,'comparison':'early_best_zero minus early_best_raw; already-development confirm','per_seed':rows,'array_sources':sources,'bootstrap_seed':20260925,'bootstrap_replicates':10000,'ci_level':.975,'quantiles':[.0125,.9875]}
    for metric,values in deltas.items():
        delta=np.mean(values,axis=0);rng=np.random.default_rng(20260925);samples=np.empty(10000)
        for index in range(10000):samples[index]=delta[rng.integers(0,len(delta),size=len(delta))].mean()
        result[metric]={'mean_delta':float(delta.mean()),'ci975':list(map(float,np.quantile(samples,[.0125,.9875]))),'users':len(delta),'gained':int((delta>0).sum()),'lost':int((delta<0).sum())}
    criteria={'all_seed_hr_positive':all(row['hit5_delta']>0 for row in rows.values()),'mean_hr_at_least_0_0005':result['hit5']['mean_delta']>=.0005,'hr_ci975_lower_positive':result['hit5']['ci975'][0]>0,'mean_ndcg_nonnegative':result['ndcg5']['mean_delta']>=0}
    result['criteria']=criteria;result['selection']='zero_cross' if all(criteria.values()) else 'raw_retained';result['final_test_allowed']=False
    return result


def run(args):
    started_at=datetime.now(timezone.utc);check_deadline(started_at,args.smoke)
    manifest=legacy(args);directory=Path(args.run_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise FileExistsError('refusing to overwrite early-cross run')
    try:
        if bool(args.smoke)!=manifest['smoke']:raise ValueError('smoke mismatch')
        if sha(args.protocol_doc) != '8d6dc4a421c84f09ed4dd626c2fbb144557b00a561cbf1de1be62baa87f6d710': raise ValueError('frozen overnight protocol hash mismatch')
        binding=control_inputs(args,manifest)
        manifest,data,records,positions,pools=protocol.load_inputs(Path(args.protocol_manifest))
        original=d.guard_targets(data,records)
        try:
            settings=manifest['settings'];config=SimpleNamespace(**settings,epochs=1,smoke=bool(args.smoke),raw_run=Path(args.raw_run),old_zero_run=Path(args.old_zero_run))
            config.device=('cpu' if args.smoke else 'cuda') if args.device=='auto' else args.device
            if not args.smoke and config.device!='cuda':raise ValueError('formal CUDA required')
            if config.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            torch.set_num_threads(4)
            environment={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'torch_threads':torch.get_num_threads()}
            if not args.smoke: check_environment(read(Path(args.raw_run)/'run_config.json')['environment'], environment)
            panels=read(Path(args.raw_run)/'panels.json')
            if panels['identity_sha256']!=r.sha256_json({key:value for key,value in panels.items() if key!='identity_sha256'}):raise ValueError('raw panel hash mismatch')
            with preserve_training_state(): regenerated=d.make_panels(data,positions,pools['train'],records['screen'],args.smoke)
            for key,value in regenerated.items():
                if r.sha256_json(value)!=r.sha256_json(panels[key]):raise ValueError('fixed diagnostic panels differ: '+key)
            r.write_json(directory/'panels.json',panels)
            factors,receipt=r._load_factors(Path(manifest['base_run']),data,config.dim,args.smoke)
            config_record={key:str(value) if isinstance(value,Path) else value for key,value in vars(config).items()}
            run_manifest={'protocol':VERSION,'started_at_utc':started_at.isoformat(),'deadline_utc':DEADLINE.isoformat(),'legacy_protocol_manifest_hash':manifest['manifest_hash'],'legacy_source_hashes':manifest['code_hashes'],'runner_sha256':sha(__file__),'protocol_doc_sha256':sha(args.protocol_doc),'controls':binding,'config':config_record,'scheduler_t_max':3,'factors':receipt,'test_future_labels_read':False,'smoke':bool(args.smoke),'environment':environment}
            r.write_json(directory/'run_manifest.json',run_manifest)
            results={}
            for seed in r.SEEDS:
                template,state=r._template(data,config,r.INIT_SEEDS[seed],factors);del template
                if seed==42:
                    tensor_bytes=sum(value.numel()*value.element_size() for value in state.values());need=4*tensor_bytes+(9*len(records['screen'])+3*len(records['confirm']))*(settings['candidates']*9+64)+(0 if args.smoke else 1024**3)
                    free=shutil.disk_usage(directory).free;r.write_json(directory/'disk_preflight.json',{'required_bytes':need,'free_bytes':free,'retained_models':3,'atomic_peak_models':4})
                    if free<need:raise OSError('insufficient early-cross disk space')
                results[str(seed)]=train_seed(data,config,directory/f'seed_{seed}/zero_cross',seed,state,positions,pools,records,panels,manifest);del state
            locked={seed:result['best'] for seed,result in results.items()};r.write_json(directory/'CHECKPOINTS_LOCKED.json',locked)
            for seed in r.SEEDS:
                base=directory/f'seed_{seed}/zero_cross';path=base/'best.pth'
                if sha(path)!=locked[str(seed)]['checkpoint_sha256']:raise ValueError('zero locked checkpoint changed')
                with preserve_training_state():
                    payload=torch.load(path,map_location='cpu',weights_only=False);model=r.SemanticTokenDIN(**payload['manifest']['model_config']);model.load_state_dict(payload['model']);del payload;model.to(torch.device(config.device))
                    metrics,arrays=evaluate_candidates(data,model,records['confirm'],pools['confirm']);del model
                    np.savez_compressed(base/'confirm_best_users.npz',**arrays);r.write_json(base/'confirm_best_metrics.json',metrics)
            comparison=paired(args,records,directory);r.write_json(directory/'paired_confirm.json',comparison)
            completion={'status':'complete','protocol':VERSION,'smoke':bool(args.smoke),'test_future_labels_read':False,'seeds':results,'paired':comparison,'evidence_hashes':{str(path.relative_to(directory)):sha(path) for path in sorted(directory.rglob('*')) if path.is_file()}}
            r.write_json(directory/'COMPLETED.json',completion);return completion
        finally:data.targets=original
    except BaseException as error:
        r.write_json(directory/'FAILED.json',{'status':'failed','type':type(error).__name__,'message':str(error)});raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy-source-dir','protocol-manifest','raw-run','old-zero-run','run-dir','protocol-doc'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--archive-receipt');parser.add_argument('--backup-receipt');parser.add_argument('--smoke',action='store_true');parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    return run(parser.parse_args(argv))

if __name__=='__main__':main()
