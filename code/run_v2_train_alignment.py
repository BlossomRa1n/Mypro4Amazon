"""One registered V2-train-pool intervention; no final-test/full entry point."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
VERSION = 'afternoon-v2-train-pool-20260925-v1'
VARIANT = 'raw_v2_train'
SYSTEM_RESERVE = 1536 * 1024**2
LOG_BUDGET = 250_000_000
CHECKPOINT_BOUND = 1_260_800_000
PAIR_KEYS = ('uid', 'position', 'candidate_ids', 'lengths', 'labels', 'pool_hit', 'auc_valid')

# Load the separately auditable new builder, not a legacy dependency by name.
_spec = importlib.util.spec_from_file_location('v2_pool_builder_contract', Path(__file__).with_name('build_v2_train_pool.py'))
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
need, safe, sha, ref, read, write = b.need, b.safe, b.sha, b.ref, b.read, b.write


def state_hashes(state):
    return {key: hashlib.sha256(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()).hexdigest() for key, value in state.items()}


def clone_state(model):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    need(all(torch.isfinite(value).all() for value in state.values()), 'nonfinite selected state')
    return state


def assert_pair(arrays, baseline, records):
    need(set(arrays) == set(baseline), 'development array fields differ')
    for key in arrays:
        need(arrays[key].shape == baseline[key].shape and arrays[key].dtype == baseline[key].dtype, 'development array shape/dtype differs: ' + key)
    for key in PAIR_KEYS:
        need(key in arrays and key in baseline and np.array_equal(arrays[key], baseline[key]), 'development pairing differs: ' + key)
    need(np.array_equal(arrays['uid'], np.asarray([row[0] for row in records])) and np.array_equal(arrays['position'], np.asarray([row[1] for row in records])), 'development record order differs')
    for name, value in arrays.items():
        need(np.isfinite(value).all(), 'nonfinite observation: ' + name)


def arrays(path):
    with np.load(safe(path), allow_pickle=False) as handle:
        return {key: handle[key].copy() for key in handle.files}


def save_arrays(path, value):
    with safe(path).open('xb') as stream:
        np.savez_compressed(stream, **value)


def system_budget(path, remaining_models, checkpoint_bytes, array_allowance, smoke=False):
    path = safe(path)
    disk = shutil.disk_usage(path)
    fs = os.statvfs(path)
    reserve = 0 if smoke else SYSTEM_RESERVE
    # One additional model is retained as an atomic-publication margin even with
    # clone-at-best and one-save-at-epoch-end implementation.
    required = (remaining_models + 1) * checkpoint_bytes + array_allowance + reserve
    need(disk.free >= required and fs.f_favail >= 64, 'system remaining worst-case budget/reserve not met')
    return dict(path=str(path), filesystem_device=path.stat().st_dev, free_bytes=disk.free, free_inodes=fs.f_favail, remaining_models=remaining_models, atomic_extra_models=1, checkpoint_bound_bytes=checkpoint_bytes, remaining_arrays_logs_bytes=array_allowance, reserve_bytes=reserve, required_free_bytes=required)


def baseline_binding(args, manifest, r, d):
    raw = safe(args.raw_run)
    done = read(raw / 'COMPLETED.json')
    locked = read(raw / 'CHECKPOINTS_LOCKED.json')
    config = read(raw / 'run_config.json')
    need(done['status'] == 'complete' and done['test_future_labels_read'] is False and done['protocol_manifest_hash'] == manifest['manifest_hash'], 'raw control protocol differs')
    settings = manifest['settings']
    need(all(config[k] == settings[k] for k in ('dim', 'token_dim', 'hist_len', 'batch_size')) and config['epochs'] == 3 and config['seeds'] == [42, 43, 44] and config['screen_checkpoints'] == [.25, .5, 1., 2., 3.], 'raw common training contract differs')
    required = ['run_config.json', 'panels.json', 'CHECKPOINTS_LOCKED.json']
    for seed in r.SEEDS:
        result = done['seeds'][str(seed)]
        need(result['variant'] == 'raw' and result['seed'] == seed and result['init_seed'] == r.INIT_SEEDS[seed] and result['precision'] == 'FP32', 'raw model/init/precision differs')
        prefix = f'seed_{seed}/raw/'
        history = read(raw / (prefix + 'history.json'))
        steps = done['seeds'][str(seed)]['steps_per_epoch']
        early = [entry for entry in history if entry['step'] <= steps]
        expected_steps = sorted(set(math.ceil(point * steps) for point in (.25, .5, 1.)))
        need([x['step'] for x in early] == expected_steps, 'raw screen grid differs')
        best = max(early, key=lambda x: d.selection_key(x['screen'], x['step']))
        need(best['step'] == locked[str(seed)]['best']['step'], 'old best is not registered early best')
        need(args.smoke or (steps == 2362 and best['step'] == {42: 591, 43: 1181, 44: 1181}[seed]), 'formal old early selection differs')
        need(locked[str(seed)]['best']['checkpoint_sha256'] == done['evidence_hashes'][prefix + 'best.pth'], 'raw best lock mismatch')
        required += [prefix + name for name in ('initial_state_audit.json', 'epoch_1_trace.json', 'history.json', 'best.pth', 'confirm_best_users.npz', 'confirm_best_metrics.json')]
        required += [prefix + f'screen_step{x}_users.npz' for x in expected_steps]
    files = {}
    for name in required:
        path = safe(raw / name)
        actual = ref(path)
        need(actual['sha256'] == done['evidence_hashes'][name], 'raw control evidence drift: ' + name)
        files[name] = actual
    binding = dict(raw_completion=ref(raw/'COMPLETED.json'), raw_locked=ref(raw/'CHECKPOINTS_LOCKED.json'), used_raw_files=files, reuse_status='registered_conditions_verified', no_raw_retraining=True)
    if not args.smoke:
        gate = read(args.archive_receipt)
        backup = read(args.backup_receipt)
        need(gate.get('formal_archive_verified') is True and gate['completion_sha256'] == binding['raw_completion']['sha256'], 'raw archive gate differs')
        need(backup['status'] == 'verified' and backup['raw_completion_sha256'] == binding['raw_completion']['sha256'] and backup['archive_receipt_sha256'] == sha(args.archive_receipt), 'raw backup gate differs')
        binding.update(archive_receipt=ref(args.archive_receipt), backup_receipt=ref(args.backup_receipt))
    return binding


def cache_binding(args, cache, manifest, data, positions):
    receipt = read(args.pool_receipt)
    need(receipt['status'] == 'complete' and receipt['mode'] == 'full' and receipt['protocol'] == VERSION and receipt['rows'] == len(positions), 'full new train cache absent')
    source = receipt['source']
    expected_source = b.source_for(args, cache, manifest, data, positions, cache.PositionRecords(data, positions), 'full')
    need(source == expected_source and receipt['smoke'] is bool(args.smoke), 'new cache complete source contract differs')
    need(source['protocol_document'] == ref(args.protocol_doc) and source['diagnostic_manifest'] == ref(args.protocol_manifest) and source['builder'] == ref(b.__file__) and source['rrf_weights'] == [2., 1., .7, .05] and source['records_count'] == len(positions) and source['positions_sha256'] == hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest(), 'new cache registered source differs')
    need(source['legacy_source_hashes'] == manifest['code_hashes'] and source['encoders_hash'] == data.manifest['encoders_hash'] and source['base_data_id'] == data.manifest['data_id'], 'new pool legacy/base differs')
    for name in ('cache', 'positions'):
        need(ref(receipt[name]['path']) == receipt[name], 'new pool file drift')
    for name, digest in receipt['evidence_hashes'].items():
        target = safe(safe(args.pool_receipt).parent / name)
        need(target.is_relative_to(safe(args.pool_receipt).parent) and sha(target) == digest, 'new pool evidence drift: ' + name)
    need(np.array_equal(np.load(receipt['positions']['path'], allow_pickle=False), positions), 'new pool positions differ')
    pools, meta = cache.load_cache(receipt['cache']['path'], source)
    need(meta['pool_hash'] == receipt['pool_hash'] and len(pools) == len(positions), 'new pool logical seal differs')
    return pools, dict(receipt=ref(args.pool_receipt), cache=receipt['cache'], positions=receipt['positions'], pool_hash=meta['pool_hash'], source=source)


def stage_budget(cfg, remaining_models):
    root = cfg.run_root
    written_other = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and p.suffix != '.pth')
    result = system_budget(root, remaining_models, cfg.checkpoint_bound, max(0, cfg.array_allowance-written_other), cfg.smoke)
    if hasattr(cfg, 'data_root'):
        result['data_disk'] = b.disk_snapshot(cfg.data_root, 0, cfg.smoke)
    return result


def train_seed(data, cfg, directory, seed, state, positions, pools, records, panels, manifest, r, d, resources):
    directory.mkdir(parents=True)
    model = r._new_variant(data, cfg, state, 'raw')
    initial = r._state_hashes(model)
    old = read(cfg.raw_run / f'seed_{seed}/raw/initial_state_audit.json')
    need(r._config(data,cfg,'raw') == read(cfg.raw_run/'COMPLETED.json')['seeds'][str(seed)]['model_config'], 'raw/new model configuration differs')
    need(initial == old['state_sha256'] and set(model.state_dict()) == set(state), 'new/old initial tensors differ')
    for name, value in model.state_dict().items():
        need(value.shape == state[name].shape and (name == 'cross_gate_logit' or torch.equal(value.cpu(), state[name])), 'initial copy/shape mismatch: ' + name)
    write(directory/'initial_state_audit.json', dict(state_sha256=initial, template_sha256=state_hashes(state), copied_tensor_keys=sorted(state), gate_difference_whitelist=['cross_gate_logit'], all_parameters_and_buffers_equal_to_raw=True, gate_value=float(model.cross_gate_logit.detach())))
    device = torch.device(cfg.device)
    model.to(device)
    view = copy.copy(data); view.train_positions = np.asarray(positions, dtype=np.int64); view.seed = seed
    dataset = r.TracedPrefixDataset(view, negatives=16, negative_policy='mixed_rrf', candidate_pools=pools['train'], candidate_ranges=((11, 25, 2), (26, 50, 2)))
    dataset.epoch = 0
    d.seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
    order = np.random.default_rng(seed).permutation(len(dataset)).astype(np.int64)
    batches = list(r._batch_chunks(order, cfg.batch_size)); steps = len(batches)
    checks = sorted(set(math.ceil(point * steps) for point in (.25, .5, 1.)))
    need(cfg.smoke or (steps == 2362 and checks == [591, 1181, 2362] and len(batches[-1]) == 95), 'formal update grid differs')
    history = []; best_key = None; best_meta = None; best_state = None
    digest = hashlib.sha256(); negative_digest = hashlib.sha256(); counts = dict(band11_25=0, band26_50=0, candidate_fallback=0, random=0); uids = set(); sizes = []
    train_seconds = 0.; eval_seconds = 0.
    base_meta = dict(protocol=VERSION, variant=VARIANT, seed=seed, init_seed=r.INIT_SEEDS[seed], model_config=r._config(data,cfg,'raw'), protocol_manifest_hash=manifest['manifest_hash'], train_pool_hash=cfg.train_pool_hash, automatic_resume=False, optimizer_saved=False, precision='FP32')
    model.train()
    for step, indexes in enumerate(batches, 1):
        began = time.monotonic(); rows = [dataset[int(index)] for index in indexes]; sizes.append(len(indexes))
        for index, row in zip(indexes, rows):
            pos = int(positions[index]); uid = int(data.uid[pos]); uids.add(uid)
            neg = np.asarray(row['neg_item_id'], dtype=np.int64)
            digest.update(np.asarray([index,pos,uid],dtype=np.int64).tobytes()); digest.update(neg.tobytes()); negative_digest.update(neg.tobytes())
            for key, count in dataset.last_sources.pop(index).items(): counts[key] += count
        batch = {key:value.to(device) for key,value in d.collate(rows).items()}
        optimizer.zero_grad(set_to_none=True)
        users = {key:batch[key] for key in d.USER_KEYS}; positive = {key:batch['pos_'+key] for key in d.ITEM_KEYS}; negative = {key:batch['neg_'+key] for key in d.ITEM_KEYS}
        pos_score, neg_score = model.forward_bpr(users,positive,negative)
        loss = F.softplus(neg_score.float()-pos_score.float()[:,None]).mean()
        need(torch.isfinite(loss), 'nonfinite loss')
        loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True); optimizer.step()
        train_seconds += time.monotonic()-began
        with (directory/'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,epoch=1,loss=float(loss.detach()),lr=optimizer.param_groups[0]['lr'],gradient_l2_before_clip=float(grad),batch_rows=len(indexes)))+'\n')
        if step in checks:
            began = time.monotonic()
            resources.append(stage_budget(cfg, 3-r.SEEDS.index(seed)))
            with d.preserve_training_state(model):
                metrics, observed = d.evaluate_candidates(data,model,records['screen'],pools['screen'])
                assert_pair(observed, arrays(cfg.raw_run/f'seed_{seed}/raw/screen_step{step}_users.npz'), records['screen'])
                save_arrays(directory/f'screen_step{step}_users.npz',observed)
                diagnostics = d.panel_diagnostics(data,model,panels,records['screen'],pools['screen'])
                key = d.selection_key(metrics,step)
                if best_key is None or key > best_key:
                    best_key = key; best_state = clone_state(model)
                    best_meta = dict(base_meta,step=step,epoch_fraction=step/steps,screen=metrics,selected_state_sha256=state_hashes(best_state),cpu_clone_includes_buffers=True)
            eval_seconds += time.monotonic()-began
            history.append(dict(step=step,epoch_fraction=step/steps,screen=metrics,diagnostics=diagnostics,train_seconds_cumulative=train_seconds,eval_seconds_cumulative=eval_seconds,selected=(best_meta['step']==step),selected_best_step=best_meta['step'],selected_state_sha256=best_meta['selected_state_sha256']))
    expected_trace = read(cfg.raw_run/f'seed_{seed}/raw/epoch_1_trace.json')
    trace = dict(sha256=digest.hexdigest(),negative_digest=negative_digest.hexdigest(),rows=len(order),order_sha256=r.sha256_array(order),batch_sizes=sizes,negative_sources=counts,covered_users=len(uids),negative_stream_expected_equal=False)
    for key in ('rows','order_sha256','batch_sizes','covered_users'):
        need(trace[key] == expected_trace[key], 'raw/new row consumption differs: '+key)
    need(uids == set(np.unique(np.asarray(data.uid)[positions]).astype(int)), 'training user coverage mismatch')
    need(sum(counts.values()) == len(positions)*16, 'negative source count mismatch')
    scheduler.step()
    write(directory/'history.json',history); write(directory/'epoch_1_trace.json',trace)
    resources.append(stage_budget(cfg,3-r.SEEDS.index(seed)))
    need(best_state is not None and state_hashes(best_state)==best_meta['selected_state_sha256'], 'selected clone changed before save')
    path=directory/'best.pth'
    need(not path.exists() and not path.with_suffix('.pth.tmp').exists(),'checkpoint output exists')
    r.atomic_torch_save(dict(model=best_state,manifest=best_meta),path)
    restored=torch.load(path,map_location='cpu',weights_only=False)
    need(set(restored['model'])==set(best_state) and all(torch.equal(restored['model'][k],v) and torch.isfinite(restored['model'][k]).all() for k,v in best_state.items()),'checkpoint reload differs')
    best_meta=dict(best_meta,checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size)
    need(path.stat().st_size <= cfg.checkpoint_bound, 'checkpoint exceeded registered bound')
    result=dict(base_meta,best=best_meta,steps_per_epoch=steps,traces={'1':trace},train_seconds=train_seconds,eval_seconds=eval_seconds)
    write(directory/'COMPLETED.json',result)
    return result


def paired(raw_run, directory, records, seeds=(42,43,44)):
    values={'hit5':[],'ndcg5':[]}; per_seed={}; sources={}
    for seed in seeds:
        newpath=directory/f'seed_{seed}/{VARIANT}/confirm_best_users.npz'; oldpath=raw_run/f'seed_{seed}/raw/confirm_best_users.npz'
        new, old=arrays(newpath), arrays(oldpath); assert_pair(new,old,records)
        row={}
        for key in values:
            delta=new[key].astype(np.float64)-old[key].astype(np.float64); values[key].append(delta); row[key+'_delta']=float(delta.mean())
        row.update(gauc_delta=float(new['auc'][new['auc_valid']].mean()-old['auc'][old['auc_valid']].mean()) if new['auc_valid'].any() else 0.,gained=int((values['hit5'][-1]>0).sum()),lost=int((values['hit5'][-1]<0).sum()))
        per_seed[str(seed)]=row;sources[str(seed)]=dict(new_arrays=ref(newpath),raw_arrays=ref(oldpath))
    result=dict(protocol=VERSION,comparison='v2-train-pool early-best minus old raw early-best; reused development confirm',per_seed=per_seed,array_sources=sources,bootstrap_seed=20260925,bootstrap_replicates=10000,ci_level=.975,quantiles=[.0125,.9875])
    for key, observations in values.items():
        delta=np.mean(observations,axis=0); rng=np.random.default_rng(20260925); samples=np.empty(10000)
        for index in range(10000): samples[index]=delta[rng.integers(0,len(delta),size=len(delta))].mean()
        result[key]=dict(mean_delta=float(delta.mean()),ci975=list(map(float,np.quantile(samples,[.0125,.9875]))),users=len(delta),gained=int((delta>0).sum()),lost=int((delta<0).sum()))
    criteria=dict(all_seed_hr_positive=all(row['hit5_delta']>0 for row in per_seed.values()),mean_hr_at_least_0_0005=result['hit5']['mean_delta']>=.0005,hr_ci975_lower_positive=result['hit5']['ci975'][0]>0,mean_ndcg_nonnegative=result['ndcg5']['mean_delta']>=0)
    result.update(criteria=criteria,supports_v2_train_pool_followup=all(criteria.values()),selection='v2_train_pool' if all(criteria.values()) else 'raw_retained',final_test_allowed=False,full_training_allowed=False)
    return result


def run(args):
    directory=safe(args.run_dir); need(not os.path.lexists(directory),'new independent run directory required')
    for existing in (args.raw_run, args.legacy_source_dir, safe(args.pool_receipt).parent):
        existing=safe(existing)
        need(not directory.is_relative_to(existing) and not existing.is_relative_to(directory), 'run output overlaps protected input')
    r,cache,manifest,data,records,positions,pools=b.legacy_inputs(args)
    import importlib
    d=importlib.import_module('run_training_diagnostics')
    binding=baseline_binding(args,manifest,r,d)
    torch.set_num_threads(4)
    env=b.environment(); env.update(torch_interop_threads=torch.get_num_interop_threads(),gpu=[dict(index=i,name=torch.cuda.get_device_name(i),total_memory=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])
    oldenv=read(safe(args.raw_run)/'run_config.json')['environment']
    if not args.smoke:
        for key in ('torch','numpy','cuda_runtime','cudnn','torch_threads','torch_interop_threads','gpu'): need(env[key]==oldenv[key],'baseline environment differs: '+key)
        need(env['python'].split()[0]==oldenv['python'].split()[0] and torch.cuda.is_available(),'formal runtime mismatch')
        need(env['matmul_allow_tf32'] is False and env['cudnn_allow_tf32'] is True and env['float32_matmul_precision'] == 'highest', 'registered fresh-process precision defaults differ')
    env['historical_tf32_recorded']=all(k in oldenv for k in ('matmul_allow_tf32','cudnn_allow_tf32','float32_matmul_precision'))
    env['historical_tf32_equality_verified']=False
    if env['historical_tf32_recorded']:
        need(all(env[k]==oldenv[k] for k in ('matmul_allow_tf32','cudnn_allow_tf32','float32_matmul_precision')), 'historical TF32 settings differ')
        env['historical_tf32_equality_verified']=True
    # Check fixed old panels before replacing the train pool.
    panels=read(safe(args.raw_run)/'panels.json')
    need(panels['identity_sha256']==r.sha256_json({k:v for k,v in panels.items() if k!='identity_sha256'}),'old panel identity drift')
    original=d.guard_targets(data,records)
    with d.preserve_training_state(): regenerated=d.make_panels(data,positions,pools['train'],records['screen'],args.smoke)
    need(all(r.sha256_json(v)==r.sha256_json(panels[k]) for k,v in regenerated.items()),'old fixed panel differs')
    newpool,poolbinding=cache_binding(args,cache,manifest,data,positions); pools=dict(pools,train=newpool)
    settings=manifest['settings']; cfg=SimpleNamespace(**settings,epochs=1,smoke=bool(args.smoke),device='cpu' if args.smoke else 'cuda',raw_run=safe(args.raw_run),run_root=directory,data_root=safe(args.pool_receipt).parent,train_pool_hash=poolbinding['pool_hash'])
    factors,factor_receipt=r._load_factors(safe(manifest['base_run']),data,cfg.dim,args.smoke)
    need(factor_receipt == read(cfg.raw_run/'run_config.json')['factors'], 'raw/new initialization factors differ')
    need(r.sha256_array(np.asarray(positions)) == read(cfg.raw_run/'run_config.json')['train_positions_sha256'], 'raw/new training positions differ')
    template,initial=r._template(data,cfg,r.INIT_SEEDS[42],factors); del template
    tensor_bytes=sum(v.numel()*v.element_size() for v in initial.values())
    del initial
    cfg.checkpoint_bound=max(CHECKPOINT_BOUND,tensor_bytes+1024**2) if not args.smoke else tensor_bytes+1024**2
    evaluation_bytes=sum(Path(item['path']).stat().st_size for name,item in binding['used_raw_files'].items() if name.endswith('_users.npz'))
    cfg.array_allowance=max(LOG_BUDGET,math.ceil(evaluation_bytes*1.25)) if not args.smoke else max(1024**2,math.ceil(evaluation_bytes*1.25))
    directory.parent.mkdir(parents=True,exist_ok=True)
    resources=[system_budget(directory.parent,3,cfg.checkpoint_bound,cfg.array_allowance,args.smoke)]
    resources[0]['data_disk']=b.disk_snapshot(cfg.data_root,0,args.smoke)
    need(args.smoke or resources[0]['filesystem_device'] != resources[0]['data_disk']['filesystem_device'], 'registered data/system split disks not present')
    directory.mkdir()
    try:
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}
        run_manifest=dict(protocol=VERSION,smoke=bool(args.smoke),started_at_utc=datetime.now(timezone.utc).isoformat(),legacy_protocol_manifest_hash=manifest['manifest_hash'],legacy_source_hashes=manifest['code_hashes'],protocol_manifest=ref(args.protocol_manifest),protocol_document=ref(args.protocol_doc),runner=ref(__file__),builder=ref(b.__file__),controls=binding,config=config,scheduler_t_max=3,factors=factor_receipt,environment=env,train_cache=poolbinding,eval_cache_refs=manifest['candidate_caches'],data_root=str(safe(args.pool_receipt).parent),system_root=str(directory),resource_budget=resources[0],test_future_labels_read=False,final_test_allowed=False,full_training_allowed=False)
        write(directory/'run_manifest.json',run_manifest); write(directory/'panels.json',panels)
        results={}
        for seed in r.SEEDS:
            resources.append(stage_budget(cfg,3-r.SEEDS.index(seed)))
            template,state=r._template(data,cfg,r.INIT_SEEDS[seed],factors); del template
            results[str(seed)]=train_seed(data,cfg,directory/f'seed_{seed}/{VARIANT}',seed,state,positions,pools,records,panels,manifest,r,d,resources); del state
        locked={seed:result['best'] for seed,result in results.items()};write(directory/'CHECKPOINTS_LOCKED.json',locked)
        for seed in r.SEEDS:
            base=directory/f'seed_{seed}/{VARIANT}'; checkpoint=base/'best.pth'
            need(sha(checkpoint)==locked[str(seed)]['checkpoint_sha256'],'locked checkpoint changed')
            resources.append(stage_budget(cfg,0))
            with d.preserve_training_state():
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                model=r.SemanticTokenDIN(**payload['manifest']['model_config']);model.load_state_dict(payload['model']);del payload;model.to(cfg.device)
                metrics,observed=d.evaluate_candidates(data,model,records['confirm'],pools['confirm']);del model
                assert_pair(observed,arrays(cfg.raw_run/f'seed_{seed}/raw/confirm_best_users.npz'),records['confirm'])
                save_arrays(base/'confirm_best_users.npz',observed);write(base/'confirm_best_metrics.json',metrics)
        comparison=paired(cfg.raw_run,directory,records['confirm']);write(directory/'paired_confirm.json',comparison)
        need(baseline_binding(args,manifest,r,d)==binding,'raw controls changed during run')
        need(ref(__file__)==run_manifest['runner'] and ref(b.__file__)==run_manifest['builder'],'new source changed during run')
        need(ref(args.protocol_doc)==run_manifest['protocol_document'] and ref(args.protocol_manifest)==run_manifest['protocol_manifest'], 'protocol source changed during run')
        need({p.name:sha(p) for p in safe(args.legacy_source_dir).glob('*.py')}==manifest['code_hashes'], 'legacy source changed during run')
        write(directory/'resource_snapshots.json',resources)
        complete=dict(status='complete',protocol=VERSION,smoke=bool(args.smoke),test_future_labels_read=False,final_test_allowed=False,full_training_allowed=False,seeds=results,paired=comparison,evidence_hashes={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()})
        write(directory/'COMPLETED.json',complete);return complete
    except BaseException as error:
        write(directory/'FAILED.json',dict(status='failed',type=type(error).__name__,message=str(error),automatic_retry=False));raise
    finally: data.targets=original


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy-source-dir','protocol-manifest','protocol-doc','raw-run','pool-receipt','run-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--archive-receipt');p.add_argument('--backup-receipt');p.add_argument('--smoke',action='store_true')
    return run(p.parse_args(argv))


if __name__=='__main__':main()
