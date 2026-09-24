"""Independent full-one-epoch executor with plan/prepared locks and one test session.

prepare accepts a root-reviewed authorization JSON with explicit SHA-256 file
references. It is not an authorization by itself. train never reads targets.
seal-models needs a separate verified local training archive receipt. test creates
its exclusive marker before constructing any test candidates or reading targets.
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
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

VERSION='full-one-epoch-test100k-20260925-v1'
DEADLINE=datetime(2026,9,25,0,4,49,tzinfo=timezone.utc)
LOCK_SHA='9416c80d3c2bda822239e19a1ffd5cc8b0497c1aca252225dee26329a9cf4303'
RECORDS_SHA='086e977d4d7bb06f453bae2596dcdddb514796add4439bb5bb5ecdd18d8732fc'
IDENTITY_SEAL_SHA='30087cca4a9d7e3db166c8bff7d41cd3ef06226b78087a7a2a0989f5e199e188'
REQUIRED_REFS=('cross_completion','cross_paired','cross_locked','cross_archive','cross_sources','cross_transfer','raw_completion','raw_init','raw_run_config','raw_archive','backup_binding','diagnostic_manifest','full_pool_completion','pilot_completion','test_lock','test_records','test_identity_seal','template')
METRICS=dict(hr5='all fixed test users',ndcg5='full future IDCG',auc='candidate proxy user-equal valid only',bootstrap_seed=20260925,bootstrap_replicates=10000,ci_quantiles=[.025,.975])
ACCEPTANCE=dict(primary_comparison='zero_cross minus raw only when two models',hr5_mean_delta_min=.0005,hr5_ci95_lower_strictly_positive=True,ndcg5_mean_nonnegative=True,raw_only_business_threshold=None)
USER_KEYS=('user_id','hist_items','hist_brands','hist_ratings','hist_time_deltas','hist_verified','hist_len','click_count','time_span','user_avg_rating','user_std_rating','user_verified_ratio','user_avg_helpful')
ITEM_KEYS=('item_id','category_id','brand_id','item_click_count','created_at_ts','item_avg_rating','item_rating_number')


def strict(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite JSON number')
    if isinstance(value,dict):
        for child in value.values():strict(child)
    if isinstance(value,list):
        for child in value:strict(child)
    return value


def read(path):
    return strict(json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON constant'))))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)


def safe_path(value,root=None):
    path=Path(value).expanduser()
    require('..' not in path.parts,'parent traversal forbidden')
    absolute=path.absolute()
    for parent in [absolute,*absolute.parents]:require(not parent.is_symlink(),'symlink evidence forbidden: '+str(parent))
    if root is not None:require(absolute.is_relative_to(Path(root).absolute()),'evidence escapes root')
    return absolute


def checked(ref):
    require(isinstance(ref,dict) and isinstance(ref.get('path'),str) and isinstance(ref.get('sha256'),str),'explicit path/SHA reference required')
    path=safe_path(ref['path']);require(path.is_file(),'missing source: '+str(path));require(sha(path)==ref['sha256'],'source SHA mismatch: '+str(path));return path


def reference(path):
    path=safe_path(path);return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}


def exclusive(path,value):
    path=safe_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def empty_directory(path):
    path=safe_path(path)
    require(not path.exists() or path.is_dir(),'output is not directory')
    if path.exists():require(not any(path.iterdir()),'output directory must be empty, including dangling symlinks')
    else:path.mkdir(parents=True)
    return path


def denied_targets(*args,**kwargs):raise PermissionError('future targets forbidden outside one locked test session')


def exact_target_guard(original,records):
    allowed={(int(uid),int(position)) for uid,position,*_ in records}
    def guard(rows):
        rows=list(rows)
        require(all(len(row)==2 and (int(row[0]),int(row[1])) in allowed for row in rows),'target access outside locked test boundaries')
        return original(rows)
    return guard


def bootstrap(delta,seed=20260925,alpha=.025):
    rng=np.random.default_rng(seed);draws=np.empty(10000)
    for i in range(10000):draws[i]=delta[rng.integers(0,len(delta),size=len(delta))].mean()
    return list(map(float,np.quantile(draws,[alpha,1-alpha])))


def bind_legacy(spec):
    manifest=read(checked(spec['refs']['diagnostic_manifest']));directory=safe_path(spec['legacy_source_dir'])
    if not spec['smoke']:
        require(manifest['manifest_hash']=='c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6' and manifest['old_protocol']['sha256']=='4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1','unregistered diagnostic/base protocol')
    require(manifest.get('manifest_hash')==canonical({key:value for key,value in manifest.items() if key!='manifest_hash'}),'diagnostic manifest canonical seal mismatch before import')
    actual={p.name:sha(p) for p in directory.glob('*.py')}
    require(actual==manifest['code_hashes'],'legacy dependency code hash mismatch')
    for name in ('run_cross_multiseed','diagnostic_protocol','diagnostic_metrics','run_training_diagnostics','cross_pool_cache'):
        if name in sys.modules:require(Path(sys.modules[name].__file__).resolve().parent==directory,'conflicting legacy imports')
    sys.path.insert(0,str(directory))
    global r,dp,dm,d,cache
    r=importlib.import_module('run_cross_multiseed');dp=importlib.import_module('diagnostic_protocol');dm=importlib.import_module('diagnostic_metrics');d=importlib.import_module('run_training_diagnostics');cache=importlib.import_module('cross_pool_cache')
    require(r.code_hashes()==actual,'imported legacy dependencies differ')
    return manifest


def checked_array(path):
    with np.load(path,allow_pickle=False) as handle:
        result={key:handle[key].copy() for key in handle.files}
    require(all(np.isfinite(value).all() for value in result.values()),'nonfinite paired array')
    return result


def cross_branch(spec):
    refs=spec['refs'];complete=read(checked(refs['cross_completion']));paired=read(checked(refs['cross_paired']));locked=read(checked(refs['cross_locked']))
    require(complete.get('status')=='complete' and complete.get('protocol')=='early-stop-zero-cross-20260925-v1' and complete.get('test_future_labels_read') is False,'cross incomplete or test exposed')
    require(complete.get('smoke')==spec['smoke'] and complete['paired']==paired,'cross paired seal mismatch')
    cross_root=Path(refs['cross_completion']['path']).parent;raw_root=Path(refs['raw_completion']['path']).parent;raw=read(checked(refs['raw_completion']))
    require(raw.get('status')=='complete' and raw.get('test_future_labels_read') is False,'raw control incomplete')
    require(paired.get('bootstrap_seed')==20260925 and paired.get('bootstrap_replicates')==10000 and paired.get('ci_level')==.975 and paired.get('quantiles')==[.0125,.9875],'cross bootstrap registration mismatch')
    for key,relative in [('raw_init','seed_42/raw/initial_state_audit.json'),('raw_run_config','run_config.json')]:require(refs[key]['sha256']==raw['evidence_hashes'][relative],'raw initialization/environment source unsealed')
    rawgate=read(checked(refs['raw_archive']));require(rawgate.get('status')=='verified' and rawgate.get('completion_sha256')==refs['raw_completion']['sha256'] and (spec['smoke'] or rawgate.get('formal_archive_verified') is True),'raw archive gate mismatch')
    backup=read(checked(refs['backup_binding']));require(backup.get('status')=='verified' and backup.get('raw_completion_sha256')==refs['raw_completion']['sha256'] and backup.get('archive_receipt_sha256')==refs['raw_archive']['sha256'],'recoverable backup source mismatch')
    for name,path in [('paired_confirm.json',checked(refs['cross_paired'])),('CHECKPOINTS_LOCKED.json',checked(refs['cross_locked']))]:require(complete['evidence_hashes'][name]==sha(path),'cross used evidence mismatch')
    deltas=[];ndcgs=[]
    for seed in (42,43,44):
        rel=f'seed_{seed}/zero_cross/confirm_best_users.npz';zp=safe_path(cross_root/rel,cross_root);require(sha(zp)==complete['evidence_hashes'][rel],'zero array seal mismatch')
        relraw=f'seed_{seed}/raw/confirm_best_users.npz';rp=safe_path(raw_root/relraw,raw_root);require(sha(rp)==raw['evidence_hashes'][relraw],'raw array seal mismatch')
        z=checked_array(zp);a=checked_array(rp)
        for key in ('uid','position','candidate_ids','lengths','labels','pool_hit'):require(np.array_equal(z[key],a[key]),'cross candidate identity mismatch')
        delta=z['hit5'].astype(float)-a['hit5'].astype(float);ndcg=z['ndcg5'].astype(float)-a['ndcg5'].astype(float)
        require(np.isclose(delta.mean(),paired['per_seed'][str(seed)]['hit5_delta'],rtol=0,atol=1e-12),'cross per-seed delta mismatch')
        deltas.append(delta);ndcgs.append(ndcg)
        require(locked[str(seed)]==complete['seeds'][str(seed)]['best'],'cross checkpoint lock mismatch')
    delta=np.mean(deltas,axis=0);ndcg=np.mean(ndcgs,axis=0);ci=bootstrap(delta,alpha=.0125)
    require(np.allclose(ci,paired['hit5']['ci975'],rtol=0,atol=1e-12) and np.isclose(delta.mean(),paired['hit5']['mean_delta'],rtol=0,atol=1e-12),'cross aggregate statistics mismatch')
    criteria={'all_seed_hr_positive':all(x.mean()>0 for x in deltas),'mean_hr_at_least_0_0005':delta.mean()>=.0005,'hr_ci975_lower_positive':ci[0]>0,'mean_ndcg_nonnegative':ndcg.mean()>=0}
    require(paired['criteria']==criteria,'cross criteria mismatch')
    choice='zero_cross' if all(criteria.values()) else 'raw_retained';require(paired['selection']==choice,'cross decision mismatch')
    for label in ('cross_archive','cross_sources','cross_transfer'):
        gate=read(checked(refs[label]));require(gate.get('status') in ('verified','complete'),'cross gate incomplete: '+label)
        require(gate.get('cross_completion_sha256')==refs['cross_completion']['sha256'],'cross gate identity missing: '+label)
    return ['raw','zero_cross'] if choice=='zero_cross' else ['raw']


def bound_test_lock(spec,manifest):
    expected=safe_path(manifest['smoke_lock'] if spec['smoke'] else Path(manifest['evidence_root'])/dp.LOCK_REL)
    require(checked(spec['refs']['test_lock'])==expected,'test identity lock path relocated')
    return expected


def load_inputs(spec):
    manifest=bind_legacy(spec)
    bound_test_lock(spec,manifest)
    for ref in spec['refs'].values():checked(ref)
    _,data,records,_,_=dp.load_inputs(checked(spec['refs']['diagnostic_manifest']))
    data.targets=denied_targets
    test,identity=dp.verify_lock(manifest['evidence_root'],data,records,manifest['smoke'],manifest.get('smoke_lock'))
    expected=[tuple(row) for row in read(checked(spec['refs']['test_records']))];require(test==expected,'fixed test records differ from verified identity lock')
    if not spec['smoke']:
        for key,digest in [('test_lock',LOCK_SHA),('test_records',RECORDS_SHA),('test_identity_seal',IDENTITY_SEAL_SHA)]:require(spec['refs'][key]['sha256']==digest,'unregistered test identity source')
        require(len(test)==100000,'test count differs')
    require(len({row[0] for row in test})==len(test) and len({row[2] for row in test})==len(test),'duplicate test identities')
    lock=read(checked(spec['refs']['test_lock']));require(lock['count']==len(test) and lock.get('formal_evaluation_allowed') is False,'test identity lock mismatch')
    require(identity==manifest['test_identity_receipt'],'test identity receipt differs')
    require(np.issubdtype(np.asarray(data.train_positions).dtype,np.integer),'noninteger training positions')
    positions=np.asarray(data.train_positions,dtype=np.int64);rows=cache.PositionRecords(data,positions)
    pool_receipt=read(checked(spec['refs']['full_pool_completion']));root=Path(spec['refs']['full_pool_completion']['path']).parent
    require(pool_receipt.get('status')=='complete' and pool_receipt.get('mode')=='full' and pool_receipt.get('test_future_labels_read') is False and pool_receipt.get('test_pool_constructed') is False,'full training cache incomplete')
    pilot=read(checked(spec['refs']['pilot_completion']))
    if not spec['smoke']:
        require(len(positions)==4731777 and pool_receipt['selected_rows']==4731777,'full training row count mismatch')
        require(pilot.get('status')=='complete' and pilot['selected_rows']>=2048 and pilot['serial_parallel_check']['rows']>=128 and pilot['serial_parallel_check']['exact_ordered_equality'] is True and pilot['original_cache_overlap']['rows']>=128 and pilot['original_cache_overlap']['exact_ordered_equality'] is True,'full pool pilot insufficient')
        require(pool_receipt['build_and_validation_seconds']<=10800,'full pool exceeded three-hour budget')
    ppath=safe_path(root/pool_receipt['positions_path'],root);require(sha(ppath)==pool_receipt['positions_sha256'],'full positions file mismatch');require(np.array_equal(np.load(ppath,allow_pickle=False),positions),'full position order drift')
    cpath=safe_path(root/pool_receipt['cache_path'],root);require(sha(cpath)==pool_receipt['cache_sha256'],'full cache sidecar mismatch')
    old=read(manifest['old_protocol']['path'])
    source=dict(base_data_id=old['base_data_id'],source_hashes=old['base_assets'],records_hash=cache.rows_hash(rows),records_count=len(rows),budget=75,candidate_policy='four-way RRF from train corpus; no future target filtering',rrf_weights=[2.,0.,.7,.05],itemcf_half_life_days=180.,smoke_fallback=spec['smoke'],encoders_hash=old['encoders_hash'],code_hashes=old['code_hashes'],cf_neighbors=300,catalog_size=len(data.items),padding='zero tail, lengths define valid ordered IDs')
    pools,metadata=cache.load_cache(cpath,source)
    require(metadata['pool_hash']==pool_receipt['pool_hash'] and metadata['records_hash']==pool_receipt['records_hash']==pool_receipt['full_records_hash'],'full cache binding mismatch')
    require(pool_receipt['full_legal_rows']==len(positions) and pool_receipt['selected_rows']==len(positions),'full pool legal rows mismatch')
    require(pool_receipt['protocol_manifest_sha256']==spec['refs']['diagnostic_manifest']['sha256'] and pool_receipt['frozen_code_hashes']==manifest['code_hashes'],'pool source dependency mismatch')
    return manifest,data,positions,pools,test,old,metadata


def environment():return dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version(),torch_threads=torch.get_num_threads())


def validate_plan(path):
    plan=read(safe_path(path));require(plan.get('protocol')==VERSION and plan.get('plan_hash')==canonical({k:v for k,v in plan.items() if k!='plan_hash'}),'final plan seal mismatch')
    require(plan.get('status')=='locked' and set(REQUIRED_REFS)<=set(plan.get('refs',{})),'plan status or required refs invalid')
    require(plan['executor_sha256']==sha(__file__),'executor code changed after plan lock')
    require(plan.get('metrics')==METRICS and plan.get('acceptance')==ACCEPTANCE,'fixed metrics/acceptance changed')
    require(plan.get('test_future_labels_read') is False and plan.get('test_pool_constructed') is False,'plan prematurely accessed test')
    for ref in plan['refs'].values():checked(ref)
    authorization=read(checked(plan['authorization']))
    require(authorization['refs']==plan['refs'] and authorization.get('status')=='authorized' and authorization['smoke']==plan['smoke'] and authorization['environment']==plan['environment'],'plan differs from authorization')
    require(authorization['authorized_by']==plan['authorized_by'] and str(safe_path(authorization['legacy_source_dir']))==plan['legacy_source_dir'] and authorization['resource_review']==plan['resource_review'],'plan authority/source/resource differs')
    require(plan['seed']==42 and plan['init_seed']==424242 and plan['epochs']==1 and plan['scheduler_t_max']==3 and plan['precision']=='FP32','locked training contract drift')
    choice=read(checked(plan['refs']['cross_paired']))['selection'];require(plan['models']==(['raw','zero_cross'] if choice=='zero_cross' else ['raw']) and choice in ('zero_cross','raw_retained'),'plan branch changed')
    if not plan['smoke']:require(plan['train_rows']==4731777 and plan['test_users']==100000 and plan['settings']==dict(dim=256,token_dim=256,hist_len=50,batch_size=256,candidates=75),'formal plan contract drift')
    return plan


def prepare(args):
    auth=read(safe_path(args.authorization));require(auth.get('status')=='authorized' and auth.get('authorized_by'),'explicit reviewed authorization required')
    require(type(auth.get('smoke')) is bool,'explicit smoke flag required');smoke=auth['smoke']
    require(set(REQUIRED_REFS)<=set(auth['refs']),'authorization missing source references')
    require(not os.path.lexists(args.output),'refusing plan overwrite')
    for ref in auth['refs'].values():checked(ref)
    require(auth['refs']['template']['sha256']=='bba570fac7b3feae2e9f8723b63d0d3c03d9afd819d40fc1defe228b925859e2','full final template changed')
    now=datetime.now(timezone.utc)
    if not smoke:
        require(now<DEADLINE,'new final round deadline expired')
        require(auth['resource_review']['status']=='approved' and auth['resource_review']['estimated_remaining_seconds']>0,'reviewed time/resource budget required')
        require((DEADLINE-now).total_seconds()>=auth['resource_review']['estimated_remaining_seconds'],'insufficient remaining registered budget')
    bind_legacy(auth);models=cross_branch(auth)
    manifest,data,positions,pools,test,old,poolmeta=load_inputs(auth)
    torch.set_num_threads(4);env=environment()
    require(env==auth['environment'],'authorized environment mismatch')
    rawenv=read(checked(auth['refs']['raw_run_config']))['environment']
    require(env=={key:(rawenv[key].split()[0] if key=='python' else rawenv[key]) for key in env},'raw/full calculation environment mismatch')
    settings=manifest['settings'];require(smoke or settings==dict(dim=256,token_dim=256,hist_len=50,batch_size=256,candidates=75),'formal model settings drift')
    plan=dict(protocol=VERSION,status='locked',smoke=smoke,authorized_by=auth['authorized_by'],authorization=reference(args.authorization),refs=auth['refs'],legacy_source_dir=str(safe_path(auth['legacy_source_dir'])),executor_sha256=sha(__file__),models=models,settings=settings,seed=42,init_seed=424242,epochs=1,scheduler_t_max=3,precision='FP32',environment=env,resource_review=auth['resource_review'],started_at_utc=now.isoformat(),deadline_utc=DEADLINE.isoformat(),base_data_id=data.manifest['data_id'],encoders_hash=data.manifest['encoders_hash'],train_rows=len(positions),train_positions_dtype=positions.dtype.str,train_positions_shape=list(positions.shape),train_positions_sha256=r.sha256_array(positions),train_records_hash=cache.rows_hash(cache.PositionRecords(data,positions)),test_records_hash=cache.rows_hash(test),test_users=len(test),full_pool_hash=poolmeta['pool_hash'],test_future_labels_read=False,test_pool_constructed=False,metrics=METRICS,acceptance=ACCEPTANCE)
    plan['plan_hash']=canonical(plan);exclusive(args.output,plan);return plan


def train(args):
    plan=validate_plan(args.final_plan);directory=empty_directory(args.run_dir)
    try:
        if not plan['smoke']:
            now=datetime.now(timezone.utc);require(now<DEADLINE,'new full training round deadline expired')
            require((DEADLINE-now).total_seconds()>=plan['resource_review']['estimated_remaining_seconds'],'remaining time insufficient at actual training start')
        manifest,data,positions,pools,test,old,poolmeta=load_inputs(plan)
        torch.set_num_threads(4);require(environment()==plan['environment'],'full training environment drift')
        require(r.sha256_array(positions)==plan['train_positions_sha256'] and cache.rows_hash(test)==plan['test_records_hash'],'full plan records changed')
        config=SimpleNamespace(**plan['settings']);device=torch.device('cpu' if plan['smoke'] else 'cuda')
        require(plan['smoke'] or torch.cuda.is_available(),'formal CUDA unavailable')
        factors,_=r._load_factors(Path(manifest['base_run']),data,config.dim,plan['smoke']);template,state=r._template(data,config,424242,factors);del template
        raw_audit=read(checked(plan['refs']['raw_init']))
        test_uids={int(row[0]) for row in test};results={};reference_trace=None
        tensor_bytes=sum(v.numel()*v.element_size() for v in state.values());required=(len(plan['models'])+1)*tensor_bytes+(0 if plan['smoke'] else 1024**3)
        require(shutil.disk_usage(directory).free>=required,'insufficient full checkpoint atomic disk budget')
        exclusive(directory/'disk_preflight.json',dict(required_bytes=required,free_bytes=shutil.disk_usage(directory).free,models=len(plan['models']),atomic_peak_models=len(plan['models'])+1))
        for variant in plan['models']:
            modeldir=directory/variant;modeldir.mkdir();model=r._new_variant(data,config,state,variant)
            hashes=r._state_hashes(model);require(hashes==raw_audit['state_sha256'],'full initialization differs from sealed raw42')
            exclusive(modeldir/'initial_state_audit.json',dict(state_sha256=hashes,copied_tensor_keys=sorted(state),gate_difference_whitelist=['cross_gate_logit']))
            model.to(device);view=copy.copy(data);view.train_positions=positions;view.seed=42;view.targets=denied_targets
            dataset=r.TracedPrefixDataset(view,negatives=16,negative_policy='mixed_rrf',candidate_pools=pools,candidate_ranges=((11,25,2),(26,50,2)))
            r.seed_all(42);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.00001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=3,eta_min=.000001)
            model.train();dataset.epoch=0;order=np.random.default_rng(42).permutation(len(dataset)).astype(np.int64);digest=hashlib.sha256();sizes=[];consumed=set();sources=dict(band11_25=0,band26_50=0,candidate_fallback=0,random=0);started=time.monotonic()
            for step,indexes in enumerate(r._batch_chunks(order,config.batch_size),1):
                sizes.append(len(indexes));rows=[dataset[int(index)] for index in indexes]
                for index,row in zip(indexes,rows):
                    position=int(positions[index]);uid=int(data.uid[position]);consumed.add(uid);digest.update(np.asarray([index,position,uid],dtype=np.int64).tobytes());digest.update(np.asarray(row['neg_item_id'],dtype=np.int64).tobytes())
                    for key,count in dataset.last_sources.pop(index).items():sources[key]+=count
                batch={key:value.to(device) for key,value in r.collate(rows).items()};optimizer.zero_grad(set_to_none=True)
                users={key:batch[key] for key in USER_KEYS};positive={key:batch['pos_'+key] for key in ITEM_KEYS};negative={key:batch['neg_'+key] for key in ITEM_KEYS}
                pos,neg=model.forward_bpr(users,positive,negative);loss=F.softplus(neg.float()-pos.float()[:,None]).mean();require(bool(torch.isfinite(loss)),'nonfinite full training loss');loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                with (modeldir/'steps.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,epoch=1,batch_rows=len(indexes),loss=float(loss.detach()),gradient_l2_before_clip=float(grad),lr=optimizer.param_groups[0]['lr']))+'\n')
            expected=set(np.unique(np.asarray(data.uid)[positions]).astype(int).tolist());require(consumed==expected and test_uids<=consumed,'actual full stream lacks required test UID training coverage')
            require(sum(sizes)==len(positions),'actual full rows not consumed once')
            if not plan['smoke']:require(len(sizes)==18484 and sizes[-1]==129,'formal update/batch contract mismatch')
            trace=dict(sha256=digest.hexdigest(),rows=len(order),order_sha256=r.sha256_array(order),batch_sizes=sizes,negative_sources=sources,covered_users=len(consumed),consumed_uid_sha256=r.sha256_array(np.asarray(sorted(consumed),dtype=np.int64)),test_users_covered=len(test_uids),test_records_hash=plan['test_records_hash'],train_positions_sha256=plan['train_positions_sha256'])
            if reference_trace is None:reference_trace=trace
            else:require(trace==reference_trace,'full model consumption traces differ')
            scheduler.step();exclusive(modeldir/'trace.json',trace);np.save(modeldir/'consumed_uids.npy',np.asarray(sorted(consumed),dtype=np.int64),allow_pickle=False)
            meta=dict(protocol=VERSION,plan_hash=plan['plan_hash'],variant=variant,seed=42,init_seed=424242,epochs=1,scheduler_t_max=3,precision='FP32',model_config=r._config(data,config,variant),trace=trace,train_seconds=time.monotonic()-started)
            checkpoint_hash=d.checked_save(model,meta,modeldir/'final.pth');del model,optimizer,scheduler
            result=dict(meta,checkpoint_sha256=checkpoint_hash);exclusive(modeldir/'COMPLETED.json',result);results[variant]=result
        completed=dict(status='complete',protocol=VERSION,plan_hash=plan['plan_hash'],models=plan['models'],test_future_labels_read=False,test_pool_constructed=False,results=results,evidence_hashes={str(path.relative_to(directory)):sha(path) for path in sorted(directory.rglob('*')) if path.is_file()})
        exclusive(directory/'TRAINING_COMPLETED.json',completed);return completed
    except BaseException as error:
        exclusive(directory/'FAILED.json',dict(status='failed',error_type=type(error).__name__,message=str(error)));raise


def verify_training(plan,run):
    run=safe_path(run);complete=read(run/'TRAINING_COMPLETED.json');require(complete.get('status')=='complete' and complete['plan_hash']==plan['plan_hash'] and complete['models']==plan['models'],'full training plan mismatch')
    require(complete['test_future_labels_read'] is False and complete['test_pool_constructed'] is False,'training accessed test')
    require(not os.path.lexists(run/'FAILED.json'),'failed full training retained')
    require(set(complete['results'])==set(plan['models']),'training result keyset mismatch')
    expected_artifacts={'disk_preflight.json'}|{f'{variant}/{name}' for variant in plan['models'] for name in ('initial_state_audit.json','trace.json','consumed_uids.npy','steps.jsonl','final.pth','COMPLETED.json')}
    require(set(complete['evidence_hashes'])==expected_artifacts,'required training artifact seal incomplete')
    for rel,expected in complete['evidence_hashes'].items():require(sha(safe_path(run/rel,run))==expected,'full training evidence changed')
    test=read(checked(plan['refs']['test_records']));test_uids={int(row[0]) for row in test};traces=[]
    raw_audit=read(checked(plan['refs']['raw_init']))
    raw_model_config=read(checked(plan['refs']['raw_completion']))['seeds']['42']['model_config']
    order=np.random.default_rng(42).permutation(plan['train_rows']).astype(np.int64)
    expected_batches=[len(chunk) for chunk in r._batch_chunks(order,plan['settings']['batch_size'])]
    for variant in plan['models']:
        result=complete['results'][variant];require(read(run/variant/'COMPLETED.json')==result,'per-model completion differs')
        audit=read(run/variant/'initial_state_audit.json');require(audit['state_sha256']==raw_audit['state_sha256'] and set(audit['copied_tensor_keys'])==set(raw_audit['state_sha256']) and audit['gate_difference_whitelist']==['cross_gate_logit'],'sealed initialization audit mismatch')
        for key,value in dict(protocol=VERSION,plan_hash=plan['plan_hash'],variant=variant,seed=42,init_seed=424242,epochs=1,scheduler_t_max=3,precision='FP32').items():require(result.get(key)==value,'final training metadata drift: '+key)
        config=result['model_config'];settings=plan['settings']
        require(config==dict(raw_model_config,cross_mode=variant),'full architecture differs from sealed raw control')
        expected_config=dict(embed_dim=settings['dim'],brand_embed_dim=min(64,settings['dim']),hidden_dims=[256,128,64] if settings['dim']>=128 else [64,32],hist_len=settings['hist_len'],dropout=.1,token_dim=settings['token_dim'],fusion='concat',cross_mode=variant)
        require(all(config.get(key)==value for key,value in expected_config.items()),'final model configuration drift')
        trace=read(run/variant/'trace.json');require(result['trace']==trace,'checkpoint trace mismatch');traces.append(trace)
        require(trace['batch_sizes']==expected_batches and sum(trace['batch_sizes'])==plan['train_rows'] and trace['order_sha256']==r.sha256_array(order),'full update/order trace mismatch')
        require(sum(trace['negative_sources'].values())==plan['train_rows']*16 and set(trace['negative_sources'])=={'band11_25','band26_50','candidate_fallback','random'} and all(type(v) is int and v>=0 for v in trace['negative_sources'].values()),'negative consumption counts invalid')
        steps=[strict(json.loads(line,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSONL')))) for line in (run/variant/'steps.jsonl').read_text().splitlines()]
        require([row['step'] for row in steps]==list(range(1,len(expected_batches)+1)) and [row['batch_rows'] for row in steps]==expected_batches and all(row['epoch']==1 and row['lr']==.001 and row['loss']>=0 and row['gradient_l2_before_clip']>=0 for row in steps),'actual update log differs from trace')
        consumed=np.load(run/variant/'consumed_uids.npy',allow_pickle=False)
        require(np.issubdtype(consumed.dtype,np.integer) and consumed.ndim==1 and np.array_equal(consumed,np.unique(consumed)) and len(consumed)==trace['covered_users'] and r.sha256_array(consumed)==trace['consumed_uid_sha256'] and test_uids<=set(consumed.tolist()),'actual UID coverage proof failed')
        require(trace['rows']==plan['train_rows'] and trace['train_positions_sha256']==plan['train_positions_sha256'] and trace['test_users_covered']==len(test) and trace['test_records_hash']==plan['test_records_hash'],'full trace plan mismatch')
        checkpoint=run/variant/'final.pth';require(sha(checkpoint)==result['checkpoint_sha256'],'final checkpoint hash changed');payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        require(payload['manifest']=={key:value for key,value in result.items() if key!='checkpoint_sha256'},'checkpoint manifest differs')
        require(set(payload['model'])==set(raw_audit['state_sha256']),'final tensor names changed')
        model=r.SemanticTokenDIN(**config);model.load_state_dict(payload['model'],strict=True)
        require(all(torch.isfinite(value).all() for value in payload['model'].values()),'nonfinite final checkpoint');del model,payload
    require(all(trace==traces[0] for trace in traces),'full model traces are not paired')
    return complete


def bind_prepared_checkpoints(plan,prepared,complete):
    run=safe_path(prepared['training_run'])
    require(checked(prepared['training_completion'])==run/'TRAINING_COMPLETED.json','prepared training completion redirected')
    require(set(prepared['checkpoints'])==set(plan['models']),'prepared checkpoint keyset differs')
    for variant in plan['models']:
        ref=prepared['checkpoints'][variant]
        require(checked(ref)==run/variant/'final.pth' and ref['sha256']==complete['results'][variant]['checkpoint_sha256'],'prepared checkpoint substituted')


def seal_models(args):
    plan=validate_plan(args.final_plan);bind_legacy(plan);run=safe_path(args.run_dir);complete=verify_training(plan,run)
    receipt=read(safe_path(args.archive_receipt));require(receipt.get('status')=='verified' and receipt.get('training_completion_sha256')==sha(run/'TRAINING_COMPLETED.json') and receipt.get('plan_hash')==plan['plan_hash'],'full local archive gate not verified')
    prepared=dict(status='prepared',protocol=VERSION,plan_hash=plan['plan_hash'],plan=reference(args.final_plan),training_run=str(run),training_completion=reference(run/'TRAINING_COMPLETED.json'),archive_receipt=reference(args.archive_receipt),models=plan['models'],checkpoints={variant:reference(run/variant/'final.pth') for variant in plan['models']},test_future_labels_read=False,test_pool_constructed=False)
    prepared['prepared_hash']=canonical(prepared);exclusive(args.output,prepared);return prepared


def final_test(args):
    plan=validate_plan(args.final_plan);prepared=read(safe_path(args.prepared));require(prepared.get('status')=='prepared' and prepared['prepared_hash']==canonical({key:value for key,value in prepared.items() if key!='prepared_hash'}),'prepared seal mismatch')
    require(prepared['plan_hash']==plan['plan_hash'] and prepared['models']==plan['models'] and prepared['test_future_labels_read'] is False and prepared['test_pool_constructed'] is False,'prepared plan mismatch')
    require(checked(prepared['plan'])==safe_path(args.final_plan),'prepared first lock mismatch')
    manifest=bind_legacy(plan);bound_test_lock(plan,manifest);complete=verify_training(plan,prepared['training_run']);bind_prepared_checkpoints(plan,prepared,complete);gate=read(checked(prepared['archive_receipt']));require(gate.get('status')=='verified' and gate.get('plan_hash')==plan['plan_hash'] and gate.get('training_completion_sha256')==prepared['training_completion']['sha256'],'prepared archive gate mismatch')
    for ref in prepared['checkpoints'].values():checked(ref)
    # A single marker next to the immutable first lock prevents selecting a
    # fresh output directory to repeat this test cohort session.
    global_marker=checked(plan['refs']['test_lock']).parent/'FINAL_TEST100K_STARTED.json';require(not os.path.lexists(global_marker),'test session already started for this locked plan')
    directory=empty_directory(args.test_dir)
    manifest,data,positions,train_pools,records,old,_=load_inputs(plan);torch.set_num_threads(4);require(environment()==plan['environment'],'test environment drift');del train_pools
    require(cache.rows_hash(records)==plan['test_records_hash'],'test record identity drift')
    device=torch.device('cpu' if plan['smoke'] else 'cuda');require(plan['smoke'] or torch.cuda.is_available(),'formal CUDA unavailable')
    required=len(records)*(75*13+64)*len(plan['models'])+len(records)*304+(0 if plan['smoke'] else 1024**3);require(shutil.disk_usage(directory).free>=required,'insufficient one-test disk budget')
    marker=dict(status='started',protocol=VERSION,plan_hash=plan['plan_hash'],prepared_hash=prepared['prepared_hash'],models=plan['models'],checkpoint_hashes={key:value['sha256'] for key,value in prepared['checkpoints'].items()},test_records_hash=plan['test_records_hash'],started_at_utc=datetime.now(timezone.utc).isoformat(),test_dir=str(directory))
    exclusive(global_marker,marker);exclusive(directory/'TEST_STARTED.json',marker)
    try:
        # Fresh data restores its native target method only after TEST_STARTED.
        # It remains guarded to this exact fixed cohort and boundary throughout.
        data=r.load_data(Path(manifest['base_run']));original=data.targets;data.targets=exact_target_guard(original,records)
        pools,poolmeta=r._load_or_make_assets(data,directory,old,records,'test_final',75)
        require(poolmeta['source']['rrf_weights']==[2.,1.,.7,.05] and poolmeta['records_hash']==plan['test_records_hash'],'test candidate contract mismatch')
        results={};arrays={}
        for variant in plan['models']:
            payload=torch.load(checked(prepared['checkpoints'][variant]),map_location='cpu',weights_only=False);model=r.SemanticTokenDIN(**payload['manifest']['model_config']);model.load_state_dict(payload['model']);del payload;model.to(device)
            metrics,values=dm.evaluate_candidates(data,model,records,pools);del model
            require(len(values['uid'])==plan['test_users'],'incomplete test denominator');np.savez_compressed(directory/f'{variant}_users.npz',**values);exclusive(directory/f'{variant}_metrics.json',metrics);results[variant]=metrics;arrays[variant]=values
        result=dict(status='complete',protocol=VERSION,plan_hash=plan['plan_hash'],prepared_hash=prepared['prepared_hash'],models=plan['models'],users=plan['test_users'],metrics=results,execution_complete=True,local_archive_verified=False,business_threshold_registered=False,test_pool_hash=poolmeta['pool_hash'])
        if plan['models']==['raw','zero_cross']:
            a,z=arrays['raw'],arrays['zero_cross']
            for key in ('uid','position','candidate_ids','lengths','labels','pool_hit'):require(np.array_equal(a[key],z[key]),'final candidate pairing mismatch')
            delta=z['hit5'].astype(float)-a['hit5'].astype(float);ndcg=z['ndcg5'].astype(float)-a['ndcg5'].astype(float);ci=bootstrap(delta)
            result['paired']=dict(hr5_delta=float(delta.mean()),hr5_ci95=ci,ndcg5_delta=float(ndcg.mean()),ndcg5_ci95=bootstrap(ndcg),gained=int((delta>0).sum()),lost=int((delta<0).sum()),bootstrap_seed=20260925,bootstrap_replicates=10000)
            result['structure_gain_accepted']=bool(delta.mean()>=.0005 and ci[0]>0 and ndcg.mean()>=0)
        else:result['structure_gain_accepted']=None
        result['evidence_hashes']={str(path.relative_to(directory)):sha(path) for path in sorted(directory.rglob('*')) if path.is_file()};exclusive(directory/'TEST_COMPLETED.json',result)
        return result
    except BaseException as error:
        exclusive(directory/'FAILED.json',dict(status='failed',plan_hash=plan['plan_hash'],error_type=type(error).__name__,message=str(error)));raise


def finalize(args):
    plan=validate_plan(args.final_plan);testdir=safe_path(args.test_dir);completion=read(testdir/'TEST_COMPLETED.json')
    require(completion.get('status')=='complete' and completion['plan_hash']==plan['plan_hash'] and completion['models']==plan['models'],'test completion mismatch')
    for relative,expected in completion['evidence_hashes'].items():require(sha(safe_path(testdir/relative,testdir))==expected,'test evidence changed')
    archive=read(safe_path(args.archive_receipt));require(archive.get('status')=='verified' and archive.get('test_completion_sha256')==sha(testdir/'TEST_COMPLETED.json') and archive.get('plan_hash')==plan['plan_hash'],'final local archive gate missing')
    result=dict(status='complete',protocol=VERSION,plan_hash=plan['plan_hash'],test_completion=reference(testdir/'TEST_COMPLETED.json'),local_archive_receipt=reference(args.archive_receipt),structure_gain_accepted=completion['structure_gain_accepted'],business_threshold_registered=False)
    exclusive(args.output,result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--authorization',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('train');p.add_argument('--final-plan',required=True);p.add_argument('--run-dir',required=True)
    p=sub.add_parser('seal-models');p.add_argument('--final-plan',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--archive-receipt',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('test');p.add_argument('--final-plan',required=True);p.add_argument('--prepared',required=True);p.add_argument('--test-dir',required=True)
    p=sub.add_parser('finalize');p.add_argument('--final-plan',required=True);p.add_argument('--test-dir',required=True);p.add_argument('--archive-receipt',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv);return {'prepare':prepare,'train':train,'seal-models':seal_models,'test':final_test,'finalize':finalize}[args.command](args)

if __name__=='__main__':main()
