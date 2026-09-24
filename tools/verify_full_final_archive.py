"""Local full-final training/test semantic archive verifier.

Original plan bytes and remote source references are never rewritten/resolved.
Explicit local inputs are trusted archived files; base-data pickle is executable
serialization and must come from the separately verified source-mapping gate.
No targets, training, scoring, SSH, deletion or shutdown. Only a new receipt is
written after success. Original source/authorization/prepared-lock authenticity
remains an independent root gate, reported explicitly in every receipt.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import run_cross_multiseed as r
import cross_pool_cache as cache
import verify_training_diagnostics_archive as diagnostic

_spec=importlib.util.spec_from_file_location('archive_full_final_contract',ROOT/'code/run_full_final_test100k.py')
contract=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(contract)
require=diagnostic.require
read=diagnostic.read_json
sha=diagnostic.sha
canonical=diagnostic.canonical_hash
safe=contract.safe_path


def sealed_tree(run,completion_name,required):
    run=safe(run)
    for path in run.rglob('*'):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()),'symlink/special archive member')
    require(not os.path.lexists(run/'FAILED.json'),'failed archive cannot verify')
    complete=read(run/completion_name)
    require(complete.get('status')=='complete','incomplete archive')
    hashes=complete.get('evidence_hashes');require(isinstance(hashes,dict) and bool(hashes),'missing evidence seal')
    require(set(hashes)==set(required),'required evidence coverage differs')
    actual={str(path.relative_to(run)) for path in run.rglob('*') if path.is_file()}
    require(actual==set(hashes)|{completion_name},'extra or missing archive files')
    verified={}
    for relative,expected in hashes.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'unsafe evidence path')
        path=safe(run/relative,run);require(sha(path)==expected,'evidence SHA mismatch: '+relative)
        verified[relative]={'sha256':expected,'size_bytes':path.stat().st_size}
        if path.suffix=='.json':read(path)
        elif path.suffix=='.jsonl':
            with path.open() as stream:
                for line in stream:diagnostic.strict_json(line)
    verified[completion_name]={'sha256':sha(run/completion_name),'size_bytes':(run/completion_name).stat().st_size}
    return complete,verified


def local_inputs(args):
    paths={name:safe(getattr(args,name)) for name in ('plan','run_dir','raw_run','test_records','positions','base_data','receipt','old_protocol','diagnostic_manifest')}
    require(not os.path.lexists(paths['receipt']),'refusing receipt overwrite')
    require(all(not paths['receipt'].is_relative_to(paths[key]) for key in ('run_dir','raw_run')),'receipt must be outside immutable archives')
    plan=read(paths['plan']);require(plan.get('protocol')==contract.VERSION and plan.get('status')=='locked','invalid final plan')
    require(plan.get('plan_hash')==canonical({key:value for key,value in plan.items() if key!='plan_hash'}),'plan canonical seal mismatch')
    require(plan.get('metrics')==contract.METRICS and plan.get('acceptance')==contract.ACCEPTANCE,'metrics/acceptance contract drift')
    require(plan.get('executor_sha256')==sha(ROOT/'code/run_full_final_test100k.py'),'archived plan executor differs from verifier contract')
    smoke=plan.get('smoke');require(type(smoke) is bool and (not smoke or getattr(args,'allow_smoke',False)),'smoke archive requires explicit allow-smoke')
    require(plan['models'] in (['raw'],['raw','zero_cross']),'invalid final model branch')
    require(plan.get('seed')==42 and plan.get('init_seed')==424242 and plan.get('epochs')==1 and plan.get('scheduler_t_max')==3 and plan.get('precision')=='FP32','training contract mismatch')
    require(plan.get('test_future_labels_read') is False and plan.get('test_pool_constructed') is False,'plan accessed test early')
    if not smoke:
        require(plan['settings']==dict(dim=256,token_dim=256,hist_len=50,batch_size=256,candidates=75) and plan['train_rows']==4731777 and plan['test_users']==100000,'formal cohort/model contract differs')
    raw=read(paths['raw_run']/'COMPLETED.json')
    require(sha(paths['raw_run']/'COMPLETED.json')==plan['refs']['raw_completion']['sha256'],'raw completion source mismatch')
    require(raw.get('status')=='complete' and raw.get('test_future_labels_read') is False and raw.get('smoke')==smoke,'invalid raw control completion')
    needed={'seed_42/raw/initial_state_audit.json','seed_42/raw/best.pth','seed_42/raw/history.json'}
    for relative in needed:
        require(sha(safe(paths['raw_run']/relative,paths['raw_run']))==raw['evidence_hashes'][relative],'raw used evidence mismatch')
    require(sha(paths['raw_run']/'seed_42/raw/initial_state_audit.json')==plan['refs']['raw_init']['sha256'],'raw initialization reference mismatch')
    require(sha(paths['test_records'])==plan['refs']['test_records']['sha256'],'test records source SHA mismatch')
    records=read(paths['test_records']);require(len(records)==plan['test_users'] and cache.rows_hash(records)==plan['test_records_hash'],'fixed test record order mismatch')
    require(all(len(row)==3 and type(row[0]) is int and type(row[1]) is int and isinstance(row[2],str) for row in records),'invalid identity schema')
    require(len({row[0] for row in records})==len(records) and len({row[2] for row in records})==len(records),'duplicate test identity')
    positions=np.load(paths['positions'],allow_pickle=False,mmap_mode='r')
    require(positions.dtype.str==plan['train_positions_dtype'] and list(positions.shape)==plan['train_positions_shape'] and positions.ndim==1 and np.issubdtype(positions.dtype,np.integer),'full positions dtype/shape differs')
    require(len(positions)==plan['train_rows'] and r.sha256_array(positions)==plan['train_positions_sha256'],'full positions SHA differs')
    diagnostic_manifest=read(paths['diagnostic_manifest'])
    require(sha(paths['diagnostic_manifest'])==plan['refs']['diagnostic_manifest']['sha256'] and diagnostic_manifest['manifest_hash']==canonical({k:v for k,v in diagnostic_manifest.items() if k!='manifest_hash'}),'local diagnostic manifest binding differs')
    old=read(paths['old_protocol'])
    require(sha(paths['old_protocol'])==diagnostic_manifest['old_protocol']['sha256'] and old['manifest_hash']==canonical({k:v for k,v in old.items() if k!='manifest_hash'}),'local original protocol binding differs')
    require(diagnostic_manifest.get('smoke')==smoke and old.get('smoke')==smoke,'local protocol smoke differs')
    if not smoke:require(diagnostic_manifest['manifest_hash']=='c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6' and sha(paths['old_protocol'])=='4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1','unregistered original protocol anchors')
    require(sha(paths['base_data'])==old['base_assets']['data.pkl'],'base bytes mismatch before trusted pickle load')
    with paths['base_data'].open('rb') as stream:value=pickle.load(stream)
    data=value['data'] if isinstance(value,dict) and 'data' in value else value
    data.targets=contract.denied_targets
    require(data.manifest['data_id']==plan['base_data_id'] and data.manifest['encoders_hash']==plan['encoders_hash'],'local base identity differs')
    require(np.array_equal(positions,np.asarray(data.train_positions)),'local positions differ from full legal base rows')
    require(np.all(positions>=0) and np.all(positions<len(data.uid)) and np.all(positions[1:]>positions[:-1]),'invalid full legal positions')
    require(cache.rows_hash(cache.PositionRecords(data,positions))==plan['train_records_hash'],'full position UID/raw order differs')
    boundaries={int(uid):int(position) for uid,position in zip(data.val_users,data.val_positions)}
    for uid,position,raw_id in records:require(2<=uid<len(data.users) and str(data.users[uid])==raw_id and boundaries.get(uid)==position,'test UID/raw/boundary mapping differs')
    return paths,plan,raw,records,positions,data


def training(paths,plan,raw,records,positions,data):
    run=paths['run_dir'];names=('initial_state_audit.json','trace.json','consumed_uids.npy','steps.jsonl','final.pth','COMPLETED.json')
    required={'disk_preflight.json'}|{f'{variant}/{name}' for variant in plan['models'] for name in names}
    complete,files=sealed_tree(run,'TRAINING_COMPLETED.json',required)
    require(complete.get('protocol')==contract.VERSION and complete.get('plan_hash')==plan['plan_hash'] and complete.get('models')==plan['models'],'training completion plan mismatch')
    require(complete.get('test_future_labels_read') is False and complete.get('test_pool_constructed') is False and set(complete['results'])==set(plan['models']),'training test-isolation/result mismatch')
    audit=read(paths['raw_run']/'seed_42/raw/initial_state_audit.json')
    rawbest=torch.load(paths['raw_run']/'seed_42/raw/best.pth',map_location='cpu',weights_only=False)
    rawconfig=raw['seeds']['42']['model_config'];require(rawbest['manifest']['model_config']==rawconfig,'raw best config mismatch')
    rawschema={key:(tuple(value.shape),value.dtype) for key,value in rawbest['model'].items()};del rawbest
    expected_uids=np.unique(np.asarray(data.uid)[positions]).astype(np.int64)
    test_uids={int(row[0]) for row in records};order=np.random.default_rng(42).permutation(len(positions)).astype(np.int64)
    batch_sizes=[len(chunk) for chunk in r._batch_chunks(order,plan['settings']['batch_size'])];traces=[]
    if not plan['smoke']:require(len(batch_sizes)==18484 and batch_sizes[-1]==129,'formal batch trace differs')
    for variant in plan['models']:
        directory=run/variant;result=complete['results'][variant]
        require(read(directory/'COMPLETED.json')==result,'per-model completion differs')
        initial=read(directory/'initial_state_audit.json')
        require(initial['state_sha256']==audit['state_sha256'] and set(initial['copied_tensor_keys'])==set(audit['state_sha256']) and initial['gate_difference_whitelist']==['cross_gate_logit'],'full initialization audit differs')
        for key,expected in dict(protocol=contract.VERSION,plan_hash=plan['plan_hash'],variant=variant,seed=42,init_seed=424242,epochs=1,scheduler_t_max=3,precision='FP32').items():require(result.get(key)==expected,'full model metadata mismatch: '+key)
        require(result['model_config']==dict(rawconfig,cross_mode=variant),'full architecture differs from raw control')
        trace=read(directory/'trace.json');require(trace==result['trace'],'model trace differs from completion');traces.append(trace)
        require(trace['rows']==len(positions) and trace['train_positions_sha256']==plan['train_positions_sha256'] and trace['order_sha256']==r.sha256_array(order) and trace['batch_sizes']==batch_sizes,'full row/order/batch trace mismatch')
        counts=trace['negative_sources'];require(set(counts)=={'band11_25','band26_50','candidate_fallback','random'} and all(type(v) is int and v>=0 for v in counts.values()) and sum(counts.values())==len(positions)*16,'mixed16 trace count mismatch')
        require(isinstance(trace['sha256'],str) and len(trace['sha256'])==64 and all(x in '0123456789abcdef' for x in trace['sha256']),'invalid consumed-row/negative digest')
        consumed=np.load(directory/'consumed_uids.npy',allow_pickle=False)
        require(consumed.dtype==np.dtype('int64') and np.array_equal(consumed,expected_uids) and trace['covered_users']==len(expected_uids) and trace['consumed_uid_sha256']==r.sha256_array(expected_uids),'actual consumed UID set differs from full position mapping')
        require(test_uids<=set(consumed.tolist()) and trace['test_users_covered']==len(test_uids) and trace['test_records_hash']==plan['test_records_hash'],'warm test UID training coverage failed')
        with (directory/'steps.jsonl').open() as stream:steps=[diagnostic.strict_json(line) for line in stream]
        require(len(steps)==len(batch_sizes),'update log count differs')
        for index,(step,size) in enumerate(zip(steps,batch_sizes),1):require(step['step']==index and step['epoch']==1 and step['batch_rows']==size and step['lr']==.001 and step['loss']>=0 and step['gradient_l2_before_clip']>=0,'invalid training update log')
        checkpoint=directory/'final.pth';require(sha(checkpoint)==result['checkpoint_sha256'],'final checkpoint SHA differs')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        require(payload['manifest']=={key:value for key,value in result.items() if key!='checkpoint_sha256'},'final checkpoint metadata differs')
        require(set(payload['model'])==set(rawschema),'checkpoint tensor keys differ')
        for key,value in payload['model'].items():require(torch.is_tensor(value) and (tuple(value.shape),value.dtype)==rawschema[key] and torch.isfinite(value).all().item(),'checkpoint shape/dtype/finite failed: '+key)
        del payload
    require(all(trace==traces[0] for trace in traces),'dual-model full consumption traces differ')
    return complete,files,'training_completion_sha256',{'models':len(plan['models']),'updates_per_model':len(batch_sizes),'actual_consumed_users':len(expected_uids),'test_users_covered':len(test_uids)}


def test_archive(paths,plan,raw,records,positions,data):
    prepared=read(paths['prepared'])
    require(prepared.get('status')=='prepared' and prepared.get('protocol')==contract.VERSION and prepared.get('prepared_hash')==canonical({k:v for k,v in prepared.items() if k!='prepared_hash'}),'prepared local seal mismatch')
    require(prepared['plan_hash']==plan['plan_hash'] and prepared['models']==plan['models'] and prepared['plan']['sha256']==sha(paths['plan']) and prepared['test_future_labels_read'] is False and prepared['test_pool_constructed'] is False,'prepared local plan binding differs')
    train_paths=dict(paths,run_dir=paths['training_run'])
    train_complete,_,_,_=training(train_paths,plan,raw,records,positions,data)
    training_sha=sha(paths['training_run']/'TRAINING_COMPLETED.json');train_receipt=read(paths['training_receipt'])
    require(prepared['training_completion']['sha256']==training_sha and train_receipt.get('status')=='verified' and train_receipt.get('plan_hash')==plan['plan_hash'] and train_receipt.get('training_completion_sha256')==training_sha and prepared['archive_receipt']['sha256']==sha(paths['training_receipt']),'prepared training/archive binding differs')
    require(set(prepared['checkpoints'])==set(plan['models']),'prepared checkpoint set differs')
    checkpoint_hashes={variant:sha(paths['training_run']/variant/'final.pth') for variant in plan['models']}
    for variant,digest in checkpoint_hashes.items():require(prepared['checkpoints'][variant]['sha256']==digest==train_complete['results'][variant]['checkpoint_sha256'],'prepared intended checkpoint differs')
    run=paths['run_dir'];metadata=read(safe(run/'candidate_test_final.json',run));required={'TEST_STARTED.json','candidate_test_final.json','candidate_test_final.json.building'}
    for name in ('items','lengths'):
        relative=metadata['files'][name]['name'];require(Path(relative).name==relative,'candidate payload path escapes root');required.add(relative)
    required|={f'{variant}_{suffix}' for variant in plan['models'] for suffix in ('users.npz','metrics.json')}
    complete,files=sealed_tree(run,'TEST_COMPLETED.json',required)
    require(complete.get('protocol')==contract.VERSION and complete.get('plan_hash')==plan['plan_hash'] and complete.get('models')==plan['models'] and complete.get('users')==plan['test_users'] and complete.get('execution_complete') is True,'test completion plan mismatch')
    require(complete.get('business_threshold_registered') is False and complete.get('local_archive_verified') is False,'test completion reporting contract differs')
    marker=read(run/'TEST_STARTED.json')
    require(read(paths['global_marker'])==marker,'global once-marker differs from session marker')
    require(marker['prepared_hash']==prepared['prepared_hash'] and marker['checkpoint_hashes']==checkpoint_hashes,'session prepared/checkpoint binding differs')
    require(marker.get('status')=='started' and marker['plan_hash']==plan['plan_hash'] and marker['prepared_hash']==complete['prepared_hash'] and marker['models']==plan['models'] and marker['test_records_hash']==plan['test_records_hash'],'one-test session marker mismatch')
    require(set(marker['checkpoint_hashes'])==set(plan['models']) and all(isinstance(v,str) and len(v)==64 for v in marker['checkpoint_hashes'].values()),'test checkpoint marker invalid')
    pools,metadata=cache.load_cache(run/'candidate_test_final.json');source=metadata['source']
    old=read(paths['old_protocol'])
    required_source=dict(source_hashes=old['base_assets'],code_hashes=old['code_hashes'],candidate_policy='four-way RRF from train corpus; no future target filtering',padding='zero tail, lengths define valid ordered IDs',base_data_id=plan['base_data_id'],encoders_hash=plan['encoders_hash'],records_hash=plan['test_records_hash'],records_count=len(records),budget=75,rrf_weights=[2.,1.,.7,.05],itemcf_half_life_days=180.,smoke_fallback=plan['smoke'],cf_neighbors=300,catalog_size=len(data.items))
    require(source==required_source,'test pool full source contract mismatch')
    require(len(pools)==len(records) and metadata['records_hash']==plan['test_records_hash'] and metadata['pool_hash']==complete['test_pool_hash'],'test pool identity mismatch')
    expected_uid=np.asarray([row[0] for row in records],dtype=np.int64);expected_position=np.asarray([row[1] for row in records],dtype=np.int64);arrays={}
    for variant in plan['models']:
        value=diagnostic.load_arrays(run/f'{variant}_users.npz');metrics=read(run/f'{variant}_metrics.json');diagnostic.verify_metrics(value,metrics)
        require(metrics==complete['metrics'][variant],'test metric file/completion mismatch')
        require(np.array_equal(value['uid'],expected_uid) and np.array_equal(value['position'],expected_position),'test array fixed user order differs')
        require(value['candidate_ids'].shape[1]<=75 and np.all(value['candidate_ids']<len(data.items)),'test candidate IDs outside catalog/budget')
        for index,pool in enumerate(pools):require(int(value['lengths'][index])==len(pool) and np.array_equal(value['candidate_ids'][index,:len(pool)],pool),'scores candidate order differs from sealed pool')
        arrays[variant]=value
    verify_comparison(plan['models'],arrays,complete)
    return complete,files,'test_completion_sha256',{'models':len(plan['models']),'test_users':len(records),'candidate_pool_hash':metadata['pool_hash'],'structure_gain_accepted':complete['structure_gain_accepted']}


def verify_comparison(models,arrays,complete):
    if models==['raw','zero_cross']:
        a,z=arrays['raw'],arrays['zero_cross']
        for key in ('uid','position','candidate_ids','lengths','labels','pool_hit'):require(np.array_equal(a[key],z[key]),'unpaired final arrays')
        delta=z['hit5'].astype(float)-a['hit5'].astype(float);ndcg=z['ndcg5'].astype(float)-a['ndcg5'].astype(float);ci=contract.bootstrap(delta);nci=contract.bootstrap(ndcg)
        expected=dict(hr5_delta=float(delta.mean()),hr5_ci95=ci,ndcg5_delta=float(ndcg.mean()),ndcg5_ci95=nci,gained=int((delta>0).sum()),lost=int((delta<0).sum()),bootstrap_seed=20260925,bootstrap_replicates=10000)
        require(complete.get('paired')==expected,'paired test bootstrap/deltas differ')
        accepted=bool(delta.mean()>=.0005 and ci[0]>0 and ndcg.mean()>=0);require(complete['structure_gain_accepted'] is accepted,'final structural acceptance differs')
    else:require(complete.get('structure_gain_accepted') is None and 'paired' not in complete,'raw-only archive claims unregistered structural comparison')


def verify(args):
    paths,plan,raw,records,positions,data=local_inputs(args)
    if args.command=='test':
        for name in ('prepared','global_marker','training_run','training_receipt'):
            require(bool(getattr(args,name,None)),'test requires explicit local '+name);paths[name]=safe(getattr(args,name))
    if 'training_run' in paths:require(not paths['receipt'].is_relative_to(paths['training_run']),'receipt must be outside immutable training archive')
    result,files,digest_field,summary=(training if args.command=='training' else test_archive)(paths,plan,raw,records,positions,data)
    receipt=dict(status='verified',formal_archive_verified=not plan['smoke'],smoke=plan['smoke'],plan_hash=plan['plan_hash'],plan_file_sha256=sha(paths['plan']),mode=args.command,summary=summary,evidence=files,local_inputs={name:{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for name,path in paths.items() if name in ('plan','positions','test_records','base_data','old_protocol','diagnostic_manifest','prepared','global_marker','training_receipt')},verifier_sha256=sha(__file__),helper_sha256={str(path.relative_to(ROOT)):sha(path) for path in (ROOT/'code/run_full_final_test100k.py',ROOT/'code/run_cross_multiseed.py',ROOT/'code/cross_pool_cache.py',ROOT/'tools/verify_training_diagnostics_archive.py')},test_future_targets_read=False,
        limitations=['Original plan bytes unchanged; remote refs, authorization and source/transfer authenticity require the independent root source mapping gate. Local prepared/training/checkpoint/global-marker bindings are independently verified.', 'Trusted local base pickle only; no future targets read or re-scoring performed. Test labels are sealed observations, not regenerated.', 'NDCG per-user finite/range and mean verified; full-future IDCG cannot be reconstructed from candidate labels without reopening targets.', 'Consumed UID set, seed42 order and mixed16 source counts verified; negative IDs are not replayed from the sealed consumption digest.', 'Receipt verifies local evidence; no SSH, deletion, shutdown or test reopening is performed.'])
    filename='TRAINING_COMPLETED.json' if args.command=='training' else 'TEST_COMPLETED.json';receipt[digest_field]=sha(paths['run_dir']/filename)
    paths['receipt'].parent.mkdir(parents=True,exist_ok=True)
    with paths['receipt'].open('x') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['training','test'])
    for name in ('plan','run-dir','raw-run','test-records','positions','base-data','receipt','old-protocol','diagnostic-manifest'):parser.add_argument('--'+name,required=True)
    for name in ('prepared','global-marker','training-run','training-receipt'):parser.add_argument('--'+name)
    parser.add_argument('--allow-smoke',action='store_true');args=parser.parse_args(argv);value=verify(args);print(json.dumps({'status':value['status'],'formal_archive_verified':value['formal_archive_verified'],'mode':value['mode'],'summary':value['summary']}))

if __name__=='__main__':main()
