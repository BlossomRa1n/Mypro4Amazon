"""Posthoc descriptive exposure groups; no causal inference, CI or selection."""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import analyze_training_exposure as e

HELPER_SHA='a5f5e160b7dc419ca7b577108e4d72638994d83800685a878cd4f2cc38c0f04b'
STABLE=('uid','position','candidate_ids','lengths','labels','pool_hit','auc_valid')
need=e.need


def load_arrays(path,records):
    with np.load(path,allow_pickle=False) as handle:value={k:handle[k].copy() for k in handle.files}
    need(set(STABLE)|{'hit5','ndcg5','scores','auc'}<=set(value),'missing array fields')
    need(all(np.isfinite(v).all() for v in value.values()),'nonfinite array')
    n=len(records);need(len(set(map(int,value['uid'])))==n,'duplicate UID')
    for key,dtype in (('uid','int64'),('position','int64'),('candidate_ids','int32'),('lengths','int32'),('hit5','int8'),('pool_hit','int8'),('ndcg5','float32'),('scores','float32'),('auc','float64')):need(value[key].dtype==np.dtype(dtype),'array dtype differs: '+key)
    for k in ('uid','position','lengths','pool_hit','auc_valid','hit5','ndcg5','auc'):need(value[k].shape==(n,),'array shape differs')
    need(np.array_equal(value['uid'],[x[0] for x in records]) and np.array_equal(value['position'],[x[1] for x in records]),'record UID/position order differs')
    ids=value['candidate_ids'];need(ids.ndim==2 and ids.shape[0]==n and value['scores'].shape==value['labels'].shape==ids.shape,'candidate shape differs')
    need(value['labels'].dtype==value['auc_valid'].dtype==np.dtype('bool'),'mask dtype differs')
    for key in ('hit5','pool_hit'):need(np.isin(value[key],[0,1]).all(),'invalid binary metric')
    need(np.all((value['ndcg5']>=0)&(value['ndcg5']<=1)),'NDCG outside bounds')
    for i,length in enumerate(value['lengths']):
        count=int(length);need(count==length and 0<=count<=ids.shape[1],'invalid length')
        row=ids[i,:count];labels=value['labels'][i,:count]
        need(np.all(row>=2) and len(set(map(int,row)))==count and np.all(ids[i,count:]==0) and not value['labels'][i,count:].any(),'candidate IDs/padding differs')
        need(bool(value['pool_hit'][i])==bool(labels.any()) and bool(value['auc_valid'][i])==bool(labels.any() and (~labels).any()),'candidate coverage/mask differs')
        need(value['hit5'][i]<=value['pool_hit'][i],'hit outside pool')
    return value


def grouped(old,new,masks):
    n=len(old['uid']);need(n>0,'empty cohort')
    need(np.array_equal(np.sum(list(masks.values()),axis=0),np.ones(n,dtype=int)),'groups not disjoint exhaustive')
    result={};weights={k:0. for k in ('hit5','ndcg5')}
    for name,mask in masks.items():
        count=int(mask.sum());row=dict(users=count,fraction=count/n,old=None,new=None,delta=None)
        if count:
            row['old']={k:float(old[k][mask].astype(float).mean()) for k in ('hit5','ndcg5','pool_hit')}
            row['new']={k:float(new[k][mask].astype(float).mean()) for k in ('hit5','ndcg5','pool_hit')}
            row['delta']={k:float((new[k][mask].astype(float)-old[k][mask].astype(float)).mean()) for k in ('hit5','ndcg5')}
            for k in weights:weights[k]+=row['delta'][k]*count/n
        result[name]=row
    overall={k:float((new[k].astype(float)-old[k].astype(float)).mean()) for k in weights}
    need(all(math.isclose(weights[k],overall[k],rel_tol=0,abs_tol=1e-12) for k in weights),'weighted overall delta not recovered')
    return dict(groups=result,overall_delta=overall,weighted_group_delta=weights,weighted_reconstruction_verified=True)


def analyze(args):
    need(e.sha(e.__file__)==HELPER_SHA,'reviewed identity helper changed')
    roots={k:e.safe(getattr(args,k)) for k in ('run_dir','raw_run','protocol_dir')};sources={str(e.safe(e.__file__)):HELPER_SHA}
    def tracked(path):
        path=e.safe(path);digest=e.sha(path)
        need(str(path) not in sources or sources[str(path)]==digest,'input drift');sources[str(path)]=digest;return path
    done={key:e.read(tracked(roots[key]/'COMPLETED.json')) for key in ('run_dir','raw_run')}
    newgate=e.read(tracked(args.verification_receipt));oldgate=e.read(tracked(args.raw_verification_receipt))
    for gate in (newgate,oldgate):need(gate['status']=='verified' and (args.allow_smoke or gate.get('formal_archive_verified') is True),'independent formal verification required')
    need(newgate['run']['completion_sha256']==sources[str(roots['run_dir']/'COMPLETED.json')] and oldgate['completion_sha256']==sources[str(roots['raw_run']/'COMPLETED.json')],'verification completion binding differs')
    need(newgate['run']['raw_archive_receipt_sha256']==sources[str(e.safe(args.raw_verification_receipt))],'raw verification receipt binding differs')
    for key,completion in done.items():need(completion['status']=='complete' and set(completion['seeds'])=={'42','43','44'} and not (roots[key]/'FAILED.json').exists(),'three completed seeds required')
    def sealed(key,name):
        p=tracked(roots[key]/name);need(p.is_relative_to(roots[key]) and sources[str(p)]==done[key]['evidence_hashes'][name],'evidence seal differs: '+name);return p
    manifest=e.read(sealed('run_dir','run_manifest.json'))
    need(manifest['controls']['raw_completion']['sha256']==sources[str(roots['raw_run']/'COMPLETED.json')],'old control differs')
    identity=e.analyze(SimpleNamespace(protocol_dir=args.protocol_dir,old_pool=args.old_pool,raw_run=args.raw_run,selected_steps=None,allow_smoke=args.allow_smoke))
    need(all(path not in sources or sources[path]==digest for path,digest in identity['input_sha256'].items()),'identity input changed between readers')
    sources.update(identity['input_sha256'])
    protocol=e.read(tracked(roots['protocol_dir']/'protocol_manifest.json'))
    records={split:e.read(tracked(roots['protocol_dir']/protocol['records'][split])) for split in ('train','screen','confirm')}
    positions=np.load(tracked(roots['protocol_dir']/protocol['ranker_train_positions']),allow_pickle=False)
    cache=e.read(tracked(args.old_pool));groups,ordered,_=e.reconstruct(records['train'],positions,cache['records_hash'])
    lookup={row[0]:i for i,row in enumerate(ordered)}
    batch=e.read(sealed('raw_run','run_config.json'))['batch_size'];steps=math.ceil(len(positions)/batch);checks=sorted(set(math.ceil(p*steps) for p in (.25,.5,1.)))
    newlock=e.read(sealed('run_dir','CHECKPOINTS_LOCKED.json'));oldlock=e.read(sealed('raw_run','CHECKPOINTS_LOCKED.json'))
    output={}
    for seed in (42,43,44):
        s=str(seed);order=np.random.default_rng(seed).permutation(len(positions)).astype(np.int64)
        trace=e.read(sealed('run_dir',f'seed_{seed}/raw_v2_train/epoch_1_trace.json'));oldtrace=e.read(sealed('raw_run',f'seed_{seed}/raw/epoch_1_trace.json'))
        for k in ('rows','order_sha256','batch_sizes','covered_users'):need(trace[k]==oldtrace[k],'exposure order differs between arms')
        need(trace['order_sha256']==e.array_hash(order),'reconstructed RNG order differs')
        def counts(step,split):
            allcounts=np.bincount(groups[order[:min(step*batch,len(order))]],minlength=len(ordered))
            return allcounts[[lookup[row[0]] for row in records[split]]]
        def pair(name,split):
            a=load_arrays(sealed('raw_run',f'seed_{seed}/raw/{name}'),records[split]);b=load_arrays(sealed('run_dir',f'seed_{seed}/raw_v2_train/{name}'),records[split])
            need(set(a)==set(b) and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a),'array dtype/shape differs')
            need(all(np.array_equal(a[k],b[k]) for k in STABLE),'paired identities/candidates/labels/masks differ')
            return a,b
        screen=[]
        for step in checks:
            a,b=pair(f'screen_step{step}_users.npz','screen');c=counts(step,'screen')
            screen.append(dict(step=step,**grouped(a,b,{'zero':c==0,'1':c==1,'2-3':(c>=2)&(c<=3),'4+':c>=4})))
        oldstep=oldlock[s]['best']['step'];newstep=newlock[s]['step'];need(oldstep in checks and newstep in checks,'selected step outside registered grid')
        need(newlock[s]==done['run_dir']['seeds'][s]['best'] and oldlock[s]['best']==done['raw_run']['seeds'][s]['best'],'checkpoint lock differs')
        a,b=pair('confirm_best_users.npz','confirm');oc=counts(oldstep,'confirm')>0;nc=counts(newstep,'confirm')>0
        confirm=grouped(a,b,{'neither_seen':~oc&~nc,'only_old_seen':oc&~nc,'only_new_seen':~oc&nc,'both_seen':oc&nc})
        expected=done['run_dir']['paired']['per_seed'][s]
        need(all(math.isclose(confirm['overall_delta'][k],expected[k+'_delta'],rel_tol=0,abs_tol=1e-12) for k in ('hit5','ndcg5')),'confirm does not recover registered per-seed delta')
        output[s]=dict(screen=screen,confirm=dict(old_best_step=oldstep,new_best_step=newstep,**confirm))
    overall={k:float(np.mean([output[s]['confirm']['overall_delta'][k] for s in output])) for k in ('hit5','ndcg5')}
    need(all(math.isclose(overall[k],done['run_dir']['paired'][k]['mean_delta'],rel_tol=0,abs_tol=1e-12) for k in overall),'three-seed mean differs from registered result')
    result=dict(scope='Posthoc descriptive exposure groups on reused development data; no causal attribution, confidence intervals, new thresholds, scoring or model selection.',seeds=output,overall_three_seed_confirm_delta=overall,empty_group_policy='old/new/delta are null when users=0',candidate_coverage_metric='pool_hit: proportion of users with at least one positive in fixed candidate pool',seen_definition='At least one own legal training prefix consumed; not a claim that unseen-user parameters were unchanged.',input_sha256=sources,script_sha256=e.sha(__file__),identity_reconstruction=identity['mapping_certification'],bootstrap_performed=False,training_or_scoring_performed=False,selection_performed=False)
    need(all(e.sha(path)==digest for path,digest in sources.items()),'source changed during analysis')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','raw-run','protocol-dir','old-pool','verification-receipt','raw-verification-receipt','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--allow-smoke',action='store_true');args=p.parse_args(argv)
    output=e.safe(args.output);need(not output.exists(),'new output required')
    for root in (args.run_dir,args.raw_run,args.protocol_dir,Path(args.old_pool).parent):need(not output.is_relative_to(e.safe(root)),'output in input tree')
    result=analyze(args)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
