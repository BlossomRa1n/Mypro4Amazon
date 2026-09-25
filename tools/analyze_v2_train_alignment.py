"""Read-only descriptive summaries of completed, sealed development artifacts.

No model loading, training, scoring, bootstrap, new threshold or selection.
Candidate changes describe fused rankings, not V2-exclusive channel provenance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SEEDS=('42','43','44')


def need(value,message):
    if not value:raise ValueError(message)


def safe(path):
    p=Path(path).absolute()
    need('..' not in p.parts and not any(x.is_symlink() for x in (p,*p.parents)),'unsafe path')
    return p


def sha(path):
    h=hashlib.sha256()
    with safe(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        result={}
        for k,v in items:need(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(safe(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def seal(root):
    root=safe(root);done=read(root/'COMPLETED.json')
    need(done['status']=='complete' and not (root/'FAILED.json').exists(),'incomplete input')
    evidence=done['evidence_hashes'];need(evidence,'empty seal')
    for name,digest in evidence.items():
        path=safe(root/name);need(path.is_relative_to(root) and sha(path)==digest,'evidence drift: '+name)
    need({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(evidence)|{'COMPLETED.json'},'unsealed files')
    return done


def bound(reference,path):
    need(sha(path)==reference['sha256'],'reference SHA differs')
    if 'bytes' in reference:need(safe(path).stat().st_size==reference['bytes'],'reference bytes differ')


def pool(path):
    path=safe(path);meta=read(path)
    need(meta['status']=='complete' and meta['schema']=='cross-pools-int32-v2' and meta['shape'][1]==75,'pool format mismatch')
    values={}
    for name in ('items','lengths'):
        item=meta['files'][name];p=safe(path.parent/item['name'])
        need(p.is_relative_to(path.parent) and sha(p)==item['sha256'],'pool payload drift')
        values[name]=np.load(p,allow_pickle=False,mmap_mode='r')
        need(values[name].dtype==np.dtype('int32'),'pool dtype mismatch')
    items,lengths=values['items'],values['lengths']
    need(list(items.shape)==meta['shape'] and lengths.shape==(len(items),),'pool shape mismatch')
    h=hashlib.sha256();h.update(b'[')
    for index,length in enumerate(lengths):
        n=int(length);ids=items[index,:n]
        need(0<=n<=75 and len(set(map(int,ids)))==n and np.all(ids>=2) and np.all(ids<meta['source']['catalog_size']) and np.all(items[index,n:]==0),'invalid pool IDs/padding')
        if index:h.update(b',')
        h.update(json.dumps(ids.tolist(),separators=(',',':')).encode())
    h.update(b']');need(h.hexdigest()==meta['pool_hash'],'pool logical SHA differs')
    return meta,items,lengths


def describe(values):
    a=np.asarray(values,dtype=np.float64);need(len(a)>0 and np.isfinite(a).all(),'nonfinite/empty observations')
    return dict(mean=float(a.mean()),quantiles=dict(zip(('min','p25','median','p75','p95','max'),map(float,np.quantile(a,[0,.25,.5,.75,.95,1])))))


def compare_pools(old,new):
    _,a,al=old;_,b,bl=new;need(len(a)==len(b),'pool rows differ')
    overlaps=[];jaccard=[];added=[];removed=[];ordered=setsame=0;bands={name:dict(overlap=[],removed=[],added=[],changed=0) for name in ('rank11_25','rank26_50')}
    for x,n,y,m in zip(a,al,b,bl):
        x=list(map(int,x[:int(n)]));y=list(map(int,y[:int(m)]));sx,sy=set(x),set(y)
        overlaps.append(len(sx&sy));ordered+=x==y;setsame+=sx==sy
        jaccard.append(len(sx&sy)/len(sx|sy) if sx|sy else 1.);added.append(len(sy-sx));removed.append(len(sx-sy))
        for name,start,end in (('rank11_25',10,25),('rank26_50',25,50)):
            u,v=set(x[start:end]),set(y[start:end]);band=bands[name]
            band['overlap'].append(len(u&v));band['removed'].append(len(u-v));band['added'].append(len(v-u));band['changed']+=u!=v
    return dict(rows=len(a),overlap_count=describe(overlaps),jaccard=describe(jaccard),added_count=describe(added),removed_count=describe(removed),identical_order_rows=ordered,identical_order_fraction=ordered/len(a),identical_set_rows=setsame,identical_set_fraction=setsame/len(a),bands={name:dict(member_change_fraction=value['changed']/len(a),**{k:describe(value[k]) for k in ('overlap','removed','added')}) for name,value in bands.items()},interpretation='Fused ranking membership changes; added IDs are not identified as V2-exclusive recall sources. Jaccard is intersection/union; two empty sets have Jaccard 1.')


def numeric_leaves(value,prefix=''):
    result={}
    if isinstance(value,dict):
        for key,item in value.items():result.update(numeric_leaves(item,prefix+('.' if prefix else '')+key))
    elif isinstance(value,(int,float)) and not isinstance(value,bool):
        need(math.isfinite(value),'nonfinite diagnostic');result[prefix]=value
    return result


def summarize_run(root,done,variant):
    need(set(done['seeds'])==set(SEEDS),'three complete seeds required')
    curves={};stats={};by_point=[[],[],[]]
    for seed in SEEDS:
        base=root/f'seed_{seed}/{variant}';result=read(base/'COMPLETED.json')
        need(result==done['seeds'][seed],'seed completion differs')
        steps=result['steps_per_epoch'];checks=sorted(set(math.ceil(p*steps) for p in (.25,.5,1.)))
        history=[row for row in read(base/'history.json') if row['step']<=steps]
        need(len(checks)==3 and [x['step'] for x in history]==checks,'three screen points required')
        curves[seed]=[dict(step=x['step'],epoch_fraction=x['epoch_fraction'],screen=x['screen'],diagnostics=x['diagnostics']) for x in history]
        for i,x in enumerate(history):by_point[i].append(numeric_leaves(dict(screen=x['screen'],diagnostics=x['diagnostics'])))
        rows=[json.loads(line) for line in (base/'steps.jsonl').read_text().splitlines()]
        rows=[x for x in rows if x['epoch']==1]
        need([x['step'] for x in rows]==list(range(1,steps+1)),'epoch1 step sequence differs')
        trace=read(base/'epoch_1_trace.json');need(sum(trace['batch_sizes'])==trace['rows'],'trace rows differ')
        losses=[x['loss'] for x in rows];grad=[x['gradient_l2_before_clip'] for x in rows]
        need(sum(trace['negative_sources'].values())==trace['rows']*16,'negative source accounting differs')
        stats[seed]=dict(best_step=result['best']['step'],negative_sources=trace['negative_sources'],negative_source_fraction={k:v/(16*trace['rows']) for k,v in trace['negative_sources'].items()},epoch1_loss=describe(losses),epoch1_gradient_before_clip=describe(grad),epoch1_clip_needed_steps=sum(x>5 for x in grad),epoch1_steps=steps,all_loss_gradient_finite=True,epoch1_train_seconds=history[-1]['train_seconds_cumulative'],epoch1_eval_seconds=history[-1]['eval_seconds_cumulative'],whole_run_train_seconds=result['train_seconds'],whole_run_eval_seconds=result['eval_seconds'])
    means=[]
    for i,observations in enumerate(by_point):
        keys=set(observations[0]);need(all(set(x)==keys for x in observations),'diagnostic fields differ across seeds')
        means.append(dict(requested_epoch_fraction=(.25,.5,1.)[i],mean_across_three_seeds={k:float(np.mean([x[k] for x in observations])) for k in sorted(keys)}))
    return dict(per_seed_curves=curves,three_seed_point_means=means,per_seed_training=stats,aggregation_note='Point summaries average the three seed-level numbers, including seed-level quantiles; they are not pooled-user quantiles. Whole-run timing includes all original epochs; epoch1 timing uses the registered 1.0 screen cumulative counters.')


def analyze(args):
    run,raw,newroot=map(safe,(args.run_dir,args.raw_run,args.pool_dir))
    done,old,complete=seal(run),seal(raw),seal(newroot)
    need(set(done['seeds'])==set(old['seeds'])==set(SEEDS),'three complete seeds required')
    need(complete.get('mode')=='full' and complete.get('rows')>0,'full training pool completion required')
    need(done.get('test_future_labels_read') is False and old.get('test_future_labels_read') is False,'test policy mismatch')
    inputs={}
    for root,completion in ((run,done),(raw,old),(newroot,complete)):
        inputs.update({str(root/name):digest for name,digest in completion['evidence_hashes'].items()})
        inputs[str(root/'COMPLETED.json')]=sha(root/'COMPLETED.json')
    manifest=read(run/'run_manifest.json');bound(manifest['controls']['raw_completion'],raw/'COMPLETED.json');bound(manifest['train_cache']['receipt'],newroot/'COMPLETED.json')
    need(manifest['train_cache']['source']==complete['source'] and manifest['train_cache']['pool_hash']==complete['pool_hash'],'pool source differs from run')
    bound(manifest['protocol_manifest'],args.protocol_manifest);protocol=read(args.protocol_manifest)
    bound(protocol['candidate_caches']['train'],args.old_pool)
    bound(complete['cache'],newroot/Path(complete['cache']['path']).name)
    bound(complete['positions'],newroot/'ranker_train_positions.npy')
    positions=np.load(safe(args.old_positions),allow_pickle=False);newpositions=np.load(newroot/'ranker_train_positions.npy',allow_pickle=False)
    need(positions.dtype==newpositions.dtype==np.dtype('int64') and positions.ndim==1 and np.array_equal(positions,newpositions),'training positions differ')
    ph=hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest()
    need(ph==complete['source']['positions_sha256']==read(raw/'run_config.json')['train_positions_sha256'],'position source binding differs')
    newpool=pool(newroot/Path(complete['cache']['path']).name);oldpool=pool(args.old_pool)
    need(newpool[0]['source']['rrf_weights']==[2,1,.7,.05] and oldpool[0]['source']['rrf_weights']==[2,0,.7,.05],'pool treatment mismatch')
    need(newpool[0]['records_hash']==oldpool[0]['records_hash'] and len(positions)==len(newpool[1]),'pool record identities differ')
    for key in ('base_data_id','source_hashes','encoders_hash','code_hashes','cf_neighbors','catalog_size','itemcf_half_life_days','candidate_policy'):
        need(newpool[0]['source'][key]==oldpool[0]['source'][key],'pool common source differs: '+key)
    need(bool(done['smoke'])==bool(args.allow_smoke),'smoke requires explicit flag')
    need(args.allow_smoke or len(positions)==604511,'formal pool rows differ')
    for p in map(safe,(args.protocol_manifest,args.old_positions,args.old_pool)):inputs[str(p)]=sha(p)
    for item in oldpool[0]['files'].values():
        p=safe(Path(args.old_pool).parent/item['name']);inputs[str(p)]=sha(p)
    paired=read(run/'paired_confirm.json');need(done['paired']==paired,'registered decision seal differs')
    result=dict(scope='Read-only descriptive development summary; no scoring/bootstrap/new thresholds or branch selection.',new=summarize_run(run,done,'raw_v2_train'),old=summarize_run(raw,old,'raw'),pool=compare_pools(oldpool,newpool),registered_decision_unchanged=paired,input_sha256=inputs,analyzer_sha256=sha(__file__),training_or_scoring_performed=False,bootstrap_performed=False)
    need(all(sha(p)==digest for p,digest in inputs.items()),'input changed during summary')
    need(read(run/'COMPLETED.json')==done and read(raw/'COMPLETED.json')==old and read(newroot/'COMPLETED.json')==complete,'completion changed during summary')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','raw-run','pool-dir','old-pool','old-positions','protocol-manifest','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--allow-smoke',action='store_true');args=parser.parse_args(argv)
    output=safe(args.output)
    need(not output.exists(),'new output required')
    for root in (args.run_dir,args.raw_run,args.pool_dir):need(not output.is_relative_to(safe(root)),'output within sealed input')
    result=analyze(args)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
