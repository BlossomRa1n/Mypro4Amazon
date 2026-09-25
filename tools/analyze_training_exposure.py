"""Identity-only exposure at registered early stops; no dataset or labels loaded.

Validation-boundary reconstruction is provisional until the exact canonical
training-record hash agrees with the sealed original candidate metadata.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SEEDS=(42,43,44)


def need(value,message):
    if not value:raise ValueError(message)


def safe(path):
    path=Path(path).absolute()
    need('..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),'unsafe path')
    return path


def sha(path):
    h=hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(rows):
        result={}
        for k,v in rows:need(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(safe(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def rows_hash(rows):
    h=hashlib.sha256();h.update(b'[')
    for i,row in enumerate(rows):
        if i:h.update(b',')
        h.update(json.dumps(list(row),ensure_ascii=True,separators=(',',':')).encode())
    h.update(b']');return h.hexdigest()


def array_hash(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def reconstruct(train,positions,expected_hash):
    need(positions.dtype==np.dtype('int64') and positions.ndim==1 and len(positions)>0 and np.all(positions>=0) and np.all(positions[1:]>positions[:-1]),'invalid positions')
    need(all(isinstance(x,list) and len(x)==3 and type(x[0]) is int and x[0]>=0 and type(x[1]) is int and x[1]>0 and isinstance(x[2],str) for x in train),'invalid identity records')
    need(len({x[0] for x in train})==len(train)==len({x[1] for x in train})==len({x[2] for x in train}),'duplicate training identity/boundary')
    ordered=sorted(train,key=lambda x:x[1]);boundaries=np.asarray([x[1] for x in ordered],dtype=np.int64)
    groups=np.searchsorted(boundaries,positions,side='right')
    need(np.all(groups<len(ordered)),'position not below any validation boundary')
    uids=np.asarray([x[0] for x in ordered],dtype=np.int64);raw=[x[2] for x in ordered]
    actual=rows_hash((int(uids[g]),int(p),raw[g]) for p,g in zip(positions,groups))
    need(actual==expected_hash,'boundary reconstruction does not match sealed training-record hash')
    need(len(np.unique(groups))==len(train),'some training users have no reconstructed rows')
    return groups,ordered,actual


def distribution(counts):
    need(len(counts)>0,'empty user cohort')
    return dict(users=len(counts),seen_users=int(np.count_nonzero(counts)),unseen_users=int(np.count_nonzero(counts==0)),consumed_rows=int(counts.sum()),mean_rows_per_user=float(counts.mean()),consumption_quantiles_all_users_including_zero=dict(zip(('min','p25','median','p75','p95','max'),map(float,np.quantile(counts,[0,.25,.5,.75,.95,1])))))


def exposure(groups,order,step,batch_size,cohorts):
    consumed=min(step*batch_size,len(order));counts=np.bincount(groups[order[:consumed]],minlength=len(cohorts['train']))
    return dict(step=step,consumed_rows=consumed,unique_users=int(np.count_nonzero(counts)),cohorts={name:distribution(counts[indexes]) for name,indexes in cohorts.items()})


def analyze(args):
    root=safe(args.protocol_dir);raw=safe(args.raw_run);inputs={}
    def tracked(path):
        path=safe(path);inputs[str(path)]=sha(path);return path
    protocol=read(tracked(root/'protocol_manifest.json'))
    need(protocol['manifest_hash']==canonical({k:v for k,v in protocol.items() if k!='manifest_hash'}),'protocol canonical hash differs')
    need(protocol.get('test_future_labels_read') is False,'protocol test policy mismatch')
    records={}
    for split in ('train','screen','confirm'):
        path=safe(root/protocol['records'][split]);need(path.is_relative_to(root),'record path escape')
        records[split]=read(tracked(path));need(rows_hash(records[split])==protocol['cohort_hashes'][split] and len(records[split])==protocol['counts'][split],'cohort identity seal differs')
        need(len({tuple(row) for row in records[split]})==len(records[split])==len({row[0] for row in records[split]}),'duplicate cohort identity/UID')
    train={tuple(x) for x in records['train']};screen={tuple(x) for x in records['screen']};confirm={tuple(x) for x in records['confirm']}
    need(not screen&confirm and train==screen|confirm and len(screen)+len(confirm)==len(records['train']),'development cohorts not exact disjoint training partition')
    path=safe(root/protocol['ranker_train_positions']);need(path.is_relative_to(root),'position path escape')
    positions=np.load(tracked(path),allow_pickle=False,mmap_mode='r')
    need(array_hash(positions)==protocol['ranker_train_positions_sha256'] and len(positions)==protocol['ranker_train_rows'],'position seal differs')
    cache=read(tracked(args.old_pool));source=cache['source']
    need(cache['status']=='complete' and cache['schema']=='cross-pools-int32-v2' and cache['shape']==[len(positions),75] and source['rrf_weights']==[2,0,.7,.05],'old training cache contract differs')
    need(source['base_data_id']==protocol['base_data_id'] and source['encoders_hash']==protocol['encoders_hash'] and source['source_hashes']==protocol['base_assets'],'old cache base source differs')
    need(source['records_count']==len(positions) and cache['records_hash']==cache['position_uid_row_hash']==source['records_hash'],'old cache record seal differs')
    groups,ordered,actual=reconstruct(records['train'],positions,cache['records_hash'])
    identity_index={tuple(row):i for i,row in enumerate(ordered)}
    cohorts={split:np.asarray([identity_index[tuple(row)] for row in rows],dtype=np.int64) for split,rows in records.items()}
    completion=read(tracked(raw/'COMPLETED.json'));configpath=tracked(raw/'run_config.json');config=read(configpath)
    need(completion['status']=='complete' and completion['test_future_labels_read'] is False and set(completion['seeds'])==set(map(str,SEEDS)),'old raw completion differs')
    need(not (raw/'FAILED.json').exists() and sha(configpath)==completion['evidence_hashes']['run_config.json'],'old raw config seal differs')
    need(config['train_positions_sha256']==array_hash(positions),'old raw position identity differs')
    lockedpath=tracked(raw/'CHECKPOINTS_LOCKED.json');need(sha(lockedpath)==completion['evidence_hashes']['CHECKPOINTS_LOCKED.json'],'old lock seal differs');locked=read(lockedpath)
    selected=read(tracked(args.selected_steps)) if args.selected_steps else {str(seed):locked[str(seed)]['best']['step'] for seed in SEEDS}
    need(set(selected)==set(map(str,SEEDS)),'selected seed set differs')
    batch=config['batch_size'];steps=math.ceil(len(positions)/batch);checks=sorted(set(math.ceil(p*steps) for p in (.25,.5,1.)))
    need(bool(protocol['smoke'])==bool(args.allow_smoke),'smoke flag differs')
    need(args.allow_smoke or (len(positions)==604511 and len(train)==100000 and len(screen)==20000 and len(confirm)==80000 and batch==256 and checks==[591,1181,2362]),'formal exposure contract differs')
    output={}
    for seed in SEEDS:
        name=f'seed_{seed}/raw/epoch_1_trace.json';tracepath=tracked(raw/name);need(sha(tracepath)==completion['evidence_hashes'][name],'raw trace seal differs');trace=read(tracepath)
        order=np.random.default_rng(seed).permutation(len(positions)).astype(np.int64)
        sizes=[min(batch,len(positions)-i) for i in range(0,len(positions),batch)]
        need(trace['order_sha256']==array_hash(order) and trace['rows']==len(positions) and trace['batch_sizes']==sizes and trace['covered_users']==len(train),'registered row order/coverage differs')
        chosen=selected[str(seed)];need(type(chosen) is int and chosen in checks,'selected step outside registered grid')
        output[str(seed)]=dict(order_sha256=array_hash(order),grid=[exposure(groups,order,step,batch,cohorts) for step in checks],selected=exposure(groups,order,chosen,batch,cohorts))
    result=dict(scope='Identity-only descriptive exposure; a seen user means at least one legal training prefix consumed. No model quality or causal claim.',mapping_certification=dict(method='Sorted validation boundaries, first boundary strictly greater than training position; accepted only after exact canonical old cache record SHA agreement.',records_hash=actual,rows=len(positions),users=len(train)),selected_steps_source='explicit caller identity-only step map' if args.selected_steps else 'sealed old raw best lock',seeds=output,input_sha256=inputs,analyzer_sha256=sha(__file__),data_pickle_loaded=False,labels_loaded=False,training_or_scoring_performed=False,decision_thresholds_changed=False)
    need(all(sha(path)==digest for path,digest in inputs.items()),'input changed during exposure analysis')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol-dir','old-pool','raw-run','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--selected-steps');parser.add_argument('--allow-smoke',action='store_true');args=parser.parse_args(argv)
    output=safe(args.output);need(not output.exists(),'new output required')
    for root in (args.protocol_dir,args.raw_run,Path(args.old_pool).parent):need(not output.is_relative_to(safe(root)),'output overlaps input tree')
    result=analyze(args)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
