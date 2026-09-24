"""Verify mapped local full-prefix pool archives without generating candidates.

No future-target calls, model loading/scoring, SSH or source-reference rewrites.
Formal frozen builder/legacy hashes are authenticated before import. Trusted base
pickle bytes are verified before loading. Original remote/local path mapping and
transfer provenance remain a separately checked root gate.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

DIAGNOSTIC='c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6'
OLD_SHA='4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1'
FORMAL_ROWS=4731777
FORMAL_DATA_ID='754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158'


def require(condition,message):
    if not condition:raise ValueError(message)


def safe(value,root=None):
    path=Path(value).absolute();require('..' not in path.parts,'parent traversal forbidden')
    for item in [path,*path.parents]:require(not item.is_symlink(),'symlink evidence forbidden')
    if root is not None:require(path.is_relative_to(root),'evidence escapes root')
    return path


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    import math
    value=json.loads(path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    def check(obj):
        if isinstance(obj,float):require(math.isfinite(obj),'nonfinite JSON number')
        elif isinstance(obj,dict):
            for child in obj.values():check(child)
        elif isinstance(obj,list):
            for child in obj:check(child)
    check(value);return value


def sealed_manifest(path):
    value=read(path);require(value.get('manifest_hash')==canonical({k:v for k,v in value.items() if k!='manifest_hash'}),'manifest canonical seal mismatch');return value


def source_for(old,data,records,cache,smoke):
    return dict(base_data_id=old['base_data_id'],source_hashes=old['base_assets'],records_hash=cache.rows_hash(records),records_count=len(records),budget=75,candidate_policy='four-way RRF from train corpus; no future target filtering',rrf_weights=[2.,0.,.7,.05],itemcf_half_life_days=180.,smoke_fallback=smoke,encoders_hash=old['encoders_hash'],code_hashes=old['code_hashes'],cf_neighbors=300,catalog_size=len(data.items),padding='zero tail, lengths define valid ordered IDs')


def check_tree(directory):
    for path in directory.rglob('*'):require(not path.is_symlink() and (path.is_file() or path.is_dir()),'symlink/special archive member')
    require(not os.path.lexists(directory/'FAILED.json'),'FAILED pool archive')


def checked_payload(directory,receipt,np,cache,source,mode,full_positions):
    check_tree(directory)
    require(receipt.get('schema')=='full-prefix-pools-v1' and receipt.get('status')=='complete' and receipt.get('mode')==mode,'pool completion mode/status mismatch')
    require(receipt.get('cpu_only') is True and receipt.get('test_future_labels_read') is False and receipt.get('test_pool_constructed') is False,'pool test-isolation mismatch')
    require(type(receipt['workers']) is int and 1<=receipt['workers']<=4,'invalid worker envelope')
    for key in ('cache_path','positions_path'):require(Path(receipt[key]).name==receipt[key],'unsafe pool payload name')
    expected_cache='candidate_final_train.json' if mode=='full' else 'candidate_pilot.json';require(receipt['cache_path']==expected_cache and receipt['positions_path']=='ranker_train_positions.npy','pool payload role mismatch')
    pospath=safe(directory/receipt['positions_path'],directory);require(sha(pospath)==receipt['positions_sha256'],'positions file SHA mismatch');positions=np.load(pospath,allow_pickle=False,mmap_mode='r')
    require(positions.dtype==np.dtype('int64') and positions.ndim==1 and len(positions)==receipt['selected_rows'],'positions dtype/row mismatch')
    expected=full_positions if mode=='full' else full_positions[np.linspace(0,len(full_positions)-1,len(positions),dtype=np.int64)]
    require(np.array_equal(positions,expected),'full/pilot position order mismatch')
    cachepath=safe(directory/receipt['cache_path'],directory);require(sha(cachepath)==receipt['cache_sha256'],'cache sidecar SHA mismatch')
    metadata=read(cachepath)
    for entry in metadata['files'].values():require(Path(entry['name']).name==entry['name'],'unsafe array payload name')
    required={'COMPLETED.json','CONSTRUCTION_STARTED.json','DISK_PREFLIGHT.json','ranker_train_positions.npy',expected_cache,expected_cache+'.building'}|{entry['name'] for entry in metadata['files'].values()}
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==required,'pool archive missing/extra files')
    start=read(directory/'CONSTRUCTION_STARTED.json')
    require(start.get('status')=='building' and all(receipt.get(k)==v for k,v in start.items() if k!='status'),'construction/completion binding differs')
    disk=read(directory/'DISK_PREFLIGHT.json');require(disk==receipt['disk_preflight'] and disk.get('schema')=='full-prefix-disk-preflight-v1' and disk.get('status')=='passed','disk preflight binding mismatch')
    payload=len(positions)*(75*4+4+8)
    require(disk['rows']==len(positions) and disk['payload_bytes']==payload and disk['metadata_allowance_bytes']>=1024**2 and disk['reserve_bytes']>=1024**3 and disk['required_free_bytes']==payload+disk['metadata_allowance_bytes']+disk['reserve_bytes'] and disk['observed_free_bytes']>=disk['required_free_bytes'],'disk preflight arithmetic/capacity mismatch')
    require(source==receipt['source'],'receipt source mismatch')
    pools,metadata=cache.load_cache(cachepath,source)
    require(metadata['pool_hash']==receipt['pool_hash'] and metadata['records_hash']==receipt['records_hash']==source['records_hash'],'pool logical seal mismatch')
    require(len(pools)==len(positions) and metadata['shape']==[len(positions),75],'pool shape mismatch')
    require(receipt['serial_parallel_check']['exact_ordered_equality'] is True and receipt['serial_parallel_check']['rows']==min(128,len(positions)),'registered serial-parallel check absent')
    return pools,metadata,{str(p.relative_to(directory)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(directory.rglob('*')) if p.is_file()}


def verify(args):
    paths={name:safe(getattr(args,name)) for name in ('full_run','old_protocol_dir','base_archive','diagnostic_manifest','legacy_source_dir','builder_source','pilot_run','receipt')}
    require(not os.path.lexists(paths['receipt']),'receipt already exists')
    require(all(not paths['receipt'].is_relative_to(paths[name]) for name in ('full_run','old_protocol_dir','base_archive','legacy_source_dir','pilot_run')),'receipt must be outside input archives')
    diagnostic=sealed_manifest(paths['diagnostic_manifest']);smoke=diagnostic.get('smoke');require(type(smoke) is bool and (not smoke or args.allow_smoke),'smoke requires explicit allow-smoke')
    require(diagnostic.get('protocol')=='raw-deterioration-20260924-v1' and diagnostic.get('test_evaluation_allowed') is False,'invalid diagnostic identity/policy')
    oldpath=paths['old_protocol_dir']/'protocol_manifest.json';old=sealed_manifest(oldpath)
    require(sha(oldpath)==diagnostic['old_protocol']['sha256'] and old.get('smoke')==smoke,'old protocol source mismatch')
    if not smoke:require(diagnostic['manifest_hash']==DIAGNOSTIC and sha(oldpath)==OLD_SHA,'unregistered formal source')
    check_tree(paths['legacy_source_dir']);code={p.name:sha(p) for p in paths['legacy_source_dir'].glob('*.py')};require(code==diagnostic['code_hashes'] and bool(code),'legacy source hash mismatch before import')
    full=read(paths['full_run']/'COMPLETED.json');pilot=read(paths['pilot_run']/'COMPLETED.json');builder_hash=sha(paths['builder_source'])
    require(builder_hash==args.builder_sha256 and full['builder_sha256']==builder_hash and pilot['builder_sha256']==builder_hash,'frozen builder hash mismatch')
    for receipt in (full,pilot):require(receipt['protocol_manifest_sha256']==sha(paths['diagnostic_manifest']) and receipt['old_protocol_sha256']==sha(oldpath) and receipt['frozen_code_hashes']==code,'pool receipt source binding mismatch')
    assets={}
    if not smoke:require({'data.pkl','svd.npy','itemcf.pkl','v2_best.pth','run_manifest.json'}<=set(old['base_assets']),'required full base assets unbound')
    for name,expected in old['base_assets'].items():
        require(Path(name).name==name,'unsafe base asset name');path=safe(paths['base_archive']/name,paths['base_archive']);require(sha(path)==expected,'base asset SHA mismatch before load: '+name);assets[name]={'path':str(path),'sha256':expected,'bytes':path.stat().st_size}
    require(assets['data.pkl']['sha256']==old['base_data_hash'],'base data file seal mismatch')
    for name in ('run_cross_multiseed','cross_pool_cache','baseline_data','future_window_data'):
        if name in sys.modules:require(Path(sys.modules[name].__file__).resolve().parent==paths['legacy_source_dir'],'fresh process required for frozen imports')
    sys.path.insert(0,str(paths['legacy_source_dir']));r=importlib.import_module('run_cross_multiseed');cache=importlib.import_module('cross_pool_cache');np=importlib.import_module('numpy')
    data=r.load_data(paths['base_archive'])
    def no_targets(*a,**k):raise PermissionError('pool archive verifier must never read targets')
    data.targets=no_targets;data.evaluation=no_targets
    require(data.manifest['data_id']==old['base_data_id'] and data.manifest['encoders_hash']==old['encoders_hash'],'base encoded identity mismatch')
    positions=np.asarray(data.train_positions)
    require(positions.dtype==np.dtype('int64') and positions.ndim==1 and len(positions)>0 and np.all(positions>=0) and np.all(positions<len(data.uid)) and np.all(positions[1:]>positions[:-1]),'invalid legal training positions')
    mask=np.arange(len(data.uid))<data.train_ends[data.uid];require(np.array_equal(mask,data.train_mask),'legal train mask drift')
    # Only train timestamps/item IDs are inspected; no future event values.
    legal=np.flatnonzero(mask);expected=legal[data.ts[legal]>data.ts[data.starts[data.uid[legal]]]]
    require(np.array_equal(positions,expected),'full legal prefix row coverage mismatch')
    require(np.array_equal(np.bincount(data.iid[mask],minlength=len(data.items)),data.train_counts),'train-only item counts mismatch')
    if not smoke:require(len(positions)==FORMAL_ROWS and data.manifest['data_id']==FORMAL_DATA_ID,'formal full row/data identity differs')
    records=cache.PositionRecords(data,positions);record_hash=cache.rows_hash(records)
    for receipt in (full,pilot):require(receipt['full_legal_rows']==len(positions) and receipt['full_records_hash']==record_hash,'full records identity mismatch')
    fullpools,metadata,files=checked_payload(paths['full_run'],full,np,cache,source_for(old,data,records,cache,smoke),'full',positions)
    pilotpositions=positions[np.linspace(0,len(positions)-1,pilot['selected_rows'],dtype=np.int64)];pilotrecords=cache.PositionRecords(data,pilotpositions)
    pilotpools,pilotmeta,pilotfiles=checked_payload(paths['pilot_run'],pilot,np,cache,source_for(old,data,pilotrecords,cache,smoke),'pilot',positions)
    for i,full_index in enumerate(np.linspace(0,len(positions)-1,len(pilotpositions),dtype=np.int64)):require(np.array_equal(pilotpools[i],fullpools[int(full_index)]),'pilot/full merged row mismatch')
    oldpospath=safe(paths['old_protocol_dir']/old['ranker_train_positions'],paths['old_protocol_dir']);oldpositions=np.load(oldpospath,allow_pickle=False,mmap_mode='r');require(r.sha256_array(oldpositions)==old['ranker_train_positions_sha256'],'old cache position seal mismatch')
    require(oldpositions.ndim==1 and np.issubdtype(oldpositions.dtype,np.integer) and np.all(oldpositions[1:]>oldpositions[:-1]),'old position order invalid')
    mapped=np.searchsorted(positions,oldpositions);require(np.all(mapped<len(positions)) and np.array_equal(positions[mapped],oldpositions),'old rows absent from full legal rows')
    oldcachepath=safe(paths['old_protocol_dir']/'candidate_train.json',paths['old_protocol_dir']);require(sha(oldcachepath)==diagnostic['candidate_caches']['train']['sha256'],'old cache sidecar seal mismatch')
    oldsource=source_for(old,data,cache.PositionRecords(data,oldpositions),cache,smoke);oldpools,oldmeta=cache.load_cache(oldcachepath,oldsource)
    indices=np.linspace(0,len(oldpositions)-1,min(128,len(oldpositions)),dtype=np.int64)
    for index in indices:require(np.array_equal(oldpools[int(index)],fullpools[int(mapped[index])]),'old/full independent overlap mismatch')
    if not smoke:
        require(pilot['selected_rows']>=2048,'pilot below registered 2048 rows')
        for receipt in (full,pilot):
            overlap=receipt['original_cache_overlap'];require(overlap['rows']==128 and overlap['exact_ordered_equality'] is True and overlap['existing_pool_hash']==oldmeta['pool_hash'] and overlap['existing_rows']==len(oldpositions),'registered old overlap receipt mismatch')
    receipt=dict(status='verified',formal_archive_verified=not smoke,smoke=smoke,builder_sha256=builder_hash,full_completion_sha256=sha(paths['full_run']/'COMPLETED.json'),pilot_completion_sha256=sha(paths['pilot_run']/'COMPLETED.json'),diagnostic_manifest_sha256=sha(paths['diagnostic_manifest']),old_protocol_sha256=sha(oldpath),rows=len(positions),full_data_id=data.manifest['data_id'],encoders_hash=data.manifest['encoders_hash'],records_hash=record_hash,pool_hash=metadata['pool_hash'],array_payload=metadata['files'],source=full['source'],files=files,pilot_files=pilotfiles,base_assets=assets,frozen_code_hashes=code,independent_old_overlap=dict(rows=len(indices),exact_ordered_equality=True,old_pool_hash=oldmeta['pool_hash']),independent_pilot_full_overlap=dict(rows=len(pilotpositions),exact_ordered_equality=True),test_future_labels_read=False,models_loaded=False,candidate_generation_performed=False,verifier_sha256=sha(Path(__file__)),limitations=['Local semantic/archive verification; original transfer and path mapping authenticity remain a root source gate.','Registered serial/parallel builder checks are validated as recorded evidence; no 4.7M candidate regeneration or model scoring.','Existing old-cache 128 rows and all pilot rows are independently compared against the full merged pool.'])
    paths['receipt'].parent.mkdir(parents=True,exist_ok=True)
    with paths['receipt'].open('x') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full-run','old-protocol-dir','base-archive','diagnostic-manifest','legacy-source-dir','builder-source','builder-sha256','pilot-run','receipt'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--allow-smoke',action='store_true');value=verify(parser.parse_args(argv));print(json.dumps({key:value[key] for key in ('status','formal_archive_verified','rows','full_data_id','pool_hash')}))

if __name__=='__main__':main()
