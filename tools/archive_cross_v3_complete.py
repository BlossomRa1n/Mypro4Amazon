#!/usr/bin/env python3
"""Read-only, one-shot complete dev+protocol archive. No scheduling or scoring."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import subprocess
import sys
import uuid
import numpy as np
import torch
import archive_cross_v3 as common
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cross_pool_cache import load_cache

REMOTE = r'''
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(sys.argv[1]).resolve();protocol=Path(sys.argv[2]).resolve()
def read(p):return json.loads(p.read_text())
marker=root/'COMPLETED.json'
if not marker.exists():
 print(json.dumps({'ready':False,'reason':'global completion marker absent'}));sys.exit(0)
c=read(marker)
if c.get('status')!='complete' or c.get('groups')!=9 or c.get('test_accessed') is not False:
 print(json.dumps({'ready':False,'reason':'global completion gate not satisfied'}));sys.exit(0)
# Inspect names only before reading any protocol contents, including markers.
allowed_protocol=set(('protocol_manifest.json','historical_users_verified.json','train_records.json','screen_records.json','confirm_records.json','test_records.json','ranker_train_positions.npy'))
for label in ('screen','confirm','train'):
 allowed_protocol.update(('candidate_'+label+'.json','candidate_'+label+'.items.npy','candidate_'+label+'.lengths.npy','candidate_'+label+'.json.building'))
for entry in protocol.rglob('*'):
 if entry.is_symlink() or (entry.is_file() and str(entry.relative_to(protocol)) not in allowed_protocol):raise ValueError('protocol contains unapproved or final-stage paths; refusing to open')
sys.path.insert(0,sys.argv[3])
import run_cross_multiseed as r
m,p=r._validate_dev_evidence(root)
if Path(m['protocol_manifest']).resolve()!=protocol/'protocol_manifest.json':raise ValueError('protocol path mismatch')
def inventory(directory):
 out={}
 for path in sorted(directory.rglob('*')):
  if path.is_symlink():raise ValueError('symlink in archive tree')
  if not path.is_file():continue
  rel=str(path.relative_to(directory))
  if directory==protocol and rel not in allowed_protocol:raise ValueError('protocol path appeared after gate; refusing to open')
  before=path.stat();h=r.sha256_file(path);after=path.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('file changed during inventory')
  out[rel]={'size':after.st_size,'sha256':h}
 return out
dev=inventory(root);pro=inventory(protocol)
allowed_markers={f'seed_{seed}/{variant}/COMPLETED.json' for seed in (42,43,44) for variant in ('raw','normalized_gated','zero_cross')}
if set(dev)!=set(m['evidence_hashes'])|{'dev_manifest.json','COMPLETED.json'}|allowed_markers:raise ValueError('unsealed dev extras/missing files')
if not set(pro).issubset(allowed_protocol):raise ValueError('protocol contains unexpected paths')
for label in ('screen','confirm','train'):
 receipt=m['candidate_cache'][label];path=Path(receipt['path']).resolve()
 if path!=protocol/('candidate_'+label+'.json'):raise ValueError('candidate path mismatch')
print(json.dumps({'ready':True,'dev':dev,'protocol':pro,'audit':{'method':'frozen _validate_dev_evidence absolute path; no evaluate/compare/targets','protocol_hash':p['manifest_hash'],'dev_manifest_hash':m['manifest_hash'],'code_hashes':r.code_hashes(),'history_scope_sha256':p.get('history_scope_sha256'),'historical_closure_sha256':p.get('historical_closure_sha256'),'test_accessed':False}},sort_keys=True))
'''

class Remote(common.SSHRemote):
    def __init__(self,args):super().__init__(args);self.code=args.remote_code
    def inventory(self):
        command=' '.join(shlex.quote(x) for x in (self.python,'-c',REMOTE,self.root,self.protocol,self.code))
        proc=subprocess.run(self.ssh+[command],check=True,capture_output=True,text=True)
        return json.loads(proc.stdout)


def tree_files(root):
    files=set()
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('symlink archive content')
        if p.is_file():files.add(str(p.relative_to(root)))
    return files


def verify_files(root,inventory):
    if tree_files(root)!=set(inventory):raise ValueError('archive tree differs from inventory')
    for rel,meta in inventory.items():
        common.safe_relative(rel);p=root/rel
        if p.stat().st_size!=meta['size'] or common.digest(p)!=meta['sha256']:raise ValueError('archive hash/size mismatch: '+rel)
        if p.suffix=='.json':common.checked_json(p)
        elif p.suffix=='.npz':common.verify_payload(p,meta)
        elif p.suffix=='.npy':np.load(p,mmap_mode='r',allow_pickle=False)


def verify_bundle(dev,protocol,inventory):
    if not inventory.get('ready') or inventory.get('audit',{}).get('test_accessed') is not False:raise ValueError('missing remote completion audit')
    verify_files(dev,inventory['dev']);verify_files(protocol,inventory['protocol'])
    m=common.checked_json(dev/'dev_manifest.json');p=common.checked_json(protocol/'protocol_manifest.json');done=common.checked_json(dev/'COMPLETED.json')
    for obj in (m,p):
        if obj.get('manifest_hash')!=common.json_hash({k:v for k,v in obj.items() if k!='manifest_hash'}):raise ValueError('manifest hash mismatch')
    if done.get('status')!='complete' or done.get('groups')!=9 or done.get('test_accessed') is not False or m.get('test_accessed') is not False:raise ValueError('global gate mismatch')
    if inventory['audit'].get('code_hashes')!=m.get('code_hashes') or m.get('code_hashes')!=p.get('code_hashes'):raise ValueError('remote audited code identity mismatch')
    for key in ('history_scope','history_scope_sha256','historical_closure_sha256'):
        if m.get(key)!=p.get(key):raise ValueError('history scope binding mismatch')
        if key!='history_scope' and inventory['audit'].get(key)!=p.get(key):raise ValueError('remote audited scope mismatch')
    if m.get('protocol_manifest_hash')!=p['manifest_hash'] or inventory['audit']['protocol_hash']!=p['manifest_hash'] or inventory['audit']['dev_manifest_hash']!=m['manifest_hash']:raise ValueError('protocol audit mismatch')
    allowed_markers={f'seed_{seed}/{variant}/COMPLETED.json' for seed in common.SEEDS for variant in common.VARIANTS}
    if set(inventory['dev'])!=set(m['evidence_hashes'])|{'dev_manifest.json','COMPLETED.json'}|allowed_markers:raise ValueError('unsealed evidence extras')
    for rel,h in m['evidence_hashes'].items():
        if inventory['dev'][rel]['sha256']!=h:raise ValueError('manifest evidence mismatch')
    allowed_protocol=set(common.PROTOCOL_FILES)
    for label in ('screen','confirm','train'):
        allowed_protocol.update((f'candidate_{label}.json',f'candidate_{label}.items.npy',f'candidate_{label}.lengths.npy',f'candidate_{label}.json.building'))
    if not set(inventory['protocol']).issubset(allowed_protocol):raise ValueError('unexpected protocol files')
    records={name:common.checked_json(protocol/p['records'][name]) for name in ('screen','confirm')}
    for label in ('screen','confirm','train'):
        receipt=m['candidate_cache'][label];side=protocol/f'candidate_{label}.json'
        if common.digest(side)!=receipt['sha256']:raise ValueError('candidate receipt mismatch')
        pools,meta=load_cache(side)
        if meta['files']!=receipt['files'] or meta['records_hash']!=receipt['records_hash'] or meta['pool_hash']!=receipt['pool_hash']:raise ValueError('candidate identity mismatch')
        if label!='train':
            if len(pools)!=len(records[label]) or meta['records_hash']!=p['cohort_hashes'][label]:raise ValueError('candidate cohort mismatch')
        elif len(pools)!=p['ranker_train_rows']:raise ValueError('train cache row mismatch')
    if m.get('seeds')!=list(common.SEEDS) or m.get('variants')!=list(common.VARIANTS):raise ValueError('nine-group identity mismatch')
    for seed in common.SEEDS:
        audit=common.checked_json(dev/f'seed_{seed}/initial_state_audit.json')
        for key in audit['copied_tensor_keys']:
            if key!='cross_gate_logit' and len({audit['state_sha256'][v][key] for v in common.VARIANTS})!=1:raise ValueError('initial state mismatch')
        for variant in common.VARIANTS:
            relative=f'seed_{seed}/{variant}';group=dev/relative
            gm=common.checked_json(group/'manifest.json')
            if any(gm.get(k)!=p.get(k) for k in ('history_scope','history_scope_sha256','historical_closure_sha256')):raise ValueError('group scope mismatch')
            common.verify_group(group,{name:inventory['dev'][relative+'/'+name] for name in common.GROUP_FILES})
            if m['checkpoint_hashes'][f'{seed}/{variant}']!=common.digest(group/'last.pth'):raise ValueError('checkpoint global seal mismatch')
            for label in ('screen','confirm'):
                names=[f'screen_epoch{x}' for x in (1,2,3)] if label=='screen' else ['confirm']
                for name in names:
                    with np.load(group/(name+'_users.npz'),allow_pickle=False) as arrays:
                        if not np.array_equal(arrays['uid'],[r[0] for r in records[label]]) or not np.array_equal(arrays['position'],[r[1] for r in records[label]]):raise ValueError('paired UID/position identity mismatch')


def copy_tree(remote,area,stage,inventory,incremental,linked):
    stage.mkdir(parents=True)
    for rel,meta in inventory.items():
        common.safe_relative(rel);target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
        source=incremental/rel if incremental and area=='dev' and rel.endswith('/last.pth') else None
        if source is not None and source.exists():
            common.reject_symlink_chain(incremental,source)
            archived=common.checked_json(source.parent/'ARCHIVE_RECEIPT.json')
            if archived.get('status')!='verified' or archived['inventory'].get('last.pth')!=meta:raise ValueError('incremental archive receipt drift')
            common.verify_payload(source,meta)
            os.link(source,target);linked.append(rel)
        else:remote.fetch(rel,target,protocol=area=='protocol')
        if target.stat().st_size!=meta['size'] or common.digest(target)!=meta['sha256']:raise ValueError('download hash/size mismatch')
    common.fsync_directory(stage)


def run(remote,local_dev,local_protocol,receipt_dir,incremental=None,dry_run=False):
    paths=[Path(x).absolute() for x in (local_dev,local_protocol,receipt_dir)]
    for path in paths:common.reject_symlink_chain(Path(path.anchor),path)
    dev,protocol,receipts=paths
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(paths) for b in paths[i+1:]):raise ValueError('archive and receipts paths must be disjoint')
    inc=Path(incremental).absolute() if incremental else None
    if inc:common.reject_symlink_chain(Path(inc.anchor),inc)
    if inc and any(inc==x or inc.is_relative_to(x) or x.is_relative_to(inc) for x in paths):raise ValueError('incremental archive must be disjoint')
    # Not-ready inventory performs only the tiny completion gate, no tree hashing.
    inventory=remote.inventory()
    if not inventory.get('ready'):return {'status':'not_ready','reason':inventory.get('reason')}
    if dry_run:return {'status':'ready','inventory':inventory,'remote_writes':False}
    receipts.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+uuid.uuid4().hex[:8]
    lock=receipts/'.complete_archive.lock';fd=None
    common.reject_symlink_chain(receipts,lock)
    receipt={'status':'started','run_id':run_id,'inventory':inventory,'hardlinks':[],'remote_writes':False}
    partial_dev=dev.with_name(dev.name+'.partial-'+run_id);partial_protocol=protocol.with_name(protocol.name+'.partial-'+run_id)
    seal=receipts/'COMPLETE.json'
    common.reject_symlink_chain(receipts,seal)
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        if dev.exists() or protocol.exists() or seal.exists():
            if not(dev.is_dir() and protocol.is_dir() and seal.is_file()):raise FileExistsError('partial publication exists; inspect without overwriting')
            saved=common.checked_json(seal)
            if saved.get('inventory')!=inventory:raise ValueError('sealed complete archive drift')
            verify_bundle(dev,protocol,inventory);receipt['status']='verified_existing';return receipt
        copy_tree(remote,'dev',partial_dev,inventory['dev'],inc,receipt['hardlinks'])
        copy_tree(remote,'protocol',partial_protocol,inventory['protocol'],None,receipt['hardlinks'])
        verify_bundle(partial_dev,partial_protocol,inventory)
        if remote.inventory()!=inventory:raise ValueError('remote immutable archive drift during copy')
        for stage,destination in ((partial_protocol,protocol),(partial_dev,dev)):
            common.reject_symlink_chain(Path(destination.anchor),destination)
            if destination.exists():raise FileExistsError(destination)
            os.rename(stage,destination);common.fsync_directory(destination.parent)
        receipt['status']='complete';common.write_json_new(seal,receipt);common.fsync_directory(receipts)
        return receipt
    except BaseException as exc:
        receipt['status']='failed';receipt['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        common.write_json_new(receipts/(run_id+'.json'),receipt)
        if fd is not None:os.close(fd);lock.unlink()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=22);p.add_argument('--control-path')
    p.add_argument('--remote-python',default='/root/miniconda3/bin/python3');p.add_argument('--remote-code',required=True)
    p.add_argument('--remote-dev',required=True);p.add_argument('--remote-protocol',required=True)
    p.add_argument('--local-dir',required=True);p.add_argument('--local-protocol',required=True);p.add_argument('--receipt-dir',required=True)
    p.add_argument('--incremental-dir');p.add_argument('--dry-run',action='store_true');return p

if __name__=='__main__':
    args=parser().parse_args();torch.set_num_threads(4)
    print(json.dumps(run(Remote(args),args.local_dir,args.local_protocol,args.receipt_dir,args.incremental_dir,args.dry_run),indent=2))
