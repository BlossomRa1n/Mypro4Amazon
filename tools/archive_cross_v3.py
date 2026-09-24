#!/usr/bin/env python3
"""One-shot read-only SSH archival of completed cross-v3 training groups.

No remote writes, cleanup, scheduling, training, or final-test access. Group
COMPLETED seals training artifacts only; confirm is appended later and is
intentionally left to the final whole-run archive.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import uuid
import numpy as np
import torch

SEEDS=(42,43,44)
VARIANTS=('raw','normalized_gated','zero_cross')
PROTOCOL_FILES=('protocol_manifest.json','historical_users_verified.json','train_records.json',
                'screen_records.json','confirm_records.json','test_records.json','ranker_train_positions.npy')
GROUP_FILES=('last.pth','history.json','manifest.json','COMPLETED.json',
             *(f'epoch_{i}_trace.json' for i in (1,2,3)),
             *(f'screen_epoch{i}_users.npz' for i in (1,2,3)))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def checked_json(path):
    with Path(path).open(encoding='utf-8') as stream:return json.load(stream)


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def write_json_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.flush();os.fsync(stream.fileno())


def safe_relative(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p)!=name:
        raise ValueError('unsafe relative inventory path')
    return p


def reject_symlink_chain(root,target):
    root=Path(root);target=Path(target)
    if not target.is_relative_to(root):raise ValueError('local archive path escapes root')
    current=root
    if current.is_symlink():raise ValueError('symlink archive root forbidden')
    for part in target.relative_to(root).parts:
        current=current/part
        if current.is_symlink():raise ValueError('symlink archive ancestor forbidden: '+str(current))


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


# The fixed script reads only the expected artifacts; shell interpolation uses
# shlex.quote and remote paths are positional data, not executable expressions.
REMOTE_INVENTORY = r'''
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();protocol=Path(sys.argv[2]).resolve() if sys.argv[2] else None
files=json.loads(sys.argv[3]);protocol_files=json.loads(sys.argv[4]);only_group=sys.argv[5]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def jsonread(p):return json.loads(p.read_text())
def inventory(directory,names):
 result={}
 for name in names:
  p=directory/name
  if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(directory):raise ValueError('unsafe/missing artifact '+str(p))
  before=p.stat();h=sha(p);after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('artifact changed during inventory')
  result[name]={'size':after.st_size,'sha256':h}
 return result
result={'groups':{}}
for seed in (42,43,44):
 for variant in ('raw','normalized_gated','zero_cross'):
  rel='seed_'+str(seed)+'/'+variant;d=root/rel
  if only_group and rel!=only_group:continue
  if d.exists() and (d.is_symlink() or not d.resolve().is_relative_to(root)):raise ValueError('unsafe group directory')
  if not (d/'COMPLETED.json').exists():continue
  done=jsonread(d/'COMPLETED.json');m=jsonread(d/'manifest.json')
  if done.get('status')!='complete' or done.get('epoch')!=3 or m.get('epochs')!=3 or m.get('seed')!=seed or m.get('variant')!=variant:raise ValueError('invalid completed group '+rel)
  inv=inventory(d,files)
  if inv['last.pth']['sha256']!=done.get('checkpoint_sha256') or inv['last.pth']['sha256']!=m.get('checkpoint_sha256'):raise ValueError('checkpoint seal mismatch '+rel)
  result['groups'][rel]=inv
if protocol:
 result['protocol']=inventory(protocol,protocol_files)
print(json.dumps(result,sort_keys=True))
'''


class SSHRemote:
    def __init__(self,args):
        if args.host.startswith('-'):raise ValueError('host cannot start with dash')
        self.ssh=['ssh','-o','BatchMode=yes','-p',str(args.port)]
        if args.control_path:self.ssh+=['-o','ControlPath='+args.control_path]
        self.ssh+=[args.host]
        self.root=args.remote_dev.rstrip('/')
        self.protocol=args.remote_protocol
        self.python=args.remote_python
    def inventory(self,include_protocol=False,only_group=""):
        command=' '.join(shlex.quote(x) for x in (self.python,'-c',REMOTE_INVENTORY,self.root,self.protocol if include_protocol else '',json.dumps(GROUP_FILES),json.dumps(PROTOCOL_FILES),only_group))
        result=subprocess.run(self.ssh+[command],check=True,capture_output=True,text=True)
        return json.loads(result.stdout)
    def recheck(self,relative,protocol=False):
        value=self.inventory(include_protocol=protocol,only_group="protocol" if protocol else relative)
        return value.get("protocol") if protocol else value["groups"].get(relative)
    def fetch(self,relative,destination,protocol=False):
        safe_relative(relative)
        root=self.protocol if protocol else self.root
        command='cat -- '+shlex.quote(str(PurePosixPath(root)/relative))
        with Path(destination).open('xb') as stream:
            subprocess.run(self.ssh+[command],stdout=stream,check=True)
            stream.flush();os.fsync(stream.fileno())


def verify_payload(path,metadata):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size!=metadata['size'] or digest(path)!=metadata['sha256']:
        raise ValueError('archive size/hash mismatch: '+str(path))
    if path.suffix=='.json':checked_json(path)
    elif path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as value:
            required=('uid','position','hit5','pool_hit','ndcg5')
            if any(k not in value for k in required):raise ValueError('missing NPZ fields')
            arrays={k:value[k] for k in value.files}
            n=len(arrays['uid'])
            if any(arrays[k].ndim!=1 or len(arrays[k])!=n or not np.isfinite(arrays[k]).all() for k in required):raise ValueError('invalid NPZ arrays')
            if len(np.unique(arrays['uid']))!=n:raise ValueError('duplicate NPZ UID')
            if any(not np.isin(arrays[k],[0,1]).all() for k in ('hit5','pool_hit')):raise ValueError('invalid NPZ hit values')
            if np.any(arrays['hit5']>arrays['pool_hit']) or np.any(arrays['ndcg5']<0) or np.any(arrays['ndcg5']>1):raise ValueError('invalid NPZ metrics')
    elif path.suffix=='.npy':
        value=np.load(path,mmap_mode='r',allow_pickle=False)
        if value.ndim!=1 or value.dtype.kind not in 'iu':raise ValueError('invalid position array')
    elif path.suffix=='.pth':
        # Restricted unpickler avoids executing arbitrary checkpoint classes.
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(saved,dict) or not all(k in saved for k in ('model','manifest','history','epoch')):raise ValueError('invalid checkpoint fields')
        if saved['epoch']!=3 or not isinstance(saved['manifest'],dict) or not isinstance(saved['history'],list) or not isinstance(saved['model'],dict) or not saved['model']:raise ValueError('invalid checkpoint metadata')
        if any(not torch.is_tensor(t) or not torch.isfinite(t).all() for t in saved['model'].values()):raise ValueError('nonfinite/invalid checkpoint tensors')
        del saved


def verify_group(directory,inventory):
    if set(inventory)!=set(GROUP_FILES):raise ValueError('unexpected completed group inventory')
    for name,meta in inventory.items():verify_payload(directory/name,meta)
    done=checked_json(directory/'COMPLETED.json');m=checked_json(directory/'manifest.json')
    if done.get('status')!='complete' or done.get('epoch')!=3 or m.get('epochs')!=3 or inventory['last.pth']['sha256']!=done.get('checkpoint_sha256') or inventory['last.pth']['sha256']!=m.get('checkpoint_sha256'):raise ValueError('local group seal mismatch')
    saved=torch.load(directory/'last.pth',map_location='cpu',weights_only=True)
    checkpoint_manifest=dict(m);checkpoint_manifest.pop('checkpoint_sha256',None)
    if saved['manifest']!=checkpoint_manifest or saved['history']!=checked_json(directory/'history.json'):raise ValueError('checkpoint manifest/history mismatch')
    for epoch in (1,2,3):
        if checked_json(directory/f'epoch_{epoch}_trace.json')!=m.get('trace_files',{}).get(str(epoch)):raise ValueError('trace manifest mismatch')


def verify_protocol(directory,inventory):
    if set(inventory)!=set(PROTOCOL_FILES):raise ValueError('unexpected protocol inventory')
    for name,meta in inventory.items():verify_payload(directory/name,meta)
    m=checked_json(directory/'protocol_manifest.json')
    if m.get('manifest_hash')!=json_hash({k:v for k,v in m.items() if k!='manifest_hash'}):raise ValueError('protocol hash mismatch')
    if m.get('protocol')!='next-cross-multiseed-20260924-v3-month-scope':raise ValueError('not frozen v3 protocol')


def archive_one(remote,local_root,relative,inventory,run_id,protocol=False):
    safe_relative(relative)
    destination=local_root/relative
    reject_symlink_chain(local_root,destination)
    expected_files=set(inventory)|{'ARCHIVE_RECEIPT.json'}
    if destination.exists():
        if destination.is_symlink() or {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}!=expected_files:raise ValueError('existing local archive is unsealed or changed')
        old=checked_json(destination/'ARCHIVE_RECEIPT.json')
        if old.get('inventory')!=inventory:raise ValueError('remote/local archive drift; refusing overwrite')
        (verify_protocol if protocol else verify_group)(destination,inventory)
        return {'relative':relative,'status':'verified_existing'}
    partial=local_root/'.partial'/run_id/relative
    reject_symlink_chain(local_root,partial)
    partial.mkdir(parents=True,exist_ok=False)
    for name,meta in inventory.items():
        safe_relative(name)
        remote_relative=name if protocol else str(PurePosixPath(relative)/name)
        remote.fetch(remote_relative,partial/name,protocol=protocol)
        verify_payload(partial/name,meta)
    (verify_protocol if protocol else verify_group)(partial,inventory)
    # Re-inventory before local publication detects remote drift during transfer.
    if hasattr(remote,'recheck'):
        now=remote.recheck(relative,protocol)
    else:
        after=remote.inventory(include_protocol=protocol)
        now=after['protocol'] if protocol else after['groups'].get(relative)
    if now!=inventory:raise ValueError('remote artifacts drifted during transfer')
    write_json_new(partial/'ARCHIVE_RECEIPT.json',{'status':'verified','inventory':inventory,'run_id':run_id,'remote_relative':relative,'confirm_deferred_to_global_archive':not protocol})
    fsync_directory(partial)
    reject_symlink_chain(local_root,destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError('archive appeared during publish')
    os.rename(partial,destination);fsync_directory(destination.parent)
    return {'relative':relative,'status':'archived','bytes':sum(x['size'] for x in inventory.values())}


def run_once(remote,local_root,include_protocol=False):
    supplied=Path(local_root).absolute()
    reject_symlink_chain(Path(supplied.anchor),supplied)
    local_root=supplied.resolve();local_root.mkdir(parents=True,exist_ok=True)
    reject_symlink_chain(local_root,local_root/"receipts")
    reject_symlink_chain(local_root,local_root/".partial")
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+uuid.uuid4().hex[:8]
    receipt={'run_id':run_id,'status':'started','groups':[],'remote_writes':False}
    lock=local_root/'.archive.lock';fd=None
    reject_symlink_chain(local_root,lock)
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        os.write(fd,run_id.encode());os.fsync(fd)
        inventory=remote.inventory(include_protocol=include_protocol)
        receipt['remote_inventory']=inventory
        allowed={f'seed_{s}/{v}' for s in SEEDS for v in VARIANTS}
        if not set(inventory.get('groups',{})).issubset(allowed):raise ValueError('unexpected remote group')
        for relative,entries in sorted(inventory.get('groups',{}).items()):
            receipt['groups'].append(archive_one(remote,local_root,relative,entries,run_id))
        if include_protocol:receipt['protocol']=archive_one(remote,local_root,'protocol',inventory['protocol'],run_id,protocol=True)
        receipt['status']='complete'
        return receipt
    except BaseException as exc:
        receipt['status']='failed';receipt['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipts=local_root/'receipts';reject_symlink_chain(local_root,receipts);receipts.mkdir(exist_ok=True)
        write_json_new(receipts/(run_id+'.json'),receipt)
        if fd is not None:os.close(fd);lock.unlink()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=22);p.add_argument('--control-path')
    p.add_argument('--remote-python',default='/root/miniconda3/bin/python3');p.add_argument('--remote-dev',required=True)
    p.add_argument('--remote-protocol');p.add_argument('--include-protocol',action='store_true')
    p.add_argument('--local-dir',required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.include_protocol and not args.remote_protocol:raise ValueError('--include-protocol requires --remote-protocol')
    torch.set_num_threads(4)
    result=run_once(SSHRemote(args),Path(args.local_dir),args.include_protocol)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':sys.exit(main())
