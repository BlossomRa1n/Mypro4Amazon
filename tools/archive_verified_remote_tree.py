"""Copy an immutable remote evidence tree and verify every byte locally.

Read-only remotely; no deletion, model scoring, automatic retry, or shutdown.
The caller must select a completed immutable tree, including its source/logs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);marker=Path(sys.argv[2])
if not root.is_absolute() or root.is_symlink() or not root.is_dir():raise ValueError('invalid remote root')
if marker.is_absolute() or '..' in marker.parts:raise ValueError('unsafe completion path')
complete=root/marker
if not complete.is_file() or complete.is_symlink():raise ValueError('completion marker absent')
obj=json.loads(complete.read_text())
if obj.get('status') not in ('complete','verified'):raise ValueError('completion is not successful')
result={}
for path in sorted(root.rglob('*')):
 if path.is_symlink():raise ValueError('symlink in evidence: '+str(path))
 if path.is_dir():continue
 if not path.is_file():raise ValueError('nonregular evidence: '+str(path))
 before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('evidence changed while hashing')
 result[str(path.relative_to(root))]={'sha256':h.hexdigest(),'bytes':after.st_size}
print(json.dumps({'root':str(root),'marker':str(marker),'files':result},sort_keys=True))
'''


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def verify_local(root,files):
    actual={}
    for p in root.rglob('*'):
        if p.is_symlink() or not(p.is_file() or p.is_dir()):raise ValueError('unsafe local entry')
        if p.is_file():actual[str(p.relative_to(root))]=p
    if set(actual)!=set(files):raise ValueError('missing or extra archive files')
    for rel,meta in files.items():
        p=actual[rel]
        if p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:raise ValueError('archive byte mismatch: '+rel)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--control-path',required=True);p.add_argument('--remote-root',required=True)
    p.add_argument('--completion',default='COMPLETED.json');p.add_argument('--remote-python',default='/root/miniconda3/bin/python')
    p.add_argument('--output',required=True);p.add_argument('--receipt',required=True)
    args=p.parse_args()
    output=Path(args.output).absolute();receipt=Path(args.receipt).absolute()
    for path in (output,receipt):
        if '..' in path.parts or any(x.is_symlink() for x in [path,*path.parents]):raise ValueError('unsafe local destination')
        if path.exists():raise FileExistsError(path)
    if receipt.is_relative_to(output):raise ValueError('receipt must be outside archived tree')
    output.parent.mkdir(parents=True,exist_ok=True);receipt.parent.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-S',args.control_path,'-o','BatchMode=yes','-p',str(args.port)]
    command=shlex.join([args.remote_python,'-c',REMOTE,args.remote_root,args.completion])
    def inventory():
        result=subprocess.run([*ssh,args.host,command],check=True,text=True,capture_output=True)
        return json.loads(result.stdout)
    first=inventory()
    inventory_path=receipt.with_suffix('.inventory.json')
    write(inventory_path,first)
    log=receipt.with_suffix('.transfer.log')
    output.mkdir()
    try:
        source=args.host+':'+shlex.quote(args.remote_root.rstrip('/')+'/')
        with log.open('xb') as stream:
            subprocess.run(['rsync','-a','--partial','-e',shlex.join(ssh),source,str(output)+'/'],check=True,stdout=stream,stderr=subprocess.STDOUT)
        verify_local(output,first['files'])
        second=inventory()
        if first!=second:raise ValueError('remote immutable evidence changed during transfer')
        completion_sha=first['files'][args.completion]['sha256']
        result={'status':'verified','verified_at_utc':datetime.now(timezone.utc).isoformat(),'remote_root':args.remote_root,'local_root':str(output),'inventory_sha256':sha(inventory_path),'completion_sha256':completion_sha,'files':len(first['files']),'bytes':sum(x['bytes'] for x in first['files'].values()),'remote_inventory_before_after_equal':True,'local_exact_coverage':True,'local_sha_size_equal':True,'local_payload_semantics_verified':False,'remote_mutated':False}
        if args.completion=='COMPLETED.json':
            complete=json.loads((output/args.completion).read_text())
            if complete.get('protocol')=='early-stop-zero-cross-20260925-v1':result['cross_completion_sha256']=completion_sha
        if args.completion=='TRAINING_COMPLETED.json':result['training_completion_sha256']=completion_sha
        if args.completion=='TEST_COMPLETED.json':result['test_completion_sha256']=completion_sha
        write(receipt,result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as error:
        write(receipt.with_suffix('.failed.json'),{'status':'failed','error_type':type(error).__name__,'error':str(error),'partial_preserved':str(output)})
        raise


if __name__=='__main__':main()
