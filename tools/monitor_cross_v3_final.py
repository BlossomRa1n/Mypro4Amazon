#!/usr/bin/env python3
"""Fixed one-stage v3 launcher/900-second monitor. Default: read-only check.

Root must approve archival, migration and resource receipts before --execute.
No SSH, automatic stage chaining, retry, cleanup, archive or shutdown.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode = True

ROOT = Path('/root/next_cross_20260924')
PYTHON = '/root/miniconda3/bin/python3'
CODE = ROOT/'src_v3/code'
BASE = ROOT/'base'
PLAN = ROOT/'final_plan_v3.json'
PROTOCOL = ROOT/'protocol_v3'
FINAL = ROOT/'final_v3'
DEV = Path('/root/autodl-tmp/next_cross_20260924_dev')
INTERVAL = 900
PROTOCOL_HASH = 'dcf65b10c266ba02b00d0493815a14b8af9aa4f47551d7c562a2010104409667'
FROZEN = {'run_cross_multiseed.py':'ec8df470eab5bc3bb39978371c9d086b9a1e834b2be63dc72648616ffa9d1e22', 'cross_pool_cache.py':'e328ad066104df75b496450cf169e6ea47846c7c8d8194e38959b7652c914505'}
CONFIG = {'dim':256,'token_dim':256,'hist_len':50,'batch_size':256,'candidates':75}
FULL_ROWS = 4731777
TEST_USERS = 17296
SCOPE_KEYS = ('history_scope','history_scope_sha256','historical_closure_sha256')


def utc(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def object_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def safe_relative(value):
    if not isinstance(value,str): raise ValueError('unsafe evidence relative path')
    rel=PurePosixPath(value)
    if not isinstance(value,str) or not value or rel.is_absolute() or '..' in rel.parts or '.' in value.split('/') or str(rel)!=value: raise ValueError('unsafe evidence relative path')
    return rel
def checked_path(path):
    path=Path(path).absolute(); current=Path(path.anchor)
    for part in path.parts[1:]:
        current/=part
        if current.is_symlink(): raise ValueError('symlink path forbidden: '+str(current))
    return path
def read_json(path,key=None):
    value=json.loads(checked_path(path).read_text())
    if key and value.get(key)!=object_hash({k:v for k,v in value.items() if k!=key}): raise ValueError('invalid '+key)
    return value
def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
def write_new(path,value):
    with checked_path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True); stream.flush(); os.fsync(stream.fileno())
    sync_dir(Path(path).parent)
def monitor_dir(stage):
    if stage not in ('final-train','final-test'): raise ValueError('unapproved stage')
    return ROOT/('monitor_'+stage.replace('-','_')+'_v3')
def command(stage):
    monitor_dir(stage)
    cmd=[PYTHON,'-u',str(CODE/'run_cross_multiseed.py'),stage,'--base-run',str(BASE),'--final-plan',str(PLAN),'--run-dir',str(FINAL),'--dim','256','--token-dim','256','--hist-len','50','--candidates','75']
    if stage=='final-train': cmd += ['--batch-size','256','--epochs','3']
    return cmd

def no_test_started():
    for tree in (PROTOCOL,FINAL):
        if tree.exists() and any(p.name.startswith('TEST_') for p in tree.rglob('*')):
            raise ValueError('test marker already exists; no second evaluation')

def no_test_outputs():
    if PROTOCOL.exists() and any(p.name.startswith('candidate_test') for p in PROTOCOL.iterdir()): raise ValueError('test candidate output already exists')
    if FINAL.exists() and any(p.name=='final_test_results.json' or p.name.endswith(('_test_users.npz','_test_metrics.json')) for p in FINAL.rglob('*')): raise ValueError('test output already exists')

def verify_train(plan,protocol):
    manifest=read_json(FINAL/'final_manifest.json','manifest_hash')
    marker=read_json(FINAL/'FINAL_TRAIN_COMPLETED.json')
    expected={'final_plan_hash':plan['plan_hash'],'protocol_manifest_hash':protocol['manifest_hash'],'models':plan['models'],'all_prefix_rows':FULL_ROWS,'test_history_coverage_users':TEST_USERS,'test_users':TEST_USERS,'test_mode':plan['test_mode'],'test_history_records_hash':plan['test_cohort_hash'],'test_future_labels_read':False}
    expected.update({k:protocol[k] for k in SCOPE_KEYS})
    if marker.get('manifest_hash')!=manifest['manifest_hash'] or any(manifest.get(k)!=v for k,v in expected.items()): raise ValueError('final training identity/coverage mismatch')
    evidence=manifest.get('evidence_hashes')
    if not evidence: raise ValueError('unsealed final training evidence')
    for rel,h in evidence.items():
        path=checked_path(FINAL/safe_relative(rel))
        if not path.is_relative_to(FINAL) or digest(path)!=h: raise ValueError('final training evidence mismatch')
    if set(manifest.get('checkpoint_hashes',{}))!=set(plan['models']): raise ValueError('top checkpoint set mismatch')
    for model in plan['models']:
        if digest(checked_path(FINAL/model/'last.pth'))!=manifest['checkpoint_hashes'][model]: raise ValueError('top checkpoint SHA mismatch')
    return manifest

def verify_completion(stage,plan,protocol):
    trained=verify_train(plan,protocol)
    if stage=='final-train':
        no_test_started()
        return {'stage':stage,'final_manifest_hash':trained['manifest_hash'],'plan_hash':plan['plan_hash'],'protocol_hash':protocol['manifest_hash'],'all_prefix_rows':trained['all_prefix_rows'],'test_history_coverage_users':trained['test_history_coverage_users'],'checkpoint_hashes':trained['checkpoint_hashes']}
    result=read_json(FINAL/'final_test_results.json','results_hash'); marker=read_json(PROTOCOL/'TEST_STARTED.json')
    expected={'plan_hash':plan['plan_hash'],'test_cohort_hash':plan['test_cohort_hash'],'test_mode':plan['test_mode'],'test_users':TEST_USERS}
    expected.update({k:protocol[k] for k in SCOPE_KEYS})
    if any(result.get(k)!=v or marker.get(k)!=v for k,v in expected.items()): raise ValueError('final test identity mismatch')
    if marker.get('status')!='complete' or marker.get('results_hash')!=result['results_hash'] or marker.get('selection_hash')!=plan['selection_hash'] or marker.get('checkpoint_hashes')!=trained['checkpoint_hashes']: raise ValueError('final test marker/result mismatch')
    if result.get('status') not in ('accepted','raw_retained') or set(result.get('models',{}))!=set(plan['models']): raise ValueError('incomplete final test model results')
    if result.get('paired',{}).get('users')!=TEST_USERS: raise ValueError('final test user count mismatch')
    outputs={}
    for model in plan['models']:
        for suffix in ('_test_users.npz','_test_metrics.json'):
            path=checked_path(FINAL/(model+suffix))
            if not path.is_file() or path.stat().st_size==0: raise ValueError('missing test output')
            outputs[path.name]=digest(path)
    return {'stage':stage,'results_hash':result['results_hash'],'final_manifest_hash':trained['manifest_hash'],'plan_hash':plan['plan_hash'],'protocol_hash':protocol['manifest_hash'],'output_sha256':outputs}

def preflight(stage):
    output=checked_path(monitor_dir(stage))
    if output.exists(): raise FileExistsError('persistent monitor lock exists; no retry')
    for path in (CODE,BASE,PLAN,PROTOCOL,FINAL,DEV): checked_path(path)
    for name,h in FROZEN.items():
        if digest(CODE/name)!=h: raise ValueError('frozen code SHA mismatch: '+name)
    no_test_started()
    no_test_outputs()
    raw=read_json(PLAN,'plan_hash')
    if raw.get('base_run')!=str(BASE) or raw.get('protocol_manifest')!=str(PROTOCOL/'protocol_manifest.json') or raw.get('selection')!=str(ROOT/'selection_v3.json'): raise ValueError('fixed plan path mismatch')
    selection=read_json(ROOT/'selection_v3.json','selection_hash')
    if selection.get('dev_run')!=str(DEV) or selection.get('status')!='accepted': raise ValueError('completed dev accepted selection required')
    sys.path.insert(0,str(CODE)); runner=importlib.import_module('run_cross_multiseed')
    if Path(runner.__file__).resolve()!=CODE/'run_cross_multiseed.py': raise ValueError('wrong imported frozen runner')
    # Revalidates full dev and pure compare of saved arrays; no new model scoring.
    plan=runner._load_plan(PLAN); protocol=runner.load_protocol(PROTOCOL/'protocol_manifest.json')
    if plan.get('final_evaluation_allowed') is not True or plan.get('winner') not in ('normalized_gated','zero_cross') or plan.get('models')!=['raw',plan['winner']]: raise ValueError('accepted locked plan required')
    if plan.get('config')!=CONFIG or plan.get('epochs')!=3 or plan.get('final_seed')!=42 or plan.get('final_init_seed')!=424242: raise ValueError('fixed training configuration mismatch')
    if protocol.get('manifest_hash')!=PROTOCOL_HASH or plan.get('protocol_manifest_hash')!=PROTOCOL_HASH or protocol.get('counts',{}).get('test')!=TEST_USERS or protocol.get('ranker_train_rows')!=604511: raise ValueError('fixed protocol identity mismatch')
    if stage=='final-train':
        if FINAL.exists() and any(FINAL.iterdir()): raise FileExistsError('final directory must be empty')
        if any(p.name.startswith(('candidate_final','candidate_test')) for p in PROTOCOL.iterdir()): raise ValueError('final cache already exists')
    else: verify_train(plan,protocol)
    return plan,protocol

def capture(argv):
    try: return {'ok':True,'output':subprocess.check_output(argv,stderr=subprocess.STDOUT,text=True,timeout=30).strip()}
    except Exception as exc: return {'ok':False,'error':str(exc)}
def file_sizes(tree,pattern='*'):
    if not tree.exists(): return []
    values=[]
    for path in sorted(tree.rglob(pattern)):
        if path.is_symlink(): values.append({'path':str(path.relative_to(tree)),'symlink':True})
        elif path.is_file(): values.append({'path':str(path.relative_to(tree)),'bytes':path.stat().st_size})
    return values

def snapshot(child,output,phase):
    value={'utc':utc(),'phase':phase,'runner_pid':child.pid,'runner_exit_code':child.poll(),'monitor_pid':os.getpid(),'interval_seconds':INTERVAL,'final_files':file_sizes(FINAL),'candidate_files':file_sizes(PROTOCOL,'candidate*'),'stdout_bytes':(output/'stdout.log').stat().st_size}
    for label,cmd in [('process',['ps','-p',str(child.pid),'-o','pid,ppid,etime,%cpu,rss,vsz,args']),('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]: value[label]=capture(cmd)
    value['memory']={}; value['cgroup']={}
    for path in (Path('/proc/meminfo'),Path('/sys/fs/cgroup/memory.current'),Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory.events')):
        try: value['memory' if path.name=='meminfo' else 'cgroup'][path.name]=path.read_text()
        except OSError as exc: value['cgroup'][path.name]={'error':str(exc)}
    value['disk_free_bytes']={str(p):shutil.disk_usage(p).free for p in (ROOT,DEV.parent)}
    value['collection_errors']=[label for label in ('process','gpu','gpu_processes') if not value[label]['ok'] and not (label=='process' and child.poll() is not None)]
    value['collection_errors'] += [name for name,entry in value['cgroup'].items() if isinstance(entry,dict) and 'error' in entry]
    write_new(output/('snapshot_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json'),value)
    return value

def run(stage,execute=False,*,gate=preflight,popen=subprocess.Popen,timer=time):
    plan,protocol=gate(stage); output=checked_path(monitor_dir(stage)); cmd=command(stage)
    code_names=set(FROZEN)|set(plan.get('code_hashes',{}))
    startup_hashes={str(p):digest(checked_path(p)) for p in [CODE/safe_relative(name) for name in sorted(code_names)]+[PLAN,ROOT/'selection_v3.json',PROTOCOL/'protocol_manifest.json']}
    summary={'stage':stage,'command':cmd,'plan_hash':plan['plan_hash'],'protocol_hash':protocol['manifest_hash'],'frozen_code_sha256':FROZEN,'startup_file_sha256':startup_hashes,'interval_seconds':INTERVAL,'execute':execute,'root_external_gate_required':True}
    if not execute: return dict(summary,status='check_passed')
    # Exclusive persistent directory is the one-shot execution lock.
    output.mkdir(exist_ok=False); sync_dir(output.parent)
    child=None
    try:
        write_new(output/'STARTED.json',dict(summary,utc=utc(),monitor_pid=os.getpid()))
        env=os.environ.copy(); env['PYTHONPATH']=str(CODE); env['PYTHONDONTWRITEBYTECODE']='1'
        with (output/'stdout.log').open('xb',buffering=0) as log:
            child=popen(cmd,cwd=str(CODE.parent),env=env,stdout=log,stderr=subprocess.STDOUT)
            write_new(output/'launch.json',dict(summary,utc=utc(),runner_pid=child.pid))
            errors=[]
            def snap(phase):
                try:
                    captured=snapshot(child,output,phase)
                    if captured and captured.get('collection_errors'): errors.append({'phase':phase,'errors':captured['collection_errors'],'utc':utc()})
                except Exception as exc: errors.append({'phase':phase,'error':str(exc),'utc':utc()})
            snap('start'); deadline=timer.monotonic()+INTERVAL
            while child.poll() is None:
                timer.sleep(min(5,max(0.1,deadline-timer.monotonic())))
                if timer.monotonic()>=deadline:
                    snap('interval'); deadline=timer.monotonic()+INTERVAL
            code=child.wait(); log.flush(); os.fsync(log.fileno()); snap('end')
        receipt={'utc':utc(),'stage':stage,'exit_code':code,'status':'failed_or_incomplete','automatic_restart':False,'snapshot_errors':errors}
        if code==0:
            try:
                if any(digest(checked_path(p))!=h for p,h in startup_hashes.items()): raise ValueError('startup code/plan/protocol changed')
                receipt['completion']=verify_completion(stage,plan,protocol); receipt['status']='complete' if not errors else 'monitoring_failed'
            except Exception as exc: receipt['completion_error']=str(exc)
        write_new(output/'exit.json',receipt); return receipt
    except BaseException as exc:
        write_new(output/'monitor_error.json',{'utc':utc(),'error':f'{type(exc).__name__}: {exc}','runner_pid':child.pid if child else None,'runner_exit_code':child.poll() if child else None,'automatic_restart':False,'manual_inspection_required':True})
        raise

def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',choices=('final-train','final-test'),required=True); p.add_argument('--execute',action='store_true'); return p
if __name__=='__main__':
    args=parser().parse_args(); result=run(args.stage,args.execute); print(json.dumps(result,indent=2)); sys.exit(0 if result['status'] in ('complete','check_passed') else 1)
