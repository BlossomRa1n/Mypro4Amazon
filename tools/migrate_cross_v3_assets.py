#!/usr/bin/env python3
"""One-shot approved immutable asset relocation. Default is strictly check-only.

Run locally on the authorized server after complete local archival and an accepted
locked plan. No SSH, training, new model evaluation, cleanup or retry is provided.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from datetime import datetime,timezone
sys.dont_write_bytecode=True
import torch

PROTOCOL_HASH='dcf65b10c266ba02b00d0493815a14b8af9aa4f47551d7c562a2010104409667'
ASSETS=(
 {'source':'/root/corrected_baseline_20260912/din_best.pth','target':'/root/autodl-tmp/next_cross_20260924_immutable/corrected_din_best.pth','bytes':1260104760,'sha256':'69724b407c42af8c1b3802f6821228c8178b4623a5c31991090b5f0131e35423'},
 {'source':'/root/next_cross_20260924/history_month_v3_20260924.tar','target':'/root/autodl-tmp/next_cross_20260924_immutable/history_month_v3_20260924.tar','bytes':248166400,'sha256':'fd7f01891d6d20a611b9b62e244a2ac45af4e80acbec04c83e9b808dbdb609a5'},
)
TENSOR_BYTES=1260683576
FULL_ROWS=4731777
TEST_USERS=17296
RESERVE=1024**3
ALLOWANCE=192*1024**2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_identity(path):
    reject_symlinks(path)
    stat=Path(path).stat()
    if not Path(path).is_file() or stat.st_nlink!=1:raise ValueError('source must be unique regular file')
    return {'device':stat.st_dev,'inode':stat.st_ino,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns,'nlink':stat.st_nlink}


def reject_symlinks(path):
    path=Path(path).absolute();current=Path(path.anchor)
    for part in path.parts[1:]:
        current/=part
        if current.is_symlink():raise ValueError('symlink input/ancestor forbidden: '+str(current))
    return path


def read_json(path):
    reject_symlinks(path)
    with Path(path).open() as stream:return json.load(stream)


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.flush();os.fsync(stream.fileno())


def verify_readable(path,kind):
    if kind=='.pth':
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(payload,dict):raise ValueError('checkpoint must be a dictionary')
        tensors=[]
        def walk(value):
            if torch.is_tensor(value):tensors.append(value)
            elif isinstance(value,dict):
                for child in value.values():walk(child)
            elif isinstance(value,(list,tuple)):
                for child in value:walk(child)
        walk(payload)
        if not tensors or any(not torch.isfinite(t).all() for t in tensors):raise ValueError('checkpoint has invalid/nonfinite tensors')
    elif kind=='.tar':
        count=0
        with tarfile.open(path,'r:') as archive:
            for member in archive:
                count+=1
                if member.isfile():
                    stream=archive.extractfile(member)
                    if stream is None:raise ValueError('unreadable tar member')
                    actual=0
                    for block in iter(lambda:stream.read(1024*1024),b''):actual+=len(block)
                    if actual!=member.size:raise ValueError('truncated tar member')
        if not count:raise ValueError('empty transport archive')
    else:raise ValueError('unapproved asset type')


def verify_asset(asset):
    source=reject_symlinks(asset['source']);target=reject_symlinks(asset['target'])
    if not source.is_file() or source.stat().st_nlink!=1:raise ValueError('source must be unique regular file')
    if target.exists() or target.with_name(target.name+'.migration-partial').exists():raise FileExistsError('target/partial exists; manual inspection required')
    before=source_identity(source)
    if before['size']!=asset['bytes'] or sha(source)!=asset['sha256']:raise ValueError('approved source identity mismatch')
    verify_readable(source,source.suffix)
    if before!=source_identity(source):raise ValueError('source changed during validation')
    return before


def budget(data_free,system_free,assets=ASSETS,reclaim_bytes=None):
    moved=sum(a['bytes'] for a in assets)
    reclaimed=moved if reclaim_bytes is None else min(moved,reclaim_bytes)
    # Current free already accounts for dev caches and existing metadata.
    data_required=moved+RESERVE+ALLOWANCE
    final_new_cache=304*(FULL_ROWS+TEST_USERS)+1024**2
    system_required=5*TENSOR_BYTES+final_new_cache+RESERVE+ALLOWANCE
    value={'data_current_free':data_free,'data_required':data_required,'data_margin_after_reserve_and_allowance':data_free-data_required,'system_current_free':system_free,'system_projected_free_after_migration':system_free+reclaimed,'system_margin_after_reserve_and_allowance':system_free+reclaimed-system_required,'reclaim_bytes':reclaimed,'final_new_cache_bytes':final_new_cache,'final_system_required':system_required,'existing_dev_caches_counted_again':False,'reserve_bytes':RESERVE,'extra_allowance_bytes':ALLOWANCE}
    if data_free<data_required or system_free+reclaimed<system_required:raise OSError('insufficient actual filesystem capacity: '+json.dumps(value))
    return value


def archive_gate(args,manifest,protocol,code_hashes):
    path=reject_symlinks(args.complete_archive_receipt)
    if sha(path)!=args.complete_archive_sha256:raise ValueError('complete archive receipt SHA mismatch')
    receipt=read_json(path);inventory=receipt.get('inventory',{});audit=inventory.get('audit',{})
    if receipt.get('status')!='complete' or receipt.get('remote_writes') is not False or inventory.get('ready') is not True or audit.get('test_accessed') is not False:raise ValueError('complete local archive receipt required')
    expected={'protocol_hash':protocol['manifest_hash'],'dev_manifest_hash':manifest['manifest_hash'],'code_hashes':code_hashes,'history_scope_sha256':protocol['history_scope_sha256'],'historical_closure_sha256':protocol['historical_closure_sha256']}
    if any(audit.get(k)!=v for k,v in expected.items()):raise ValueError('complete archive audit identity mismatch')
    dev=Path(args.dev_dir);pro=Path(manifest['protocol_manifest']).parent
    allowed_dev=set(manifest['evidence_hashes'])|{'dev_manifest.json','COMPLETED.json'}|{f'seed_{seed}/{variant}/COMPLETED.json' for seed in (42,43,44) for variant in ('raw','normalized_gated','zero_cross')}
    if set(inventory.get('dev',{}))!=allowed_dev:raise ValueError('complete archive dev inventory mismatch')
    for rel,meta in inventory['dev'].items():
        path=reject_symlinks(dev/rel)
        expected_sha=manifest['evidence_hashes'].get(rel)
        if expected_sha is None:expected_sha=sha(path)
        if meta.get('sha256')!=expected_sha or meta.get('size')!=path.stat().st_size:raise ValueError('complete archive dev seal mismatch')
    allowed_pro={'protocol_manifest.json','historical_users_verified.json','train_records.json','screen_records.json','confirm_records.json','test_records.json','ranker_train_positions.npy'}
    for label in ('screen','confirm','train'):
        allowed_pro.update((f'candidate_{label}.json',f'candidate_{label}.items.npy',f'candidate_{label}.lengths.npy',f'candidate_{label}.json.building'))
    entries={str(p.relative_to(pro)) for p in pro.rglob('*') if p.is_file() or p.is_symlink()}
    if not entries.issubset(allowed_pro) or set(inventory.get('protocol',{}))!=entries:raise ValueError('complete archive protocol inventory mismatch')
    for rel,meta in inventory['protocol'].items():
        path=reject_symlinks(pro/rel)
        if meta.get('size')!=path.stat().st_size or meta.get('sha256')!=sha(path):raise ValueError('complete archive protocol seal mismatch')
    receipt_sha=sha(reject_symlinks(args.complete_archive_receipt))
    if receipt_sha!=args.complete_archive_sha256:raise ValueError('complete archive receipt changed during validation')
    return {'path':str(reject_symlinks(args.complete_archive_receipt)),'sha256':receipt_sha,'run_id':receipt.get('run_id'),'local_tree_reverification':'required on archival host before transferring this original receipt'}


def official_gate(args):
    code=reject_symlinks(args.code_dir);dev=reject_symlinks(args.dev_dir);plan_path=reject_symlinks(args.final_plan)
    launch=read_json(args.capacity_receipt)
    if launch.get('status')!='pass' or launch.get('protocol_hash')!=PROTOCOL_HASH or launch.get('R')!=604511 or launch.get('N')!=TEST_USERS:raise ValueError('capacity receipt identity mismatch')
    for key,asset in zip(('required_post_dev_migration','additional_post_dev_migration'),ASSETS):
        record=launch.get(key,{})
        if any(record.get(k)!=v for k,v in asset.items()):raise ValueError('unapproved migration asset')
    # Disable bytecode writes; these official validators read proof/evidence only.
    sys.dont_write_bytecode=True;sys.path.insert(0,str(code))
    import run_cross_multiseed as r
    raw_plan=read_json(plan_path)
    protocol_path=reject_symlinks(raw_plan['protocol_manifest'])
    for entry in protocol_path.parent.iterdir():
        if entry.name.startswith(('TEST_','candidate_final','candidate_test')):raise ValueError('final stage already started; migration forbidden')
    manifest,protocol=r._validate_dev_evidence(dev.resolve())
    plan=r._load_plan(plan_path.resolve())
    if not plan.get('final_evaluation_allowed') or plan.get('winner') not in ('normalized_gated','zero_cross'):raise ValueError('migration allowed only for accepted locked winner')
    reject_symlinks(plan['protocol_manifest'])
    selection=read_json(plan['selection'])
    if Path(selection['dev_run']).resolve()!=dev.resolve() or selection.get('status')!='accepted':raise ValueError('selection does not bind completed dev')
    if protocol['manifest_hash']!=PROTOCOL_HASH or plan['protocol_manifest_hash']!=PROTOCOL_HASH or protocol['counts']['test']!=TEST_USERS or protocol['ranker_train_rows']!=604511:raise ValueError('locked protocol identity mismatch')
    archive=archive_gate(args,manifest,protocol,r.code_hashes())
    return {'protocol_hash':PROTOCOL_HASH,'dev_manifest_hash':manifest['manifest_hash'],'plan_hash':plan['plan_hash'],'winner':plan['winner'],'code_hashes':r.code_hashes(),'complete_archive':archive,'capacity_receipt_sha256':sha(args.capacity_receipt),'final_plan_sha256':sha(plan_path),'protected_roots':[str(code.resolve()),str(dev.resolve()),str(Path(plan['protocol_manifest']).resolve().parent)]}


def copy_and_switch(asset,identity):
    source=Path(asset['source']);target=Path(asset['target'])
    reject_symlinks(source);reject_symlinks(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    reject_symlinks(target)
    temporary=target.with_name(target.name+'.migration-partial');link=source.with_name(source.name+'.migration-link')
    if target.exists() or temporary.exists() or link.exists() or link.is_symlink():raise FileExistsError('migration output already exists')
    with source.open('rb') as inp,temporary.open('xb') as out:
        shutil.copyfileobj(inp,out,length=1024*1024);out.flush();os.fsync(out.fileno())
    if temporary.stat().st_size!=asset['bytes'] or sha(temporary)!=asset['sha256']:raise ValueError('copied asset identity mismatch')
    verify_readable(temporary,source.suffix)
    if source_identity(source)!=identity or sha(source)!=asset['sha256']:raise ValueError('source changed before atomic switch')
    reject_symlinks(target)
    if target.exists():raise FileExistsError(target)
    # Atomic no-replace publication. Retain the temporary hardlink as evidence;
    # it names the same destination inode and consumes no second content payload.
    os.link(temporary,target);fsync_dir(target.parent)
    os.symlink(str(target),link)
    if os.readlink(link)!=str(target) or sha(link)!=asset['sha256']:raise ValueError('staged source symlink mismatch')
    # Destination content is durable and independently verified before replacing
    # only the original directory entry. No content-deletion operation exists.
    reject_symlinks(source)
    if source_identity(source)!=identity:raise ValueError('source changed at switch')
    os.replace(link,source);fsync_dir(source.parent)
    if not source.is_symlink() or os.readlink(source)!=str(target) or sha(source)!=asset['sha256']:raise ValueError('post-switch source verification failed')
    verify_readable(source,source.suffix)
    return {'source':str(source),'target':str(target),'bytes':asset['bytes'],'sha256':asset['sha256'],'source_is_expected_symlink':True}


def remaining_capacity(assets,data_parent,system_parent):
    reclaimed=sum(min(a['bytes'],Path(a['source']).stat().st_blocks*512) for a in assets)
    return budget(shutil.disk_usage(data_parent).free,shutil.disk_usage(system_parent).free,assets,reclaimed)


def storage_check(assets):
    data_parent=Path(assets[0]['target']).parent
    while not data_parent.exists():data_parent=data_parent.parent
    system_parent=Path(assets[0]['source']).parent
    reject_symlinks(data_parent);reject_symlinks(system_parent)
    source_dev=system_parent.stat().st_dev;target_dev=data_parent.stat().st_dev
    if source_dev==target_dev:raise ValueError('migration requires separate source/target filesystems')
    for asset in assets:
        if Path(asset['source']).stat().st_dev!=source_dev:raise ValueError('sources not on expected filesystem')
        parent=Path(asset['target']).parent
        while not parent.exists():parent=parent.parent
        if parent.stat().st_dev!=target_dev:raise ValueError('targets not on expected filesystem')
    reclaimed=sum(min(a['bytes'],Path(a['source']).stat().st_blocks*512) for a in assets)
    capacity=budget(shutil.disk_usage(data_parent).free,shutil.disk_usage(system_parent).free,assets,reclaimed)
    return capacity,data_parent,system_parent


def run(args,*,gate=official_gate,assets=ASSETS,storage=storage_check,remaining=remaining_capacity):
    # Dependency arguments support synthetic tests; CLI always uses approved constants.
    receipt_dir=reject_symlinks(args.execution_dir)
    lock=reject_symlinks(Path(assets[0]['target']).parent/'.asset-migration.started')
    if receipt_dir==lock or receipt_dir.is_relative_to(lock) or lock.is_relative_to(receipt_dir):raise ValueError('execution directory overlaps global lock')
    if lock.exists():raise FileExistsError('global migration evidence exists; no automatic retry')
    if receipt_dir.exists():raise FileExistsError('execution evidence exists; no automatic retry')
    for asset in assets:
        for path in (Path(asset['source']),Path(asset['target'])):
            if path==receipt_dir or path.is_relative_to(receipt_dir):raise ValueError('execution directory overlaps an asset')
    proof=gate(args)
    identities=[verify_asset(asset) for asset in assets]
    for root in proof.get('protected_roots',[]):
        root=Path(root)
        if receipt_dir==root or receipt_dir.is_relative_to(root) or root.is_relative_to(receipt_dir):
            raise ValueError('execution receipt must be outside protected evidence trees')
    capacity,data_parent,system_parent=storage(assets)
    record={'status':'check_passed','proof':proof,'capacity':capacity,'assets':list(assets),'execute':bool(args.execute),'utc':datetime.now(timezone.utc).isoformat()}
    if not args.execute:return record
    # Fixed global lock prevents concurrent attempts with different receipt paths.
    reject_symlinks(lock);reject_symlinks(receipt_dir)
    lock.parent.mkdir(parents=True,exist_ok=True);lock.mkdir(exist_ok=False);write_new(lock/'STARTED.json',{'execution_dir':str(receipt_dir),'record':record});fsync_dir(lock);fsync_dir(lock.parent)
    receipt_dir.mkdir(parents=True,exist_ok=False);write_new(receipt_dir/'STARTED.json',record);fsync_dir(receipt_dir)
    completed=[]
    try:
        for index,(asset,identity) in enumerate(zip(assets,identities)):
            remaining(assets[index:],data_parent,system_parent)
            completed.append(copy_and_switch(asset,identity));write_new(receipt_dir/f'asset_{len(completed)}.json',completed[-1])
        after_capacity=remaining([],data_parent,system_parent)
        record.update(status='complete',completed=completed,after_capacity=after_capacity,after_free={'data':after_capacity['data_current_free'],'system':after_capacity['system_current_free']})
        write_new(receipt_dir/'COMPLETED.json',record);fsync_dir(receipt_dir);return record
    except BaseException as exc:
        write_new(receipt_dir/'FAILED.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','completed':completed,'no_automatic_retry':True});fsync_dir(receipt_dir);raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-dir',required=True);p.add_argument('--dev-dir',required=True);p.add_argument('--final-plan',required=True);p.add_argument('--capacity-receipt',required=True);p.add_argument('--complete-archive-receipt',required=True);p.add_argument('--complete-archive-sha256',required=True);p.add_argument('--execution-dir',required=True);p.add_argument('--execute',action='store_true');return p

if __name__=='__main__':
    torch.set_num_threads(4);print(json.dumps(run(parser().parse_args()),indent=2))
