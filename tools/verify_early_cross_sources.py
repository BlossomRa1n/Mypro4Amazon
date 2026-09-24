"""Authenticate archived early-cross code, monitor, cohorts and candidate arrays.

Complements model/metric semantic verification and byte-transfer receipts.
Only existing development arrays are read; final test records are not opened.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
from verify_training_diagnostics_archive import sha as file_sha, read_json as file_json, require, canonical_hash


FROZEN_SOURCES = {
    'run_early_stop_cross.py': 'cacafb6b0916fd5f934debeb3f1dc73d1677dea5afe0b857fa807d89782f3282',
    'archive_verified.json': '31ec73c1e56238b49df173c0cb97713e8d7e34d4daf581ed4f59542a94e2d95d',
    'monitor_training_diagnostics.py': '90438083cab02ae387dbd5c6638e2d8ae62e9b51b37845f530b3eb6e8e7a1344',
    'OVERNIGHT_CROSS_PROTOCOL_20260925.md': '8d6dc4a421c84f09ed4dd626c2fbb144557b00a561cbf1de1be62baa87f6d710',
    'RECOVERY_BACKUP_BINDING.json': '37a23966049016bd4e63aa3392c5f9522e7f82a155868c1c4a1c1c547c630494',
    'test_early_stop_cross.py': '05ccd16b5ff6f68e2498c1a4b182f8ca437241613c1681c0898fe2c17122b4c7',
}


def safe_path(value):
    path=Path(value).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)), 'unsafe path: '+str(path))
    return path


def sha(path):
    path=safe_path(path)
    require(path.is_file(), 'missing input file: '+str(path))
    return file_sha(path)


def read_json(path):
    return file_json(safe_path(path))


def checked_receipt(value, readonly):
    path=safe_path(value)
    require(not os.path.lexists(path), 'receipt already exists')
    require(not any(path==tree or tree in path.parents for tree in map(safe_path,readonly)), 'receipt is inside read-only input tree')
    return path


def check_sources(sources, cross, remote):
    require(sources==FROZEN_SOURCES, 'frozen source set or hashes differ')
    for name,digest in FROZEN_SOURCES.items():
        require(sha(cross/'src'/name)==digest and sha(remote/'src'/name)==digest, 'deployed source mismatch: '+name)


def check_transfer(receipt_path, local_root, remote_root, marker, cross_sha=None, rehash=False):
    local_root=safe_path(local_root)
    transfer=read_json(receipt_path)
    inventory_path=receipt_path.with_suffix('.inventory.json')
    inventory=read_json(inventory_path)
    require(transfer.get('status')=='verified', 'transfer not verified')
    require(transfer.get('inventory_sha256')==sha(inventory_path), 'transfer inventory hash differs')
    require(transfer.get('remote_root')==inventory.get('root')==remote_root, 'transfer remote root differs')
    require(transfer.get('local_root')==str(local_root.resolve()), 'transfer local root differs')
    require(inventory.get('marker')==marker, 'transfer completion marker differs')
    for flag in ('remote_inventory_before_after_equal','local_exact_coverage','local_sha_size_equal'):
        require(transfer.get(flag) is True, 'transfer check failed: '+flag)
    require(transfer.get('remote_mutated') is False, 'remote transfer mutation')
    files=inventory['files']
    actual={str(p.relative_to(local_root)):p for p in local_root.rglob('*') if not p.is_dir() or p.is_symlink()}
    require(set(actual)==set(files), 'transfer local file coverage differs')
    for name,ref in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe inventory path')
        path=safe_path(local_root/name)
        require(path.is_file() and path.stat().st_size==ref['bytes'], 'transfer local file size differs: '+name)
        if rehash:require(sha(path)==ref['sha256'], 'transfer local hash differs: '+name)
    require(transfer.get('files')==len(files) and transfer.get('bytes')==sum(ref['bytes'] for ref in files.values()), 'transfer inventory totals differ')
    require(sha(local_root/marker)==files[marker]['sha256']==transfer.get('completion_sha256'), 'transfer completion hash differs')
    if cross_sha is not None:
        require(transfer.get('cross_completion_sha256')==cross_sha==transfer.get('completion_sha256'), 'transfer cross completion differs')
    return files


def check_control_launch(launch):
    run='/root/autodl-tmp/early_stop_cross_20260925_v1'
    source='/root/early_stop_cross_20260925/src/'
    expected={
        '--legacy-source-dir':'/root/training_diagnostics_20260924/src_v2/code',
        '--protocol-manifest':'/root/training_diagnostics_20260924/protocol_v2/protocol_manifest.json',
        '--raw-run':'/root/autodl-tmp/training_diagnostics_20260924_raw_v1',
        '--old-zero-run':'/root/autodl-tmp/next_cross_20260924_dev',
        '--run-dir':run, '--protocol-doc':source+'OVERNIGHT_CROSS_PROTOCOL_20260925.md',
        '--archive-receipt':source+'archive_verified.json',
        '--backup-receipt':source+'RECOVERY_BACKUP_BINDING.json', '--device':'cuda',
    }
    require(launch.get('run_dir')==run and launch.get('control_dir')=='/root/early_stop_cross_20260925/control_v1', 'control launch roots differ')
    require(launch.get('automatic_restart') is False and launch.get('shutdown_allowed') is False, 'control launch policy differs')
    command=launch.get('command',[])
    require(isinstance(command,list) and len(command)==2+2*len(expected) and command[:2]==['/root/miniconda3/bin/python',source+'run_early_stop_cross.py'], 'control launch runner differs')
    require(len(set(command[2::2]))==len(expected) and dict(zip(command[2::2],command[3::2]))==expected, 'control launch source arguments differ')


def checked_arrays(run, completed, split, per_seed):
    require(set(completed['seeds'])=={'42','43','44'}, 'cross seed set differs')
    paths=sorted(run.glob(f'seed_*/zero_cross/{split}*_users.npz'))
    require(len(paths)==3*per_seed, 'unexpected cross array count')
    for seed in ('42','43','44'):
        require(sum(p.parent.parent.name=='seed_'+seed for p in paths)==per_seed, 'unexpected per-seed array count')
    for path in paths:
        require(sha(path)==completed['evidence_hashes'].get(str(path.relative_to(run))), 'cross array completion seal differs: '+str(path))
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--receipt',required=True)
    args=p.parse_args();root=safe_path(args.repo)
    cross=root/'server_snapshot/early_stop_cross_20260925'
    raw=root/'server_snapshot/training_diagnostics_20260924'
    original=root/'server_snapshot/next_cross_20260924/protocol_complete'
    run=cross/'run';remote=cross/'remote_root'
    readonly=(run,remote,cross/'src',raw,original)
    receipt=checked_receipt(args.receipt,readonly)
    completed=read_json(run/'COMPLETED.json');done_sha=sha(run/'COMPLETED.json')
    require(completed.get('status')=='complete' and completed.get('test_future_labels_read') is False,'cross completion/test gate failed')
    sources=read_json(cross/'source_manifest.json')
    check_sources(sources,cross,remote)
    run_inventory=check_transfer(cross/'transfer_verified.json',run,'/root/autodl-tmp/early_stop_cross_20260925_v1','COMPLETED.json',done_sha)
    check_transfer(cross/'control_transfer_verified.json',remote,'/root/early_stop_cross_20260925','control_v1/COMPLETED.json',rehash=True)
    require(sha(run/'run_manifest.json')==run_inventory['run_manifest.json']['sha256'], 'run manifest transfer seal differs')
    manifest=read_json(run/'run_manifest.json')
    require(manifest['runner_sha256']==sources['run_early_stop_cross.py'],'runner code seal mismatch')
    require(manifest['protocol_doc_sha256']==sources['OVERNIGHT_CROSS_PROTOCOL_20260925.md'],'cross document drift')
    protocol_path=raw/'remote_root/protocol_v2/protocol_manifest.json'
    protocol=read_json(protocol_path)
    require(protocol['manifest_hash']=='c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6','wrong legacy protocol')
    require(canonical_hash({k:v for k,v in protocol.items() if k!='manifest_hash'})==protocol['manifest_hash'],'legacy seal drift')
    require(manifest['legacy_source_hashes']==protocol['code_hashes'],'cross legacy source map differs')
    for name,digest in protocol['code_hashes'].items():
        require(sha(raw/'remote_root/src_v2/code'/name)==digest,'local frozen dependency differs')
    require(sha(original/'protocol_manifest.json')==protocol['old_protocol']['sha256']=='4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1','old cohort protocol drift')
    old=read_json(original/'protocol_manifest.json')
    count=0
    for split,total in (('screen',9),('confirm',3)):
        records=read_json(original/f'{split}_records.json')
        require(canonical_hash(records)==old['cohort_hashes'][split],'original cohort order changed')
        side=original/f'candidate_{split}.json';meta=read_json(side)
        require(sha(side)==protocol['candidate_caches'][split]['sha256'],'original pool sidecar drift')
        require(meta['records_hash']==old['cohort_hashes'][split],'pool/cohort binding differs')
        for ref in meta['files'].values():
            require(Path(ref['name']).name==ref['name'] and sha(original/ref['name'])==ref['sha256'],'pool payload drift')
        items=np.load(original/meta['files']['items']['name'],allow_pickle=False,mmap_mode='r')
        lengths=np.load(original/meta['files']['lengths']['name'],allow_pickle=False,mmap_mode='r')
        uid=np.asarray([row[0] for row in records]);position=np.asarray([row[1] for row in records])
        paths=checked_arrays(run,completed,split,total//3)
        for path in paths:
            with np.load(path,allow_pickle=False) as arrays:
                require(np.array_equal(arrays['uid'],uid) and np.array_equal(arrays['position'],position),'cross cohort identity/order mismatch')
                require(np.array_equal(arrays['candidate_ids'],items) and np.array_equal(arrays['lengths'],lengths),'cross candidate identity/order mismatch')
            count+=1
    check_control_launch(read_json(remote/'control_v1/launch.json'))
    monitor=read_json(remote/'control_v1/COMPLETED.json')
    require(monitor['status']=='complete' and monitor['returncode']==0 and monitor['failed_marker'] is False and monitor['completed_marker'] is True,'supervisor completion failed')
    require(read_json(remote/'control_v1/status.json')==monitor,'supervisor current status differs from completion')
    result=dict(status='verified',cross_completion_sha256=done_sha,source_files=len(sources),frozen_dependency_files=len(protocol['code_hashes']),cohort_candidate_arrays=count,monitor_exit=0,protocol_manifest_sha256=sha(protocol_path),source_manifest_sha256=sha(cross/'source_manifest.json'),transfer_sha256=sha(cross/'transfer_verified.json'),control_transfer_sha256=sha(cross/'control_transfer_verified.json'),test100k_accessed=False)
    receipt=checked_receipt(receipt,readonly)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
