"""Authenticate completed full-pool source, transfer and pilot provenance gates.

Reads local frozen source/JSON and hashes pool archive bytes only. No source
imports, model loading, candidate generation, future-target access or SSH.
Run only after both transfers and the formal semantic archive verifier finish.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

BUILDER_SHA='b6534421c6c65d794205a5a90844cd5a11343685a870c04f545d86dc5a6aa0d3'
TEST_SHA='2791ab2d120d46c5c1e46f69b084ea9ec0cb909eb805f24e9ff5ad66e204c565'
SUPERVISOR_SHA='e8c67a0118c49e0f2fd01563457c2a535b22905a591cee71f9ac60c822303c0a'
VERIFIER_SHA='53c388ad768d4c124b94fb426a4374370c8634575e69ca32cc781b95eb62d75a'
PILOT_SHA='1bd80ad658977dcedde5b5d459756df9be5b653acb13f392debf7ad7c9a1fbc3'
DIAGNOSTIC_SHA='aca3a5258389d7a8387352a8d33e596198df6380ba428d8059db2106859f45a9'
OLD_SHA='4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1'
REMOTE='/root/full_prefix_pools_20260925'
FULL='/root/autodl-tmp/full_prefix_pools_20260925_full_v1'
PILOT='full_prefix_pools_20260925_pilot_v1'
LEGACY='/root/training_diagnostics_20260924/src_v2/code'
COMMAND=['/root/miniconda3/bin/python','-u',REMOTE+'/src/build_full_prefix_pools.py',
         '--legacy-source-dir',LEGACY,'--protocol-manifest',
         '/root/training_diagnostics_20260924/protocol_v2/protocol_manifest.json',
         '--output-dir',FULL,'--workers','4']


def require(condition, message):
    if not condition:raise ValueError(message)


def safe(value):
    path=Path(value).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)), 'unsafe input/receipt path: '+str(path))
    return path


def digest(stream):
    value=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def sha(value):
    path=safe(value);require(path.is_file(),'missing regular input: '+str(path))
    with path.open('rb') as stream:return digest(stream)


def read(value):
    return json.loads(safe(value).read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def output_path(value, readonly):
    path=safe(value)
    require(not os.path.lexists(path),'receipt already exists')
    require(not any(path==tree or tree in path.parents for tree in map(safe,readonly)), 'receipt inside read-only input tree')
    return path


def tree_files(root):
    root=safe(root);require(root.is_dir(),'missing input tree: '+str(root))
    result={}
    for path in root.rglob('*'):
        safe(path)
        require(path.is_file() or path.is_dir(),'special input tree member')
        if path.is_file():result[path.relative_to(root).as_posix()]=path
    return result


def relative(name):
    path=PurePosixPath(name)
    require(isinstance(name,str) and name and not path.is_absolute() and '..' not in path.parts and path.as_posix()==name,'unsafe/noncanonical inventory member')
    return path


def transfer(receipt_path, local, remote, marker):
    receipt=read(receipt_path);inventory_path=receipt_path.with_suffix('.inventory.json');inventory=read(inventory_path)
    require(receipt.get('status')=='verified' and receipt.get('inventory_sha256')==sha(inventory_path),'transfer inventory seal differs')
    require(receipt.get('local_root')==str(safe(local).resolve()),'transfer local root differs')
    require(receipt.get('remote_root')==inventory.get('root')==remote and inventory.get('marker')==marker,'transfer remote role/root/marker differs')
    for key in ('remote_inventory_before_after_equal','local_exact_coverage','local_sha_size_equal'):
        require(receipt.get(key) is True,'transfer check not true: '+key)
    require(receipt.get('remote_mutated') is False,'transfer mutated remote input')
    files=inventory['files'];actual=tree_files(local)
    require(set(actual)==set(files),'transfer exact local coverage differs')
    for name,entry in files.items():
        relative(name);path=actual[name]
        require(type(entry['bytes']) is int and entry['bytes']>=0 and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'transfer file SHA/size differs: '+name)
    require(receipt.get('files')==len(files) and receipt.get('bytes')==sum(x['bytes'] for x in files.values()),'transfer totals differ')
    require(receipt.get('completion_sha256')==files[marker]['sha256']==sha(local/marker),'transfer completion differs')
    return files


def check_launch(value):
    require(value.get('command')==COMMAND,'formal launch command changed')
    require(value.get('automatic_restart') is False and value.get('test_allowed') is False and value.get('interval_seconds')==900,'formal supervisor policy changed')


def check_supervisor(archive, remote):
    require(sha(archive/'supervise_full.py')==sha(remote/'supervise_full.py')==SUPERVISOR_SHA,'frozen supervisor source differs')


def check_control(control, status):
    require(control.get('status')=='complete' and control.get('returncode')==0 and control.get('completed') is True,'control completion failed')
    required={'utc','pid','returncode','elapsed_seconds','free_bytes','memory_current','memory_max','completed','log_tail'}
    require(set(status)==required and set(control)==required|{'status','peak_child_rss_kib'},'control final status fields differ')
    require(status=={key:value for key,value in control.items() if key not in ('status','peak_child_rss_kib')},'control final status differs')


def check_completion(value, mode):
    require(value.get('schema')=='full-prefix-pools-v1' and value.get('status')=='complete' and value.get('mode')==mode,'pool completion role/status differs')
    require(value.get('workers')==4 and value.get('cpu_only') is True and value.get('test_future_labels_read') is False and value.get('test_pool_constructed') is False,'pool completion execution policy differs')
    require(value.get('builder_sha256')==BUILDER_SHA and value.get('frozen_source_dir')==LEGACY and value.get('protocol_manifest_sha256')==DIAGNOSTIC_SHA and value.get('old_protocol_sha256')==OLD_SHA,'pool completion frozen source differs')
    expected=FULL if mode=='full' else '/root/autodl-tmp/'+PILOT
    require(value['disk_preflight']['output_directory']==expected,'pool completion output role differs')
    require(value.get('full_legal_rows')==4731777 and value.get('selected_rows')==(4731777 if mode=='full' else 2048),'pool completion row scope differs')


def check_pilot(archive, pilot, semantic):
    completion=sha(pilot/'COMPLETED.json')
    require(completion==sha(archive/'pilot_COMPLETED.json')==PILOT_SHA==semantic['pilot_completion_sha256'],'pilot differs from independently captured completion')
    extraction=read(archive/'pilot_extraction_receipt.json')
    require(extraction.get('status')=='verified' and extraction.get('local_completion_matches_previously_captured') is True and extraction.get('completion_sha256')==completion and extraction.get('tar_sha256')==sha(archive/'pilot_archive.tar.gz'),'pilot extraction receipt differs')
    actual=tree_files(pilot);sealed=semantic['pilot_files'];require(set(actual)==set(sealed),'pilot semantic file coverage differs')
    seen=set();members=0;root_seen=False
    with tarfile.open(safe(archive/'pilot_archive.tar.gz'),'r:gz') as stream:
        for member in stream:
            members+=1;path=relative(member.name)
            if path.as_posix()==PILOT:
                require(member.isdir() and not root_seen,'invalid/duplicate pilot tar root');root_seen=True;continue
            require(path.parts[0]==PILOT and len(path.parts)==2 and member.isfile(),'unsafe pilot tar member')
            name=path.parts[1];require(name not in seen and name in actual,'duplicate/extra pilot tar payload');seen.add(name)
            ref=sealed[name];require(member.size==ref['bytes']==actual[name].stat().st_size,'pilot tar size differs')
            with stream.extractfile(member) as payload:require(digest(payload)==ref['sha256']==sha(actual[name]),'pilot tar/extracted SHA differs')
    require(root_seen and seen==set(actual) and extraction.get('safe_members')==members,'pilot extraction coverage differs')


def verify(repo, receipt_value):
    repo=safe(repo);archive=repo/'server_snapshot/full_prefix_pools_20260925'
    full=archive/'full_run';remote=archive/'remote_root';pilot=archive/'pilot_extracted'/PILOT
    readonly=(full,remote,archive/'src',archive/'pilot_extracted',repo/'tools',repo/'tests')
    receipt=output_path(receipt_value,readonly)
    # All readiness checks precede any output directory creation or writes.
    for path in (full/'COMPLETED.json',remote/'control_v1/COMPLETED.json',archive/'transfer_verified.json',archive/'control_transfer_verified.json',archive/'archive_verified.json'):
        require(safe(path).is_file(),'prerequisite not ready: '+str(path))
    fullfiles=transfer(archive/'transfer_verified.json',full,FULL,'COMPLETED.json')
    transfer(archive/'control_transfer_verified.json',remote,REMOTE,'control_v1/COMPLETED.json')
    require(not any(path.name=='FAILED.json' for tree in (full,remote/'control_v1') for path in tree.rglob('*')),'FAILED marker exists')
    for name,expected,local in (('build_full_prefix_pools.py',BUILDER_SHA,repo/'tools/build_full_prefix_pools.py'),('test_full_prefix_pools.py',TEST_SHA,repo/'tests/test_full_prefix_pools.py')):
        require(sha(local)==sha(archive/'src'/name)==sha(remote/'src'/name)==expected,'frozen pool source differs: '+name)
    require(sha(repo/'tools/verify_full_prefix_pool_archive.py')==VERIFIER_SHA,'semantic verifier source differs')
    check_supervisor(archive,remote)
    launch=read(remote/'control_v1/launch.json');check_launch(launch)
    require(read(archive/'full_launch.json')==launch,'captured original launch differs')
    control=read(remote/'control_v1/COMPLETED.json');status=read(remote/'control_v1/status.json')
    check_control(control,status)
    semantic=read(archive/'archive_verified.json')
    require(semantic.get('status')=='verified' and semantic.get('formal_archive_verified') is True and semantic.get('smoke') is False,'formal semantic gate absent')
    require(semantic.get('verifier_sha256')==VERIFIER_SHA and semantic.get('builder_sha256')==BUILDER_SHA,'semantic source identity differs')
    require(semantic.get('diagnostic_manifest_sha256')==DIAGNOSTIC_SHA and semantic.get('old_protocol_sha256')==OLD_SHA,'semantic protocol source differs')
    for key in ('test_future_labels_read','models_loaded','candidate_generation_performed'):
        require(semantic.get(key) is False,'semantic isolation check differs: '+key)
    require(semantic.get('full_completion_sha256')==sha(full/'COMPLETED.json'),'full semantic completion differs')
    require(semantic.get('files')==fullfiles,'full semantic/transfer payload map differs')
    fullvalue=read(full/'COMPLETED.json');pilotvalue=read(pilot/'COMPLETED.json')
    for value,mode in ((fullvalue,'full'),(pilotvalue,'pilot')):
        check_completion(value,mode)
        require(value['frozen_code_hashes']==semantic['frozen_code_hashes'],'semantic frozen code map differs')
    require(fullvalue['source']==semantic['source'] and fullvalue['pool_hash']==semantic['pool_hash'] and fullvalue['records_hash']==semantic['records_hash'] and fullvalue['selected_rows']==semantic['rows'],'semantic full pool source differs')
    check_pilot(archive,pilot,semantic)
    result=dict(status='verified',full_completion_sha256=sha(full/'COMPLETED.json'),pilot_completion_sha256=PILOT_SHA,
        builder_sha256=BUILDER_SHA,test_source_sha256=TEST_SHA,supervisor_sha256=SUPERVISOR_SHA,archive_verified_sha256=sha(archive/'archive_verified.json'),
        transfer_sha256=sha(archive/'transfer_verified.json'),control_transfer_sha256=sha(archive/'control_transfer_verified.json'),
        transfer_inventory_sha256=sha(archive/'transfer_verified.inventory.json'),control_inventory_sha256=sha(archive/'control_transfer_verified.inventory.json'),
        control_completion_sha256=sha(remote/'control_v1/COMPLETED.json'),launch_sha256=sha(remote/'control_v1/launch.json'),
        pilot_extraction_receipt_sha256=sha(archive/'pilot_extraction_receipt.json'),full_files=len(fullfiles),
        local_full_root=str(full),remote_full_root=FULL,local_control_root=str(remote),remote_control_root=REMOTE,
        test_future_labels_read=False,models_loaded=False,candidate_generation_performed=False,remote_mutated=False)
    receipt=output_path(receipt,readonly);receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True);parser.add_argument('--receipt',required=True)
    args=parser.parse_args();print(json.dumps(verify(args.repo,args.receipt)))


if __name__=='__main__':main()
