"""Validate local byte-identical mappings for remote final-plan dependencies.

No remote execution, source-reference rewriting, future-target reads or scoring.
Mappings are supplied by the operator; all referenced source bytes are checked.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from verify_training_diagnostics_archive import read_json, sha, canonical_hash, require


def safe_path(value):
    path=Path(value).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)), 'unsafe source mapping or receipt')
    return path


def checked_receipt(value, readonly):
    path=safe_path(value)
    require(not os.path.lexists(path), 'source receipt already exists')
    require(not any(path==tree or tree in path.parents for tree in map(safe_path,readonly)), 'receipt is inside read-only input tree')
    return path


def checked_local(path, expected, used):
    path=safe_path(path)
    require(path.is_file() and sha(path)==expected,'local source bytes differ: '+str(path))
    used[str(path)]={'sha256':expected,'bytes':path.stat().st_size}
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization',required=True);p.add_argument('--mapping',required=True)
    p.add_argument('--legacy-code',required=True);p.add_argument('--base-assets',required=True)
    p.add_argument('--evidence-root',required=True);p.add_argument('--old-protocol',required=True)
    p.add_argument('--executor',required=True);p.add_argument('--receipt',required=True)
    args=p.parse_args();receipt=checked_receipt(args.receipt,())
    authorization=safe_path(args.authorization);mapping_path=safe_path(args.mapping)
    auth=read_json(authorization);mapping=read_json(mapping_path);used={}
    readonly=[safe_path(x) for x in (args.legacy_code,args.base_assets,args.evidence_root)]
    readonly += [safe_path(x).parent for x in (args.old_protocol,args.executor,authorization,mapping_path,*mapping.values())]
    receipt=checked_receipt(receipt,readonly)
    authorization_sha=sha(authorization);mapping_sha=sha(mapping_path)
    require(auth.get('status')=='authorized' and auth.get('smoke') is False,'formal reviewed authorization required')
    require(set(mapping)==set(auth['refs']),'source reference map coverage mismatch')
    refs={}
    for name,ref in auth['refs'].items():
        require(isinstance(ref.get('path'),str) and Path(ref['path']).is_absolute(),'remote reference must be absolute')
        refs[name]=checked_local(mapping[name],ref['sha256'],used)
    diagnostic=read_json(refs['diagnostic_manifest'])
    require(diagnostic.get('manifest_hash')==canonical_hash({k:v for k,v in diagnostic.items() if k!='manifest_hash'})=='c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6','diagnostic source seal mismatch')
    code=safe_path(args.legacy_code)
    require({p.name for p in code.glob('*.py')}==set(diagnostic['code_hashes']),'legacy dependency set differs')
    for name,digest in diagnostic['code_hashes'].items():
        require(Path(name).name==name,'unsafe legacy source name')
        checked_local(code/name,digest,used)
    oldpath=checked_local(args.old_protocol,diagnostic['old_protocol']['sha256'],used)
    require(diagnostic['old_protocol']['sha256']=='4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1','wrong legacy cohort protocol')
    old=read_json(oldpath)
    require(old['manifest_hash']==canonical_hash({k:v for k,v in old.items() if k!='manifest_hash'}),'old protocol self-seal mismatch')
    for name,digest in old['base_assets'].items():
        require(Path(name).name==name,'unsafe base asset name')
        checked_local(Path(args.base_assets)/name,digest,used)
    for rel,digest in diagnostic['test_identity_receipt']['verified_evidence'].items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'unsafe identity source reference')
        checked_local(Path(args.evidence_root)/rel,digest,used)
    require(sha(refs['test_lock'])=='9416c80d3c2bda822239e19a1ffd5cc8b0497c1aca252225dee26329a9cf4303','test lock differs')
    require(sha(refs['test_records'])=='086e977d4d7bb06f453bae2596dcdddb514796add4439bb5bb5ecdd18d8732fc','test records differ')
    require(sha(refs['test_identity_seal'])=='30087cca4a9d7e3db166c8bff7d41cd3ef06226b78087a7a2a0989f5e199e188','test identity seal differs')
    for name in ('cross_archive','cross_sources','cross_transfer'):
        gate=read_json(refs[name]);require(gate.get('status')=='verified' and gate.get('cross_completion_sha256')==auth['refs']['cross_completion']['sha256'],'cross prerequisite gate differs')
    for name in ('full_pool_completion','pilot_completion'):
        pool=read_json(refs[name]);require(pool.get('status')=='complete' and pool.get('test_future_labels_read') is False and pool.get('test_pool_constructed') is False,'pool construction not qualified')
    executor=safe_path(args.executor)
    require(sha(executor)=='ae8756bde5ef024fbafb2078f615d348e4e7a71d51d13afa9a8bf50b284df862','final executor differs from audited source')
    result={'status':'verified','authorization_sha256':authorization_sha,'mapping_sha256':mapping_sha,'executor_sha256':sha(executor),'references':{name:{'remote_path':auth['refs'][name]['path'],'local_path':str(path),'sha256':auth['refs'][name]['sha256']} for name,path in refs.items()},'verified_source_files':len(used),'files':used,'test_future_labels_read':False,'remote_references_rewritten':False,'limitations':['Checks local dependency identity and exact bytes; the remote executor rechecks deployed sources and original paths.', 'Resource feasibility and real run semantic/transfer gates remain separate.']}
    require(sha(authorization)==authorization_sha and sha(mapping_path)==mapping_sha,'authorization or mapping changed during verification')
    receipt=checked_receipt(receipt,readonly)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'verified','verified_source_files':len(used),'authorization_sha256':result['authorization_sha256']}))


if __name__=='__main__':main()
