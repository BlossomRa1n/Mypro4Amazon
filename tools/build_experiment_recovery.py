#!/usr/bin/env python3
"""Build and independently restore a local source recovery archive.

Includes current Git bundle plus permitted working-tree text and small receipts.
Explicit large dependencies are SHA-verified external references, never copied or
interpreted. This archive is not a standalone data/model backup. No network,
training, scoring, source mutation, or mutation of earlier recovery evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time

ROOTS = {'code', 'docs', 'tests', 'tools', 'archive'}
EXTS = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.sh', '.bat', '.ps1', '.rst', '.lock', '.html', '.css', '.js', '.ts'}
DENIED = {'server_snapshot', '.git', '.credentials', '.ssh', 'amazon_reviews', 'user_data', 'prediction_result', 'node_modules', '__pycache__', '.venv', 'venv', 'env', 'model_data', 'tmp_data'}
SECRET = re.compile(rb'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|(?:AKIA|ASIA)[A-Z0-9]{16}|(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|https?://[^\s/:]+:[^\s/@]+@|(?im:^\s*(?:export\s+)?(?:AWS_SECRET_ACCESS_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GITHUB_TOKEN|[A-Z_]*PASSWORD)\s*=\s*["\']?[A-Za-z0-9/+_=-]{12,})')
MAX_FILE = 16 * 1024 * 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe(path):
    p = Path(os.path.abspath(path))
    need('..' not in Path(path).parts, 'parent traversal rejected')
    need(not any(q.is_symlink() for q in [p, *p.parents]), 'symlink rejected')
    return p


def relative(name):
    p = PurePosixPath(name)
    need(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name and '\x00' not in name, 'unsafe relative path')
    need(str(p) == name, 'noncanonical relative path')
    return p


def allowed(name):
    p = relative(name)
    if p.name in {'.test_output.txt', 'tmp_resume.txt'}:
        return False
    if any(x.lower() in DENIED or x.lower().startswith('.env') or x.lower() in {'id_rsa', 'id_ed25519', 'credentials', 'secrets'} for x in p.parts):
        return False
    if any(x in p.name.lower() for x in ('credential', 'private_key', 'secret_key')) or p.stem.lower() in {'secret', 'secrets'} or p.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}:
        return False
    return (len(p.parts) == 1 or p.parts[0] in ROOTS) and (p.suffix.lower() in EXTS or p.name in {'.gitignore', 'Dockerfile', 'Makefile'})


def digest(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return {'sha256': h.hexdigest(), 'bytes': safe(path).stat().st_size}


def checked_bytes(path):
    p = safe(path)
    need(p.is_file() and stat.S_ISREG(p.stat().st_mode), 'regular file required')
    need(p.stat().st_size <= MAX_FILE, 'source or receipt exceeds size limit')
    b = p.read_bytes()
    need(not SECRET.search(b), 'potential credential found; content suppressed')
    need(b'\x00' not in b, 'binary source or receipt rejected')
    return b


def write(path, data):
    p = safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data)


def write_json(path, data):
    write(path, (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode())


def publish_verified(path, data):
    """Publish a complete, synced receipt without replacing any existing file."""
    temporary=path.with_suffix('.json.pending')
    write_json(temporary,data)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.link(temporary,path)
    temporary.unlink()


def git_result(repo, *args, stdin=None):
    env={k:v for k,v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0',GIT_OPTIONAL_LOCKS='0',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_LITERAL_PATHSPECS='1',GIT_NO_LAZY_FETCH='1',GIT_ALLOW_PROTOCOL='file',GIT_NO_REPLACE_OBJECTS='1')
    # check-ignore accepts literal paths, not pathspecs; Git rejects the
    # literal-pathspec flag for this command. Keep all other isolation intact.
    if args and args[0]=='check-ignore':env.pop('GIT_LITERAL_PATHSPECS')
    return subprocess.run(['git','-c','core.hooksPath=/dev/null','-c','core.fsmonitor=false','-c','protocol.allow=never','-c','protocol.file.allow=always','-C',str(repo),*args],input=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)


def git(repo, *args, stdin=None):
    result=git_result(repo,*args,stdin=stdin)
    need(result.returncode==0,'local Git operation failed: '+args[0])
    return result.stdout


def source_preflight(repo):
    config=git(repo,'config','--local','--list').decode().lower()
    need(not any('promisor=' in line or 'partialclone' in line or line.startswith('filter.') for line in config.splitlines()),'partial/promisor/filter repository unsupported')
    gitdir=safe(git(repo,'rev-parse','--absolute-git-dir').decode().strip())
    need(not any((gitdir/'objects/pack').glob('*.promisor')),'promisor object pack unsupported')
    need(not (gitdir/'objects/info/alternates').exists(),'alternate object store unsupported')


def names(raw):
    return [x.decode('utf-8') for x in raw.split(b'\0') if x]


def snapshot(repo):
    symbolic=git_result(repo,'symbolic-ref','-q','HEAD')
    need(symbolic.returncode in (0,1),'symbolic HEAD lookup failed')
    return {'head':git(repo,'rev-parse','HEAD').decode().strip(),'symbolic_HEAD':symbolic.stdout.decode().strip() if symbolic.returncode==0 else None,'refs':git(repo,'for-each-ref','--format=%(objectname) %(refname) %(symref)').decode(),'status_hex':git(repo,'status','--porcelain=v1','-z','--untracked-files=all').hex()}


def inventory(repo):
    current = names(git(repo, 'ls-files', '--cached', '--others', '--exclude-standard', '-z'))
    head = names(git(repo, 'ls-tree', '-r', '--name-only', '-z', 'HEAD'))
    allnames = sorted(set(current + head))
    selected = [n for n in allnames if allowed(n)]
    for raw in (git(repo,'ls-tree','-rz','HEAD'),git(repo,'ls-files','--stage','-z')):
        for row in raw.split(b'\0'):
            if not row:continue
            metadata,name=row.split(b'\t',1)
            need(name.decode() not in selected or metadata.split()[0] in (b'100644',b'100755'),'selected symlink/gitlink unsupported')
    present = []
    for n in selected:
        p = safe(repo / n)
        if p.exists():
            data = checked_bytes(p)
            present.append({'path': n, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'mode': stat.S_IMODE(p.stat().st_mode) & 0o777})
    return selected, present, [n for n in allnames if n not in selected]


def load_specs(path):
    data = checked_bytes(path)
    specs = json.loads(data)
    need(isinstance(specs, list), 'dependency manifest must be a list')
    need(len({x['path'] for x in specs}) == len(specs), 'duplicate dependency reference')
    for item in specs:
        need(set(item) == {'path', 'sha256', 'bytes'}, 'reference fields must be path/sha256/bytes')
        need(isinstance(item['path'], str) and re.fullmatch('[0-9a-f]{64}', item['sha256']) and type(item['bytes']) is int and item['bytes'] >= 0, 'invalid dependency reference')
    return specs


def verify_refs(repo, specs, receipts=False):
    result = []
    for item in specs:
        path = safe(item['path'] if Path(item['path']).is_absolute() else repo/item['path'])
        need(not any(x.lower().startswith('.env') or x.lower() in {'.git','.ssh','.credentials','credentials','secrets'} for x in path.parts), 'credential/dependency path rejected')
        need(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'regular dependency required')
        need(digest(path) == {key:item[key] for key in ('sha256','bytes')}, 'dependency SHA/byte mismatch')
        if receipts: checked_bytes(path)
        result.append(dict(item, path=str(path)))
    return result


def extract_safe(archive, destination, members):
    need(not destination.exists(), 'extraction destination exists')
    expected = {item['path']:item for item in members}
    destination.mkdir()
    with tarfile.open(archive, 'r:gz') as tar:
        entries = tar.getmembers(); seen = set()
        for member in entries:
            relative(member.name)
            need(member.isfile() and member.name not in seen and member.name in expected, 'unsafe tar entry')
            need(member.size == expected[member.name]['bytes'] and member.mode == expected[member.name]['mode'], 'tar metadata differs')
            seen.add(member.name)
        need(seen == set(expected), 'tar member set differs')
        for member in entries:
            target = destination/member.name; target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as out:
                shutil.copyfileobj(source,out,1024*1024)
            os.chmod(target, member.mode)
            need(digest(target) == {k:expected[member.name][k] for k in ('sha256','bytes')}, 'extracted SHA differs')


def diff(repo, state, selected, staged=False):
    args = ['diff','--no-ext-diff','--no-textconv','--no-renames','--binary','--full-index']
    if staged: args += ['--cached',state['head']]
    return git(repo,*args,'--',*selected)


def restore_verify(repo, extracted, work, state, selected, files):
    bundle=extracted/'git/repository.bundle'; target=work/'restored_repository'
    bootstrap=work/'bootstrap';bootstrap.mkdir();git(bootstrap,'init','-q')
    git(bootstrap,'bundle','verify',str(bundle))
    refs={line.split()[1]:line.split()[0] for line in git(bootstrap,'bundle','list-heads',str(bundle)).decode().splitlines()}
    for line in state['refs'].splitlines():
        oid,name,*_=line.split();need(refs.get(name)==oid,'bundle refs differ')
    need(state['head'] in refs.values(),'HEAD absent from bundle')
    git(bootstrap,'clone','--no-checkout','--no-local',str(bundle),str(target))
    for line in git(target,'for-each-ref','--format=%(refname)').decode().splitlines():git(target,'update-ref','--no-deref','-d',line)
    for line in state['refs'].splitlines():
        oid,name,*symref=line.split()
        if not symref:git(target,'update-ref','--no-deref',name,oid)
    for line in state['refs'].splitlines():
        oid,name,*symref=line.split()
        if symref:git(target,'symbolic-ref',name,symref[0])
    if state['symbolic_HEAD']:git(target,'symbolic-ref','HEAD',state['symbolic_HEAD'])
    else:git(target,'update-ref','--no-deref','HEAD',state['head'])
    git(target,'read-tree',state['head'])
    indexed=set(names(git(target,'ls-files','-z')))
    checkout=sorted(indexed & set(selected)); excluded=sorted(indexed-set(selected))
    if excluded:git(target,'update-index','--skip-worktree','-z','--stdin',stdin=b''.join(n.encode()+b'\0' for n in excluded))
    if checkout:git(target,'checkout-index','-u','--stdin','-z',stdin=b''.join(n.encode()+b'\0' for n in checkout))
    for kind in ('staged','unstaged'):
        patch=extracted/f'git/{kind}.patch'
        if patch.stat().st_size:
            git(target,'apply','--binary',*(['--index'] if kind=='staged' else []),str(patch))
    indexed=set(names(git(target,'ls-files','-z')))
    for item in files:
        if item['path'] not in indexed:
            write(target/item['path'],(extracted/'working_tree'/item['path']).read_bytes())
            os.chmod(target/item['path'],item['mode'])
    for item in files:
        p=target/item['path']
        need(digest(p)=={k:item[k] for k in ('sha256','bytes')},'restored working-tree SHA differs')
        need(stat.S_IMODE(p.stat().st_mode)==item['mode'],'restored working-tree mode differs')
    present={x['path'] for x in files}
    need(all(not (target/name).exists() for name in selected if name not in present),'restored deletion differs')
    for kind in ('staged','unstaged'):
        need(diff(target,state,selected,kind=='staged')==(extracted/f'git/{kind}.patch').read_bytes(),'restored index/worktree patch differs')
    need(snapshot(target)==state,'restored HEAD/refs/full Git status differs')
    return dict(head_refs_status_verified=True,source_files_verified=len(files),deletions_verified=len(set(selected)-present),staged_and_unstaged_patches_verified=True,excluded_tracked_paths_unmaterialized=len(excluded),path=str(target))


def execute(args):
    repo=safe(args.repo);output=safe(args.output)
    need(git(repo,'rev-parse','--show-toplevel').decode().strip()==str(repo),'repository root required')
    source_preflight(repo)
    need(not output.exists() and not repo.is_relative_to(output),'new independent output required')
    if output.is_relative_to(repo):
        probe=git_result(repo,'check-ignore','-q','--stdin','-z',stdin=os.fsencode(output)+b'\0')
        need(probe.returncode in (0,1),'local Git ignore check failed')
        need(probe.returncode==0,'output within repository must be Git ignored')
    output.mkdir(parents=True)
    try:
        source_identity=digest(__file__)
        inputs={key:dict(path=str(safe(getattr(args,key))),**digest(getattr(args,key))) for key in ('external_assets','receipts')}
        specs=load_specs(args.external_assets);small=load_specs(args.receipts)
        assets=verify_refs(repo,specs);receipts=verify_refs(repo,small,True)
        need(len({x['path'] for x in assets+receipts})==len(assets+receipts),'asset/receipt roles overlap')
        need(all(not Path(x['path']).is_relative_to(output) for x in assets+receipts),'output overlaps dependency')
        state=snapshot(repo);selected,files,excluded=inventory(repo)
        need(files,'empty selected source inventory')
        changed=set(names(git(repo,'diff','--name-only','-z'))) | set(names(git(repo,'diff','--cached','--name-only','-z',state['head']))) | set(names(git(repo,'ls-files','--others','--exclude-standard','-z')))
        need(not (changed-set(selected)),'changed or untracked excluded paths prevent exact source recovery')
        need(not any(x['path'] in {str(repo/name) for name in selected} for x in assets),'external asset overlaps included source')
        payload=output/'payload';payload.mkdir();work=output/'verification';work.mkdir()
        for item in files:
            dest=payload/'working_tree'/item['path'];write(dest,checked_bytes(repo/item['path']));os.chmod(dest,item['mode'])
            need(digest(dest)=={k:item[k] for k in ('sha256','bytes')},'source changed during copy')
        write_json(payload/'git/state.json',state);write(payload/'git/status.porcelain.z',bytes.fromhex(state['status_hex']))
        for kind in ('staged','unstaged'):
            patch=diff(repo,state,selected,kind=='staged');need(not SECRET.search(patch),'potential credential in patch; content suppressed');write(payload/f'git/{kind}.patch',patch)
        git(repo,'bundle','create',str(payload/'git/repository.bundle'),'--all','HEAD')
        for index,item in enumerate(receipts):write(payload/f'receipts/{index:04d}.txt',checked_bytes(item['path']))
        write(payload/'helper/build_experiment_recovery.py',checked_bytes(__file__))
        write(payload/'RESTORE.md',b'''# Local source recovery\n\nThe bundle contains complete reachable Git history, including excluded historical files. Credential scanning covers only current permitted text, patches and explicit receipts; historical objects have not been credential-audited. Keep the archive private.\n\nThis is not a standalone data/model backup. External assets must be available separately and checked against recovery.json. Remote URLs, local Git configuration, hooks, ignored files and excluded working-tree files are not restored.\n\nCold restore uses only this extracted payload and Python's standard library: import helper/build_experiment_recovery.py with importlib, read git/state.json and recovery.json, create a new empty work directory, then call restore_verify(None, extracted_payload_path, new_work_path, state, recovery['selected_paths'], recovery['working_tree_files']). It creates an empty bootstrap Git repository, validates and clones the bundled history, restores symbolic refs/HEAD, applies staged and unstaged patches and untracked text, and verifies all source hashes/modes and full Git status. No original repository or network is used.\n\nBefore extraction verify the outer archive SHA from VERIFIED.json. Extract only regular, uniquely named, canonical paths and verify manifest.json hashes, sizes and modes. Use extract_safe with manifest file descriptors plus the separately trusted manifest descriptor when automating. Never overwrite an existing output.\n''')
        write_json(payload/'recovery.json',dict(schema=1,head=state['head'],selected_paths=selected,excluded_paths=excluded,working_tree_files=files,external_assets=assets,receipts=receipts,input_manifests=inputs,helper=source_identity,scope='Current Git history, permitted source/docs/tests/tools/root text and explicit small receipts; large assets remain external dependencies. Excluded historical files are not materialized during restore.',standalone_data_model_backup=False,network_used=False,training_or_scoring_performed=False))
        members=[dict(path=p.relative_to(payload).as_posix(),**digest(p),mode=stat.S_IMODE(p.stat().st_mode)) for p in sorted(payload.rglob('*')) if p.is_file()]
        write_json(payload/'manifest.json',dict(schema=1,files=members));members.append(dict(path='manifest.json',**digest(payload/'manifest.json'),mode=stat.S_IMODE((payload/'manifest.json').stat().st_mode)))
        archive=output/'experiment_recovery.tar.gz'
        with archive.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz',compresslevel=6) as tar:
            for item in members:tar.add(payload/item['path'],arcname=item['path'],recursive=False)
        extracted=work/'extracted';extract_safe(archive,extracted,members)
        verification=restore_verify(repo,extracted,work,state,selected,files)
        need(snapshot(repo)==state and inventory(repo)==(selected,files,excluded),'source changed during capture')
        need(verify_refs(repo,specs)==assets and verify_refs(repo,small,True)==receipts,'dependency changed during capture')
        need(digest(__file__)==source_identity,'helper source changed')
        for key,item in inputs.items():need(digest(item['path'])=={k:item[k] for k in ('sha256','bytes')},'input manifest changed')
        result=dict(status='verified',archive=dict(path=str(archive),**digest(archive)),manifest_sha256=digest(payload/'manifest.json')['sha256'],head=state['head'],external_assets=assets,receipt_count=len(receipts),restore_verification=verification,source_and_dependencies_unchanged=True,standalone_data_model_backup=False,network_used=False,training_or_scoring_performed=False)
        publish_verified(output/'VERIFIED.json',result);return result
    except BaseException as error:
        write_json(output/'FAILED.json',dict(status='failed',error_type=type(error).__name__,error=str(error),automatic_retry=False));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('repo','external-assets','receipts','output'):parser.add_argument('--'+field,required=True)
    args=parser.parse_args(argv)
    try:print(json.dumps(execute(args),indent=2))
    except Exception as error:parser.exit(1,f'{type(error).__name__}: {error}\n')


if __name__=='__main__':main()
