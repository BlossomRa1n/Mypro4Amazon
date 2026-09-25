"""Isolated local Git repositories exercise source recovery and hostile inputs."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_experiment_recovery as b


def git(repo,*args):
    result=subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=True)
    return result.stdout


def fixture(root):
    repo=root/'repo';repo.mkdir();git(repo,'init','-q');git(repo,'config','user.name','Local Test');git(repo,'config','user.email','test@example.invalid')
    (repo/'code').mkdir();(repo/'docs').mkdir();(repo/'assets').mkdir()
    (repo/'.gitignore').write_text('ignored/\n')
    (repo/'code/a.py').write_text('x=1\n');(repo/'docs/delete.md').write_text('delete me\n')
    (repo/'code/script.sh').write_text('#!/bin/sh\ntrue\n');(repo/'assets/old.bin').write_bytes(b'opaque\x00historical')
    git(repo,'add','.');git(repo,'commit','-qm','initial');git(repo,'branch','extra');git(repo,'tag','v1')
    git(repo,'update-ref','refs/remotes/origin/main','HEAD');git(repo,'symbolic-ref','refs/remotes/origin/HEAD','refs/remotes/origin/main')
    (repo/'code/a.py').write_text('x=2\n');git(repo,'add','code/a.py');(repo/'code/a.py').write_text('x=3\n')
    (repo/'docs/delete.md').unlink();git(repo,'add','docs/delete.md')
    (repo/'code/script.sh').chmod(0o755)
    (repo/'code/new.py').write_text('new=True\n');(repo/'code/staged.py').write_text('staged=True\n');git(repo,'add','code/staged.py')
    (repo/'ignored').mkdir();(repo/'ignored/skip.bin').write_bytes(b'ignored')
    asset=root/'model.bin';asset.write_bytes(b'large opaque asset\x00'*100)
    receipt=root/'receipt.json';receipt.write_text('{"status":"verified"}\n')
    external=root/'external.json';external.write_text(json.dumps([dict(path=str(asset),**b.digest(asset))]))
    receipts=root/'receipts.json';receipts.write_text(json.dumps([dict(path=str(receipt),**b.digest(receipt))]))
    return SimpleNamespace(repo=str(repo),output=str(root/'new_backup'),external_assets=str(external),receipts=str(receipts))


class RecoveryTests(unittest.TestCase):
    def test_current_bundle_restores_refs_full_status_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root);repo=Path(args.repo);before=b.snapshot(repo)
            result=b.execute(args)
            self.assertEqual(result['status'],'verified');self.assertFalse(result['standalone_data_model_backup'])
            restored=Path(result['restore_verification']['path']);self.assertEqual(b.snapshot(restored),before)
            self.assertEqual((restored/'code/a.py').read_text(),'x=3\n');self.assertFalse((restored/'docs/delete.md').exists())
            self.assertFalse((restored/'assets/old.bin').exists());self.assertFalse((restored/'ignored').exists())
            payload=Path(args.output)/'payload';manifest=json.loads((payload/'recovery.json').read_text())
            self.assertIn('assets/old.bin',manifest['excluded_paths']);self.assertEqual(len(manifest['external_assets']),1)
            self.assertFalse(any(p.name=='model.bin' for p in Path(args.output).rglob('*')))
            self.assertEqual(b.snapshot(repo),before)
            with self.assertRaises(ValueError):b.execute(args)

    def test_bad_asset_hash_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root);spec=json.loads(Path(args.external_assets).read_text());spec[0]['sha256']='0'*64;Path(args.external_assets).write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'dependency SHA'):b.execute(args)
            self.assertTrue((Path(args.output)/'FAILED.json').is_file());self.assertFalse((Path(args.output)/'VERIFIED.json').exists())

    def test_git_environment_isolation_detached_and_cold_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root);repo=Path(args.repo)
            git(repo,'update-ref','--no-deref','HEAD',git(repo,'rev-parse','HEAD').decode().strip())
            before=b.snapshot(repo);index=b.digest(repo/'.git/index');foreign=root/'foreign_index';foreign.write_bytes(b'do not touch')
            with patch.dict(os.environ,{'GIT_DIR':str(repo/'.git'),'GIT_WORK_TREE':str(repo),'GIT_INDEX_FILE':str(foreign),'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'core.bare','GIT_CONFIG_VALUE_0':'true','GIT_OBJECT_DIRECTORY':'/missing/objects'}):result=b.execute(args)
            self.assertIsNone(before['symbolic_HEAD']);self.assertEqual(b.snapshot(repo),before);self.assertEqual(b.digest(repo/'.git/index'),index);self.assertEqual(foreign.read_bytes(),b'do not touch')
            extracted=Path(args.output)/'verification/extracted';recovery=json.loads((extracted/'recovery.json').read_text())
            repo.rename(root/'unavailable_source');work=root/'cold';work.mkdir()
            cold=b.restore_verify(None,extracted,work,before,recovery['selected_paths'],recovery['working_tree_files'])
            self.assertEqual(b.snapshot(Path(cold['path'])),before)

    def test_promisor_and_historical_symlink_refused(self):
        for kind in ('promisor','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();args=fixture(root);repo=Path(args.repo)
                if kind=='promisor':git(repo,'config','remote.origin.promisor','true')
                else:
                    path=repo/'code/link.py';path.symlink_to('a.py');git(repo,'add','code/link.py');path.unlink();path.write_text('regular now\n')
                with self.assertRaises(ValueError):b.execute(args)

    def test_source_and_external_drift_fail_closed(self):
        for kind in ('source','asset'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();args=fixture(root);original=b.restore_verify
                def changed(*a,**kw):
                    result=original(*a,**kw)
                    target=Path(args.repo)/'code/a.py' if kind=='source' else root/'model.bin'
                    with target.open('ab') as stream:stream.write(b'changed')
                    return result
                with patch.object(b,'restore_verify',side_effect=changed),self.assertRaises(ValueError):b.execute(args)
                self.assertTrue((Path(args.output)/'FAILED.json').exists());self.assertFalse((Path(args.output)/'VERIFIED.json').exists())

    def test_secret_path_changed_exclusion_and_symlinks_rejected(self):
        for kind in ('secret','excluded','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();args=fixture(root);repo=Path(args.repo)
                if kind=='secret':
                    marker='-----BEGIN '+ 'PRIVATE '+ 'KEY-----\n'
                    self.assertIsNotNone(b.SECRET.search(marker.encode()))
                    (repo/'code/key.txt').write_text(marker)
                elif kind=='excluded':(repo/'assets/old.bin').write_bytes(b'drift')
                else:(repo/'code/link.py').symlink_to(repo/'code/a.py')
                with self.assertRaises(ValueError):b.execute(args)
                self.assertTrue((Path(args.output)/'FAILED.json').exists())

    def test_safe_extract_rejects_traversal_links_duplicates(self):
        for kind in ('traversal','symlink','duplicate'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();archive=root/'bad.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    member=tarfile.TarInfo('../escape' if kind=='traversal' else 'file.txt');member.size=1;member.mode=0o644
                    if kind=='symlink':member.type=tarfile.SYMTYPE;member.linkname='/outside';member.size=0
                    tar.addfile(member,io.BytesIO(b'x'))
                    if kind=='duplicate':tar.addfile(member,io.BytesIO(b'x'))
                with self.assertRaises(ValueError):b.extract_safe(archive,root/'extract',[dict(path='file.txt',sha256='0'*64,bytes=1,mode=0o644)])
                self.assertFalse((root/'escape').exists())

    def test_atomic_receipt_and_nonignored_output_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root)
            args.output=str(Path(args.repo)/'accidental_output')
            with self.assertRaisesRegex(ValueError,'must be Git ignored'):b.execute(args)
            self.assertFalse(Path(args.output).exists())
            target=root/'VERIFIED.json';b.publish_verified(target,{'status':'verified'})
            with self.assertRaises(FileExistsError):b.publish_verified(target,{'status':'changed'})
            self.assertEqual(json.loads(target.read_text()),{'status':'verified'})

    def test_ignored_repository_output_restores_with_literal_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root);repo=Path(args.repo)
            before=b.snapshot(repo)
            args.output=str(repo/'ignored'/'backup [*] :literal\nnew')
            with patch.dict(os.environ,{'GIT_LITERAL_PATHSPECS':'1','GIT_GLOB_PATHSPECS':'1'}):
                result=b.execute(args)
            self.assertEqual(result['status'],'verified')
            self.assertTrue((Path(args.output)/'VERIFIED.json').is_file())
            self.assertEqual(b.snapshot(repo),before)
            restored=Path(result['restore_verification']['path'])
            self.assertEqual(b.snapshot(restored),before)

    def test_nonignored_literal_output_cannot_match_ignored_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root);repo=Path(args.repo)
            args.output=str(repo/'ignored*'/'backup')
            with patch.dict(os.environ,{'GIT_LITERAL_PATHSPECS':'1','GIT_GLOB_PATHSPECS':'1'}):
                with self.assertRaisesRegex(ValueError,'must be Git ignored'):b.execute(args)
            self.assertFalse(Path(args.output).exists())


if __name__=='__main__':unittest.main()
