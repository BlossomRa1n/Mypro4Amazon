import json,shutil,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from unittest.mock import patch
import numpy as np
import archive_cross_v3 as a
import archive_cross_v3_complete as c
ROOT=Path(__file__).resolve().parents[1]
FROZEN=None

class FakeRemote:
    def __init__(self):self.ready=True;self.corrupt=False;self.fetches=[]
    def inventory(self):
        if not self.ready:return {'ready':False,'reason':'global completion marker absent'}
        inventories={name:{str(p.relative_to(FROZEN/name)):{'size':p.stat().st_size,'sha256':a.digest(p)} for p in (FROZEN/name).rglob('*') if p.is_file()} for name in ('dev','protocol')}
        d=a.checked_json(FROZEN/'dev/dev_manifest.json');p=a.checked_json(FROZEN/'protocol/protocol_manifest.json')
        return dict(ready=True,**inventories,audit={'protocol_hash':p['manifest_hash'],'dev_manifest_hash':d['manifest_hash'],'test_accessed':False,'code_hashes':d['code_hashes'],'history_scope_sha256':p['history_scope_sha256'],'historical_closure_sha256':p.get('historical_closure_sha256')})
    def fetch(self,rel,destination,protocol=False):
        self.fetches.append((protocol,rel));shutil.copyfile(FROZEN/('protocol' if protocol else 'dev')/rel,destination)
        if self.corrupt:destination.write_bytes(b'bad')

class CompleteArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global FROZEN
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'code'))
        import run_cross_multiseed as runner
        from tests.test_cross_multiseed import _fixture
        cls.fixture_tmp=tempfile.TemporaryDirectory()
        FROZEN=Path(cls.fixture_tmp.name).resolve()
        base,history=_fixture(FROZEN)
        runner.prepare(SimpleNamespace(base_run=str(base),run_dir=str(FROZEN/'protocol'),historical_users=str(history),train_users=4,screen_users=2,confirm_users=2,cohort_seed=runner.COHORT_SEED,test_mode='all_fresh',smoke=True))
        args=SimpleNamespace(base_run=str(base),protocol_manifest=str(FROZEN/'protocol/protocol_manifest.json'),run_dir=str(FROZEN/'dev'),seeds=list(runner.SEEDS),variants=list(runner.VARIANTS),epochs=3,dim=8,token_dim=8,hist_len=5,batch_size=8,candidates=10,smoke=True)
        with patch.object(runner.torch.cuda,'is_available',return_value=False):runner.dev(args)

    @classmethod
    def tearDownClass(cls):cls.fixture_tmp.cleanup()

    def test_not_ready_and_dry_run_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();remote=FakeRemote();remote.ready=False
            self.assertEqual(c.run(remote,root/'dev',root/'protocol',root/'receipts',dry_run=True)['status'],'not_ready')
            self.assertEqual(list(root.iterdir()),[])
            remote.ready=True
            self.assertEqual(c.run(remote,root/'dev',root/'protocol',root/'receipts',dry_run=True)['status'],'ready');self.assertEqual(list(root.iterdir()),[])

    def test_frozen_success_hardlink_reuse_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();inc=root/'incremental';group=inc/'seed_42/raw';group.mkdir(parents=True)
            source=FROZEN/'dev/seed_42/raw/last.pth';shutil.copyfile(source,group/'last.pth')
            metadata={'size':source.stat().st_size,'sha256':a.digest(source)}
            (group/'ARCHIVE_RECEIPT.json').write_text(json.dumps({'status':'verified','inventory':{'last.pth':metadata}}))
            remote=FakeRemote();result=c.run(remote,root/'dev',root/'protocol',root/'receipts',inc)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['hardlinks'],['seed_42/raw/last.pth'])
            self.assertEqual((group/'last.pth').stat().st_ino,(root/'dev/seed_42/raw/last.pth').stat().st_ino)
            self.assertFalse((root/'dev/seed_42/raw/ARCHIVE_RECEIPT.json').exists())
            self.assertFalse(any(rel=='seed_42/raw/last.pth' for _,rel in remote.fetches))
            result=c.run(remote,root/'dev',root/'protocol',root/'receipts',inc);self.assertEqual(result['status'],'verified_existing')

    def test_hash_mismatch_preserves_partial_no_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();remote=FakeRemote();remote.corrupt=True
            with self.assertRaisesRegex(ValueError,'hash'):c.run(remote,root/'dev',root/'protocol',root/'receipts')
            self.assertFalse((root/'dev').exists());self.assertFalse((root/'protocol').exists());self.assertTrue(list(root.glob('dev.partial-*')))
            receipts=list((root/'receipts').glob('*.json'));self.assertEqual(a.checked_json(receipts[0])['status'],'failed')

    def test_existing_unsealed_refused_and_receipts_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'dev').mkdir();(root/'dev/user_file').write_text('keep')
            with self.assertRaises(FileExistsError):c.run(FakeRemote(),root/'dev',root/'protocol',root/'receipts')
            self.assertEqual((root/'dev/user_file').read_text(),'keep')
            with self.assertRaisesRegex(ValueError,'disjoint'):c.run(FakeRemote(),root/'dev',root/'protocol',root/'dev/receipts')

    def test_global_gate_and_unsealed_extra_rejected(self):
        inv=FakeRemote().inventory();inv['audit']['test_accessed']=True
        with self.assertRaisesRegex(ValueError,'audit'):c.verify_bundle(FROZEN/'dev',FROZEN/'protocol',inv)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();shutil.copytree(FROZEN/'dev',root/'dev');(root/'dev/unsealed').write_text('x')
            with self.assertRaisesRegex(ValueError,'inventory'):c.verify_bundle(root/'dev',FROZEN/'protocol',FakeRemote().inventory())

    def test_remote_drift_prevents_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();remote=FakeRemote();original=remote.inventory;calls=0
            def inventory():
                nonlocal calls
                calls+=1;value=original()
                if calls>1:value['audit']['protocol_hash']='changed'
                return value
            remote.inventory=inventory
            with self.assertRaisesRegex(ValueError,'drift'):c.run(remote,root/'dev',root/'protocol',root/'receipts')
            self.assertFalse((root/'dev').exists());self.assertFalse((root/'protocol').exists())

    def test_publish_interruption_keeps_partial_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();rename=c.os.rename;calls=0
            def fail_second(source,dest):
                nonlocal calls
                calls+=1
                if calls==2:raise OSError('simulated publish interruption')
                return rename(source,dest)
            with patch.object(c.os,'rename',side_effect=fail_second):
                with self.assertRaises(OSError):c.run(FakeRemote(),root/'dev',root/'protocol',root/'receipts')
            self.assertTrue((root/'protocol').exists());self.assertFalse((root/'dev').exists());self.assertTrue(list(root.glob('dev.partial-*')))
            with self.assertRaises(FileExistsError):c.run(FakeRemote(),root/'dev',root/'protocol',root/'receipts')

    def test_confirm_identity_tamper_even_resealed_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();shutil.copytree(FROZEN/'dev',root/'dev');inv=FakeRemote().inventory()
            rel='seed_42/raw/confirm_users.npz';file=root/'dev'/rel
            with np.load(file) as source:values={k:source[k].copy() for k in source.files}
            values['uid']+=1000;np.savez_compressed(file,**values)
            m=a.checked_json(root/'dev/dev_manifest.json');m['evidence_hashes'][rel]=a.digest(file);m.pop('manifest_hash');m['manifest_hash']=a.json_hash(m);(root/'dev/dev_manifest.json').write_text(json.dumps(m))
            inv['dev'][rel]={'size':file.stat().st_size,'sha256':a.digest(file)}
            manifest=root/'dev/dev_manifest.json';inv['dev']['dev_manifest.json']={'size':manifest.stat().st_size,'sha256':a.digest(manifest)};inv['audit']['dev_manifest_hash']=m['manifest_hash']
            with self.assertRaisesRegex(ValueError,'UID/position'):c.verify_bundle(root/'dev',FROZEN/'protocol',inv)

    def test_remote_gate_refuses_final_marker_without_opening_it(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();dev=root/'dev';protocol=root/'protocol';dev.mkdir();protocol.mkdir()
            (dev/'COMPLETED.json').write_text(json.dumps({'status':'complete','groups':9,'test_accessed':False}))
            marker=protocol/'TEST_STARTED.json';marker.write_text('must not read')
            prelude="from pathlib import Path\n_original=Path.open\ndef guarded(self,*args,**kwargs):\n if self.name=='TEST_STARTED.json': raise RuntimeError('READ_TRAP')\n return _original(self,*args,**kwargs)\nPath.open=guarded\n"
            proc=subprocess.run([sys.executable,'-c',prelude+c.REMOTE,str(dev),str(protocol),str(ROOT/'code')],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0);self.assertIn('refusing to open',proc.stderr);self.assertNotIn('READ_TRAP',proc.stderr)

if __name__=='__main__':unittest.main()
