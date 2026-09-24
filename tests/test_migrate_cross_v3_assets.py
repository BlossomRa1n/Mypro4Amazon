import io,json,sys,tarfile,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import migrate_cross_v3_assets as m

class MigrationTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'system';target=root/'data';source.mkdir();target.mkdir()
        torch.save({'model':{'weight':torch.ones(2)}},source/'model.pth')
        with tarfile.open(source/'history.tar','w') as archive:
            value=b'immutable history evidence';info=tarfile.TarInfo('proof.txt');info.size=len(value);archive.addfile(info,io.BytesIO(value))
        assets=[dict(source=str(source/name),target=str(target/name),bytes=(source/name).stat().st_size,sha256=m.sha(source/name)) for name in ('model.pth','history.tar')]
        args=SimpleNamespace(execution_dir=str(root/'receipts'),execute=False)
        return args,assets
    def run_fixture(self,args,assets):
        return m.run(args,assets=assets,gate=lambda a:{'accepted':True},storage=lambda a:({'test_capacity':True},Path(a[0]['target']).parent,Path(a[0]['source']).parent),remaining=lambda a,d,s:m.budget(100*1024**3,100*1024**3,a))

    def test_check_only_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);before={str(p.relative_to(root)) for p in root.rglob('*')}
            result=self.run_fixture(args,assets);self.assertEqual(result['status'],'check_passed')
            self.assertEqual(before,{str(p.relative_to(root)) for p in root.rglob('*')});self.assertFalse(Path(args.execution_dir).exists())

    def test_success_preserves_bytes_and_expected_links_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
            result=self.run_fixture(args,assets);self.assertEqual(result['status'],'complete')
            for asset in assets:
                source=Path(asset['source']);target=Path(asset['target']);self.assertTrue(source.is_symlink());self.assertEqual(source.readlink(),target);self.assertEqual(m.sha(source),asset['sha256']);self.assertEqual(m.sha(target),asset['sha256'])
            with self.assertRaises(FileExistsError):self.run_fixture(args,assets)

    def test_hash_or_symlink_failure_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
            assets[0]['sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'identity'):self.run_fixture(args,assets)
            self.assertFalse(Path(args.execution_dir).exists());self.assertFalse(Path(assets[0]['target']).exists())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);p=Path(assets[0]['source']);other=p.with_name('real.pth');p.rename(other);p.symlink_to(other)
            with self.assertRaisesRegex(ValueError,'symlink'):self.run_fixture(args,assets)

    def test_switch_failure_preserves_destination_source_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
            with patch.object(m.os,'replace',side_effect=OSError('simulated switch failure')):
                with self.assertRaises(OSError):self.run_fixture(args,assets)
            self.assertTrue(Path(assets[0]['source']).is_file());self.assertFalse(Path(assets[0]['source']).is_symlink());self.assertTrue(Path(assets[0]['target']).is_file())
            self.assertTrue((Path(args.execution_dir)/'FAILED.json').exists())
            with self.assertRaises(FileExistsError):self.run_fixture(args,assets)

    def test_gate_capacity_and_existing_target_fail_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
            def refuse(a):raise ValueError('no winner')
            with self.assertRaisesRegex(ValueError,'no winner'):m.run(args,assets=assets,gate=refuse)
            self.assertFalse(Path(args.execution_dir).exists())
            Path(assets[0]['target']).write_text('retain')
            with self.assertRaises(FileExistsError):self.run_fixture(args,assets)
            self.assertEqual(Path(assets[0]['target']).read_text(),'retain')
        with self.assertRaises(OSError):m.budget(0,0)

    def test_unreadable_checkpoint_and_tar_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p=root/'nan.pth';torch.save({'model':{'w':torch.tensor(float('nan'))}},p)
            with self.assertRaises(ValueError):m.verify_readable(p,'.pth')
            p=root/'bad.tar';p.write_bytes(b'not a tar')
            with self.assertRaises(tarfile.ReadError):m.verify_readable(p,'.tar')

    def test_target_appears_during_publish_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
            original=m.os.link
            def race(source,target):
                Path(target).write_bytes(b'preserve competing target')
                return original(source,target)
            with patch.object(m.os,'link',side_effect=race):
                with self.assertRaises(FileExistsError):self.run_fixture(args,assets)
            self.assertEqual(Path(assets[0]['target']).read_bytes(),b'preserve competing target')
            self.assertFalse(Path(assets[0]['source']).is_symlink())
            self.assertTrue((Path(args.execution_dir)/'FAILED.json').exists())

    def test_hardlink_or_source_mutation_before_switch_refused(self):
        for hardlink in (True,False):
            with self.subTest(hardlink=hardlink),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
                original=m.shutil.copyfileobj
                def mutate(inp,out,length):
                    original(inp,out,length)
                    source=Path(assets[0]['source'])
                    if hardlink:m.os.link(source,source.with_name('extra-hardlink'))
                    else:source.touch()
                with patch.object(m.shutil,'copyfileobj',side_effect=mutate):
                    with self.assertRaises(ValueError):self.run_fixture(args,assets)
                self.assertFalse(Path(assets[0]['source']).is_symlink());self.assertFalse(Path(assets[0]['target']).exists())

    def test_global_lock_blocks_alternate_receipt_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
            self.run_fixture(args,assets);args.execution_dir=str(root/'another-receipt')
            with self.assertRaisesRegex(FileExistsError,'global migration'):self.run_fixture(args,assets)
            self.assertFalse(Path(args.execution_dir).exists())

    def test_actual_final_capacity_failure_keeps_migration_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
            def capacity(remaining,data,system):
                return m.budget(100*1024**3,100*1024**3 if remaining else 0,remaining)
            with self.assertRaises(OSError):
                m.run(args,assets=assets,gate=lambda a:{},storage=lambda a:({},root/'data',root/'system'),remaining=capacity)
            self.assertTrue(all(Path(a['source']).is_symlink() for a in assets))
            self.assertTrue((Path(args.execution_dir)/'FAILED.json').exists());self.assertFalse((Path(args.execution_dir)/'COMPLETED.json').exists())

    def test_receipt_symlink_ancestor_is_refused_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,assets=self.fixture(root);args.execute=True
            (root/'alias').symlink_to(root/'system',target_is_directory=True);args.execution_dir=str(root/'alias'/'receipts')
            with self.assertRaisesRegex(ValueError,'symlink'):self.run_fixture(args,assets)
            self.assertFalse((root/'system'/'receipts').exists())

    def test_archive_gate_requires_complete_bound_original_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();dev=root/'dev';pro=root/'protocol';dev.mkdir();pro.mkdir()
            (dev/'dev_manifest.json').write_text('{}');(dev/'COMPLETED.json').write_text('{}');(pro/'protocol_manifest.json').write_text('{}')
            for seed in (42,43,44):
                for variant in ('raw','normalized_gated','zero_cross'):
                    p=dev/f'seed_{seed}/{variant}/COMPLETED.json';p.parent.mkdir(parents=True);p.write_text('{}')
            manifest={'manifest_hash':'devhash','protocol_manifest':str(pro/'protocol_manifest.json'),'evidence_hashes':{}}
            protocol={'manifest_hash':'prohash','history_scope_sha256':'scope','historical_closure_sha256':'closure'};code={'runner':'codehash'}
            def inventory(tree):return {str(p.relative_to(tree)):{'sha256':m.sha(p),'size':p.stat().st_size} for p in tree.rglob('*') if p.is_file()}
            receipt={'status':'complete','remote_writes':False,'run_id':'local_verified','inventory':{'ready':True,'dev':inventory(dev),'protocol':inventory(pro),'audit':{'test_accessed':False,'protocol_hash':'prohash','dev_manifest_hash':'devhash','code_hashes':code,'history_scope_sha256':'scope','historical_closure_sha256':'closure'}}}
            path=root/'COMPLETE.json';path.write_text(json.dumps(receipt));args=SimpleNamespace(complete_archive_receipt=str(path),complete_archive_sha256=m.sha(path),dev_dir=str(dev))
            self.assertEqual(m.archive_gate(args,manifest,protocol,code)['sha256'],m.sha(path))
            receipt['status']='started';path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.archive_gate(args,manifest,protocol,code)
            args.complete_archive_sha256=m.sha(path)
            with self.assertRaisesRegex(ValueError,'complete local archive'):m.archive_gate(args,manifest,protocol,code)
            receipt['status']='complete';receipt['inventory']['audit']['dev_manifest_hash']='wrong';path.write_text(json.dumps(receipt));args.complete_archive_sha256=m.sha(path)
            with self.assertRaisesRegex(ValueError,'audit identity'):m.archive_gate(args,manifest,protocol,code)

if __name__=='__main__':unittest.main()
