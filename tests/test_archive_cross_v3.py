import hashlib,json,shutil,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import archive_cross_v3 as a

class LocalRemote:
    def __init__(self,root):self.root=root;self.corrupt=False
    def inventory(self,include_protocol=False):
        d=self.root/'seed_42/raw'
        return {'groups':{'seed_42/raw':{n:{'size':(d/n).stat().st_size,'sha256':a.digest(d/n)} for n in a.GROUP_FILES}}}
    def fetch(self,relative,destination,protocol=False):
        shutil.copyfile(self.root/relative,destination)
        if self.corrupt:Path(destination).write_bytes(b'corrupt')

def fixture(root):
    d=root/'seed_42/raw';d.mkdir(parents=True)
    traces={str(i):{'rows':2,'batch_sizes':[2]} for i in (1,2,3)}
    m={'epochs':3,'seed':42,'variant':'raw','trace_files':traces}
    history=[{'epoch':i,'loss':.5} for i in (1,2,3)]
    torch.save({'model':{'weight':torch.ones(2)},'manifest':m,'history':history,'epoch':3},d/'last.pth')
    h=a.digest(d/'last.pth');(d/'manifest.json').write_text(json.dumps(dict(m,checkpoint_sha256=h)))
    (d/'COMPLETED.json').write_text(json.dumps({'status':'complete','epoch':3,'checkpoint_sha256':h}))
    (d/'history.json').write_text(json.dumps(history))
    for i in (1,2,3):
        (d/f'epoch_{i}_trace.json').write_text(json.dumps(traces[str(i)]))
        np.savez(d/f'screen_epoch{i}_users.npz',uid=np.array([2,3]),position=np.array([5,8]),hit5=np.array([0,1],dtype='i1'),pool_hit=np.ones(2,dtype='i1'),ndcg5=np.array([0.,1.],dtype='f4'))
    return d

class ArchiveTests(unittest.TestCase):
    def test_atomic_publish_and_idempotent_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();d=fixture(root/'remote');remote=LocalRemote(root/'remote');local=root/'local'
            real=a.os.rename;observed=[]
            def checked(src,dst):
                self.assertTrue((src/'ARCHIVE_RECEIPT.json').exists());self.assertFalse(dst.exists());observed.append(dst);return real(src,dst)
            with patch.object(a.os,'rename',side_effect=checked):first=a.run_once(remote,local)
            self.assertEqual(first['groups'][0]['status'],'archived');self.assertEqual(len(observed),1)
            checkpoint=local/'seed_42/raw/last.pth';before=checkpoint.stat().st_mtime_ns
            second=a.run_once(remote,local);self.assertEqual(second['groups'][0]['status'],'verified_existing');self.assertEqual(before,checkpoint.stat().st_mtime_ns)
            self.assertEqual(len(list((local/'receipts').glob('*.json'))),2)

    def test_hash_failure_preserves_partial_no_published_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root/'remote');remote=LocalRemote(root/'remote');remote.corrupt=True;local=root/'local'
            with self.assertRaisesRegex(ValueError,'hash mismatch'):a.run_once(remote,local)
            self.assertFalse((local/'seed_42/raw').exists());self.assertTrue(list((local/'.partial').rglob('last.pth')))
            receipts=list((local/'receipts').glob('*.json'));self.assertEqual(a.checked_json(receipts[0])['status'],'failed')

    def test_drift_never_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();d=fixture(root/'remote');remote=LocalRemote(root/'remote');local=root/'local';a.run_once(remote,local)
            before=a.digest(local/'seed_42/raw/history.json');(d/'history.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'drift'):a.run_once(remote,local)
            self.assertEqual(a.digest(local/'seed_42/raw/history.json'),before)

    def test_npz_and_checkpoint_readability_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,write in [('bad.npz',lambda p:np.savez(p,uid=np.array([2]))),('bad.pth',lambda p:torch.save({'model':{'w':torch.tensor(float('nan'))},'manifest':{},'history':[],'epoch':3},p))]:
                p=root/name;write(p)
                with self.assertRaises(ValueError):a.verify_payload(p,{'size':p.stat().st_size,'sha256':a.digest(p)})

    def test_remote_change_before_publish_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root/'remote');remote=LocalRemote(root/'remote');local=root/'local';original=remote.inventory;calls=0
            def drifting(*args,**kwargs):
                nonlocal calls
                calls+=1;value=original(*args,**kwargs)
                if calls>1:value['groups']['seed_42/raw']['last.pth']['sha256']='changed'
                return value
            remote.inventory=drifting
            with self.assertRaisesRegex(ValueError,'during transfer'):a.run_once(remote,local)
            self.assertFalse((local/'seed_42/raw').exists())

    def test_paths_and_lock_do_not_overwrite(self):
        for name in ('../x','/tmp/a','x/../a'):
            with self.assertRaises(ValueError):a.safe_relative(name)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root/'remote');local=root/'local';local.mkdir();(local/'.archive.lock').write_text('another process')
            with self.assertRaises(FileExistsError):a.run_once(LocalRemote(root/'remote'),local)
            self.assertEqual((local/'.archive.lock').read_text(),'another process')

    def test_symlink_ancestors_rejected_without_outside_write(self):
        for relative in ('seed_42','.partial','receipts'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();fixture(root/'remote');local=root/'local';local.mkdir();outside=root/'outside';outside.mkdir()
                (local/relative).symlink_to(outside,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'symlink'):a.run_once(LocalRemote(root/'remote'),local)
                self.assertEqual(list(outside.iterdir()),[])

if __name__=='__main__':unittest.main()
