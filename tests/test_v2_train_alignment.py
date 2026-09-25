"""Local synthetic integration and fail-closed tests; no server or final test."""
import copy
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch._dynamo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import run_v2_train_alignment as v
import run_cross_multiseed as r
import run_training_diagnostics as d
import cross_pool_cache as cache
from tests.test_cross_multiseed import _fixture


class AlignmentTests(unittest.TestCase):
    def test_clone_all_buffers_independent_and_reload(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3),torch.nn.Dropout(.2))
        model.train(); model(torch.randn(8,3))
        with d.preserve_training_state(model): selected=v.clone_state(model)
        before=v.state_hashes(selected)
        model(torch.randn(8,3))
        with torch.no_grad(): model[0].weight.add_(2)
        self.assertEqual(v.state_hashes(selected),before)
        self.assertIn('1.num_batches_tracked',selected)
        self.assertTrue(model.training)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'best.pth';r.atomic_torch_save({'model':selected},path)
            self.assertEqual(v.state_hashes(torch.load(path,weights_only=False)['model']),before)
        self.assertGreater(d.selection_key({'hr5':.5,'ndcg5':.3},1),d.selection_key({'hr5':.5,'ndcg5':.3},2))

    def test_pairing_mask_and_nonfinite_rejected(self):
        a={key:np.array([1]) for key in v.PAIR_KEYS};a.update(uid=np.array([1]),position=np.array([2]),score=np.array([.5]))
        v.assert_pair(a,copy.deepcopy(a),[(1,2,'u')])
        for key in v.PAIR_KEYS:
            bad=copy.deepcopy(a);bad[key][0]+=1
            with self.assertRaises(ValueError):v.assert_pair(a,bad,[(1,2,'u')])
        bad=copy.deepcopy(a);bad['score'][0]=np.nan
        with self.assertRaises(ValueError):v.assert_pair(bad,a,[(1,2,'u')])

    def test_remaining_disk_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            with patch.object(v.shutil,'disk_usage',return_value=SimpleNamespace(free=1000)):
                self.assertEqual(v.system_budget(root,3,100,50,True)['required_free_bytes'],450)
                with self.assertRaises(ValueError):v.system_budget(root,3,100,50,False)
            (root/'existing.pth').write_bytes(b'x'*100)
            (root/'arrays.npz').write_bytes(b'x'*10)
            cfg=SimpleNamespace(run_root=root,checkpoint_bound=100,array_allowance=50,smoke=True)
            self.assertEqual(v.stage_budget(cfg,2)['required_free_bytes'],340)

    def test_real_three_seed_one_epoch_new_pool_pipeline(self):
        torch.set_num_threads(4)
        keep=os.environ.get('V2_SMOKE_KEEP_DIR')
        if keep: Path(keep).mkdir(parents=True,exist_ok=False)
        with (contextlib.nullcontext(keep) if keep else tempfile.TemporaryDirectory()) as tmp:
            root=Path(tmp).resolve();base,_=_fixture(root);data=r.load_data(base)
            allrecords=[(int(u),int(p),str(data.users[u])) for u,p in zip(data.val_users,data.val_positions)]
            records=dict(train=allrecords[:4],screen=allrecords[4:6],confirm=allrecords[6:8])
            uids={x[0] for x in records['train']};positions=np.asarray([p for p in data.train_positions if int(data.uid[p]) in uids])
            def pool(uid):return [int(x) for x in data.active_items if x not in data.train_sets[uid]][:75]
            # Deliberately different ranked train pool exercises natural RNG drift.
            pools=dict(train=[list(reversed(pool(int(data.uid[p])))) for p in positions])
            for split in ('screen','confirm'):
                pools[split]=[[x for x in pool(row[0]) if x not in data.targets([row[:2]])[0]] for row in records[split]]
            original=data.targets;accesses=[];allowed={x[:2] for group in records.values() for x in group}
            def spy(rows):
                rows=list(rows);self.assertTrue(set(rows)<=allowed);accesses.extend(rows);return original(rows)
            data.targets=spy
            old=root/'old.json';v.write(old,dict(base_data_id=data.manifest['data_id'],base_assets={},encoders_hash=data.manifest['encoders_hash'],code_hashes={}))
            legacy=root/'legacy';legacy.mkdir();(legacy/'fixture.py').write_text('x=1\n')
            manifest=dict(smoke=True,base_run=str(base),manifest_hash='fixture',old_protocol=dict(path=str(old)),code_hashes={'fixture.py':v.sha(legacy/'fixture.py')},candidate_caches={},settings=dict(dim=8,token_dim=8,hist_len=5,batch_size=4,candidates=75))
            protocol=root/'protocol.json';v.write(protocol,manifest)
            adapter=ModuleType('diagnostic_protocol');adapter.load_inputs=lambda path:(manifest,data,records,positions,pools)
            with patch.dict(sys.modules,{'diagnostic_protocol':adapter}):oldresult=d.run(SimpleNamespace(protocol_manifest=str(protocol),run_dir=str(root/'raw'),smoke=True))
            common=dict(smoke=True,legacy_source_dir=str(legacy),protocol_manifest=str(protocol),protocol_doc=str(ROOT/'docs/AFTERNOON_V2_TRAIN_POOL_PROTOCOL_20260925.md'))
            legacy_result=(r,cache,manifest,data,records,positions,pools)
            with patch.object(v.b,'legacy_inputs',return_value=legacy_result):
                v.b.build(SimpleNamespace(**common,mode='pilot',output_dir=str(root/'pilot'),pilot_receipt=None))
                v.b.build(SimpleNamespace(**common,mode='full',output_dir=str(root/'pool'),pilot_receipt=str(root/'pilot/COMPLETED.json')))
                args=SimpleNamespace(**common,raw_run=str(root/'raw'),pool_receipt=str(root/'pool/COMPLETED.json'),run_dir=str(root/'new'),archive_receipt=None,backup_receipt=None)
                result=v.run(args)
                self.assertFalse(result['test_future_labels_read']);self.assertTrue(accesses)
                self.assertEqual(len(list((root/'new').rglob('*.pth'))),3)
                self.assertEqual(len(list((root/'new').rglob('screen_step*_users.npz'))),9)
                self.assertEqual(len(list((root/'new').rglob('confirm_best_users.npz'))),3)
                self.assertEqual(v.read(root/'new/panels.json'),v.read(root/'raw/panels.json'))
                for seed in r.SEEDS:
                    output=root/f'new/seed_{seed}/{v.VARIANT}'
                    trace=v.read(output/'epoch_1_trace.json');oldtrace=v.read(root/f'raw/seed_{seed}/raw/epoch_1_trace.json')
                    self.assertEqual(trace['order_sha256'],oldtrace['order_sha256'])
                    self.assertNotEqual(trace['sha256'],oldtrace['sha256'])
                    payload=torch.load(output/'best.pth',weights_only=False)
                    self.assertEqual(v.state_hashes(payload['model']),payload['manifest']['selected_state_sha256'])
                    history=v.read(output/'history.json')
                    self.assertEqual(payload['manifest']['step'],max(history,key=lambda x:d.selection_key(x['screen'],x['step']))['step'])
                with self.assertRaises(ValueError):v.run(args)
                config=v.read(root/'new/run_manifest.json')['config']
                for key in ('raw_run','run_root','data_root'):config[key]=Path(config[key])
                cfg=SimpleNamespace(**config);factors,_=r._load_factors(base,data,8,True)
                template,state=r._template(data,cfg,r.INIT_SEEDS[42],factors);del template
                # Identical old pool proves clone-at-best reproduces old per-point save.
                parity=v.train_seed(data,cfg,root/'parity',42,state,positions,pools,records,v.read(root/'raw/panels.json'),manifest,r,d,[])
                oldbest=torch.load(root/'raw/seed_42/raw/best.pth',weights_only=False)
                paritybest=torch.load(root/'parity/best.pth',weights_only=False)
                self.assertEqual(v.state_hashes(oldbest['model']),v.state_hashes(paritybest['model']))
                self.assertEqual(parity['traces']['1']['sha256'],v.read(root/'raw/seed_42/raw/epoch_1_trace.json')['sha256'])
                for newarray in (root/'parity').glob('screen_step*_users.npz'):
                    oldarray=root/'raw/seed_42/raw'/newarray.name
                    a,c=v.arrays(newarray),v.arrays(oldarray)
                    for key in a:np.testing.assert_array_equal(a[key],c[key])
                state['user_embedding.weight'][2,0]+=1
                with self.assertRaisesRegex(ValueError,'initial tensors differ'):
                    v.train_seed(data,cfg,root/'rejected_initial',42,state,positions,pools,records,v.read(root/'raw/panels.json'),manifest,r,d,[])
                template,state=r._template(data,cfg,r.INIT_SEEDS[42],factors);del template
                tracepath=root/'raw/seed_42/raw/epoch_1_trace.json';original_trace=tracepath.read_text();bad=json.loads(original_trace);bad['rows']-=1;tracepath.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'row consumption differs'):
                    v.train_seed(data,cfg,root/'rejected_trace',42,state,positions,pools,records,v.read(root/'raw/panels.json'),manifest,r,d,[])
                self.assertFalse((root/'rejected_trace/best.pth').exists());tracepath.write_text(original_trace)
                # Changed old source fails its sealed evidence binding before training.
                target=root/'raw/seed_42/raw/initial_state_audit.json';target.write_text(target.read_text()+'\n')
                args.run_dir=str(root/'rejected')
                with self.assertRaisesRegex(ValueError,'raw control evidence drift'):v.run(args)
                self.assertFalse((root/'rejected/COMPLETED.json').exists())
                target.write_text(target.read_text()[:-1])


if __name__=='__main__':unittest.main()
