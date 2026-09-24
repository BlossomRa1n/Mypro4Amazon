import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import run_cross_multiseed as r
from tests.test_cross_multiseed import _fixture

class RuntimeTests(unittest.TestCase):
    def test_cohort_can_use_historical_users_for_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base,hist=_fixture(root)
            obj=json.loads(hist.read_text());obj['raw_user_ids']=[f'u{i:03}' for i in range(12)];hist.write_text(json.dumps(obj))
            args=SimpleNamespace(base_run=str(base),run_dir=str(root/'p'),historical_users=str(hist),train_users=4,screen_users=2,confirm_users=2,test_users=2,cohort_seed=r.COHORT_SEED,smoke=True)
            m=r.prepare(args)
            self.assertTrue(set(x[2] for x in r._records(root/'p',m,'train')) <= set(obj['raw_user_ids']))
            with self.assertRaises(FileExistsError):r.prepare(args)
            (root/'p'/'ranker_train_positions.npy').write_bytes(b'bad')
            with self.assertRaises(ValueError):r.load_protocol(root/'p'/'protocol_manifest.json')

    def test_sampler_sources_and_seed_are_actual(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,_=_fixture(Path(tmp));data=r.load_data(base);view=copy.copy(data);view.train_positions=data.train_positions[:3];view.seed=42
            pools=[]
            for p in view.train_positions:
                uid=int(data.uid[p]); pools.append([x for x in data.active_items if x not in data.train_sets[uid]][:75])
            ds=r.TracedPrefixDataset(view,16,'mixed_rrf',pools)
            a=ds[0]['neg_item_id'].copy();self.assertEqual(ds.last_sources[0],dict(band11_25=2,band26_50=2,candidate_fallback=0,random=12))
            view.seed=43;b=ds[0]['neg_item_id'];self.assertFalse(np.array_equal(a,b))
            ds.candidate_pools=[p[:11] for p in pools];ds[0]
            self.assertEqual(ds.last_sources[0]['band11_25'],1)
            self.assertEqual(ds.last_sources[0]['candidate_fallback'],3)
            self.assertEqual(ds.last_sources[0]['random'],12)

    def test_cpu_nine_group_smoke_blocks_test_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,hist=_fixture(root)
            p=SimpleNamespace(base_run=str(base),run_dir=str(root/'p'),historical_users=str(hist),train_users=4,screen_users=2,confirm_users=2,test_users=2,cohort_seed=r.COHORT_SEED,smoke=True)
            m=r.prepare(p);test_uids={x[0] for x in r._records(root/'p',m,'test')}
            data=r.load_data(base);old_positions=data.train_positions.copy();original=data.targets
            def guarded(records):
                self.assertFalse({x[0] for x in records}&test_uids)
                return original(records)
            data.targets=guarded
            a=SimpleNamespace(base_run=str(base),protocol_manifest=str(root/'p'/'protocol_manifest.json'),run_dir=str(root/'dev'),seeds=list(r.SEEDS),variants=list(r.VARIANTS),epochs=3,dim=8,token_dim=8,hist_len=5,batch_size=8,candidates=10,smoke=True)
            with patch.object(r,'validate_base',return_value=data): result=r.dev(a)
            np.testing.assert_array_equal(data.train_positions,old_positions)
            self.assertFalse(result['test_accessed']);self.assertEqual(len(result['checkpoint_hashes']),9)
            for seed in r.SEEDS:
                traces=[]
                for variant in r.VARIANTS:
                    d=root/'dev'/f'seed_{seed}'/variant
                    traces.append(json.loads((d/'epoch_3_trace.json').read_text()))
                    self.assertEqual(sum(traces[-1]['batch_sizes']),traces[-1]['rows'])
                    expected=[len(x) for x in r._batch_chunks(np.arange(traces[-1]['rows']),8)]
                    self.assertEqual(traces[-1]['batch_sizes'],expected)
                    checkpoint=torch.load(d/'last.pth',weights_only=False)
                    self.assertEqual(checkpoint['epoch'],3)
                    self.assertEqual(checkpoint['manifest']['history_scope_sha256'],m['history_scope_sha256'])
                    self.assertGreater(int(checkpoint['model']['head.1.num_batches_tracked']),3)
                    self.assertTrue(all(torch.isfinite(x).all() for x in checkpoint['model'].values()))
                    hist=json.loads((d/'history.json').read_text())
                    self.assertEqual(hist[-1]['steps'],len(traces[-1]['batch_sizes']))
                    if variant=='zero_cross':self.assertEqual(hist[-1]['cross_token_norm'],0.)
                self.assertEqual(traces[0],traces[1]);self.assertEqual(traces[0],traces[2])
            with self.assertRaises(FileExistsError):r.dev(a)

if __name__=='__main__':unittest.main()

class DiskBudgetTests(unittest.TestCase):
    def test_stage_local_binary_pool_budget_and_safety_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';cache=root/'cache';run.mkdir();cache.mkdir()
            usage=SimpleNamespace(free=100*1024**3)
            with patch.object(r.shutil,'disk_usage',return_value=usage):
                receipt=r._disk_preflight(run,cache,1000,50,75,'dev',9,10)
            self.assertEqual(receipt['candidate_pool_estimate_bytes'],50*304+1024**2)
            self.assertEqual(receipt['atomic_peak_models'],10)
            self.assertEqual(receipt['lifetime_atomic_peak_models'],14)
            self.assertEqual(receipt['safety_reserve_per_filesystem_bytes'],1024**3)
            with patch.object(r.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(OSError,'insufficient dev disk'):
                    r._disk_preflight(run,cache,1000,50,75,'dev',9,10)
