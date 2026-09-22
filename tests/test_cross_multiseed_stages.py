"""Fail-closed guards for immutable selection and one-shot evaluation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import run_cross_multiseed as runner


class StageGuardTests(unittest.TestCase):
    def arrays(self):
        return dict(uid=np.array([2, 3]), position=np.array([4, 8]),
                    hit5=np.array([0, 1]), pool_hit=np.array([1, 1]), ndcg5=np.array([0., 1.]))

    def test_rejects_duplicate_nan_and_wrong_length(self):
        for key, value in [('uid', np.array([2, 2])), ('ndcg5', np.array([np.nan, 1.])),
                           ('hit5', np.array([1])), ('position', np.array([[4, 8]]))]:
            arrays = self.arrays(); arrays[key] = value
            with self.assertRaises(ValueError):
                runner._validate_arrays(arrays)

    def test_rejects_wrong_cohort_and_impossible_hit(self):
        arrays = self.arrays()
        with self.assertRaises(ValueError):
            runner._validate_arrays(arrays, [(3, 8, 'b'), (2, 4, 'a')])
        arrays['pool_hit'] = np.array([1, 0])
        with self.assertRaises(ValueError):
            runner._validate_arrays(arrays)

    def test_exclusive_marker_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'TEST_STARTED.json'
            runner._exclusive_json(p, {'state': 'started'})
            with self.assertRaises(FileExistsError):
                runner._exclusive_json(p, {'state': 'again'})
            self.assertEqual(json.loads(p.read_text()), {'state': 'started'})

    def test_tampered_and_cross_protocol_selection_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'selection.json'
            selection = {'protocol_manifest_hash': 'other'}
            selection['selection_hash'] = runner.sha256_json(selection)
            p.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'different protocol'):
                runner._verified_selection(p, {'manifest_hash': 'expected'})
            selection['protocol_manifest_hash'] = 'expected'
            p.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'hash'):
                runner._verified_selection(p, {'manifest_hash': 'expected'})

    def test_smoke_and_no_winner_plans_cannot_authorize_final(self):
        for smoke, status in [(True, 'accepted'), (False, 'no_winner')]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); selection_path = root / 'selection.json'
                selection_path.write_text('{}')
                protocol = dict(manifest_hash='p', smoke=smoke, base_run=tmp, base_data_id='d',
                                base_assets={}, cohort_hashes={'test': 't'})
                selection = dict(status=status, winner='zero_cross' if status == 'accepted' else None)
                args = SimpleNamespace(selection=str(selection_path), protocol_manifest=str(root/'protocol.json'), output=str(root/'plan.json'))
                with patch.object(runner, 'load_protocol', return_value=protocol), patch.object(runner, '_verified_selection', return_value=selection), patch.object(runner, '_verify_base'):
                    plan = runner.lock(args)
                self.assertFalse(plan['final_evaluation_allowed'])
                self.assertEqual(plan['models'], ['raw'])
                with patch.object(runner, '_load_plan', return_value=plan):
                    with self.assertRaises(ValueError):
                        runner.final_test(SimpleNamespace(final_plan=str(root/'plan.json')))

    def test_final_requires_training_receipt_and_protocol_global_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'final'; run.mkdir()
            protocol = {'counts': {'test': 100000}, 'smoke': False}
            plan = dict(final_evaluation_allowed=True, base_run=tmp, protocol_manifest=str(root/'protocol.json'),
                        plan_hash='locked', models=['raw', 'zero_cross'])
            args = SimpleNamespace(final_plan=str(root/'plan.json'), base_run=tmp, run_dir=str(run),
                                   dim=256, token_dim=256, hist_len=50, candidates=75)
            with patch.object(runner, '_load_plan', return_value=plan), patch.object(runner, 'load_protocol', return_value=protocol):
                with self.assertRaises(FileNotFoundError):
                    runner.final_test(args)
                manifest = dict(final_plan_hash='locked', models=plan['models'], test_future_labels_read=False,
                                test_history_coverage_users=100000, evidence_hashes={'receipt.txt': 'fake'})
                manifest['manifest_hash'] = runner.sha256_json(manifest)
                (run/'final_manifest.json').write_text(json.dumps(manifest))
                (run/'FINAL_TRAIN_COMPLETED.json').write_text(json.dumps({'manifest_hash': 'wrong'}))
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    runner.final_test(args)
                self.assertFalse((root/'TEST_STARTED.json').exists())

    def test_second_run_directory_cannot_bypass_global_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(final_evaluation_allowed=True, base_run=tmp, protocol_manifest=str(root/'protocol.json'),
                        plan_hash='locked', models=['raw', 'zero_cross'])
            protocol = {'counts': {'test': 100000}, 'smoke': False}
            (root/'TEST_STARTED.json').write_text('{"status":"started"}')
            for name in ('final', 'different-final'):
                run = root/name; run.mkdir(); (run/'receipt').write_text('ok')
                manifest = dict(final_plan_hash='locked', models=plan['models'], test_future_labels_read=False,
                                test_history_coverage_users=100000,
                                evidence_hashes={'receipt': runner.sha256_file(run/'receipt')})
                manifest['manifest_hash'] = runner.sha256_json(manifest)
                (run/'final_manifest.json').write_text(json.dumps(manifest))
                (run/'FINAL_TRAIN_COMPLETED.json').write_text(json.dumps({'manifest_hash': manifest['manifest_hash']}))
                args = SimpleNamespace(final_plan=str(root/'plan.json'), base_run=tmp, run_dir=str(run),
                                       dim=256, token_dim=256, hist_len=50, candidates=75)
                with patch.object(runner, '_load_plan', return_value=plan), patch.object(runner, 'load_protocol', return_value=protocol), patch.object(runner, 'evaluate') as evaluate:
                    with self.assertRaisesRegex(RuntimeError, 'second test'):
                        runner.final_test(args)
                    evaluate.assert_not_called()

    def test_fast_checkpoint_cannot_pass_final_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run=root/'final'; run.mkdir()
            plan = dict(final_evaluation_allowed=True, base_run=tmp, protocol_manifest=str(root/'protocol.json'),
                        plan_hash='locked', test_cohort_hash='test', models=['raw', 'zero_cross'])
            hashes = {}
            for variant in plan['models']:
                (run/variant).mkdir()
                runner.torch.save({'manifest': {'final': False}}, run/variant/'last.pth')
                hashes[variant] = runner.sha256_file(run/variant/'last.pth')
            manifest = dict(final_plan_hash='locked', models=plan['models'], test_future_labels_read=False,
                            test_history_coverage_users=100000, checkpoint_hashes=hashes,
                            evidence_hashes={f'{v}/last.pth': h for v, h in hashes.items()})
            manifest['manifest_hash'] = runner.sha256_json(manifest)
            (run/'final_manifest.json').write_text(json.dumps(manifest))
            (run/'FINAL_TRAIN_COMPLETED.json').write_text(json.dumps({'manifest_hash': manifest['manifest_hash']}))
            args = SimpleNamespace(final_plan=str(root/'plan.json'), base_run=tmp, run_dir=str(run),
                                   dim=256, token_dim=256, hist_len=50, candidates=75)
            with patch.object(runner, '_load_plan', return_value=plan), patch.object(runner, 'load_protocol', return_value={'counts': {'test':100000}, 'smoke':False}), patch.object(runner, 'evaluate') as evaluate:
                with self.assertRaisesRegex(ValueError, 'full-training'):
                    runner.final_test(args)
                evaluate.assert_not_called()
            self.assertFalse((root/'TEST_STARTED.json').exists())

    def test_final_train_and_test_real_artifact_integration(self):
        # Only the authorization boundary is mocked; production smoke cannot lock.
        from test_cross_multiseed import _fixture
        runner.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base, history=_fixture(root); protocol_dir=root/'protocol'
            protocol = runner.prepare(SimpleNamespace(base_run=str(base), run_dir=str(protocol_dir),
                historical_users=str(history), train_users=4, screen_users=2, confirm_users=2,
                test_users=2, cohort_seed=runner.COHORT_SEED, smoke=True))
            plan = dict(final_evaluation_allowed=True, base_run=str(base), protocol_manifest=str(protocol_dir/'protocol_manifest.json'),
                        plan_hash='fixture-only', selection_hash='fixture-only', models=['raw','zero_cross'],
                        winner='zero_cross', test_cohort_hash=protocol['cohort_hashes']['test'])
            args=SimpleNamespace(final_plan=str(root/'plan.json'),base_run=str(base),run_dir=str(root/'final'),
                dim=8,token_dim=8,hist_len=5,batch_size=8,epochs=3,candidates=75)
            with patch.object(runner, '_load_plan', return_value=plan):
                manifest=runner.final_train(args)
                self.assertEqual(manifest['test_history_coverage_users'], 2)
                result=runner.final_test(args)
                self.assertEqual(result['paired']['users'], 2)
                marker=protocol_dir/'TEST_STARTED.json'
                self.assertEqual(json.loads(marker.read_text())['status'], 'complete')
                with self.assertRaisesRegex(RuntimeError, 'second test'):
                    runner.final_test(args)
                import shutil
                shutil.copytree(root/'final', root/'copied-final')
                args.run_dir=str(root/'copied-final')
                with patch.object(runner, 'evaluate') as evaluate:
                    with self.assertRaisesRegex(RuntimeError, 'second test'):
                        runner.final_test(args)
                    evaluate.assert_not_called()

    def test_registered_selection_rules_and_user_level_seed_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev=Path(tmp)
            (dev/'dev_manifest.json').write_text('{}')
            manifest={'seeds':list(runner.SEEDS), 'variants':list(runner.VARIANTS), 'epochs':3, 'smoke':False, 'test_accessed':False}
            protocol={'manifest_hash':'fixture'}
            n=8
            baseline=np.array([0,0,0,0,1,1,1,1], dtype=np.int8)
            improved=np.ones(n, dtype=np.int8)
            for scenario in ('positive', 'reversed_seed', 'ndcg_regression'):
                for seed in runner.SEEDS:
                    for variant in runner.VARIANTS:
                        directory=dev/f'seed_{seed}'/variant; directory.mkdir(parents=True,exist_ok=True)
                        hits=baseline.copy() if variant != 'normalized_gated' else improved.copy()
                        ndcg=np.full(n, .5 if variant != 'normalized_gated' else .6)
                        if variant == 'normalized_gated' and scenario == 'reversed_seed' and seed == 44:
                            hits=np.zeros(n,dtype=np.int8)
                        if variant == 'normalized_gated' and scenario == 'ndcg_regression':
                            ndcg=np.full(n,.4)
                        for label in ('screen_epoch3','confirm'):
                            np.savez(directory/f'{label}_users.npz',uid=np.arange(n),position=np.arange(n)*10,
                                     hit5=hits,pool_hit=np.ones(n,dtype=np.int8),ndcg5=ndcg)
                with patch.object(runner, '_validate_dev_evidence', return_value=(manifest,protocol)):
                    result=runner.compare(SimpleNamespace(dev_run=str(dev),output=str(dev/'unused.json'),return_only=True))
                challenge=result['challenges']['normalized_gated']
                self.assertEqual(challenge['confirm_pooled_users'],n)
                if scenario == 'positive':
                    self.assertTrue(challenge['accepted'])
                    self.assertEqual(result['winner'],'normalized_gated')
                    self.assertGreater(challenge['confirm_hr5_ci97_5'][0],0)
                else:
                    self.assertFalse(challenge['accepted'])
                    self.assertEqual(result['status'],'no_winner')
                    self.assertIsNone(result['winner'])

    def test_forged_selection_decision_fails_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'selection.json'
            value = dict(protocol_manifest_hash='p', dev_run=tmp, status='accepted')
            value['selection_hash'] = runner.sha256_json(value)
            p.write_text(json.dumps(value))
            with patch.object(runner, 'compare', return_value={'status': 'no_winner'}):
                with self.assertRaisesRegex(ValueError, 'decision'):
                    runner._verified_selection(p, {'manifest_hash': 'p'})


if __name__ == '__main__':
    unittest.main()
