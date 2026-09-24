import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch._dynamo
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import diagnostic_metrics as m
import run_training_diagnostics as d
import run_cross_multiseed as r
from tests.test_cross_multiseed import _fixture


class DiagnosticsTests(unittest.TestCase):
    def test_auc_ties_and_invalid_masks(self):
        self.assertEqual(m.pair_auc([1., 1., 0.], [1, 0, 0]), (.75, True))
        self.assertEqual(m.pair_auc([1., 2.], [0, 0]), (0., False))
        self.assertEqual(m.pair_auc([1.], [1]), (0., False))

    def test_rng_and_batchnorm_are_unchanged(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.5))
        model.train(); before = model[0].running_mean.clone()
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        with m.preserve_training_state(model):
            random.random(); np.random.rand(); torch.rand(4); model(torch.ones(4, 3))
        actual = (random.random(), np.random.rand(), torch.rand(4))
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        self.assertEqual(actual[0], random.random()); self.assertEqual(actual[1], np.random.rand()); self.assertTrue(torch.equal(actual[2], torch.rand(4)))
        self.assertTrue(model.training); self.assertTrue(torch.equal(before, model[0].running_mean))

    def test_selection_tiebreak_and_checkpoint_reload(self):
        self.assertGreater(d.selection_key(dict(hr5=.5, ndcg5=.3), 1), d.selection_key(dict(hr5=.5, ndcg5=.3), 2))
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 1); path = Path(tmp) / 'best.pth'
            digest = d.checked_save(model, {'step': 1}, path)
            self.assertEqual(digest, r.sha256_file(path)); self.assertFalse(path.with_suffix('.pth.tmp').exists())
            self.assertTrue(torch.equal(model.weight, torch.load(path, weights_only=False)['model']['weight']))

    def test_real_protocol_prepare_load_and_three_seed_smoke(self):
        import diagnostic_protocol as protocol
        from cross_pool_cache import PositionRecords, rows_hash
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, hist = _fixture(root); data = r.load_data(base)
            old_dir = root / 'old_protocol'
            old = r.prepare(SimpleNamespace(base_run=str(base), run_dir=str(old_dir), historical_users=str(hist), train_users=4, screen_users=2, confirm_users=2, test_users=2, cohort_seed=r.COHORT_SEED, smoke=True))
            records = {split: r._records(old_dir, old, split) for split in ('train', 'screen', 'confirm')}
            positions = np.load(old_dir / old['ranker_train_positions'])
            for split in records:
                rows = PositionRecords(data, positions) if split == 'train' else records[split]
                r._load_or_make_assets(data, old_dir, old, rows, split, 75)
            evidence = root / 'evidence'; (evidence / 'server_snapshot').mkdir(parents=True)
            def save(name, obj):
                relative = 'server_snapshot/' + name + '.json'; target = evidence / relative
                r.write_json(target, obj); return dict(path=relative, sha256=r.sha256_file(target))
            registry = save('registry', dict(unresolved_sources_within_reviewed_scope=[]))
            audit = save('audit', {})
            eligible, _ = r._eligible_records(data, set()); excluded = sorted({row[2] for row in records['train']})
            exclusions = save('excluded', dict(raw_user_ids=excluded, count=len(excluded), raw_user_ids_sha256=r.sha256_json(excluded)))
            available = sorted({row[2] for row in eligible} - set(excluded))
            chosen = sorted(np.random.default_rng(20260924).choice(np.asarray(available), 2, replace=False).tolist())
            by_raw = {row[2]: row for row in eligible}; test = [by_raw[raw] for raw in chosen]
            test_ref = save('locked_records', test)
            lock = dict(formal_evaluation_allowed=False, base_data_id=data.manifest['data_id'], source_registry=registry, independent_stageab_audit=audit, exclusions=exclusions, records=test_ref, count=2, cohort_records_hash=rows_hash(test), cohort_raw_hash=r.sha256_json(chosen), eligible_available_population=len(available), population_sorted_raw_ids_sha256=r.sha256_json(available), cohort_seed=20260924)
            lock_path = evidence / 'lock.json'; r.write_json(lock_path, lock)
            path = root / 'diagnostic.json'
            args = SimpleNamespace(output=str(path), old_protocol=str(old_dir/'protocol_manifest.json'), base_run=str(base), evidence_root=str(evidence), smoke=True, smoke_lock=str(lock_path))
            protocol.prepare(args)
            _, guarded, _, _, _ = protocol.load_inputs(path)
            with self.assertRaises(PermissionError): guarded.targets([test[0][:2]])
            result = d.run(SimpleNamespace(protocol_manifest=str(path), run_dir=str(root/'run'), smoke=True))
            self.assertFalse(result['test_future_labels_read']); self.assertEqual(len(result['seeds']), 3)
            self.assertIn('supports_early_stopping_followup', result['confirm'])

    def test_smoke_three_seeds_old_path_parity_and_test_spy(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, _ = _fixture(root); data = r.load_data(base)
            all_records = [(int(u), int(p), str(data.users[u])) for u, p in zip(data.val_users, data.val_positions)]
            records = dict(train=all_records[:4], screen=all_records[4:6], confirm=all_records[6:8])
            train_uids = {x[0] for x in records['train']}; positions = np.asarray([p for p in data.train_positions if int(data.uid[p]) in train_uids])
            def pool(uid): return [int(x) for x in data.active_items if x not in data.train_sets[uid]][:75]
            pools = dict(train=[pool(int(data.uid[p])) for p in positions], screen=[pool(x[0]) for x in records['screen']], confirm=[pool(x[0]) for x in records['confirm']])
            # Include real positives to exercise HR/NDCG and AUC paths.
            for split in ('screen', 'confirm'):
                for index, record in enumerate(records[split]): pools[split][index][-1] = min(data.targets([record[:2]])[0])
            manifest = dict(smoke=True, base_run=str(base), manifest_hash='fixture', settings=dict(dim=8, token_dim=8, hist_len=5, batch_size=4))
            original = data.targets; accesses = []
            allowed = {x[:2] for rows in records.values() for x in rows}
            def spy(rows):
                rows = list(rows); self.assertTrue(set(rows) <= allowed); accesses.extend(rows); return original(rows)
            data.targets = spy
            adapter = types.ModuleType('diagnostic_protocol'); adapter.load_inputs = lambda path: (manifest, data, records, positions, pools)
            args = SimpleNamespace(protocol_manifest=str(root/'protocol.json'), run_dir=str(root/'run'), smoke=True)
            with patch.dict(sys.modules, {'diagnostic_protocol': adapter}): result = d.run(args)
            self.assertFalse(result['test_future_labels_read']); self.assertEqual(len(result['seeds']), 3)
            self.assertTrue(accesses); self.assertEqual(len(list((root/'run').rglob('*.pth'))), 6)
            config = SimpleNamespace(dim=8, token_dim=8, hist_len=5, batch_size=4, epochs=3)
            factors, _ = r._load_factors(base, data, 8, True)
            template, state = r._template(data, config, r.INIT_SEEDS[42], factors); del template
            old = r._train_model(data, config, root/'old', 42, 'raw', state, factors, positions, pools['train'], records['screen'], pools['screen'], True)
            new = root/'run'/'seed_42'/'raw'
            self.assertEqual(old['trace_files'], result['seeds']['42']['traces'])
            before = torch.load(root/'old'/'seed_42'/'raw'/'last.pth', weights_only=False)['model']
            after = torch.load(new/'last.pth', weights_only=False)['model']
            for key in before: self.assertTrue(torch.equal(before[key], after[key]), key)
            model = r.SemanticTokenDIN(**r._config(data, config, 'raw')); model.load_state_dict(after)
            old_metrics, old_arrays = r.evaluate(data, model, records['screen'], pools['screen'])
            metrics, arrays = m.evaluate_candidates(data, model, records['screen'], pools['screen'])
            for key in old_metrics: self.assertEqual(metrics[key], old_metrics[key])
            for key in old_arrays: np.testing.assert_array_equal(arrays[key], old_arrays[key])
            history = json.loads((new/'history.json').read_text()); self.assertEqual(sum(len(x['requested_points']) for x in history), 5)
            best = max(history, key=lambda x: d.selection_key(x['screen'], x['step']))
            self.assertEqual(result['seeds']['42']['best']['step'], best['step'])
            with patch.dict(sys.modules, {'diagnostic_protocol': adapter}), self.assertRaises(FileExistsError): d.run(args)
            d.guard_targets(data, records)
            with self.assertRaises(PermissionError): data.targets([all_records[-1][:2]])

if __name__ == '__main__': unittest.main()
