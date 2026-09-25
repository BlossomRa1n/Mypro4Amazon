"""Synthetic train-only guards and real cache publication; no formal data."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import build_v2_train_pool as b
import run_cross_multiseed as r
import cross_pool_cache as cache
from tests.test_cross_multiseed import _fixture


class V2PoolTests(unittest.TestCase):
    def test_guards_and_unchanged_original(self):
        with tempfile.TemporaryDirectory() as temp:
            base, _ = _fixture(Path(temp))
            data = r.load_data(base)
            positions = np.asarray(data.train_positions)
            guarded = b.guard_training_data(data, positions)
            pos = int(positions[0]); uid = int(data.uid[pos])
            guarded.user_features(uid, pos)
            for call in (lambda: guarded.targets([(uid, pos)]), lambda: guarded.evaluation(), lambda: np.asarray(guarded.iid), lambda: guarded.user_features(uid, int(data.val_positions[0])), lambda: guarded.iid[int(data.val_positions[0])]):
                with self.assertRaises((PermissionError, ValueError)):
                    call()
            self.assertIsInstance(data.iid, np.ndarray)

    def test_actual_pilot_cache_and_failed_attempt_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); base, _ = _fixture(root)
            data = r.load_data(base); positions = np.asarray(data.train_positions)
            protocol = root / 'protocol.json'; b.write(protocol, {})
            old = root / 'old.json'; b.write(old, dict(base_data_id=data.manifest['data_id'], base_assets={}, encoders_hash=data.manifest['encoders_hash'], code_hashes={}))
            legacy = root / 'legacy'; legacy.mkdir(); (legacy / 'fixture.py').write_text('x=1\n')
            manifest = dict(base_run=str(base), old_protocol=dict(path=str(old)), code_hashes={'fixture.py': b.sha(legacy / 'fixture.py')})
            args = SimpleNamespace(mode='pilot', output_dir=str(root/'pilot'), smoke=True, legacy_source_dir=str(legacy), protocol_manifest=str(protocol), protocol_doc=str(ROOT/'docs/AFTERNOON_V2_TRAIN_POOL_PROTOCOL_20260925.md'), pilot_receipt=None)
            with patch.object(b, 'legacy_inputs', return_value=(r, cache, manifest, data, {}, positions, {})):
                result = b.build(args)
                self.assertEqual(result['source']['rrf_weights'], [2., 1., .7, .05])
                self.assertFalse(result['target_access_performed'])
                pools, meta = cache.load_cache(root/'pilot/candidate_pilot.json')
                self.assertEqual(len(pools), len(positions))
                self.assertEqual(meta['source'], result['source'])
                with self.assertRaises(ValueError):
                    b.build(args)
                args.output_dir = str(root/'failed')
                with patch.object(r, 'RecallPoolBuilder', side_effect=RuntimeError('synthetic failure')):
                    with self.assertRaises(RuntimeError):
                        b.build(args)
                self.assertTrue((root/'failed/FAILED.json').is_file())
                self.assertFalse((root/'failed/COMPLETED.json').exists())

    def test_disk_and_symlink_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(b.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
                with self.assertRaises(ValueError):
                    b.disk_snapshot(root, 100, False)
            link = root/'link'; link.symlink_to(root/'missing')
            with self.assertRaises(ValueError):
                b.safe(link)


if __name__ == '__main__':
    unittest.main()
