"""CPU-only equivalence and fail-closed tests for the standalone full builder."""
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/build_full_prefix_pools.py'
sys.path.insert(0, str(ROOT / 'tools'))
import build_full_prefix_pools as b
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
import run_cross_multiseed as r
from cross_pool_cache import PositionRecords, build_cache, rows_hash
from tests.test_cross_multiseed import _fixture


class BrokenBuilder:
    def __call__(self, records): raise RuntimeError('intentional interrupted worker')


class FullPrefixTests(unittest.TestCase):
    def test_formal_pins_reject_resealed_substitutions(self):
        real_path = ROOT / 'server_snapshot/training_diagnostics_20260924/remote_root/protocol_v2/protocol_manifest.json'
        manifest = b.read_manifest(real_path)
        b.validate_formal_pins(manifest, False)
        changed = dict(manifest, base_run='/tmp/substituted-base')
        changed.pop('manifest_hash')
        changed['manifest_hash'] = r.sha256_json(changed)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'resealed.json'; path.write_text(json.dumps(changed))
            resealed = b.read_manifest(path)
            with self.assertRaisesRegex(ValueError, 'unregistered formal diagnostic'):
                b.validate_formal_pins(resealed, False)
            args = SimpleNamespace(workers=4, pilot_rows=17, legacy_source_dir=str(ROOT/'code'),
                                   protocol_manifest=str(path), output_dir=str(Path(td)/'out'), smoke=False)
            with self.assertRaisesRegex(ValueError, 'unregistered formal diagnostic'): b.run(args)
            self.assertFalse((Path(td)/'out').exists())
        changed = dict(manifest, old_protocol=dict(manifest['old_protocol'], sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'original protocol'):
            b.validate_formal_pins(changed, False)
        b.validate_formal_pins(changed, True)

    def test_disk_preflight_rejects_low_space_before_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            required = 1476314424 + 1024**2 + 1024**3
            with patch.object(b.shutil, 'disk_usage', return_value=SimpleNamespace(free=required-1)):
                with self.assertRaisesRegex(OSError, 'insufficient output filesystem'):
                    b.disk_preflight(output, b.FORMAL_ROWS)
            receipt = json.loads((output/'DISK_PREFLIGHT.json').read_text())
            self.assertEqual(receipt['status'], 'insufficient_space')
            self.assertEqual(receipt['payload_bytes'], 1476314424)
            self.assertEqual(receipt['reserve_bytes'], 1024**3)
            self.assertEqual(receipt['required_free_bytes'], required)
            self.assertEqual(list(output.glob('*.npy*')), [])
            self.assertFalse((output/'COMPLETED.json').exists())
        with tempfile.TemporaryDirectory() as td:
            with patch.object(b.shutil, 'disk_usage', return_value=SimpleNamespace(free=required)):
                self.assertEqual(b.disk_preflight(Path(td), b.FORMAL_ROWS)['status'], 'passed')

    def formal_builder(self, root):
        base, _ = _fixture(root)
        data = r.load_data(base)
        generator = np.random.default_rng(942)
        neighbors = generator.integers(2, len(data.items), (len(data.items), 300), dtype=np.int32)
        scores = generator.random(neighbors.shape, dtype=np.float32)
        with (base / 'itemcf.pkl').open('wb') as f: pickle.dump((neighbors, scores), f)
        (base / 'v2_best.pth').write_bytes(b'never loaded: v2 weight is zero')
        positions = b.validate_positions(data)
        return data, r.RecallPoolBuilder(data, base, 75, b.WEIGHTS, False), positions

    def test_real_frozen_algorithm_parallel_exact_order_and_guard(self):
        with tempfile.TemporaryDirectory() as td:
            data, builder, positions = self.formal_builder(Path(td))
            records = PositionRecords(data, positions)
            expected = builder(records)
            b.guard_data(data)
            with b.OrderedBuilder(builder, 4) as parallel:
                actual = []
                for start in range(0, len(records), 13): actual.extend(parallel(records[start:start+13]))
            self.assertEqual(actual, expected)
            with self.assertRaises(PermissionError): data.targets([(2, 7)])
            future = int(data.train_ends[2])
            with self.assertRaises(PermissionError): _ = data.iid[future]
            with self.assertRaises(PermissionError): _ = np.asarray(data.iid)
            with self.assertRaises(PermissionError): data.evaluation('test')

    def test_worker_exception_leaves_no_finished_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'candidate.json'
            records = [(2, 1, 'u'), (2, 2, 'u')]
            with b.OrderedBuilder(BrokenBuilder(), 4) as parallel:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    build_cache(target, records, 75, {'records_hash': rows_hash(records)}, parallel)
            self.assertFalse(target.exists())
            self.assertTrue(target.with_suffix('.json.building').exists())
            with self.assertRaises(FileExistsError):
                build_cache(target, records, 75, {'records_hash': rows_hash(records)}, lambda x: [[]]*len(x))

    def test_row_order_coverage_workers_and_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data, _, positions = self.formal_builder(Path(td))
            data.train_positions = positions[::-1]
            with self.assertRaisesRegex(ValueError, 'reordering'): b.validate_positions(data)
            data.train_positions = positions[1:]
            with self.assertRaisesRegex(ValueError, 'coverage'): b.validate_positions(data)
            data.train_positions = positions
            data.train_mask = data.train_mask.copy(); data.train_mask[0] = not data.train_mask[0]
            with self.assertRaisesRegex(ValueError, 'mask drift'): b.validate_positions(data)
            for bad in (0, 5, -1, True, 1.5):
                with self.assertRaises(ValueError): b.validate_workers(bad)
            asset = Path(td) / 'asset'; asset.write_text('a'); digest = b.sha(asset); asset.write_text('b')
            with self.assertRaisesRegex(ValueError, 'drift'): b.checked(asset, digest)

    def test_swapped_worker_envelopes_rejected(self):
        class WrongPool:
            def map(self, fn, tasks, chunksize): return [b._work(t) for t in tasks][::-1]
        parallel = b.OrderedBuilder(lambda rows: [[2] for row in rows], 4)
        parallel.pool = WrongPool()
        b._CONTEXT = parallel.builder
        with self.assertRaisesRegex(ValueError, 'order mismatch'):
            parallel([(2, i, 'u') for i in range(9)])
        b._CONTEXT = None

    def test_cli_frozen_smoke_pilot_receipt_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); base, _ = _fixture(root); data = r.load_data(base)
            code = ROOT / 'server_snapshot/training_diagnostics_20260924/remote_root/src_v2/code'
            assets = {name: b.sha(base/name) for name in ('data.pkl', 'run_manifest.json')}
            def save_manifest(path, value):
                value['manifest_hash'] = r.sha256_json(value)
                path.write_text(json.dumps(value)); return value
            old_path = root/'old.json'
            old = save_manifest(old_path, dict(smoke=True, base_run=str(base), base_assets=assets,
                base_data_hash=assets['data.pkl'], base_data_id=data.manifest['data_id'],
                encoders_hash=data.manifest['encoders_hash'], code_hashes=b.code_hashes(code)))
            diagnostic = root/'diagnostic.json'
            save_manifest(diagnostic, dict(protocol='raw-deterioration-20260924-v1', smoke=True,
                base_run=str(base), old_protocol=dict(path=str(old_path), sha256=b.sha(old_path)),
                code_hashes=b.code_hashes(code), test_evaluation_allowed=False))
            output=root/'out'
            command=[sys.executable,str(TOOL),'--legacy-source-dir',str(code),'--protocol-manifest',str(diagnostic),
                     '--output-dir',str(output),'--workers','4','--smoke','--pilot-rows','17']
            result=subprocess.run(command,text=True,capture_output=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads((output/'COMPLETED.json').read_text())
            self.assertEqual(receipt['mode'],'pilot')
            self.assertEqual(receipt['selected_rows'],17)
            self.assertFalse(receipt['test_future_labels_read'])
            self.assertEqual(receipt['disk_preflight']['status'], 'passed')
            self.assertEqual(receipt['disk_preflight']['payload_bytes'], 17*312)
            self.assertTrue((output/'candidate_pilot.json').exists())
            self.assertFalse((output/'candidate_final_train.json').exists())
            self.assertEqual(receipt['source']['code_hashes'],old['code_hashes'])
            result=subprocess.run(command,text=True,capture_output=True,timeout=90)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('refuse to overwrite',result.stderr)
            payload=diagnostic.read_text(); diagnostic.write_text(payload.replace('raw-deterioration','changed'))
            result=subprocess.run(command[:-2],text=True,capture_output=True,timeout=90)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('manifest hash mismatch',result.stderr)


if __name__ == '__main__': unittest.main()
