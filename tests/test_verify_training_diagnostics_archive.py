import json
import numpy as np
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import verify_training_diagnostics_archive as verifier
import tests.test_training_diagnostics as fixtures
import run_training_diagnostics as runner


class ArchiveVerificationTests(unittest.TestCase):
    def test_real_tiny_archive_and_hash_nan_missing_rejections(self):
        original_run = runner.run
        def run_and_verify(args):
            result = original_run(args)
            run = Path(args.run_dir); root = run.parent
            config = SimpleNamespace(run_dir=str(run), protocol_manifest=args.protocol_manifest, receipt=str(root/'verified.json'), allow_smoke=True)
            receipt = verifier.verify(config)
            self.assertEqual(receipt['status'], 'verified'); self.assertFalse(receipt['formal_archive_verified']); self.assertEqual(receipt['counts']['checkpoints'], 6)
            config.receipt = str(root/'rejected.json'); config.allow_smoke = False
            with self.assertRaisesRegex(ValueError, 'smoke archive'): verifier.verify(config)
            config.allow_smoke = True
            file = run/'panels.json'; saved = file.read_bytes(); file.write_bytes(saved+b' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): verifier.verify(config)
            file.write_bytes(saved)
            file.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'): verifier.verify(config)
            file.write_bytes(saved)
            # Reseal a malicious NaN payload to ensure strict parsing, rather than
            # relying only on the checksum mismatch to reject nonfinite content.
            complete_path = run/'COMPLETED.json'; original_complete = complete_path.read_bytes()
            file.write_text('{"bad": NaN}')
            complete = json.loads(original_complete); complete['evidence_hashes']['panels.json'] = verifier.sha(file)
            complete_path.write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'nonfinite JSON'): verifier.verify(config)
            file.write_bytes(saved); complete_path.write_bytes(original_complete)
            steps_path = run/'seed_42/raw/steps.jsonl'; steps_saved = steps_path.read_bytes()
            steps_path.write_text(steps_saved.decode().replace('"loss": ', '"probe": 1e999, "loss": ', 1))
            complete = json.loads(original_complete); complete['evidence_hashes']['seed_42/raw/steps.jsonl'] = verifier.sha(steps_path)
            complete_path.write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'nonfinite JSON number'): verifier.verify(config)
            steps_path.write_bytes(steps_saved); complete_path.write_bytes(original_complete)
            array_path = sorted((run/'seed_42/raw').glob('screen_step*_users.npz'))[0]; arrays_saved = array_path.read_bytes()
            with np.load(array_path, allow_pickle=False) as handle: arrays = {key: handle[key].copy() for key in handle.files}
            arrays['lengths'] = arrays['lengths'].astype(float); arrays['lengths'][0] -= .5
            np.savez_compressed(array_path, **arrays)
            complete = json.loads(original_complete); complete['evidence_hashes'][str(array_path.relative_to(run))] = verifier.sha(array_path)
            complete_path.write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'integer dtype required: lengths'): verifier.verify(config)
            array_path.write_bytes(arrays_saved); complete_path.write_bytes(original_complete)
            config_path = run/'run_config.json'; config_saved = config_path.read_bytes()
            bad_config = json.loads(config_saved); bad_config['dim'] += 1; config_path.write_text(json.dumps(bad_config))
            complete = json.loads(original_complete); complete['evidence_hashes']['run_config.json'] = verifier.sha(config_path); complete_path.write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'run setting mismatch'): verifier.verify(config)
            config_path.write_bytes(config_saved); complete_path.write_bytes(original_complete)
            outside = root/'panels-copy.json'; outside.write_bytes(saved); file.unlink(); file.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'symlink'): verifier.verify(config)
            file.unlink(); file.write_bytes(saved)
            second_array = sorted((run/'seed_43/raw').glob('screen_step*_users.npz'))[0]; second_saved = second_array.read_bytes()
            with np.load(second_array, allow_pickle=False) as handle: arrays = {key: handle[key].copy() for key in handle.files}
            arrays['position'][0] += 1; np.savez_compressed(second_array, **arrays)
            complete = json.loads(original_complete); complete['evidence_hashes'][str(second_array.relative_to(run))] = verifier.sha(second_array); complete_path.write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'unstable screen identity'): verifier.verify(config)
            second_array.write_bytes(second_saved); complete_path.write_bytes(original_complete)
            self.assertFalse(Path(config.receipt).exists())
            return result
        with patch.object(runner, 'run', side_effect=run_and_verify):
            fixtures.DiagnosticsTests('test_real_protocol_prepare_load_and_three_seed_smoke').test_real_protocol_prepare_load_and_three_seed_smoke()

if __name__ == '__main__': unittest.main()
