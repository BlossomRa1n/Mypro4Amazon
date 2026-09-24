"""Exercise the verifier on actual tiny training output and resealed corruptions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import verify_early_stop_cross_archive as v
import verify_training_diagnostics_archive as raw_v
from tests import test_early_stop_cross as early_fixture


class CrossArchiveTests(unittest.TestCase):
    def test_real_cross_archive_and_adversarial_rejections(self):
        original=subprocess.run
        verified=[]
        def intercept(command,*args,**kwargs):
            result=original(command,*args,**kwargs)
            if result.returncode or 'run_early_stop_cross.py' not in str(command):return result
            def arg(name):return command[command.index('--'+name)+1]
            run=Path(arg('run-dir'));root=run.parent;raw=Path(arg('raw-run'))
            raw_receipt=root/'raw_verified.json'
            raw_v.verify(SimpleNamespace(run_dir=str(raw),protocol_manifest=arg('protocol-manifest'),receipt=str(raw_receipt),allow_smoke=True))
            bundle=root/'bundle';source=bundle/'src';source.mkdir(parents=True)
            for name,path in [('run_early_stop_cross.py',Path(command[1])),('OVERNIGHT_CROSS_PROTOCOL_20260925.md',Path(arg('protocol-doc')))]:shutil.copy2(path,source/name)
            source_manifest=bundle/'source_manifest.json'
            source_manifest.write_text(json.dumps({p.name:v.sha(p) for p in source.iterdir()}))
            config=SimpleNamespace(run_dir=str(run),raw_run=str(raw),old_zero_run=arg('old-zero-run'),
                protocol_manifest=arg('protocol-manifest'),raw_archive_receipt=str(raw_receipt),
                source_manifest=str(source_manifest),receipt=str(root/'cross_verified.json'),allow_smoke=True)
            receipt=v.verify(config);verified.append(receipt)
            self.assertEqual(receipt['counts']['checkpoints'],3)
            self.assertTrue(receipt['bootstrap_independently_recomputed'])
            self.assertFalse(receipt['formal_archive_verified'])
            self.assertEqual(receipt['cross_completion_sha256'],receipt['completion_sha256'])
            config.receipt=str(root/'rejected.json')
            config.allow_smoke=False
            with self.assertRaisesRegex(ValueError,'smoke cannot'):v.verify(config)
            config.allow_smoke=True
            completion=run/'COMPLETED.json';original_completion=completion.read_bytes()
            def mutate(relative,change,message,paired=False):
                path=run/relative;saved=path.read_bytes();value=json.loads(saved);change(value);path.write_text(json.dumps(value))
                complete=json.loads(original_completion);complete['evidence_hashes'][relative]=v.sha(path)
                if paired:complete['paired']=value
                completion.write_text(json.dumps(complete))
                try:
                    with self.assertRaisesRegex(ValueError,message):v.verify(config)
                    self.assertFalse(Path(config.receipt).exists())
                finally:path.write_bytes(saved);completion.write_bytes(original_completion)
            mutate('paired_confirm.json',lambda obj:obj['hit5']['ci975'].__setitem__(0,obj['hit5']['ci975'][0]+.1),'independently recomputed CI975',True)
            mutate('paired_confirm.json',lambda obj:obj.__setitem__('selection','invalid'),'selection result mismatch',True)
            mutate('seed_42/zero_cross/initial_state_audit.json',lambda obj:obj.__setitem__('gate_value',9.),'initial state differs')
            mutate('run_manifest.json',lambda obj:obj['config'].__setitem__('epochs',2),'epoch/scheduler')
            for corruption in ('shape','dtype'):
                checkpoint=run/'seed_42/zero_cross/best.pth'
                locked_path=run/'CHECKPOINTS_LOCKED.json'
                seed_path=run/'seed_42/zero_cross/COMPLETED.json'
                originals={path:path.read_bytes() for path in (checkpoint,locked_path,seed_path,completion)}
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                key=next(key for key,tensor in payload['model'].items() if tensor.is_floating_point() and tensor.ndim>1)
                tensor=payload['model'][key]
                payload['model'][key]=tensor.reshape(-1) if corruption=='shape' else tensor.to(torch.float64)
                torch.save(payload,checkpoint)
                digest=v.sha(checkpoint)
                lock=json.loads(originals[locked_path]);lock['42']['checkpoint_sha256']=digest;locked_path.write_text(json.dumps(lock))
                seed=json.loads(originals[seed_path]);seed['best']['checkpoint_sha256']=digest;seed_path.write_text(json.dumps(seed))
                complete=json.loads(originals[completion]);complete['seeds']['42']=seed
                for path in (checkpoint,locked_path,seed_path):complete['evidence_hashes'][str(path.relative_to(run))]=v.sha(path)
                completion.write_text(json.dumps(complete))
                try:
                    with self.assertRaisesRegex(ValueError,'tensor shape/dtype differs'):v.verify(config)
                    self.assertFalse(Path(config.receipt).exists())
                finally:
                    for path,content in originals.items():path.write_bytes(content)
            (run/'UNSEALED.txt').write_text('x')
            with self.assertRaisesRegex(ValueError,'unsealed archive'):v.verify(config)
            (run/'UNSEALED.txt').unlink()
            (run/'FAILED.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'failed marker'):v.verify(config)
            (run/'FAILED.json').unlink()
            path=run/'seed_42/zero_cross/confirm_best_users.npz';saved=path.read_bytes()
            with np.load(path,allow_pickle=False) as data:arrays={key:data[key].copy() for key in data.files}
            arrays['position'][0]+=1;np.savez_compressed(path,**arrays)
            complete=json.loads(original_completion);complete['evidence_hashes'][str(path.relative_to(run))]=v.sha(path);completion.write_text(json.dumps(complete))
            try:
                with self.assertRaisesRegex(ValueError,'paired identities'):v.verify(config)
            finally:path.write_bytes(saved);completion.write_bytes(original_completion)
            return result
        with patch.object(subprocess,'run',side_effect=intercept):
            early_fixture.EarlyCrossTests('test_smoke_with_sealed_legacy_controls').test_smoke_with_sealed_legacy_controls()
        self.assertEqual(len(verified),1)

if __name__=='__main__':unittest.main()
