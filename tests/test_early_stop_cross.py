import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import tests.test_training_diagnostics as fixtures
import run_training_diagnostics as d
import run_cross_multiseed as r
import diagnostic_protocol

class EarlyCrossTests(unittest.TestCase):
    def test_guard_boundaries(self):
        import importlib.util
        from datetime import timedelta
        path=Path(__file__).resolve().parents[1]/'code/run_early_stop_cross.py';spec=importlib.util.spec_from_file_location('early_cross_guard',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.r=r
        module.check_deadline(module.DEADLINE-timedelta(microseconds=1))
        with self.assertRaises(ValueError):module.check_deadline(module.DEADLINE)
        module.check_deadline(module.DEADLINE,True)
        env=dict(python='3.12.4',torch='2.14',numpy='2.5.2',cuda_runtime='13',cudnn=1,torch_threads=4)
        module.check_environment(env,dict(env))
        with self.assertRaises(ValueError):module.check_environment(env,dict(env,numpy='2.4'))
        module.validate_backup_binding(dict(status='verified',raw_completion_sha256='a',archive_receipt_sha256='b'),'a','b')
        with self.assertRaises(ValueError):module.validate_backup_binding(dict(status='verified'),'a','b')
        value={'evidence_hashes':{}};value['manifest_hash']=r.sha256_json(value);module.validate_old_manifest(value,'fixture',True)
        value['evidence_hashes']['x']='bad'
        with self.assertRaises(ValueError):module.validate_old_manifest(value,'fixture',True)

    def test_smoke_with_sealed_legacy_controls(self):
        original=d.run
        def run_cross(args):
            completed=original(args);raw=Path(args.run_dir);root=raw.parent
            manifest,data,records,positions,pools=diagnostic_protocol.load_inputs(args.protocol_manifest)
            config=SimpleNamespace(**manifest['settings'],epochs=3)
            factors,_=r._load_factors(Path(manifest['base_run']),data,config.dim,True)
            old=root/'oldzero';old.mkdir()
            for seed in r.SEEDS:
                template,state=r._template(data,config,r.INIT_SEEDS[seed],factors);del template
                r._train_model(data,config,old,seed,'zero_cross',state,factors,positions,pools['train'],records['screen'],pools['screen'],True)
            old_manifest={'evidence_hashes':{str(p.relative_to(old)):r.sha256_file(p) for p in old.rglob('*') if p.is_file()}}
            old_manifest['manifest_hash']=r.sha256_json(old_manifest);r.write_json(old/'dev_manifest.json',old_manifest)
            legacy=root/'legacy';legacy.mkdir();source=Path(__file__).resolve().parents[1]/'code'
            for path in source.glob('*.py'):shutil.copy2(path,legacy/path.name)
            command=[sys.executable,str(source/'run_early_stop_cross.py'),'--legacy-source-dir',str(legacy),'--protocol-manifest',args.protocol_manifest,'--raw-run',str(raw),'--old-zero-run',str(old),'--run-dir',str(root/'cross'),'--protocol-doc',str(source.parent/'docs/OVERNIGHT_CROSS_PROTOCOL_20260925.md'),'--smoke']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            cross=json.loads((root/'cross/COMPLETED.json').read_text());self.assertEqual(len(list((root/'cross').rglob('*.pth'))),3)
            self.assertFalse(cross['test_future_labels_read']);self.assertEqual(cross['paired']['bootstrap_seed'],20260925);self.assertEqual(cross['paired']['ci_level'],.975)
            for seed in r.SEEDS:
                history=json.loads((root/f'cross/seed_{seed}/zero_cross/history.json').read_text());self.assertEqual([p for entry in history for p in entry['requested_points']],[.25,.5,1.])
                self.assertEqual(cross['seeds'][str(seed)]['traces']['1'],completed['seeds'][str(seed)]['traces']['1'])
                self.assertEqual(history[-1]['step'],cross['seeds'][str(seed)]['steps_per_epoch'])
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            return completed
        with patch.object(d,'run',side_effect=run_cross):fixtures.DiagnosticsTests('test_real_protocol_prepare_load_and_three_seed_smoke').test_real_protocol_prepare_load_and_three_seed_smoke()

if __name__=='__main__':unittest.main()
