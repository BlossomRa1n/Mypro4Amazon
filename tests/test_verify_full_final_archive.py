import json
import numpy as np
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_full_final_archive as verifier
import tests.test_full_final_test100k as fixture


class FullArchiveTests(unittest.TestCase):
    def test_actual_dual_model_arrays_and_resealed_paired_rejection(self):
        original=fixture.f.final_test;seen=[]
        def evaluate(args):
            result=original(args);root=Path(args.test_dir)
            arrays={}
            for variant in ('raw','zero_cross'):
                with np.load(root/f'{variant}_users.npz',allow_pickle=False) as handle:arrays[variant]={key:handle[key].copy() for key in handle.files}
            verifier.verify_comparison(['raw','zero_cross'],arrays,result);seen.append(True)
            bad=json.loads(json.dumps(result));bad['paired']['hr5_ci95'][0]+=.01
            with self.assertRaisesRegex(ValueError,'bootstrap/deltas'):verifier.verify_comparison(['raw','zero_cross'],arrays,bad)
            bad=json.loads(json.dumps(result));bad['structure_gain_accepted']=not result['structure_gain_accepted']
            with self.assertRaisesRegex(ValueError,'structural acceptance'):verifier.verify_comparison(['raw','zero_cross'],arrays,bad)
            return result
        with patch.object(fixture.f,'final_test',side_effect=evaluate):fixture.FullFinalTests('test_full_cpu_training_prepared_and_one_test_session').test_full_cpu_training_prepared_and_one_test_session()
        self.assertEqual(seen,[True])

    def test_real_cli_archives_and_tamper_rejection(self):
        original=subprocess.run;seen=[]
        def intercept(command,*args,**kwargs):
            result=original(command,*args,**kwargs)
            if isinstance(command,list) and any(str(x).endswith('/code/run_full_final_test100k.py') for x in command) and result.returncode==0:
                index=next(i for i,x in enumerate(command) if str(x).endswith('/code/run_full_final_test100k.py'));phase=command[index+1]
                if phase not in ('train','test'):return result
                def arg(name):return Path(command[command.index(name)+1]).resolve()
                planpath=arg('--final-plan');root=planpath.parent;plan=json.loads(planpath.read_text());run=arg('--run-dir' if phase=='train' else '--test-dir')
                spec=SimpleNamespace(command='training' if phase=='train' else 'test',plan=str(planpath),run_dir=str(run),raw_run=str(root/'run'),test_records=plan['refs']['test_records']['path'],positions=str(root/'fullpool/ranker_train_positions.npy'),base_data=str(root/'base/data.pkl'),receipt=str(root/f'local_verified_{phase}.json'),allow_smoke=True,old_protocol=str(root/'old_protocol/protocol_manifest.json'),diagnostic_manifest=str(root/'diagnostic.json'),prepared=str(root/'prepared.json'),global_marker=str(root/'evidence/FINAL_TEST100K_STARTED.json'),training_run=str(root/'fulltrain'),training_receipt=str(root/'full_archive.json'))
                planbytes=planpath.read_bytes();receipt=verifier.verify(spec);self.assertEqual(receipt['status'],'verified');self.assertFalse(receipt['formal_archive_verified']);self.assertEqual(planbytes,planpath.read_bytes());seen.append(spec.command)
                spec.receipt=str(root/f'rejected_{phase}.json');spec.allow_smoke=False
                with self.assertRaisesRegex(ValueError,'allow-smoke'):verifier.verify(spec)
                spec.allow_smoke=True
                target=run/('raw/steps.jsonl' if phase=='train' else 'raw_metrics.json');saved=target.read_bytes();target.write_bytes(saved+b' ')
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):verifier.verify(spec)
                target.write_bytes(saved)
                completepath=run/('TRAINING_COMPLETED.json' if phase=='train' else 'TEST_COMPLETED.json');completebytes=completepath.read_bytes();complete=json.loads(completebytes)
                if phase=='train':
                    complete['evidence_hashes'].pop('raw/initial_state_audit.json');completepath.write_text(json.dumps(complete))
                    with self.assertRaisesRegex(ValueError,'evidence coverage'):verifier.verify(spec)
                    completepath.write_bytes(completebytes)
                    target.write_text(saved.decode().replace('"loss":','"infinity":1e999,"loss":',1));complete['evidence_hashes']['raw/steps.jsonl']=verifier.sha(target);complete['evidence_hashes']['raw/initial_state_audit.json']=json.loads(completebytes)['evidence_hashes']['raw/initial_state_audit.json'];completepath.write_text(json.dumps(complete))
                    with self.assertRaisesRegex(ValueError,'nonfinite JSON'):verifier.verify(spec)
                    target.write_bytes(saved);completepath.write_bytes(completebytes)
                else:
                    complete['metrics']['raw']['hr5']+=.01;completepath.write_text(json.dumps(complete))
                    with self.assertRaisesRegex(ValueError,'metric file/completion'):verifier.verify(spec)
                    completepath.write_bytes(completebytes)
                if phase=='test':
                    markerpath=run/'TEST_STARTED.json';markerbytes=markerpath.read_bytes();globalpath=Path(spec.global_marker);globalbytes=globalpath.read_bytes()
                    badmarker=json.loads(markerbytes);badmarker['prepared_hash']='f'*64;markerpath.write_text(json.dumps(badmarker));globalpath.write_text(json.dumps(badmarker))
                    badcomplete=json.loads(completebytes);badcomplete['prepared_hash']='f'*64;badcomplete['evidence_hashes']['TEST_STARTED.json']=verifier.sha(markerpath);completepath.write_text(json.dumps(badcomplete))
                    with self.assertRaisesRegex(ValueError,'prepared/checkpoint binding'):verifier.verify(spec)
                    markerpath.write_bytes(markerbytes);globalpath.write_bytes(globalbytes);completepath.write_bytes(completebytes)
                oldreceipt=spec.receipt;spec.receipt=str(Path(spec.raw_run)/'forbidden_receipt.json')
                with self.assertRaisesRegex(ValueError,'outside immutable'):verifier.verify(spec)
                spec.receipt=oldreceipt
                extra=run/'extra.txt';extra.write_text('unsealed')
                with self.assertRaisesRegex(ValueError,'extra or missing'):verifier.verify(spec)
                extra.unlink();self.assertFalse(Path(spec.receipt).exists())
            return result
        with patch.object(subprocess,'run',side_effect=intercept):
            fixture.FullFinalTests('test_real_prepare_to_finalize_subprocess').test_real_prepare_to_finalize_subprocess()
        self.assertEqual(seen,['training','test'])

if __name__=='__main__':unittest.main()
