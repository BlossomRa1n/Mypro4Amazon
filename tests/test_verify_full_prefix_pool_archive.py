import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import tests.test_full_final_test100k as fixture

class FullPoolArchiveTests(unittest.TestCase):
    def test_real_builder_archive_and_tampered_payload(self):
        original=subprocess.run;seen=[];source=Path(__file__).resolve().parents[1]
        def intercept(command,*args,**kwargs):
            result=original(command,*args,**kwargs)
            if isinstance(command,list) and any(str(x).endswith('/tools/build_full_prefix_pools.py') for x in command) and result.returncode==0:
                def arg(name):return Path(command[command.index(name)+1]).resolve()
                full=arg('--output-dir');root=full.parent;legacy=arg('--legacy-source-dir');diag=arg('--protocol-manifest');pilot=root/'pilot_archive'
                build=[sys.executable,str(source/'tools/build_full_prefix_pools.py'),'--legacy-source-dir',str(legacy),'--protocol-manifest',str(diag),'--output-dir',str(pilot),'--workers','1','--pilot-rows','8','--smoke']
                answer=original(build,capture_output=True,text=True);self.assertEqual(answer.returncode,0,answer.stderr)
                import hashlib
                builder=source/'tools/build_full_prefix_pools.py';digest=hashlib.sha256(builder.read_bytes()).hexdigest()
                verify=[sys.executable,str(source/'tools/verify_full_prefix_pool_archive.py'),'--full-run',str(full),'--old-protocol-dir',str(root/'old_protocol'),'--base-archive',str(root/'base'),'--diagnostic-manifest',str(diag),'--legacy-source-dir',str(legacy),'--builder-source',str(builder),'--builder-sha256',digest,'--pilot-run',str(pilot),'--receipt',str(root/'pool_verified.json'),'--allow-smoke']
                answer=original(verify,capture_output=True,text=True);self.assertEqual(answer.returncode,0,answer.stderr);receipt=json.loads((root/'pool_verified.json').read_text());self.assertEqual(receipt['status'],'verified');self.assertFalse(receipt['test_future_labels_read']);self.assertFalse(receipt['models_loaded']);seen.append(True)
                verify[verify.index('--receipt')+1]=str(root/'rejected.json');position=full/'ranker_train_positions.npy';saved=position.read_bytes();position.write_bytes(saved+b'x')
                answer=original(verify,capture_output=True,text=True);self.assertNotEqual(answer.returncode,0);self.assertIn('positions file SHA mismatch',answer.stderr);self.assertFalse((root/'rejected.json').exists());position.write_bytes(saved)
                no_smoke=verify[:-1];answer=original(no_smoke,capture_output=True,text=True);self.assertNotEqual(answer.returncode,0);self.assertIn('allow-smoke',answer.stderr)
                disk=full/'DISK_PREFLIGHT.json';saved=disk.read_bytes();bad=json.loads(saved);bad['observed_free_bytes']=0;disk.write_text(json.dumps(bad))
                answer=original(verify,capture_output=True,text=True);self.assertNotEqual(answer.returncode,0);self.assertIn('disk preflight',answer.stderr);disk.write_bytes(saved)
            return result
        with patch.object(subprocess,'run',side_effect=intercept):fixture.FullFinalTests('test_real_prepare_to_finalize_subprocess').test_real_prepare_to_finalize_subprocess()
        self.assertEqual(seen,[True])

if __name__=='__main__':unittest.main()
