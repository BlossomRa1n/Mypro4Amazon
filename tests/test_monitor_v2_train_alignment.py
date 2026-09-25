import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import monitor_v2_train_alignment as m


class MonitorV2Tests(unittest.TestCase):
    def test_complete_and_failed_children_exit_promptly(self):
        for code in (0,3):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();data=root/'pool';data.mkdir();run=root/'run';control=root/'control'
                marker='COMPLETED.json' if code==0 else 'FAILED.json'
                script="import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); s=p/'seed_42/raw_v2_train'; s.mkdir(parents=True); (s/'steps.jsonl').write_text(json.dumps({'step':591})+'\\n'+ '{partial'); (p/sys.argv[2]).write_text('{}'); print('child evidence',flush=True); sys.exit(int(sys.argv[3]))"
                args=SimpleNamespace(run_dir=str(run),control_dir=str(control),data_dir=str(data),interval=900,no_gpu_probe=True,command=[sys.executable,'-c',script,str(run),marker,str(code)])
                self.assertEqual(m.supervise(args),code)
                status=json.loads((control/'status.json').read_text())
                self.assertLess(status['elapsed_seconds'],10)
                self.assertEqual(status['latest_steps']['seed_42/raw_v2_train/steps.jsonl']['step'],591)
                self.assertEqual(set(status['filesystems']),{'data','system'})
                self.assertEqual(status['filesystems']['data']['reserve_bytes'],2*1024**3)
                self.assertEqual(status['filesystems']['system']['reserve_bytes'],1536*1024**2)
                for fs in status['filesystems'].values():self.assertIsInstance(fs['filesystem_device'],int);self.assertGreater(fs['free_inodes'],0)
                self.assertTrue((control/marker).is_file());self.assertTrue((run/marker).is_file())
                launch=json.loads((control/'launch.json').read_text())
                self.assertFalse(launch['automatic_restart']);self.assertFalse(launch['shutdown_allowed'])
                self.assertEqual(len(launch['monitor_source_sha256']),64)
                self.assertIn('child evidence',(control/'training.log').read_text())
                with self.assertRaises(FileExistsError):m.supervise(args)

    def test_missing_marker_and_overlapping_roots_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();data=root/'pool';data.mkdir()
            args=SimpleNamespace(run_dir=str(root/'run'),control_dir=str(root/'control'),data_dir=str(data),interval=900,no_gpu_probe=True,command=[sys.executable,'-c','pass'])
            self.assertEqual(m.supervise(args),1)
            self.assertTrue((root/'control/FAILED.json').is_file())
            args.run_dir=str(data/'bad');args.control_dir=str(root/'unused')
            with self.assertRaises(ValueError):m.supervise(args)
            self.assertFalse((root/'unused').exists())


if __name__=='__main__':unittest.main()
