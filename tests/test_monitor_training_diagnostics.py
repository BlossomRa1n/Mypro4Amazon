import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import monitor_training_diagnostics as monitor


class MonitorDiagnosticsTests(unittest.TestCase):
    def test_success_and_failed_child_preserve_evidence(self):
        for code in (0, 3):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); run = root/'run'; control = root/'control'
                marker = 'COMPLETED.json' if code == 0 else 'FAILED.json'
                script = "import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/sys.argv[2]).write_text(json.dumps({'status':'dummy'})); print('dummy evidence',flush=True); sys.exit(int(sys.argv[3]))"
                result = monitor.supervise(SimpleNamespace(run_dir=str(run), control_dir=str(control), interval=900, no_gpu_probe=True, command=[sys.executable, '-c', script, str(run), marker, str(code)]))
                self.assertEqual(result, code)
                self.assertTrue((run/marker).exists()); self.assertTrue((control/marker).exists())
                status = json.loads((control/'status.json').read_text())
                self.assertEqual(status['returncode'], code); self.assertLess(status['elapsed_seconds'], 10)
                self.assertIn('dummy evidence', (control/'training.log').read_text())
                self.assertTrue((control/'status.jsonl').read_text().strip())
                with self.assertRaises(FileExistsError): monitor.supervise(SimpleNamespace(run_dir=str(run), control_dir=str(control), interval=900, command=['false']))

    def test_missing_completion_marker_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = monitor.supervise(SimpleNamespace(run_dir=str(root/'run'), control_dir=str(root/'control'), interval=900, no_gpu_probe=True, command=[sys.executable, '-c', 'pass']))
            self.assertEqual(result, 1); self.assertTrue((root/'control'/'FAILED.json').exists())

if __name__ == '__main__': unittest.main()
