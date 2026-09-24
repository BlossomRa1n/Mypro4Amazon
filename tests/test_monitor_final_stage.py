from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import monitor_final_stage as m

class FinalStageMonitorTests(unittest.TestCase):
    def test_matching_stage_markers_and_zero_exit_succeed(self):
        for stage,marker in [('training','TRAINING_COMPLETED.json'),('test','TEST_COMPLETED.json')]:
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);run=root/'run'
                command=[sys.executable,'-c',"import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.mkdir();(p/sys.argv[2]).write_text('{}')",str(run),marker]
                self.assertEqual(m.supervise(SimpleNamespace(stage=stage,run_dir=str(run),control_dir=str(root/'control'),interval=900,no_gpu_probe=True,command=command)),0)
                self.assertTrue((root/'control/COMPLETED.json').exists())
    def test_missing_or_wrong_stage_marker_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run'
            command=[sys.executable,'-c',"import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.mkdir();(p/'COMPLETED.json').write_text('{}')",str(run)]
            self.assertEqual(m.supervise(SimpleNamespace(stage='training',run_dir=str(run),control_dir=str(root/'control'),interval=900,no_gpu_probe=True,command=command)),1)
            self.assertTrue((root/'control/FAILED.json').exists())

if __name__=='__main__':unittest.main()
