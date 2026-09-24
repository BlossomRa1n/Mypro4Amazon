import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import monitor_cross_v3_final as m
import run_cross_multiseed as runner
from tests.test_cross_multiseed import _fixture


class Clock:
    def __init__(self): self.now=0
    def monotonic(self): return self.now
    def sleep(self,seconds): self.now+=seconds
class Child:
    pid=123456
    def __init__(self,clock,code=0,until=901): self.clock=clock; self.code=code; self.until=until
    def poll(self): return self.code if self.clock.now>=self.until else None
    def wait(self): return self.code


class MonitorTests(unittest.TestCase):
    def paths(self,root):
        stack=ExitStack()
        for key,value in {'ROOT':root,'CODE':root/'src_v3/code','BASE':root/'base','PLAN':root/'final_plan_v3.json','PROTOCOL':root/'protocol_v3','FINAL':root/'final_v3','DEV':root/'dev'}.items(): stack.enter_context(patch.object(m,key,value))
        m.CODE.mkdir(parents=True,exist_ok=True);m.PROTOCOL.mkdir(exist_ok=True)
        for path in [m.CODE/name for name in m.FROZEN]+[m.PLAN,m.ROOT/'selection_v3.json',m.PROTOCOL/'protocol_manifest.json']:path.write_text('{}')
        return stack

    def test_exact_fixed_commands_and_parser(self):
        train=m.command('final-train'); test=m.command('final-test')
        self.assertEqual(train[:4],[m.PYTHON,'-u',str(m.CODE/'run_cross_multiseed.py'),'final-train'])
        self.assertEqual(train[-4:],['--batch-size','256','--epochs','3'])
        self.assertNotIn('--epochs',test); self.assertEqual(m.INTERVAL,900)
        for cmd in (train,test):
            parsed=runner.parser().parse_args(cmd[3:]);self.assertEqual(parsed.command,cmd[3])
            self.assertEqual(cmd[cmd.index('--run-dir')+1],'/root/next_cross_20260924/final_v3')
            self.assertEqual(cmd[cmd.index('--final-plan')+1],'/root/next_cross_20260924/final_plan_v3.json')
        with self.assertRaises(ValueError):m.command('dev')
        with self.assertRaises(SystemExit):m.parser().parse_args(['--stage','final-test','--interval','5'])

    def test_check_only_has_no_writes_and_does_not_spawn(self):
        with tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
            gate=lambda s:({'plan_hash':'plan'},{'manifest_hash':'protocol'})
            with patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
                self.assertEqual(m.run('final-train',gate=gate)['status'],'check_passed')
            self.assertFalse(m.monitor_dir('final-train').exists())

    def test_single_child_interval_raw_log_and_persistent_lock(self):
        with tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
            clock=Clock(); child=Child(clock); phases=[]; commands=[]
            def popen(cmd,**kw):
                commands.append(cmd); kw['stdout'].write(b'\xffRAW LOSS secret bytes\n'); return child
            gate=lambda s:({'plan_hash':'plan'},{'manifest_hash':'protocol'})
            with patch.object(m,'snapshot',side_effect=lambda c,o,p:phases.append((p,clock.now))),patch.object(m,'verify_completion',return_value={'plan_hash':'plan'}):
                result=m.run('final-train',True,gate=gate,popen=popen,timer=clock)
            self.assertEqual(result['status'],'complete'); self.assertEqual(len(commands),1)
            self.assertEqual(phases,[('start',0),('interval',900),('end',905)])
            self.assertEqual((m.monitor_dir('final-train')/'stdout.log').read_bytes(),b'\xffRAW LOSS secret bytes\n')
            with self.assertRaises(FileExistsError):m.run('final-train',True,gate=gate,popen=popen,timer=clock)
            self.assertEqual(len(commands),1)

    def test_nonzero_or_missing_completion_never_reports_success(self):
        for code in (0,9):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
                clock=Clock(); child=Child(clock,code,0)
                with patch.object(m,'snapshot',return_value={}),patch.object(m,'verify_completion',side_effect=ValueError('missing marker')) as verify:
                    result=m.run('final-test',True,gate=lambda s:({'plan_hash':'p'},{'manifest_hash':'h'}),popen=lambda *a,**kw:child,timer=clock)
                self.assertEqual(result['status'],'failed_or_incomplete')
                if code:verify.assert_not_called()
                self.assertTrue((m.monitor_dir('final-test')/'exit.json').exists())

    def test_snapshot_never_reads_training_log_or_metric_contents(self):
        with tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
            m.FINAL.mkdir(); out=m.monitor_dir('final-train');out.mkdir()
            (out/'stdout.log').write_text('PRIVATE LOSS DATA');(m.FINAL/'raw_test_metrics.json').write_text('PRIVATE METRICS')
            child=Child(Clock(),until=0)
            original=Path.read_text
            def read(path,*args,**kwargs):
                if path.is_relative_to(out) or path.is_relative_to(m.FINAL):raise AssertionError('read model output contents')
                return original(path,*args,**kwargs)
            with patch.object(m,'capture',return_value={'ok':True,'output':'resource only'}),patch.object(Path,'read_text',read):
                result=m.snapshot(child,out,'start')
            self.assertNotIn('PRIVATE',json.dumps(result));self.assertEqual(result['final_files'],[{'path':'raw_test_metrics.json','bytes':15}])

    def test_frozen_hash_and_marker_fail_before_official_gate(self):
        with tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
            (m.CODE/'run_cross_multiseed.py').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'frozen code SHA'):m.preflight('final-train')
            (m.PROTOCOL/'TEST_STARTED.json').write_text('do not read')
            with self.assertRaisesRegex(ValueError,'test marker'):m.no_test_started()
            (m.PROTOCOL/'TEST_STARTED.json').unlink();(m.PROTOCOL/'TEST_STARTED.json').symlink_to('/missing')
            with self.assertRaisesRegex(ValueError,'test marker'):m.no_test_started()

    def test_real_frozen_runner_artifacts_completion_and_tamper(self):
        runner.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
            base,history=_fixture(m.ROOT);(m.PROTOCOL/'protocol_manifest.json').unlink()
            protocol=runner.prepare(SimpleNamespace(base_run=str(base),run_dir=str(m.PROTOCOL),historical_users=str(history),train_users=4,screen_users=2,confirm_users=2,test_users=2,cohort_seed=runner.COHORT_SEED,smoke=True))
            plan={'final_evaluation_allowed':True,'base_run':str(base),'protocol_manifest':str(m.PROTOCOL/'protocol_manifest.json'),'plan_hash':'fixture-only','selection_hash':'fixture-only','models':['raw','zero_cross'],'winner':'zero_cross','test_cohort_hash':protocol['cohort_hashes']['test'],'test_mode':protocol['test_mode']}
            args=SimpleNamespace(final_plan=str(m.PLAN),base_run=str(base),run_dir=str(m.FINAL),dim=8,token_dim=8,hist_len=5,batch_size=8,epochs=3,candidates=75)
            with patch.object(runner,'_load_plan',return_value=plan),patch.object(runner.torch.cuda,'is_available',return_value=False):
                trained=runner.final_train(args)
                with patch.object(m,'FULL_ROWS',trained['all_prefix_rows']),patch.object(m,'TEST_USERS',2):
                    value=m.verify_completion('final-train',plan,protocol);self.assertEqual(value['final_manifest_hash'],trained['manifest_hash'])
                    runner.final_test(args)
                    result=m.verify_completion('final-test',plan,protocol);self.assertIn('results_hash',result);self.assertNotIn('paired',result)
                    marker=m.PROTOCOL/'TEST_STARTED.json';obj=json.loads(marker.read_text());obj['results_hash']='wrong';marker.write_text(json.dumps(obj))
                    with self.assertRaisesRegex(ValueError,'marker/result'):m.verify_completion('final-test',plan,protocol)
                    (m.FINAL/'raw/last.pth').write_bytes(b'corrupt')
                    with self.assertRaisesRegex(ValueError,'evidence|checkpoint'):m.verify_train(plan,protocol)

    def test_hash_contract_unicode_and_escape_rejected_before_read(self):
        self.assertEqual(m.object_hash({'label':'历史闭合'}),runner.sha256_json({'label':'历史闭合'}))
        for rel in ('../outside','/outside','a/../../outside','a/./b','a//b'):
            with self.assertRaisesRegex(ValueError,'unsafe evidence'):m.safe_relative(rel)
        self.assertEqual(str(m.safe_relative('seed_42/raw/last.pth')),'seed_42/raw/last.pth')

    def test_existing_test_payloads_without_marker_block_replay(self):
        for name in ('final_test_results.json','raw_test_users.npz','raw_test_metrics.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
                m.FINAL.mkdir();(m.FINAL/name).write_text('existing')
                with self.assertRaisesRegex(ValueError,'test output'):m.no_test_outputs()
        with tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
            (m.PROTOCOL/'candidate_test_final.json.building').write_text('retained')
            with self.assertRaisesRegex(ValueError,'test candidate'):m.no_test_outputs()

    def test_resource_collection_failure_marks_monitoring_failed(self):
        with tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
            clock=Clock();child=Child(clock,0,0)
            with patch.object(m,'snapshot',return_value={'collection_errors':['gpu']}),patch.object(m,'verify_completion',return_value={}):
                result=m.run('final-train',True,gate=lambda s:({'plan_hash':'p'},{'manifest_hash':'h'}),popen=lambda *a,**kw:child,timer=clock)
            self.assertEqual(result['status'],'monitoring_failed');self.assertEqual(len(result['snapshot_errors']),2)

    def test_postexit_code_drift_fails_even_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp,self.paths(Path(tmp).resolve()):
            clock=Clock();child=Child(clock,0,0)
            def popen(*args,**kwargs):
                (m.CODE/'run_cross_multiseed.py').write_text('changed');return child
            with patch.object(m,'snapshot',return_value={}),patch.object(m,'verify_completion') as completion:
                result=m.run('final-train',True,gate=lambda s:({'plan_hash':'p'},{'manifest_hash':'h'}),popen=popen,timer=clock)
            self.assertEqual(result['status'],'failed_or_incomplete');self.assertIn('startup code',result['completion_error']);completion.assert_not_called()

if __name__=='__main__':unittest.main()
