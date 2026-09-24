import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import tests.test_training_diagnostics as fixtures
import run_cross_multiseed as r
import run_training_diagnostics as d
import diagnostic_metrics as dm

spec=importlib.util.spec_from_file_location('fullfinal',Path(__file__).resolve().parents[1]/'code/run_full_final_test100k.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
f.r=r;f.d=d;f.dm=dm
import cross_pool_cache
f.cache=cross_pool_cache

class FullFinalTests(unittest.TestCase):
    def test_real_prepare_to_finalize_subprocess(self):
        import tests.test_early_stop_cross as cross_tests
        original=d.run; source=Path(__file__).resolve().parents[1]; original_subprocess=subprocess.run
        def run_and_full(args):
            result=original(args)
            # The enclosing cross fixture creates its control tree after raw
            # returns. Hook its successful cross subprocess, then execute final.
            return result
        def process(command,*pargs,**kwargs):
            result=original_subprocess(command,*pargs,**kwargs)
            if isinstance(command,list) and any(str(x).endswith('/code/run_early_stop_cross.py') for x in command) and result.returncode==0:
                def arg(name):return Path(command[command.index(name)+1]).resolve()
                root=arg('--run-dir').parent; legacy=arg('--legacy-source-dir');diag=arg('--protocol-manifest');cross=arg('--run-dir');raw=arg('--raw-run')
                manifest=json.loads(diag.read_text());evidence=Path(manifest['evidence_root']);identity=evidence/'lock.json';identityseal=evidence/'identity_seal.json';r.write_json(identityseal,{'lock_sha256':f.sha(identity)})
                def runcli(*parts):
                    answer=original_subprocess([sys.executable,*map(str,parts)],capture_output=True,text=True)
                    self.assertEqual(answer.returncode,0,answer.stderr);return answer
                fullpool=root/'fullpool';runcli(source/'tools/build_full_prefix_pools.py','--legacy-source-dir',legacy,'--protocol-manifest',diag,'--output-dir',fullpool,'--workers','1','--smoke')
                refs={}
                def add(name,path):refs[name]=f.reference(path)
                for name,path in [('cross_completion',cross/'COMPLETED.json'),('cross_paired',cross/'paired_confirm.json'),('cross_locked',cross/'CHECKPOINTS_LOCKED.json'),('raw_completion',raw/'COMPLETED.json'),('raw_init',raw/'seed_42/raw/initial_state_audit.json'),('raw_run_config',raw/'run_config.json'),('diagnostic_manifest',diag),('full_pool_completion',fullpool/'COMPLETED.json'),('pilot_completion',fullpool/'COMPLETED.json'),('test_lock',identity),('test_records',evidence/'server_snapshot/locked_records.json'),('test_identity_seal',identityseal),('template',source/'docs/FULL_FINAL_PLAN_TEMPLATE_20260925.md')]:add(name,path)
                for label in ('cross_archive','cross_sources','cross_transfer'):
                    path=root/(label+'.json');r.write_json(path,dict(status='verified',cross_completion_sha256=refs['cross_completion']['sha256']));add(label,path)
                path=root/'raw_archive.json';r.write_json(path,dict(status='verified',completion_sha256=refs['raw_completion']['sha256'],formal_archive_verified=False));add('raw_archive',path)
                path=root/'backup_binding.json';r.write_json(path,dict(status='verified',raw_completion_sha256=refs['raw_completion']['sha256'],archive_receipt_sha256=refs['raw_archive']['sha256']));add('backup_binding',path)
                rawenv=json.loads((raw/'run_config.json').read_text())['environment'];env={key:rawenv[key] for key in ('python','torch','numpy','cuda_runtime','cudnn','torch_threads')};env['python']=env['python'].split()[0]
                auth=root/'authorization.json';r.write_json(auth,dict(status='authorized',authorized_by='synthetic fixture',smoke=True,legacy_source_dir=str(legacy),environment=env,resource_review=dict(status='approved',estimated_remaining_seconds=10),refs=refs))
                executor=source/'code/run_full_final_test100k.py';plan=root/'final_plan.json';runcli(executor,'prepare','--authorization',auth,'--output',plan)
                training=root/'fulltrain';runcli(executor,'train','--final-plan',plan,'--run-dir',training)
                planobj=json.loads(plan.read_text());archive=root/'full_archive.json';r.write_json(archive,dict(status='verified',plan_hash=planobj['plan_hash'],training_completion_sha256=f.sha(training/'TRAINING_COMPLETED.json')))
                prepared=root/'prepared.json';runcli(executor,'seal-models','--final-plan',plan,'--run-dir',training,'--archive-receipt',archive,'--output',prepared)
                test=root/'finaltest';runcli(executor,'test','--final-plan',plan,'--prepared',prepared,'--test-dir',test)
                archive=root/'test_archive.json';r.write_json(archive,dict(status='verified',plan_hash=planobj['plan_hash'],test_completion_sha256=f.sha(test/'TEST_COMPLETED.json')))
                runcli(executor,'finalize','--final-plan',plan,'--test-dir',test,'--archive-receipt',archive,'--output',root/'FINAL_COMPLETED.json')
                rerun=original_subprocess([sys.executable,str(executor),'test','--final-plan',str(plan),'--prepared',str(prepared),'--test-dir',str(root/'test_again')],capture_output=True,text=True);self.assertNotEqual(rerun.returncode,0)
            return result
        with patch.object(subprocess,'run',side_effect=process):cross_tests.EarlyCrossTests('test_smoke_with_sealed_legacy_controls').test_smoke_with_sealed_legacy_controls()

    def test_boundaries_reject_paths_json_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();bad=root/'bad.json';bad.write_text('{"x":1e999}')
            with self.assertRaises(ValueError):f.read(bad)
            diagnostic=root/'tampered_diagnostic.json';value={'manifest_hash':'fixed','code_hashes':{'evil.py':'bad'},'base_run':'mutated'};r.write_json(diagnostic,value)
            with patch.object(f.importlib,'import_module') as imported,self.assertRaisesRegex(ValueError,'canonical seal mismatch'):
                f.bind_legacy({'smoke':True,'refs':{'diagnostic_manifest':f.reference(diagnostic)},'legacy_source_dir':str(root)})
            imported.assert_not_called()
            target=root/'target';target.write_text('x');link=root/'link';link.symlink_to(target)
            with self.assertRaises(ValueError):f.safe_path(link)
            out=root/'out';out.mkdir();(out/'dangling').symlink_to(root/'missing')
            with self.assertRaises(ValueError):f.empty_directory(out)
            guard=f.exact_target_guard(lambda rows:rows,[(2,8,'a')]);self.assertEqual(guard([(2,8)]),[(2,8)])
            with self.assertRaises(ValueError):guard([(2,9)])
            with self.assertRaises(PermissionError):f.denied_targets([])

    def test_full_cpu_training_prepared_and_one_test_session(self):
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();base,_=fixtures._fixture(root);data=r.load_data(base)
            test=[(int(u),int(p),str(data.users[u])) for u,p in list(zip(data.val_users,data.val_positions))[-2:]]
            positions=np.asarray(data.train_positions,dtype=np.int64)
            pools=[[int(x) for x in data.active_items if int(x) not in data.train_sets[int(data.uid[p])]][:75] for p in positions]
            settings=dict(dim=8,token_dim=8,hist_len=5,batch_size=8,candidates=75);config=SimpleNamespace(**settings)
            factors,_=r._load_factors(base,data,8,True);template,state=r._template(data,config,424242,factors);del template
            model=r._new_variant(data,config,state,'raw');audit=root/'audit.json';r.write_json(audit,dict(state_sha256=r._state_hashes(model)));del model
            rawcomplete=root/'raw_complete.json';r.write_json(rawcomplete,{'seeds':{'42':{'model_config':r._config(data,config,'raw')}}})
            testpath=root/'test_records.json';r.write_json(testpath,test);lock=root/'identity_lock.json';r.write_json(lock,{'count':2})
            manifest=dict(base_run=str(base),smoke_lock=str(lock));old={};poolmeta={'pool_hash':'fixture'}
            plan=dict(protocol=f.VERSION,plan_hash='fixture-plan',smoke=True,models=['raw','zero_cross'],settings=settings,environment=f.environment(),refs={'raw_init':f.reference(audit),'raw_completion':f.reference(rawcomplete),'test_records':f.reference(testpath),'test_lock':f.reference(lock)},train_rows=len(positions),train_positions_sha256=r.sha256_array(positions),test_records_hash=cross_pool_cache.rows_hash(test),test_users=len(test))
            planpath=root/'plan.json';r.write_json(planpath,plan)
            def inputs(_):
                value=r.load_data(base);value.targets=f.denied_targets
                return manifest,value,positions,pools,test,old,poolmeta
            run=root/'training'
            with patch.object(f,'validate_plan',return_value=plan),patch.object(f,'load_inputs',side_effect=inputs):
                result=f.train(SimpleNamespace(final_plan=str(planpath),run_dir=str(run)))
            self.assertFalse(result['test_future_labels_read']);self.assertEqual(result['results']['raw']['trace'],result['results']['zero_cross']['trace']);self.assertEqual(result['results']['raw']['trace']['test_users_covered'],2)
            completepath=run/'TRAINING_COMPLETED.json';completebytes=completepath.read_bytes();bad=json.loads(completebytes);bad['evidence_hashes'].pop('raw/initial_state_audit.json');r.write_json(completepath,bad)
            with self.assertRaisesRegex(ValueError,'artifact seal incomplete'):f.verify_training(plan,run)
            completepath.write_bytes(completebytes)
            tracepath=run/'raw/trace.json';tracebytes=tracepath.read_bytes();badtrace=json.loads(tracebytes);badtrace['order_sha256']='0'*64;r.write_json(tracepath,badtrace)
            bad=json.loads(completebytes);bad['results']['raw']['trace']=badtrace;bad['evidence_hashes']['raw/trace.json']=f.sha(tracepath)
            modelcomplete=run/'raw/COMPLETED.json';modelbytes=modelcomplete.read_bytes();r.write_json(modelcomplete,bad['results']['raw']);bad['evidence_hashes']['raw/COMPLETED.json']=f.sha(modelcomplete);r.write_json(completepath,bad)
            with self.assertRaisesRegex(ValueError,'update/order trace'):f.verify_training(plan,run)
            tracepath.write_bytes(tracebytes);modelcomplete.write_bytes(modelbytes);completepath.write_bytes(completebytes)
            copiedlock=root/'copiedlock.json';copiedlock.write_bytes(lock.read_bytes());copiedplan=dict(plan,refs=dict(plan['refs'],test_lock=f.reference(copiedlock)))
            with self.assertRaisesRegex(ValueError,'lock path relocated'):f.bound_test_lock(copiedplan,manifest)
            archive=root/'training_archive.json';r.write_json(archive,dict(status='verified',training_completion_sha256=f.sha(run/'TRAINING_COMPLETED.json'),plan_hash=plan['plan_hash']))
            prepared=root/'FINAL_MODELS_PREPARED.json'
            with patch.object(f,'validate_plan',return_value=plan),patch.object(f,'bind_legacy'):
                f.seal_models(SimpleNamespace(final_plan=str(planpath),run_dir=str(run),archive_receipt=str(archive),output=str(prepared)))
            prepared_obj=json.loads(prepared.read_text());copiedcheckpoint=root/'substituted.pth';copiedcheckpoint.write_bytes((run/'raw/final.pth').read_bytes());prepared_obj['checkpoints']['raw']=f.reference(copiedcheckpoint)
            with self.assertRaisesRegex(ValueError,'checkpoint substituted'):f.bind_prepared_checkpoints(plan,prepared_obj,result)
            archive_saved=archive.read_bytes();r.write_json(archive,dict(status='verified',training_completion_sha256='wrong',plan_hash=plan['plan_hash']))
            with patch.object(f,'validate_plan',return_value=plan),patch.object(f,'bind_legacy'),self.assertRaises(ValueError):
                f.seal_models(SimpleNamespace(final_plan=str(planpath),run_dir=str(run),archive_receipt=str(archive),output=str(root/'BAD_PREPARED.json')))
            archive.write_bytes(archive_saved)
            calls=[]
            def build(data,directory,old,records,label,budget):
                self.assertTrue((root/'FINAL_TEST100K_STARTED.json').exists());self.assertTrue((directory/'TEST_STARTED.json').exists());calls.append(1)
                result=[]
                for uid,position,_ in records:
                    self.assertTrue(data.targets([(uid,position)]));result.append([int(x) for x in data.active_items if int(x) not in data.train_sets[uid]][:75])
                return result,dict(source={'rrf_weights':[2.,1.,.7,.05]},records_hash=plan['test_records_hash'],pool_hash='testfixture')
            args=SimpleNamespace(final_plan=str(planpath),prepared=str(prepared),test_dir=str(root/'test'))
            with patch.object(f,'validate_plan',return_value=plan),patch.object(f,'bind_legacy',return_value=manifest),patch.object(f,'load_inputs',side_effect=inputs),patch.object(r,'_load_or_make_assets',side_effect=build):
                result=f.final_test(args);self.assertEqual(result['users'],2);self.assertEqual(calls,[1])
                args.test_dir=str(root/'different_test_dir')
                with self.assertRaises(ValueError):f.final_test(args)
            final_archive=root/'test_archive.json';r.write_json(final_archive,dict(status='verified',plan_hash=plan['plan_hash'],test_completion_sha256=f.sha(root/'test/TEST_COMPLETED.json')))
            with patch.object(f,'validate_plan',return_value=plan):
                f.finalize(SimpleNamespace(final_plan=str(planpath),test_dir=str(root/'test'),archive_receipt=str(final_archive),output=str(root/'FINAL_COMPLETED.json')))
            self.assertTrue((root/'FINAL_COMPLETED.json').exists())

if __name__=='__main__':unittest.main()
