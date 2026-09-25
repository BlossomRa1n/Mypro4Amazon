import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_v2_exposure_groups as g
from tests.test_analyze_training_exposure import fixture as identity_fixture,write


def fixture(root):
    args=identity_fixture(root);raw=Path(args.raw_run);new=root/'new';new.mkdir()
    protocol=g.e.read(Path(args.protocol_dir)/'protocol_manifest.json')
    locks=g.e.read(raw/'CHECKPOINTS_LOCKED.json');oldseeds={};newseeds={};newlocks={};deltas={}
    for seed in (42,43,44):
        oldseeds[str(seed)]={'best':locks[str(seed)]['best']};newlocks[str(seed)]={'step':4};newseeds[str(seed)]={'best':newlocks[str(seed)]}
        trace=raw/f'seed_{seed}/raw/epoch_1_trace.json';write(new/f'seed_{seed}/raw_v2_train/epoch_1_trace.json',g.e.read(trace))
        for split in ('screen','confirm'):
            records=g.e.read(Path(args.protocol_dir)/protocol['records'][split]);n=len(records)
            for name in ([f'screen_step{x}_users.npz' for x in (1,2,4)] if split=='screen' else ['confirm_best_users.npz']):
                for directory,variant,improved in ((raw,'raw',False),(new,'raw_v2_train',True)):
                    values=dict(uid=np.array([x[0] for x in records],dtype=np.int64),position=np.array([x[1] for x in records],dtype=np.int64),candidate_ids=np.tile(np.array([2,3],dtype=np.int32),(n,1)),lengths=np.full(n,2,dtype=np.int32),labels=np.tile([True,False],(n,1)),pool_hit=np.ones(n,dtype=np.int8),auc_valid=np.ones(n,dtype=bool),scores=np.tile([1.,0.],(n,1)).astype(np.float32),auc=np.ones(n,dtype=np.float64),hit5=np.array([1 if improved else 0,1],dtype=np.int8),ndcg5=np.array([.5 if improved else 0,.75],dtype=np.float32))
                    p=directory/f'seed_{seed}/{variant}'/name;p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**values)
        deltas[str(seed)]={'hit5_delta':.5,'ndcg5_delta':.25}
    olddone=g.e.read(raw/'COMPLETED.json');olddone['seeds']=oldseeds
    def complete(directory,value):
        value['evidence_hashes']={p.relative_to(directory).as_posix():g.e.sha(p) for p in directory.rglob('*') if p.is_file() and p!=directory/'COMPLETED.json'};write(directory/'COMPLETED.json',value)
    complete(raw,olddone)
    write(new/'CHECKPOINTS_LOCKED.json',newlocks)
    write(new/'run_manifest.json',{'controls':{'raw_completion':{'sha256':g.e.sha(raw/'COMPLETED.json')}}})
    complete(new,dict(status='complete',seeds=newseeds,paired=dict(per_seed=deltas,hit5={'mean_delta':.5},ndcg5={'mean_delta':.25})))
    oldgate=root/'old_verified.json';write(oldgate,dict(status='verified',formal_archive_verified=False,completion_sha256=g.e.sha(raw/'COMPLETED.json')))
    gate=root/'new_verified.json';write(gate,dict(status='verified',formal_archive_verified=False,run=dict(completion_sha256=g.e.sha(new/'COMPLETED.json'),raw_archive_receipt_sha256=g.e.sha(oldgate))))
    return SimpleNamespace(**vars(args),run_dir=str(new),verification_receipt=str(gate),raw_verification_receipt=str(oldgate))


class GroupTests(unittest.TestCase):
    def test_four_disjoint_groups_recover_known_unequal_weights(self):
        old=dict(uid=np.arange(7),hit5=np.array([0,1,0,1,0,1,0]),ndcg5=np.array([0,.5,0,.5,0,.5,0]),pool_hit=np.ones(7))
        new=dict(uid=np.arange(7),hit5=np.array([1,0,1,1,1,0,1]),ndcg5=np.array([.5,0,.5,.5,.5,0,.5]),pool_hit=np.ones(7))
        membership=np.array([0,1,1,2,2,2,3]);result=g.grouped(old,new,{str(k):membership==k for k in range(4)})
        self.assertEqual([result['groups'][str(k)]['users'] for k in range(4)],[1,2,3,1])
        self.assertAlmostEqual(result['weighted_group_delta']['hit5'],2/7)
        self.assertAlmostEqual(result['weighted_group_delta']['ndcg5'],1/7)

    def test_sealed_synthetic_pipeline_empty_and_weighted_reconstruction(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root);before={str(p):g.e.sha(p) for p in root.rglob('*') if p.is_file()}
            result=g.analyze(args)
            self.assertEqual(result['overall_three_seed_confirm_delta'],dict(hit5=.5,ndcg5=.25))
            for seed,value in result['seeds'].items():
                self.assertEqual(len(value['screen']),3)
                c=value['confirm'];self.assertEqual(c['groups']['only_old_seen']['users'],0);self.assertIsNone(c['groups']['only_old_seen']['delta'])
                self.assertEqual(sum(v['users'] for v in c['groups'].values()),2)
                self.assertEqual(c['weighted_group_delta'],c['overall_delta'])
                self.assertEqual(value['screen'][-1]['groups']['2-3']['users'],2)
            self.assertFalse(result['selection_performed']);self.assertEqual(before,{str(p):g.e.sha(p) for p in root.rglob('*') if p.is_file()})

    def test_gate_pairing_uid_finite_and_trace_tamper_rejected(self):
        for kind in ('gate','pair','uid','finite','trace'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();args=fixture(root);new=Path(args.run_dir)
                if kind=='gate':
                    gate=g.e.read(args.verification_receipt);gate['run']['completion_sha256']='0'*64;write(Path(args.verification_receipt),gate)
                else:
                    if kind=='trace':
                        p=new/'seed_42/raw_v2_train/epoch_1_trace.json';v=g.e.read(p);v['order_sha256']='0'*64;write(p,v)
                    else:
                        p=new/'seed_42/raw_v2_train/screen_step1_users.npz'
                        with np.load(p) as f:v={k:f[k].copy() for k in f.files}
                        if kind=='pair':v['candidate_ids'][0,1]=4
                        elif kind=='uid':v['uid'][1]=v['uid'][0]
                        else:v['ndcg5'][0]=np.nan
                        np.savez_compressed(p,**v)
                    # Even a coherent re-seal cannot bypass semantic checks.
                    done=g.e.read(new/'COMPLETED.json');done['evidence_hashes'][p.relative_to(new).as_posix()]=g.e.sha(p);write(new/'COMPLETED.json',done)
                    gate=g.e.read(args.verification_receipt);gate['run']['completion_sha256']=g.e.sha(new/'COMPLETED.json');write(Path(args.verification_receipt),gate)
                with self.assertRaises(ValueError):g.analyze(args)

    def test_formal_receipt_required_without_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            args=fixture(Path(temp).resolve());args.allow_smoke=False
            with self.assertRaisesRegex(ValueError,'formal verification'):g.analyze(args)


if __name__=='__main__':unittest.main()
