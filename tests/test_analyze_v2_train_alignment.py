import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_v2_train_alignment as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False))


def ref(path):return dict(path=str(path),sha256=a.sha(path),bytes=path.stat().st_size)


def seal(root,done):
    done=dict(done,evidence_hashes={p.relative_to(root).as_posix():a.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='COMPLETED.json'})
    # Include per-seed COMPLETED, excluding only the root completion.
    done['evidence_hashes']={p.relative_to(root).as_posix():a.sha(p) for p in root.rglob('*') if p.is_file() and p!=root/'COMPLETED.json'}
    write(root/'COMPLETED.json',done);return done


def fixture(root):
    oldpool=root/'oldpool';newpool=root/'pool';oldpool.mkdir();newpool.mkdir()
    positions=np.arange(20,dtype=np.int64);np.save(root/'positions.npy',positions);np.save(newpool/'ranker_train_positions.npy',positions)
    common=dict(base_data_id='fixture',source_hashes={},encoders_hash='enc',code_hashes={},cf_neighbors=300,catalog_size=200,itemcf_half_life_days=180.,candidate_policy='RRF',positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest())
    metas=[]
    for directory,new in ((oldpool,False),(newpool,True)):
        items=np.tile(np.arange(2,77,dtype=np.int32),(20,1));lengths=np.full(20,75,dtype=np.int32)
        if new:items[0,0]=100;items[1,[10,25]]=items[1,[25,10]]
        np.save(directory/'items.npy',items);np.save(directory/'lengths.npy',lengths)
        meta=dict(status='complete',schema='cross-pools-int32-v2',shape=[20,75],records_hash='records',source=dict(common,rrf_weights=[2,1 if new else 0,.7,.05]),pool_hash=hashlib.sha256(json.dumps(items.tolist(),separators=(',',':')).encode()).hexdigest(),files={key:dict(name=key+'.npy',sha256=a.sha(directory/(key+'.npy'))) for key in ('items','lengths')})
        write(directory/'candidate.json',meta);metas.append(meta)
    pc=seal(newpool,dict(status='complete',mode='full',rows=20,source=metas[1]['source'],pool_hash=metas[1]['pool_hash'],cache=ref(newpool/'candidate.json'),positions=ref(newpool/'ranker_train_positions.npy')))
    protocol=root/'protocol.json';write(protocol,dict(candidate_caches={'train':ref(oldpool/'candidate.json')}))
    paired=dict(criteria={'frozen':True},selection='raw_retained',bootstrap_replicates=10000)
    for directory,variant in ((root/'raw','raw'),(root/'new','raw_v2_train')):
        seeds={}
        for seed in a.SEEDS:
            base=directory/f'seed_{seed}/{variant}';base.mkdir(parents=True)
            steps=4;history=[dict(step=s,epoch_fraction=s/4,screen={'hr5':.1,'ndcg5':.05},diagnostics={'train':{'bpr_loss':.5},'screen':{'pair_auc':.8}},train_seconds_cumulative=float(s),eval_seconds_cumulative=.1*s) for s in (1,2,4)]
            write(base/'history.json',history);write(base/'epoch_1_trace.json',dict(rows=20,batch_sizes=[5]*4,negative_sources=dict(band11_25=40,band26_50=40,candidate_fallback=0,random=240)))
            (base/'steps.jsonl').write_text('\n'.join(json.dumps(dict(step=s,epoch=1,loss=.5,gradient_l2_before_clip=6 if s==2 else 2)) for s in range(1,5))+'\n')
            result=dict(steps_per_epoch=steps,best={'step':1},train_seconds=4.,eval_seconds=.4);write(base/'COMPLETED.json',result);seeds[seed]=result
        write(directory/'run_config.json',{'train_positions_sha256':common['positions_sha256']})
        if variant=='raw':rawdone=seal(directory,dict(status='complete',seeds=seeds,smoke=True,test_future_labels_read=False))
        else:
            write(directory/'paired_confirm.json',paired)
            write(directory/'run_manifest.json',dict(controls={'raw_completion':ref(root/'raw/COMPLETED.json')},train_cache=dict(receipt=ref(newpool/'COMPLETED.json'),source=pc['source'],pool_hash=pc['pool_hash']),protocol_manifest=ref(protocol)))
            seal(directory,dict(status='complete',seeds=seeds,smoke=True,test_future_labels_read=False,paired=paired))
    return SimpleNamespace(run_dir=str(root/'new'),raw_run=str(root/'raw'),pool_dir=str(newpool),old_pool=str(oldpool/'candidate.json'),old_positions=str(root/'positions.npy'),protocol_manifest=str(protocol),allow_smoke=True)


class AnalysisTests(unittest.TestCase):
    def test_full_readonly_summary_and_expected_pool_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root);before={str(p):a.sha(p) for p in root.rglob('*') if p.is_file()}
            result=a.analyze(args)
            self.assertEqual(result['pool']['rows'],20);self.assertEqual(result['pool']['identical_order_rows'],18);self.assertEqual(result['pool']['identical_set_rows'],19)
            self.assertEqual(result['pool']['bands']['rank11_25']['member_change_fraction'],.05)
            self.assertEqual(result['pool']['overlap_count']['quantiles']['min'],74)
            self.assertAlmostEqual(result['pool']['jaccard']['quantiles']['min'],74/76)
            self.assertEqual(result['pool']['added_count']['quantiles']['max'],1)
            for seed in a.SEEDS:self.assertEqual(result['new']['per_seed_training'][seed]['epoch1_clip_needed_steps'],1)
            self.assertEqual(result['new']['per_seed_training']['42']['negative_source_fraction']['random'],.75)
            self.assertEqual(len(result['new']['three_seed_point_means']),3)
            self.assertEqual(result['registered_decision_unchanged'],a.read(root/'new/paired_confirm.json'))
            self.assertFalse(result['bootstrap_performed']);self.assertFalse(result['training_or_scoring_performed'])
            self.assertEqual(before,{str(p):a.sha(p) for p in root.rglob('*') if p.is_file()})

    def test_incomplete_seed_positions_and_pool_corruption_fail(self):
        for kind in ('seed','positions','pool','loss'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();args=fixture(root)
                if kind=='seed':
                    done=a.read(root/'new/COMPLETED.json');del done['seeds']['44'];write(root/'new/COMPLETED.json',done)
                elif kind=='positions':np.save(root/'positions.npy',np.arange(1,21,dtype=np.int64))
                elif kind=='pool':
                    with (root/'oldpool/items.npy').open('ab') as stream:stream.write(b'changed')
                else:
                    path=root/'new/seed_42/raw_v2_train/steps.jsonl';path.write_text(path.read_text().replace('0.5','NaN',1));seal(root/'new',a.read(root/'new/COMPLETED.json'))
                with self.assertRaises(ValueError):a.analyze(args)


if __name__=='__main__':unittest.main()
