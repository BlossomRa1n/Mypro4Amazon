import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_training_exposure as a


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(root):
    protocol=root/'protocol';protocol.mkdir();raw=root/'raw';raw.mkdir()
    rows=[[5,9,'u5'],[2,19,'u2'],[8,29,'u8'],[1,39,'u1']]
    records=dict(train=[rows[i] for i in (2,0,3,1)],screen=[rows[2],rows[0]],confirm=[rows[3],rows[1]])
    positions=np.array([1,3,6,11,12,15,21,22,26,31,34,36],dtype=np.int64);np.save(protocol/'positions.npy',positions)
    manifest=dict(smoke=True,test_future_labels_read=False,records={k:k+'.json' for k in records},cohort_hashes={k:a.rows_hash(v) for k,v in records.items()},counts={k:len(v) for k,v in records.items()},ranker_train_positions='positions.npy',ranker_train_positions_sha256=a.array_hash(positions),ranker_train_rows=12,base_data_id='fixture',encoders_hash='enc',base_assets={})
    manifest['manifest_hash']=a.canonical(manifest);write(protocol/'protocol_manifest.json',manifest)
    for split,value in records.items():write(protocol/(split+'.json'),value)
    canonical_rows=[(rows[i//3][0],int(p),rows[i//3][2]) for i,p in enumerate(positions)];recordhash=a.rows_hash(canonical_rows)
    cache=dict(status='complete',schema='cross-pools-int32-v2',shape=[12,75],records_hash=recordhash,position_uid_row_hash=recordhash,source=dict(records_hash=recordhash,records_count=12,rrf_weights=[2,0,.7,.05],base_data_id='fixture',encoders_hash='enc',source_hashes={}))
    write(root/'candidate_train.json',cache)
    write(raw/'run_config.json',dict(batch_size=3,train_positions_sha256=a.array_hash(positions)))
    write(raw/'CHECKPOINTS_LOCKED.json',{str(s):dict(best={'step':1 if s==42 else 2}) for s in a.SEEDS})
    for seed in a.SEEDS:
        order=np.random.default_rng(seed).permutation(12).astype(np.int64)
        write(raw/f'seed_{seed}/raw/epoch_1_trace.json',dict(order_sha256=a.array_hash(order),rows=12,batch_sizes=[3]*4,covered_users=4))
    write(raw/'COMPLETED.json',dict(status='complete',test_future_labels_read=False,seeds={str(s):{} for s in a.SEEDS},evidence_hashes={p.relative_to(raw).as_posix():a.sha(p) for p in raw.rglob('*') if p.is_file()}))
    return SimpleNamespace(protocol_dir=str(protocol),old_pool=str(root/'candidate_train.json'),raw_run=str(raw),selected_steps=None,allow_smoke=True)


class ExposureTests(unittest.TestCase):
    def test_identity_reconstruction_and_exact_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=fixture(root);before={str(p):a.sha(p) for p in root.rglob('*') if p.is_file()}
            report=a.analyze(args)
            self.assertEqual(report['mapping_certification']['rows'],12)
            for seed in a.SEEDS:
                entry=report['seeds'][str(seed)];self.assertEqual([x['consumed_rows'] for x in entry['grid']],[3,6,12])
                final=entry['grid'][-1];self.assertEqual(final['unique_users'],4)
                self.assertEqual(final['cohorts']['screen']['seen_users'],2);self.assertEqual(final['cohorts']['confirm']['unseen_users'],0)
                self.assertEqual(final['cohorts']['train']['consumption_quantiles_all_users_including_zero']['min'],3)
                order=np.random.default_rng(seed).permutation(12);self.assertEqual(entry['grid'][0]['unique_users'],len(set((order[:3]//3).tolist())))
            self.assertFalse(report['labels_loaded']);self.assertEqual(before,{str(p):a.sha(p) for p in root.rglob('*') if p.is_file()})
            selected=root/'selected.json';write(selected,{str(s):4 for s in a.SEEDS});args.selected_steps=str(selected)
            self.assertTrue(all(x['selected']['unique_users']==4 for x in a.analyze(args)['seeds'].values()))

    def test_mapping_and_order_corruption_rejected_even_if_resealed(self):
        for kind in ('mapping','order','selected'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();args=fixture(root)
                if kind=='mapping':
                    cache=a.read(args.old_pool);cache['records_hash']=cache['position_uid_row_hash']=cache['source']['records_hash']='0'*64;write(Path(args.old_pool),cache)
                elif kind=='order':
                    raw=Path(args.raw_run);path=raw/'seed_42/raw/epoch_1_trace.json';trace=a.read(path);trace['order_sha256']='0'*64;write(path,trace);done=a.read(raw/'COMPLETED.json');done['evidence_hashes']['seed_42/raw/epoch_1_trace.json']=a.sha(path);write(raw/'COMPLETED.json',done)
                else:
                    selected=root/'selected.json';write(selected,{str(s):3 for s in a.SEEDS});args.selected_steps=str(selected)
                with self.assertRaises(ValueError):a.analyze(args)

    def test_resealed_duplicate_screen_or_confirm_rejected(self):
        for split in ('screen','confirm'):
            with self.subTest(split=split),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();args=fixture(root);protocol=Path(args.protocol_dir)
                path=protocol/(split+'.json');rows=a.read(path);rows.append(rows[0]);write(path,rows)
                manifest=a.read(protocol/'protocol_manifest.json');manifest['counts'][split]=len(rows);manifest['cohort_hashes'][split]=a.rows_hash(rows)
                manifest['manifest_hash']=a.canonical({k:v for k,v in manifest.items() if k!='manifest_hash'});write(protocol/'protocol_manifest.json',manifest)
                with self.assertRaisesRegex(ValueError,'duplicate cohort'):a.analyze(args)


if __name__=='__main__':unittest.main()
