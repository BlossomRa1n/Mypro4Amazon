"""Adversarial independent V2 archive checks; only small synthetic artifacts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_v2_verifier', ROOT/'tools/verify_v2_train_alignment.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def make_cache(root):
    items = np.zeros((3,75), dtype='<i4'); items[:, :3] = [[2,3,4],[3,5,8],[6,7,9]]
    lengths = np.full(3,3,dtype='<i4')
    np.save(root/'items.npy', items); np.save(root/'lengths.npy', lengths)
    source = dict(records_hash='a'*64, records_count=3, budget=75, catalog_size=10)
    logical = hashlib.sha256(json.dumps(items[:,:3].tolist(), separators=(',', ':')).encode()).hexdigest()
    meta = dict(schema='cross-pools-int32-v2', status='complete', dtype='<i4', lengths_dtype='<i4', budget=75,
                block_rows=1024, source=source, shape=[3,75], records_hash='a'*64, position_uid_row_hash='a'*64,
                pool_hash=logical, files={k:dict(name=k+'.npy',sha256=v.sha(root/(k+'.npy'))) for k in ('items','lengths')})
    write(root/'cache.json',meta)
    files={p.name:dict(sha256=v.sha(p),bytes=p.stat().st_size) for p in root.iterdir()}
    return meta,files


def paired_fixture():
    differences={'hit5':[np.array([1.,0.,-1.,0.]),np.array([1.,1.,0.,0.]),np.array([1.,0.,0.,0.])],
                 'ndcg5':[np.array([.3,0.,-.3,0.]),np.array([.3,.3,0.,0.]),np.array([.3,0.,0.,0.])]}
    sources={str(seed):dict(raw_arrays_sha256='a'*64,new_arrays_sha256='b'*64) for seed in (42,43,44)}
    result=dict(protocol=v.VERSION,final_test_allowed=False,full_training_allowed=False,per_seed={},array_sources=sources,
                bootstrap_seed=20260925,bootstrap_replicates=10000,ci_level=.975,quantiles=[.0125,.9875])
    for key,values in differences.items():
        average=np.mean(values,axis=0);rng=np.random.default_rng(20260925)
        means=np.array([average[rng.integers(0,len(average),size=len(average))].mean() for _ in range(10000)])
        result[key]=dict(mean_delta=float(average.mean()),users=len(average),gained=int((average>0).sum()),lost=int((average<0).sum()),ci975=np.quantile(means,[.0125,.9875]).tolist())
        for seed,row in zip(v.SEEDS,values):
            item=result['per_seed'].setdefault(seed,{})
            item[key+'_delta']=float(row.mean())
            if key=='hit5':item.update(gained=int((row>0).sum()),lost=int((row<0).sum()))
    result['criteria']=dict(all_seed_hr_positive=False,mean_hr_at_least_0_0005=True,hr_ci975_lower_positive=False,mean_ndcg_nonnegative=True)
    result.update(selection='raw_retained',supports_v2_train_pool_followup=False)
    return differences,result,sources


class IndependentV2Tests(unittest.TestCase):
    def test_cache_valid_and_resealed_padding_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();meta,files=make_cache(root)
            self.assertEqual(v.verify_cache_payload(root,root/'cache.json',files)['pool_hash'],meta['pool_hash'])
            items=np.load(root/'items.npy');items[0,4]=9;np.save(root/'items.npy',items)
            meta['files']['items']['sha256']=v.sha(root/'items.npy');write(root/'cache.json',meta)
            files['items.npy']['sha256']=v.sha(root/'items.npy')
            with self.assertRaisesRegex(ValueError,'IDs/duplicates/padding'):
                v.verify_cache_payload(root,root/'cache.json',files)

    def test_logical_hash_and_duplicate_rows_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();meta,files=make_cache(root)
            meta['pool_hash']='b'*64;write(root/'cache.json',meta)
            with self.assertRaisesRegex(ValueError,'logical pool hash'):
                v.verify_cache_payload(root,root/'cache.json',files)
            items=np.load(root/'items.npy');items[0,1]=items[0,0];np.save(root/'items.npy',items)
            meta['files']['items']['sha256']=v.sha(root/'items.npy');write(root/'cache.json',meta)
            files['items.npy']['sha256']=v.sha(root/'items.npy')
            with self.assertRaisesRegex(ValueError,'IDs/duplicates/padding'):
                v.verify_cache_payload(root,root/'cache.json',files)

    def test_disk_reserve_budget_and_inode_gates(self):
        gate=dict(path='/tmp/new',filesystem_device=1,free_bytes=9000,remaining_bytes=5000,reserve_bytes=3000,free_inodes=16)
        v.disk_check(gate,3000)
        for key,value in [('free_bytes',7999),('reserve_bytes',2999),('free_inodes',15),('remaining_bytes',-1)]:
            changed=dict(gate,**{key:value})
            with self.assertRaises(ValueError):v.disk_check(changed,3000)

    def test_source_mapping_needs_bytes_hash_and_safe_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();path=root/'source.py';path.write_text('x=1\n')
            ref=dict(path='/remote/source.py',sha256=v.sha(path),bytes=path.stat().st_size)
            v.bound_file(ref,path,expected_name='source.py')
            with self.assertRaisesRegex(ValueError,'byte count'):
                v.bound_file(dict(ref,bytes=0),path)
            with self.assertRaisesRegex(ValueError,'unsafe source'):
                v.bound_file(dict(ref,path='/remote/../source.py'),path)
            link=root/'link.py';link.symlink_to(path)
            with self.assertRaisesRegex(ValueError,'symlink'):
                v.bound_file(ref,link)

    def test_evidence_exact_set_and_failed_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();write(root/'x.json',dict(x=1));complete=dict(evidence_hashes={'x.json':v.sha(root/'x.json')});write(root/'COMPLETED.json',complete)
            v.sealed_tree(v.safe_root(root),complete)
            (root/'extra.txt').write_text('extra')
            with self.assertRaisesRegex(ValueError,'unsealed'):
                v.sealed_tree(root,complete)
            write(root/'FAILED.json',{})
            with self.assertRaisesRegex(ValueError,'failed marker'):
                v.safe_root(root)

    def test_strict_json_rejects_duplicates_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'bad.json'
            for value in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e400}'):
                path.write_text(value)
                with self.assertRaises(ValueError):v.read_json(path)

    def test_pairing_includes_auc_valid_mask(self):
        left={key:np.array([1,2]) for key in v.STABLE};right={k:a.copy() for k,a in left.items()}
        v.paired_equal(left,right)
        right['auc_valid'][0]=3
        with self.assertRaisesRegex(ValueError,'paired identities'):
            v.paired_equal(left,right)

    def test_statistics_recomputed_and_adaptive_settings_rejected(self):
        differences,paired,sources=paired_fixture();v.recompute_paired(differences,paired,sources)
        for key,value,message in [('bootstrap_seed',2,'registration'),('supports_v2_train_pool_followup',True,'followup decision'),('selection','v2_train_pool','selection result')]:
            changed=copy.deepcopy(paired);changed[key]=value
            with self.assertRaisesRegex(ValueError,message):v.recompute_paired(differences,changed,sources)
        changed=copy.deepcopy(paired);changed['hit5']['ci975'][0]+=.1
        with self.assertRaisesRegex(ValueError,'independently recomputed CI975'):
            v.recompute_paired(differences,changed,sources)

    def test_actual_new_run_and_resealed_semantic_corruptions(self):
        # Use the separately implemented small training integration as an input
        # producer; the verifier never imports or calls it in production.
        from tests import test_v2_train_alignment as producer
        original_run=producer.v.run; receipts=[]
        def intercept(args):
            result=original_run(args)
            run=Path(args.run_dir);root=run.parent
            if run.name!='new':return result
            context=dict(smoke=True,protocol=v.read_json(args.protocol_manifest),protocol_path=Path(args.protocol_manifest),
                         document=Path(args.protocol_doc),positions=np.load(root/'pool/ranker_train_positions.npy'))
            pool=dict(root=root/'pool',complete=v.read_json(root/'pool/COMPLETED.json'),meta=v.read_json(root/'pool/candidate_train_v2.json'))
            config=SimpleNamespace(run_dir=run,raw_run=Path(args.raw_run),raw_archive_receipt=None,backup_receipt=None,
                                   runner_source=ROOT/'code/run_v2_train_alignment.py',builder_source=ROOT/'code/build_v2_train_pool.py')
            receipt=v.verify_run(config,context,pool);receipts.append(receipt)
            self.assertEqual(receipt['counts'],dict(checkpoints=3,screen_arrays=9,confirm_arrays=3))
            self.assertFalse(receipt['formal_archive_verified'])
            self.assertTrue(receipt['bootstrap_independently_recomputed'])
            completion=run/'COMPLETED.json';original_complete=completion.read_bytes()
            def mutate(relative,change,message,paired=False,seed=False):
                path=run/relative;before=path.read_bytes();value=json.loads(before);change(value);write(path,value)
                complete=json.loads(original_complete);complete['evidence_hashes'][relative]=v.sha(path)
                if paired:complete['paired']=value
                if seed:complete['seeds']['42']=value
                write(completion,complete)
                try:
                    with self.assertRaisesRegex(ValueError,message):v.verify_run(config,context,pool)
                finally:path.write_bytes(before);completion.write_bytes(original_complete)
            mutate('paired_confirm.json',lambda value:value['hit5']['ci975'].__setitem__(0,value['hit5']['ci975'][0]+.05),'independently recomputed CI975',paired=True)
            mutate('paired_confirm.json',lambda value:value.__setitem__('supports_v2_train_pool_followup',not value['supports_v2_train_pool_followup']),'followup decision',paired=True)
            mutate('seed_42/raw_v2_train/initial_state_audit.json',lambda value:value.__setitem__('all_parameters_and_buffers_equal_to_raw',False),'initial raw equality')
            mutate('run_manifest.json',lambda value:value['config'].__setitem__('epochs',2),'epoch/scheduler')
            mutate('seed_42/raw_v2_train/history.json',lambda value:value[0].__setitem__('selected_best_step',999),'sequential CPU best')
            mutate('resource_snapshots.json',lambda value:value[1].__setitem__('remaining_models',1),'system free/remaining/inode')
            # Resign the run after changing labels/UIDs; pairing must still fail.
            path=run/'seed_42/raw_v2_train/confirm_best_users.npz';before=path.read_bytes()
            with np.load(path,allow_pickle=False) as archive:values={key:archive[key].copy() for key in archive.files}
            values['position'][0]+=1;np.savez_compressed(path,**values)
            complete=json.loads(original_complete);complete['evidence_hashes'][str(path.relative_to(run))]=v.sha(path);write(completion,complete)
            try:
                with self.assertRaisesRegex(ValueError,'paired identities'):v.verify_run(config,context,pool)
            finally:path.write_bytes(before);completion.write_bytes(original_complete)
            return result
        with patch.object(producer.v,'run',side_effect=intercept):
            producer.AlignmentTests('test_real_three_seed_one_epoch_new_pool_pipeline').test_real_three_seed_one_epoch_new_pool_pipeline()
        self.assertEqual(len(receipts),1)


if __name__=='__main__':unittest.main()
