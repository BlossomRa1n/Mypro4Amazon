import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cross_pool_cache import build_cache, load_cache, rows_hash, pool_hash, PositionRecords
import run_cross_multiseed as r
from tests.test_cross_multiseed import _fixture


class PoolCacheTests(unittest.TestCase):
    def test_streaming_hash_exact_canonical_json(self):
        rows=[(2,3,'中\\文'),(3,5,'x')]
        self.assertEqual(rows_hash(rows),r.sha256_json([list(x) for x in rows]))
        pools=[[],[2,5],[9]]
        self.assertEqual(pool_hash(pools),r.sha256_json(pools))

    def test_blocks_preserve_ragged_and_cache_seal(self):
        rows=[(2,3,'u'),(2,4,'u'),(3,6,'v'),(4,9,'w')]
        expected=[[],[2],[9,3,4],[5,6]]
        source={'records_hash':rows_hash(rows),'records_count':len(rows),'budget':3,'asset':'original'}
        for block in (1,3,10):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'pool.json'
                builder=lambda rs:[expected[rows.index(tuple(row))] for row in rs]
                pools,meta=build_cache(path,rows,3,source,builder,block_rows=block)
                self.assertEqual([x.tolist() for x in pools],expected)
                self.assertEqual(meta['pool_hash'],pool_hash(expected))
                self.assertIsInstance(pools.items,np.memmap)
                self.assertFalse(pools.items.flags.writeable)
                with self.assertRaises(ValueError):load_cache(path,dict(source,asset='changed'))
                altered=dict(meta);altered['shape']=[len(rows)-1,3];path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'row count'):load_cache(path,source)
                path.write_text(json.dumps(meta))
                payload=path.parent/meta['files']['items']['name']
                with payload.open('r+b') as f:f.seek(-1,2);f.write(b'\x7f')
                with self.assertRaisesRegex(ValueError,'hash'):load_cache(path,source)

    def test_incomplete_and_exclusive_builder_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pool.json';records=[(2,3,'u')];source={'records_hash':rows_hash(records)}
            def fail(rows):raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):build_cache(path,records,3,source,fail)
            self.assertFalse(path.exists())
            with self.assertRaises(FileExistsError):build_cache(path,records,3,source,lambda rs:[[2]])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pool.json';path.with_suffix('.json.building').write_text('owned')
            with self.assertRaises(FileExistsError):build_cache(path,records,3,source,lambda rs:[[2]])

    def test_lazy_records_and_memmap_sampler_equal_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,_=_fixture(root);data=r.load_data(base)
            records=PositionRecords(data,data.train_positions)
            expected=[(int(data.uid[p]),int(p),str(data.users[int(data.uid[p])])) for p in data.train_positions]
            self.assertEqual(list(records),expected)
            builder=r.RecallPoolBuilder(data,base,75,[2,0,.7,.05],True)
            old=r._build_recall_pools(data,base,records,75,[2,0,.7,.05],True)
            pools,_=build_cache(root/'p.json',records,75,{'records_hash':rows_hash(records)},builder,block_rows=3)
            self.assertEqual([x.tolist() for x in pools],old)
            ds1=r.TracedPrefixDataset(data,16,'mixed_rrf',old)
            ds2=r.TracedPrefixDataset(data,16,'mixed_rrf',pools)
            for epoch in (0,1,2):
                ds1.epoch=ds2.epoch=epoch
                for i in range(len(records)):
                    np.testing.assert_array_equal(ds1[i]['neg_item_id'],ds2[i]['neg_item_id'])
                    self.assertEqual(ds1.last_sources.pop(i),ds2.last_sources.pop(i))

    def test_formal_builder_matches_frozen_baseline_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,_=_fixture(root);data=r.load_data(base)
            records=PositionRecords(data,data.train_positions[:11])
            neighbors=np.tile(np.arange(2,len(data.items)),(len(data.items),1))
            similarity=np.ones_like(neighbors,dtype=np.float32)
            # Construct the context without requiring a fake production checkpoint.
            builder=object.__new__(r.RecallPoolBuilder)
            builder.data=data;builder.budget=75;builder.weights=[2,0,.7,.05];builder.smoke=False;builder.v2=None
            builder.neighbors=neighbors;builder.similarity=similarity
            builder.hot=[int(i) for i in np.argsort(-data.train_counts,kind='stable') if i>=2 and data.train_counts[i]>0]
            builder.category_hot={cat:[i for i in builder.hot if data.item_category[i]==cat] for cat in range(2,len(data.categories))}
            expected,_=r.candidate_pools(data,[(u,p) for u,p,_ in records],[{} for _ in records],(neighbors,similarity),75,fusion_mode='rrf',half_life_days=180.,fusion_weights=builder.weights)
            for chunk in (1,3,20):
                actual=[]
                for start in range(0,len(records),chunk):actual.extend(builder(records[start:start+chunk]))
                self.assertEqual(actual,[list(x) for x in expected])

    def test_lazy_batch_chunks_keep_singleton_merge(self):
        for n in (1,2,8,9,10,16,17):
            old=[np.arange(n)[i:i+8].tolist() for i in range(0,n,8)]
            if len(old)>1 and len(old[-1])==1:old[-2].extend(old.pop())
            self.assertEqual(list(r._batch_chunks(np.arange(n),8)),old)

    def test_v2_crosses_1024_boundary_with_original_fp32_order(self):
        import pickle
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,_=_fixture(root);data=r.load_data(base)
            records=(list(PositionRecords(data,data.train_positions))*16)[:1031]
            rng=np.random.default_rng(924)
            neighbors=rng.integers(2,len(data.items),size=(len(data.items),300),dtype=np.int32)
            similarity=rng.random(neighbors.shape,dtype=np.float32)
            with (base/'itemcf.pkl').open('wb') as f:pickle.dump((neighbors,similarity),f)
            r.torch.set_num_threads(1);r.seed_all(42)
            v2=r.make_model(data,SimpleNamespace(dim=8,hist_len=5),'v2',r.torch.device('cpu'))
            r.torch.save({'model':v2.state_dict()},base/'v2_best.pth')
            weights=[2.,1.,.7,.05]
            with patch.object(r.torch.cuda,'is_available',return_value=False):
                old=r._build_recall_pools(data,base,records,75,weights,False)
                builder=r.RecallPoolBuilder(data,base,75,weights,False)
                actual,meta=build_cache(root/'p.json',records,75,
                    dict(records_hash=rows_hash(records),budget=75,catalog_size=len(data.items)),builder)
            self.assertEqual([row.tolist() for row in actual],old)
            self.assertEqual(meta['pool_hash'],pool_hash(old))
