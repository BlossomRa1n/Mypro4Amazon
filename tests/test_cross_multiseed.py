"""Small protocol guards for the registered cross runner."""

import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from future_window_data import FutureWindowData
from run_cross_multiseed import (load_historical_users, prepare,
                                 sha256_file)
from token_models import SemanticTokenDIN


def _month_scope():
    return dict(schema_version=1,kind='month_to_freeze',timezone='Asia/Shanghai',
                start_inclusive='2026-09-01T00:00:00+08:00',end_exclusive='2026-09-24T01:54:44+08:00',
                exposure_policy='evaluation_or_selection',training_prefix_allowed=True)


def _fixture(root: Path):
    rows = []
    for user in range(14):
        for tick in range(8):
            rows.append((f"u{user:03}", f"i{user:03}_{tick}", 4.0,
                         1_700_000_000_000 + tick * 86_400_000, "fixture"))
    frame = pd.DataFrame(rows, columns=("user_id", "parent_asin", "rating", "timestamp", "category"))
    data = FutureWindowData(frame, hist_len=5, test_users=2)
    base = root / "base"; base.mkdir()
    with (base / "data.pkl").open("wb") as stream:
        pickle.dump({"data": data}, stream, protocol=5)
    (base / "run_manifest.json").write_text(json.dumps({"args": {"dim": 8, "hist_len": 5}}))
    source = root / "history-source.txt"; source.write_text("fixture historical source")
    history = root / "historical.json"
    import run_cross_multiseed as r
    scope=_month_scope()
    history.write_text(json.dumps({
        "history_scope":scope,"history_scope_sha256":r.sha256_json(scope),
        "schema_version": 1, "raw_user_ids": [f"u{i:03}" for i in range(12)],
        "sources": [{"path": str(source), "sha256": sha256_file(source), "mode": "fixture"}],
        "completeness": {"attested": True, "note": "fixture source"},
    }))
    return base, history


class CrossProtocolTests(unittest.TestCase):
    def test_prepare_uses_raw_ids_and_exact_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, historical = _fixture(root)
            args = type("Args", (), {"base_run": str(base), "run_dir": str(root / "protocol"),
                "historical_users": str(historical), "train_users": 4, "screen_users": 2,
                "confirm_users": 2, "test_users": 2, "cohort_seed": 20260922, "smoke": True})()
            manifest = prepare(args)
            self.assertEqual(manifest["counts"], {"train": 4, "screen": 2, "confirm": 2, "test": 2})
            screen = json.loads((root / "protocol" / "screen_records.json").read_text())
            confirm = json.loads((root / "protocol" / "confirm_records.json").read_text())
            test = json.loads((root / "protocol" / "test_records.json").read_text())
            self.assertTrue({row[2] for row in screen}.isdisjoint({row[2] for row in confirm}))
            self.assertTrue({row[2] for row in test}.isdisjoint({row[2] for row in screen + confirm}))
            self.assertFalse(manifest["test_future_labels_read"])

    def test_unverifiable_history_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps({"schema_version": 1, "raw_user_ids": ["u"],
                                        "sources": [], "completeness": {"attested": True, "note": "x"}}))
            with self.assertRaises(ValueError):
                load_historical_users(path)

    def test_zero_cross_keeps_shape_and_zeroes_complete_slot(self):
        model = SemanticTokenDIN(num_users=4, num_items=8, num_brands=4,
                                 num_categories=4, embed_dim=8, token_dim=8,
                                 hist_len=5, cross_mode="zero_cross")
        self.assertEqual(model.cross_gate_logit.numel(), 1)
        self.assertEqual(model.TOKEN_COUNT, 6)


if __name__ == "__main__":
    unittest.main()

class DynamicFreshTests(unittest.TestCase):
    def args(self, base, history, run):
        from types import SimpleNamespace
        return SimpleNamespace(base_run=str(base),run_dir=str(run),historical_users=str(history),
                               train_users=4,screen_users=2,confirm_users=2,cohort_seed=20260922,
                               test_mode='all_fresh',smoke=True)

    def test_all_fresh_dynamic_one_and_three(self):
        import run_cross_multiseed as r
        for historic_count in (11,13):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base,hist=_fixture(root)
                obj=json.loads(hist.read_text());obj['raw_user_ids']=[f'u{i:03}' for i in range(historic_count)];hist.write_text(json.dumps(obj))
                run=root/'p';m=prepare(self.args(base,hist,run))
                self.assertEqual(m['counts']['test'],14-historic_count)
                self.assertEqual({x[2] for x in r._records(run,m,'test')},{f'u{i:03}' for i in range(historic_count,14)})
                self.assertEqual(r.load_protocol(run/'protocol_manifest.json'),m)
                self.assertTrue({x[2] for x in r._records(run,m,'train')}.issubset(obj['raw_user_ids']))

    def test_empty_fresh_and_insufficient_history_rejected(self):
        for historic_count in (0,3,14):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base,hist=_fixture(root)
                obj=json.loads(hist.read_text());obj['raw_user_ids']=[f'u{i:03}' for i in range(historic_count)];hist.write_text(json.dumps(obj))
                with self.assertRaisesRegex(ValueError,'insufficient'):prepare(self.args(base,hist,root/'p'))

    def test_rehashed_subset_identity_and_v1_rejected(self):
        import run_cross_multiseed as r
        for corruption in ('subset','mapping','raw_hash','v1','v2','smoke_type','mode','scope'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base,hist=_fixture(root);run=root/'p';m=prepare(self.args(base,hist,run))
                if corruption in ('subset','mapping'):
                    rows=json.loads((run/'test_records.json').read_text())
                    if corruption=='subset':rows=rows[:1]
                    else:rows[0][2]='fake-raw'
                    (run/'test_records.json').write_text(json.dumps(rows))
                    m['counts']['test']=len(rows);m['cohort_hashes']['test']=r._record_hash(rows)
                    m['cohort_raw_hashes']['test']=r.sha256_json(sorted(x[2] for x in rows))
                    m['eligible_fresh_count']=len(rows);m['eligible_fresh_records_hash']=r._record_hash(rows)
                elif corruption=='raw_hash':m['cohort_raw_hashes']['test']='tampered'
                elif corruption=='v1':m['protocol']='next-cross-multiseed-20260922-v1'
                elif corruption=='v2':m['protocol']='next-cross-multiseed-20260924-v2'
                elif corruption=='scope':m['history_scope']=dict(m['history_scope'],end_exclusive='2026-09-21T12:00:00+08:00');m['history_scope_sha256']=r.sha256_json(m['history_scope'])
                elif corruption=='smoke_type':m['smoke']='true'
                else:m['test_mode']='sampled'
                m.pop('manifest_hash');m['manifest_hash']=r.sha256_json(m)
                (run/'protocol_manifest.json').write_text(json.dumps(m))
                with self.assertRaises(ValueError):r.load_protocol(run/'protocol_manifest.json')

    def test_false_attestation_and_unresolved_fail(self):
        for field in ('attested','unresolved_sources'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);_,hist=_fixture(root);obj=json.loads(hist.read_text())
                obj['completeness'][field]=False if field=='attested' else ['source-without-identity']
                hist.write_text(json.dumps(obj))
                with self.assertRaises(ValueError):load_historical_users(hist)

    def test_formal_counts_and_parser_have_no_requested_test_n(self):
        import run_cross_multiseed as r
        from types import SimpleNamespace
        r._cohort_sizes(SimpleNamespace(train_users=100000,screen_users=20000,confirm_users=80000,test_mode='all_fresh'),False)
        for bad in (dict(train=99999,screen=20000,confirm=79999),dict(train=100000,screen=30000,confirm=70000)):
            with self.assertRaises(ValueError):r._cohort_sizes(SimpleNamespace(**{k+'_users':v for k,v in bad.items()},test_mode='all_fresh'),False)
        args=r.parser().parse_args(['prepare','--base-run','b','--run-dir','r','--historical-users','h','--test-mode','all_fresh'])
        self.assertFalse(hasattr(args,'test_users'))

class HistoricalClosureTests(unittest.TestCase):
    def test_formal_requires_and_verifies_closure_receipt(self):
        import run_cross_multiseed as r
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,history=_fixture(root)
            with self.assertRaisesRegex(ValueError,'closure_receipt'):load_historical_users(history,require_closure=True)
            obj=json.loads(history.read_text());registry=root/'source-registry.json';registry.write_text(json.dumps({'history_scope':obj['history_scope'],'history_scope_sha256':obj['history_scope_sha256'],'sources':[dict(x,status='recovered') for x in obj['sources']]}))
            receipt=root/'closure.json';proof=dict(schema_version=1,status='complete',scope_complete=True,
                unresolved_sources=[],auditor='synthetic-test',completed_at_utc='2026-09-24T00:00:00Z',
                scope=obj['history_scope'],history_scope_sha256=obj['history_scope_sha256'],raw_union_count=len(obj['raw_user_ids']),source_count=len(obj['sources']),raw_union_sha256=r.sha256_json(sorted(obj['raw_user_ids'])),
                source_registry={'path':registry.name,'sha256':sha256_file(registry)},source_registry_sha256=sha256_file(registry))
            def publish(value):
                receipt.write_text(json.dumps(value));obj['completeness']['closure_receipt']={'path':receipt.name,'sha256':sha256_file(receipt)};history.write_text(json.dumps(obj))
            publish(proof)
            ids,_=load_historical_users(history,require_closure=True);self.assertEqual(ids,set(obj['raw_user_ids']))
            for key,value in [('status','partial'),('scope_complete',False),('unresolved_sources',['gap']),('raw_union_sha256','wrong'),('source_registry_sha256','wrong'),('source_count',999),('raw_union_count',999),('scope',dict(obj['history_scope'],end_exclusive='2026-09-21T12:00:00+08:00'))]:
                publish(dict(proof,**{key:value}))
                with self.assertRaises(ValueError):load_historical_users(history,require_closure=True)
            publish(proof)
            registry.write_text(json.dumps({'history_scope':obj['history_scope'],'history_scope_sha256':obj['history_scope_sha256'],'sources':[dict(obj['sources'][0],status='unresolved')]}))
            changed=dict(proof,source_registry={'path':registry.name,'sha256':sha256_file(registry)},source_registry_sha256=sha256_file(registry))
            publish(changed)
            with self.assertRaisesRegex(ValueError,'unresolved source status'):load_historical_users(history,require_closure=True)
            registry.write_text(json.dumps({'history_scope':dict(obj['history_scope'],end_exclusive='2026-09-21T12:00:00+08:00'),'history_scope_sha256':obj['history_scope_sha256'],'sources':[dict(x,status='recovered') for x in obj['sources']]}))
            changed=dict(proof,source_registry={'path':registry.name,'sha256':sha256_file(registry)},source_registry_sha256=sha256_file(registry))
            publish(changed)
            with self.assertRaisesRegex(ValueError,'registry history scope'):load_historical_users(history,require_closure=True)
            publish(proof);registry.write_text('changed')
            with self.assertRaisesRegex(ValueError,'registry hash'):load_historical_users(history,require_closure=True)


class MonthScopeTests(unittest.TestCase):
    def test_month_scope_has_no_old_allhistory_fresh_bound(self):
        import run_cross_multiseed as r
        m=dict(smoke=False,cohort_seed=r.COHORT_SEED,test_mode='all_fresh',counts=dict(r.FORMAL_COUNTS,test=30000))
        r._validate_counts(m)

    def test_scope_rejects_old_month_naive_future_and_changed_policy(self):
        import run_cross_multiseed as r
        scope=_month_scope()
        self.assertEqual(r._validate_history_scope(scope),r.sha256_json(scope))
        for key,value in [('start_inclusive','2026-08-01T00:00:00+08:00'),('end_exclusive','2026-09-20T12:00:00'),('end_exclusive','2099-09-20T12:00:00+08:00'),('exposure_policy','all_training'),('training_prefix_allowed',False),('kind','all_history')]:
            with self.assertRaises(ValueError):r._validate_history_scope(dict(scope,**{key:value}))

    def test_scope_binding_rejects_change_even_with_recomputed_hash(self):
        import run_cross_multiseed as r
        a={'history_scope':_month_scope(),'history_scope_sha256':r.sha256_json(_month_scope()),'historical_closure_sha256':'proof'}
        b=dict(a);b['history_scope']=dict(a['history_scope'],end_exclusive='2026-09-21T12:00:00+08:00');b['history_scope_sha256']=r.sha256_json(b['history_scope'])
        with self.assertRaises(ValueError):r._validate_scope_binding(a,b)
