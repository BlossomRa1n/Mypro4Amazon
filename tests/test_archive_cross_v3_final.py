import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "code"))
import archive_cross_v3 as common
import archive_cross_v3_final as final
from cross_pool_cache import CACHE_VERSION, pool_hash, PoolRows


def seal(value, key):
    value = dict(value); value[key] = common.json_hash(value); return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def cache(protocol, label, records_hash, rows):
    items = np.zeros((len(rows), 4), dtype="<i4"); lengths = np.zeros(len(rows), dtype="<i4")
    for index, row in enumerate(rows): items[index, :len(row)] = row; lengths[index] = len(row)
    ip = protocol / f"candidate_{label}.items.npy"; lp = protocol / f"candidate_{label}.lengths.npy"
    np.save(ip, items); np.save(lp, lengths)
    files = {"items": {"name": ip.name, "sha256": common.digest(ip)},
             "lengths": {"name": lp.name, "sha256": common.digest(lp)}}
    meta = {"schema": CACHE_VERSION, "status": "complete", "dtype": "<i4", "lengths_dtype": "<i4",
            "shape": list(items.shape), "budget": 4, "block_rows": 1024,
            "records_hash": records_hash, "position_uid_row_hash": records_hash,
            "pool_hash": pool_hash(PoolRows(items, lengths, {})),
            "source": {"records_hash": records_hash, "records_count": len(rows), "budget": 4, "catalog_size": 20},
            "files": files}
    side = protocol / f"candidate_{label}.json"; write_json(side, meta)
    (protocol / f"candidate_{label}.json.building").write_text("retained lock\n")
    return {"path": "/remote/protocol_v3/" + side.name, "sha256": common.digest(side),
            "files": files, "records_hash": records_hash, "pool_hash": meta["pool_hash"]}


def npz(path, uids, positions):
    np.savez_compressed(path, uid=np.asarray(uids, dtype="i8"), position=np.asarray(positions, dtype="i8"),
                        hit5=np.asarray([0, 1], dtype="i1"), pool_hit=np.ones(2, dtype="i1"),
                        ndcg5=np.asarray([0., 1.], dtype="f4"))


def build_fixture(root):
    remote = root / "remote"; protocol = remote / "protocol_v3"; run = remote / "final_v3"
    protocol.mkdir(parents=True); run.mkdir(); models = ["raw", "normalized_gated"]
    counts = {"train": 4, "screen": 2, "confirm": 2, "test": 2}
    records = {"train": [[1, 10, "a"]], "screen": [[1, 10, "a"], [2, 20, "b"]],
               "confirm": [[1, 10, "a"], [2, 20, "b"]], "test": [[1, 11, "a"], [2, 21, "b"]]}
    hashes = {k: common.json_hash(v) for k, v in records.items()}
    for name, value in records.items(): write_json(protocol / f"{name}_records.json", value)
    np.save(protocol / "ranker_train_positions.npy", np.asarray([1, 2], dtype="i8"))
    write_json(protocol / "historical_users_verified.json", {"status": "complete"})
    caches = {}
    for label in ("screen", "confirm", "train", "final_train", "test_final"):
        rh = hashes["test"] if label == "test_final" else hashes.get(label, hashes["train"])
        caches[label] = cache(protocol, label, rh, [[2, 3], [3, 4]] if label != "train" else [[2, 3]])
    scope = {"month": "2026-09", "policy": "fixture"}; scope_hash = common.json_hash(scope)
    positions_hash = common.digest(protocol / "ranker_train_positions.npy")
    pm = seal({"protocol": "next-cross-multiseed-20260924-v3-month-scope", "counts": counts,
               "records": {k: f"{k}_records.json" for k in records}, "cohort_hashes": hashes,
               "ranker_train_rows": 2, "ranker_train_positions_sha256": positions_hash,
               "test_mode": "all_fresh", "history_scope": scope, "history_scope_sha256": scope_hash,
               "historical_closure_sha256": "c" * 64}, "manifest_hash")
    write_json(protocol / "protocol_manifest.json", pm)
    selection = seal({"status": "accepted", "winner": models[1], "accepted_candidates": models[1:],
                      "protocol_manifest_hash": pm["manifest_hash"]}, "selection_hash")
    write_json(remote / "selection_v3.json", selection)
    plan = seal({"selection_hash": common.digest(remote / "selection_v3.json"), "protocol_manifest_hash": pm["manifest_hash"],
                 "winner": models[1], "models": models, "final_evaluation_allowed": True,
                 "test_users": 2, "test_cohort_hash": hashes["test"], "test_mode": "all_fresh",
                 "final_init_seed": 424242,
                 "history_scope": scope, "history_scope_sha256": scope_hash,
                 "historical_closure_sha256": "c" * 64}, "plan_hash")
    write_json(remote / "final_plan_v3.json", plan)
    for log in final.LOG_FILES: (remote / log).write_text(log + " complete\n")
    write_json(run / "disk_preflight.json", {"stage": "final", "status": "pass"})
    write_json(run / "initial_state_audit.json", {"models": models, "copied_tensor_keys": ["w", "cross_gate_logit"],
               "state_sha256": {v: {"w": "same", "cross_gate_logit": v} for v in models}})
    train_hash = positions_hash
    for epoch in (1, 2, 3): write_json(run / f"seed_42/epoch_{epoch}_trace.json", {"epoch": epoch, "rows": 2})
    top_hashes = {}
    for offset, variant in enumerate(models):
        group = run / f"seed_42/{variant}"; group.mkdir(parents=True)
        history = [{"epoch": x, "loss": .1 + offset} for x in (1, 2, 3)]
        base_manifest = {"protocol": pm["protocol"], "variant": variant, "seed": 42, "init_seed": 424242, "epochs": 3,
                         "train_positions_sha256": train_hash, "trace_files": {str(x): {"epoch": x, "rows": 2} for x in (1, 2, 3)},
                         "history_scope": scope, "history_scope_sha256": scope_hash, "historical_closure_sha256": "c" * 64}
        torch.save({"model": {"w": torch.tensor([1. + offset])}, "manifest": base_manifest,
                    "history": history, "epoch": 3}, group / "last.pth")
        checkpoint_hash = common.digest(group / "last.pth")
        write_json(group / "manifest.json", dict(base_manifest, checkpoint_sha256=checkpoint_hash))
        write_json(group / "history.json", history)
        write_json(group / "COMPLETED.json", {"status": "complete", "epoch": 3, "checkpoint_sha256": checkpoint_hash})
        for epoch in (1, 2, 3):
            write_json(group / f"epoch_{epoch}_trace.json", {"epoch": epoch, "rows": 2})
            npz(group / f"screen_epoch{epoch}_users.npz", [1, 2], [10, 20])
        top_manifest = dict(base_manifest, final=True, final_plan_hash=plan["plan_hash"], test_users=2,
                            test_mode="all_fresh", test_cohort_hash=hashes["test"])
        (run / variant).mkdir()
        torch.save({"model": {"w": torch.tensor([1. + offset])}, "manifest": top_manifest,
                    "history": history, "epoch": 3}, run / variant / "last.pth")
        write_json(run / variant / "manifest.json", top_manifest)
        top_hashes[variant] = common.digest(run / variant / "last.pth")
        metrics = {"hr5": .5, "pool_hit": 1.0, "ndcg5": .5, "users": 2}
        write_json(run / f"{variant}_test_metrics.json", metrics)
        npz(run / f"{variant}_test_users.npz", [1, 2], [11, 21])
    evidence = {str(p.relative_to(run)): common.digest(p) for p in sorted(run.rglob("*")) if p.is_file()
                and not p.name.endswith("_test_metrics.json") and not p.name.endswith("_test_users.npz")}
    fm = seal({"protocol_manifest_hash": pm["manifest_hash"], "final_plan_hash": plan["plan_hash"],
               "models": models, "test_users": 2, "train_positions_sha256": train_hash,
               "test_mode": "all_fresh", "test_history_records_hash": hashes["test"], "test_future_labels_read": False,
               "test_history_coverage_users": 2, "all_prefix_rows": 2, "history_scope": scope,
               "history_scope_sha256": scope_hash, "historical_closure_sha256": "c" * 64,
               "candidate_cache": caches["final_train"], "checkpoint_hashes": top_hashes,
               "evidence_hashes": evidence}, "manifest_hash")
    write_json(run / "final_manifest.json", fm)
    write_json(run / "FINAL_TRAIN_COMPLETED.json", {"manifest_hash": fm["manifest_hash"]})
    result = seal({"plan_hash": plan["plan_hash"], "test_users": 2, "test_cohort_hash": hashes["test"],
                   "test_mode": "all_fresh", "history_scope": scope, "history_scope_sha256": scope_hash,
                   "historical_closure_sha256": "c" * 64,
                   "models": {v: json.loads((run / f"{v}_test_metrics.json").read_text()) for v in models},
                   "candidate_cache": caches["test_final"],
                   "paired": {"users": 2, "gained": 0, "lost": 0, "hr5_delta": 0.0, "ndcg5_delta": 0.0,
                              "hr5_ci95": [-0.1, 0.1]},
                   "status": "raw_retained"}, "results_hash")
    write_json(run / "final_test_results.json", result)
    write_json(protocol / "TEST_STARTED.json", {"status": "complete", "plan_hash": plan["plan_hash"],
               "selection_hash": plan["selection_hash"], "checkpoint_hashes": top_hashes,
               "test_users": 2, "test_mode": "all_fresh", "test_cohort_hash": hashes["test"],
               "history_scope": scope, "history_scope_sha256": scope_hash,
               "historical_closure_sha256": "c" * 64, "results_hash": result["results_hash"]})
    return remote, pm["manifest_hash"], counts


class FakeRemote:
    def __init__(self, root): self.root = root; self.ready = True; self.corrupt = False; self.drift = False
    def inventory(self):
        if not self.ready: return {"ready": False, "reason": "FINAL_TRAIN_COMPLETED absent"}
        areas = {}
        for area, base in (("final_v3", self.root / "final_v3"), ("protocol_v3", self.root / "protocol_v3")):
            areas[area] = {str(p.relative_to(base)): {"size": p.stat().st_size, "sha256": common.digest(p), "remote.original": str(p)}
                           for p in base.rglob("*") if p.is_file()}
        areas["control"] = {name: {"size": (self.root / name).stat().st_size, "sha256": common.digest(self.root / name), "remote.original": str(self.root / name)}
                            for name in ("selection_v3.json", "final_plan_v3.json", *final.LOG_FILES)}
        results = json.loads((self.root / "final_v3/final_test_results.json").read_text())
        plan = json.loads((self.root / "final_plan_v3.json").read_text())
        selection = json.loads((self.root / "selection_v3.json").read_text())
        fm = json.loads((self.root / "final_v3/final_manifest.json").read_text())
        value = {"ready": True, "winner": plan["winner"], "models": plan["models"], "areas": areas,
                 "audit": {"protocol_hash": plan["protocol_manifest_hash"], "plan_hash": plan["plan_hash"],
                           "selection_hash": selection["selection_hash"], "final_manifest_hash": fm["manifest_hash"],
                           "results_hash": results["results_hash"], "test_marker_results_hash": results["results_hash"],
                           "official_validation": "fixture", "remote_writes": False}}
        value["inventory_hash"] = common.json_hash(value)
        if self.drift: value["audit"]["plan_hash"] = "changed"
        return value
    def fetch(self, original, destination):
        shutil.copyfile(original, destination)
        if self.corrupt: destination.write_bytes(b"bad")


class FinalArchiveTests(unittest.TestCase):
    def test_remote_not_ready_gate_does_not_open_final_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); (root / "final_v3").mkdir(); (root / "protocol_v3").mkdir(); (root / "src_v3/code").mkdir(parents=True)
            (root / "final_v3/FINAL_TRAIN_COMPLETED.json").write_text("not-json-and-must-not-open")
            prelude = "from pathlib import Path\n_original=Path.open\ndef guarded(self,*a,**k):\n if self.name=='FINAL_TRAIN_COMPLETED.json':raise RuntimeError('READ_TRAP')\n return _original(self,*a,**k)\nPath.open=guarded\n"
            proc = subprocess.run([sys.executable, "-c", prelude + final.REMOTE, str(root), str(root / "final_v3"),
                                   str(root / "protocol_v3"), str(root / "src_v3/code"), str(root / "selection_v3.json"),
                                   str(root / "final_plan_v3.json"), json.dumps(final.LOG_FILES), final.PROTOCOL_HASH,
                                   json.dumps(final.COUNTS), str(final.FULL_PREFIX_ROWS)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0); self.assertIn("TEST_STARTED absent", proc.stdout); self.assertNotIn("READ_TRAP", proc.stderr)

    def test_frozen_runner_smoke_produces_exact_final_structure(self):
        import run_cross_multiseed as runner
        from tests.test_cross_multiseed import _fixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); fixture = root / "fixture"; fixture.mkdir(); base, history = _fixture(fixture)
            remote = root / "remote"; remote.mkdir(); protocol = remote / "protocol_v3"
            manifest = runner.prepare(SimpleNamespace(base_run=str(base), run_dir=str(protocol), historical_users=str(history),
                train_users=4, screen_users=2, confirm_users=2, cohort_seed=runner.COHORT_SEED,
                test_mode="all_fresh", smoke=True))
            selection = seal({"status": "accepted", "winner": "normalized_gated", "accepted_candidates": ["normalized_gated"],
                              "protocol_manifest_hash": manifest["manifest_hash"]}, "selection_hash")
            write_json(remote / "selection_v3.json", selection)
            plan = seal({"protocol_manifest": str(protocol / "protocol_manifest.json"), "base_run": str(base.resolve()),
                    "selection": str(remote / "selection_v3.json"), "selection_hash": common.digest(remote / "selection_v3.json"),
                    "protocol_manifest_hash": manifest["manifest_hash"], "test_cohort_hash": manifest["cohort_hashes"]["test"],
                    "test_mode": manifest["test_mode"], "test_users": manifest["counts"]["test"],
                    "models": ["raw", "normalized_gated"], "winner": "normalized_gated", "final_evaluation_allowed": True,
                    "final_init_seed": runner.INIT_SEEDS[42], "history_scope": manifest["history_scope"],
                    "history_scope_sha256": manifest["history_scope_sha256"],
                    "historical_closure_sha256": manifest["historical_closure_sha256"]}, "plan_hash")
            write_json(remote / "final_plan_v3.json", plan)
            for log in final.LOG_FILES: (remote / log).write_text(log + " fixture\n")
            args = SimpleNamespace(final_plan=str(remote / "final_plan_v3.json"), base_run=str(base), run_dir=str(remote / "final_v3"),
                                   epochs=3, dim=8, token_dim=8, hist_len=5, batch_size=8, candidates=10)
            with patch.object(runner, "_load_plan", return_value=plan), patch.object(runner.torch.cuda, "is_available", return_value=False):
                runner.final_train(args); runner.final_test(args)
            # final-train/test reuses the pre-existing dev screen cache; add the
            # two other prior-dev caches so the fixture represents the complete
            # protocol tree present at the real final stage.
            cache(protocol, "confirm", manifest["cohort_hashes"]["confirm"], [[2, 3], [3, 4]])
            cache(protocol, "train", manifest["cohort_hashes"]["train"], [[2, 3]] * manifest["ranker_train_rows"])
            actual = {str(p.relative_to(remote / "final_v3")) for p in (remote / "final_v3").rglob("*") if p.is_file()}
            self.assertEqual(actual, final.final_names(plan["models"]))
            fm = json.loads((remote / "final_v3/final_manifest.json").read_text())
            test_outputs = {"final_test_results.json", *(f"{v}_test_metrics.json" for v in plan["models"]),
                            *(f"{v}_test_users.npz" for v in plan["models"])}
            expected = final.final_names(plan["models"]) - {"final_manifest.json", "FINAL_TRAIN_COMPLETED.json"} - test_outputs
            self.assertEqual(set(fm["evidence_hashes"]), expected)
            inventory = FakeRemote(remote).inventory()
            with (base / "data.pkl").open("rb") as stream:
                full_rows = len(__import__("pickle").load(stream)["data"].train_positions)
            with patch.object(final, "PROTOCOL_HASH", manifest["manifest_hash"]), patch.object(final, "COUNTS", manifest["counts"]), patch.object(final, "FULL_PREFIX_ROWS", full_rows):
                bundle = root / "bundle"; shutil.copytree(remote / "final_v3", bundle / "final_v3")
                shutil.copytree(remote / "protocol_v3", bundle / "protocol_v3"); (bundle / "control").mkdir()
                for name in ("selection_v3.json", "final_plan_v3.json", *final.LOG_FILES):
                    shutil.copyfile(remote / name, bundle / "control" / name)
                final.verify_bundle(bundle, inventory, str(remote))

    def test_complete_hardlink_reuse_and_existing_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); remote_root, ph, counts = build_fixture(root)
            with patch.object(final, "PROTOCOL_HASH", ph), patch.object(final, "COUNTS", counts), patch.object(final, "FULL_PREFIX_ROWS", 2):
                reuse = root / "protocol_complete"; shutil.copytree(remote_root / "protocol_v3", reuse)
                local = root / "archive"; receipts = root / "receipts"
                result = final.run(FakeRemote(remote_root), local, receipts, str(remote_root), reuse)
                self.assertEqual(result["status"], "complete")
                self.assertIn("protocol_v3/ranker_train_positions.npy", result["hardlinks"])
                self.assertEqual((reuse / "ranker_train_positions.npy").stat().st_ino,
                                 (local / "protocol_v3/ranker_train_positions.npy").stat().st_ino)
                self.assertEqual(final.run(FakeRemote(remote_root), local, receipts, str(remote_root), reuse)["status"], "verified_existing")
                self.assertFalse((receipts / ".final_archive.lock").exists())

    def test_not_ready_and_dry_run_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); rr, ph, counts = build_fixture(root); remote = FakeRemote(rr); remote.ready = False
            with patch.object(final, "PROTOCOL_HASH", ph), patch.object(final, "COUNTS", counts), patch.object(final, "FULL_PREFIX_ROWS", 2):
                self.assertEqual(final.run(remote, root / "a", root / "r", str(rr), dry_run=True)["status"], "not_ready")
                remote.ready = True
                self.assertEqual(final.run(remote, root / "a", root / "r", str(rr), dry_run=True)["status"], "ready")
                self.assertFalse((root / "a").exists()); self.assertFalse((root / "r").exists())

    def test_failure_preserves_partial_and_lock_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); rr, ph, counts = build_fixture(root); remote = FakeRemote(rr); remote.corrupt = True
            with patch.object(final, "PROTOCOL_HASH", ph), patch.object(final, "COUNTS", counts), patch.object(final, "FULL_PREFIX_ROWS", 2):
                with self.assertRaisesRegex(ValueError, "hash"): final.run(remote, root / "archive", root / "receipts", str(rr))
                self.assertTrue(list(root.glob("archive.partial-*")))
                self.assertTrue((root / "receipts/.final_archive.lock").exists())
                with self.assertRaises(FileExistsError): final.run(FakeRemote(rr), root / "archive", root / "receipts", str(rr))

    def test_remote_original_and_inventory_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); rr, ph, counts = build_fixture(root); inv = FakeRemote(rr).inventory()
            with patch.object(final, "PROTOCOL_HASH", ph), patch.object(final, "COUNTS", counts), patch.object(final, "FULL_PREFIX_ROWS", 2):
                inv["areas"]["control"]["selection_v3.json"]["remote.original"] = "/wrong"
                inv["inventory_hash"] = common.json_hash({k: v for k, v in inv.items() if k != "inventory_hash"})
                with self.assertRaisesRegex(ValueError, "remote.original"): final.verify_metadata(inv, str(rr))
                inv = FakeRemote(rr).inventory(); inv["inventory_hash"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "result hash"): final.verify_metadata(inv, str(rr))

    def test_four_checkpoint_identity_and_test_pairing_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); rr, ph, counts = build_fixture(root); remote = FakeRemote(rr)
            with patch.object(final, "PROTOCOL_HASH", ph), patch.object(final, "COUNTS", counts), patch.object(final, "FULL_PREFIX_ROWS", 2):
                final.run(remote, root / "archive", root / "receipts", str(rr))
                archive = root / "archive"
                top = archive / "final_v3/raw/last.pth"; payload = torch.load(top, weights_only=True)
                payload["manifest"] = torch.load(archive / "final_v3/seed_42/raw/last.pth", weights_only=True)["manifest"]
                torch.save(payload, top)
                inv = remote.inventory(); inv["areas"]["final_v3"]["raw/last.pth"] = {"size": top.stat().st_size, "sha256": common.digest(top), "remote.original": str(rr / "final_v3/raw/last.pth")}
                inv["inventory_hash"] = common.json_hash({k: v for k, v in inv.items() if k != "inventory_hash"})
                with self.assertRaisesRegex(ValueError, "sealed final training|top checkpoint|collapsed"): final.verify_bundle(archive, inv, str(rr))

    def test_remote_drift_keeps_unpublished_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); rr, ph, counts = build_fixture(root); remote = FakeRemote(rr); calls = 0; original = remote.inventory
            def inventory():
                nonlocal calls; calls += 1; value = original()
                if calls > 1: value["audit"]["plan_hash"] = "changed"
                return value
            remote.inventory = inventory
            with patch.object(final, "PROTOCOL_HASH", ph), patch.object(final, "COUNTS", counts), patch.object(final, "FULL_PREFIX_ROWS", 2):
                with self.assertRaisesRegex(ValueError, "drift"): final.run(remote, root / "archive", root / "receipts", str(rr))
                self.assertFalse((root / "archive").exists()); self.assertTrue((root / "receipts/.final_archive.lock").exists())


if __name__ == "__main__": unittest.main()
