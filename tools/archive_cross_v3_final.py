#!/usr/bin/env python3
"""Read-only, one-shot archive of a completed accepted cross-v3 final run.

This tool never trains, scores, cleans, migrates, shuts down, or writes remotely.
It is intentionally inapplicable to the no-winner branch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import uuid

import numpy as np
import torch

import archive_cross_v3 as common
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from cross_pool_cache import load_cache

PROTOCOL_HASH = "dcf65b10c266ba02b00d0493815a14b8af9aa4f47551d7c562a2010104409667"
COUNTS = {"train": 100000, "screen": 20000, "confirm": 80000, "test": 17296}
FULL_PREFIX_ROWS = 4731777
WINNERS = ("normalized_gated", "zero_cross")
LOG_FILES = ("compare_v3.log", "lock_v3.log", "final_train_v3.log", "final_test_v3.log")
PROTOCOL_BASE = (
    "protocol_manifest.json", "historical_users_verified.json", "train_records.json",
    "screen_records.json", "confirm_records.json", "test_records.json",
    "ranker_train_positions.npy", "TEST_STARTED.json",
)


def candidate_names(label: str) -> tuple[str, str, str, str]:
    return (f"candidate_{label}.json", f"candidate_{label}.items.npy",
            f"candidate_{label}.lengths.npy", f"candidate_{label}.json.building")


def final_names(models: list[str]) -> set[str]:
    names = {"disk_preflight.json", "initial_state_audit.json", "final_manifest.json", "FINAL_TRAIN_COMPLETED.json",
             "final_test_results.json"}
    for epoch in (1, 2, 3):
        names.add(f"seed_42/epoch_{epoch}_trace.json")
    for variant in models:
        names.update((f"{variant}/last.pth", f"{variant}/manifest.json",
                      f"{variant}_test_metrics.json", f"{variant}_test_users.npz",
                      f"seed_42/{variant}/last.pth", f"seed_42/{variant}/history.json",
                      f"seed_42/{variant}/manifest.json", f"seed_42/{variant}/COMPLETED.json"))
        for epoch in (1, 2, 3):
            names.update((f"seed_42/{variant}/epoch_{epoch}_trace.json",
                          f"seed_42/{variant}/screen_epoch{epoch}_users.npz"))
    return names


REMOTE = r'''
import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(sys.argv[1]).resolve();final=Path(sys.argv[2]).resolve();protocol=Path(sys.argv[3]).resolve();code=Path(sys.argv[4]).resolve()
selection=Path(sys.argv[5]).resolve();plan_path=Path(sys.argv[6]).resolve();logs=json.loads(sys.argv[7])
expected_protocol=sys.argv[8];expected_counts=json.loads(sys.argv[9]);expected_full_rows=int(sys.argv[10])
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def inv(p):
 if p.is_symlink() or not p.is_file():raise ValueError('unsafe/missing allowed artifact '+str(p))
 before=p.stat();h=sha(p);after=p.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('artifact changed during inventory')
 return {'size':after.st_size,'sha256':h,'remote.original':str(p)}
# The two tiny gates are the only contents opened before official validation.
train_done=final/'FINAL_TRAIN_COMPLETED.json';test_marker=protocol/'TEST_STARTED.json'
if not train_done.exists():print(json.dumps({'ready':False,'reason':'FINAL_TRAIN_COMPLETED absent'}));sys.exit(0)
if not test_marker.exists():print(json.dumps({'ready':False,'reason':'TEST_STARTED absent'}));sys.exit(0)
started=read(test_marker)
if started.get('status')!='complete':print(json.dumps({'ready':False,'reason':'TEST_STARTED is not complete'}));sys.exit(0)
sys.path.insert(0,str(code))
import run_cross_multiseed as r
# _load_plan performs the frozen evidence-only compare(return_only=True) validation.
# It does not call evaluate/targets and does not write selection output.
plan=r._load_plan(plan_path)
proto=r.load_protocol(Path(plan['protocol_manifest']))
if Path(plan['protocol_manifest']).resolve()!=protocol/'protocol_manifest.json':raise ValueError('protocol path mismatch')
if Path(plan['selection']).resolve()!=selection:raise ValueError('selection path mismatch')
if proto['manifest_hash']!=expected_protocol or proto.get('counts')!=expected_counts:raise ValueError('formal protocol identity mismatch')
sel=r._checked_json(selection,'selection_hash')
winner=plan.get('winner');models=plan.get('models')
if sel.get('status')!='accepted' or winner not in ('normalized_gated','zero_cross') or models!=['raw',winner] or not plan.get('final_evaluation_allowed'):raise ValueError('not an accepted uniquely locked winner plan')
fm=r._checked_json(final/'final_manifest.json','manifest_hash');done=r._checked_json(train_done)
results=r._checked_json(final/'final_test_results.json','results_hash')
base=r.validate_base(Path(plan['base_run']),proto)
full_positions_hash=r.sha256_array(__import__('numpy').asarray(base.train_positions,dtype='int64'))
if done.get('manifest_hash')!=fm['manifest_hash'] or fm.get('final_plan_hash')!=plan['plan_hash'] or fm.get('models')!=models:raise ValueError('final training completion mismatch')
if started.get('results_hash')!=results['results_hash'] or results.get('plan_hash')!=plan['plan_hash'] or set(results.get('models',{}))!=set(models):raise ValueError('completed test/result mismatch')
for key in ('history_scope','history_scope_sha256','historical_closure_sha256'):
 expected=proto.get(key)
 for name,obj in (('plan',plan),('final_manifest',fm),('test_marker',started),('results',results)):
  if obj.get(key)!=expected:raise ValueError(name+' scope binding mismatch')
if fm.get('test_future_labels_read') is not False or fm.get('test_history_coverage_users')!=expected_counts['test'] or fm.get('all_prefix_rows')!=expected_full_rows or not fm.get('evidence_hashes') or fm.get('train_positions_sha256')!=full_positions_hash:raise ValueError('incomplete final training coverage/seal')
for name,obj in (('plan',plan),('test_marker',started),('results',results)):
 if obj.get('test_mode')!=proto.get('test_mode') or obj.get('test_users')!=expected_counts['test'] or obj.get('test_cohort_hash')!=proto['cohort_hashes']['test']:raise ValueError(name+' test identity mismatch')
if fm.get('test_mode')!=proto.get('test_mode') or fm.get('test_users')!=expected_counts['test'] or fm.get('test_history_records_hash')!=proto['cohort_hashes']['test']:raise ValueError('final_manifest test identity mismatch')
for obj in (plan,fm,results,started):
 if obj.get('protocol_manifest_hash',expected_protocol)!=expected_protocol and obj is not results:raise ValueError('protocol binding mismatch')
 if obj.get('test_users',expected_counts['test'])!=expected_counts['test']:raise ValueError('test count mismatch')
if started.get('plan_hash')!=plan['plan_hash'] or started.get('selection_hash')!=plan['selection_hash'] or started.get('checkpoint_hashes')!=fm['checkpoint_hashes']:raise ValueError('TEST_STARTED identity mismatch')
allowed_final={'disk_preflight.json','initial_state_audit.json','final_manifest.json','FINAL_TRAIN_COMPLETED.json','final_test_results.json'}
for epoch in (1,2,3):allowed_final.add('seed_42/epoch_'+str(epoch)+'_trace.json')
for variant in models:
 allowed_final.update((variant+'/last.pth',variant+'/manifest.json',variant+'_test_metrics.json',variant+'_test_users.npz','seed_42/'+variant+'/last.pth','seed_42/'+variant+'/history.json','seed_42/'+variant+'/manifest.json','seed_42/'+variant+'/COMPLETED.json'))
 for epoch in (1,2,3):allowed_final.update(('seed_42/'+variant+'/epoch_'+str(epoch)+'_trace.json','seed_42/'+variant+'/screen_epoch'+str(epoch)+'_users.npz'))
actual_final=set()
for p in final.rglob('*'):
 if p.is_symlink():raise ValueError('symlink in final tree')
 if p.is_file():actual_final.add(str(p.relative_to(final)))
if actual_final!=allowed_final:raise ValueError('final tree differs from fixed allowlist')
protocol_allowed={'protocol_manifest.json','historical_users_verified.json','train_records.json','screen_records.json','confirm_records.json','test_records.json','ranker_train_positions.npy','TEST_STARTED.json'}
for label in ('screen','confirm','train','final_train','test_final'):
 protocol_allowed.update(('candidate_'+label+'.json','candidate_'+label+'.items.npy','candidate_'+label+'.lengths.npy','candidate_'+label+'.json.building'))
actual_protocol=set()
for p in protocol.rglob('*'):
 if p.is_symlink():raise ValueError('symlink in protocol tree')
 if p.is_file():actual_protocol.add(str(p.relative_to(protocol)))
if actual_protocol!=protocol_allowed:raise ValueError('protocol tree differs from fixed allowlist')
areas={'final_v3':{rel:inv(final/rel) for rel in sorted(allowed_final)},'protocol_v3':{rel:inv(protocol/rel) for rel in sorted(protocol_allowed)}}
control={'selection_v3.json':selection,'final_plan_v3.json':plan_path}
for name in logs:
 if name not in ('compare_v3.log','lock_v3.log','final_train_v3.log','final_test_v3.log'):raise ValueError('unapproved log name')
 control[name]=root/name
areas['control']={name:inv(path) for name,path in control.items()}
# Check all final-train evidence against the seal. Four checkpoint files are
# inventoried independently; no equality assumption is made between copies.
for rel,h in fm.get('evidence_hashes',{}).items():
 if rel not in areas['final_v3'] or areas['final_v3'][rel]['sha256']!=h:raise ValueError('sealed final training evidence mismatch '+rel)
for variant in models:
 if areas['final_v3'][variant+'/last.pth']['sha256']!=fm['checkpoint_hashes'][variant]:raise ValueError('top checkpoint seal mismatch')
for receipt in (fm.get('candidate_cache'),results.get('candidate_cache')):
 p=Path(receipt['path']).resolve()
 if p.parent!=protocol or p.name not in ('candidate_final_train.json','candidate_test_final.json') or sha(p)!=receipt['sha256']:raise ValueError('final candidate receipt mismatch')
result={'ready':True,'winner':winner,'models':models,'areas':areas,'audit':{'protocol_hash':proto['manifest_hash'],'plan_hash':plan['plan_hash'],'selection_hash':sel['selection_hash'],'final_manifest_hash':fm['manifest_hash'],'results_hash':results['results_hash'],'test_marker_results_hash':started['results_hash'],'official_validation':'frozen _load_plan; compare(return_only=True), no evaluate/targets/write','remote_writes':False}}
canonical=json.dumps(result,sort_keys=True,separators=(',',':')).encode();result['inventory_hash']=hashlib.sha256(canonical).hexdigest()
print(json.dumps(result,sort_keys=True))
'''


class SSHRemote:
    def __init__(self, args):
        if args.host.startswith("-"):
            raise ValueError("host cannot start with dash")
        self.ssh = ["ssh", "-o", "BatchMode=yes", "-p", str(args.port)]
        if args.control_path:
            self.ssh += ["-o", "ControlPath=" + args.control_path]
        self.ssh += [args.host]
        self.python = args.remote_python
        self.root = args.remote_root.rstrip("/")
        self.final = args.remote_final.rstrip("/")
        self.protocol = args.remote_protocol.rstrip("/")
        self.code = args.remote_code.rstrip("/")
        self.selection = args.remote_selection
        self.plan = args.remote_plan
        for value in (self.root, self.final, self.protocol, self.code, self.selection, self.plan):
            posix = PurePosixPath(value)
            if not posix.is_absolute() or ".." in posix.parts or str(posix) != value:
                raise ValueError("remote paths must be normalized absolute paths")
        expected = PurePosixPath(self.root)
        identities = {"final": (PurePosixPath(self.final), expected / "final_v3"),
                      "protocol": (PurePosixPath(self.protocol), expected / "protocol_v3"),
                      "code": (PurePosixPath(self.code), expected / "src_v3" / "code"),
                      "selection": (PurePosixPath(self.selection), expected / "selection_v3.json"),
                      "plan": (PurePosixPath(self.plan), expected / "final_plan_v3.json")}
        if not expected.is_absolute() or any(actual != wanted for actual, wanted in identities.values()):
            raise ValueError("remote paths must be the fixed cross-v3 root layout")

    def inventory(self):
        argv = (self.python, "-B", "-c", REMOTE, self.root, self.final, self.protocol,
                self.code, self.selection, self.plan, json.dumps(LOG_FILES),
                PROTOCOL_HASH, json.dumps(COUNTS, sort_keys=True), str(FULL_PREFIX_ROWS))
        proc = subprocess.run(self.ssh + [" ".join(shlex.quote(x) for x in argv)],
                              check=True, capture_output=True, text=True)
        return json.loads(proc.stdout)

    def fetch(self, remote_original: str, destination: Path):
        original = PurePosixPath(remote_original)
        if not original.is_absolute() or ".." in original.parts:
            raise ValueError("remote.original must be absolute and normalized")
        command = "cat -- " + shlex.quote(str(original))
        with destination.open("xb") as stream:
            subprocess.run(self.ssh + [command], stdout=stream, check=True)
            stream.flush(); os.fsync(stream.fileno())


def _inventory_hash(inventory: dict) -> str:
    value = dict(inventory); value.pop("inventory_hash", None)
    return common.json_hash(value)


def _expected_remote(area: str, rel: str, remote_root: str) -> str:
    if area == "control":
        return str(PurePosixPath(remote_root) / rel)
    return str(PurePosixPath(remote_root) / area / rel)


def verify_metadata(inventory: dict, remote_root: str) -> None:
    if not inventory.get("ready"):
        raise ValueError("missing completed remote inventory")
    if inventory.get("inventory_hash") != _inventory_hash(inventory):
        raise ValueError("remote inventory result hash mismatch")
    if inventory.get("models") != ["raw", inventory.get("winner")] or inventory.get("winner") not in WINNERS:
        raise ValueError("winner/model identity mismatch")
    audit = inventory.get("audit", {})
    if audit.get("protocol_hash") != PROTOCOL_HASH or audit.get("results_hash") != audit.get("test_marker_results_hash"):
        raise ValueError("protocol or result completion mismatch")
    expected_final = final_names(inventory["models"])
    expected_protocol = set(PROTOCOL_BASE)
    for label in ("screen", "confirm", "train", "final_train", "test_final"):
        expected_protocol.update(candidate_names(label))
    if set(inventory.get("areas", {}).get("final_v3", {})) != expected_final:
        raise ValueError("final inventory is not the fixed allowlist")
    if set(inventory.get("areas", {}).get("protocol_v3", {})) != expected_protocol:
        raise ValueError("protocol inventory is not the fixed allowlist")
    if set(inventory.get("areas", {}).get("control", {})) != {"selection_v3.json", "final_plan_v3.json", *LOG_FILES}:
        raise ValueError("control inventory is not the fixed allowlist")
    for area, files in inventory.get("areas", {}).items():
        if area not in ("final_v3", "protocol_v3", "control"):
            raise ValueError("unexpected inventory area")
        for rel, meta in files.items():
            common.safe_relative(rel)
            expected = _expected_remote(area, rel, remote_root)
            if meta.get("remote.original") != expected or not PurePosixPath(expected).is_absolute():
                raise ValueError("remote.original identity mismatch: " + rel)
            if not isinstance(meta.get("size"), int) or meta["size"] < 0 or len(meta.get("sha256", "")) != 64:
                raise ValueError("invalid inventory metadata")


def _files(root: Path) -> set[str]:
    found = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in local archive")
        if path.is_file():
            found.add(str(path.relative_to(root)))
    return found


def verify_checkpoint(path: Path, variant: str, *, top: bool, plan: dict, final_manifest: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(("model", "manifest", "history", "epoch")) - set(payload):
        raise ValueError("invalid checkpoint fields")
    manifest = payload["manifest"]
    if (payload["epoch"] != 3 or manifest.get("epochs") != 3 or manifest.get("variant") != variant
            or manifest.get("seed") != 42 or manifest.get("init_seed") != plan.get("final_init_seed")):
        raise ValueError("checkpoint variant/seed mismatch")
    if manifest.get("train_positions_sha256") != final_manifest["train_positions_sha256"]:
        raise ValueError("checkpoint training identity mismatch")
    if top:
        if (manifest.get("final") is not True or manifest.get("final_plan_hash") != plan["plan_hash"]
                or manifest.get("test_cohort_hash") != plan["test_cohort_hash"]
                or manifest.get("test_mode") != plan["test_mode"] or manifest.get("test_users") != plan["test_users"]):
            raise ValueError("top checkpoint is not locked final result")
    elif manifest.get("final") is True:
        raise ValueError("training checkpoint unexpectedly has final wrapper identity")
    if not payload["model"] or any(not torch.is_tensor(x) or not torch.isfinite(x).all() for x in payload["model"].values()):
        raise ValueError("invalid checkpoint tensors")
    for key in ("history_scope", "history_scope_sha256", "historical_closure_sha256"):
        if manifest.get(key) != plan.get(key): raise ValueError("checkpoint scope binding mismatch")
    return payload


def verify_bundle(root: Path, inventory: dict, remote_root: str) -> None:
    verify_metadata(inventory, remote_root)
    expected = {f"{area}/{rel}" for area, values in inventory["areas"].items() for rel in values}
    if _files(root) != expected:
        raise ValueError("local archive tree differs from inventory")
    for area, values in inventory["areas"].items():
        for rel, meta in values.items():
            path = root / area / rel
            if path.stat().st_size != meta["size"] or common.digest(path) != meta["sha256"]:
                raise ValueError("archive hash/size mismatch: " + area + "/" + rel)
            if path.suffix == ".json": common.checked_json(path)
            elif path.suffix == ".npz": common.verify_payload(path, meta)
            elif path.suffix == ".npy": np.load(path, mmap_mode="r", allow_pickle=False)
            elif path.suffix == ".pth": pass
            elif path.suffix in (".log", ".building"): path.read_text(encoding="utf-8")
            else: raise ValueError("unsupported archive file type")
    control = root / "control"; final = root / "final_v3"; protocol = root / "protocol_v3"
    selection = common.checked_json(control / "selection_v3.json")
    plan = common.checked_json(control / "final_plan_v3.json")
    pm = common.checked_json(protocol / "protocol_manifest.json")
    fm = common.checked_json(final / "final_manifest.json")
    done = common.checked_json(final / "FINAL_TRAIN_COMPLETED.json")
    marker = common.checked_json(protocol / "TEST_STARTED.json")
    results = common.checked_json(final / "final_test_results.json")
    for obj, key in ((selection, "selection_hash"), (plan, "plan_hash"), (pm, "manifest_hash"),
                     (fm, "manifest_hash"), (results, "results_hash")):
        if obj.get(key) != common.json_hash({k: v for k, v in obj.items() if k != key}):
            raise ValueError(key + " mismatch")
    audit = inventory["audit"]
    if (audit.get("selection_hash") != selection["selection_hash"] or audit.get("plan_hash") != plan["plan_hash"]
            or audit.get("final_manifest_hash") != fm["manifest_hash"] or audit.get("results_hash") != results["results_hash"]):
        raise ValueError("downloaded control/final hashes differ from remote audit")
    winner = inventory["winner"]; models = ["raw", winner]
    if selection.get("status") != "accepted" or plan.get("winner") != winner or plan.get("models") != models:
        raise ValueError("selection/plan winner mismatch")
    if plan.get("selection_hash") != common.digest(control / "selection_v3.json"):
        raise ValueError("plan selection file binding mismatch")
    if plan.get("protocol_manifest_hash") != PROTOCOL_HASH or pm.get("manifest_hash") != PROTOCOL_HASH or pm.get("counts") != COUNTS:
        raise ValueError("formal protocol mismatch")
    if done.get("manifest_hash") != fm["manifest_hash"] or marker.get("status") != "complete" or marker.get("results_hash") != results["results_hash"]:
        raise ValueError("final completion marker mismatch")
    if (marker.get("plan_hash") != plan["plan_hash"] or marker.get("selection_hash") != plan["selection_hash"]
            or marker.get("checkpoint_hashes") != fm.get("checkpoint_hashes")):
        raise ValueError("test marker plan/checkpoint binding mismatch")
    if results.get("plan_hash") != plan["plan_hash"] or set(results.get("models", {})) != set(models):
        raise ValueError("final result identity mismatch")
    for key in ("history_scope", "history_scope_sha256", "historical_closure_sha256"):
        for name, obj in (("plan", plan), ("final manifest", fm), ("marker", marker), ("results", results)):
            if obj.get(key) != pm.get(key): raise ValueError(name + " scope binding mismatch")
    if (fm.get("test_future_labels_read") is not False or fm.get("test_history_coverage_users") != COUNTS["test"]
            or fm.get("all_prefix_rows") != FULL_PREFIX_ROWS or not fm.get("evidence_hashes")
            or not fm.get("train_positions_sha256")):
        raise ValueError("final training coverage/seal mismatch")
    for name, obj in (("plan", plan), ("marker", marker), ("results", results)):
        if (obj.get("test_mode") != pm.get("test_mode") or obj.get("test_users") != COUNTS["test"]
                or obj.get("test_cohort_hash") != pm["cohort_hashes"]["test"]):
            raise ValueError(name + " test identity mismatch")
    if (fm.get("test_mode") != pm.get("test_mode") or fm.get("test_users") != COUNTS["test"]
            or fm.get("test_history_records_hash") != pm["cohort_hashes"]["test"]):
        raise ValueError("final manifest test identity mismatch")
    test_outputs = {"final_test_results.json", *(f"{v}_test_metrics.json" for v in models),
                    *(f"{v}_test_users.npz" for v in models)}
    expected_evidence = final_names(models) - {"final_manifest.json", "FINAL_TRAIN_COMPLETED.json"} - test_outputs
    if set(fm.get("evidence_hashes", {})) != expected_evidence:
        raise ValueError("final training evidence set mismatch")
    for rel, digest in fm["evidence_hashes"].items():
        common.safe_relative(rel)
        if common.digest(final / rel) != digest:
            raise ValueError("sealed final training evidence mismatch")
    for label, receipt in (("final_train", fm.get("candidate_cache")),
                           ("test_final", results.get("candidate_cache"))):
        if not isinstance(receipt, dict): raise ValueError("missing final candidate receipt")
        sidecar = protocol / f"candidate_{label}.json"
        if common.digest(sidecar) != receipt.get("sha256"):
            raise ValueError("candidate receipt hash mismatch")
        pools, metadata = load_cache(sidecar)
        if (metadata.get("files") != receipt.get("files") or metadata.get("pool_hash") != receipt.get("pool_hash")
                or metadata.get("records_hash") != receipt.get("records_hash")):
            raise ValueError("candidate payload receipt mismatch")
        expected_rows = FULL_PREFIX_ROWS if label == "final_train" else COUNTS["test"]
        if len(pools) != expected_rows or metadata.get("shape", [None])[0] != expected_rows:
            raise ValueError("candidate cache row count mismatch")
        if label == "test_final" and metadata.get("records_hash") != pm["cohort_hashes"]["test"]:
            raise ValueError("test candidate cohort mismatch")
    records = common.checked_json(protocol / pm["records"]["test"])
    uids = np.asarray([x[0] for x in records]); positions = np.asarray([x[1] for x in records])
    if len(records) != COUNTS["test"]:
        raise ValueError("test cohort size mismatch")
    for variant in models:
        trained = verify_checkpoint(final / f"seed_42/{variant}/last.pth", variant, top=False,
                                    plan=plan, final_manifest=fm)
        top = verify_checkpoint(final / f"{variant}/last.pth", variant, top=True,
                                plan=plan, final_manifest=fm)
        group_manifest = common.checked_json(final / f"seed_42/{variant}/manifest.json")
        group_done = common.checked_json(final / f"seed_42/{variant}/COMPLETED.json")
        group_history = common.checked_json(final / f"seed_42/{variant}/history.json")
        without_checkpoint = dict(group_manifest); without_checkpoint.pop("checkpoint_sha256", None)
        if trained["manifest"] != without_checkpoint or trained["history"] != group_history:
            raise ValueError("training checkpoint sidecar/history mismatch")
        if group_done.get("status") != "complete" or group_done.get("checkpoint_sha256") != common.digest(final / f"seed_42/{variant}/last.pth") or group_manifest.get("checkpoint_sha256") != group_done.get("checkpoint_sha256"):
            raise ValueError("training checkpoint marker mismatch")
        for epoch in (1, 2, 3):
            trace = common.checked_json(final / f"seed_42/{variant}/epoch_{epoch}_trace.json")
            shared = common.checked_json(final / f"seed_42/epoch_{epoch}_trace.json")
            if trace != group_manifest.get("trace_files", {}).get(str(epoch)) or trace != shared:
                raise ValueError("paired training trace mismatch")
            if trace.get("rows") != FULL_PREFIX_ROWS:
                raise ValueError("full training trace row mismatch")
        if top["manifest"] != common.checked_json(final / f"{variant}/manifest.json"):
            raise ValueError("top checkpoint manifest sidecar mismatch")
        if top["history"] != trained["history"] or len(top["history"]) != 3 or any(not np.isfinite(float(row.get("loss", float("nan")))) for row in top["history"]):
            raise ValueError("top checkpoint history mismatch")
        if set(trained["model"]) != set(top["model"]) or any(not torch.equal(trained["model"][k], top["model"][k]) for k in trained["model"]):
            raise ValueError("paired checkpoint tensors differ")
        # The top checkpoint is a separately sealed wrapper and must not be
        # reduced to an assumed duplicate of the training checkpoint.
        if trained["manifest"] == top["manifest"]:
            raise ValueError("four-checkpoint wrapper identities collapsed")
        if common.digest(final / f"{variant}/last.pth") != fm["checkpoint_hashes"][variant]:
            raise ValueError("top checkpoint hash mismatch")
        with np.load(final / f"{variant}_test_users.npz", allow_pickle=False) as arrays:
            if not np.array_equal(arrays["uid"], uids) or not np.array_equal(arrays["position"], positions):
                raise ValueError("test UID/position identity mismatch")
            calculated = {"hr5": float(np.mean(arrays["hit5"])), "pool_hit": float(np.mean(arrays["pool_hit"])),
                          "ndcg5": float(np.mean(arrays["ndcg5"])), "users": int(len(arrays["uid"]))}
        metrics = common.checked_json(final / f"{variant}_test_metrics.json")
        if metrics != results["models"][variant]:
            raise ValueError("test metrics/result mismatch")
        for key, value in calculated.items():
            if key not in metrics or (not np.isclose(metrics[key], value) if isinstance(value, float) else metrics[key] != value):
                raise ValueError("test metrics do not match user arrays")
    audit = common.checked_json(final / "initial_state_audit.json")
    for key in audit.get("copied_tensor_keys", []):
        if key != "cross_gate_logit" and len({audit["state_sha256"][v][key] for v in models}) != 1:
            raise ValueError("paired initialization mismatch")
    with np.load(final / "raw_test_users.npz", allow_pickle=False) as raw, np.load(final / f"{winner}_test_users.npz", allow_pickle=False) as candidate:
        for key in ("uid", "position", "pool_hit"):
            if not np.array_equal(raw[key], candidate[key]): raise ValueError("paired test identity mismatch")
        delta = candidate["hit5"].astype(np.float64) - raw["hit5"].astype(np.float64)
        ndcg = candidate["ndcg5"].astype(np.float64) - raw["ndcg5"].astype(np.float64)
        paired = results.get("paired", {})
        if (paired.get("users") != len(delta) or paired.get("gained") != int(np.sum(delta > 0))
                or paired.get("lost") != int(np.sum(delta < 0))
                or not np.isclose(paired.get("hr5_delta"), float(delta.mean()))
                or not np.isclose(paired.get("ndcg5_delta"), float(ndcg.mean()))):
            raise ValueError("paired result does not match user arrays")
    if results.get("status") not in ("accepted", "raw_retained"):
        raise ValueError("invalid final result status")
    paired = results["paired"]
    ci = paired.get("hr5_ci95")
    if (not isinstance(ci, list) or len(ci) != 2 or any(not np.isfinite(float(x)) for x in ci) or ci[0] > ci[1]):
        raise ValueError("invalid paired confidence interval")
    accepted = (paired.get("hr5_ci95", [float("-inf")])[0] > 0 and paired["hr5_delta"] >= 0.0005
                and paired["ndcg5_delta"] >= 0)
    if results["status"] != ("accepted" if accepted else "raw_retained"):
        raise ValueError("final result status disagrees with sealed paired rule")


def _find_reuse(source: Path | None, area: str, rel: str, meta: dict) -> Path | None:
    if source is None or area != "protocol_v3": return None
    candidate = source / rel
    if not candidate.exists(): return None
    common.reject_symlink_chain(source, candidate)
    if candidate.is_file() and candidate.stat().st_size == meta["size"] and common.digest(candidate) == meta["sha256"]:
        return candidate
    return None


def run(remote, local_dir, receipt_dir, remote_root, protocol_source=None, dry_run=False):
    destination = Path(local_dir).absolute(); receipts = Path(receipt_dir).absolute()
    source = Path(protocol_source).absolute() if protocol_source else None
    for path in (destination, receipts, *(() if source is None else (source,))):
        common.reject_symlink_chain(Path(path.anchor), path)
    if destination == receipts or destination.is_relative_to(receipts) or receipts.is_relative_to(destination):
        raise ValueError("archive and receipts must be disjoint")
    if source and (source == destination or source.is_relative_to(destination) or destination.is_relative_to(source)):
        raise ValueError("reuse source must be disjoint")
    if source and (source == receipts or source.is_relative_to(receipts) or receipts.is_relative_to(source)):
        raise ValueError("receipts and reuse source must be disjoint")
    inventory = remote.inventory()
    if not inventory.get("ready"):
        return {"status": "not_ready", "reason": inventory.get("reason"), "remote_writes": False}
    verify_metadata(inventory, remote_root)
    if dry_run: return {"status": "ready", "inventory": inventory, "remote_writes": False}
    receipts.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    lock = receipts / ".final_archive.lock"; seal = receipts / "FINAL_ARCHIVE.json"
    common.reject_symlink_chain(receipts, lock); common.reject_symlink_chain(receipts, seal)
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, run_id.encode()); os.fsync(fd); os.close(fd)
    partial = destination.with_name(destination.name + ".partial-" + run_id)
    record = {"status": "started", "run_id": run_id, "inventory": inventory,
              "hardlinks": [], "remote_writes": False}
    release_lock = False
    try:
        if destination.exists() or seal.exists():
            if not (destination.is_dir() and seal.is_file()):
                raise FileExistsError("partial publication exists; inspect without overwriting")
            saved = common.checked_json(seal)
            if saved.get("inventory") != inventory:
                raise ValueError("sealed archive inventory drift")
            verify_bundle(destination, inventory, remote_root)
            record["status"] = "verified_existing"
            release_lock = True
            return record
        partial.mkdir(parents=True, exist_ok=False)
        for area, values in inventory["areas"].items():
            for rel, meta in values.items():
                target = partial / area / rel; target.parent.mkdir(parents=True, exist_ok=True)
                reused = _find_reuse(source, area, rel, meta)
                if reused:
                    os.link(reused, target); record["hardlinks"].append(f"{area}/{rel}")
                else:
                    remote.fetch(meta["remote.original"], target)
                if target.stat().st_size != meta["size"] or common.digest(target) != meta["sha256"]:
                    raise ValueError("download hash/size mismatch")
        verify_bundle(partial, inventory, remote_root)
        if remote.inventory() != inventory:
            raise ValueError("remote immutable archive drift during copy")
        if destination.exists(): raise FileExistsError(destination)
        common.reject_symlink_chain(Path(destination.anchor), destination)
        os.rename(partial, destination); common.fsync_directory(destination.parent)
        record["status"] = "complete"
        common.write_json_new(seal, record); common.fsync_directory(receipts)
        release_lock = True
        return record
    except BaseException as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
        # Keep both the partial tree and exclusive lock as unexplained-failure evidence.
        raise
    finally:
        common.write_json_new(receipts / (run_id + ".json"), record)
        if release_lock:
            lock.unlink(); common.fsync_directory(receipts)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True); p.add_argument("--port", type=int, default=22); p.add_argument("--control-path")
    p.add_argument("--remote-python", default="/root/miniconda3/bin/python3")
    p.add_argument("--remote-root", required=True); p.add_argument("--remote-code", required=True)
    p.add_argument("--remote-final", required=True); p.add_argument("--remote-protocol", required=True)
    p.add_argument("--remote-selection", required=True); p.add_argument("--remote-plan", required=True)
    p.add_argument("--local-dir", required=True); p.add_argument("--receipt-dir", required=True)
    p.add_argument("--protocol-source"); p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args(); torch.set_num_threads(4)
    print(json.dumps(run(SSHRemote(args), args.local_dir, args.receipt_dir, args.remote_root,
                         args.protocol_source, args.dry_run), indent=2))
