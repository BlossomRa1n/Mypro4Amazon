"""Registered cross validation protocol for the next future-window round.

This runner is deliberately independent from the historical experiment
scripts.  It has five small, auditable stages:

``prepare`` builds a raw-ID cohort and immutable training/evaluation records;
``dev`` trains the nine registered models without opening test targets;
``compare`` applies the pre-registered paired rule; ``lock`` writes the final
plan; and ``final-train``/``final-test`` are the only path to fresh test
labels.  The implementation also supports a clearly marked ``--smoke`` mode
for the tiny CPU fixtures used by the local tests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from baseline_data import PrefixDataset, collate, fingerprint
from baseline_runtime import seed_all
from run_baseline import candidate_pools, make_model, recall, write_json
from run_token_experiments import model_config, sha256_file
from token_models import SemanticTokenDIN


PROTOCOL_VERSION = "next-cross-multiseed-20260922-v1"
COHORT_SEED = 20260922
SEEDS = (42, 43, 44)
INIT_SEEDS = {42: 424242, 43: 424243, 44: 424244}
VARIANTS = ("raw", "normalized_gated", "zero_cross")
FORMAL_COUNTS = {"train": 100000, "screen": 20000, "confirm": 80000, "test": 100000}
GATE_LOGIT = -2.944439


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return _sha_bytes(arr.tobytes())


def sha256_json(value) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode())


def atomic_torch_save(value, path: Path) -> str:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def load_data(base_run: Path):
    path = Path(base_run) / "data.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        value = pickle.load(stream)
    data = value.get("data") if isinstance(value, dict) and "data" in value else value
    if not hasattr(data, "users") or not hasattr(data, "train_positions"):
        raise ValueError("data.pkl does not contain a compatible BenchmarkData object")
    return data


def _source_record(path: Path, mode: str = "declared"):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "mode": mode}


def load_historical_users(path: str | Path) -> tuple[set[str], dict]:
    """Load and verify the raw-ID exclusion contract.

    A source is intentionally required even when a caller supplies a complete
    list by hand.  This prevents a bare hash or an untraceable comment from
    being treated as proof that a cohort is fresh.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema_version") != 1:
        raise ValueError("historical users schema_version must be 1")
    ids = obj.get("raw_user_ids")
    if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("historical users must contain a raw_user_ids list")
    if len(set(ids)) != len(ids):
        raise ValueError("historical raw_user_ids contain duplicates")
    sources = obj.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("historical users require at least one verifiable source")
    checked = []
    for source in sources:
        if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
            raise ValueError("historical source requires path and sha256")
        source_path = Path(source["path"]).expanduser()
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        actual = sha256_file(source_path)
        if actual != source["sha256"]:
            raise ValueError(f"historical source hash mismatch: {source_path}")
        checked.append(dict(source, path=str(source_path), sha256=actual))
    completeness = obj.get("completeness")
    if not isinstance(completeness, dict) or completeness.get("attested") is not True:
        raise ValueError("historical users completeness.attested must be true")
    note = str(completeness.get("note", "")).strip()
    if not note:
        raise ValueError("historical users completeness requires a note")
    normalized = dict(obj)
    normalized["raw_user_ids"] = sorted(ids)
    normalized["sources"] = checked
    normalized["source_file"] = str(path.resolve())
    normalized["source_file_sha256"] = sha256_file(path)
    return set(ids), normalized


def _data_user_raw(data, uid: int) -> str:
    value = data.users[int(uid)]
    return str(value)


def _eligible_records(data, historical: set[str]):
    # ``eval_selection`` is a deterministic permutation created by the base
    # preparation.  It does not alter the base user/item encoding.
    selection = np.asarray(getattr(data, "eval_selection", np.arange(len(data.val_users))), dtype=np.int64)
    records = []
    excluded_present = set()
    train_uids = set(np.asarray(data.uid)[np.asarray(data.train_positions)].astype(int).tolist())
    for index in selection.tolist():
        uid = int(data.val_users[int(index)])
        raw = _data_user_raw(data, uid)
        if raw in historical:
            excluded_present.add(raw)
        position = int(data.val_positions[int(index)])
        # The prefix itself may contain multiple optimization rows.  A user is
        # eligible only when at least one of those rows exists in the base
        # training positions.
        has_row = uid in train_uids
        if has_row:
            records.append((uid, position, raw))
    return records, excluded_present


def _record_hash(records: Sequence[tuple[int, int, str]]) -> str:
    return sha256_json([[int(uid), int(position), str(raw)] for uid, position, raw in records])


def _cohort_sizes(args, smoke: bool):
    values = {
        "train": int(args.train_users), "screen": int(args.screen_users),
        "confirm": int(args.confirm_users), "test": int(args.test_users),
    }
    if any(v < 1 for v in values.values()):
        raise ValueError("cohort sizes must be positive")
    if not smoke and values != FORMAL_COUNTS:
        raise ValueError(f"formal cohort sizes must be exactly {FORMAL_COUNTS}; use --smoke for a fixture")
    if values["screen"] + values["confirm"] != values["train"]:
        raise ValueError("screen + confirm must equal train users")
    return values


def prepare(args):
    smoke = bool(args.smoke)
    sizes = _cohort_sizes(args, smoke)
    if not smoke and int(args.cohort_seed) != COHORT_SEED:
        raise ValueError(f"formal cohort seed must be {COHORT_SEED}")
    base = Path(args.base_run).resolve()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise FileExistsError("prepare requires an empty evidence directory")
    data = load_data(base)
    if getattr(data, "protocol", "") != "positive-first-interaction-user-prefix80-future20-v1":
        raise ValueError("prepare requires FutureWindowData prefix80 protocol")
    if not data.manifest.get("data_id"):
        raise ValueError("base data manifest is missing data_id")
    if not smoke:
        if data.hist_len != 50: raise ValueError("actual full-base history length must be 50")
        spec = json.loads((base / "run_manifest.json").read_text())["args"]
        for key, expected in {"sample_users": 0, "dim": 256, "hist_len": 50, "cf_neighbors": 300}.items():
            if spec.get(key) != expected: raise ValueError(f"full-base {key} must equal {expected}")
        for name in ("v2_best.pth", "svd.npy", "itemcf.pkl"):
            if not (base / name).exists(): raise FileNotFoundError(base / name)
    historical, historical_manifest = load_historical_users(args.historical_users)
    eligible, excluded_present = _eligible_records(data, historical)
    fresh = [row for row in eligible if row[2] not in historical]
    if len(fresh) < sizes["test"] or len(eligible) - sizes["test"] < sizes["train"]:
        raise ValueError(f"insufficient eligible users: total={len(eligible)}, fresh={len(fresh)}, "
                         f"required test={sizes['test']}, train={sizes['train']}")
    rng = np.random.default_rng(int(args.cohort_seed))
    test_records = [fresh[int(i)] for i in rng.permutation(len(fresh))[:sizes["test"]]]
    test_uids = {row[0] for row in test_records}
    remaining = [row for row in eligible if row[0] not in test_uids]
    train_records = [remaining[int(i)] for i in rng.permutation(len(remaining))[:sizes["train"]]]
    screen_records = train_records[:sizes["screen"]]
    confirm_records = train_records[sizes["screen"]:]
    raw_sets = {name: {row[2] for row in rows} for name, rows in (
        ("test", test_records), ("train", train_records), ("screen", screen_records), ("confirm", confirm_records))}
    if raw_sets["screen"] & raw_sets["confirm"] or raw_sets["train"] != raw_sets["screen"] | raw_sets["confirm"]:
        raise AssertionError("cohort split is not disjoint")
    if raw_sets["test"] & raw_sets["train"]:
        raise AssertionError("fresh test overlaps fast-train")

    # Fast training rows are an independent view over the full base corpus.
    selected_raw = raw_sets["train"]
    selected_positions = np.asarray(
        [int(pos) for pos in np.asarray(data.train_positions).tolist()
         if _data_user_raw(data, int(data.uid[int(pos)])) in selected_raw], dtype=np.int64)
    if not len(selected_positions):
        raise ValueError("selected fast-train cohort has no optimization rows")
    seen_uids = set(np.asarray(data.uid)[selected_positions].astype(int).tolist())
    expected_uids = {int(row[0]) for row in train_records}
    if seen_uids != expected_uids:
        missing = expected_uids - seen_uids
        raise ValueError(f"fast-train users missing optimization rows: {len(missing)}")

    records = {"test": test_records, "train": train_records,
               "screen": screen_records, "confirm": confirm_records}
    for name, rows in records.items():
        (run / f"{name}_records.json").write_text(
            json.dumps([[int(u), int(p), raw] for u, p, raw in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(run / "ranker_train_positions.npy", selected_positions)
    base_path_hashes = {}
    for name in ("data.pkl", "run_manifest.json", "v2_best.pth", "svd.npy", "itemcf.pkl"):
        path = base / name
        if path.exists():
            base_path_hashes[name] = sha256_file(path)
    manifest = {
        "protocol": PROTOCOL_VERSION, "smoke": smoke, "cohort_seed": int(args.cohort_seed),
        "base_run": str(base), "base_data_id": data.manifest.get("data_id"),
        "base_data_hash": sha256_file(base / "data.pkl"), "base_assets": base_path_hashes,
        "encoders_hash": data.manifest["encoders_hash"], "code_hashes": code_hashes(),
        "historical_users": historical_manifest, "historical_intersection_count": len(excluded_present),
        "historical_intersection_raw_ids": sorted(excluded_present),
        "candidate_count": len(eligible), "counts": sizes,
        "cohort_intersections": {"screen_confirm": len(raw_sets["screen"] & raw_sets["confirm"]),
                                 "screen_test": len(raw_sets["screen"] & raw_sets["test"]),
                                 "confirm_test": len(raw_sets["confirm"] & raw_sets["test"]),
                                 "train_test": len(raw_sets["train"] & raw_sets["test"])},
        "cohort_hashes": {name: _record_hash(rows) for name, rows in records.items()},
        "cohort_raw_hashes": {name: sha256_json(sorted(raws)) for name, raws in raw_sets.items()},
        "records": {name: f"{name}_records.json" for name in records},
        "ranker_train_positions": "ranker_train_positions.npy",
        "ranker_train_positions_sha256": sha256_array(selected_positions),
        "ranker_train_rows": int(len(selected_positions)),
        "full_base_train_rows": int(len(data.train_positions)),
        "fresh_test_excludes_history_raw_ids": True,
        "test_future_labels_read": False,
        "test_records_are_locked": True,
    }
    if (run / "protocol_manifest.json").exists():
        old = json.loads((run / "protocol_manifest.json").read_text(encoding="utf-8"))
        if old.get("manifest_hash") != sha256_json({k: v for k, v in old.items() if k != "manifest_hash"}):
            raise ValueError("existing protocol manifest hash is invalid")
        if old.get("cohort_hashes") != manifest["cohort_hashes"] or old.get("base_data_id") != manifest["base_data_id"]:
            raise ValueError("run directory already contains a different cohort or base data")
    manifest["manifest_hash"] = sha256_json(manifest)
    write_json(run / "protocol_manifest.json", manifest)
    write_json(run / "historical_users_verified.json", historical_manifest)
    return manifest


def load_protocol(path: str | Path):
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    expected = obj.get("manifest_hash")
    actual = sha256_json({k: v for k, v in obj.items() if k != "manifest_hash"})
    if not expected or expected != actual:
        raise ValueError("protocol manifest hash mismatch")
    if obj.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol manifest")
    for name in obj.get("records", {}).values():
        if not (path.parent / name).exists():
            raise FileNotFoundError(path.parent / name)
    name = obj["ranker_train_positions"]
    if not (path.parent / name).exists():
        raise FileNotFoundError(path.parent / name)
    records = {label: _records(path.parent, obj, label) for label in ("train", "screen", "confirm", "test")}
    uids = {label: {r[0] for r in rows} for label, rows in records.items()}
    if uids["train"] != uids["screen"] | uids["confirm"] or uids["screen"] & uids["confirm"] or uids["test"] & uids["train"]:
        raise ValueError("cohort intersection violation")
    historical = set(obj["historical_users"]["raw_user_ids"])
    if historical & {r[2] for r in records["test"]}: raise ValueError("test is not fresh")
    positions = np.load(path.parent / obj["ranker_train_positions"])
    if sha256_array(positions) != obj["ranker_train_positions_sha256"]:
        raise ValueError("training positions hash mismatch")
    return obj


def _records(run: Path, manifest: dict, name: str):
    rows = json.loads((run / manifest["records"][name]).read_text(encoding="utf-8"))
    rows = [(int(uid), int(position), str(raw)) for uid, position, raw in rows]
    if len(rows) != manifest["counts"][name] or _record_hash(rows) != manifest["cohort_hashes"][name]:
        raise ValueError(f"{name} record identity/hash mismatch")
    if len({row[0] for row in rows}) != len(rows): raise ValueError("duplicate cohort UID")
    return rows


def code_hashes():
    root = Path(__file__).parent
    return {p.name: sha256_file(p) for p in sorted(root.glob("*.py"))}


def validate_base(base, protocol):
    base = Path(base).resolve()
    if base != Path(protocol["base_run"]).resolve(): raise ValueError("base path mismatch")
    for name, digest in protocol["base_assets"].items():
        if sha256_file(base / name) != digest: raise ValueError(f"base asset drift: {name}")
    if protocol.get("code_hashes") != code_hashes(): raise ValueError("protocol code drift")
    data = load_data(base)
    if data.manifest["data_id"] != protocol["base_data_id"] or data.manifest["encoders_hash"] != protocol["encoders_hash"]:
        raise ValueError("base data or encoding identity mismatch")
    historical = protocol["historical_users"]
    if sha256_file(historical["source_file"]) != historical["source_file_sha256"]: raise ValueError("historical source drift")
    load_historical_users(historical["source_file"])
    return data


def _source_hashes(base: Path):
    result = {}
    for name in ("v2_best.pth", "svd.npy", "itemcf.pkl", "run_manifest.json", "data.pkl"):
        path = base / name
        result[name] = sha256_file(path) if path.exists() else "missing"
    return result


def _catalog(data):
    active = np.asarray(getattr(data, "active_items", np.arange(2, len(data.items))), dtype=np.int64)
    counts = np.asarray(getattr(data, "train_counts", np.zeros(len(data.items))), dtype=np.int64)
    return sorted((int(item) for item in active if int(item) >= 2), key=lambda item: (-int(counts[item]), item))


def _build_pools(data, records: Sequence[tuple[int, int, str]], budget: int):
    catalog = _catalog(data)
    pools = []
    for uid, _, _ in records:
        known = set(data.train_sets[int(uid)])
        pool = [item for item in catalog if item not in known][:int(budget)]
        if len(pool) < min(int(budget), 5):
            # A tiny fixture can have fewer active products; retaining every
            # available item is still auditable and keeps metrics finite.
            pool = [item for item in catalog if item not in known]
        pools.append(pool)
    return pools


def _build_recall_pools(data, base: Path, records, budget: int, weights, smoke: bool):
    """Build the registered four-channel RRF pools for a real base run.

    The compact hot-catalog implementation is intentionally available only to
    smoke fixtures.  A formal run must have both the V2 checkpoint and ItemCF
    asset, and delegates channel construction to the versioned baseline helper
    (which uses prefix history only).
    """
    if smoke:
        return _build_pools(data, records, budget)
    for name in ("v2_best.pth", "itemcf.pkl", "run_manifest.json"):
        if not (base / name).exists():
            raise FileNotFoundError(f"formal candidate construction requires {base / name}")
    with (base / "itemcf.pkl").open("rb") as stream:
        cf = pickle.load(stream)
    if not smoke and np.asarray(cf[0]).shape[1] != 300:
        raise ValueError("formal ItemCF must have 300 neighbors")
    pairs = [(int(uid), int(position)) for uid, position, _ in records]
    if weights[1] == 0:
        v2_scores = [{} for _ in pairs]
    else:
        source_args = json.loads((base / "run_manifest.json").read_text(encoding="utf-8"))["args"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        v2 = make_model(data, SimpleNamespace(**source_args), "v2", device)
        saved = torch.load(base / "v2_best.pth", map_location="cpu", weights_only=False)
        v2.load_state_dict(saved["model"]); v2.eval()
        _, v2_scores = recall(v2, data, pairs, device, budget)
        del v2, saved
    pools, _ = candidate_pools(data, pairs, v2_scores, cf, budget,
                               fusion_mode="rrf", half_life_days=180.,
                               fusion_weights=list(weights))
    return [list(pool) for pool in pools]


def _pool_hash(pools):
    return sha256_json([[int(item) for item in pool] for pool in pools])


def _write_pools(path: Path, records, pools, source):
    value = {"records_hash": _record_hash(records), "pool_hash": _pool_hash(pools),
             "position_uid_row_hash": _record_hash(records), "source": source,
             "budget": max((len(x) for x in pools), default=0),
             "shape": [len(pools), max((len(x) for x in pools), default=0)],
             "pools": pools}
    # Hash the canonical payload (rather than its own self-referential JSON
    # wrapper) so a cache can be checked without trusting its prose fields.
    value["payload_sha256"] = sha256_json({"records_hash": value["records_hash"],
                                           "pool_hash": value["pool_hash"],
                                           "shape": value["shape"], "pools": value["pools"]})
    write_json(path, value)
    return value


def _load_or_make_assets(data, run: Path, manifest: dict, records, label, budget):
    path = run / f"candidate_{label}.json"
    weights = [2.0, 0.0, 0.7, 0.05] if label in ("train", "final_train") else [2.0, 1.0, 0.7, 0.05]
    source = {"base_data_id": manifest["base_data_id"], "source_hashes": manifest["base_assets"],
              "records_hash": _record_hash(records), "budget": int(budget),
              "candidate_policy": "four-way RRF from train corpus; no future target filtering",
              "rrf_weights": weights, "itemcf_half_life_days": 180.0,
              "smoke_fallback": bool(manifest.get("smoke"))}
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("records_hash") != source["records_hash"] or obj.get("source") != source:
            raise ValueError(f"candidate cache provenance mismatch: {path}")
        pools = obj.get("pools")
        if not isinstance(pools, list) or len(pools) != len(records):
            raise ValueError(f"candidate cache row count mismatch: {path}")
        if obj.get("pool_hash") != _pool_hash(pools):
            raise ValueError(f"candidate cache content hash mismatch: {path}")
        expected_payload = sha256_json({"records_hash": obj.get("records_hash"),
                                       "pool_hash": obj.get("pool_hash"),
                                       "shape": obj.get("shape"), "pools": pools})
        if obj.get("payload_sha256") != expected_payload:
            raise ValueError(f"candidate cache payload hash mismatch: {path}")
        return obj["pools"], obj
    pools = _build_recall_pools(data, Path(manifest["base_run"]), records, budget, weights,
                                bool(manifest.get("smoke")))
    return pools, _write_pools(path, records, pools, source)


def _load_factors(base: Path, data, dim: int, smoke: bool):
    path = base / "svd.npy"
    if path.exists():
        factors = np.asarray(np.load(path), dtype=np.float32)
        if factors.shape != (len(data.items), dim):
            raise ValueError(f"SVD shape {factors.shape} does not match {(len(data.items), dim)}")
        return factors, {"path": str(path), "sha256": sha256_file(path), "synthetic": False}
    if not smoke:
        raise FileNotFoundError(path)
    generator = np.random.default_rng(20260922)
    factors = generator.normal(0, 0.02, size=(len(data.items), dim)).astype(np.float32)
    factors[:2] = 0
    return factors, {"path": "synthetic-smoke-svd", "sha256": sha256_array(factors), "synthetic": True}


def _model_args(data, args):
    return SimpleNamespace(dim=int(args.dim), hist_len=int(args.hist_len), token_dim=int(args.token_dim))


def _config(data, args, mode):
    config = model_config(data, _model_args(data, args), "concat")
    config["cross_mode"] = mode
    return config


def _template(data, args, init_seed: int, factors: np.ndarray):
    seed_all(init_seed)
    template = SemanticTokenDIN(**_config(data, args, "raw"))
    with torch.no_grad():
        template.item_embedding.weight.copy_(torch.as_tensor(factors))
        template.item_embedding.weight[:2].zero_()
        template.user_embedding.weight[:2].zero_()
    state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    return template, state


def _new_variant(data, args, template_state, mode):
    model = SemanticTokenDIN(**_config(data, args, mode))
    state = {key: value.clone() for key, value in template_state.items()}
    state["cross_gate_logit"] = torch.tensor(GATE_LOGIT if mode == "normalized_gated" else 10.0,
                                               dtype=state["cross_gate_logit"].dtype)
    model.load_state_dict(state, strict=True)
    return model


def _state_hashes(model):
    return {key: _sha_bytes(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
            for key, value in model.state_dict().items()}


def _audit_initial_states(models: dict[str, torch.nn.Module], template_state):
    hashes = {name: _state_hashes(model) for name, model in models.items()}
    keys = sorted(template_state)
    for key in keys:
        if key == "cross_gate_logit":
            continue
        values = [model.state_dict()[key].detach().cpu() for model in models.values()]
        if not all(torch.equal(values[0], value) for value in values[1:]):
            raise AssertionError(f"initial state differs outside gate: {key}")
    gates = {name: float(model.cross_gate_logit.detach().cpu()) for name, model in models.items()}
    if abs(gates["raw"] - gates["zero_cross"]) > 1e-8 or abs(gates["raw"] - gates["normalized_gated"]) < 1e-4:
        raise AssertionError("cross gate differs outside the registered whitelist")
    return {"state_sha256": hashes, "gate_values": gates,
            "copied_tensor_keys": keys, "gate_difference_whitelist": ["cross_gate_logit"]}


def _batch_chunks(order: np.ndarray, batch_size: int):
    chunks = [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2].extend(chunks[-1]); chunks.pop()
    return chunks


class TracedPrefixDataset(PrefixDataset):
    """Same registered band/fallback policy, with actual source accounting."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_sources = {}

    def _mixed_negatives(self, idx, uid, target, rng):
        seen = set(self.data.train_sets[uid]); seen.add(int(target))
        pool = self._ranked_pool(idx)
        selected = []
        counts = {"band11_25": 0, "band26_50": 0, "candidate_fallback": 0, "random": 0}
        for (start, stop, quota), label in zip(self.candidate_ranges, ("band11_25", "band26_50")):
            eligible = sorted(set(int(x) for x in pool[start - 1:stop] if int(x) >= 2 and int(x) not in seen))
            chosen = rng.choice(eligible, min(quota, len(eligible)), replace=False)
            selected.extend(map(int, chosen)); seen.update(map(int, chosen)); counts[label] += len(chosen)
        missing = 4 - len(selected)
        eligible = sorted(set(int(x) for x in pool if int(x) >= 2 and int(x) not in seen))
        chosen = rng.choice(eligible, min(missing, len(eligible)), replace=False)
        selected.extend(map(int, chosen)); seen.update(map(int, chosen)); counts["candidate_fallback"] = len(chosen)
        random_count = 16 - len(selected)
        random = self.data.negatives(uid, target, random_count, rng, excluded=seen)
        selected.extend(map(int, random)); counts["random"] = random_count
        if len(selected) != 16 or len(set(selected)) != 16:
            raise AssertionError("invalid mixed negative stream")
        self.last_sources[int(idx)] = counts
        return np.asarray(selected, dtype=np.int64)


def _train_model(data, args, run: Path, seed: int, variant: str, template_state, factors,
                 train_positions, train_candidate_pools, screen_records, screen_pools, smoke=False):
    directory = run / f"seed_{seed}" / variant
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite training evidence: {directory}")
    init_seed = INIT_SEEDS.get(seed, 424242 + int(seed) - 42)
    model = _new_variant(data, args, template_state, {
        "raw": "raw", "normalized_gated": "normalized_gated", "zero_cross": "zero_cross"}[variant])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Shallow view shares the immutable train-corpus arrays, while the selected
    # rows and sampler seed are local to this training invocation.
    view = copy.copy(data)
    view.train_positions = np.asarray(train_positions, dtype=np.int64)
    view.seed = int(seed)
    dataset = TracedPrefixDataset(view, negatives=16, negative_policy="mixed_rrf",
                            candidate_pools=train_candidate_pools,
                            candidate_ranges=((11, 25, 2), (26, 50, 2)))
    seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=1e-6)
    history = []
    traces = {}
    for epoch in range(int(args.epochs)):
        model.train()
        dataset.epoch = epoch
        order = np.random.default_rng(seed + epoch).permutation(len(dataset)).astype(np.int64)
        digest = hashlib.sha256()
        source_counts = {"band11_25": 0, "band26_50": 0, "candidate_fallback": 0, "random": 0}
        consumed_uids = set()
        chunks = _batch_chunks(order, int(args.batch_size))
        total = 0.0
        started = time.monotonic()
        for indexes in chunks:
            rows = [dataset[int(index)] for index in indexes]
            for index, row in zip(indexes, rows):
                position = int(view.train_positions[int(index)])
                uid = int(view.uid[position]); consumed_uids.add(uid)
                digest.update(np.asarray([int(index), position, uid], dtype=np.int64).tobytes())
                digest.update(np.asarray(row["neg_item_id"], dtype=np.int64).tobytes())
                for key, count in dataset.last_sources[int(index)].items():
                    source_counts[key] += count
            batch = {key: value.to(device) for key, value in collate(rows).items()}
            optimizer.zero_grad(set_to_none=True)
            users = {key: batch[key] for key in (
                "user_id", "hist_items", "hist_brands", "hist_ratings", "hist_time_deltas",
                "hist_verified", "hist_len", "click_count", "time_span", "user_avg_rating",
                "user_std_rating", "user_verified_ratio", "user_avg_helpful")}
            positive = {key: batch["pos_" + key] for key in (
                "item_id", "category_id", "brand_id", "item_click_count", "created_at_ts",
                "item_avg_rating", "item_rating_number")}
            negative = {key: batch["neg_" + key] for key in positive}
            pos_score, neg_score = model.forward_bpr(users, positive, negative)
            loss = F.softplus(neg_score.float() - pos_score.float()[:, None]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss in {variant} seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        expected_uids = set(np.asarray(data.uid)[np.asarray(train_positions)].astype(int).tolist())
        if consumed_uids != expected_uids:
            raise AssertionError("actual training stream did not cover every selected user")
        trace = {"sha256": digest.hexdigest(), "rows": int(len(order)),
                 "order_sha256": sha256_array(order), "batch_sizes": [len(c) for c in chunks],
                 "negative_sources": source_counts, "covered_users": len(consumed_uids)}
        trace_path = run / f"seed_{seed}" / f"epoch_{epoch + 1}_trace.json"
        if trace_path.exists() and json.loads(trace_path.read_text()) != trace:
            raise ValueError("actual consumed stream differs across variants")
        if not trace_path.exists(): write_json(trace_path, trace)
        traces[str(epoch + 1)] = trace
        write_json(directory / f"epoch_{epoch + 1}_trace.json", trace)
        scheduler.step()
        model.eval()
        metrics, arrays = evaluate(data, model, screen_records, screen_pools)
        np.savez_compressed(directory / f"screen_epoch{epoch + 1}_users.npz",
                            uid=np.asarray([int(x[0]) for x in screen_records], dtype=np.int64),
                            position=np.asarray([int(x[1]) for x in screen_records], dtype=np.int64), **arrays)
        history.append({"epoch": epoch + 1, "loss": total / max(len(chunks), 1),
                        "steps": len(chunks), "seconds": time.monotonic() - started,
                        "screen": metrics,
                        "cross_gate": float(torch.sigmoid(model.cross_gate_logit).detach()),
                        "cross_token_norm": cross_token_norm(data, model, screen_records),
                        "cross_type_offset_norm": float(model.token_type[4].detach().norm())})
        write_json(directory / "history.json", history)
    manifest = {"protocol": PROTOCOL_VERSION, "variant": variant, "seed": int(seed),
                "init_seed": int(init_seed), "epochs": int(args.epochs), "train_rows": len(train_positions),
                "train_positions_sha256": sha256_array(np.asarray(train_positions)),
                "model_config": _config(data, args, variant), "negative_policy": "12 random + 2 ranks11-25 + 2 ranks26-50",
                "trace_files": traces, "screen_pool_hash": _pool_hash(screen_pools),
                "base_data_id": data.manifest.get("data_id")}
    checkpoint_hash = atomic_torch_save({"model": model.state_dict(), "manifest": manifest,
                                         "history": history, "epoch": int(args.epochs)}, directory / "last.pth")
    saved = torch.load(directory / "last.pth", map_location="cpu", weights_only=False)
    for key, tensor in saved["model"].items():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor.cpu(), model.state_dict()[key].cpu()):
            raise ValueError(f"checkpoint reload/finite validation failed: {key}")
    del saved
    manifest["checkpoint_sha256"] = checkpoint_hash
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "COMPLETED.json", {"status": "complete", "checkpoint": "last.pth",
                                               "checkpoint_sha256": checkpoint_hash, "epoch": int(args.epochs)})
    return manifest


def cross_token_norm(data, model, records):
    uid, position, _ = records[0]
    device = next(model.parameters()).device
    user = {k: v.to(device) for k, v in collate([data.user_features(uid, position)]).items()}
    # A training-prefix item makes this diagnostic independent of future labels.
    item = int(data.iid[int(data.starts[uid])])
    features = {k: torch.as_tensor(v, device=device) for k, v in data.item_features([item]).items()}
    with torch.inference_mode():
        return float(model._tokenize(model._encode_user(user), features)[:, 4].norm(dim=-1).mean())


def evaluate(data, model, records, pools):
    if len(records) != len(pools): raise ValueError("candidate row count mismatch")
    for pool in pools:
        if len(set(pool)) != len(pool) or any(int(item) < 2 or int(item) >= len(data.items) for item in pool):
            raise ValueError("candidate IDs must be unique valid catalog IDs")
    model.eval()
    hits, ndcgs, pool_hits = [], [], []
    with torch.inference_mode():
        for (uid, position, _), pool in zip(records, pools):
            user = collate([data.user_features(int(uid), int(position))])
            device = next(model.parameters()).device
            user = {key: value.to(device) for key, value in user.items()}
            encoded = model._encode_user(user)
            ids = [int(item) for item in pool]
            if not ids:
                ranking = []
            else:
                encoded = {key: (value.expand(len(ids), *value.shape[1:])
                                  if torch.is_tensor(value) and value.shape[0] == 1 else value)
                           for key, value in encoded.items()}
                items = {key: torch.as_tensor(value, device=device) for key, value in data.item_features(ids).items()}
                scores = model._score_item(encoded, items).float().cpu().numpy().tolist()
                if not np.all(np.isfinite(np.asarray(scores, dtype=np.float64))):
                    raise FloatingPointError("nonfinite candidate score")
                ranking = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))]
            targets = set(int(item) for item in data.targets([(int(uid), int(position))])[0])
            top = ranking[:5]
            hit = bool(set(top) & targets)
            hits.append(int(hit)); pool_hits.append(int(bool(set(ids) & targets)))
            ndcgs.append(sum(1.0 / math.log2(i + 2) for i, item in enumerate(top) if item in targets) /
                         max(sum(1.0 / math.log2(i + 2) for i in range(min(5, len(targets)))), 1e-12))
    hits_arr = np.asarray(hits, dtype=np.int8)
    ndcg_arr = np.asarray(ndcgs, dtype=np.float32)
    pool_arr = np.asarray(pool_hits, dtype=np.int8)
    metrics = {"hr5": float(hits_arr.mean()) if len(hits_arr) else 0.0,
               "ndcg5": float(ndcg_arr.mean()) if len(ndcg_arr) else 0.0,
               "pool_hit": float(pool_arr.mean()) if len(pool_arr) else 0.0,
               "users": int(len(records))}
    return metrics, {"hit5": hits_arr, "ndcg5": ndcg_arr, "pool_hit": pool_arr}


def _validate_dev_args(args, manifest):
    if not manifest.get("smoke") and (int(args.epochs) != 3 or tuple(args.seeds) != SEEDS or tuple(args.variants) != VARIANTS or
                                       int(args.dim) != 256 or int(args.token_dim) != 256 or int(args.hist_len) != 50 or int(args.batch_size) != 256 or int(args.candidates) != 75):
        raise ValueError("formal dev requires fixed 256 dimensions, hist_len 50, batch 256, seeds 42/43/44, three variants and 3 epochs")
    if int(args.epochs) < 1:
        raise ValueError("epochs must be positive")


def dev(args):
    protocol_path = Path(args.protocol_manifest).resolve()
    manifest = load_protocol(protocol_path)
    _validate_dev_args(args, manifest)
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()): raise FileExistsError("dev requires an empty evidence directory")
    torch.set_num_threads(4)
    base = Path(args.base_run).resolve()
    if str(base) != str(Path(manifest["base_run"]).resolve()):
        raise ValueError("base-run does not match protocol manifest")
    data = validate_base(base, manifest)
    if data.manifest.get("data_id") != manifest.get("base_data_id"):
        raise ValueError("base data_id differs from protocol")
    screen = _records(protocol_path.parent, manifest, "screen")
    confirm = _records(protocol_path.parent, manifest, "confirm")
    # Test records are deliberately not loaded here.  This is the guard that
    # keeps a replacement data.targets(test_records) from being called in dev.
    # Estimate before candidate caches allocate disk/RAM or any training begins.
    with torch.device("meta"):
        sizing_model = SemanticTokenDIN(**_config(data, args, "raw"))
    tensor_bytes = sum(x.numel() * x.element_size() for x in sizing_model.state_dict().values())
    del sizing_model
    same_filesystem = run.stat().st_dev == protocol_path.parent.stat().st_dev
    pool_bytes = (manifest["ranker_train_rows"] + len(screen) + len(confirm)
                  + len(data.train_positions) + manifest["counts"]["test"]) * 75 * 32
    # Nine dev files plus two final copies per variant remain retained. Reserve
    # one extra model and a second cache payload for atomic-write peaks.
    estimate = 14 * tensor_bytes + 2 * pool_bytes
    free = shutil.disk_usage(run).free
    cache_free = shutil.disk_usage(protocol_path.parent).free
    write_json(run / "disk_preflight.json", {"model_tensor_bytes": tensor_bytes,
        "retained_models": 13, "atomic_peak_models": 14, "candidate_pool_estimate_bytes": pool_bytes,
        "candidate_atomic_peak_bytes": 2 * pool_bytes, "same_filesystem": same_filesystem,
        "required_estimate_bytes": estimate, "free_bytes": free, "cache_free_bytes": cache_free,
        "note": "JSON candidate pools require substantial host RAM; this is a conservative disk estimate, not a capacity guarantee"})
    if same_filesystem:
        if free < estimate: raise OSError(f"insufficient disk: need estimated {estimate}, available {free}")
    elif free < 14 * tensor_bytes or cache_free < 2 * pool_bytes:
        raise OSError("insufficient model or protocol-cache filesystem capacity")
    screen_pools, screen_cache = _load_or_make_assets(data, protocol_path.parent, manifest, screen, "screen", int(args.candidates))
    confirm_pools, confirm_cache = _load_or_make_assets(data, protocol_path.parent, manifest, confirm, "confirm", int(args.candidates))
    train_positions = np.load(protocol_path.parent / manifest["ranker_train_positions"], mmap_mode="r")
    train_records = _records(protocol_path.parent, manifest, "train")
    expected_positions = data.train_positions[np.isin(data.uid[data.train_positions], [r[0] for r in train_records])]
    if not np.array_equal(train_positions, expected_positions): raise ValueError("selected prefix rows differ from cohort")
    boundaries = dict(zip(map(int, data.val_users), map(int, data.val_positions)))
    for records in (train_records, screen, confirm):
        for uid, position, raw in records:
            if data.users[uid] != raw or boundaries.get(uid) != position: raise ValueError("cohort raw/encoded identity mismatch")
    train_catalog_records = [(int(data.uid[int(pos)]), int(pos), _data_user_raw(data, int(data.uid[int(pos)])))
                             for pos in train_positions.tolist()]
    train_pools, train_cache = _load_or_make_assets(data, protocol_path.parent, manifest,
                                                     train_catalog_records, "train", max(75, int(args.candidates)))
    factors, factor_source = _load_factors(base, data, int(args.dim), bool(manifest.get("smoke")))
    # The pool cache for training is indexed by selected positions, not by
    # the fast-train user's split record.  Its sidecar therefore includes the
    # exact row hash and cannot be accidentally reused for another cohort.
    train_pool_hash = _pool_hash(train_pools)
    if len(train_pools) != len(train_positions):
        raise ValueError("training candidate pool does not match selected rows")
    # Build one complete template per seed, then explicitly copy every tensor
    # into all three variants.  This keeps the only allowed initial difference
    # (the registered gate value) visible in an audit file.
    dev_manifest = {"protocol": PROTOCOL_VERSION, "protocol_manifest": str(protocol_path),
                    "protocol_manifest_hash": manifest["manifest_hash"], "base_data_id": manifest["base_data_id"],
                    "seeds": [int(x) for x in args.seeds], "variants": list(args.variants),
                    "epochs": int(args.epochs), "smoke": bool(manifest.get("smoke")), "test_accessed": False,
                    "candidate_cache": {"screen": screen_cache, "confirm": confirm_cache, "train": train_cache},
                    "factor_source": factor_source, "screen_pool_hash": _pool_hash(screen_pools),
                    "confirm_pool_hash": _pool_hash(confirm_pools), "train_pool_hash": train_pool_hash}
    dev_manifest["config"] = {key: getattr(args, key) for key in ("dim", "token_dim", "hist_len", "batch_size", "epochs", "candidates")}
    dev_manifest["code_hashes"] = code_hashes()
    dev_manifest["candidate_cache"] = {
        label: {"path": str(protocol_path.parent / f"candidate_{label}.json"),
                "sha256": sha256_file(protocol_path.parent / f"candidate_{label}.json"),
                "pool_hash": value["pool_hash"], "records_hash": value["records_hash"]}
        for label, value in dev_manifest["candidate_cache"].items()}
    for seed in args.seeds:
        seed = int(seed)
        if seed not in INIT_SEEDS and not manifest.get("smoke"):
            raise ValueError(f"unsupported formal seed {seed}")
        seed_dir = run / f"seed_{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
        template, template_state = _template(data, args, INIT_SEEDS.get(seed, 424242 + seed - 42), factors)
        del template
        audit = {"state_sha256": {}, "gate_values": {}, "copied_tensor_keys": sorted(template_state),
                 "gate_difference_whitelist": ["cross_gate_logit"]}
        for variant in args.variants:
            model = _new_variant(data, args, template_state, variant)
            for key, value in model.state_dict().items():
                if key != "cross_gate_logit" and not torch.equal(value.cpu(), template_state[key]):
                    raise AssertionError(f"template copy mismatch: {key}")
            audit["state_sha256"][variant] = _state_hashes(model)
            audit["gate_values"][variant] = float(model.cross_gate_logit.detach())
            del model
        write_json(seed_dir / "initial_state_audit.json", audit)
        for variant in args.variants:
            _train_model(data, args, run, seed, variant, template_state, factors,
                         train_positions, train_pools, screen, screen_pools, bool(manifest.get("smoke")))
    # Confirm is intentionally evaluated only after all nine epoch-3
    # checkpoints have been written.  No test target or test record is read.
    for seed in args.seeds:
        seed = int(seed)
        for variant in args.variants:
            directory = run / f"seed_{seed}" / variant
            saved = torch.load(directory / "last.pth", map_location="cpu", weights_only=False)
            model = SemanticTokenDIN(**saved["manifest"]["model_config"])
            model.load_state_dict(saved["model"]); model.to("cuda" if torch.cuda.is_available() else "cpu"); model.eval()
            metrics, arrays = evaluate(data, model, confirm, confirm_pools)
            np.savez_compressed(directory / "confirm_users.npz",
                                uid=np.asarray([int(x[0]) for x in confirm], dtype=np.int64),
                                position=np.asarray([int(x[1]) for x in confirm], dtype=np.int64), **arrays)
            write_json(directory / "confirm_metrics.json", metrics)
            del model
    dev_manifest["checkpoint_hashes"] = {
        f"{seed}/{variant}": sha256_file(run / f"seed_{seed}" / variant / "last.pth")
        for seed in args.seeds for variant in args.variants}
    dev_manifest["test_accessed"] = False
    dev_manifest["evidence_hashes"] = {str(p.relative_to(run)): sha256_file(p) for p in sorted(run.rglob("*")) if p.is_file() and p.name not in ("dev_manifest.json", "COMPLETED.json")}
    dev_manifest["manifest_hash"] = sha256_json(dev_manifest)
    write_json(run / "dev_manifest.json", dev_manifest)
    write_json(run / "COMPLETED.json", {"status": "complete", "test_accessed": False,
                                         "groups": len(args.seeds) * len(args.variants)})
    return dev_manifest


def _checked_json(path, hash_key=None):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if hash_key and value.get(hash_key) != sha256_json({k: v for k, v in value.items() if k != hash_key}):
        raise ValueError(f"{hash_key} mismatch: {path}")
    return value


def _exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _code_hashes():
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _verify_base(protocol):
    base = Path(protocol["base_run"])
    for name, expected in protocol["base_assets"].items():
        if sha256_file(base / name) != expected:
            raise ValueError(f"base asset changed: {name}")
    if sha256_file(base / "data.pkl") != protocol["base_data_hash"]:
        raise ValueError("base data changed")


def _validate_arrays(value, records=None):
    required = ("uid", "position", "hit5", "ndcg5", "pool_hit")
    if any(key not in value for key in required):
        raise ValueError("missing paired array")
    n = len(value["uid"])
    if n == 0 or any(value[key].ndim != 1 or len(value[key]) != n for key in required):
        raise ValueError("invalid paired array shape")
    if len(set(value["uid"].tolist())) != n:
        raise ValueError("duplicate paired user")
    if any(not np.all(np.isfinite(value[key])) for key in required):
        raise ValueError("nonfinite paired evidence")
    if any(not np.all(np.isin(value[key], [0, 1])) for key in ("hit5", "pool_hit")):
        raise ValueError("nonbinary hit evidence")
    if np.any(value["ndcg5"] < 0) or np.any(value["ndcg5"] > 1) or np.any(value["hit5"] > value["pool_hit"]):
        raise ValueError("invalid metric evidence")
    if records is not None and (not np.array_equal(value["uid"], [r[0] for r in records]) or
                                not np.array_equal(value["position"], [r[1] for r in records])):
        raise ValueError("evaluation cohort differs from registered records")


def _validate_dev_evidence(dev):
    manifest = _checked_json(dev / "dev_manifest.json", "manifest_hash")
    protocol_path = Path(manifest["protocol_manifest"])
    protocol = load_protocol(protocol_path)
    if manifest.get("protocol") != PROTOCOL_VERSION or manifest.get("protocol_manifest_hash") != protocol["manifest_hash"]:
        raise ValueError("development protocol mismatch")
    if manifest.get("smoke") != protocol.get("smoke") or manifest.get("test_accessed") is not False:
        raise ValueError("development smoke/test identity mismatch")
    if tuple(manifest["seeds"]) != SEEDS or tuple(manifest["variants"]) != VARIANTS:
        raise ValueError("development requires nine unique registered groups")
    if not protocol.get("smoke") and (manifest["epochs"] != 3 or protocol["counts"] != FORMAL_COUNTS):
        raise ValueError("formal epochs/cohort counts mismatch")
    complete = _checked_json(dev / "COMPLETED.json")
    if complete.get("status") != "complete" or complete.get("groups") != 9 or complete.get("test_accessed") is not False:
        raise ValueError("development is incomplete")
    evidence = manifest.get("evidence_hashes")
    if not evidence:
        raise ValueError("development has no sealed evidence hashes; rerun dev with current runner")
    actual_files = {str(p.relative_to(dev)) for p in dev.rglob("*") if p.is_file() and p.name not in ("dev_manifest.json", "COMPLETED.json")}
    if not actual_files.issubset(evidence):
        raise ValueError("unsealed development evidence")
    for relative, expected in evidence.items():
        path = (dev / relative).resolve()
        if not path.is_relative_to(dev) or sha256_file(path) != expected:
            raise ValueError(f"development evidence hash mismatch: {relative}")
    if manifest.get("code_hashes") != protocol.get("code_hashes") or manifest.get("code_hashes") != _code_hashes():
        raise ValueError("development code drift")
    if not protocol.get("smoke") and manifest.get("config") != {"dim": 256, "token_dim": 256, "hist_len": 50, "batch_size": 256, "epochs": 3, "candidates": 75}:
        raise ValueError("development configuration drift")
    for receipt in manifest["candidate_cache"].values():
        if sha256_file(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError("candidate cache changed")
    expected_records = {name: _records(protocol_path.parent, protocol, name) for name in ("screen", "confirm")}
    for name, records in expected_records.items():
        if len(records) != protocol["counts"][name]:
            raise ValueError("registered cohort size mismatch")
    for seed in SEEDS:
        seed_dir = dev / f"seed_{seed}"
        audit = _checked_json(seed_dir / "initial_state_audit.json")
        hashes = audit["state_sha256"]
        for key in audit["copied_tensor_keys"]:
            if key != "cross_gate_logit" and len({hashes[v][key] for v in VARIANTS}) != 1:
                raise ValueError("shared initialization mismatch")
        trace_reference = None
        for variant in VARIANTS:
            directory = seed_dir / variant
            m = _checked_json(directory / "manifest.json")
            checkpoint_hash = sha256_file(directory / "last.pth")
            done = _checked_json(directory / "COMPLETED.json")
            if done.get("status") != "complete" or done.get("checkpoint_sha256") != checkpoint_hash or m.get("checkpoint_sha256") != checkpoint_hash or manifest["checkpoint_hashes"].get(f"{seed}/{variant}") != checkpoint_hash:
                raise ValueError("checkpoint provenance mismatch")
            if m.get("epochs") != manifest["epochs"] or m.get("seed") != seed or m.get("variant") != variant or m.get("base_data_id") != protocol["base_data_id"]:
                raise ValueError("model manifest mismatch")
            if m.get("train_positions_sha256") != protocol["ranker_train_positions_sha256"]:
                raise ValueError("training cohort mismatch")
            traces = m["trace_files"]
            if len(traces) != manifest["epochs"]:
                raise ValueError("missing epoch trace")
            for epoch, trace in traces.items():
                trace_path = directory / f"epoch_{epoch}_trace.json"
                if _checked_json(trace_path) != trace or trace["rows"] != protocol["ranker_train_rows"]:
                    raise ValueError("invalid consumed trace")
            if trace_reference is None:
                trace_reference = traces
            elif traces != trace_reference:
                raise ValueError("training streams differ between variants")
            history = _checked_json(directory / "history.json")
            if len(history) != manifest["epochs"] or any(not math.isfinite(float(row["loss"])) for row in history):
                raise ValueError("invalid training history")
            for name in ("screen", "confirm"):
                label = f"screen_epoch{manifest['epochs']}" if name == "screen" else "confirm"
                _validate_arrays(_load_arrays(directory, label), expected_records[name])
    return manifest, protocol


def _bootstrap_ci(delta: np.ndarray, seed: int, reps: int = 10000, alpha: float = 0.0125):
    if len(delta) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    # Integer bootstrap counts are materially cheaper than allocating a
    # (reps x users) matrix for the 80k formal confirm set.
    values = np.asarray(delta, dtype=np.float64)
    means = np.empty(reps, dtype=np.float64)
    chunk = 250
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))]


def _load_arrays(directory: Path, name: str):
    path = directory / f"{name}_users.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as value:
        result = {key: np.asarray(value[key]) for key in value.files}
    _validate_arrays(result)
    return result


def _paired(candidate, baseline, seed):
    _validate_arrays(candidate)
    _validate_arrays(baseline)
    for key in ("uid", "position", "pool_hit"):
        if not np.array_equal(candidate[key], baseline[key]):
            raise ValueError(f"paired {key} identity mismatch")
    if len(set(zip(candidate["uid"].tolist(), candidate["position"].tolist()))) != len(candidate["uid"]):
        raise ValueError("paired evaluation contains duplicate user/position rows")
    for value in (candidate, baseline):
        if not np.all(np.isin(value["hit5"], [0, 1])) or not np.all(np.isin(value["pool_hit"], [0, 1])):
            raise ValueError("paired hit metrics must be finite binary arrays")
        if not np.all(np.isfinite(value["ndcg5"])) or np.any(value["ndcg5"] < 0) or np.any(value["ndcg5"] > 1):
            raise ValueError("paired ndcg5 must be finite in [0,1]")
    if not np.all(np.isfinite(candidate["ndcg5"])) or not np.all(np.isfinite(baseline["ndcg5"])):
        raise ValueError("nonfinite paired metric")
    delta_hr = candidate["hit5"].astype(np.float64) - baseline["hit5"].astype(np.float64)
    delta_ndcg = candidate["ndcg5"].astype(np.float64) - baseline["ndcg5"].astype(np.float64)
    return {"hr5_delta": float(delta_hr.mean()), "ndcg5_delta": float(delta_ndcg.mean()),
            "gained": int(np.sum(delta_hr > 0)), "lost": int(np.sum(delta_hr < 0)),
            "users": int(len(delta_hr))}


def compare(args):
    dev = Path(args.dev_run).resolve()
    dev_manifest, protocol = _validate_dev_evidence(dev)
    if dev_manifest.get("test_accessed") is not False:
        raise ValueError("dev manifest indicates test access")
    output = Path(args.output).resolve()
    results = {"protocol": PROTOCOL_VERSION, "dev_run": str(dev),
               "protocol_manifest_hash": protocol["manifest_hash"], "code_hashes": _code_hashes(),
               "dev_manifest_hash": sha256_file(dev / "dev_manifest.json"),
               "seeds": dev_manifest["seeds"], "variants": dev_manifest["variants"], "challenges": {}}
    for variant in ("normalized_gated", "zero_cross"):
        per_seed = {}
        screen_deltas = []; confirm_deltas = []; screen_ndcgs = []; confirm_ndcgs = []
        confirm_hr_rows = []; confirm_ndcg_rows = []
        screen_hr_rows = []; screen_ndcg_rows = []
        reference_ids = None
        for seed in dev_manifest["seeds"]:
            root = dev / f"seed_{int(seed)}"
            raw = _load_arrays(root / "raw", "confirm")
            candidate = _load_arrays(root / variant, "confirm")
            ids = np.stack([candidate["uid"], candidate["position"]], axis=1)
            if reference_ids is None:
                reference_ids = ids
            elif not np.array_equal(reference_ids, ids):
                raise ValueError(f"confirm UID/position cohort differs across seed {seed}")
            confirm = _paired(candidate, raw, COHORT_SEED)
            screen_label = f"screen_epoch{int(dev_manifest['epochs'])}"
            raw_screen = _load_arrays(root / "raw", screen_label)
            candidate_screen = _load_arrays(root / variant, screen_label)
            screen_ids = np.stack([candidate_screen["uid"], candidate_screen["position"]], axis=1)
            if not np.array_equal(np.stack([raw_screen["uid"], raw_screen["position"]], axis=1), screen_ids):
                raise ValueError(f"screen UID/position cohort differs within seed {seed}")
            screen = _paired(candidate_screen, raw_screen, COHORT_SEED)
            per_seed[str(seed)] = {"screen": screen, "confirm": confirm}
            screen_deltas.append(screen["hr5_delta"]); confirm_deltas.append(confirm["hr5_delta"])
            screen_ndcgs.append(screen["ndcg5_delta"]); confirm_ndcgs.append(confirm["ndcg5_delta"])
            confirm_hr_rows.append(candidate["hit5"].astype(np.float64) - raw["hit5"].astype(np.float64))
            confirm_ndcg_rows.append(candidate["ndcg5"].astype(np.float64) - raw["ndcg5"].astype(np.float64))
            screen_hr_rows.append(candidate_screen["hit5"].astype(np.float64) - raw_screen["hit5"].astype(np.float64))
            screen_ndcg_rows.append(candidate_screen["ndcg5"].astype(np.float64) - raw_screen["ndcg5"].astype(np.float64))
        # First average each user's paired delta over the three training
        # seeds, then bootstrap users.  The three seed observations are not
        # counted as 3x independent users.
        pooled = np.mean(np.stack(confirm_hr_rows, axis=0), axis=0)
        pooled_ndcg = np.mean(np.stack(confirm_ndcg_rows, axis=0), axis=0)
        pooled_screen = np.mean(np.stack(screen_hr_rows, axis=0), axis=0)
        pooled_screen_ndcg = np.mean(np.stack(screen_ndcg_rows, axis=0), axis=0)
        pooled_ci = _bootstrap_ci(pooled, COHORT_SEED)
        pooled_ndcg_ci = _bootstrap_ci(pooled_ndcg, COHORT_SEED + 1)
        result = {"per_seed": per_seed, "screen_mean_hr5_delta": float(np.mean(screen_deltas)),
                  "screen_mean_ndcg5_delta": float(np.mean(screen_ndcgs)),
                  "screen_user_mean_hr5_delta": float(np.mean(pooled_screen)),
                  "screen_user_mean_ndcg5_delta": float(np.mean(pooled_screen_ndcg)),
                  "confirm_seed_hr5_deltas": confirm_deltas,
                  "confirm_seed_hr5_std": float(np.std(confirm_deltas, ddof=1)) if len(confirm_deltas) > 1 else 0.0,
                  "confirm_seed_ndcg5_deltas": confirm_ndcgs,
                  "confirm_seed_ndcg5_std": float(np.std(confirm_ndcgs, ddof=1)) if len(confirm_ndcgs) > 1 else 0.0,
                  "screen_seed_hr5_std": float(np.std(screen_deltas, ddof=1)) if len(screen_deltas) > 1 else 0.0,
                  "confirm_mean_hr5_delta": float(np.mean(confirm_deltas)),
                  "confirm_mean_ndcg5_delta": float(np.mean(confirm_ndcgs)),
                  "confirm_hr5_ci97_5": pooled_ci,
                  "confirm_ndcg5_ci97_5": pooled_ndcg_ci,
                  "confirm_pooled_users": int(len(pooled)),
                  "accepted": bool(np.mean(screen_deltas) > 0 and np.mean(screen_ndcgs) >= 0 and
                                   all(x > 0 for x in confirm_deltas) and np.mean(confirm_deltas) >= 0.0005 and
                                   pooled_ci[0] > 0 and np.mean(confirm_ndcgs) >= 0)}
        results["challenges"][variant] = result
    accepted = [name for name, value in results["challenges"].items() if value["accepted"]]
    if accepted and not dev_manifest.get("smoke"):
        accepted.sort(key=lambda name: (-results["challenges"][name]["confirm_mean_hr5_delta"],
                                        -results["challenges"][name]["confirm_mean_ndcg5_delta"],
                                        0 if name == "zero_cross" else 1))
        results.update({"status": "accepted", "winner": accepted[0], "accepted_candidates": accepted})
    else:
        results.update({"status": "smoke_only" if dev_manifest.get("smoke") else "no_winner",
                        "winner": None, "accepted_candidates": []})
    if accepted and not dev_manifest.get("smoke") and len(accepted) == 2:
        a, b = accepted
        av, bv = results["challenges"][a], results["challenges"][b]
        if abs(av["confirm_mean_hr5_delta"] - bv["confirm_mean_hr5_delta"]) <= 1e-12:
            accepted.sort(key=lambda name: (-results["challenges"][name]["confirm_mean_ndcg5_delta"], 0 if name == "zero_cross" else 1))
            results["winner"] = accepted[0]
            results["accepted_candidates"] = accepted
    results["selection_hash"] = sha256_json(results)
    if not getattr(args, "return_only", False):
        _exclusive_json(output, results)
    return results


def _verified_selection(path, protocol):
    selection = _checked_json(path, "selection_hash")
    if selection.get("protocol_manifest_hash") != protocol["manifest_hash"]:
        raise ValueError("selection belongs to a different protocol")
    recomputed = compare(SimpleNamespace(dev_run=selection["dev_run"], output=str(path), return_only=True))
    if selection != recomputed:
        raise ValueError("selection differs from validated registered decision")
    return selection


def lock(args):
    selection_path = Path(args.selection).resolve()
    protocol_path = Path(args.protocol_manifest).resolve()
    protocol = load_protocol(protocol_path)
    selection = _verified_selection(selection_path, protocol)
    allowed = selection.get("status") == "accepted" and not protocol.get("smoke")
    winner = selection.get("winner") if allowed else None
    if allowed and winner not in ("normalized_gated", "zero_cross"):
        raise ValueError("invalid accepted winner")
    _verify_base(protocol)
    plan = {"protocol": PROTOCOL_VERSION, "selection": str(selection_path),
            "selection_hash": sha256_file(selection_path), "protocol_manifest": str(protocol_path),
            "protocol_manifest_hash": protocol["manifest_hash"], "base_run": protocol["base_run"],
            "base_data_id": protocol["base_data_id"], "base_assets": protocol["base_assets"],
            "code_hashes": _code_hashes(), "epochs": 3,
            "config": {"dim": 256, "token_dim": 256, "hist_len": 50, "batch_size": 256, "candidates": 75},
            "final_seed": 42, "final_init_seed": INIT_SEEDS[42],
            "test_cohort_hash": protocol["cohort_hashes"]["test"],
            "models": ["raw", winner] if allowed else ["raw"],
            "winner": winner, "final_evaluation_allowed": allowed,
            "test_future_labels_read": False}
    plan["plan_hash"] = sha256_json(plan)
    _exclusive_json(Path(args.output).resolve(), plan)
    return plan


def _load_plan(path: Path):
    plan = _checked_json(path, "plan_hash")
    if plan.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("invalid final plan protocol")
    selection_path = Path(plan["selection"])
    if sha256_file(selection_path) != plan.get("selection_hash"):
        raise ValueError("selection hash no longer matches locked final plan")
    protocol = load_protocol(Path(plan["protocol_manifest"]))
    if protocol["manifest_hash"] != plan.get("protocol_manifest_hash"):
        raise ValueError("protocol manifest no longer matches locked final plan")
    selection = _verified_selection(selection_path, protocol)
    allowed = selection["status"] == "accepted" and not protocol.get("smoke")
    if plan.get("final_evaluation_allowed") != allowed or plan.get("winner") != (selection["winner"] if allowed else None):
        raise ValueError("plan authorization differs from verified selection")
    expected_models = ["raw", selection["winner"]] if allowed else ["raw"]
    if plan.get("models") != expected_models or plan.get("test_cohort_hash") != protocol["cohort_hashes"]["test"]:
        raise ValueError("plan models/cohort mismatch")
    if plan.get("code_hashes") != _code_hashes() or plan.get("base_assets") != protocol["base_assets"]:
        raise ValueError("locked code/base changed")
    if plan.get("config") != {"dim": 256, "token_dim": 256, "hist_len": 50, "batch_size": 256, "candidates": 75}:
        raise ValueError("locked model configuration mismatch")
    if plan.get("epochs") != 3 or plan.get("final_seed") != 42 or plan.get("final_init_seed") != INIT_SEEDS[42]:
        raise ValueError("locked training configuration mismatch")
    _verify_base(protocol)
    return plan


def final_train(args):
    plan_path = Path(args.final_plan).resolve()
    plan = _load_plan(plan_path)
    if not plan.get("final_evaluation_allowed"):
        raise ValueError("selection has no accepted winner; final training/evaluation is not authorized")
    base = Path(args.base_run).resolve()
    if str(base) != str(Path(plan["base_run"]).resolve()):
        raise ValueError("base-run does not match final plan")
    protocol = load_protocol(Path(plan["protocol_manifest"]))
    if not protocol.get("smoke") and (int(args.epochs) != 3 or int(args.dim) != 256 or
                                       int(args.token_dim) != 256 or int(args.hist_len) != 50 or int(args.batch_size) != 256 or int(args.candidates) != 75):
        raise ValueError("formal final-train requires the locked 256-dim, hist_len 50, batch 256, 3-epoch configuration")
    data = validate_base(base, protocol)
    factors, _ = _load_factors(base, data, int(args.dim), bool(protocol.get("smoke")))
    test = _records(Path(plan["protocol_manifest"]).parent, protocol, "test")
    all_positions = np.asarray(data.train_positions, dtype=np.int64)
    test_uids = {int(row[0]) for row in test}
    covered = set(np.asarray(data.uid)[all_positions].astype(int).tolist())
    if not test_uids.issubset(covered):
        raise ValueError("final training does not cover all test user prefixes")
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise FileExistsError("refusing to overwrite final training evidence")
    catalog_records = [(int(data.uid[int(pos)]), int(pos), _data_user_raw(data, int(data.uid[int(pos)])))
                       for pos in all_positions.tolist()]
    pools, cache = _load_or_make_assets(data, Path(plan["protocol_manifest"]).parent, protocol,
                                        catalog_records, "final_train", max(75, int(args.candidates)))
    # A deterministic, unlabelled screen proxy is used for the training loop;
    # no future test labels are read.  The final checkpoint itself is evaluated
    # only by final-test.
    screen_records = _records(Path(plan["protocol_manifest"]).parent, protocol, "screen")
    screen_pools, _ = _load_or_make_assets(data, Path(plan["protocol_manifest"]).parent, protocol,
                                           screen_records, "screen", int(args.candidates))
    template, state = _template(data, args, INIT_SEEDS[42], factors)
    del template
    initial_audit = {"state_sha256": {}, "copied_tensor_keys": sorted(state), "gate_values": {},
                     "gate_difference_whitelist": ["cross_gate_logit"]}
    for variant in plan["models"]:
        audit_model = _new_variant(data, args, state, variant)
        for key, value in audit_model.state_dict().items():
            if key != "cross_gate_logit" and not torch.equal(value.cpu(), state[key]):
                raise ValueError("final initialization mismatch")
        initial_audit["state_sha256"][variant] = _state_hashes(audit_model)
        initial_audit["gate_values"][variant] = float(audit_model.cross_gate_logit.detach())
        del audit_model
    write_json(run / "initial_state_audit.json", initial_audit)
    reference_traces = None
    for variant in plan["models"]:
        trained = _train_model(data, args, run, 42, variant, state, factors, all_positions, pools,
                     screen_records, screen_pools, bool(protocol.get("smoke")))
        if reference_traces is None:
            reference_traces = trained["trace_files"]
        elif reference_traces != trained["trace_files"]:
            raise ValueError("final paired training streams differ")
        for trace in trained["trace_files"].values():
            if trace["rows"] != len(all_positions) or trace["covered_users"] != len(covered):
                raise ValueError("actual final training coverage mismatch")
        # Keep the final model under a clear, unambiguous path.  Do not copy it
        # over the fast-development checkpoint tree.
        source = run / "seed_42" / variant / "last.pth"
        target_dir = run / variant; target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "last.pth"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        payload["manifest"].update({"final": True, "final_plan_hash": plan["plan_hash"],
                                     "test_cohort_hash": plan["test_cohort_hash"],
                                     "test_future_labels_read": False})
        atomic_torch_save(payload, target)
        write_json(target_dir / "manifest.json", payload["manifest"])
    final_manifest = {"protocol": PROTOCOL_VERSION, "final_plan_hash": plan["plan_hash"],
                      "test_history_coverage_users": len(test_uids), "all_prefix_rows": int(len(all_positions)),
                      "models": plan["models"], "test_future_labels_read": False,
                      "checkpoint_hashes": {v: sha256_file(run / v / "last.pth") for v in plan["models"]}}
    final_manifest["train_positions_sha256"] = sha256_array(all_positions)
    final_manifest["protocol_manifest_hash"] = protocol["manifest_hash"]
    final_manifest["evidence_hashes"] = {str(p.relative_to(run)): sha256_file(p)
                                         for p in sorted(run.rglob("*")) if p.is_file()}
    final_manifest["manifest_hash"] = sha256_json(final_manifest)
    write_json(run / "final_manifest.json", final_manifest)
    _exclusive_json(run / "FINAL_TRAIN_COMPLETED.json", {"manifest_hash": final_manifest["manifest_hash"]})
    return final_manifest


def final_test(args):
    plan_path = Path(args.final_plan).resolve(); plan = _load_plan(plan_path)
    if not plan.get("final_evaluation_allowed"):
        raise ValueError("final test is forbidden without a unique accepted winner")
    if str(Path(args.base_run).resolve()) != str(Path(plan["base_run"]).resolve()):
        raise ValueError("base-run does not match final plan")
    protocol = load_protocol(Path(plan["protocol_manifest"]))
    if not protocol.get("smoke") and (int(args.dim) != 256 or int(args.token_dim) != 256 or int(args.hist_len) != 50 or int(args.candidates) != 75):
        raise ValueError("formal final-test requires the locked 256-dim, hist_len 50 configuration")
    run = Path(args.run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    final_manifest = _checked_json(run / "final_manifest.json", "manifest_hash")
    complete = _checked_json(run / "FINAL_TRAIN_COMPLETED.json")
    if complete.get("manifest_hash") != final_manifest["manifest_hash"] or final_manifest.get("final_plan_hash") != plan["plan_hash"] or final_manifest.get("models") != plan["models"] or final_manifest.get("test_future_labels_read") is not False:
        raise ValueError("final training is incomplete or belongs to another plan")
    if final_manifest.get("test_history_coverage_users") != protocol["counts"]["test"]:
        raise ValueError("final test users lack trained histories")
    if not final_manifest.get("evidence_hashes"):
        raise ValueError("final training evidence is unsealed")
    for relative, expected in final_manifest["evidence_hashes"].items():
        evidence = (run / relative).resolve()
        if not evidence.is_relative_to(run) or sha256_file(evidence) != expected:
            raise ValueError("final training evidence changed")
    marker = Path(plan["protocol_manifest"]).parent / "TEST_STARTED.json"
    if marker.exists():
        raise RuntimeError("TEST_STARTED.json exists; refusing a second test evaluation")
    for variant in plan["models"]:
        if not (run / variant / "last.pth").exists():
            raise FileNotFoundError(run / variant / "last.pth")
    checkpoint_hashes = {variant: sha256_file(run / variant / "last.pth") for variant in plan["models"]}
    if checkpoint_hashes != final_manifest["checkpoint_hashes"]:
        raise ValueError("final checkpoint hash mismatch")
    for variant in plan["models"]:
        payload = torch.load(run / variant / "last.pth", map_location="cpu", weights_only=False)
        saved = payload["manifest"]
        if saved.get("final") is not True or saved.get("final_plan_hash") != plan["plan_hash"] or saved.get("test_cohort_hash") != plan["test_cohort_hash"] or saved.get("epochs") != 3 or saved.get("train_positions_sha256") != final_manifest["train_positions_sha256"]:
            raise ValueError("checkpoint is not the locked full-training result")
    marker_value = {"protocol": PROTOCOL_VERSION, "plan_hash": plan["plan_hash"],
                    "selection_hash": plan["selection_hash"], "test_cohort_hash": plan["test_cohort_hash"],
                    "checkpoint_hashes": checkpoint_hashes, "status": "started", "started_at": time.time()}
    base = validate_base(Path(plan["base_run"]), protocol)
    if sha256_array(np.asarray(base.train_positions, dtype=np.int64)) != final_manifest["train_positions_sha256"]:
        raise ValueError("full training positions changed")
    _exclusive_json(marker, marker_value)
    test = _records(Path(plan["protocol_manifest"]).parent, protocol, "test")
    factors, _ = _load_factors(Path(plan["base_run"]), base, int(args.dim), bool(protocol.get("smoke")))
    results = {"protocol": PROTOCOL_VERSION, "plan_hash": plan["plan_hash"], "test_cohort_hash": plan["test_cohort_hash"], "models": {}}
    pools, _ = _load_or_make_assets(base, Path(plan["protocol_manifest"]).parent, protocol, test,
                                    "test_final", int(args.candidates))
    for variant in plan["models"]:
        payload = torch.load(run / variant / "last.pth", map_location="cpu", weights_only=False)
        model = SemanticTokenDIN(**payload["manifest"]["model_config"])
        model.load_state_dict(payload["model"])
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")); model.eval()
        metrics, arrays = evaluate(base, model, test, pools)
        np.savez_compressed(run / f"{variant}_test_users.npz",
                            uid=np.asarray([int(x[0]) for x in test], dtype=np.int64),
                            position=np.asarray([int(x[1]) for x in test], dtype=np.int64), **arrays)
        write_json(run / f"{variant}_test_metrics.json", metrics)
        results["models"][variant] = metrics
    raw = np.load(run / "raw_test_users.npz"); winner = np.load(run / f"{plan['winner']}_test_users.npz")
    _paired(winner, raw, COHORT_SEED)
    delta = winner["hit5"].astype(np.float64) - raw["hit5"].astype(np.float64)
    ndcg_delta = winner["ndcg5"].astype(np.float64) - raw["ndcg5"].astype(np.float64)
    results["paired"] = {"hr5_delta": float(delta.mean()), "hr5_ci95": _bootstrap_ci(delta, COHORT_SEED, alpha=0.025),
                          "ndcg5_delta": float(ndcg_delta.mean()), "gained": int(np.sum(delta > 0)),
                          "lost": int(np.sum(delta < 0)), "users": int(len(delta))}
    results["status"] = "accepted" if results["paired"]["hr5_ci95"][0] > 0 and results["paired"]["hr5_delta"] >= 0.0005 and results["paired"]["ndcg5_delta"] >= 0 else "raw_retained"
    results["results_hash"] = sha256_json(results)
    write_json(run / "final_test_results.json", results)
    marker_value.update({"status": "complete", "results_hash": results["results_hash"]})
    write_json(marker, marker_value)
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base-run", required=True); prep.add_argument("--run-dir", required=True)
    prep.add_argument("--historical-users", required=True); prep.add_argument("--train-users", type=int, default=100000)
    prep.add_argument("--screen-users", type=int, default=20000); prep.add_argument("--confirm-users", type=int, default=80000)
    prep.add_argument("--test-users", type=int, default=100000); prep.add_argument("--cohort-seed", type=int, default=COHORT_SEED)
    prep.add_argument("--smoke", action="store_true")
    devp = sub.add_parser("dev")
    devp.add_argument("--base-run", required=True); devp.add_argument("--protocol-manifest", required=True); devp.add_argument("--run-dir", required=True)
    devp.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); devp.add_argument("--variants", nargs="+", default=list(VARIANTS))
    devp.add_argument("--epochs", type=int, default=3); devp.add_argument("--dim", type=int, default=256); devp.add_argument("--token-dim", type=int, default=256)
    devp.add_argument("--hist-len", type=int, default=50); devp.add_argument("--batch-size", type=int, default=256); devp.add_argument("--candidates", type=int, default=75); devp.add_argument("--smoke", action="store_true")
    cmp = sub.add_parser("compare"); cmp.add_argument("--dev-run", required=True); cmp.add_argument("--output", required=True)
    lockp = sub.add_parser("lock"); lockp.add_argument("--selection", required=True); lockp.add_argument("--protocol-manifest", required=True); lockp.add_argument("--output", required=True)
    fp = sub.add_parser("final-train"); fp.add_argument("--base-run", required=True); fp.add_argument("--final-plan", required=True); fp.add_argument("--run-dir", required=True); fp.add_argument("--dim", type=int, default=256); fp.add_argument("--token-dim", type=int, default=256); fp.add_argument("--hist-len", type=int, default=50); fp.add_argument("--batch-size", type=int, default=256); fp.add_argument("--epochs", type=int, default=3); fp.add_argument("--candidates", type=int, default=75)
    ftp = sub.add_parser("final-test"); ftp.add_argument("--base-run", required=True); ftp.add_argument("--final-plan", required=True); ftp.add_argument("--run-dir", required=True); ftp.add_argument("--dim", type=int, default=256); ftp.add_argument("--token-dim", type=int, default=256); ftp.add_argument("--hist-len", type=int, default=50); ftp.add_argument("--candidates", type=int, default=75)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare": return prepare(args)
    if args.command == "dev": return dev(args)
    if args.command == "compare": return compare(args)
    if args.command == "lock": return lock(args)
    if args.command == "final-train": return final_train(args)
    if args.command == "final-test": return final_test(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
