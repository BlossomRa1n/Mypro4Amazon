"""Read-only reconstruction audit for the completed controlled token suite.

No training, new model evaluation, candidate generation or deletion is done.
Only the requested JSON report is written. Full negative reconstruction is
the default; --skip-full-negatives produces development-only evidence.
"""
import argparse
import gc
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
DATA_ID = "754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158"
VARIANTS = ("din_random", "din_svd", "semantic_concat_random", "semantic_concat_svd")
SOURCE_FILES = ("run_controlled_token_experiments.py", "run_token_experiments.py",
                "baseline_data.py", "future_window_data.py", "baseline_runtime.py",
                "run_baseline.py", "model_ext.py", "token_models.py", "model.py", "config.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_option(command, name, default=None):
    require(isinstance(command, list) and all(isinstance(value, str) for value in command),
            "monitor command must be an argument list")
    found = []
    for index, value in enumerate(command):
        if value == name:
            require(index + 1 < len(command), "Missing command value: " + name)
            found.append(command[index + 1])
        elif value.startswith(name + "="):
            found.append(value[len(name) + 1:])
    require(len(found) <= 1, "Repeated command option: " + name)
    return found[0] if found else default


def verify_sources(run_dir, manifest):
    evidence = {}
    for name in SOURCE_FILES:
        candidates = [run_dir / "source_snapshot" / name,
                      run_dir / "source_snapshot" / "code" / name]
        existing = [path for path in candidates if path.is_file()]
        require(existing, "Missing source snapshot: " + name)
        actual = file_hash(CODE / name)
        require(all(file_hash(path) == actual for path in existing), "Source/snapshot drift: " + name)
        evidence[name] = {"sha256": actual, "snapshot": str(existing[0])}
    require(evidence["run_controlled_token_experiments.py"]["sha256"] == manifest["runner_sha256"],
            "Runner does not match the hash recorded by this run")
    return evidence


def reconstruct_order(sample_count, batch_size, seed, epoch):
    """Use DataLoader itself, including its generator base-seed draw."""
    count = sample_count // batch_size * batch_size
    require(count > 0, "No full training batches")
    order = np.empty(count, dtype=np.int64)
    offset = 0
    loader = DataLoader(range(sample_count), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed + epoch),
                        num_workers=0, drop_last=True)
    for batch in loader:
        values = batch.numpy()
        order[offset:offset + len(values)] = values
        offset += len(values)
    require(offset == count, "Rebuilt loader length differs")
    require(np.all((order >= 0) & (order < sample_count)) and len(np.unique(order)) == count,
            "Rebuilt order contains invalid or duplicate sample indices")
    digest = hashlib.sha256(order.astype("<i8", copy=False).tobytes()).hexdigest()
    return order, digest


def negatives_for(data, idx, epoch, count=4):
    position = int(data.train_positions[int(idx)])
    uid, target = int(data.uid[position]), int(data.iid[position])
    rng = np.random.default_rng(np.random.SeedSequence([data.seed, epoch, int(idx)]))
    negatives = np.asarray(data.negatives(uid, target, count, rng), dtype=np.int64)
    require(negatives.shape == (count,) and len(set(negatives.tolist())) == count,
            f"Invalid negative count/duplicates at epoch={epoch}, sample={idx}")
    excluded = data.train_sets[uid]
    require(all(2 <= int(item) < len(data.items) and int(item) in data.active_set and
                int(item) != target and int(item) not in excluded for item in negatives),
            f"Illegal negative at epoch={epoch}, sample={idx}")
    return uid, target, negatives


def audit_negatives(data, order, epoch, dataset_class, full=True):
    digest = hashlib.sha256()
    if full:
        for offset, idx in enumerate(order):
            _, _, negatives = negatives_for(data, int(idx), epoch)
            digest.update(np.asarray([int(idx), *negatives.tolist()], dtype="<i8").tobytes())
            if offset and offset % 1000000 == 0:
                print(f"epoch {epoch + 1}: reconstructed {offset:,}/{len(order):,} negative samples", flush=True)
    probes = np.unique(np.linspace(0, len(data.train_positions) - 1,
                                  min(128, len(data.train_positions)), dtype=np.int64))
    datasets = {variant: dataset_class(data, 4) for variant in VARIANTS}
    for dataset in datasets.values():
        dataset.epoch = epoch
    probe_digest = hashlib.sha256()
    for idx in probes:
        uid, target, expected = negatives_for(data, int(idx), epoch)
        for variant, dataset in datasets.items():
            row = dataset[int(idx)]
            require(int(row["user_id"]) == uid and int(row["pos_item_id"]) == target,
                    variant + ": PrefixDataset row identity differs")
            require(np.array_equal(np.asarray(row["neg_item_id"]), expected),
                    variant + ": PrefixDataset negatives differ")
        probe_digest.update(np.asarray([int(idx), *expected.tolist()], dtype="<i8").tobytes())
    return {"full_reconstruction": bool(full), "effective_samples_checked": len(order) if full else 0,
            "ordered_sample_and_negative_sha256": digest.hexdigest() if full else None,
            "negative_ids_per_sample": 4, "legality_checks": "unique, active, in-range, not target, not train positive",
            "four_group_prefix_dataset_probe_count": len(probes),
            "probe_sample_and_negative_sha256": probe_digest.hexdigest()}


def audit_pool(data, records, pools, budget=75):
    require(len(records) == len(pools), "Candidate cache/user count differs")
    digest = hashlib.sha256()
    pool_hit = np.empty(len(records), dtype=np.int8)
    for index, ((uid, position), pool) in enumerate(zip(records, pools)):
        ids = list(pool)
        require(len(ids) == budget and len(set(ids)) == budget,
                f"Candidate pool {index}: expected exactly {budget} unique IDs")
        require(all(isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)) and
                    2 <= int(item) < len(data.items) for item in ids),
                f"Candidate pool {index}: illegal item ID")
        end = data.history_end(uid, position)
        seen = set(data.iid[data.starts[uid]:end].tolist())
        require(not seen.intersection(ids), f"Candidate pool {index}: observed history item is present")
        target = set(data.iid[position:data.ends[uid]].tolist())
        pool_hit[index] = bool(target.intersection(ids))
        digest.update(np.asarray([uid, position, len(ids), *ids], dtype="<i8").tobytes())
    return pool_hit, {"users": len(records), "candidates_per_user": budget,
                      "ordered_identity_and_candidate_ids_sha256": digest.hexdigest(),
                      "pool_hits": int(pool_hit.sum()), "pool_hr": float(pool_hit.mean())}


def audit_run(run_dir, skip_full_negatives=False):
    run_dir = Path(run_dir).resolve()
    require(not (run_dir / "SUPERSEDED.json").exists(), "Run is superseded")
    monitor = read_json(run_dir / "monitor_status.json")
    require(monitor["status"] == "complete" and type(monitor["exit_code"]) is int and monitor["exit_code"] == 0,
            "Training must have completed with exit_code=0 before protocol audit")
    completion = read_json(run_dir / "COMPLETED.json")
    require(completion["status"] == "complete" and completion["suite"] == "controlled_din_token_v2" and
            completion["variants"] == list(VARIANTS),
            "Four-group completion marker is missing")
    manifest = read_json(run_dir / "suite_manifest.json")
    fixed = {"suite": "controlled_din_token_v2", "protocol": "future-window", "data_id": DATA_ID,
             "seed": 42, "init_seed": 424242, "epochs": 3, "negative_count": 4,
             "candidate_budget": 75, "fusion_mode": "rrf", "fusion_weights": [2., 1., .7, .05],
             "itemcf_half_life_days": 180., "screen_users": 100000, "test_users": 100000}
    for key, expected in fixed.items():
        require(manifest.get(key) == expected, "Fixed protocol differs: " + key)
    sources = verify_sources(run_dir, manifest)
    provenance = {}
    for name in ("supplemental_capture.json", "sha256_supplemental.json"):
        path = run_dir / "source_snapshot" / name
        require(path.is_file(), "Missing supplemental snapshot provenance: " + name)
        provenance[name] = {"path": str(path), "sha256": file_hash(path), "record": read_json(path)}
    sys.path.insert(0, str(CODE))
    from baseline_data import PrefixDataset, fingerprint
    from future_window_data import FutureWindowData

    source = manifest["source"]
    base = Path(source["base_run"]).resolve()
    command = monitor["command"]
    require(Path(command_option(command, "--base-run", str(base))).resolve() == base,
            "Monitor/base_run differs from suite source")
    require(int(command_option(command, "--max-train-steps", "0")) == 0,
            "Formal audit rejects truncated training")
    cf_dir = Path(command_option(command, "--cf-run", str(base)) or str(base)).resolve()
    assets = {"base_manifest": base / "run_manifest.json", "base_v2": base / "v2_best.pth",
              "svd": base / "svd.npy", "itemcf": cf_dir / "itemcf.pkl"}
    asset_hashes = {}
    for key, path in assets.items():
        actual = file_hash(path)
        require(actual == source[key], "Source asset hash drift: " + key)
        asset_hashes[key] = {"path": str(path), "sha256": actual}
    base_spec = read_json(assets["base_manifest"])
    base_args = base_spec["args"]
    require(base_args["seed"] == 42 and base_args["protocol"] == "future-window",
            "Base data seed/protocol differs")
    cf_manifest = cf_dir / "run_manifest.json"
    cf_spec = read_json(cf_manifest)
    require(cf_spec["args"]["cf_neighbors"] == 300 and
            cf_spec["args"]["itemcf_half_life_days"] == 180.,
            "ItemCF manifest must specify 300 neighbors and 180-day half life")
    asset_hashes["cf_manifest"] = {"path": str(cf_manifest), "sha256": file_hash(cf_manifest)}
    data_path = base / "data.pkl"
    asset_hashes["data_pickle"] = {"path": str(data_path), "sha256": file_hash(data_path)}
    with data_path.open("rb") as stream:
        data = pickle.load(stream)["data"]
    require(isinstance(data, FutureWindowData) and
            data.protocol == "positive-first-interaction-user-prefix80-future20-v1",
            "Loaded data is not the FutureWindowData prefix80/future20 protocol")
    require(data.manifest["data_id"] == source["data_id"] == DATA_ID and data.seed == 42,
            "Loaded data identity/seed differs")
    arrays = {"uid": data.uid, "iid": data.iid, "ts": data.ts, "rating": data.rating,
              "train_mask": data.train_mask, "brand": data.item_brand, "category": data.item_category}
    for name, values in arrays.items():
        require(hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest() ==
                data.manifest["hashes"][name], "Data array hash differs: " + name)
    factors = np.load(assets["svd"], mmap_mode="r")
    require(factors.shape == (len(data.items), int(base_args["dim"])), "SVD shape differs")
    del factors
    with assets["itemcf"].open("rb") as stream:
        neighbors, similarities = pickle.load(stream)
    require(neighbors.shape == similarities.shape == (len(data.items), 300), "ItemCF asset must have width 300")
    cf_shape = list(neighbors.shape)
    del neighbors, similarities
    gc.collect()

    batch_size, sample_count = int(base_args["din_batch"]), len(data.train_positions)
    require(batch_size >= 2 and sample_count >= 128, "Invalid formal sample/batch size")
    histories = {variant: read_json(run_dir / variant / "history.json") for variant in VARIANTS}
    for variant, history in histories.items():
        require(len(history) == 3 and [entry["epoch"] for entry in history] == [1, 2, 3],
                variant + ": expected three completed epochs")
    training = []
    for epoch in range(3):
        hashes, order = {}, None
        for variant in VARIANTS:
            rebuilt, digest = reconstruct_order(sample_count, batch_size, 42, epoch)
            hashes[variant] = digest
            require(histories[variant][epoch]["steps"] == len(rebuilt) // batch_size,
                    variant + ": recorded training steps do not match rebuilt loader")
            if order is None:
                order = rebuilt
            else:
                require(digest == hashes[VARIANTS[0]], "Group training-order reconstruction differs")
                del rebuilt
        training.append({"epoch": epoch + 1, "effective_sample_count": len(order),
                         "steps": len(order) // batch_size, "dropped_samples": sample_count - len(order),
                         "order_sha256_by_group": hashes,
                         "negatives": audit_negatives(data, order, epoch, PrefixDataset, not skip_full_negatives)})
        del order
        print(f"epoch {epoch + 1}: order and negative reconstruction verified", flush=True)

    records_by_split = {label: data.evaluation(split, manifest[label + "_users"])
                        for label, split in (("screen", "val"), ("test", "test"))}
    require(not set(uid for uid, _ in records_by_split["screen"]).intersection(
                uid for uid, _ in records_by_split["test"]), "Development and test users overlap")
    candidates = {}
    for label, records in records_by_split.items():
        require(len(records) == 100000 and len({uid for uid, _ in records}) == 100000,
                label + ": wrong evaluation count/duplicate users")
        records_hash = fingerprint(records)
        require(records_hash == manifest["evaluation_records"][label], label + ": recorded evaluation identity differs")
        key = fingerprint({"label": label, "records": records, "data_id": DATA_ID,
                           "v2": source["base_v2"], "itemcf": source["itemcf"], "budget": 75,
                           "fusion_mode": "rrf", "weights": [2., 1., .7, .05], "half_life": 180.})
        cache = run_dir / "cache" / f"pools_{label}_{key}.pkl"
        require(cache.name in manifest["candidate_cache_files"] and cache.is_file(),
                label + ": expected fingerprinted candidate cache is missing")
        cache_hash = file_hash(cache)
        with cache.open("rb") as stream:
            pools = pickle.load(stream)
        pool_hit, summary = audit_pool(data, records, pools)
        identities = np.asarray(records, dtype=np.int64)
        for variant in VARIANTS:
            path = run_dir / f"{variant}_{label}_final_users.npz"
            with np.load(path, allow_pickle=False) as final:
                require(np.array_equal(final["uid"], identities[:, 0]) and
                        np.array_equal(final["position"], identities[:, 1]),
                        variant + "/" + label + ": final identities differ from data.evaluation")
                require(np.array_equal(final["pool_hit"], pool_hit),
                        variant + "/" + label + ": final pool_hit differs from actual candidate IDs")
        candidates[label] = {**summary, "cache_path": str(cache), "cache_sha256": cache_hash,
                             "cache_key": key, "records_fingerprint": records_hash,
                             "all_four_final_identity_and_pool_hits_match": True}
        del pools
        gc.collect()
    return {"status": "development_only" if skip_full_negatives else "passed",
            "safe_shutdown_eligible": not skip_full_negatives, "full_negative_reconstruction": not skip_full_negatives,
            "run_dir": str(run_dir), "data_id": DATA_ID, "source_code": sources, "assets": asset_hashes,
            "snapshot_provenance": provenance, "auditor_sha256": file_hash(__file__),
            "reconstruction_runtime": {"torch": torch.__version__, "numpy": np.__version__},
            "itemcf_shape": cf_shape, "itemcf_neighbors": 300, "itemcf_half_life_days": 180.,
            "training": {"sample_count": sample_count, "batch_size": batch_size,
                         "training_workers": int(base_args["workers"]), "reconstruction_workers": 0,
                         "epochs": training}, "candidates": candidates, "dev_test_users_disjoint": True,
            "checks": ["frozen source and recorded runner hash", "source asset hashes and protocol",
                       "four deterministic DataLoader order rebuilds per epoch",
                       ("full effective negative reconstruction plus four actual PrefixDataset probes"
                        if not skip_full_negatives else "development-only PrefixDataset probes; full sweep omitted"),
                       "actual candidate IDs, history exclusions, targets and final row identities"],
            "limitations": [
                "Order and negatives are deterministic reconstructions from frozen code/configuration, not recorded per-batch training traces.",
                "DataLoader workers only fetch map-style samples; worker count does not change the main-process sampler order. Reconstruction uses workers=0.",
                "Negative reconstruction runs once per epoch over effective shuffled indices; four-group equality is additionally checked with 128 distributed actual PrefixDataset samples.",
                "Snapshot equality anchors the runner to its recorded hash; dependency snapshots require their own retained provenance. This audit does not recreate launch-time evidence.",
                "Candidate caches are checked directly and matched to final arrays; V2 recall and RRF generation are not rerun.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to RUN_DIR/protocol_audit.json")
    parser.add_argument("--skip-full-negatives", action="store_true",
                        help="Development-only shortcut; output explicitly fails the safe-shutdown eligibility condition")
    args = parser.parse_args()
    try:
        result = audit_run(args.run_dir, skip_full_negatives=args.skip_full_negatives)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"Protocol audit failed: {error}\n")
    output = args.output or args.run_dir / "protocol_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Protocol audit {result['status']}: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
