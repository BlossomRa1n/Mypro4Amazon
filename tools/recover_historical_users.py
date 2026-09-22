#!/usr/bin/env python3
"""Build an explicitly incomplete, conservative raw-user exclusion registry.

No torch/sklearn imports. JSON predictions are tokenized incrementally and only
string user_id values are retained. NPZ loading imports numpy only on demand.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


class RecoveryError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_tokens(path):
    """Bounded token reader; never materializes an entire prediction object."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, position, eof = "", 0, False
        while True:
            if position >= len(buffer):
                buffer, position = stream.read(65536), 0
                if not buffer:
                    return
            char = buffer[position]
            if char.isspace():
                position += 1
                continue
            if char in "{}[]:,":
                position += 1
                yield char, None
                continue
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    # Number tokens may end at the current chunk boundary.
                    if not eof and (end == len(buffer) or buffer[end] not in " \t\n\r{}[]:,\""):
                        raise ValueError("possibly incomplete token")
                    break
                except ValueError:
                    if eof:
                        raise RecoveryError("Invalid prediction JSON")
                    buffer = buffer[position:]
                    position = 0
                    more = stream.read(65536)
                    eof = not more
                    buffer += more
                    if len(buffer) > 8 * 1024 * 1024:
                        raise RecoveryError("Prediction JSON token exceeds limit")
            position = end
            if isinstance(value, (dict, list)):
                raise RecoveryError("Unexpected compound JSON token")
            yield "scalar", value


def prediction_users(path):
    tokens = iter(json_tokens(path))
    users = set()

    def parse(token, depth=0, user_value=False, capture=True):
        if depth > 100:
            raise RecoveryError("Prediction JSON nesting exceeds limit")
        kind, value = token
        if user_value:
            if kind != "scalar" or not isinstance(value, str) or not value:
                raise RecoveryError("Prediction user_id must be a nonempty raw string")
            users.add(value)
        if kind == "scalar":
            return
        if kind == "{":
            token = next(tokens)
            if token[0] == "}":
                return
            while True:
                if token[0] != "scalar" or not isinstance(token[1], str):
                    raise RecoveryError("Invalid JSON object key")
                key = token[1]
                if next(tokens)[0] != ":":
                    raise RecoveryError("Invalid JSON object separator")
                child_capture = capture and key not in {"labels", "targets", "metrics", "y_true", "ground_truth"}
                parse(next(tokens), depth + 1, child_capture and key == "user_id", child_capture)
                token = next(tokens)
                if token[0] == "}":
                    return
                if token[0] != ",":
                    raise RecoveryError("Invalid JSON object delimiter")
                token = next(tokens)
        elif kind == "[":
            token = next(tokens)
            if token[0] == "]":
                return
            while True:
                parse(token, depth + 1, capture=capture)
                token = next(tokens)
                if token[0] == "]":
                    return
                if token[0] != ",":
                    raise RecoveryError("Invalid JSON array delimiter")
                token = next(tokens)
        else:
            raise RecoveryError("Invalid JSON value")

    try:
        parse(next(tokens))
        if next(tokens, None) is not None:
            raise RecoveryError("Trailing prediction JSON data")
    except StopIteration as exc:
        raise RecoveryError("Truncated prediction JSON") from exc
    return users


def csv_users(path):
    users = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.reader(stream)
        header = next(rows, [])
        if "user_id" not in header:
            return None
        if header.count("user_id") != 1:
            raise RecoveryError("Duplicate user_id CSV header")
        index = header.index("user_id")
        for row in rows:
            if len(row) != len(header) or not row[index]:
                raise RecoveryError("Invalid CSV user row")
            users.add(row[index])
    return users


def load_identity(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    users = value.get("users")
    source_hash = value.get("source_sha256")
    if (not isinstance(value.get("data_id"), str) or not value["data_id"]
            or not isinstance(users, list) or not users
            or not all(isinstance(x, str) and x for x in users)
            or len(set(users)) != len(users)
            or not isinstance(source_hash, str) or len(source_hash) != 64
            or any(char not in "0123456789abcdef" for char in source_hash)):
        raise RecoveryError("Identity requires data_id, source SHA256 and unique ordered raw users")
    value["identity_path"] = str(path)
    value["identity_sha256"] = sha256(path)
    return value


def nearest_manifest(path):
    for parent in Path(path).resolve().parents:
        manifest = parent / "suite_manifest.json"
        if manifest.is_file():
            with manifest.open() as stream:
                value = json.load(stream)
            nested = value.get("source", {}).get("data_id")
            top = value.get("data_id")
            if nested and top and nested != top:
                raise RecoveryError("Conflicting manifest data IDs")
            if not (nested or top):
                raise RecoveryError("Nearest suite manifest has no data_id")
            return manifest, nested or top
    raise RecoveryError("No ancestor suite manifest")


def npz_users(path, identities):
    import numpy as np
    manifest, data_id = nearest_manifest(path)
    if data_id not in identities:
        raise RecoveryError("NPZ data_id has no matching identity mapping")
    mapping = identities[data_id]["users"]
    with np.load(path, allow_pickle=False) as archive:
        if "uid" not in archive.files:
            raise RecoveryError("NPZ missing uid")
        uid = archive["uid"]
        if uid.ndim != 1 or uid.dtype.kind not in "iu":
            raise RecoveryError("NPZ uid must be a one-dimensional integer array")
        if uid.size and (int(uid.min()) < 0 or int(uid.max()) >= len(mapping)):
            raise RecoveryError("NPZ uid outside verified mapping")
        result = {mapping[int(index)] for index in uid}
        if result.intersection({"__PAD__", "__UNKNOWN__"}):
            raise RecoveryError("NPZ uid references a reserved encoder index")
    return result, data_id, manifest


def raw_list(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    if isinstance(value, dict):
        value = value.get("raw_user_ids", value.get("users"))
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise RecoveryError("Legacy input must contain a raw user string list")
    return set(value)


def source_csv(path):
    parts = {part.lower() for part in Path(path).parts}
    return bool(parts.intersection({"raw", "raw_data", "source_data", "amazon_data", "amazon_reviews", "5core", "rating_only"}))


def recover(inventory_path, identity_paths, conservative_paths, legacy_paths, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise RecoveryError("Output directory must be new; existing evidence will not be overwritten")
    identities, registry, users, sources = {}, [], set(), []
    for path in identity_paths:
        value = load_identity(path)
        data_id = value["data_id"]
        if data_id in identities and identities[data_id]["users"] != value["users"]:
            raise RecoveryError("Conflicting identity mappings for one data_id")
        identities[data_id] = value
    with Path(inventory_path).open() as stream:
        inventory = json.load(stream)
    assets = inventory.get("assets")
    if not isinstance(assets, list):
        raise RecoveryError("Inventory assets must be a list")

    def record(path, mode, recovered, **extra):
        entry = {"path": str(path), "sha256": sha256(path), "mode": mode,
                 "count": len(recovered), "status": "recovered", **extra}
        users.update(recovered)
        registry.append(entry)
        sources.append(entry.copy())

    for path in conservative_paths:
        value = load_identity(path)
        record(path, "conservative_entire_encoder", set(value["users"]) - {"__PAD__", "__UNKNOWN__"},
               data_id=value["data_id"], mapping_source_sha256=value.get("source_sha256"))
    for path in legacy_paths:
        with Path(path).open() as stream:
            legacy = json.load(stream)
        provenance = ({"original_source": legacy.get("source"),
                       "original_source_sha256": legacy.get("source_sha256")}
                      if isinstance(legacy, dict) else {})
        del legacy
        record(path, "legacy_raw_user_list", raw_list(path), **provenance)
    seen = set()
    for asset in assets:
        raw_path = asset if isinstance(asset, str) else asset.get("path", asset.get("file"))
        if not isinstance(raw_path, str):
            raise RecoveryError("Inventory asset missing path")
        path = Path(raw_path)
        if raw_path in seen:
            continue
        seen.add(raw_path)
        mode = ("prediction_json" if path.suffix == ".json" and "predictions" in path.name.lower()
                else "uid_npz" if path.name.endswith("_users.npz")
                else "user_csv" if path.suffix.lower() == ".csv" and not source_csv(path)
                and (path.name.lower().startswith("preds_")
                     or any(word in path.name.lower() for word in ("prediction", "result", "recommendation"))) else None)
        if mode is None:
            continue
        try:
            extra = {}
            if mode == "prediction_json":
                recovered = prediction_users(path)
            elif mode == "user_csv":
                recovered = csv_users(path)
                if recovered is None:
                    registry.append({"path": str(path), "mode": mode, "status": "skipped_no_user_id_header"})
                    continue
            else:
                recovered, data_id, manifest = npz_users(path, identities)
                extra = {"data_id": data_id, "manifest_path": str(manifest),
                         "manifest_sha256": sha256(manifest),
                         "mapping_source_sha256": identities[data_id].get("source_sha256"),
                         "identity_path": identities[data_id]["identity_path"],
                         "identity_sha256": identities[data_id]["identity_sha256"]}
            record(path, mode, recovered, **extra)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            entry = {"path": str(path), "mode": mode, "status": "unresolved",
                     "reason": str(exc) if isinstance(exc, RecoveryError) else type(exc).__name__}
            if path.is_file():
                entry["sha256"] = sha256(path)
            registry.append(entry)
    completeness = {"attested": False,
                    "reason": "Conservative partial recovery; earliest historical sources remain uncertain."}
    output.mkdir(parents=True, exist_ok=False)
    exclusions = {"schema_version": 1, "raw_user_ids": sorted(users), "sources": sources,
                  "completeness": completeness}
    summary = {"schema": 1, "completeness": completeness, "unique_raw_user_count": len(users),
               "inventory_sha256": sha256(inventory_path), "sources": registry,
               "encoders": [{"data_id": key, "user_count": len(value["users"]) -
                             sum(x in {"__PAD__", "__UNKNOWN__"} for x in value["users"]),
                             "outside_recovered_union": sum(x not in users and x not in {"__PAD__", "__UNKNOWN__"}
                                                            for x in value["users"]),
                             "source_sha256": value.get("source_sha256"),
                             "identity_path": value["identity_path"],
                             "identity_sha256": value["identity_sha256"]}
                            for key, value in identities.items()]}
    for name, value in (("historical_users_partial.json", exclusions), ("registry.json", summary)):
        with (output / name).open("w") as stream:
            json.dump(value, stream, ensure_ascii=True)
            stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--identity", action="append", required=True, type=Path)
    parser.add_argument("--conservative-identity", action="append", default=[], type=Path)
    parser.add_argument("--legacy-users", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = recover(args.inventory, args.identity, args.conservative_identity, args.legacy_users, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, "Historical recovery failed: " + (str(exc) if isinstance(exc, RecoveryError) else type(exc).__name__) + "\n")
    print(json.dumps({"unique_raw_user_count": result["unique_raw_user_count"],
                      "unresolved_sources": sum(x["status"] == "unresolved" for x in result["sources"]),
                      "completeness": result["completeness"]}))


if __name__ == "__main__":
    main()
