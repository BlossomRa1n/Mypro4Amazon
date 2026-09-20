"""Package a completed, audited run; verify its downloaded archive locally.

This tool never shuts down a server, removes training artifacts, or commits
data. The archive and raw user arrays belong in ignored server_snapshot/.
"""
import argparse
import hashlib
import json
import math
import pickle
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite_json(value, label):
    if isinstance(value, float):
        require(math.isfinite(value), label + ": nonfinite JSON number")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, label + "/" + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{label}/{index}")


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    finite_json(value, str(path))
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def readable(path):
    if path.suffix == ".json":
        read_json(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            require(arrays.files, str(path) + ": empty NPZ")
            for name in arrays.files:
                value = arrays[name]
                require(value.dtype.kind in "biuf", str(path) + ": unexpected array dtype")
                require(np.isfinite(value).all(), str(path) + ": nonfinite array " + name)
    elif path.suffix == ".pkl":
        # Only our authenticated, SHA-verified candidate caches are unpickled.
        require(path.name.startswith("pools_") and path.parent.name == "cache",
                str(path) + ": unexpected pickle in evidence package")
        with path.open("rb") as stream:
            pools = pickle.load(stream)
        require(isinstance(pools, list) and len(pools) == 100000, str(path) + ": wrong pool count")
        for pool in pools:
            require(len(pool) == 75, str(path) + ": wrong candidate count")
            require(len(set(pool)) == 75, str(path) + ": duplicate candidate")
            if isinstance(pool, dict):
                require(all(math.isfinite(float(value)) for value in pool.values()),
                        str(path) + ": nonfinite candidate score")


def completion_gate(run):
    monitor = read_json(run / "monitor_status.json")
    require(monitor["status"] == "complete" and monitor["exit_code"] == 0,
            "Training monitor has not completed successfully")
    require(read_json(run / "COMPLETED.json")["status"] == "complete", "Missing completion")
    require(not (run / "SUPERSEDED.json").exists(), "Run is superseded")
    require(read_json(run / "controlled_paired_verified.json")["status"] == "validated",
            "Four-way result validation did not pass")
    protocol = read_json(run / "protocol_audit.json")
    require(protocol["status"] == "passed", "Protocol audit did not pass")
    require(protocol.get("safe_shutdown_eligible") is True,
            "Protocol audit does not certify the full required checks")
    for name in ("results.json", "suite_manifest.json", "train.log"):
        require((run / name).is_file(), "Missing required artifact: " + name)
    log = (run / "train.log").read_text(encoding="utf-8", errors="replace")
    require(not re.search(r"Traceback|\b(?:FloatingPointError|RuntimeError|ValueError|"
                          r"AssertionError|OutOfMemoryError|Killed|nan|inf)\b", log, re.IGNORECASE),
            "Training log contains an unexplained failure or nonfinite marker")


def build(run, output_dir):
    run, output_dir = run.resolve(), output_dir.resolve()
    require(output_dir != run and run not in output_dir.parents, "Archive must be outside run directory")
    completion_gate(run)
    payload = {}
    excluded = {"EVIDENCE_MANIFEST.json", "SHA256SUMS"}
    for path in sorted(run.rglob("*")):
        require(not path.is_symlink(), "Evidence contains a symlink: " + str(path))
        if not path.is_file():
            continue
        relative = path.relative_to(run).as_posix()
        if relative in excluded:
            continue
        require(path.suffix != ".tmp", "Incomplete temporary artifact: " + relative)
        readable(path)
        payload[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest = {"status": "packaged", "source_run": str(run),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "file_count": len(payload), "files": payload}
    write_json(run / "EVIDENCE_MANIFEST.json", manifest)
    checksum_paths = sorted(payload) + ["EVIDENCE_MANIFEST.json"]
    (run / "SHA256SUMS").write_text(
        "".join(f"{digest(run / name)}  {name}\n" for name in checksum_paths), encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / (run.name + "_evidence.tar.gz")
    require(not archive.exists(), "Archive already exists; verify it rather than overwrite evidence")
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for name in sorted(payload) + ["EVIDENCE_MANIFEST.json", "SHA256SUMS"]:
            stream.add(run / name, arcname=name, recursive=False)
    # Reject any source changed while the package was being built.
    for name, entry in payload.items():
        require(digest(run / name) == entry["sha256"], "Source changed during archive: " + name)
    checksum = digest(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{checksum}  {archive.name}\n", encoding="utf-8")
    return {"status": "packaged", "archive": str(archive), "archive_sha256": checksum,
            "archive_bytes": archive.stat().st_size, "payload_files": len(payload)}


def verify(archive, checksum_file, destination):
    expected = checksum_file.read_text(encoding="utf-8").split()[0]
    actual = digest(archive)
    require(actual == expected, "Downloaded archive SHA-256 mismatch")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as stream:
        names = set()
        for member in stream.getmembers():
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts,
                    "Unsafe archive member: " + member.name)
            require(member.name not in names, "Duplicate archive member: " + member.name)
            names.add(member.name)
            target = destination.joinpath(*path.parts)
            require(destination in target.resolve().parents, "Archive path escapes destination")
            require(not target.is_symlink(), "Destination contains a symlink")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if target.exists():
                # Keep existing evidence if a previous download was interrupted.
                require(digest(target) == hashlib.sha256(source.read()).hexdigest(),
                        "Existing destination differs: " + member.name)
            else:
                with target.open("xb") as output:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(block)
    manifest = read_json(destination / "EVIDENCE_MANIFEST.json")
    require(names == set(manifest["files"]) | {"EVIDENCE_MANIFEST.json", "SHA256SUMS"},
            "Archive inventory differs from manifest")
    for line in (destination / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", 1)
        require(name in names, "Checksum entry is not in archive")
        require(digest(destination / name) == checksum, "SHA256SUMS mismatch: " + name)
    for name, entry in manifest["files"].items():
        path = destination / name
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                "Payload checksum/size mismatch: " + name)
        readable(path)
    completion_gate(destination)
    return {"status": "verified", "verified_utc": datetime.now(timezone.utc).isoformat(),
            "archive": str(archive.resolve()), "archive_sha256": actual,
            "local_directory": str(destination), "payload_files": len(manifest["files"]),
            "all_payload_sha256_matched": True, "json_npz_candidate_caches_readable": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("build")
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--archive", type=Path, required=True)
    check.add_argument("--checksum-file", type=Path, required=True)
    check.add_argument("--extract-to", type=Path, required=True)
    check.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.run_dir, args.output_dir)
    else:
        result = verify(args.archive, args.checksum_file, args.extract_to)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.receipt, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
