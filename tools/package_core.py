"""Create and verify a source-only backup with a per-file SHA-256 manifest."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--description", default="Core source backup before unattended optimization; data, weights, environments and SSH credentials excluded.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = []
    for directory in ("code", "tests", "tools", "docs"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file()
                     and not p.is_symlink() and "__pycache__" not in p.parts
                     and p.suffix in (".py", ".md", ".sh", ".bat", ".json", ".toml", ".txt"))
    names = ("README.md", "RESULTS.md", "CHANGELOG.md", "FUTURE_WORK.md",
             "AGENTS.md", "CLAUDE.md", "requirements.txt", ".gitignore")
    paths.extend(root / name for name in names if (root / name).is_file())
    paths.extend(p for p in root.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".bat"))
    paths = sorted(set(paths))
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "description": args.description,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            name, content = path.relative_to(root).as_posix(), path.read_bytes()
            archive.writestr(name, content)
            manifest["files"].append({"path": name, "bytes": len(content),
                                      "sha256": hashlib.sha256(content).hexdigest()})
        archive.writestr("BACKUP_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for entry in manifest["files"]:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "  " + output.name + "\n", encoding="ascii")
    print(json.dumps({"path": str(output), "files": len(paths), "bytes": output.stat().st_size,
                      "sha256": digest, "verified": True}, indent=2))


if __name__ == "__main__":
    main()
