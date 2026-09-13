"""Remove only optimizer-free checkpoints whose exact weights remain in best/latest."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOTS = [Path("/root/baseline_379k"), Path("/root/token_fix"),
         *[Path("/root/autodl-tmp") / name for name in
           ("exp_3a1", "exp_3a2", "exp_3a3", "exp_stageA", "exp_stageB")]]


def inspect(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "optimizer_state_dict" in checkpoint or "optimizer" in checkpoint:
        return None
    state = checkpoint.get("model_state_dict")
    if state is None:
        return None
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return {"weights_sha256": digest.hexdigest(), "epoch": checkpoint.get("epoch"),
            "config": checkpoint.get("config"), "metrics": checkpoint.get("metrics"),
            "best_auc": checkpoint.get("best_auc"), "patience_counter": checkpoint.get("patience_counter")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit", default="/root/cleanup_audit_20260912.json")
    args = parser.parse_args()
    plan = []
    for root in ROOTS:
        model_dir = root / "model_data"
        retained = {}
        for path in sorted(model_dir.glob("*_best.pth")) + sorted(model_dir.glob("*_latest.pth")):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                continue
            metadata = inspect(path)
            if metadata is None:
                continue
            identity = (metadata["weights_sha256"], metadata["epoch"], json.dumps(metadata["config"], sort_keys=True))
            if identity in retained and path.name.endswith("_latest.pth"):
                plan.append({"remove": str(path), "retain": retained[identity], "bytes": path.stat().st_size, "metadata": metadata})
            else:
                retained[identity] = str(path)
        for path in sorted((model_dir / "checkpoints").glob("*_epoch*.pth")):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                continue
            metadata = inspect(path)
            if metadata is None:
                continue
            identity = (metadata["weights_sha256"], metadata["epoch"], json.dumps(metadata["config"], sort_keys=True))
            if identity in retained:
                plan.append({"remove": str(path), "retain": retained[identity], "bytes": path.stat().st_size, "metadata": metadata})
    audit = {"applied": args.apply, "bytes": sum(p["bytes"] for p in plan), "files": plan}
    Path(args.audit).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"apply": args.apply, "gib": audit["bytes"] / 1024 ** 3,
                      "files": [{k: v for k, v in p.items() if k != "metadata"} for p in plan]}, indent=2), flush=True)
    if args.apply:
        for entry in plan:
            if not Path(entry["retain"]).is_file():
                raise RuntimeError("Retained checkpoint disappeared")
            Path(entry["remove"]).unlink()
        print("Cleanup complete; all unique weights and optimizer states retained.", flush=True)


if __name__ == "__main__":
    main()
