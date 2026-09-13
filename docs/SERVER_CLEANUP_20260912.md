# Server Disk Cleanup, 2026-09-12

Host: AutoDL instance accessed on SSH port 48427.

| Disk | Reclaimed | Available after cleanup |
| --- | ---: | ---: |
| System `/` (30 GiB) | 5.134 GiB | about 24 GiB |
| Data `/root/autodl-tmp` (50 GiB) | 12.836 GiB | about 14 GiB |
| Total | 17.970 GiB | |

Removed 14 duplicate checkpoint files across `baseline_379k`, `token_fix`, `exp_3a1`, `exp_3a2`, `exp_3a3`, `exp_stageA` and `exp_stageB`. For every deletion, sorted tensor names, dtypes, shapes and exact bytes produced the same SHA-256 as a retained checkpoint, and epoch/configuration matched. Removed checkpoint metadata was recorded in the audit. Checkpoints containing optimizer state were excluded from cleanup.

All best models, unique weights (including Stage B epoch 1), training data, logs, metrics, existing recall artifacts and the installed Python runtime were retained. After deletion, all 14 removed paths were verified absent and all corresponding retained paths verified present.

Full audit: server `/root/cleanup_audit_20260912.json`; local `server_snapshot/2026-09-12/cleanup_audit_20260912.json`. Cleanup implementation: `tools/cleanup_server_checkpoints.py`.

Available space is the measurement immediately after cleanup, before the corrected baseline creates new artifacts.
