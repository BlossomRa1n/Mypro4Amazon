# Server Disk Cleanup, 2026-09-19

Host: AutoDL instance `connect.weste.seetacloud.com:43050`.

## Result

| Disk | Before | After | Reclaimed |
| --- | ---: | ---: | ---: |
| System `/` (30 GiB) | 27 GiB used, 3.9 GiB free | 16 GiB used, 15 GiB free | about 11.1 GiB |
| Data `/root/autodl-tmp` (50 GiB) | 45 GiB used, 5.3 GiB free | unchanged | 0 |

No training process was running when cleanup started. The future-window baseline had a completed marker and exit code 0.

## Removed

Only restart snapshots and temporary files were removed:

- `/root/future_window_baseline_din_latest.pth`
- `/root/corrected_baseline_20260912/din_latest.pth`
- `/root/corrected_baseline_20260912/v2_latest.pth`
- `/root/baseline_379k/model_data/din_ext_latest.pth`
- `/root/token_fix/model_data/din_token_latest.pth`
- Smoke-run `din_latest.pth` and `v2_latest.pth` files under `/root/corrected_baseline_smoke_20260912`
- `/tmp/final_supporting_results_20260913.tar.gz`
- `/tmp/optimization_completed_results_20260912.tar.gz`
- `/tmp/train_din.log`

Total file sizes removed: 11,897,685,988 bytes. These files were not needed for inference after the runs completed. The corresponding `*_best.pth` files remain.

## Retained

All best checkpoints, data artifacts, metrics, result JSON/CSV files, current project code, the Python environment, CUDA runtime, and the active data-disk experiment directory were retained. In particular, `/root/autodl-tmp/future_window_baseline_20260919` still contains `din_best.pth`, `v2_best.pth`, `data.pkl`, `itemcf.pkl`, `svd.npy`, validation/test predictions, and `results.json`.

## Not removed yet

Historical experiment directories such as `/root/optimization_20260912`, `/root/followup_20260912`, `/root/candidate_train_20260912`, `/root/final_optimization_20260912`, and `/root/corrected_baseline_20260912` still contain archived best models and evidence. They can be archived or deleted later if those comparisons are no longer needed.

## Data-disk audit and cleanup

The data disk was audited after the final evaluation. The largest directories are raw `amazon_data` (about 19 GiB), legacy `user_data` (about 9.2 GiB after cleanup), `structural_20260912` (about 5.3 GiB), Stage A/B runs (about 2.6 GiB each), and the current future-window baseline (about 2.6 GiB). Raw data, the current baseline, and final result directories were retained.

Before deletion, SHA-256 confirmed that each structural experiment's `final.pth` matched its `best.pth`. The five redundant `final.pth` names were removed while the `best.pth` files and experiment metadata were retained. The old `/root/autodl-tmp/user_data/model_data/item_embeddings_v2.pkl` was also removed because it matched `item_embeddings_v2_phase2.pkl` byte-for-byte. The data disk increased from about 5.3 GiB to 5.6 GiB free; the structural files were hard-linked duplicate names, so removing those names did not duplicate-free additional blocks.

`two_tower_v2_latest.pth` was checked separately and did **not** match `two_tower_v2_best_phase2.pth`; both remain. This exception overrides the earlier audit suggestion to remove the latest file.

The rejected Category-weight `0.4` eval-only run temporarily created about 1 GiB of duplicate `data.pkl`, `svd.npy`, and `itemcf.pkl` files. After its result snapshot was copied locally, those three cache files were removed; its 31 MiB results/prediction directory remains. The data disk now has about 5.5 GiB free. Raw data, the future-window baseline, accepted final results, and legacy model directories remain intact.
