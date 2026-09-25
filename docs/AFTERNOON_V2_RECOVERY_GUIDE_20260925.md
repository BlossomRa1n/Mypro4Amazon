# 本轮结果、恢复与重放入口

日期：2026-09-25。适用于本轮V2训练候选池对齐实验。先读结果报告，再决定是否需要恢复；恢复本身不会重新训练或评价test。

## 1. 结果与证据入口

- 正式结论：`AFTERNOON_V2_RESULTS_20260925.md`。本轮未通过晋级门槛，保留旧训练池及raw cross。
- 固定科学合同：`AFTERNOON_V2_TRAIN_POOL_PROTOCOL_20260925.md`，SHA-256 `e0b5b95aaf2d9b39f55849c41c1733906af1eb8e4ba3de14973f0859c150ac05`。
- 本地证据根：`server_snapshot/optimization_20260925_evening/`，以下称证据根。训练、候选、逐用户开发结果和身份资料均不提交Git。
- 正式独验：证据根下`TRAINING_RUN_INDEPENDENT_VERIFIED.json`。来源、模型和数组验收与统计重算已完成；这不是重新训练得到第二次独立实验。
- 本轮没有打开旧10万人test重新评分。历史test结果不能用来选择本轮模型。

## 2. 必须一起保留的三部分

| 部分 | 位置与范围 |
|---|---|
| 主体恢复包 | 项目父目录`recovery_capture_v1/experiment_recovery.tar.gz`，64,690,743字节；Git历史、245份当前源码/文档/测试和220份小回执 |
| 外部大资产 | `recovery_capture_v1/payload/recovery.json`中的290条外部引用，合计14,887,326,184字节；保留对应本地文件与路径映射 |
| 后续补充 | 独立闭包审计发现的历史验收/身份锁依赖及主包之后的收尾文件，必须以最终收尾记录指定的补充清单和包为准 |

主包SHA-256为`157e69a28d5233a9779eec399971ce7df2d9047ac1b45eb047671c0c878d8925`；主`VERIFIED.json`的SHA为`5f18c1ae884e941d8faa74104b341dba671d13b269e4c80208101b24f7c65462`；内层manifest SHA为`b1dc193acaa09ffaffe3930a3c0ff32952f5d0354d55d6ffaf1f835436e63327`。

**65MB主包不能单独恢复全部模型与数据。** 主包的源码冷恢复已成功，但不能因此宣称主包包含所有重放依赖。后续遗漏项通过新补充包修复，旧主包不修改、不删除。

## 3. 恢复步骤

1. 复制主包、其`VERIFIED.json`、最终补充包及所有外部资产到有足够空间的本地存储；保留原始副本。GitHub只保存项目源码和说明，不能代替模型/数据备份。
2. 按最终收尾清单逐文件校验SHA-256、大小和资产映射。先核外层压缩包，再核manifest及成员集合；遇到不符就停止，不覆盖旧文件、不重新生成回执冒充一致。
3. 在全新空目录安全解压，只接受唯一、普通、规范相对路径文件。禁止绝对路径、父目录穿越和符号链接。主包内`RESTORE.md`说明恢复接口；先验证helper字节，再使用它。
4. 通过封存helper的`restore_verify(None, extracted_payload, new_work_directory, state, selected_paths, working_tree_files)`恢复Git历史、分支/refs、暂存区、未提交修改及未跟踪源码。恢复目录不依赖原项目或网络；大资产按清单另行映射。
5. 应用最终补充范围并核验完整输入闭包。只有相同代码、环境、数据/资产及协议可用时，才考虑另行授权重放；不得直接重跑旧一次性test入口。

真实独立冷恢复产物位于项目父目录`independent_cold_restore_audit_v1/`，回执`INDEPENDENT_COLD_RESTORE_VERIFIED.json` SHA为`281a80ef0a55e4aafb5baf72e73aed043effb21dd60ed53cf8f3d534acfcda6a`。它核对474个安全成员、245个源码文件、HEAD/refs/status和暂存/非暂存补丁；651个排除的历史路径未落地。后续闭包审计单独记录，不与源码恢复成功混同。

## 4. 只复算已有结果时

不需要服务器或GPU，也不需要重新打开未来目标。已执行命令及参数保存在证据根：

| 目的 | 已执行请求及对应工具 |
|---|---|
| 完整正式验收 | `independent_run_verification_v1/request.json`；`tools/verify_v2_train_alignment.py` |
| 描述性统计、固定点曲线、pool变化 | `descriptive_analysis_execution_v1/request.json`；`tools/analyze_v2_train_alignment.py` |
| 所选停止点用户历史消费覆盖 | `tools/analyze_training_exposure.py`，只重建已封训练行顺序 |
| 已见/未见用户分组 | `tools/analyze_v2_exposure_groups.py`，使用已存配对开发数组；分组是描述性分析 |

复算必须改用**新的输出路径**，不能覆盖冻结回执；保留输入SHA和实际退出码。正式验收约76秒是本次本地实测，不保证其他机器耗时。固定的10,000次bootstrap采用用户维度配对，不把三seed当作240,000个独立用户。

## 5. 范围与保密边界

主体包保留Git可达历史；当前文本、补丁和明确的小回执已做凭据形状预检，但没有对全部历史对象完成凭据审计，因此恢复包只作私人备份。它不包含操作系统镜像、完整Python环境、Git hooks、SSH配置或私钥。原运行环境及版本记录是复现参考，不是离线环境安装包。

文献核验错误及被拒绝/提前停止的本地操作记录作为事故材料保留，不能被当成训练或恢复成功证据。只有实际训练、传输、统计、恢复和关机返回才能支撑对应完成声明。最终关机和Git结果以独立收尾记录为准。

## 6. 可复制的本地验收命令

以下从已执行的两个 `request.json` 参数转换为项目根相对路径，将三处历史输入映射到主清单已封存且逐字节／SHA相同的 canonical 位置，并将输出换到新目录；不执行训练或新模型评分。完整验收会读取既有模型及 screen/confirm 开发数组（包括已保存的开发标签）并重算统计，不打开最终 test 评分数组，也不重新调用数据集未来目标。先完成主包和补充的闭包核验；缺失输入时停止，不能修改协议或生成替代回执绕过门禁。

先进入恢复后的项目根，设置已有的本地科学环境和一个从未使用的输出目录。`VERIFY_PY` 必须指向已经安装所需依赖的 Python；示例是本次已用路径，迁移机器时应替换。`CHECK_OUT` 是新路径占位，不指向任何冻结证据目录。`mkdir` 若报告已存在，停止并换一个新目录。

```sh
cd '/Users/admin/projects/分布式训练/AI/Mypro4Amazon'
VERIFY_PY='/Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python'
CHECK_OUT="$PWD/server_snapshot/local_revalidation_NEW_RUN"
mkdir "$CHECK_OUT"
```

完整正式验收命令：

```sh
"$VERIFY_PY" tools/verify_v2_train_alignment.py \
  --pilot-dir server_snapshot/optimization_20260925_evening/pilot_run \
  --pool-dir server_snapshot/optimization_20260925_evening/full_pool_run \
  --protocol-manifest server_snapshot/training_diagnostics_20260924/prelaunch_server/protocol_v2/protocol_manifest.json \
  --old-protocol-manifest server_snapshot/next_cross_20260924/protocol_complete/protocol_manifest.json \
  --old-train-cache server_snapshot/next_cross_20260924/protocol_complete/candidate_train.json \
  --protocol-doc docs/AFTERNOON_V2_TRAIN_POOL_PROTOCOL_20260925.md \
  --legacy-source-dir server_snapshot/training_diagnostics_20260924/remote_root/src_v2/code \
  --builder-source server_snapshot/optimization_20260925_evening/training_source_archive_v1/code/build_v2_train_pool.py \
  --run-dir server_snapshot/optimization_20260925_evening/training_run \
  --raw-run server_snapshot/training_diagnostics_20260924/run \
  --raw-archive-receipt server_snapshot/training_diagnostics_20260924/archive_verified.json \
  --backup-receipt server_snapshot/training_diagnostics_20260924/RECOVERY_BACKUP_BINDING.json \
  --runner-source server_snapshot/optimization_20260925_evening/training_source_archive_v1/code/run_v2_train_alignment.py \
  --receipt "$CHECK_OUT/training_verified.json"
```

描述性汇总命令：

```sh
"$VERIFY_PY" tools/analyze_v2_train_alignment.py \
  --run-dir server_snapshot/optimization_20260925_evening/training_run \
  --raw-run server_snapshot/training_diagnostics_20260924/run \
  --pool-dir server_snapshot/optimization_20260925_evening/full_pool_run \
  --old-pool server_snapshot/next_cross_20260924/protocol_complete/candidate_train.json \
  --old-positions server_snapshot/next_cross_20260924/protocol_complete/ranker_train_positions.npy \
  --protocol-manifest server_snapshot/training_diagnostics_20260924/prelaunch_server/protocol_v2/protocol_manifest.json \
  --output "$CHECK_OUT/descriptive_summary.json"
```

三处映射已经实际逐字节与SHA核对：旧 `remote_root/protocol_v2/protocol_manifest.json` 对应 `prelaunch_server/protocol_v2/protocol_manifest.json`；旧 `protocol_v3` 下的清单和positions对应 `protocol_complete` 下同名文件。旧协议同目录的train/screen/confirm records须随 `protocol_complete` 一起保留。该映射基于内容一致，不能仅凭文件名互换。迁移时只替换本地位置，不改任何已封JSON内的绝对路径或SHA；验收器按显式参数映射原远端来源。保存命令、退出码和新结果 SHA；本指南中的命令未在编写期间重新执行。

### 环境和最小输入范围

- 原训练版本入口：`training_run/run_manifest.json.environment`；旧控制版本入口：`training_diagnostics_20260924/run/run_config.json.environment`。原训练为 Python 3.12.3、NumPy 2.5.2、PyTorch 2.3.0+cu121、CUDA 12.1、cuDNN 8902；这是历史记录，不是自动安装处方。
- `requirements.txt` 声明依赖但未固定版本。本地验收使用 CPU；其真实解释器和参数见上述 request。独立恢复 helper 只需 Python 标准库与 Git。Matplotlib 仅属于另建的绘图环境，不是本轮训练或验收新增依赖。
- 完整本地验收至少需要：两个协议清单及旧协议同目录的 records/positions、旧 train cache 的元数据和 items/lengths、原 `src_v2/code` 完整源码集合、冻结新协议、封印 builder/runner、pilot 和 full pool 完整树、新 run、旧 raw 完整树、旧 `archive_verified.json` 与 `RECOVERY_BACKUP_BINDING.json`，以及验收器调用的固定 SHA 原 reader。逐文件以最终闭包补充清单为准。
- 描述汇总需要既有新旧 run、full pool、旧 train pool 与 positions；它不加载模型或调用新评分，但会读取已有开发结果。只复算这些产物不需要反序列化 `base_archive/data.pkl`。
- 重播训练需要更大的闭包：原合法训练底座 `data.pkl/itemcf.pkl/svd.npy/v2_best.pth/run_manifest.json`、全部旧候选缓存及训练/开发身份、协议身份锁与回执、源码、pilot门禁和旧raw复用门禁。身份锁证明用于拒绝测试访问；它不是读取未来标签的许可。不得把本地验收的最小输入误当作训练闭包。训练重播需另行授权，本指南没有给出启动命令。

### 当前 helper 与主包内 helper

主包内 helper 保留原 SHA，不替换。原版本的仓库内输出检查受 `GIT_LITERAL_PATHSPECS` 与 `check-ignore` 不兼容影响，曾在创建目录前拒绝实际已忽略的路径；主包因此在项目父目录成功生成。后续源码只修复此兼容分支并新增两项真实 Git 测试；该版本与本指南需由补充包覆盖。旧主包冷恢复仍使用其已封存 helper，不需要这个输出检查修复。

若另行授权创建新源码恢复包，两个清单须是最终经过 SHA/bytes 核验、互不重叠的显式依赖清单；输出必须不存在。下例中的清单和输出都是待替换的新路径，不能拿旧清单绑定已修改的源码或可变状态：

```sh
python3 tools/build_experiment_recovery.py \
  --repo "$PWD" \
  --external-assets '/ABS/FINAL/external_assets.json' \
  --receipts '/ABS/FINAL/receipts.json' \
  --output '/ABS/NEW/recovery_output'
```

helper 会重新验证依赖、恢复 Git 状态并检查源未变；没有 `VERIFIED.json` 就不能声称完成。它不复制外部大资产，也不会补全遗漏的依赖清单。
