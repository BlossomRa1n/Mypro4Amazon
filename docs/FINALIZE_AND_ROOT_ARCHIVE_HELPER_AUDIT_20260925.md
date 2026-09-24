# Finalize与fresh-root归档helper审计

状态：**代码审计PASS；尚未执行正式finalize或fresh-root归档，不是操作授权或关机授权。** 审计者只读helper、依赖和runbook并运行全mock/临时文件轻量测试，没有SSH、训练、评分或正式test指标/预测读取。

| 文件，均位于 `server_snapshot/full_final_test100k_20260925/` | SHA-256 |
|---|---|
| `finalize_archive_only.py` | `e9c3b4211bbf66dc317593af3657b0944f490ea1190c85478550e75b19726843` |
| `test_finalize_archive_only.py` | `463ee18e317c79b65b1070554731ddd8eff6d31fddc7079761042962cc6cf992` |
| `FINALIZE_AND_ROOT_ARCHIVE_RUNBOOK.md` | `9a1833aa4c808a644d9ac30b7a75371a30b9a584248841f36e4a76742784ef50` |

本runbook版本只在先前已审说明后追加两个独立helper入口与operations冻结要求。Marker helper及其test保持原SHA `448265298e9a5fbecb338a090d85d0632ba406ba45504686b8afb5dd371c4832` / `37375d3f2c9cf71b5365e424490f3263a70148fcc89a9987263bf6ebe3111e73`；先前marker审计文档内runbook SHA是当时被审版本的历史记录，本审计列出当前扩展版本。

## 1. 入口与本地前置门禁

Helper只有 `finalize-only` 与 `archive-final-root` 两个命令，均要求主执行者单独给出GO，不能pipeline自动跨阶段。固定使用既有SSH control socket、主机/端口、远端Python、原final root及原train/test目录，没有train/test/score/shutdown入口。全新输出目录、receipt、inventory、transfer/failure log与execution回执必须不存在，已有输出停止并保留。发现final operations root或指定closeout_gates目录内已发布CLOSEOUT_VERIFIED时拒绝执行；主执行者仍须对其他位置的closeout和完整operations冻结负责。

本地门禁复用固定SHA的seal、closeout、watcher模块，不修改冻结执行器。训练链绑定原auth/plan、训练completion/checkpoint、run/control transfers与真实watcher命令和exit；prepared需精确canonical、raw checkpoint和实际seal成功回执。Test链核run/control完整inventory与SHA/size、控制argv与exit0、正式semantic receipt精确证据覆盖、inputs/helpers、真实watcher完成和原路径global-marker来源。Global/session marker与prepared/checkpoint/records/目录一致；TEST_COMPLETED必须是raw-only、100k、execution_complete、无paired、structure_gain_accepted null和business_threshold_registered false。实际指标正确性沿用固定且已执行的semantic gate，不在helper内重新评分。

## 2. Finalize操作及原文保全

远端每个操作检查与本地验收字节绑定的auth/plan/prepared、训练和test完整已知run/control文件、marker、training archive receipt以及冻结executor/monitor SHA。扫描/proc精确argv确认无active executor/monitor，正式run/control无FAILED/FAILURE且completion exit0。Preflight/upload要求远端test archive receipt和FINAL_COMPLETED不存在；upload仅写已验收receipt，xb、flush/fsync并读回SHA/size。

Upload后再次核本地完整门禁及远端before-finalize条件，执行固定冻结executor的唯一finalize命令。真实command、脚本SHA、起止UTC、stdout/stderr及returncode写入新目录；仅exit0后下载原FINAL_COMPLETED。下载阶段对final路径作安全和regular存在性检查，不解析其JSON；原stdout及local FINAL_COMPLETED原字节先保留，再在本地要求精确schema、plan/protocol、raw-only null、business false，以及test completion和archive receipt的精确远端path/SHA/size。末尾再次核本地门禁后才发布finalize COMPLETED。

首版REMOTE_CHECK曾在download分支先解析/校验远端final后再输出，坏JSON或wrong-contract时不能保留原文；审计要求修复，当前版本已将download独立为opaque读回，archive仍保留契约校验。新增回归case直接执行真实mode dispatch与读回tail，验证坏JSON和不合法contract都会输出并先保存原字节，再由本地拒绝。

所有非零、格式错误和timeout停止且无自动重试；stdout/stderr、raw下载及partial保留。Timeout明确记remote outcome unknown，不能把本地SSH退出当成远端操作未发生，也不能删除已有输出后盲目重跑。未知结果必须先由主执行者只读调查。

## 3. Fresh-root完整归档

`archive-final-root`独立要求本地FINAL_COMPLETED精确合同、成功finalize回执和同SHA，再将本地final原字节加入远端bound，并重新确认远端全部前置门禁和空闲。执行固定SHA的 `tools/archive_verified_remote_tree.py`，实际argv使用绝对工具路径、实际可执行Python路径以及固定远端root/FINAL_COMPLETED角色。冻结工具提供两次远端inventory相等、local exact coverage与全部SHA/size；helper再次运行本地transfer校验且要求归档内FINAL_COMPLETED与已下载原文相同。

只有真实archive exit0、tool执行前后SHA相同、transfer链通过、stdout最后JSON精确等于receipt且stderr为空，才发布 `final_remote_root_transfer_execution.json` 及archive COMPLETED。Execution记录实际argv、tool/receipt/inventory SHA、stdout/stderr path/SHA/bytes，符合已审closeout schema。旧plan_only草稿、首次失败日志、旧snapshot与旧archive不被回写。

## 4. 独立轻量测试与操作限制

审计者运行：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 server_snapshot/full_final_test100k_20260925/test_finalize_archive_only.py -v
```

**5项测试，0.042秒，全部PASS。** 包含无GO/已有输出前停止、finalize完整mock成功、非零/timeout/坏download保全、真实download分支坏原文保全、archive真实argv和execution schema mock、路径穿越/symlink与xb保护。网络和正式执行均mock，测试stdout的verified是fixture结果，不代表正式finalize或归档已完成。实现者同5项0.042秒PASS；首版4项虽通过但未覆盖上述真实download分支缺口，不能把它解释为当前5项之外的正式执行。

正式命令应使用已验证含NumPy的诊断venv，因为本地链模块会导入NumPy。每一步仍须主执行者按真实状态单独放行。完成finalize/fresh-root后，结果报告与工作总结在关机前写完并冻结，再进行汇总closeout和源码恢复补充；closeout后不得修改operations_root、OPERATIONS_STATE、checklist或已绑定报告。关机、closure、commit/push使用独立外部记录，并保留恢复与explicit shutdown GO门禁。
