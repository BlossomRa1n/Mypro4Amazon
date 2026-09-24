# 原路径全局test marker只读捕获审计

状态：**代码审计PASS；尚未实际捕获正式test marker，本文不是启动test或关机授权。** 审计者只读脚本与运行手册并执行隔离mock测试，没有SSH、正式test指标/预测读取、训练、评分或正式marker写入。

| 文件，均位于 `server_snapshot/full_final_test100k_20260925/` | SHA-256 |
|---|---|
| `capture_global_marker.py` | `448265298e9a5fbecb338a090d85d0632ba406ba45504686b8afb5dd371c4832` |
| `test_capture_global_marker.py` | `37375d3f2c9cf71b5365e424490f3263a70148fcc89a9987263bf6ebe3111e73` |
| `FINALIZE_AND_ROOT_ARCHIVE_RUNBOOK.md` | `e6a6b6fd35799604998ff7866b531b53b8ddf730b92eb11140282530766f9061` |

Helper只接受marker-capture命令，只通过既有SSH control socket执行只读检查。它固定本次原authorization和final plan SHA，检查local prepared canonical、raw-only/100k/同plan、seal_models_v1成功回执、prepared原plan path/SHA/size和唯一raw checkpoint角色。原全局marker路径由auth test_lock父目录推导，固定会话marker位于正式test run。它不允许通过新身份目录或新test目录重置一次性限制。

远端只读流程核同plan/prepared原字节、prepared身份及实际raw checkpoint SHA。Global marker与session marker各读取三次，要求各自前中后原字节相等，二者原字节也相等。所有路径及祖先拒绝symlink和`..`。返回值包含真实请求、SHA/size、原始base64字节及checkpoint结果；本地验证同一请求、角色/路径、三读稳定性，以及marker精确schema：status/protocol、plan/prepared hash、raw-only、checkpoint hash、test records hash、原test目录和带时区时间。

本地输出全部排他创建。脚本先保存真实stdout/stderr/exit，再保存global/session原始readback，即使后续链不合法也保留证据。正式 `GLOBAL_FINAL_TEST100K_STARTED.json` 和source verified回执只在验证成功且本地source chain再次相同后创建。非零、坏响应、SHA漂移或超时均停止，无自动重试；timeout保存partial stdout/stderr和unknown状态，不把SSH结束当成远端状态证明。

Source回执符合已审closeout schema，另绑定实际request/exit/stdout/session原文SHA。捕获后才能启动要求已有本地全局marker的test watcher。该捕获仅证明既有marker的原路径和字节来源，不证明test已经评分完成，也不替代TEST_COMPLETED、模型/候选/逐用户归档及语义校验。

Runbook将真实test归档通过后的finalize与新final remote-root archive分开，要求精确原plan/executor、无active final进程、固定输出不存在、只xb上传已经验收的test semantic回执，记录真实command/stdout/stderr/exit，并先下载保留原FINAL_COMPLETED字节再验收。最终root transfer使用已审固定archive工具，要求新output和真实两inventory/完整SHA/size；execution回执必须在exit0后绑定实际stdout最终JSON、receipt和inventory，不能从runbook占位符合成成功证据。旧plan_only草稿、旧snapshot及失败记录保留。该文档只描述操作，不能自行授权test、finalize或shutdown。

审计者独立运行 `PYTHONDONTWRITEBYTECODE=1 /Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python -m unittest discover -s server_snapshot/full_final_test100k_20260925 -p test_capture_global_marker.py -v`，2项测试 **0.013秒PASS**。覆盖原路径/前后SHA/size/actual checkpoint漂移及完整mock成功、invalid、非零、timeout和已有输出保全。全部网络调用mock；打印的verified和marker SHA来自临时fixture，不是正式test已启动的事实。实现者另报告同2项0.013秒通过。

正式执行仅可在主执行者已授权的一次test实际发布marker后进行。任何已有marker/输出或异常均保留并调查，不删除后重复开测。恢复包、完整closeout和关机授权仍是后续独立门禁。
