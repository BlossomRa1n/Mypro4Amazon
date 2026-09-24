# 唯一test启动及独立watcher helper审计

状态：**代码审计PASS；没有实际SSH、test启动、watcher启动、训练或评分。** 本审只读源码与已封印依赖，并执行隔离mock测试；正式test指标、预测与未来targets未读。Helper PASS不替代主执行者分别放行两个操作。

| 文件，均位于 `server_snapshot/full_final_test100k_20260925/` | SHA-256 |
|---|---|
| `launch_test_once.py` | `2304d501bfbde75d4b6ce23881ffeed426933fe18b8b96b5a602a499cdb52cf8` |
| `test_launch_test_once.py` | `8830e7d239f3f6819c8dd2b7f10d457cfe3ea6f01fddbebf9752bc768c169d97` |
| `FINAL_TEST_LAUNCH_RUNBOOK.md` | `356ce1aa9a09ec0be0dd00889b9e81b908265def749c9be5c13617c7fe2c6680` |

Runbook新增两个helper入口、真实marker capture条件、准确runtime和上述运营资源门禁，与实现一致；先前seal审计中列出的runbook SHA保留为其历史被审版本，不代表此扩展说明改变冻结执行器。

Helper提供 `launch-test --go` 与 `start-watcher --go` 两个独立入口，无自动衔接或retry。固定原auth/plan、raw-only训练及prepared链；local训练门禁复用已审seal helper，核实际训练语义/transfer/watcher完成、checkpoint字节与seal COMPLETED。新输出必须不存在，已知closeout发布后拒绝写operations。现有本地test_launch_v1或远端永久test_launch_once.lock均阻断重复启动，失败不删除锁或现场。

远端一次启动脚本首先按本地验收SHA/size核auth、plan、prepared、训练semantic、TRAINING_COMPLETED、唯一raw checkpoint和training control完整五文件；核冻结executor/monitor SHA。原test identity lock按精确path/SHA校验，全局marker由其父目录推导；原test run、control、launch日志/PID/前置回执/永久锁、global marker、FINAL_COMPLETED须全部不存在。扫描/proc精确executor或monitor argv确认无active final进程，training run/control无FAILED/FAILURE、monitor completion为训练exit0。

脚本额外检查授权截止北京时间2026-09-25 08:04:49前至少剩3600秒、data盘至少2GiB/system盘至少512MiB、cgroup内存上限至少48GiB且剩20GiB、至少4 CPU，以及唯一指定GPU空闲与容量。上述是运营余量，不修改冻结scientific plan或重新选模型；实际冻结executor仍独立检查科学合同、环境、身份及test预算。任何前置失败不启动进程，仍保留本地真实请求/退出。

通过前置后，远端先exclusive建立永久lock与preflight，再以准确argv detached启动冻结monitor：test阶段、interval900、原run/control目录，child为原Python `-u` 冻结executor test+原plan/prepared/test-dir。Stdout/stderr绑定新exclusive nohup日志，stdin为DEVNULL，记录真实PID、argv和launch回执。Local记录实际SSH argv、script SHA、request、UTC、原stdout/stderr及exit，验证ack后才写LAUNCHED。Timeout、非零或非法ack均记结果未知、保留partial、不自动retry；LAUNCHED只证明进程启动记录，不代表test已发布marker或完成。

`start-watcher`先重新核训练/模型链，再使用已审marker helper的真实local chain、原路径provenance、原stdout response、exit0和local marker原字节验证。只有实际marker capture通过且实际test_launch回执存在才继续。Watcher全部run/control/receipts/inventory/log输出须缺失，永久watcher launch lock先落地，使用已验证非symlink REALPY与明确PYTHONPATH启动固定SHA watcher，记录真实cmd、PID、环境增量、marker-source SHA。该入口只负责开始本地监视与之后的归档门禁，不重新启动远端test。进程创建或回执写入异常均保留lock/log并报告未知，不重新尝试。

审计者独立运行：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 server_snapshot/full_final_test100k_20260925/test_launch_test_once.py -v
```

**4项mock测试，0.021秒，全部PASS。** 覆盖无GO/已有输出前停止、冻结interval和child argv、launch成功/非零/timeout/非法响应、partial保全、marker拒绝则不启动watcher、准确runtime与PYTHONPATH/detached参数、路径穿越与symlink。远端REMOTE_LAUNCH只进行了代码审阅和compile检查，没有模拟全部/proc/cgroup/GPU/文件系统guard，也没有真实网络或正式test执行；mock输出launched不得作为正式启动证据。

本helper不覆盖最终科学验收、finalize、fresh-root、两报告冻结、closeout、补充恢复或关机。实际启动失败/未知时，保留全部唯一marker、lock、PID和日志，由主执行者只读调查；不得换目录、删除marker或更改身份重开test。
