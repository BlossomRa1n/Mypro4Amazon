# Final计划prepare-only部署脚本审计

状态：**代码审计PASS；脚本尚未实际运行；须主执行者明确prepare-only GO且所有前置实际门禁齐备。** 本次仅只读审查本地代码并运行离线测试，没有SSH、部署、训练、评分或test未来目标访问。

| 文件 | SHA-256 |
|---|---|
| `server_snapshot/full_final_test100k_20260925/prepare_final_plan.py` | `2277d96c14b06f29d88fd610698c4d06e03ed409d62294c7a6e12a09fd463436` |
| 同目录 `test_prepare_offline.py` | `56ebadd2379a2d495e8b063560a3c897d20b97efea92054edd7d683078f1ca47` |
| 冻结executor，仅调用prepare入口 | `ae8756bde5ef024fbafb2078f615d348e4e7a71d51d13afa9a8bf50b284df862` |

## 1. 操作范围与前置绑定

脚本只接受显式 `--go-prepare-only`，并要求已有SSH Unix control socket。它读取主执行者assembler产生的authorization、copy manifest及独立 `server_snapshot/final_gates_20260925/source_mapping_verified.json`，核对formal授权状态、smoke=false、原授权/mapping的文件SHA、已审executor SHA、固定环境和legacy源码路径。

source gate的reference键集合及每项原remote path/SHA必须精确等于authorization。每个待上传copy必须匹配manifest SHA及gate对应remote ref；上传目标集合必须精确覆盖授权中全部 `authorization_inputs/` 引用，本地staging文件集合也须完全相等。它不改写原authorization或来源引用，只上传相同字节的已绑定副本，并把copy manifest、source gate和authorization一并排他写至固定远端角色。

授权的真实性、资源预算和门禁实际通过仍由主执行者负责；脚本不会把self-declared authorized当成新的用户授权。该source mapping gate必须先实际执行成功，代码审计本身不替代它。

## 2. 保全、失败和超时

首个远端调用前，所有9个本地输出位置统一检查安全路径与lexists，拒绝symlink、父目录穿越和任何已有回执/日志/计划。原稿的preflight stdout覆盖写入现已改为exclusive二进制写入。远端preflight拒绝已有authorization、输入树、plan、准备锁和训练/test目录；检查executor本体与所有祖先无symlink并匹配冻结SHA，再以新目录创建 `prepare_once.lock`。

远端上传在mkdir前检查目标及祖先无symlink，并用xb拒绝覆盖，上传后重读SHA。远端executor与下载plan都检查本体及祖先路径。计划原始字节下载后以xb保存，即使后续合同验证失败也保留原文件，不能删掉后重新prepare。

SSH配置包含BatchMode、连接与keepalive限制；普通remote preflight/upload/download调用有120秒上限，唯一prepare子进程有900秒上限。失败回执排他保存当前phase、异常类型、stdout/stderr和不自动重试标记；prepare的重定向日志在非零退出或超时后读入回执。超时显式记录 `remote_outcome_unknown=true`，因为本地SSH终止不证明远端prepare已取消。远端lock、上传partial、原始plan及本地日志均保留，主执行者须只读检查实际结果后决定恢复方式，不可盲目重跑。

## 3. Prepare与下载计划验收

唯一执行命令是冻结executor的 `prepare --authorization ... --output ...`，没有train、seal-models、test、finalize、删除、重试或关机入口。prepare源码已审：它验证原cross开发选择/已封数组、来源、合法训练池、固定test身份锁与训练prefix覆盖条件；test targets保持拒绝，不生成test候选、不统计test命中、不进行模型评分。该阶段可以按注册方式重算已有开发比较，不构成新的训练实验或开放test。

下载的plan必须通过canonical seal，authorization path/SHA/bytes必须指向本次原上传文件，refs、authorized_by、environment、legacy_source_dir及resource_review逐项等于原授权。本轮严格要求已封selection为raw_retained且模型集合为 `['raw']`，不将未知选择值回退成双模型。

计划还核对固定protocol、locked/formal状态、seed42/init424242、完整1epoch、T_max3、FP32、4,731,777训练行、100k身份、模型维度/batch/history、无test标签/pool访问，以及与冻结executor相同的metrics/acceptance。成功回执仅表示计划prepare已验证，明确training_started=false和test_started=false；后续train仍须主执行者单独满足实际资源/期限和启动门禁。

本raw-only分支最终只能报告锁定full1模型的绝对test指标。它不证明full1相对full3更优，也不提供新的结构比较；不能在test后挑选停止点、seed或cross。计划prepare与训练后模型准备锁是两个不同阶段，前者不能代替后者。

## 4. 离线测试

审计者独立运行 `/Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python -m unittest discover -s server_snapshot/full_final_test100k_20260925 -p test_prepare_offline.py -v`，5项测试0.016秒通过。测试覆盖mock完整验证→上传→prepare→下载→合同验收链，source mapping多余引用在首远端调用前拒绝、所有已有输出不覆盖、悬空symlink拒绝、timeout保留unknown状态，以及remote非零退出保留phase/stdout/stderr。测试对网络全部mock，没有SSH或真实prepare调用；实现者最终v3另报告5项0.019秒通过。

所有已发现必须修复项均已关闭。正式执行应使用上表最终版本和主执行者当前完整实际授权文件，保存真实prepare退出与计划字节；本文不先行宣称生产prepare完成。
