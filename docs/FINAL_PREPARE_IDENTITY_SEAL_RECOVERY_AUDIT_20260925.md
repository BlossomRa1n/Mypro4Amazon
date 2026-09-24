# Final prepare缺失身份seal的恢复审计

状态：**唯一身份seal补部署已实际verified；同auth、同executor的恢复prepare已exit0且实际plan独立验收PASS。** 审计者只读源码、本地身份元数据、真实补部署/恢复回执及下载plan，没有SSH、远端补文件、prepare、训练或评分。本文不替代后续训练、模型准备、test及关机门禁。

## 1. 已确定的失败边界

首次prepare实际退出码为1，退出记录UTC时间为2026-09-24T19:46:26.498364+00:00（上海2026-09-25 03:46:26.498364）。保存的traceback唯一失败点是冻结executor `prepare()` 中对全部authorization refs执行 `checked(ref)`，缺少：

`/root/training_diagnostics_20260924/evidence/server_snapshot/test100k_feasibility_20260924/locked_test100k/SHA256_MANIFEST.json`

失败发生在 `bind_legacy`、`cross_branch`、`load_inputs` 和plan写入之前；它是已声明依赖尚未部署的前置检查失败，不是训练或模型算法失败。首次 `prepare_FAILURE.json` 明确 `remote_outcome_unknown=false`，原stdout/stderr、command、exit、failure和远端 `prepare_once.lock` 全部保留。仍须恢复前只读确认没有其他活跃prepare或后来产生的plan/train/test状态，不能单凭本地failure假定远端持续不变。

当前authorization SHA为 `d14f939f9f94f6a256fd361d0b35f2a29b9204325dfeaac008d1e04a85457f5e`，executor SHA为 `ae8756bde5ef024fbafb2078f615d348e4e7a71d51d13afa9a8bf50b284df862`。授权 `test_identity_seal` 已固定原目录路径和SHA `30087cca4a9d7e3db166c8bff7d41cd3ef06226b78087a7a2a0989f5e199e188`；本地原身份seal字节具有同一SHA。因此恢复不需要重写authorization、改变引用或修改冻结executor。

## 2. 身份校验语义

冻结 `diagnostic_protocol.evidence_refs()` 只递归给定JSON对象中的显式 `path`/`sha256` 引用，校验相应文件；它不glob、枚举或对整个目录作闭集校验。`verify_lock()` 分别遍历锁、reviewed registry、独立StageAB audit及旧monthly registry，再复核身份映射、排除集、合法prefix资格和固定随机抽样。当前诊断manifest的 `test_identity_receipt.verified_evidence` 有54项，未包含 `SHA256_MANIFEST.json`。

缺失seal是额外冻结的身份包清单，内容只有4项身份文件的路径、SHA、大小和 `cohort_identity_only=true`，不含test未来目标或候选。将已封存的相同字节添加到原identity目录，不改变显式引用集合，也不改变 `verify_lock()` 返回的identity receipt；lock、records、registry、audit及所有54项既有来源必须逐字节保持不变。

Final executor独立核验seal的固定SHA，且 `bound_test_lock()` 强制原test lock路径与manifest evidence_root相同。后续全局一次性marker仍位于该原目录 `FINAL_TEST100K_STARTED.json`。本恢复不能迁移身份目录、复制锁来换marker位置、删除marker或重新生成任何身份文件。

## 3. 最小恢复门禁与操作

主执行者授权后的补部署仅限以下顺序：

1. 使用既有连接只读检查无活跃prepare、无本地/远端final plan、无已启动或完成的train/test产物、无原identity目录全局test marker。检查缺seal目标及所有祖先无symlink，目标 `lexists=false`；完整核验其余authorization refs和executor当前字节。
2. 保存远端evidence树恢复前的文件、SHA、大小清单，检查本地seal SHA与auth固定引用一致，并核对清单所列4项远端身份文件原字节。只用exclusive `xb` 将同一seal字节写到已授权的原路径；写后完成flush/fsync，并重读SHA/大小。既有父目录、锁、records、legacy src、protocol、authorization及既有日志不改写。
3. 保存恢复后evidence清单，要求相对于恢复前唯一差异是新增该固定seal文件，所有原文件SHA/大小保持不变。补部署receipt记录源/目标路径、SHA/大小、时间、命令/退出及前后清单SHA。旧本地remote_root archive/snapshot保留原状；明确远端部署树新增一个文件，不能声称旧归档已原样包含它或重写旧传输回执。
4. 补部署通过后，再只读复核全auth refs、executor、无plan及无活跃prepare、原test marker不存在和完整剩余预算。使用同一authorization、同一executor、同一 `--output .../final_plan.json` 直接调用prepare。原上传脚本已有排他锁和日志，不能整段重跑，也不删除 `prepare_once.lock`。
5. 再次prepare使用全新的 `prepare_recovery_v1` 排他启动记录/日志/退出/验收文件；保留所有首次failure证据。新启动锁、记录和输出统一首写前检查，拒绝覆盖、并发重复和symlink。失败或超时保留partial并停止，不自动重试；SSH超时不能证明远端进程已取消。
6. 若退出0，原始plan只下载一次、以xb保存，按已审prepare合同核验canonical seal、authorization path/SHA/bytes、全部refs、raw-only、固定环境/参数/metrics/acceptance、4,731,777行、100k身份和两项test未访问标志。最终receipt须清楚标注本次是缺资产恢复后的prepare成功；仍没有启动train或test。

恢复时任何额外缺失、SHA漂移、已有plan、活跃进程或test marker均必须停止，不扩展成批量补部署或绕过原门禁。授权截止、150分钟预算和最晚上海2026-09-25 05:34:49实际train启动限制保持不变。此次无模型训练和评分，不构成新增优化试验；它只恢复已登记final流程的依赖完整性。

## 4. 审计结论

在上述实际门禁全部通过且唯一补入字节为授权seal的条件下，同auth、同executor再次prepare不改变身份校验验收语义，恢复方案PASS。Medium执行者负责远端只读核验及唯一文件补部署，Low执行者在主执行者放行后负责一次prepare及本地验收；审计者不执行这些动作。下节只依据实际回执和独立读回更新已完成部分。

## 5. 实际恢复与plan验收

补部署于UTC2026-09-24 19:49:49.194963开始、19:49:50.308157结束（上海2026-09-25 03:49），returncode=0、status=`repaired_verified`。独立读回 `identity_seal_repair_execution.json` 的前后清单证明：28项授权引用中，恢复前唯一不满足项为缺失的身份seal，恢复后28项全部匹配；远端evidence文件从55个变为56个，唯一新增正是949字节的固定seal，零删除、零原文件SHA/size变化。回执记录文件及父目录fsync完成、前后无活跃prepare、plan/train/test及原全局test marker均不存在，旧 `prepare_once.lock` 保留。

审计者还独立用本地文件重算原prepare日志/脚本的全部已记录SHA，均与补部署回执一致；对本地冻结remote_root/evidence中原55文件逐一重算SHA和大小，均与恢复前清单一致。本地旧归档没有补写seal，旧failure没有被覆盖。远端新增的seal由独立补部署回执说明，不能回写声称旧snapshot本来包含它。

恢复prepare前的独立preflight于UTC19:51:22.571665完成：同authorization SHA、28项refs通过、无活跃final进程、所有计划/训练/test输出不存在，剩余15,206.428335秒，超过9,000秒。`prepare_recovery_v1/command.json` 绑定补部署回执SHA，命令仍是同一executor、同一authorization及同一final_plan输出路径，未重新运行上传流程。

恢复prepare于UTC19:58:48.568960退出0（上海03:58:48.568960）。下载plan的 `started_at_utc` 为UTC19:51:24.296478。恢复成功回执为verified，training_started/test_started均false；这两个标志描述prepare阶段，不能代替后续实际训练状态。未发生test评分，原失败仍按缺部署依赖解释，不算新增优化实验。

审计者另独立重算实际plan canonical hash和文件SHA，复核authorization path/SHA/8,457字节、refs/authorized_by/environment/legacy_source_dir/resource_review逐项相同，并通过local mapping重读全部28项引用SHA。固定raw-only、seed42/init424242、FP32、完整1epoch/T_max3、4,731,777训练行、100k身份、dim/token_dim256、hist50、batch256、75候选、完整metrics/acceptance及两项test未访问标志全部通过。没有导入executor、加载模型或读取test预测。

| 实际证据，均位于 `server_snapshot/full_final_test100k_20260925/` | SHA-256 |
|---|---|
| `identity_seal_repair_execution.json` | `f4b5319eacfd754345202662f676571e82564e284f8623981a564626e60704dd` |
| `identity_seal_repair_preflight.json` | `8f52fa23587227eec71109701a1c4a2f672aa651e807f4a37f83417a3520578d` |
| `prepare_recovery_v1/preflight.json` | `b1737d51bc39d9a7baa3c2bac9e73b61e5df91fd647cbc8445d15ac9a917110b` |
| `prepare_recovery_v1/command.json` | `8b5d2ce525f3e6fdf72ea3f72ae75856472b215fe7901d69d23be9520ba4aaf5` |
| `prepare_recovery_v1/exit.json` | `d3e03b587e4d86eecb5edea72dca2019596ec64e7306ecbe62cb3e958ca0cbfd` |
| `prepare_recovery_v1/recovery_verified.json` | `26c9076d0609fa40a5de91ced80f4b0a9c12cd760c8847329548a14ac2953d73` |
| `final_plan.json` | `ab8caa81e32142a07095b57c445d5f8afc1c3a6ee626868ed1b379547f9dd893` |

实际plan canonical hash为 `092aafb2a776c1e84d8bec2dcda9ee31312b131dfe7ced9ff2c6bf039fe8015f`。以上说明恢复prepare已完成，不先行宣称full训练、一次test或最终归档完成。原cross结果文档、final模板和其余授权引用继续保持原字节；后续最终结果应写独立文档。raw-only没有full3或zero对照，最终绝对指标不能用来宣称结构或full1相对full3的优化收益。
