# Full1epoch独立最终入口执行前审计

状态：**代码审计PASS；正式final计划、模型准备与100k test仍分别受门禁控制**。审计范围为新独立入口及其测试，不修改冻结raw/cross协议与原三个diagnostic代码。本审计只读代码与已有证据并撰写文档，没有运行训练、连接服务器或读取test目标。

| 对象 | SHA-256 |
|---|---|
| `code/run_full_final_test100k.py` | `ae8756bde5ef024fbafb2078f615d348e4e7a71d51d13afa9a8bf50b284df862` |
| `tests/test_full_final_test100k.py` | `7e41aae077b99ff2195ee7e3d7ae7a7e2c92448dace38990aba35e0631ea79cd` |
| `docs/FULL_FINAL_PLAN_TEMPLATE_20260925.md` | `bba570fac7b3feae2e9f8723b63d0d3c03d9afd819d40fc1defe228b925859e2` |

## 1. 预先锁定的五阶段

入口为新的 `prepare → train → seal-models → test → finalize`，不调用旧 `all_fresh` 最终执行入口。prepare需要主执行者已审查的authorization及明确路径/SHA引用；模板不是授权。诊断manifest在任何用户指定源码import之前先重算canonical seal，再与固定正式 `c82aa0…` 核对，旧protocol SHA固定为 `4afac7…`；legacy Python文件名/SHA集合必须与封印完全相同。

prepare独立读取raw/zero封印数组复算cross决策：三seed HR方向、跨seed用户平均ΔHR、10,000次bootstrap的97.5%CI与NDCG保护条件一致后，唯一模型集合为raw一个，或raw后zero两个。raw完整归档/备份以及cross归档/source/transfer回执均要求绑定实际完成SHA。正式身份固定100k原锁、原records及原seal，不重新抽样。

正式 `final_plan.json` 绑定authorization、executor SHA、完整源引用、资源评审、环境、模型集合、训练行/候选/测试身份、指标和接受规则。validate_plan重算seal并核对locked状态、必要引用、授权人、legacy目录与资源配置；训练seed42/init424242、1epoch、FP32、T_max3及固定metrics/acceptance不能漂移。prepare和实际train开始都执行UTC `2026-09-25T00:04:49Z` 新轮截止检查及剩余预算门槛；已经开始的同一注册轮允许按原范围完成。

## 2. 全量训练与模型准备锁

训练使用原base全部4,731,777合法prefix行，原16负例及原行随机排列，18,484次update、最后129行，AdamW/clip5/FP32/初始化不变。全量候选复核positions、完整source与logical/file SHA，同时要求正式2048行pilot、128串并行及旧缓存重叠一致；full pool构建验证超过3小时拒绝进入正式final。

每个模型初始化state hash逐项匹配原raw42，最终完整model_config（含用户/商品/类别/品牌维度）匹配原raw，仅按分支修改cross_mode。保存完整step日志、逐行/负例消费hash、batch sizes/order hash、negative source计数、实际消费UID数组/hash和test覆盖。每个test UID必须至少实际消费一个自身合法训练目标。双模型trace完全一致；只保留一个固定full终点，没有screen选点或额外best。

`verify_training` 重查完整必要artifact集合、初始hash、实际step序列/非负有限loss与grad、固定batch/update/order、负例总数、唯一消费UID及测试覆盖、trace/模型manifest/完成回执相等。checkpoint在CPU按真实完整配置strict load并检查所有tensor有限。模型准备锁还必须绑定已验证的本地训练归档回执。

本次审查关闭了“验证一个训练目录却评分另一份prepared checkpoint”的缺口：prepared的模型键集合、checkpoint精确路径/SHA、training completion精确路径必须对应同一个已验收训练run的结果。自封修改prepared为另一文件（即使同字节副本）也不能替代。

## 3. 一次test隔离与评分

test先重新验证plan、prepared、所有模型与归档，再要求输出目录为空。全局 `FINAL_TEST100K_STARTED.json` 固定在原诊断manifest所指身份锁目录；入口要求test_lock精确等于该原路径，不能复制同字节身份锁到新目录避开一次门槛。排他写全局marker及会话marker后，才恢复被精确固定 `(uid,boundary)` 守卫限制的targets并构造test候选。失败保留现场，不删marker、不换目录重试；悬空symlink和路径越界也拒绝。

`run_cross_multiseed._load_or_make_assets` helper只消费调用方传入的固定records并生成/校验source cache，不执行旧17,296/all_fresh人群选择。`test_final`标签走 `[2,1,0.7,0.05]`，75容量；冻结 `RecallPoolBuilder` 的V2使用FP32全catalog矩阵打分、每128用户内部batch，保持ItemCF/category/hot/历史排除/排序原规则。没有近似召回、future target过滤或补正例。

真实评分调用冻结 `diagnostic_metrics.evaluate_candidates`：完整100k分母、候选有限score、固定item-ID tie规则、whole-future IDCG，保存UID/position/candidates/lengths/scores/labels/HR/NDCG/pool_hit/候选AUC及valid mask。其上下文保存恢复RNG、各module mode与BN buffers；buffer改变会失败。所有已准备模型使用同一pool，双模型稳定列逐项相等。

双模型只做zero−raw单seed10,000次配对bootstrap，seed20260925、95%CI、ΔHR≥0.0005/CI下界>0/ΔNDCG≥0。raw单模型仅报告绝对指标，结构收益字段为空，不创造业务阈值。TEST_COMPLETED与本地完整归档回执绑定后才允许finalize写最终完成标记；执行完成与结构收益成立分别记录。

## 4. 验证证据与剩余外部门槛

实现者报告此冻结版本3项测试通过，用时17.869秒。审计者阅读测试未重复训练。新增真实tiny CLI链实际生成旧诊断/raw/cross/full-pool来源，经authorization→prepare→train→seal-models→test→finalize独立subprocess完成，不mock核心load_inputs/validate_plan/cross_branch；另含真实双模型tiny训练及边界/对抗检查，覆盖重复test目录、复制身份锁、重定向checkpoint、缺少初始审计、重封错误order、非有限JSON/越界targets/symlink，以及保留固定hash字段但篡改manifest在import前拒绝。tiny fixture的归档外部门槛为模拟已验证回执，不能替代正式本地恢复验收，也不证明正式CUDA效果。

主执行者仍须先实际验收cross/source/transfer及旧证据备份、正式full pool与资源预算，再生成审核授权和计划。预算须包含大缓存反复source/SHA/logic读回、所有注册模型、V2 test pool、评分及本地传输/验证，不能只算GPU训练。训练完成后要实际验证本地checkpoint可读和trace/UID覆盖，才写模型准备锁；最终评分后要核验逐文件SHA/size、数组及模型可读性和可恢复备份，才finalize和安全关机。代码PASS本身不绕过这些外部门槛。

科学解释见 `TRAINING_COHORT_COVERAGE_REVIEW_20260925.md`：screen/confirm属于fast训练用户，但早期点覆盖尚未完成；full1改变覆盖、规模和update次数。没有full3对照时不能称其修复了full3退化，也不能把开发三seed结论写成最终三seed确认。
