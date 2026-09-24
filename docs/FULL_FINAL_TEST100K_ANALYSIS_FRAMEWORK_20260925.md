# 最终raw-only test100k结果分析框架

状态：**待最终证据门禁与主执行者放行后填写实际结果。** 本文仅基于冻结源码、协议与已封印开发报告准备结构，不读取正式test指标、预测数组或未来targets，不运行评分或新增实验。最终报告文件固定为 `docs/FULL_FINAL_TEST100K_RESULTS_20260925.md`；本文不替代该报告，也不是test、finalize或关机授权。已有auth-bound文档保持原字节。

## 1. 报告首段应回答的事实

使用可区分的三类状态：完整执行和本地验收是否通过；结构收益是否有可检验对照；是否存在预注册业务门槛。此次锁定分支为raw-only，`structure_gain_accepted=null`，`business_threshold_registered=false`；raw-only验收允许报告绝对性能，不能把null写成结构收益“通过”或“未通过”。最终执行状态只依据实际TRAINING_COMPLETED、prepared、唯一marker、TEST_COMPLETED、FINAL_COMPLETED及本地门禁填写，不用文件名、mock日志或辅助脚本PASS代替真实状态。

若任一完整性门禁不通过，明确写已完成/未完成范围并保留错误，不能发布有效效果结论。正式test没有完成时，不能使用开发confirm数值充当test100k结果。

## 2. 固定合同与模型身份

报告引用原authorization、plan原字节SHA与canonical hash，以及唯一raw checkpoint和prepared SHA。训练合同为seed42/init424242、全量4,731,777条合法prefix、完整1epoch、18,484次update、末batch129、FP32、batch256、dim/token_dim256、hist50、AdamW学习率0.001/weight decay1e−5/clip5、Cosine T_max3、mixed16。训练pool权重 `[2,0,0.7,0.05]`，test候选pool权重 `[2,1,0.7,0.05]`，容量75，实际length定义有效候选。列出实际环境及训练trace，不将冻结期望自动视为实际通过。

固定test为原封印100,000个有序 `(uid, boundary_position)`。报告必须绑定test lock/records/seal身份、全部实际训练UID覆盖、candidate pool hash、有序UID/position与唯一全局marker原路径来源。Test未来目标仅在唯一已注册test内读取；后续分析使用本地已封印观测，不重新调用targets或模型评分。

## 3. 唯一绝对指标表

| 指标 | 最终读取字段 | 分母与解释 |
|---|---|---|
| HR@5 | `raw_metrics.json: hr5` | 完整100,000用户；top5至少一个future正例。并列score按item ID升序排序。 |
| NDCG@5 | `raw_metrics.json: ndcg5` | 完整100,000用户；IDCG按完整future目标数量计算，不能改成候选内正例数量。 |
| pool_hit | `raw_metrics.json: pool_hit` | 完整100,000用户；有效候选内至少一个future正例。 |
| 候选GAUC | `raw_metrics.json: gauc` | 只在候选内同时存在正负例的有效用户上，用户等权AUC；原score并列计0.5。 |
| AUC有效人数与覆盖率 | `auc_valid_users`、`auc_coverage` | 同时给出人数/100,000与百分比，不改变前三项分母。 |

字段必须与TEST_COMPLETED中的raw metrics逐字段一致，并由正式本地语义门禁绑定raw_users.npz和candidate cache。报告可按批准的只读范围核已封印数组聚合一致性，但不新增重评分、cohort过滤、bootstrap、显著性检验、post-hoc分组或模型选择。GAUC是当前候选负例代理口径，不能表述为全catalog、曝光标签或全用户GAUC。若AUC有效人数为0，实现返回的gauc=0只是约定值，应报告无有效用户，不能解释为真实性能为0。

表中HR/pool_hit/coverage建议同时保留足够精度的比例和百分比，NDCG/GAUC以0–1值报告；若报告命中人数，必须来自通过门禁的封印hit5/pool_hit聚合或精确整数一致性核验，不能由显示后四舍五入百分比反推。

## 4. 科学结论边界

开发停止策略已有支持：封印raw early-best−epoch3-last平均ΔHR为+0.005275，95%区间[+0.004420833,+0.006120833]，三seed均正。这支持fast开发合同下保留按注册screen规则选择的early checkpoint，不证明full1最优或用户ID机制因果。

共同早期停止下的cross选择为raw_retained：zero−raw平均ΔHR +0.0000416667，97.5%区间[−0.0004500,+0.000529167]，未达到晋级规则。该结果不证明结构等价、非劣效或early raw显著更好。旧固定epoch3结果仍为zero−raw −0.0015916667，97.5%区间[−0.0023958854,−0.0007958333]；适用条件不同，不能用一个区间跨零、另一个不跨零论证交互显著。

Full迁移是冻结训练合同的实际执行事实，100k是该唯一raw模型的绝对验收事实。没有full3或zero最终对照，不能报告full1−full3收益、最终结构收益、全量优化的因果收益或未注册业务达标；不得把不同cohort、不同训练暴露或不同候选分布的绝对数值直接解释为改进幅度。完整执行、本地可恢复归档、科学效果及业务价值分别表述。

## 5. 证据台账与故障说明

最终报告或附录至少绑定下列实际来源及SHA/size，并核执行回执对应真实command/stdout/stderr/exit：原authorization与28项refs、plan与source mapping；TRAINING_COMPLETED/per-model completion/trace/init/checkpoint/consumed UID；训练run与control的完整transfer及语义门禁；prepared/seal实际回执；原路径global marker读回与session marker；TEST_COMPLETED/raw metrics/raw arrays/candidate cache；test run与control的完整transfer及语义门禁；FINAL_COMPLETED及其真实执行证据；finalize之后的fresh-root完整归档和真实transfer execution。报告只写冻结时已经完成的门禁；后续汇总closeout、恢复补充包、关机及推送结果由独立closure文档记录，避免报告与绑定它的回执互相自引用。

如实列出两个部署缺件及修复：首次server CPU fixture缺protocol文档而报错，v2补齐50项docs且原126项零删零改后通过；首次正式prepare因遗漏原identity seal退出，按已有相同SHA补充949byte原seal，证据55→56、28项auth refs完整后恢复prepare通过。保留旧失败日志、旧锁与旧archive，不能写成初次全部通过。修复只恢复已授权源证据，不能表述为重置test或重建身份。

归档验证器局限需保留：本地封印array及pool/metrics可验，但NDCG完整future IDCG无法仅从候选labels独立重建；消耗UID/order/mixed16 counts已核，negative IDs不由digest重放；source/remote authenticity由独立来源与transfer门禁补充。上述局限不能通过重新读future targets消除。

## 6. 收尾状态与交付

报告时间统一注明北京时间/UTC及具体日期；区分训练、test、监督、传输与总墙钟，不把阶段训练计时当端到端耗时。授权截止为北京时间2026-09-25 08:04:49；截止不新开轮，已注册并开始的范围按原规则完成，保存证据不因到时跳过。

最终结果报告与 `docs/OVERNIGHT_WORK_SUMMARY_20260925.md` 在关机前完成并冻结；closeout将绑定两报告及整个operations_root，因此之后不得修改已绑定报告、OPERATIONS_STATE或checklist。最终报告可写“正式执行与语义归档已完成，汇总closeout、补充恢复、关机及推送状态见后续closure”，前提是前半句有真实门禁支持。`docs/OVERNIGHT_CLOSEOUT_20260925.md` 可在关机后新建，记录真实后续状态及回执SHA，随最终commit提交；commit SHA及push结果保存到ignored的独立sibling回执，避免自引用。

恢复补充包仅证明捕获HEAD与selected source/index可恢复，原大资产/Git bundle仍是明确外部依赖。关机须在必要证据完整本地SHA/size、可读性、恢复门禁和无active任务检查之后，由主执行者单独放行。SSH断开不能单独证明已关机；closure应注明实际平台或新连接验证结果，没有证据时写未确认。关机后的closure文件及最终Git commit不在关机前补充包捕获范围内，应由后续commit/push回执说明，不把它们追写入旧恢复包。
