# 夜间最小cross复验与最终验收门槛

协议ID：`early-stop-zero-cross-20260925-v1`。本文件是后续新协议，不修改 `raw-deterioration-20260924-v1` 或旧epoch3 cross协议。执行前把本文、实际源码、原始控制证据和环境SHA绑定manifest；本文件不表示实验已经执行。

## 1. 已解决什么，本轮只再回答什么

完整训练退化诊断已经达到四项开发支持门槛：raw按screen保留best比epoch3的confirm HR平均提高0.005275，95%CI=[0.004420833,0.006120833]，三个seed均正，NDCG非负。它提供避免部署退化checkpoint的操作方法；无需为了用满夜间预算强制增加学习率或正则优化。

下一轮只问：**在相同的早期选模规则下，去掉额外cross槽是否优于raw？** 只新增zero_cross，不改变DIN内部历史attention交互，不新增normalized/gated、负例、学习率、weight decay、attention或初始化因素。

夜间授权窗口固定为北京时间2026-09-25 00:04:49至08:04:49，包含此前准备时间，最多3个新优化轮。该数量是上限，不是必须完成的任务量；到08:04:49不新开轮，已经开始的轮按已注册范围完成后停止，不借新名称滚动续时。任何额外优化轮均需先形成具体新协议，不能根据confirm临时扩展本轮。

## 2. 开始前与固定资产

先完成本轮raw诊断的完整本地归档验收，并形成主体可恢复备份：当前代码/未提交改动、git状态、配置与环境、协议与日志、数据/召回/model资产路径及SHA、恢复步骤与实际读取验证。若只备份代码tar而无受影响的资产与模型恢复链，不能据此删除服务器上的唯一副本。

继续使用原full base、100k训练用户604,511条固定prefix行、20k screen、80k confirm、全部既有训练/评估候选cache，身份与资产SHA不变。新100k test仅做身份封印与互斥检查，不建pool、不统计覆盖、不评分。

raw控制来源是本轮完整归档的三个seed及五点history、screen数组和best模型；zero构造调用旧共同模板并仅把 `cross_mode` 设为 `zero_cross`。该模式将加上type offset后的整个额外cross槽置零，不能称为单独删除纯乘积的严格因果实验，也未关闭DIN attention内部乘积。

## 3. 最少新增训练：3个zero模型，各到1epoch

| 项目 | 固定合同 |
|---|---|
| 新训练 | zero_cross×seeds42/43/44，各完整1epoch；初始化424242/424243/424244 |
| raw控制 | 复用已归档raw；共同参数张量复制与原模板一致，gate白名单规则沿用旧协议 |
| optimizer与精度 | 原AdamW lr1e−3、weight_decay1e−5、FP32、clip5、batch256；不变 |
| scheduler | 保留原 `CosineAnnealingLR(T_max=3, eta_min=1e−6)` 路径；只运行首epoch，不把T_max改1，也不压缩schedule |
| 训练数据/负例 | 同一604,511行、原mixed16 sampler和fallback、seed/epoch/index RNG、顺序与末batch95；每seed2,362updates |
| 新评估点 | step591、1181、2362，即请求0.25、0.5、1epoch；不增加更早点或事后延长 |

raw控制也只在同样三个点中按HR@5、NDCG@5、较早step词典序重算best。预先核对结果必须仍为seed42 step591、seed43/44 step1181，并与已有best checkpoint binding一致；不能在zero得到结果后另挑raw点。虽然这些raw点及confirm已被开发观察过，双方应用的选择规则、预算窗口相同；结论保持开发复验性质。

每个zero的最佳checkpoint原子写入并重读校验；只强制保留best模型，epoch1末端保留history/指标/完整trace，不强制再留一份last模型。若best在epoch1，best本身就是末次模型。每个seed消费epoch1的完整trace必须逐字段等于同seed raw及旧zero的epoch1封印trace；若仅配置相同而真实order/负例流不同，本轮失败。history/数组/诊断在各点保存，RNG、BN buffer、mode与batch invariance守卫沿用本轮已验证实现。

三个zero best全部锁定后，才统一对固定confirm评分；raw使用已锁定best的confirm数组，不重新选模或用confirm筛点。不得据第一个seed的结果决定是否补后两个seed。若可复用原panel，必须绑定原实际面板数组/正负ID SHA；若实现需要重新载入生成，其实际结果必须完全相等。

## 4. 唯一主要比较与统计

主要比较只有 `early_best_zero − early_best_raw` 的HR@5。三个seed、同一80k用户先计算配对差，再按用户跨seed均值，进行10,000次用户配对bootstrap，seed20260925。固定使用**双侧97.5%百分位区间**，分位点0.0125与0.9875。

本轮只注册一个challenger；97.5%比单比较通常95%更保守，保持root已选门槛，不因只跑一个挑战者而临时放宽。它不自动创建第二个normalized/gated比较，也不声称能自动校正未来任意自适应夜间轮次。新增轮次必须重新声明比较与累计开发使用，不将复用confirm写成全新独立发现。

zero替换raw的开发支持条件同时为：三个seed各自confirm HR差严格正；均值至少+0.0005；97.5%CI下界严格正；平均NDCG差非负；所有身份、候选、初始化、消费轨迹、数值和本地归档检查通过。NDCG区间可作描述性附表，GAUC/面板/norm/BN只做机制诊断，不作为临时替代晋级指标。

未满足条件即保留raw早停方案，记录no_winner并停止本轮；不把不显著等同于无作用，也不自动追加normalized/gated或userLR。若同一个双侧区间上界<0且三seed差均负，可以说在此固定早期策略下raw有配对优势；区间跨0则不能宣称等效。

旧epoch3的zero−raw差值来自既有封印对照，可与本轮方向和幅度并列描述，回答旧结论是否依赖停止点；这不是本轮第二个接受检验。不得用两个各自显著/不显著的结论直接宣布“交互显著”，也不临时追加差中差检验后挑有利叙述。

## 5. 预算与停止规则

已有实测raw每seed一epoch约470秒。三zero训练预计约23.5分钟，九次screen+诊断约6分钟，三次confirm及统计约6–10分钟；加入测试、封印和本地归档，计划45–60分钟。它是基于上一轮的估算，不是完成时间保证；磁盘和传输可能更慢。

新模型留存预算为3个zero best及最多1个原子写入临时模型，另外包括raw控制所需的已有3个best、全部数组、日志和安全余量。先按实际tensor bytes与score array上界做disk preflight；没有本地可读备份前不删唯一checkpoint，不能为挤空间悄悄减少seed、人数、评估点或精度。

任何身份/资产SHA、候选配对、真实消费流、loss/grad/score有限性、BN不变性、batch invariance或checkpoint重读失败，停止本轮，保留证据；不得跨错误继续confirm。预算检查、备份失败、磁盘不足也可形成有依据的停止状态。通过早停开发验证加完成本轮cross归因，已足以结束优化，不要求达到夜间3轮上限。

## 6. Full训练与100k test：另锁最终计划后才执行

快速训练100k与新test100k身份不相交，不能把fast模型直接用于warm-user test验收。也不能直接将fast的0.25/0.5个epoch迁移到full随机训练流：届时会有test用户ID尚未得到梯度更新。不得通过改训练行顺序或额外覆盖pass偷偷添加未经开发验证的新训练因素。

若夜间预算足够，本协议推荐的最小最终规模迁移方案是：**提前固定seed42/init424242、完整1个full epoch、原batch/顺序/负例/FP32与T_max3调度路径，消费全部4,731,777条合法prefix训练行后取最终模型。** 训练末必须核验所有full合法行按注册顺序消费一次、所有计划test UID至少消费1条合法prefix训练目标；未通过则最终test仍封存。full这一个epoch不再按test或新的screen搜索停止点。

这个完整1epoch是为了保证warm-user覆盖而提前固定的最终训练规则，是规模迁移假设，不是声称原fast的0.25/0.5选择策略被原封复制。若不接受该迁移假设，今晚应停止在cross开发结论，后续先注册full停止策略验证，不临时篡改顺序以兼顾半epoch覆盖。

按行数比例7.827估算，完整1个full epoch约61–62分钟/模型，另需full训练候选cache、评估pool、校验与归档时间：

| Cross开发结果 | 最小最终模型集合 | 估算纯训练 | 最终test范围 |
|---|---|---:|---|
| zero未达支持门槛 | raw，seed42，full1epoch | 约62分钟 | 仅锁定raw模型的100k一次验收；不宣称结构增益 |
| zero达支持门槛，拟验证结构收益 | raw基线＋zero候选，均seed42/full1epoch | 约124分钟 | 两个模型预先锁定，在同一固定pool/100k会话一次评分后配对比较 |

上述两个分支仍须在启动前形成final-plan：固定模型配置和数量、init/训练预算、数据/身份/召回/源码SHA、候选和指标、最终接受或未通过的业务解释，尤其禁止在看到test结果后回到本100k挑checkpoint/seed/cross。若有raw+zero，最终唯一比较为zero−raw；预先固定同用户10,000次配对bootstrap、95%CI及HR均值至少+0.0005、CI下界>0、NDCG非负的验收条件。它只有一个seed，不声称三seed最终确认；未过则记录最终结构收益未验收，不再读取test推动重训。开发winner选择已经在test之前完成。

若只有raw，报告全用户HR/NDCG/pool_hit和候选GAUC/有效覆盖；当前用户没有给绝对业务达标值，不能凭单模型test成绩自创达标门槛或声称优于未评分基线。原“用户平均命中”等解释须以这次锁定指标为准。

最终test开放必须同时满足：cross开发选择已完成且不再继续test后的结构选型；本轮诊断与cross全部本地归档验证通过；主体可恢复备份通过；final-plan锁定；full模型训练及100k用户prefix覆盖通过；全量资产/pool缓存可用且SHA/空间核算通过。没有足够剩余时间容纳预计训练、pool构建、一次验收与本地归档时，不启动该最终阶段，继续封存100k test并报告未完成项。

时间截止是08:04:49，不会因为full阶段准备缓慢而重置；若最终阶段已按预算允许开跑，在截止时正在执行的已锁定轮按原范围完成，然后停止，不能以此启动新实验。安全关机始终等待本地完整size/SHA与可读校验，不能因夜间截止跳过归档。

## 7. 交付与报告边界

至少保存协议/源码/环境与旧控制绑定、三个zero初始化审计及epoch1完整trace、九个screen数组/history、三个best checkpoint及锁定SHA、三个confirm数组与复用raw的来源SHA、配对统计、门槛决策、监控和全部本地归档回执。报告分开写“早停操作方法有效”“cross在固定早期规则下的开发对照结论”“full迁移与test是否实际完成”。

只有实际执行并通过的阶段才写完成；结束时可以如实是“早停已解决部署退化、cross已完成、full/test因时间或资产预算未启动”。本轮不为追求表面完成而牺牲test隔离、用户覆盖或备份可恢复性。
