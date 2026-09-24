# Full 1 epoch 与固定100k验收计划模板

模板ID：`full-one-epoch-test100k-20260925-template-v1`。状态：**TEMPLATE_UNAUTHORIZED；未锁定、未启动、test仍封存**。

本文件定义可审查的最终阶段合同及完成标准，不是正式计划、开测回执或允许读取test的凭据。它承接已冻结的 `early-stop-zero-cross-20260925-v1`，不修改原raw诊断、旧epoch3实验、旧17,296人test或冻结夜间协议。正式入口应为新的独立模块；旧 `all_fresh` final入口不能通过替换人数或跳过守卫复用。

## 1. 必须先填实并封印的输入

实施者须生成不可变 `final_plan.json`，以下项目均为实际路径与SHA-256，不接受占位符、空值、未完成状态或仅有路径。记录文件字节SHA与需要的canonical hash，并明确两者含义；计划的canonical hash排除自身hash字段。保存模板本文的文件SHA。

| 锁定项 | 正式计划必须绑定的内容 | 当前状态 |
|---|---|---|
| Cross决策 | 新cross `COMPLETED.json`、`paired_confirm.json`、三个checkpoint锁定表、完整归档与源/身份/trace/传输回执；决策须重算一致 | 待cross完成 |
| Raw控制 | 原诊断完成封印、固定raw best及来源、原协议与旧对照封印、可恢复备份回执 | 引用已验收资产，待填实际来源 |
| 模型集合 | 下节确定的一个分支及其依据，不能在看test后改分支 | 待cross决策 |
| 数据与初始化 | full base `data_id`、encoders、原始数据/合法prefix统计、SVD/ItemCF/V2资产、共同初始化模板及全部相关文件SHA | 待填实际封印 |
| 全量训练行 | 原base全部4,731,777条 `train_positions` 的dtype/shape/SHA；对应 `PositionRecords` 有序hash | 待full pool与源校验 |
| Full训练pool | 完整cache sidecar及items/lengths文件SHA、logical pool hash、source/config/records绑定与校验回执 | 待构建验收 |
| 固定100k身份 | 下述身份锁、records、复核历史来源与排除集；不能重新抽样 | 已锁身份，正式计划仍须绑定 |
| 代码与环境 | 独立final executor、测试、只读legacy源码、依赖、Python/Torch/NumPy/CUDA/cuDNN与计算线程设置 | 待实现审查 |
| 资源与时间 | 各文件系统实际free/模型原子写入峰值、RAM/VRAM、pool与训练实测估计、传输/归档余量、UTC开始时间与固定deadline | 待启动前复核 |

固定test来源为 `server_snapshot/test100k_feasibility_20260924/locked_test100k/`；身份锁文件SHA为 `9416c80d3c2bda822239e19a1ffd5cc8b0497c1aca252225dee26329a9cf4303`，records文件SHA为 `086e977d4d7bb06f453bae2596dcdddb514796add4439bb5bb5ecdd18d8732fc`，封印清单SHA为 `30087cca4a9d7e3db166c8bff7d41cd3ef06226b78087a7a2a0989f5e199e188`。正式计划验证唯一encoded UID/raw UID各100,000、排序与边界映射、合法历史资格、与fast/screen/confirm及已批准本月历史排除的互斥。StageAB使用已经独立复算的19,878人确定性重建并集，保留其他来源；不重新扩大/缩小历史范围，不调用旧 `all_fresh` 选择逻辑，也不将旧17,296人与本100k当作独立test重复计数。

在最终计划锁定和后述模型准备门禁通过前，只允许test身份及合法prefix资格读取；test候选、future targets、覆盖命中、模型分数均禁止读取。这里的身份/训练覆盖是UID及训练行事实，不是候选命中率。

## 2. Cross结果只决定预先定义的分支

| 已封印开发决策 | 正式模型集合及顺序 | 100k用途 |
|---|---|---|
| `raw_retained`；zero任一开发晋级条件未通过 | `raw`一个模型 | 报告锁定raw的绝对指标，不检验结构增益 |
| `zero_cross`；全部开发晋级与完整性条件通过 | `raw`然后`zero_cross`，两个模型均必须完成 | 唯一结构比较为`zero_cross − raw` |

不因资源不足把第二分支改成仅zero却仍宣称验证了结构收益；资源不足时保持test封存。若cross失败、未完成或无有效归档，最终阶段不能启动。未晋级不等于证明zero无效；不引入第三种模型、额外seed、训练点或临时新超参数。

最终阶段只注册一个确定范围的轮次，模型数由上述分支固定。它不自动增加夜间优化轮数上限，也不允许在test之后重新开发并再次使用同一100k。

## 3. 固定训练合同与warm用户覆盖

所有最终模型采用同一full base、seed42、init424242、dim/token_dim256、hist_len50、batch256、FP32、AdamW lr=0.001、weight_decay=0.00001、clip_norm=5，以及原 `CosineAnnealingLR(T_max=3, eta_min=0.000001)` 路径。只运行第一个完整epoch；不能把scheduler的T_max改成1。

共同模板在所有用户/商品ID的原维度上构造，复用封印SVD与原随机初始化规则。初始化审计必须证明模型共享张量逐项相等，并与封印raw seed42的同维度模板一致；仅允许协议中已声明的gate差异。`zero_cross`将type offset之后的整个额外cross槽置零，不修改DIN attention内部交互。

训练行严格取原base全部4,731,777条合法prefix `train_positions`，按原行轴建立只读 `PositionRecords`；不构造重排的按用户覆盖pass。每个模型均以原 `default_rng(seed + epoch)` 的epoch0排列与原batch切分消费一次：正式配置下18,484次update、末batch129行。完整行数、原位置hash、排列hash、每batch大小、逐行 `(index, position, uid)` 及16个负例ID的消费hash、各负例来源计数、实际消费用户集合均须记录。双模型时这些trace必须逐字段相同。

负例维持原 `mixed_rrf` 16个、候选rank11–25取2个与26–50取2个、原fallback/随机混合及seed/epoch/index RNG。Full训练pool为75容量的原三路有效RRF：权重 `[2.0, 0.0, 0.7, 0.05]`，ItemCF邻居300、180天半衰期、同category/hot/tie/order/padding规则；V2权重零，不启用V2打分。保持全合法训练prefix资产与full legal user train_sets排除，不把它描述成严格逐行或全局日历时间因果。

完整epoch完成后，须验证每个计划test UID至少实际消费过1条自己的合法prefix训练目标；仅检查“在数据中存在”或预训练pool包含该UID不够。该门禁确保测试用户的ID embedding经过本次训练更新路径，不能用fast模型中未参与训练的ID来代表warm-user验收。覆盖只基于训练trace及身份边界，不能读取future targets。

最终模型固定为完整epoch结束状态，只有一个训练终点。可在该终点对原20k screen做一次已注册的描述性诊断，但不得据此挑点、换seed、延长/缩短epoch或取消另一个模型；不必保存另一个“best”。各模型保存一个唯一最终checkpoint，重读核验tensor名称/shape/有限性、模型配置、init/seed、完整trace及计划hash。full1epoch是预先声明的规模迁移选择，不能称为原fast 0.25/0.5早停规则的逐字复现，也不能据此宣称相对full3epoch有收益。

## 4. 两阶段锁定与一次test入口

第一阶段锁定 `final_plan.json` 后才允许最终训练。计划必须包含确定的模型数量、源/训练pool封印、指标和接受规则，且test候选/targets未读。Full训练pool可在cross期间提前构建，因为它只消费合法训练prefix；这不授权任何test pool构建。

第二阶段在所有注册模型训练、实际UID覆盖、checkpoint重读、完整trace、资源和归档门禁通过后，写独立不可变 `FINAL_MODELS_PREPARED.json`，绑定原计划hash、所有最终checkpoint及manifest/trace/SHA。不能修改第一阶段计划来迁就产物。模型准备完成前不建test pool；准备回执本身也不是测试成功回执。

Test入口重新验证两层锁定及所有必要源文件，确认测试目录为空且不存在任意旧 `TEST_STARTED`、`TEST_COMPLETED`、`FAILED`、候选cache、results、scores或逐用户数组。悬空symlink也视为已存在；封印证据路径不得越出根目录、包含`..`或穿过symlink。然后以排他创建的 `TEST_STARTED.json` 固定唯一会话，绑定计划、准备回执、模型hash、100k有序records和开始时间，之后才可构建test pool或读取future targets。出现故障即保留现场并停止；不自动删marker、重开有利运行或挑选已成功的局部结果。

Test pool固定为75容量、四路RRF权重 `[2.0, 1.0, 0.7, 0.05]`，使用同一封印ItemCF、V2、category/hot资产及既有排序/去重/历史排除规则。V2采用原FP32全catalog候选逻辑与128用户内部batch边界，不换approximate/top-k实现，不按future targets过滤或补入正例。记录cache sidecar、items/lengths SHA、有序records hash及logical pool hash。两模型只能使用这一个相同pool；训练三路、评估四路是声明的固定分布差异，不能临时“修正”。

一次会话依次评分所有已锁模型，candidate与target读取只能发生在这个测试阶段。目标读取守卫只允许固定100k的精确 `(uid, boundary_position)`；不允许其他UID、历史边界或训练中的泄漏读取。评分前后保持RNG、model mode与BN buffers，所有loss/grad/score/模型tensor须有限。禁止看test决定checkpoint、seed、cross结构、pool、指标、bootstrap或报告人群。

## 5. 指标、统计与成功定义

所有主指标分母是完整固定100,000用户，包括pool未命中用户。HR@5为top5至少一个future正例的用户比例；NDCG@5沿用封印实现的完整future目标数IDCG规则及分数并列按item ID升序的排序；pool_hit为75容量实际有效pool至少一个future正例的比例。保存逐用户UID/position、候选ID/有效length、score、label、hit5、ndcg5、pool_hit、AUC及auc_valid，重算与汇总一致。

候选GAUC仅在同一候选中正负皆有的用户上计算用户等权AUC，score并列计0.5，同时报告有效用户数及覆盖率；它是既有候选负例代理指标，不代表全catalog AUC。无效用户不能从HR/NDCG/pool_hit分母剔除，也不能改报“命中用户内HR”替代全用户结果。

若只有raw，成功定义为：完整执行、100k一次验收和本地可恢复归档通过，给出绝对HR/NDCG/pool_hit/候选GAUC及覆盖。未给定绝对业务达标值，因此不能自创业务门槛或宣称优于未评分基线。该分支没有zero结构收益检验，也没有full1对full3或fast早停收益检验。

若有raw与zero，唯一主要比较为同用户 `zero_hit5 − raw_hit5`，仅一个seed。10,000次用户配对bootstrap，固定seed20260925，双侧95%百分位CI，分位点0.025与0.975；同一用户的两个模型结果一起抽样，不能分别bootstrap模型。结构收益通过须同时满足平均ΔHR≥0.0005、95%CI下界>0、平均ΔNDCG≥0以及全部完整性门禁。报告ΔHR、区间、gained/lost、ΔNDCG及描述性区间，不声称三seed最终确认。零未通过时记录“最终结构收益未验收”；不能借此回到100k重新选型、训练或开测。

执行成功、结构收益验收和业务达标是不同事实。存在完整执行但结构门槛未过的合法结论；模型或数据完整性失败则不能出有效效果结论。所有结论仅涵盖当前固定cohort、资产、pool与训练合同，不将候选统计写成全catalog效果或严格机制因果。

## 6. 资源、时间与终止

夜间窗口固定为北京时间2026-09-25 00:04:49至08:04:49，即UTC `2026-09-24T16:04:49Z`至`2026-09-25T00:04:49Z`。新最终轮入口必须在deadline前，且启动前有足够估算容纳已注册全部模型、pool、一次评分、验证和本地归档。已经开始的合规注册轮可按原范围完成；不能在截止后另开轮、补新variant或滚动延长授权。

Full pool暂以4个CPU worker独立只读原builder、保持每行计算及原行顺序为候选工程实现；它不改变训练合同。正式前须完成有代表性的2048行CPU pilot、与串行原builder至少128个重叠行候选ID/顺序/length完全一致、targets调用为零及source/records/cache重读校验。单个512均匀样本或理想线性加速只能用来估算，不能证明完整耗时；并行还要计初始化、IPC、写盘与封印。若full pool实际耗时超过3小时，或剩余预算不足，不勉强启动full训练。

原16负例raw实测按行数比例估计full训练约61–62分钟/模型；它不含候选构建、终点评估、封印和传输，旧不同配置的约23分钟full记录不得作为本合同训练预算。启动前以实际pool进度、CPU竞争和磁盘写入情况更新估算记录，不改统计或模型范围。

磁盘按每个实际文件系统独立核算。full训练items/lengths净payload为1,438,460,208字节，test pool净payload为30,400,000字节，另加NPY/JSON、逐用户arrays、checkpoint约1.26GB/个、原子保存临时模型及至少1GiB余量；已存在资产不能重复扣除，但不同盘free不能相加掩盖单盘不足。只保留必要最终checkpoint，不为减少空间遗漏模型/trace，也不删除尚无本地SHA/size与可读备份的唯一原始证据。

任一源/身份/cache/init/trace/coverage/有限性/重读/空间/一次test门禁失败立即停止并保留错误与现场。可正常交付“cross完成、full/test未启动或未完成”；不得为了表面闭环跳过测试隔离或可恢复归档。

## 7. 最终交付清单

正式交付至少包括：锁定final plan及源版本、cross选择来源、初始化与全量训练消费/UID覆盖审计、每个最终checkpoint及读回回执、模型准备锁、test唯一会话marker及cache封印、完整100k逐用户数组与指标/配对统计、门槛决策、monitor退出状态，以及远端到本地完整文件清单的size/SHA、可读性与可恢复备份回执。checkpoint、manifest、模型数量、pool身份、用户顺序和统计互相绑定。

只有全部注册范围完成才写 `FINAL_COMPLETED`；结构条件是否通过作为独立字段保留。安全关机等所有必要证据本地验收完成，不因08:04:49到时跳过保存。最终报告明确区分已证实的raw早期checkpoint操作收益、cross开发选择、full规模迁移事实和100k是否实际评分。
