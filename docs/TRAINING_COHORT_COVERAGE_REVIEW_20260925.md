# 训练用户覆盖与full迁移限制补充审查

状态：**本地身份/训练顺序只读复核完成；不改变冻结协议或选择规则；test仍封存**。这是 `FULL_FINAL_PLAN_TEMPLATE_20260925.md` 的解释补充，不改其已绑定文件SHA，不构成正式计划或测试授权。本次没有训练、评分、构造候选或读取test future targets。

## 1. Fast训练与开发人群并不互斥

独立读取原 `server_snapshot/next_cross_20260924/protocol_complete/` 三份records，在encoded UID和raw UID两种身份上得到相同结果：train有100,000人、screen有20,000人、confirm有80,000人；train∩screen=20,000，train∩confirm=80,000，screen∩confirm=0，train恰等于screen∪confirm。因此不能把本轮开发曲线解释成整个screen/confirm都属于从未训练的冷用户。冻结诊断协议与 `diagnostic_protocol.load_inputs` 已明确检查这一并集关系。

| 身份输入 | 文件SHA-256 |
|---|---|
| train records | `df66a62de582c702722e981dcd5b88ecce2ed6a68dae76c40ffe727d98876626` |
| screen records | `f57889f2ced2db20b115da75caf945bc4f80b0e08d408f68f514d33d08037312` |
| confirm records | `affba61d997af8c93229be870f8645754503853a6f23da110cc1d9b910f5eeac` |
| base data.pkl | `e71413ee4eead36954650c1ddb99c8334ead9edecee75ab6b9c01012efd9bb06` |

只反序列化既有base以访问UID数组和训练位置，没有调用数据集方法或检查未来项。原604,511位置升序且其实际UID集合正好是train100k。每seed按冻结 `default_rng(seed).permutation(604511)` 重建epoch0顺序，数组字节SHA均与封印 `epoch_1_trace.json.order_sha256` 完全一致；该epoch实际trace均记录覆盖100,000用户，2,362次update，末batch95。

## 2. 早期点仍有尚未消费训练目标的用户

所属train cohort不等于在某个早期checkpoint之前已经消费过该用户训练行。下表按原batch256和封印顺序复算，计数仅指“至少消费过一条该UID合法prefix训练目标”。

| Seed | Step | 已消费行 | 已覆盖train UID | 已覆盖screen / 20k | 尚未消费screen | 已覆盖confirm / 80k | 尚未消费confirm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 591 | 151,296 | 66,583 | 13,381 | 6,619 | 53,202 | 26,798 |
| 42 | 1,181 | 302,336 | 88,707 | 17,750 | 2,250 | 70,957 | 9,043 |
| 43 | 591 | 151,296 | 66,641 | 13,353 | 6,647 | 53,288 | 26,712 |
| 43 | 1,181 | 302,336 | 88,793 | 17,839 | 2,161 | 70,954 | 9,046 |
| 44 | 591 | 151,296 | 66,590 | 13,319 | 6,681 | 53,271 | 26,729 |
| 44 | 1,181 | 302,336 | 88,832 | 17,775 | 2,225 | 71,057 | 8,943 |
| 42/43/44，各自 | 2,362 | 604,511 | 100,000 | 20,000 | 0 | 80,000 | 0 |

原raw最佳点42→591、43/44→1,181，所以其confirm尚未消费人数分别为26,798、9,046、8,943；epoch3 last则全部已消费。这种早期覆盖差异与同用户训练次数、共享参数训练进展同时变化，是解释退化时需要保留的因素。尚未消费不能写成该embedding张量完全未改变：dense embedding参数可能受AdamW衰减等全参数优化操作影响，历史与商品表示及共享网络也在变化；本复核只证明该UID有没有自己的训练目标消费路径，不证明某个维度梯度必非零。

新cross保留相同cohort、顺序和早期点，且逐字段trace对齐，因此不因这项事后发现暂停或修改既定公平对照。但它仍是当前开发数据的操作性比较，不能由此唯一识别“用户ID记忆”机制。任何按已消费/未消费身份分组的已有数组分析只能描述，不能增加接受检验、重新选checkpoint或修改晋级门槛。

## 3. Full1epoch是不同的训练规模合同

Full1epoch消费4,731,777条合法prefix、18,484次update、末batch129；与fast0.25/0.5的151,296/302,336行、591/1,181次update不等价。除了覆盖所有固定test UID的合法历史训练目标，训练用户总量、共享参数更新次数和行分布也变了。即便超参数、共同初始化与候选算法保持固定，也不能宣称原fast最佳停止点直接证明full1最佳，或证明full3退化已被修复。

最终阶段必须在模型准备锁前通过实际消费UID覆盖门禁，使固定test每个UID至少参与一条自身合法prefix目标；只存在于base、pool或embedding表都不足以证明覆盖。这里的warm资格是训练目标消费事实，不是test候选命中，也不保证每个用户embedding的每次梯度均非零。Full仅raw分支报告绝对指标；raw+zero分支只支持该full合同下单seed结构配对比较。没有full3对照时，不报告full1相对full3收益。

## 4. 现有confirm数组的事后分组描述

主执行者随后用已封印raw best/last逐用户数组作只读分组，未重新评分、bootstrap或读取test100k。分析源码为 `tools/analyze_early_user_coverage.py`（SHA `989cfe92c507cf3ffa6173cf4631b021987610ea990b30119daab9f2a1fe33ca`），结果为 `server_snapshot/training_diagnostics_20260924/analysis/early_user_coverage.json`（SHA `ce6f453777a3cd4073eed90166857c360fb124ef9fd51c83581c9f8cc2f447bd`）。本审计阅读了源码和结果，没有重复运行该统计。源码核对原数组封印、UID/position、候选/label/valid mask配对、permutation SHA及两组加权分解等于全体差值。

下表均为best−last的比例差，不是百分点；“已消费”在每个seed的best时点定义，分组集合随seed变化。

| Seed | 已消费组ΔHR@5 | 未消费组ΔHR@5 | 已消费组ΔNDCG@5 | 未消费组ΔNDCG@5 |
|---:|---:|---:|---:|---:|
| 42 | +0.0054885 | +0.0046272 | +0.0024366 | +0.0025650 |
| 43 | +0.0054542 | +0.0029847 | +0.0023171 | +0.0021482 |
| 44 | +0.0055167 | +0.0049200 | +0.0023609 | +0.0029291 |

三个seed的两个组中，best候选GAUC也均高于last；有效人数与覆盖率详见结果JSON，GAUC仍只在各组候选内有正负样本的用户上计算。因此现有数据不支持“总体退化全部只是后续新纳入用户引起”的描述，早期已消费组也持续呈现后期退化。

这仍不是曝光的随机处理实验。未消费组的平均完整合法训练行约2.4–3.1，而已消费组约6.5–7.5；两组pool_hit也不同。不得用组间绝对HR或GAUC差解释训练用户ID的因果影响，也不能据这些事后切分建立新门槛。该描述只缩小解释范围，既定raw/cross决定和full/test合同保持不变。
