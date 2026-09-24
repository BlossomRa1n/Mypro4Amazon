# 早期停止下的 raw / zero_cross 复验结果

**结论：按预注册规则保留 raw。** 三seed、固定80k confirm上，early-best zero−raw 的HR@5均值为 **+0.0000416667（+0.0041667个百分点）**，97.5%配对区间 **[−0.0004500,+0.000529167]**，未满足晋级条件。这个结果不证明两种结构等价，也不能把旧epoch3的raw优势外推为所有停止策略下cross都必需。当前更有直接操作支持的改进是停止策略。

## 1. 执行与证据状态

正式运行于北京时间2026-09-25 **02:00:20**开始，监督进程于 **02:40:22**确认完成，exit0、无失败marker，墙钟约40.06分钟。三seed zero_cross各训练1epoch，复用已封印raw前三个评估点作对照；未改冻结协议、runner或最终计划模板。本报告仅对已有开发结果作只读分析，没有训练、重新评分、重造候选或打开test100k未来标签。

本地已有完整3个checkpoint、9份screen及3份confirm数组。独立归档验证器已给出正式语义PASS，覆盖模型tensor形状/dtype/有限性、完整证据SHA、三组配对来源、初始状态/训练trace、既定选择和注册bootstrap重算。两个传输回执分别确认run的39文件共3,996,425,341字节，以及source/control树的95文件共745,209字节；两次远端inventory相等，本地文件集合与SHA/size精确匹配。

本报告作者另以 `allow_pickle=False` 读回全部12份数组，逐份核对global COMPLETED中的SHA，确认数值有限，并核对与对应raw的UID、position、candidate IDs、lengths、labels、pool_hit及AUC有效mask逐项相等；从已有逐用户值复算了三seed差值及gained/lost。没有新增bootstrap或接受检验。

**Cross来源加强门禁现已通过：** `source_cohort_verified_v2.json` 由修复后的冻结工具实际执行产生，检查12数组与原cohort/pool、6个部署来源、61个冻结依赖，以及安全输出、固定源码pin、两transfer和控制launch/completion绑定。原回执保留；新旧输出字段为兼容而相同，故SHA相同，验证器版本与实际执行证据另见 `SOURCE_TRANSFER_VERIFIERS_AUDIT_20260925.md`。这补齐cross来源门禁，但不表示full pool、final映射、模型准备或test等后续门禁已经通过，也不构成test启动授权。

## 2. 共同停止规则与screen选择

每seed的候选停止点固定为step591、1181、2362，对应请求的0.25、0.5、1epoch；第一点实际为591/2362=0.250211685epoch。各模型按screen HR@5、NDCG@5、较早step依次决胜，锁定best后才读取confirm。本次zero选出的三个停止点与raw完全一致。

| Seed | raw / zero共同best step | 实际epoch | raw best screen HR@5 | zero best screen HR@5 |
|---:|---:|---:|---:|---:|
| 42 | 591 | 0.250211685 | 5.085% | 5.050% |
| 43 | 1181 | 0.5 | 5.120% | 5.095% |
| 44 | 1181 | 0.5 | 5.025% | 5.005% |
| 三seed均值 | 各自best | — | 5.076667% | 5.050000% |

共同固定时点的screen均值如下，仅作曲线描述。best是根据同一screen选择的，不能把best的screen差当成独立确认收益。

| 请求点 | raw HR@5 | zero HR@5 |
|---|---:|---:|
| 0.25epoch | 5.036667% | 5.008333% |
| 0.5epoch | 5.025000% | 4.978333% |
| 1epoch | 4.846667% | 4.836667% |

screen pool_hit始终为12.255%，2,451/20,000用户进入候选GAUC计算；HR/NDCG仍对全20k计算。zero各seed从自己的best到1epoch，screen HR和NDCG均下降，但个别中间点有回升，不能概括为每步单调下降。本轮只新增zero至1epoch的早期观察，不重训epoch2/3。

## 3. 固定confirm的预注册比较

下表差值均为zero−raw；HR显示百分比，ΔHR显示百分点，NDCG显示0–1比例差。gained/lost指该seed同一用户从raw未命中到zero命中、以及反向变化的真实人数。

| Seed | raw HR@5 | zero HR@5 | ΔHR，百分点 | ΔNDCG@5 | gained / lost |
|---:|---:|---:|---:|---:|---:|
| 42 | 4.61625% | 4.64625% | +0.0300 | +0.0001378064 | 349 / 325 |
| 43 | 4.74500% | 4.73750% | −0.0075 | −0.0000033310 | 493 / 499 |
| 44 | 4.70000% | 4.69000% | −0.0100 | +0.0001919445 | 521 / 529 |
| 三seed均值 | **4.687083%** | **4.691250%** | **+0.0041667** | **+0.0001088066** | 不合并成240k独立用户 |

统计方法保持注册方案：先对每位用户跨三seed平均差值，再对80,000个用户进行10,000次配对bootstrap，seed20260925，97.5%区间分位点为0.0125/0.9875。HR区间换算为百分点是 **[−0.045000,+0.052917]**。NDCG均值差为+0.0001088066，描述性97.5%区间为 **[−0.0001015974,+0.0003117170]**。区间反映固定三个训练seed下的用户抽样波动，不覆盖所有可能训练seed的不确定性。

| 预注册晋级条件 | 实际结果 | 通过 |
|---|---|---|
| 三个seed的ΔHR都严格为正 | 一正两负 | 否 |
| 平均ΔHR至少0.0005，即0.05个百分点 | 0.0000416667 | 否 |
| HR的97.5%区间下界大于0 | −0.0004500 | 否 |
| 平均ΔNDCG非负 | +0.0001088066 | 是 |

因此 `selection=raw_retained`，`final_test_allowed=false`。这是未通过去cross方案晋级门槛的决定，不是raw已显著优于early-zero的结论。HR区间同时包含负值、零与正值，上界还略高于注册的0.0005收益门槛，尤其不能解释为已证明无有用差异；本轮未注册等价/非劣效界限及检验。

三seed平均逐用户HR差为正、负、零的人数为1,040、1,026、77,934；它们不是某个单一模型的gained/lost，也不是跨seed去重后一个可部署模型的净命中人数。NDCG平均差为正/负的人数为2,205/2,243，人数方向和均值方向不必相同，因为变化幅度不同。

raw与zero的confirm pool_hit都为12.1275%，AUC有效人数都为9,702/80,000；本轮差值来自固定候选内排序，不能归为召回覆盖变化。候选GAUC仅在同时存在候选正负例的这一子集上等权计算，负例仍是既定候选代理口径，不是全用户或真实曝光标签上的GAUC。

## 4. 旧epoch3的raw优势与新结果如何并存

旧协议的固定epoch3比较仍保持原封印：zero−raw平均ΔHR **−0.0015916667**，97.5%区间 **[−0.0023958854,−0.0007958333]**，三seed都为负；旧平均raw HR为4.159583%，zero为4.000417%。该旧结论支持“在固定epoch3条件下，raw优于zero”，原 `no_winner` 选择不能被本轮回写。

新结果问的是采用共同早期选择规则后的结构比较。两个方案都在相同seed、相同停止step下比较，平均HR约4.69%，结构差接近零。一个模型结构在训练后期维持相对优势，与在较早且总体更好的停止点没有被本次试验证实优势，并不矛盾。旧raw优势不能据此被否定，也不能被提升为跨停止点的普遍架构必要性。

从两个既有点估计看，结构差由−0.0015916667变成+0.0000416667；这一变化只作描述。本报告没有针对“停止点×结构”进行新增差中差/交互检验，不能把“旧区间不跨零、新区间跨零”本身当成两种效应差异显著的证据。confirm早已参与开发，也不能称本轮为全新的独立holdout验证。

本次zero清零的是完整额外cross token，包括该槽的公共type offset；DIN历史注意力内部的交互保持不变。因此比较不能解释为“DIN所有交互无效”，也不能唯一量化用户ID与商品逐元素乘积的纯信息贡献。共同初始化、训练行顺序和负例流对齐减小了无关差异，但仍未分离用户ID记忆、共享参数漂移、head优化或负例目标失配等机制。

## 5. 为什么停止策略仍是当前更强的行动依据

已封印raw诊断中，early-best−epoch3-last平均ΔHR为 **+0.005275**，95%区间 **[+0.004420833,+0.006120833]**，三seed均正，平均ΔNDCG为+0.002400666；全部注册开发支持条件通过。它直接支持在当前fast开发合同下，按注册screen规则保留早期checkpoint，避免部署后期退化checkpoint。本轮结构复验则未产生去cross晋级证据。

两轮使用不同注册区间水平，不能仅把数值大小拿来作新统计比较；这里的行动优先级来自停止策略自身已满足注册门槛、结构改动尚未满足，而非对二者另作显著性排序。没有必要为了用满最多三个新优化轮而追加未经注册的参数或架构搜索。

覆盖限制也保持原审查结论：fast train100k恰等于screen20k∪confirm80k；但早期checkpoint尚未消费所有这些UID的训练目标。共同best处，三个seed尚未消费的confirm UID分别为26,798、9,046、8,943。raw和zero使用同一顺序及停止step，因此本轮结构比较的这一暴露范围一致；它仍影响早期与晚期、fast与full之间的解释。

既有raw数组的事后分组显示：已消费组和未消费组在三个seed中均为best的HR/NDCG/候选GAUC高于last，所以退化不能全部归结为后期新纳入UID。分组并非随机处理，且历史长度和pool_hit不同；不能据此识别用户ID训练的因果效应。未消费也不表示embedding逐比特不变，AdamW衰减和共享网络仍可能变化。详见 `TRAINING_COHORT_COVERAGE_REVIEW_20260925.md`。

## 6. 成本与后续条件分支

zero三seed的纯训练记录约1,418.45秒（23.64分钟），screen加诊断记录约366.04秒（6.10分钟）；这些小计不包含全部准备、confirm、bootstrap、归档与传输成本，不能替代约40.06分钟的实际监督墙钟。

按注册分支，后续若来源v2、完整pool、可恢复归档、资源及正式计划等门禁全部通过，则进行 **raw-only、seed42、完整1epoch**：4,731,777条合法prefix、18,484次update、末batch129，init424242，batch256，dim/token_dim256，hist50，FP32，AdamW学习率0.001、weight decay1e−5、clip5，Cosine T_max保持3，mixed16及训练pool权重 `[2,0,0.7,0.05]` 不变。实际训练消费UID必须覆盖全部固定test100k的合法历史目标；通过模型准备锁后才允许一次固定test验收，其候选权重为 `[2,1,0.7,0.05]`。

full1epoch与fast0.25/0.5epoch的训练人群、样本数和更新数不同，不能宣称早期开发结果已经证明full1最优。raw-only最终验收只能报告锁定模型绝对表现；没有full3对照就不报告full1相对full3收益，也不在test上补做结构选择或重新选checkpoint。冻结最终模板仍为模板，不因本报告自动转为授权计划。

夜间授权窗口固定为北京时间2026-09-25 **00:04:49–08:04:49**，最多三个新优化轮；截止后不新开轮，已注册并开始的轮可按既定范围完成。test、关机及任何唯一远端证据删除仍分别服从原门禁，不能由本次 `raw_retained` 决定自动触发。

## 7. 主要证据定位

| 本地文件，均位于 `server_snapshot/early_stop_cross_20260925/` | SHA-256 |
|---|---|
| `run/COMPLETED.json` | `868974cd450fe300b350be419e04a4aaac8d04d615df133f4b942d9b10fe148c` |
| `run/paired_confirm.json` | `7915c49a459f95fda214d80078040dcff17728d53f6f73b13ebe766ea4fcb4ba` |
| `archive_verified.json` | `14a8eae2d7c8ca59f85571418c57a5dd405b8465a3390e453eab675e1547979b` |
| `transfer_verified.json` | `9070a14a2436f57bf4425eec1eba05c05bdc3894b3d962f15c8f04adcd23105e` |
| `control_transfer_verified.json` | `cc5ec5d6e034769df57eab67a84b053d9bf90cdd97d9a3d3b90e07790a242ffd` |
| `source_cohort_verified_v2.json`，旧同内容回执保留 | `b53e48b40ddb5d6c6a26aa8158eeec5be3977d70e3babc995f63ec4d7651ae8d` |
| `run/seed_42/zero_cross/confirm_best_users.npz` | `85886c9857ac10a45797f115c0144bf7d2c7159daf7233b0b65f8062ac524986` |
| `run/seed_43/zero_cross/confirm_best_users.npz` | `d0e6b571c35e2ab8405f83f49c43de739cc923aefb3a094d55e4cf28c5100283` |
| `run/seed_44/zero_cross/confirm_best_users.npz` | `0780637189af3107e5ed85e535ebde1069e68e0f3dbfb1e350a0b63ee906a01f` |

其余9份screen、每seed history/trace/初始审计/step日志及模型SHA均见global COMPLETED的 `evidence_hashes` 和正式归档回执。旧epoch3比较来自 `server_snapshot/next_cross_20260924/control_complete/selection_v3.json`；raw早期对last来自 `server_snapshot/training_diagnostics_20260924/run/confirm_best_last.json` 与相同目录的封印COMPLETED。所有原始统计、旧选择、冻结协议与final模板保持不变。
