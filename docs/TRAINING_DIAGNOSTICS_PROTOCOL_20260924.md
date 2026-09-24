# Raw 训练退化诊断预注册协议

协议 ID：`raw-deterioration-20260924-v1`。本文件规定下一次服务器开发诊断的范围、判据与证据；不代表实验已运行或通过。执行前将本文、实际源码、环境与数据资产的 SHA 写入新协议 manifest。若实现与本文不一致，必须在第一次新训练前合拢并重新封印，不能在看到结果后修改规则。

## 1. 本轮只回答一个主问题

在相同 raw 模型、初始化、训练顺序、负例和最大三 epoch 预算下，按预定 screen 规则保留的 checkpoint，是否稳定优于固定 epoch-3 checkpoint？同时记录足以定位训练退化的损失、排序、范数及 BatchNorm 证据。

本轮只训练 `raw × seeds(42,43,44)`，不增加 zero、normalized、gate、token、attention、负例配比或正则实验。既有九组实验的 no_winner 结论与原 v3 文件保持原样；本轮是新预注册的开发复验。旧 confirm 已参与开发，不能将此次区间写成全新独立验证集上的确认。

完成本诊断后停止自动训练流程。即使达到下文门槛，也只取得为下一阶段注册停止策略的开发证据，不自动启动全量训练或最终 test。

## 2. 数据契约与封存 test

沿用 `server_snapshot/next_cross_20260924/protocol_complete/` 中已经锁定的 train、screen、confirm 记录及训练行顺序；服务器使用其已校验副本。所有正式人数必须精确满足：

| 角色 | 用户数 | 可使用内容 |
|---|---:|---|
| 快速 ranker 训练 | 100,000 | 全部合法历史前缀；604,511 条既定 prefix 训练行 |
| screen | 20,000 | 固定未来窗口与固定 75 候选，用于五个点的选模 |
| confirm | 80,000 | 全部模型锁定后统一比较 best 与 last |
| 新锁定 test | 100,000 | 仅身份、边界和资格校验；未来标签、候选与评分禁止访问 |

screen 与 confirm 不相交，二者并集正好是快速训练 100k 用户。因此这里验证的是已知用户未来推荐。新 test 与这三个开发角色均不相交；最终全量训练仍可使用 test 用户合法历史前缀，但本轮不进行此训练。

固定 full base 的 `data_id` 是 `754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158`，`data.pkl` 文件 SHA 是 `e71413ee4eead36954650c1ddb99c8334ead9edecee75ab6b9c01012efd9bb06`。SVD、V2、真实 CF300、编码器和原候选 cache 均绑定原资产 SHA，不重新拟合或静默重建。训练行数组 canonical bytes SHA 是 `b09afbf0d67a51ebd7218185204bdf848756eb4637444c12bbc71df45835387f`。

新 test 取自 `server_snapshot/test100k_feasibility_20260924/locked_test100k/`：

| 锁定对象 | SHA-256 |
|---|---|
| `test_records.json` 文件 | `086e977d4d7bb06f453bae2596dcdddb514796add4439bb5bb5ecdd18d8732fc` |
| 排序 raw-ID 集合 canonical JSON | `9790c65a3ef4ef995205a9d61d548465649d74860018b316de61fd6666161e02` |
| `test100k_manifest.json` | `9416c80d3c2bda822239e19a1ffd5cc8b0497c1aca252225dee26329a9cf4303` |
| `reviewed_source_registry.json` | `71e86224e9db7b2f3d0a2e55b70d684e75b95b443ceae525b5694ef1f326c93b` |
| `reviewed_excluded_raw_users.json` | `1f54c976e38b1ab0b347ac79ed7c0af3090fedf48cf1f7e6c6f699837175a6e2` |

新 adapter 必须明确采用固定名单契约，验证以上文件、全部 100k 身份唯一性、UID/raw-ID/边界映射、prefix 训练资格、身份互斥及审核后的排除来源。禁止截断、补人、重抽，也禁止为了通过旧 `all_fresh` 守卫而修改旧 v3 manifest 或调用其 final-test 路径。旧 17,296 证书保留为历史证据；它与新 test 有交集，不算第二份独立 test。

开发评分器仅允许固定 screen/confirm 的 `(uid, position)`；训练面板仅用合法训练行自己的已观测目标。禁止生成 test pool、统计 test pool_hit、调用 test targets，或把 test 未来项用于过滤负例。加载包含整个数据集的既有 pickle 不等于授权检查其封存后缀；运行代码必须只访问已授权记录。本轮无 final-test 命令。

## 3. 必须保留的训练设置

| 项目 | 固定值 |
|---|---|
| 模型 | `SemanticTokenDIN`，concat，raw cross；DIN 内部 attention 保留 |
| 尺寸 | embed_dim=token_dim=256，hist_len=50，head=[256,128,64]，dropout=0.1 |
| 初始化 | user 随机初始化；item 复制既定 SVD；padding/UNK 前两行置零；不是 user-SVD 均值初始化 |
| 训练 seed / 初始化 seed | 42/424242、43/424243、44/424244 |
| 精度 | FP32；不改 BF16/AMP |
| 优化器 | AdamW，lr=1e-3，weight_decay=1e-5；梯度 L2 clip=5 |
| 调度器 | CosineAnnealingLR，T_max=3，eta_min=1e-6；每完整 epoch 后 step 一次 |
| 训练目标 | `mean(softplus(negative_score-positive_score))`，同一 forward 混合正负候选供 BatchNorm |
| batch | 256；沿用末尾单样本并入前一 batch 的既定规则，不漏行 |
| 训练负例 | 每行 16 个：通常 12 random + ranks11–25 取2 + ranks26–50 取2；保留候选不足时原 fallback |
| 顺序与负采样 | 第 e 个从0开始的epoch：`default_rng(seed+e).permutation(N)`；沿用 TracedPrefixDataset 的逐行 SeedSequence 与去重规则 |

“12+4”是目标配比，rank band 被已知训练正例过滤后会触发原 candidate/random fallback；每 epoch 记录实际来源计数。不得为了让日志恰好 12+4 而改变 sampler。

训练候选原 cache 的 RRF 权重按 `[ItemCF,V2,category,hot]` 为 `[2,0,0.7,0.05]`；screen/confirm 为 `[2,1,0.7,0.05]`，budget=75、CF300、ItemCF half-life=180天。训练 3 路而评估 4 路、训练抽16负例而评估75候选，是需要解释和后续控制实验的分布差异；它们本身不是实现错误。本轮固定它们以隔离 checkpoint 选择问题。

每 epoch 保留真实消费的 index/position/uid/negative IDs digest、order SHA、batch_sizes、各负例来源计数与用户覆盖数；604,511 行和全部 100k 训练用户必须消费一次。比较旧路径时应验证相同 seed 的采样轨迹与末次模型，不能只检查配置字符串。

## 4. 评估时点与 best 选择

令 B 为按既定 batch 规则完成一个 epoch 的 optimizer update 数。正式 N=604,511 时 B=2,362。全局 step 集合为 `ceil(.25B), ceil(.5B), B, 2B, 3B`，即 591、1,181、2,362、4,724、7,086；小型 smoke 如遇重复 step，只运行一次并记录对应全部请求点。分数在相应 optimizer update 后计算。

每点对全部固定 20k screen 用户评分，同一 pool、同一目标窗口、不注入正例。按以下词典序选择每 seed 的 best：HR@5 较高优先；HR 完全相等则 NDCG@5 较高优先；二者均相等则较早 global step 优先。不对结果四舍五入后比较，不采用 AUC、训练 loss、confirm 或人工观感选模。每 seed 独立应用完全相同规则，不能从三个 seed 挑最好者。

保存 best.pth 与 epoch-3 last.pth，各自包含模型、seed、step、配置和协议 binding，并执行 atomic write、SHA、reload tensor/finite 校验。best 如果就是 last，仍明确记录二者相同模型状态；不能强制选择较早点制造收益。完成三 seed 后形成一次 `CHECKPOINTS_LOCKED.json`，确认六个 checkpoint SHA 后才开始 confirm。

本轮必须跑完三 epoch以取得完整 last 对照。screen 下降不触发自适应停止或学习率变化；也不因最优点在3而延长至4。每点评估必须 `eval()+inference_mode()`，保存/恢复 Python、NumPy、Torch CPU/CUDA RNG 和各 module mode，检查 BatchNorm 等 buffers 未改变。评估不调用 optimizer/scheduler step、不修改梯度。本轮 checkpoint 不承诺 optimizer 恢复；进程失败不自动从模型权重续训。

## 5. 固定诊断面板与指标

首次训练前生成 `panels.json` 并封印，全部 seed 和 checkpoint 重用。正式规模固定512训练行与512 screen用户。取样用一个独立 `numpy.default_rng(20260924)`，按如下调用顺序执行并保存实际结果，不能只存 seed：

1. 从既定训练行数组索引均匀无放回取512，排序索引后访问；不是512个独立用户，允许同用户有多行。独立 dataset view 的 seed=20260924、epoch=0，以原16 mixed sampler固定每行负例。正例为该训练行 item；不读取该用户未来后缀，不用未来标签过滤训练面板负例。
2. 用同一 RNG 的后续状态从固定 screen records 索引无放回取512并排序。每人正例固定为 `data.iid[position]`，即未来窗口边界第一事件；不是最大分数或最小 item-ID 正例。仅此面板使用单正例，真实候选指标仍用整个未来窗口。
3. 每 screen 面板用户从排序后的 train-active item 集合中均匀无放回取50负例，排0/1、完整已观测prefix和该用户全部已授权未来目标；按面板排序依次调用同一 RNG。正式不足50则失败，不缩小。此未来过滤仅服务已经开放的 screen 采样诊断，不用于训练或实际召回池。

每个训练评估点输出如下指标：

| 数据/指标 | 精确定义 |
|---|---|
| train panel pair-AUC/BPR | 固定单正例对16mixed负例；每行正负对均值，先对同用户行均值，再对有效用户等权均值 |
| screen panel sampled user-AUC/BPR | 固定单正例对50random负例；用户等权；BPR=`mean(logaddexp(0,neg-pos))` |
| 全screen / 全confirm HR@5 | 每人真实75候选前5与未来唯一目标集有交集为1；全部固定用户等权，包括未召回目标用户 |
| NDCG@5 | 二元相关，DCG按rank折扣；IDCG使用整个未来唯一目标数 `min(5,|targets|)`，不只用池内正例数 |
| pool_hit | 真实候选中至少有一个未来目标为1；固定池与目标时跨 checkpoint 完全相等 |
| 候选 GAUC | 仅池内同时有正例与负例用户：所有正负score对中胜出计1，原始score精确相等计0.5；先每用户AUC，再用户等权 |

真实75候选按 score 降序、精确平分按 item-ID 升序排序。AUC 的平分规则不使用 item-ID 打破平局。无正或无负用户的 AUC 记为无效并从 GAUC 分母剔除；必须同时报告有效人数与覆盖率，保存有效 mask，不能把填充0当作有效AUC。GAUC 不是全体用户表现，也不把未观测候选称为真实曝光未点击标签。

train 面板与 screen 面板使用不同负例分布和数目；两者绝对 AUC/BPR 差不能解释成纯泛化差距。主要看各自固定样本上的纵向曲线，再结合实际75候选指标。训练面板按行抽样，对活跃用户仍有抽中倾向，不能将其当成全体训练用户的无偏估计。512小面板只用于诊断，不用于新的接受检验或调参。

每点保留 train/screen 正负margin分位数，以及固定train面板正例的 user norm、item norm、`u*v` norm、完整cross token norm、独立 type-offset norm；分位数至少为min/p25/median/p75/p95/max。完整cross token包含offset，不能与原始乘积混称。全表embedding范数可另报，但全表包含大量未在快速100k ranker训练的用户，不替代面板统计。

同时记录全局参数L2、每步clip前梯度L2、评估点clip后梯度L2、每层BN running mean/variance分位数及 num_batches_tracked。这里只记录相关变化，不以范数变大或BN均值漂移单独判定机制因果。

batch invariance 固定检查 screen records 的前8人，对每人原pool第一候选分别单独评分、与原完整pool一起评分。score必须有限；每个差值必须满足 `abs(s_alone-s_pool) <= 1e-5 + 1e-5*abs(s_pool)`。失败立即终止并保留回执，不将受影响评分用于confirm。通过仅说明这些固定探针未见批次依赖，不证明所有用户、所有混合batch都通过。

保存每个 screen checkpoint、confirm best/last 的 uid、position、候选ID、有效length、原始scores、labels、hit5、ndcg5、pool_hit、AUC与有效mask，以便完全离线重算。train/screen面板的身份、正负item固定数组与逐点汇总一并封印。

## 6. 一次统一 confirm 与统计门槛

仅在三 seed 完成、best/last全部锁定且输入/数值/不变性检查通过后，统一对固定80k confirm的六个模型评分。confirm只评best和last，不评其它checkpoint，也不据confirm改变best选择。

统计前必须确认所有六份数组的 uid/position 严格等于manifest的80k记录原顺序且唯一；candidate IDs、lengths、labels、pool_hit在六份中完全一致。任何错位、缺行、池差异或非有限值导致失败，不能排序拼凑或删用户补救。

唯一主比较为 `best-raw − epoch3-raw` 的 HR@5。对用户u先计算三个seed配对差的均值 `d_u = mean_s(hit_best[s,u]-hit_last[s,u])`，估计量为80k用户的均值。以 `default_rng(20260924)` 对用户索引有放回抽80k，共10,000次，使用bootstrap均值的2.5%与97.5%分位数给95%百分位区间。不能把同一用户跨seed当成240k独立样本；此区间仅反映固定3seed下用户抽样波动，不覆盖所有训练seed的不确定性。

提前固定同时满足以下条件，才记为 `supports_early_stopping_followup=true`：

1. 三个seed各自confirm HR差值都严格大于0；
2. 跨seed平均confirm HR差值至少+0.0005，即+0.05个百分点；
3. 主HR差值95%CI下界严格大于0；
4. 跨seed平均confirm NDCG差值非负；
5. 全部数据、配对、有限值、batch invariance与归档前检查通过。

只注册这一个主比较，使用95%CI；不沿用旧两挑战者比较的97.5%CI，也不为五个screen点分别做显著性筛选。NDCG可按相同用户bootstrap给描述性区间，其非负均值是保护条件，不宣称第二项独立显著发现。每seed报告 gained/lost 用户数及差值；跨seed用户均值正负计数必须标明其是“seed平均差为正/负的人数”，不能称为某个单模型新增命中人数。

未通过任一项记为 `supports_early_stopping_followup=false` 并列出未过项，不等于证明早停永远无效。输入或运行检查失败则整个run失败，不生成有效的支持结论。三个seed的best点、固定各点曲线、GAUC、面板、norm/BN都是诊断性证据，不增加主要比较。confirm复用与screen选择的开发性质必须随结果披露。

## 7. 时间可用性审计与解释边界

执行前形成时间审计回执，明确“通过的边界”“现存近似”“尚无证据证明的事项”。本地源码已支持下述区分，但服务器实际资产仍须以来源与SHA绑定确认：

| 路径 | 目前边界 | 可以和不可以据此断言的事项 |
|---|---|---|
| 用户历史/序列/评分统计 | `history_end` 按 ts严格小于当前target排除同时间事件；hist最多50 | 应核验训练面板与screen的history边界；同时间事件不能跨进历史 |
| item popularity/average/created与归一化常量 | 基于所有用户合法训练prefix聚合 | 不包含各用户held-out suffix的设计，不等于每个训练行或全球部署时点可用 |
| SVD、ItemCF、V2 | 复用full合法prefix拟合/选定资产 | 会包含某训练行之后的训练prefix信息；不声称逐prefix因果或交叉拟合 |
| train negative exclusion | `train_sets[uid]` 为该用户完整合法训练prefix | 可排除当前训练行之后的已知训练正例，属于回顾式负例采样；不得用test后缀再过滤 |
| 原训练候选 | 当前prefix发query，召回底座仍是fullprefix拟合 | query时间严格不代表召回底座也严格；应保存来源而非更名成因果召回 |
| active catalog/编码/静态品类brand | active由合法train统计，ID词表及静态metadata有历史可用性假设 | 不证明逐时间点上市可见性；需注明静态信息假设 |
| test身份排除 | 审核后的本月已评估用户排除；StageAB确定性重建 | 身份隔离不自动证明上述特征时间因果；未保存的原RNG快照/NumPy版本限制仍保留 |

这些近似均固定为本轮现有benchmark条件，足以比较同一训练轨迹的停止策略；不能从结果直接宣称上线因果收益。发现任何实际held-out/test未来标签进入训练目标、统计拟合、候选选择、训练负例过滤，或来源证明与绑定冲突时，停止正式诊断，封存失败证据并另行修复；不能边修数据边继续本轮比较。

以下模式用于决定下一项独立实验，不在本轮临时追加：

- 固定train面板持续改善、screen面板与真实候选共同退化：支持进一步检查过拟合，但不足以唯一归因用户ID记忆。
- screen随机负例AUC改善而真实pool的GAUC/HR下降：支持分布或Top-5目标失配的调查；16与75本身不构成bug证明。
- 候选GAUC改善而HR下降：可能是大量中低位正负对改善而前5恶化，不能用AUC替代业务主指标。
- score随batch构成变化超阈值、BN buffer被评估更新：先修评估路径；当前结果不能用于正式选型。
- 所有best落在0.25：峰值可能更早，下轮可预注册更密时点；本轮不加0.1/0.125，不读更多confirm。
- 所有best落在3或未过confirm门槛：本轮没有支持更早停止的开发证据，不自动延长训练或开放test。

## 8. 停止、监控与归档

输入SHA、身份/边界、训练行完整性、候选行数/唯一性、固定参数、梯度/损失/score有限性、checkpoint reload、BN不变性、batch invariance或confirm配对任一失败，立即退出本次run并写 `FAILED.json`。不覆盖原run，不自动换seed、降人数、减候选、换精度或继续confirm。磁盘不足以原子写入和保留6模型及数组时也不得启动；中途容量不足保留已有产物并失败退出。

不存在“训练loss低于某个值即可通过”或“screen下降就重启”的规则；固定三epoch上限与预定完成后停机才是正常流程。硬件/进程故障后的重启必须用新run路径与相同封印契约，标明失败尝试；若已有confirm结果，不能把重复评分写成又一次独立确认。无optimizer状态的权重文件不能被称为精确resume。

服务器通过独立于客户端连接的脚本运行，建议训练入口由 `nohup`/独立会话托管，pid、退出码、启动命令、UTC时间和日志路径落盘。监控以900秒间隔记录进程状态、GPU、磁盘、最新step/产物和终止状态；客户端断网后仍能运行。900秒是服务器监控周期，不要求客户端阻塞等待，也不许可对外发送消息。监控只能读状态，不改训练设置或自动发起新实验。

服务器完成不等于允许关机。先将本轮新协议、源码与环境、panels、所有history/step/trace、6份checkpoint、screen/confirm逐用户数组、锁定表、统计、失败/完成回执、监控日志与SHA清单归档回本地；本地逐文件核对size/SHA，解析JSON/NPZ，并对checkpoint执行可读性与有限值检查。只有本地可读校验回执齐全后，才能按既有授权执行安全关机。网络中断或只完成服务器打包时保持服务器与资产可恢复，不能仅凭一条COMPLETED日志关机。

## 9. 开跑前与报告证据

正式运行前必须通过有意义的小型验证：AUC平分/无效分母、未来标签访问拒绝、RNG/BN不变性、best平局规则、checkpoint重读、全3seed smoke、原raw训练路径与新诊断路径相同seed的采样trace及末次tensor一致性、旧HR/NDCG口径一致性、固定100k adapter身份与封印校验。smoke产物与正式产物分目录，不能把合成人数当正式通过。

最终报告至少给出每seed五点screen曲线及best位置、best/last confirm的HR/NDCG/GAUC/覆盖、主差值区间与逐seed方向、面板纵向趋势、cross分位数/BN/batch探针、时间审计限制、实际训练/评估耗时、协议与6模型SHA、本地归档回执及 `test_future_labels_read=false` 的代码守卫证据。结论只在本轮固定资产与开发样本范围内成立。

本协议不打开新test；停止策略、训练目标、负例和最终模型规则后续全部确定并另行锁定后，才安排一次最终验收。

## 依据

- `AGENTS.md`；`docs/CROSS_DIN_ATTRIBUTION_20260924.md`；`docs/TEST100K_RECOVERY_20260924.md`。
- 既有raw路径：`code/run_cross_multiseed.py`、`code/baseline_data.py`、`code/future_window_data.py`、`code/token_models.py`。
- 新执行入口与指标：`code/run_training_diagnostics.py`、`code/diagnostic_metrics.py`；新固定名单adapter使用独立 `code/diagnostic_protocol.py` 契约。
- 原协议及固定候选：`server_snapshot/next_cross_20260924/protocol_complete/`；新身份锁定：`server_snapshot/test100k_feasibility_20260924/locked_test100k/`。
