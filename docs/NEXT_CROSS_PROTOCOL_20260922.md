# 下一轮 cross 验证方案（2026-09-22）

方案由 Ultra 定义；实现交由 Medium。本文是预注册的工程与实验约定，不表示正式实验已经运行。本次授权范围是本地方案、实现、自动测试和 smoke，不启动服务器训练。

## 先看结论

- 快速训练仍只优化固定 **100,000 个用户**的训练前缀。
- **20,000 screen + 80,000 confirm**，两者互斥，合计覆盖这 100,000 个训练用户的未来窗口。两者都是开发集。
- 另预留 **100,000 fresh test** 用户；其未来标签不参与开发或选型。最终全量训练时可使用这些用户的历史前缀，最终一次验收才读取其未来标签。
- 下一轮只做 **raw / normalized_gated / zero_cross × 3 seeds = 9 组**；固定 12 随机 + 4 中等负例、BPR、3 epoch。
- 只有开发确认通过的唯一候选，才进入“全量 raw 基线 + 胜者”的一次最终测试；无胜者就保留 raw，不消耗新 test。
- 所有最终 checkpoint 必须保留并校验，不能再默认删除。

## 1. 为什么需要改造现有程序

实际代码采用 `FutureWindowData`：按用户较早约 80% 正反馈作历史，较晚约 20% 为完整未来目标集合；同时间戳不跨边界。screen/test 使用不同用户，但这些用户的历史前缀都可以参与训练。这是老用户未来推荐验证。

现有 `load_data(..., sample_users=100000)` 是先把整个数据范围缩小到 100,000 用户再编码，不只是缩小精排优化行。现有 `NegativeScreenSuite.run()` 无条件评估 screen 和 test；还默认删除 checkpoint。这两项都不满足本轮要求。

因此，新协议保留**全量 base 的用户/商品编码、训练统计和召回资产**，另存独立的 `ranker_train_positions`，仅选 100,000 用户的行进行快速精排训练。不修改 base `train_mask`、`train_sets` 或数据身份，不重编号，不因不同 seed 改变 cohort。全量表空间不代表全量精排优化；它仍会占用全量 embedding 的内存，且与上轮缩小商品空间的 100k 实验不能直接横比绝对指标或耗时。

若输入仍是旧 100k base，或者排除历史已评估用户后不足 100k fresh test，准备阶段**直接失败并输出缺口**。禁止自动缩小 test、重复用户、借用旧 test，或把 `min(requested, available)` 当作完成。

## 2. 用户与数据资产

| 集合 | 人数 | 可用历史前缀训练 | 可读未来标签 | 用途 |
|---|---:|---|---|---|
| fast-train | 100,000 | 快筛：是 | 仅 screen/confirm 路径 | 固定规模精排优化 |
| screen | 20,000 | 是，属于 fast-train | 每 epoch 可评估 | 过程诊断、提前发现异常 |
| confirm | 80,000 | 是，属于 fast-train | 九组固定 epoch-3 模型全部保存后评一次 | 开发确认和唯一候选选择 |
| fresh test | 100,000 | 快筛：不使用其训练样本；最终全量：是 | 仅最终锁定验收可读 | 最终成对验收 |

集合约束：`screen ∩ confirm = ∅`，`screen ∪ confirm = fast-train`，`fresh-test ∩ fast-train = ∅`，`fresh-test ∩ historical-evaluated = ∅`。screen/confirm 不要求从历史未看过的用户产生；它们不能被称为全新 holdout。

快筛没有来自 fresh-test 用户训练样本的梯度，但 dense embedding 配合 AdamW 时，全表 weight decay 仍会衰减未采样的行；因此不能声称这些 embedding 数值完全不变。最终模型从共同模板全量重训，并实际覆盖这些用户的历史。这不涉及读取未来标签。

固定 cohort seed `20260922`。先从合格用户中排除历史已评估用户，再抽 fresh test；再从其补集中抽 fast-train，最后固定拆成 20k screen 和 80k confirm。用户必须具备合法 future-window 边界和至少一条可优化的 prefix 样本。每个 screen/confirm 用户都必须在实际训练 batch 的优化行中出现；final test 用户也必须在最终全量训练的实际优化行中出现。

历史排除以 **raw user ID** 为唯一跨数据集标识。不同 `data_id` 的 encoded UID 不能直接做并集。来源登记包括此前全量开发/测试、token 四组实验、100k 负样本/cross 实验，以及用于选择召回配置的历史评估。优先从逐用户预测/NPZ 加对应编码表恢复；也可保守排除整个历史实验用户子集。只有 hash、没有对应 raw IDs 或可验证编码表时，不允许宣称 fresh；准备阶段应要求补齐来源，或提供完整且有来源说明的保守排除名单。

最低数据条件不只是 200k 总用户，而是：至少 100k 满足 fresh 条件的合格用户，且其补集中还至少有 100k 可训练、可评估用户。报告输出候选数、历史排除数、各交集大小和实际样本数。

最终 test 的 raw 用户身份可以在准备时锁定、计数并哈希；其未来标签不得用于采样分层、特征统计、候选矿池、提前指标或选择。现有 pickle 物理上含完整数据，代码隔离只是防误用约束，不宣称密码学封存。开发路径必须禁止调用 `targets(test_records)` 或 test 评估和候选缓存生成。

## 3. 固定条件和九组内容

三个训练 seed：`42, 43, 44`；对应初始化 seed：`424242, 424243, 424244`。cohort、开发用户、召回资产、评估候选池、训练正样本行在三个 seed 间固定；初始化、样本顺序和负例抽样可以随训练 seed 改变，但**同一 seed 内三个变体完全共享**。

固定参数：semantic_concat、随机用户初始化、SVD 商品初始化、256 维、3 epoch、batch 256、历史 50、BPR、AdamW lr `1e-3` / weight decay `1e-5`、候选 75、RRF 权重 `[2.0, 1.0, 0.7, 0.05]`、ItemCF 邻居 300、半衰期 180 天。学习率调度与现有 runner 保持一致。正式模式不允许 `max_train_steps` 截断或偷偷接受 base manifest 覆盖本轮参数。

| 变体 | cross 行为 | 解释 |
|---|---|---|
| raw | 当前原始逐元素乘积 | 基线 |
| normalized_gated | user/item L2 归一化，乘积乘 0.32，再乘 sigmoid(gate)，初值 0.05 | 复验当前优化候选 |
| zero_cross | 保持六个槽及参数形状，cross 在 token-type offset 后整槽置零，从头重训 | 是否需要 cross 的结构对照 |

0.32 延续上轮初始化尺度匹配假设；本轮不按结果调 scale 或 gate 初值。门控与归一化一起变化，因此不能单独归因“gate 是收益原因”。现有 gate 不覆盖 token-type offset；保持该定义以复验现有候选，并记录 gate、cross token norm 和 cross type-offset norm，避免把 sigmoid 数值直接解释成整个分支的贡献比例。

每 seed 先构造并完成商品 SVD/随机用户初始化的一个完整 concat 模板，随后对三个新模型显式复制**全部同形状参数和 buffer**，包括 token projections、head 首层、BatchNorm 统计、type offset。只允许 `cross_gate_logit` 按变体定义不同；zero_cross 通过配置改变，不允许改变参数形状。复制后逐 tensor 校验并保存 hash，不能只调用相同 seed。构造完成后再重置训练 RNG。

最终选模点固定为第 3 epoch；screen 用于监控，不选每个模型的最好 epoch。九组都训练完，才能一次性读 confirm；异常修复须保留旧产物及失败说明，不能悄悄只补跑有利组合。

## 4. 负样本、训练顺序和候选证据

本轮负例固定为每条正样本 **12 随机 + RRF 11–25 名中 2 个 + 26–50 名中 2 个**。各区间不足时按现有确定性规则从剩余候选、再从随机池补足，记录实际来源与 fallback 数，不能把补足后仍称为恰好四个中等负例。去重，排除 PAD/UNK、目标和该用户全部已知训练正例；不读取外层未来标签过滤负例。

训练池只为选定的 `ranker_train_positions` 建立；pool sidecar 必须绑定 base data_id、encoders_hash、position/UID 行序 hash、来源资产 hash、候选策略、shape 和文件 SHA-256。仅 shape 相同不允许复用旧池。可用全量训练前缀拟合的三路 ItemCF/category/hot 池，权重 `[2.0, 0.0, 0.7, 0.05]`；评估仍固定四路 ItemCF/V2/category/hot。两者区别写入 manifest。

每 seed/epoch 生成确定性的样本索引排列和负例，三变体复用；可预生成，也可确定性重建，但必须保存实际消费的 `(row index, position, uid, negatives)` 流的 SHA-256 及样本数，并逐变体比对。DataLoader worker 数不得改变样本语义。不得靠 manifest 文案宣称顺序/负例相同。

训练不丢样本，避免单条训练样本用户因 `drop_last` 未获更新。BatchNorm 的尾部单样本应确定性合并到前一 batch；所有变体使用相同批次边界。正式完成时校验 screen/confirm（final 时为 test）每个用户实际训练行数大于零。

screen、confirm 候选按固定 records 生成一次并缓存，九组复用。保存候选逐行 item ID 和顺序 hash、UID/position hash、来源 V2/ItemCF hash；逐用户结果保存 UID、position、hit5、ndcg5、pool_hit，不能只记录汇总 HR。所有数值必须有限，候选池命中率须各变体相同。

## 5. 预注册选择规则

主指标为 confirm 的 HR@5；NDCG@5 是不回退护栏。报告 screen 与 confirm 分开，不把两者汇总成额外的独立样本。

两个预注册挑战是 `normalized_gated - raw` 和 `zero_cross - raw`。对每个用户先求三个 seed 的配对差值均值，再按用户成对 bootstrap 10,000 次，固定分析 seed `20260922`。两个挑战使用 Bonferroni 后的双侧 **97.5% CI**（每个尾部 1.25%），同时给出每个训练 seed 的差值及三 seed 均值、标准差。这种区间反映固定三次训练下的用户抽样误差，不能声称充分覆盖所有训练随机性；不得将同一用户的三个 seed 当成三名独立用户。

候选需要同时满足：

1. screen 的三 seed 平均 HR@5 差值为正，平均 NDCG@5 不下降；
2. confirm 三个 seed 的 HR@5 差值均为正；
3. confirm 平均 HR@5 提升至少 `0.0005`（0.05 个百分点，80k 用户约 40 个净命中量级）；
4. confirm HR@5 的上述 97.5% 配对 CI 下界大于 0，平均 NDCG@5 差值不小于 0；
5. 所有控制校验、可读性和有限性审计通过。

不满足即“不足以接受”，不等同于等效或无效。若两候选都通过，选 confirm 平均 HR@5 较高者；差值在 `1e-12` 内时比较平均 NDCG@5，再按 `zero_cross` 优先作确定性平局规则。候选之间的差异另给描述性区间，不宣称胜者显著胜过另一候选。

三 seed 是初步稳定性复验；本轮不自发加 seed，不根据 confirm 继续调 gate，也不打开 test 寻找反转。

## 6. 最终 100k test 的隔离和一次验收

开发通过后先写不可变 `selection.json` 和 `final_plan.json`：锁定 raw + 唯一胜者、3 epoch、最终训练 seed 42 / init seed 424242、全量可训练 prefix 行、固定 test cohort、召回/负例配置及代码/来源 hash。最终两个模型重新从同一模板训练，不能拿快速 100k 模型充当全量 baseline。

全量训练必须覆盖 test 用户的历史前缀并通过实际优化行覆盖校验。这样测的是新评估用户的未来推荐，随机用户 embedding 不会停留在未训练状态；使用他们的历史并不等于使用其未来标签。

最终 test 只对锁定的两模型执行一次成对评估；固定 epoch，不按 test 选 checkpoint、seed 或架构。不默认自动从 dev 跳转正式测试。建立带排他写入的 `TEST_STARTED.json`：绑定 selection/plan/checkpoint/test-cohort hash；已有 marker 时拒绝重新启动评分。若中途失败保留 evidence 和 partial outputs，通过人工审计后才决定是否安全续接同一次计算，不能删 marker 后重试。对已保存的用户级数组重新计算报告不算再次模型测试。

最终验收报告 paired HR@5 的双侧 95% CI、NDCG@5、净增/丢失命中用户；接受仍要求 HR 区间下界 > 0、点估计达到预注册最小收益且 NDCG 不下降，否则保留 raw。该验收是固定最终 seed 的结果，训练随机性依据来自前面的开发多 seed，不能把一次最终训练称作最终多 seed 确证。

## 7. 本轮保留的限制

本轮不重构严格全局时间回放。用户历史严格早于其未来目标；但训练矿池和商品 dense 统计仍由全量 train-corpus 拟合，可能包含某条训练 prefix 时点之后的训练记录，故**不是严格 prefix-causal，也不是全局上线时点回放**。fresh test 的新鲜性只指未来标签未用于方案选择，不修复这种训练样本内部时点近似。

本轮与上一轮的用户/商品范围、开发规模、seed 数变化，绝对 HR 不作跨轮收益比较。最终全量与快速 100k 同样只比较各阶段内部的 raw 与候选。召回六路整合、四路训练矿池、严格 causal 统计重构、dense 压缩和多组门控消融不在本轮。

## 8. 工程接口和证据要求

建议新建独立 `code/run_cross_multiseed.py`，分 `prepare / dev / compare / lock / final-train / final-test` 子命令；不要复用会无条件执行旧 test 的 `.run()`。实际实现可拆为 cohort 工具与 runner，但以下接口契约不变：

```text
prepare --base-run FULL_BASE --run-dir RUN
        --historical-users RAW_IDS_WITH_PROVENANCE
        --train-users 100000 --screen-users 20000 --confirm-users 80000
        --test-users 100000 --cohort-seed 20260922
dev     --base-run FULL_BASE --protocol-manifest RUN/protocol_manifest.json
        --run-dir RUN/dev --seeds 42 43 44
        --variants raw normalized_gated zero_cross --epochs 3
compare --dev-run RUN/dev --output RUN/selection.json
lock    --selection RUN/selection.json --protocol-manifest RUN/protocol_manifest.json
        --output RUN/final_plan.json
final-train --base-run FULL_BASE --final-plan RUN/final_plan.json
            --run-dir RUN/final
final-test --base-run FULL_BASE --final-plan RUN/final_plan.json
           --run-dir RUN/final
```

dev 默认无 `test-users` 开关，不生成 test metrics；final 缺计划、缺历史排除来源、数据不足、checkpoint 来源不符、控制审计未通过均硬失败。smoke 使用单独 `--smoke` 身份和小人数，不可生成可接受的正式 selection/final-plan。

永久保留九组最终 epoch-3 `last.pth` checkpoint 和最终 raw/胜者 checkpoint（若正式阶段执行）；不能提供“默认删除”的路径，也不把固定末 epoch 文件误命名为 `best.pth`。每份 checkpoint 含完整模型配置、cross/ablation 配置、数据/协议/训练行/初始化身份、epoch、history 引用和 SHA-256。原子写入后立即重载校验；保存最终 gate 和范数诊断。中间恢复 checkpoint 可原子更新；最终 checkpoint 不覆写。

磁盘准备必须按实际模型 tensor 字节、9 份开发模型、2 份未来最终模型及其训练来源副本（共13份文件）、原子写入峰值、候选/负例池和日志预估。当前预检按14份 checkpoint 和候选缓存临时写入留余量，并包含最终 test 的候选行；若运行目录与协议缓存位于不同文件系统，分别检查可用空间。全量 embedding 会使 checkpoint 远大于旧 100k base；不能承诺原剩余空间足够。可按组下载并验证本地归档后迁移服务器副本，但至少有一份验证过的完整 checkpoint 长期保留，本次实现不自动清理证据。

证据目录包括 protocol/cohort manifest、历史 raw-ID 排除来源及 SHA、训练正例 positions、各 split UID/position、代码快照/hash、base/V2/SVD/ItemCF/训练池 hash、逐 tensor 初始状态审计、实际顺序/负例 batch trace hash、每 epoch history、checkpoint/最终 gate、逐用户 NPZ、候选缓存、配对分析、selection、完整日志、所有输出 SHA-256 清单。

## 9. 本地必须验证的风险

1. 小 fixture 上精确人数、互斥、历史排除、raw-ID 跨编码映射、数据不足硬失败；排除来源缺失硬失败。
2. fast-train 只优化选中用户，screen/confirm 的用户实际在优化行出现；final 的 test 历史可训练但未来记录从不进入训练。
3. dev 的 test target/evaluator 用会报错的替身测试，仍能完成九组和 confirm；final 无锁定计划或二次调用必须失败。
4. 三变体所有共同初始 tensor 实际相等，gate 差异仅白名单；zero_cross 整槽为零且重新训练。
5. 同 seed 顺序/负例 trace 相同，不同 seed 改变训练随机性但不改变 cohort；错位训练池即使 shape 相同也失败。
6. tiny CPU 端到端完成九组，checkpoint 全部存在且重载，history/NPZ/manifest 可读、指标有限；smoke 不能触发正式选择。
7. 人工构造的指标验证选择门槛、配对用户一致性、多 seed 非独立处理和无胜者时不触碰 test。

## 10. 给 Medium 的实施拆解

1. 实现独立 cohort/协议构建函数和 CLI，保存 full-base 编码下的 train/screen/confirm/test records；历史 raw ID 来源可验证，所有数量硬断言。优先新增文件，避免改变旧报告复现路径。
2. 实现只包含 selected `train_positions` 的训练视图或显式 positions dataset 参数；不改 base 训练统计、train_mask 和数据身份。采样 seed 与 cohort/data seed 分离。
3. 新 runner 构建完整 concat 模板并复制；九组固定 epoch-3；确定性 batch sampler 保留所有行，写实际 batch trace；三路训练池 sidecar 验证，screen/confirm 缓存分别共享。
4. dev 只记录 screen 每 epoch，九组完成后 confirm；保留 checkpoint，审计 source/config/hash、warm 覆盖、NaN 和所有 paired UID。
5. 独立分析/选择工具实现三个 seed 用户级配对、两个挑战 97.5% CI 和固定门槛，输出机器可读 selection。缺任意组/epoch/控制证据则拒绝选型。
6. final-plan/final 隔离通道实现 plan 哈希、全量 prefix 训练、仅 raw+唯一胜者、test 一次 marker；没有接受候选时只输出无需最终评估的结论。
7. 编写上述有意义的风险测试和 tiny smoke，记录执行结果。不得执行服务器正式训练；完成后交 Ultra 审计高风险逻辑。

## 11. 当前实现状态（本地）

- 已由 Medium 实现独立 runner：`code/run_cross_multiseed.py`，包含 `prepare`、`dev`、`compare`、`lock`、`final-train`、`final-test` 六个阶段。
- `token_models.py` 已加入 `zero_cross` 模式：保留参数形状和 cross gate，在 token-type offset 后把 cross 槽置零。
- formal candidate cache 要求 V2 checkpoint、ItemCF 和 run manifest，并调用四路 RRF；训练池权重 `[2.0, 0.0, 0.7, 0.05]`，开发/最终评估权重 `[2.0, 1.0, 0.7, 0.05]`。只有显式 smoke 才允许热门度 fixture fallback。
- dev 只为校验身份读取 test 的 UID/position/raw-ID 元数据，不读取未来标签、不构造 test 候选或生成 test metrics；九组完成后才评估 confirm。训练顺序、负例流、batch 边界、初始化 tensor hash、checkpoint hash 和 test 一次性 marker 均有证据字段。
- 最终本地 `prepare + dev smoke（3 seed × 3 variant，每组3 epoch）+ compare + smoke-only lock` 已完成；没有启动服务器正式训练。九份 checkpoint、history、候选缓存和完整 manifest 保留在 `baseline_runs/next_cross_smoke_20260922_audited/`（本地保存，不提交模型）。`selection.status=smoke_only`，`final_evaluation_allowed=false`。
- 最终本地检查：`py_compile` 与 `git diff --check` 通过；项目 **67 项测试全部通过**。其中新增 17 项，覆盖记录/来源校验、真实负例来源与训练流、九组 CPU 集成、test-label 调用拒绝、篡改/NaN/伪造选型、最终训练→测试实际小数据流程，以及重复测试和换目录绕过拒绝。
- 最终验收通道的成功路径采用 tiny fixture，仅在测试中替换已审核计划的加载入口；生产 CLI 仍拒绝 smoke 晋升。手工构造的用户级 NPZ 还验证了三 seed 正收益接受、单 seed 反向拒绝、NDCG 回退拒绝及 bootstrap 用户数为 N 而非 3N。
- Ultra 已完成高风险逻辑复审，通过本地实现范围。修复包括每 epoch 恢复训练模式、不同 seed 的实际负例抽样、独立训练行视图、实际消费 trace、GPU 自动使用、初始化复制审计、协议/代码/来源资产绑定、不可覆写结果与协议目录排他 test 标记。
- 正式实验仍需先恢复可验证的历史 raw-user 排除清单、组装正确全量资产并做服务器磁盘/内存/GPU 预检。候选 JSON 的全量 RAM 峰值和正式 GPU 性能尚未实测；本地通过不代表已获得新模型收益。

可复现检查命令（激活项目依赖环境后）：

```text
PYTHONPATH=code python -m unittest discover -s tests -v
python -m py_compile code/run_cross_multiseed.py code/token_models.py
git diff --check
```

本地验证记录：`docs/NEXT_CROSS_LOCAL_VALIDATION_20260922.json`；小数据产物及日志的 SHA-256 清单保存在上述 smoke 目录内。

**无卡预检更新（2026-09-22）：** 用户开机后恢复到了旧全量评分日志，774,585 名当前用户已在 8 月 28 日被实际评估；当前 raw 用户总数为 828,005。仅这一来源就使 fresh 上限降为 53,420，合并本次已恢复来源及保守排除后进一步降至 34,087，再按正式训练/评估资格过滤只剩最多 20,247，仍未完成其余历史来源认证。因此本协议的 fresh 100k 在现有数据上不可执行。人数/启动保护保持不变，未运行正式训练；详情及待批准的最小修订见 `docs/NEXT_CROSS_SERVER_PREFLIGHT_20260922.md`。

## 12. 历史用户排除资料与启动前置条件

2026-09-22 本地盘点结果：已归档逐用户 NPZ，但它们只有编码后的 `uid` 和 `position`，没有 raw user ID。本地仓库未找到对应 `data.pkl` 或编码表；`token_evidence_20260921.tgz` 也不含这两类文件。因此目前不能可靠地产生完整历史 raw-ID 排除名单，不能将不同 data_id 的数字 UID 直接合并后声称 test 是 fresh。

| 历史来源 | 已有资料 | 仍需恢复的映射/证据 |
|---|---|---|
| 全量 token 四组 | `server_snapshot/token_control_v2_20260920/`；screen/test 各 100k；data_id `754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158` | `/root/autodl-tmp/future_window_baseline_20260919/data.pkl` 的 `users` 编码表，须核对 data_id/encoders_hash |
| 100k 负样本、token 消融及 common-cross | `evidence/token_evidence_20260921.tgz` 与 `evidence/common_cross_screen_20260922/`；screen/test 各 20k；data_id `d68d1f3a050d1f374b759aea2559fb30ce0bd6cbfe3b690df7d611f6ee426a93` | `/root/autodl-tmp/token_negative_screen_base_20260921/data.pkl` 的 `users` 编码表；可保守排除其整个 100k 用户子集 |
| 更早基线、召回调参及 future-window 开发 | 现有汇报、来源 manifest | 对应 raw-ID 预测文件，或历史 NPZ + 同版本编码表；需逐次登记，不能只覆盖最近一次 cross 实验 |

恢复资料只需 CPU/无卡模式。操作顺序：从历史产物提取用户身份、验证其编码来源、合并 raw IDs 并登记所有来源 SHA-256；完整性审计通过后才设置 `completeness.attested=true`。哈希只能证明文件没有变化，不能自动证明历史实验清单完整，也不能证明手填的 raw IDs 确实来自该文件。

`--historical-users` 的 JSON 格式如下。此示例故意未通过完整性签认，不能作为正式输入：

```json
{
  "schema_version": 1,
  "raw_user_ids": ["实际恢复的用户ID"],
  "sources": [
    {
      "path": "相对于此JSON的来源文件路径",
      "sha256": "来源文件的实际SHA-256",
      "mode": "NPZ + 经data_id和encoders_hash验证的编码表"
    }
  ],
  "completeness": {
    "attested": false,
    "note": "逐项登记历史评估来源，未完成前保持false"
  }
}
```

数据规模也须重新检查：旧全量 base 有 828,007 个编码用户、378,913 个编码商品和 4,731,777 条训练 prefix。固定 100k 优化用户并不会缩小 embedding 表；单是两张 256 维 float32 用户/商品表就约 1.15 GiB。九个开发 checkpoint 至少约 10.4 GiB，尚未包含其他参数、候选缓存、最终两个模型和原子写入峰值。因此不能沿用旧小 base 的空间/耗时估计。服务器预检要按实际模型和缓存测量，确认磁盘与内存，再启动 GPU 训练。

旧 `future_window_baseline_20260919` 的来源 manifest 记录 `cf_neighbors=100`，不能直接把这整个目录当作本轮合格 base。需组装独立且可追溯的全量资产目录：同一 data_id/编码的 data、V2、SVD，加经验证的 300 邻居 ItemCF；来源文件与实际 shape 都须相符。不能只把 manifest 的数值改成 300 来通过检查。
