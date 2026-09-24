# 下一轮 cross 修订与实现验收（2026-09-24）

本文记录用户已批准的最小协议修订及 Ultra 对 Medium 实现的约束。它接续 `NEXT_CROSS_PROTOCOL_20260922.md`，仅覆盖本文明确修订的条目；原有模型、训练、比较、一次测试和证据隔离要求继续有效。本文是设计与验收标准，不是历史完整性认证、资源通过证明或实验结果。

**当前生效范围见第10节。** 用户随后明确将fresh排除期缩小为2026年9月1日至固定冻结时刻的评估/选型实验；第1–9节保留先前设计与审计事实，其中“全部历史”、六月缺口和20,247人数警戒均不得继续作为v3月度协议的启动约束。

用户已授权依次完成修订、恢复审计、资源检查和满足条件后的正式运行。授权不会代替 fresh 身份证据；任何无法闭合的来源、零名 fresh 或容量不足都必须在正式训练前停止。服务器连接、部署、启动与关闭由主代理统一管理；本文作者不连接服务器。

## 1. 生效修订与不变条件

| 项目 | 本轮锁定值 |
|---|---|
| 快速训练用户 | 100,000，全部从历史排除名单中的合格用户选取 |
| 开发拆分 | screen 20,000 + confirm 80,000，互斥且并集等于快速训练用户 |
| 最终 test | `test_mode=all_fresh`，完整历史审计后全部剩余合格 raw 用户，人数记为 N |
| 当前可用人数陈述 | N ≤20,247 是现有证据的上限，尚未认证，不能写为实际 N |
| 顺序 | 先完成历史审计并锁定全部 fresh，后抽快速训练用户，再开始开发 |
| 变体与 seed | raw / normalized_gated / zero_cross × 42 / 43 / 44，共九组 |
| 训练 | 3 epoch，BPR，16 负例：12 随机 + RRF 11–25 中 2 个 + 26–50 中 2 个 |
| 编码与初始化 | 全量 encoder/资产；同 seed 从共同模板显式复制全部同形参数及 buffer，仅既定 gate 白名单不同 |
| 选模与最终流程 | 原确认规则不变；固定 epoch 3；唯一胜者才与全量重训 raw 做最终一次成对验收 |

其余正式参数原样保留，包括 256 维、历史 50、batch 256、候选 75、AdamW 和学习率设置、CF300、半衰期 180 天、训练矿池三路及评估四路 RRF。不得借性能改造加入 AMP、改精度、调 gate/scale、换召回规则、减 epoch、截断训练行、变更随机数消费或降低比较阈值。

人数变少只改变最终估计的精度和适用人群，不改变接受条件。仍使用最终成对 HR@5 双侧 95% 区间、点估计至少 0.0005、NDCG@5 不回退；不得根据 N 或最终结果临时放宽。报告应明确结论只适用于锁定的剩余 fresh cohort，不能把不通过解释为等效。除 N>0 外本次不凭空增设另一个事后人数门槛；若用户另设统计精度目标，需要先修订协议，再开始开发。

## 2. 历史完整性是启动条件

当前归档是 144 个已恢复来源、25 个缺 UID 的 NPZ，早期训练评估、失败/重跑和真实数据 smoke 尚未全部闭合。原 partial registry 及 `completeness.attested=false` 原样保留，新审计产物写入新目录，禁止覆写旧证据或仅把布尔值改成 true。

完整审计至少交付以下内容：

1. 覆盖已知实验目录、日志、manifest、预测及用户级结果的来源 inventory，记录范围、检索方法、文件 SHA-256、已排除目录和仍存在的缺口。
2. 每个来源归类为“恢复了实际 raw 用户”“由可验证的整个实验用户集合保守覆盖”“证明确实未读评估标签”，或“未解决”。没有 UID 的结果数组本身不能恢复身份；只有证明其必然属于某个已排除 raw 用户全集，才可标为保守覆盖。
3. 编码映射必须绑定原 pickle SHA、data_id、encoders_hash/fingerprint 和产生结果的 manifest；不同 data_id 的整数 UID 不直接合并。给定文件的 SHA 只证明文件身份，不证明实验目录完整或手填映射正确。
4. 每项保守覆盖需要对应来源、覆盖集合的 SHA、覆盖集合如何包含该评估用户的推理和支持文件。不能因为某来源“看起来与旧 100k 类似”而消除 unresolved。
5. 完整性声明记录审计人/角色、审计时间、范围、闭合结论与所有支持文件 hash。机器可检查的 registry 必须无未解决来源；早期来源缺口也必须被证据闭合或由可验证全集保守覆盖。

正式实现将闭合证明固定为 `completeness.closure_receipt={path,sha256}`。该 JSON 必须有 `schema_version=1`、`status=complete`、`scope_complete=true`、`unresolved_sources=[]`、auditor/UTC 完成时间/scope、排序 raw 用户并集的 canonical SHA，以及 source_registry 的路径与 SHA。runner 不仅验 registry 文件 SHA，还逐项读取 source 状态；仅接受 `recovered`、`conservatively_covered`、`proven_no_evaluation`，核对每项来源文件 SHA；保守覆盖必须附带经 hash 验证的 supporting_sources。receipt 自称 complete 而 registry 仍有 unresolved、未签认状态、缺来源或文件漂移，均拒绝。合成 smoke 可以省略闭合证明，但不能进入 final。

若必要的保守覆盖包含全量当前用户，则 N=0，停止本轮正式实验。这是有效的审计结论，不得为保住 N 而缩小排除范围。若无法证明历史完整性，则完成工程、自动测试和证据归档，保留阻断原因，不新开一轮开发或测试来绕过 fresh 要求。

## 3. 动态 cohort 与 manifest 契约

### 3.1 正式准备

为此次修订使用新的 protocol version，例如 `next-cross-multiseed-20260924-v2`。旧 v1 manifest 不得按新含义悄悄继续；旧产物仍按其原协议保留。

正式 CLI 明确使用 `--test-mode all_fresh`，或者提供同义的单一正式入口。本轮实现从 CLI 移除 `--test-users`，tiny smoke 同样按其合成历史集合推导全部 fresh；不得接受任意请求人数，也不得用默认 100,000 或 `min(requested, available)` 伪装动态契约。smoke 始终不能晋升正式。

在不调用 `targets()`、不读未来目标内容的条件下，按原 `_eligible_records` 的历史可训练性和 future-window 边界资格建立 E，以已认证历史 raw ID 排除并集 H 计算：

```text
fresh F = {r in E | r.raw_uid not in H}
historical train candidates D = {r in E | r.raw_uid in H}
test = all of F; N = |F|
fast-train = cohort_seed 20260922 下从 D 无放回抽取 100,000
screen = fast-train 的前 20,000；confirm = 余下 80,000
```

先固定 F 的全部成员和确定性顺序，再抽 D。允许保持原 seeded permutation 排序方式，但随机数消费规则必须固定并被测试；不允许依赖 chunk 大小改变 cohort。N=0 或 |D|<100,000 时直接失败，报告 E、H∩E、F 和缺口。用户 raw ID、UID、position 的对应必须验证，raw/UID/评估记录不得重复；screen 与 confirm 的 raw/UID 均互斥。

主代理确认保留 20,247 这一**特定数据版本的恢复证据异常警戒**：仅在正式模式且 base data_id 为 `754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158` 时，若重算 F 超过旧证据上限，停止并要求审计来源矛盾。它不是用户设定的 test 人数上限，不会截取或抽样 F，不限制其他 base 或合成 smoke。

快速训练 `ranker_train_positions` 仍为原全量 `data.train_positions` 中属于 fast-train 的全部行，保留原顺序。它是独立视图，不覆写 base 训练 mask/sets/positions，不重编码。实际训练消费覆盖全部 screen/confirm 用户；全量 final-train 覆盖全部 test 用户历史。

### 3.2 保存并复验的字段

protocol 至少保存 `test_mode`、动态 `counts.test=N`、固定三项开发 counts、E/H∩E/F/D 的人数、历史完整性证明及 SHA、全部 cohort 的有序 records hash 与 raw-set hash、train/test 的来源规则、ranker positions hash、base data/encoder/assets hashes、code hashes、test 未读取声明。可新增统计字段，不删除现有证据。

`load_protocol` 不能只相信 manifest 自带 hash：应验证 record count/UID/raw 唯一性、raw/UID 双层集合关系、`fast-train ⊆ H`、`test ∩ H=∅`、screen∪confirm=train、N>0、正式 counts 和 mode。`validate_base` 必须用原数据资格重新算 E/F 并证实 test 正好等于全部 F，而非 F 的一个可任意挑选子集；同时复核 UID/raw/position 和历史来源链。

禁止在开发阶段生成 test 候选或调用 test targets。加载/校验 test 用户身份及窗口边界是允许的；“未读取 test records”不能替代真实的 targets 隔离测试。

动态 N 必须贯穿 `dev`、`compare`、`lock`、`_load_plan`、`final-train`、`final-test`、checkpoint metadata、资源预检、最终 NPZ、最终报告和排他 marker。使用统一正式 counts 校验，避免残留 `counts == FORMAL_COUNTS` 将 N 又限定到 100k，或相反把 100k/20k/80k 全部放宽。

final plan 显式绑定 `test_mode`、N、有序 cohort hash、raw-set hash、protocol hash、历史证明 hash。最终训练覆盖数和最终用户级数组长度必须等于 N，并核对身份数组，不接受只报告一个相同数字。`TEST_STARTED.json` 在任何最终评分/targets 调用之前排他创建，保留失败证据；换 run-dir 不得绕过同一 protocol 的 marker。

## 4. 训练候选池的分块 memmap 改造

主要目标是移除数百万训练行的 Python dictionary/list 与完整 JSON 峰值。只把最后结果转为 NPY，而前面仍构建全部 records、全部 pools 或完整 JSON 字符串，不算完成。

### 4.1 有界构建与读取

- 以已锁定的 `train_positions` 为唯一行轴，提供按需索引/切片的 records 视图；不要为 4,731,777 行先构造 `(uid, position, raw)` tuple 列表或 `.tolist()`。dev 和 final_train 共用该路径。
- 每个连续块仅计算该块候选，直接写 `open_memmap` 的 int32 `[rows,75]` 文件。块大小可作资源参数，但不得改变结果或 sampler 随机数；写出顺序严格等于 positions 顺序。
- ItemCF、全局 hot/category 索引等不随块变化的资产每次构建只加载/计算一次。不得每 2,000 行重新读约 0.9 GB ItemCF，也不得每块重复编码全量 V2 商品。训练路径 V2 权重仍为 0。
- 候选不足 75 时以 PAD=0 尾部填充，另存每行有效长度，或由只读包装层明确去除 PAD。sampler 看到的有效候选顺序必须与旧 list 完全一致，rank11–25/rank26–50 的边界绝不能被移位。空池、短池和少于 16 可用负例的失败语义不得静默变化。
- 封存后用 `mmap_mode='r'` 读取。候选文件不可被训练修改，不为 hash 把全量数组转换为 Python 对象。记录、候选语义 hash 均采用固定版本的流式编码；若继续原 JSON hash，必须逐行输出与原 canonical JSON 相同的字节。
- 可同时紧凑化 screen/confirm 缓存，但不是最低必需项；无论格式，记录顺序、候选内容、来源和两级文件封存要求相同。test 缓存仅在最终 marker 已建立后产生。

### 4.2 缓存身份与原子完成

sidecar 至少绑定：格式版本、完成状态、dtype/端序、shape/有效长度规则、budget、base data_id、encoders_hash、positions/UID 行序 hash、source asset 文件 SHA、候选策略、RRF weights、CF300、half-life、代码 hash、候选语义 hash、NPY 文件 SHA、长度文件 SHA。

实现独占 builder 或等效锁。先写临时数据，flush/fsync 并核验行数/值/校验和，再原子发布数据，最后发布完整 sidecar；未封存临时文件不允许作为缓存命中。失败保留可识别的 partial 证据；不得补一个手写 `complete=true` 让旧文件通过。数据、sidecar 任一被改、位置序列被换、同 shape 不同 source 或只剩孤立数据文件时必须拒绝复用。

dev/final 的 evidence receipt 必须封存 sidecar **及其所有数据文件**；只 hash sidecar 文本不够。final-train 也必须把 final_train 候选池 receipt 纳入最终证据，避免路径在 protocol 目录导致漏封。是否支持续接是独立工程选择；首版可拒绝部分缓存并从新目录重建，不得偷用部分结果。

### 4.3 等价要求

用冻结的旧构建逻辑作为测试 oracle，比较多个 chunk 大小（包含 1、非整除边界、大于样本总数）、短/空池、重复 UID 不同 position、相同分数 tie、seen 过滤和无有效 category。每行有效候选 ID 及顺序必须完全一致；canonical hash 在等价语义下不因 chunk 改变。

同 seed/epoch 的每行负例、fallback 来源计数、batch 边界、实际消费 `(index, position, uid, negatives)` trace 必须与旧 list 路径一致。训练 batch 不能因候选存储格式或读取 worker 变化而变化。现有全参数初始化/训练消费流/ checkpoint 再读取审计继续保留。

## 5. 有条件的评估与 category 加速

### 5.1 Category 提前停止

`run_baseline.candidate_pools(return_details=False)` 可用按现有 `category_hot` 顺序的 generator/islice 找到前 20 个非 seen 商品后停止。现有 `eligible_all[:20]` 的顺序是权威，不能先改变排序再取 20，也不能每个用户只截取全目录前 20 后才去除 seen。

`return_details=True` 仍须计算完整 `cat_full` 可达集合，不得用前 20 替代。测试须比较优化前后的 pools、cf_rankings，以及 details=True 的 channels/full_channels/seen；覆盖 quota 和 RRF、tie、前 20 大量被 seen 排除、空 category 等情况。此改造不得影响既有离线 attribution。

### 5.2 批量 evaluate

可复用旧 token runner 的“按用户编码、展平候选、按候选微批评分”结构，但保留新 runner 的严格身份/有限性审计。每块保存用户顺序和每行长度，评分后按 offsets 分回原行；模型始终 eval/inference_mode，不更新 BatchNorm，不产生 dropout。

每个用户仍用 `(-score, item_id)` 排序。空池、1–4 候选、不同长度、多目标、相同分数、最后不足一块均须保持原语义。hits/pool_hits 是 int8，ndcg 是 float32；指标计算公式和平均方式保持不变。NaN/Inf、非法/重复候选在写入结果前拒绝。

最低对照为旧逐用户 evaluator 与新路径在三个变体、固定随机权重及训练后权重上的比较：逐用户 top5/UID/position/hit5/pool_hit 完全一致，ndcg5 数组完全一致，汇总一致。浮点 scores 可以使用预先固定的 FP32 容差（建议 atol=1e-6、rtol=1e-5），但不得用分数容差掩盖排名变化；tie fixture 必须确定性保持 item_id 次序。

CPU 测试及服务器 GPU 上历史开发用户的固定 slice 都要验证。真实 GPU 验证只用历史已排除用户，不触及 fresh test 的候选或标签。若 GPU batching 造成 top5 不等价，保持逐用户评分或进一步修复；不要声称“结果大致相同”就直接用于正式实验。最终采用的 evaluator 参数与代码在 prepare 前锁定。

## 6. 最低自动测试与审计证据

| 范围 | 必须覆盖的行为 |
|---|---|
| all_fresh | 取尽全部 eligible\historical；N 可为 1 或不同非 100k 值；N=0 拒绝；历史 train 不足拒绝；固定开发 counts/seed 不放宽 |
| 污染与认证 | attested=false、缺来源、source hash 漂移、unresolved 未闭合拒绝；raw/UID 重复、假映射、train 非历史、test 被摘成子集拒绝 |
| manifest | 旧协议拒绝；重算 manifest hash 也不能绕过错误 counts/mode/成员；N 在 plan/coverage/NPZ/marker 不一致拒绝 |
| memmap | 多块大小与旧 oracle 全行相同；有效长度/PAD正确；same shape不同数据/行序/资产拒绝；文件篡改、partial和缺sidecar拒绝 |
| sampler | 三变体实际 trace 一致；新旧存储负例及 fallback 一致；尾部单样本合并仍覆盖全部用户 |
| evaluator | 三变体新旧逐用户数组和top5一致；空/短/ragged/tie/非有限值；CPU及受控GPU验证 |
| category | return_details=False/True 的原合同均保持，完整可达集合未截断 |
| 阶段隔离 | 全九组 CPU smoke 的 test-target spy 零调用；无 test cache；confirm 只在九 checkpoint 后；smoke/no-winner拒绝final |
| 一次验收 | 动态N下全量训练历史覆盖；fast checkpoint拒绝；global marker排他；失败/复制run-dir不允许再次评分 |

运行现有完整单元测试、针对修改的回归测试、Python 编译和 diff whitespace 检查。输出实际测试命令、环境、数量、耗时和失败记录；不复用旧“74通过”数字作为本次结果。九组 smoke 仍是合成 tiny fixture，不是一次新真实数据选型实验。

Ultra 复审应逐项给出 PASS/BLOCKED/FAIL 和证据路径：动态人数、历史证明、候选行等价、采样 trace、批量评分等价、阶段守卫、磁盘/RAM/GPU 预检。只有全部必要项通过才通知主代理允许正式 prepare/dev；代码能跑与实验获准是两个独立结论。

## 7. 正式启动、监控与收尾边界

按顺序执行：历史闭合 → 工程与本地测试 → Ultra 代码/证据复审 → 组装真实 CF300 全量 base → 服务器 GPU/CPU/RAM/磁盘测量及历史 slice 等价检查 → 冻结代码/来源/全部 fresh → 正式九组 → 原规则比较 → 有唯一胜者才锁 plan/全量成对训练/一次 final test → 归档校验与服务器关闭。

可以并行恢复审计与工程，不能越过前置条件开始真实开发训练。正式 code hash 一旦封存不热改；修复保留失败 run 和说明，并重新审计后使用新目录。确认或 test 已读取后的修复必须显式讨论其对一次性/选择性的影响，不能只补有利组。

资源预检保留至少 13 份 checkpoint、14 份原子写入模型峰值估算，另加紧凑候选、长度/sidecar、日志/证据和实际余量。候选净磁盘字节改按 int32 与真实 N 计算，不沿用 JSON 的 32 字节/ID 估计；同时记录真实 RSS/GPU 峰值，NPY 变小不代表巨型模型、AdamW、原始 data 和 CF 已适配 RAM/VRAM。不能自动删旧证据腾空间。

Low 只负责已审计命令的启动与 15 分钟间隔监控。监控记录当前阶段、进程、已完成行/step/epoch、吞吐、剩余空间和资源异常；不改变参数、不跳过护栏、不擅自删除 marker/重跑。状态无变化时保持安静，失败、完成或需要处理时报告主代理。若 preflight 或历史审计阻断，保存阻断报告、归档已完成工程与证据，不启动新实验。最终关闭服务器仍遵循用户的收尾授权。

## 8. Ultra 工程复审（2026-09-24）

**结论：工程实现有条件 PASS；正式实验仍受历史认证、真实资产和服务器资源前置条件约束。本审计不授权绕过这些条件。** 审阅范围为 Medium 的 v2 runner、`cross_pool_cache.py` 和本轮回归测试；Ultra 仅修改本文，未连接服务器、未启动真实数据训练/评估。

最终冻结证据：`evidence/cross_v2_frozen_validation_20260924_v2/`，包括 `validation_receipt.json`、`unittest.log`、`smoke_summary.json` 及合成 protocol/dev 产物。旧 `cross_v2_local_validation_20260924/` 和无 `_v2` 的冻结目录保留原样；它们对应此前代码，不得作为最终续跑依据。

| 审计项 | 状态 | 已验证内容或剩余条件 |
|---|---|---|
| all_fresh 与开发 cohort | PASS | 固定 100k/20k/80k 与 seed；仅历史用户训练；全部 fresh 重算；UID/raw/position、raw hash、确定性抽样和 N 全链路核验；旧协议及可篡改子集拒绝 |
| 历史完整性代码守卫 | PASS | 正式 closure receipt 和真实 registry 双层验证；虚报 complete、真实 unresolved、缺证据或 SHA 漂移拒绝 |
| 真实 fresh 身份认证 | BLOCKED | 本工程审计未收到可正式签认的完整历史证明；N 尚未锁定，20,247 仍只作证据上限 |
| 训练候选缓存 | PASS | 按需 PositionRecords、固定 1,024 行块、只读 int32 memmap/有效长度、流式 canonical hash、source/row/dtype/shape/ID/PAD校验、独占构建和 fsync；sidecar 与 payload 双封存 |
| 候选与采样等价 | PASS（本地） | 三路及四路 CPU synthetic oracle 逐行一致；1031 行跨1024边界并含7行尾块；多个块大小的训练采样、fallback和hash等价 |
| 模型评估与 attribution | PASS（范围保持） | 本轮保留原逐用户 evaluator，没有采用批量评分；category提前停止仅位于新固定RRF builder，原 `candidate_pools(return_details=True)` 完整集合路径未修改 |
| 训练证据与阶段隔离 | PASS | 九组初始化和3 epoch训练流一致；尾部合并保留；真实batch_sizes证据；test-target spy无访问，无test候选，smoke/no-winner不能final，排他marker不能换目录绕过 |
| 动态 N final 链 | PASS（合成） | 现有全文件集成测试通过；Ultra额外对N=1/N=3执行合成final_train/final_test，仅替换授权入口，实际coverage/paired数组长度等于N，marker complete |
| 资源预算 | PASS（工程）、BLOCKED（正式） | 二进制缓存按真实行数/动态N预算；dev新增10模型峰值与final新增5模型峰值逐阶段检查，记录全生命周期14峰值；主代理仍需验证服务器全流程容量分布、RSS/VRAM和真实资产 |
| 正式 GPU/真实历史 slice | BLOCKED | 未由本地CPU测试替代；真实环境候选等价/吞吐与峰值仍须在历史排除用户上完成，不触及fresh test |

本轮复审发现并修复了实际问题：分批 iterator 被训练循环耗尽后，尾部再次遍历导致 `batch_sizes=[]`；现改为训练时记录实际批次大小，回归测试验证总行数、step数和合并边界。还补齐了 cache builder 排他写入、数据 fsync、缓存来源与实际 payload封存、必需的final cache receipt、marker前动态身份校验，以及closure与registry内容不一致的拒绝路径。

最终冻结 runner SHA-256：`1c7ad8d9960de223ff9cb479df4ade63cb448e27f6c914fe467a5da5f7c8ef2f`；cache SHA-256：`e328ad066104df75b496450cf169e6ea47846c7c8d8194e38959b7652c914505`。Medium 最终封存记录为 **89 tests / 4.326秒全部通过**；独立九组CPU smoke三轮共 **1.730秒**，`test_accessed=false`、无test cache、`selection_status=smoke_only`、`final_evaluation_allowed=false`；Python编译及diff检查通过。Ultra独立复跑最终版本为 **89 tests / 4.288秒全部通过**，并核对 receipt 中六个代码/测试文件、unittest log、smoke summary 的 SHA 全部匹配；直接调用 `_validate_dev_evidence` 对封存的九组开发产物复验通过，protocol 为v2、counts为合成4/2/2/2、test未访问且final禁用。

复现环境为 `/private/tmp/token-control-local-env/bin/python`；测试命令：`PYTHONPATH=code /private/tmp/token-control-local-env/bin/python -m unittest discover -s tests -q`。所有新增运行均为合成CPU fixture，未产生新的真实模型收益结论。只有主代理补齐表中正式BLOCKED项后，才可进入正式prepare/dev；若历史无法认证或F为空，按第2节结束为审计阻断，完成归档与收尾。

## 9. 真实 GPU 预检证据追加复审（2026-09-24）

本节记录第8节本地工程审计之后收到的服务器历史用户预检，不修改先前审计时点的事实。**结论：限定历史 slice 的候选等价、初始化复制及两步 FP32 BPR probe 通过；正式磁盘容量 BLOCKED，正式全过程 RAM/VRAM 峰值及历史闭合仍未认证。** Ultra仅在本地核查下载证据，未连接服务器、未重新运行真实训练；冻结runner/cache代码未改变，不重复完整单元测试。

证据目录：`server_snapshot/next_cross_20260924/gpu_preflight/`。`final_seal.json` SHA-256为 `0c6ef197c511a35b899ec89575256546dabf8294f1a3676e3f77c19276e50070`，其中20个文件逐项校验通过，包括probe脚本、日志、结果、资源说明、初始化审计、身份记录和三组候选NPY/sidecar。封存结果中的runner/cache哈希与第8节冻结版本一致。Ultra还调用真实`load_cache`复验128行、129行和512行缓存：文件与语义hash、dtype、shape、有效长度、ID/PAD和只读memmap均通过。

历史身份仅使用已恢复partial名单内的确定成员：原`dev_selection`中首129名历史用户，以及原`train_positions`中首512条历史用户训练行。Ultra独立核对该名单文件SHA与result引用一致，641条身份记录均在名单内，两组records hash与封存值相同。这证明probe触及用户的已排除成员身份，**不将partial名单升级为完整历史认证，也不证明剩余用户fresh**。脚本在实例及类两层禁止`targets()`，封存计数为0；没有未来标签评分、变体选优、完整epoch或模型checkpoint输出。

| 追加审计项 | 状态 | 证据与边界 |
|---|---|---|
| 真实历史 slice 候选等价 | PASS（固定样本） | 四路128/129行覆盖V2的128行边界，三路512条训练前缀；旧逻辑与新缓存每行候选ID、顺序及canonical hash完全一致；使用全量encoder和已绑定hash的资产 |
| 全量模型初始化复制 | PASS（seed 42） | 三变体每份50项参数/buffer hash；除既定`cross_gate_logit`白名单外完全相同；脚本逐项检查FP32和显式复制 |
| FP32 BPR有限性 | PASS（两步probe） | raw变体，batch256×2，16负例；随机6144、11–25档1024、26–50档1024、fallback0；loss、梯度及更新后的参数/buffer均有限，无AMP |
| 实测资源 | 已记录，非正式容量通过 | probe进程累计峰值RSS 5.984 GiB；CUDA分配峰值6.644 GiB、预留峰值9.020 GiB；两步合计512行，约937.46行/秒，仅为短probe观测 |
| 正式dev磁盘 | BLOCKED | 10份模型tensor峰值加1 GiB余量的最低需求为13,680,577,584字节，尚未含候选池；`/root`仅余11,038,056,448字节，`/root/autodl-tmp`仅余10,352,476,160字节，均不足 |
| 正式全过程RAM/VRAM | BLOCKED（尚未证实） | probe优化前已释放CPU模板state及factors，正式runner保留这些对象；两步峰值不覆盖正式对象生命周期、全部cohort/cache和checkpoint写入峰值 |
| 正式历史完整性与fresh锁定 | BLOCKED | 本probe不提供closure receipt或完整source registry；仍须独立闭合历史来源并锁定N |

候选构建实测为：四路128行旧4.558秒/新1.829秒，129行旧3.425秒/新1.850秒；三路512行旧7.456秒/新1.224秒。这些小样本包含资产加载、缓存和热身差异，不能直接外推成全量耗时保证。保留原逐用户evaluator，本次未引入或声称验证新的批量评分路径。

模型tensor实测1,260,683,576字节；磁盘最低缺口分别为2,642,521,136和3,328,101,424字节，且未计候选缓存、序列化附加开销与后续证据增长。全量4,731,777行训练候选及长度的净payload另需1,438,460,208字节，再加NPY头和sidecar。两块盘的空闲空间不能相加后宣称当前单一dev输出目录已满足检查，不能自动删除旧证据绕过阻断。资源说明已明确上述范围，与`result.status=passed`的局部probe含义一致；后者不构成正式prepare/dev授权。

## 10. 用户授权的本月fresh范围与v3锁定（2026-09-24）

用户明确指示：**“不要管6月，只要这些人没有被用于本月的实验，就可以使用。”** 据此，fresh的排除范围改为北京时间2026-09-01 00:00:00起至主代理锁定的2026-09-24 01:54:44止，按半开区间`[start_inclusive,end_exclusive)`处理。这里“用于实验”沿用评估标签曝光或模型/参数/配置选型口径；仅使用既定历史prefix训练不触发排除。九组、100k/20k/80k、all_fresh、训练与比较规则均不变。这是用户主动修订协议，不是把原全部历史缺口签认为已解决。

### 10.1 唯一scope对象

以下对象固定保存为`history_scope`；正式模式不接受别名字段、naive时间、缺失字段、不同月份/时区/结束时刻或更弱的曝光策略：

```json
{
  "schema_version": 1,
  "kind": "month_to_freeze",
  "timezone": "Asia/Shanghai",
  "start_inclusive": "2026-09-01T00:00:00+08:00",
  "end_exclusive": "2026-09-24T01:54:44+08:00",
  "exposure_policy": "evaluation_or_selection",
  "training_prefix_allowed": true
}
```

使用既有canonical JSON规则`sort_keys=True,ensure_ascii=True,separators=(',',':')`计算`history_scope_sha256`。结束时刻必须晚于开始且不在未来；不得随每次启动重新计算`now`，不得用文件复制/下载时间替代实际实验时点。冻结后至正式prepare期间不得另行用候选fresh用户做评估/选型；若发生新的标签曝光，原锁定不再有效，需要保留旧证据并重新审计新freeze，不能只改名单或证书日期。

### 10.2 月度来源归类与身份证明

月度`H_Sept`只并入scope内真实评估、选型或其可验证保守覆盖集合；原`F=E\H`及历史训练候选规则现明确为`F=E\H_Sept`、`D=E∩H_Sept`。fast-train仍从D固定seed抽100,000人，screen20,000与confirm80,000不变；N为全部F人数，不抽样、不设20,247上限。若D不足100,000、N=0或scope内来源不能闭合，正式运行仍停止。

来源registry必须区分“排除贡献来源”与“仅身份/时间/映射支持来源”。2026-08-28及其他九月前的编码器、原始输入、日志可以证明UID→raw映射、输入版本或程序身份；**它们自身不产生月度排除用户**。不得因九月某次10,000人抽样评估使用了八月774,585人验证全集，便不经证明把整个八月全集当作九月实际曝光；只有确有scope内运行、并给出可验证的包含关系与必要性时，才能对该运行作保守覆盖，必须明确标注过度覆盖的人数及依据。

每个运行记录实际评估/选型发生时段、时间证据路径/SHA、用户身份或selection重建规则、来源代码/配置/映射hash。mtime、目录命名和当前源代码单独不足以证明运行时刻或旧抽样逻辑。跨起止边界的运行应以实际发生的曝光识别；无法辨别但可能落入scope的来源保留unresolved，或由有明确九月运行依据的集合保守覆盖。仅prefix训练、候选生成及`targets_calls=0`且无其他评分/选型的资源probe不因读取用户身份而成为评估排除来源。六月及确定九月前来源不再构成本协议的完整性缺口，旧全部历史审计保留原样。

正式`historical_users`与registry保存完全一致的`history_scope`及`history_scope_sha256`；closure receipt的`scope`必须与此对象逐字段一致，并携带相同`history_scope_sha256`。closure仍要求complete、scope_complete、空unresolved、审计人、UTC完成时间、raw_union_sha256和可核验registry路径/SHA；追加`raw_union_count`与`source_count`和实际内容一致。receipt只能认证这个明确月度范围，不可沿用旧all-history receipt或把`attested=false`改成true。registry逐来源的状态、文件hash和保守覆盖支持链继续检查；scope内未解决来源不可被scope外支持文件掩盖。

### 10.3 全链路锁定与证据隔离

新协议版本为`next-cross-multiseed-20260924-v3-month-scope`。v1/v2 manifest、selection、final plan与checkpoint不自动升级；正式prepare使用新空目录。v3移除原base-specific20,247异常守卫：该数字属于旧all-history定义，在新scope下不是可使用人数上限或异常依据。未来若新增上界，必须同时绑定data_id、history_scope_sha256和来源证据，不能跨scope复用。

scope对象、`history_scope_sha256`及closure SHA随protocol、dev证据、selection、final plan、最终训练manifest/checkpoint、排他marker和结果保留；重载时复验与protocol/证书一致，不能只检查某处自带hash。`historical_users_verified.json`保存新认证原件的来源hash。更改scope后即使重新算manifest hash也应拒绝；test仍须重算为该scope下全部eligible fresh，UID/raw/position、顺序和coverage的既有校验不放宽。final marker建立前完成scope/proof验证，失败不触碰test targets。

candidate cache及模型算法保持第8–9节已审计逻辑；只调整协议与证明元数据，并为变更重新冻结code hashes。旧GPU候选/初始化probe可作为未变算法的支持证据，必须明确其旧代码hash，不能冒称新v3全链路验证。

### 10.4 v3新增验收与资源重算

新增测试至少覆盖：缺scope、错误kind/月/时区、naive时间、无效或漂移end、未来end、错误policy/boolean、scope与hash不符、history/closure/registry不一致、raw/source计数错误；旧v2拒绝；对同一正式base允许N>20,247且仍取全部F；scope变更并重算外层hash不能绕过plan/checkpoint/marker校验；scope外仅支持来源不直接贡献排除。保留全部已有阶段、身份、动态N、缓存与采样测试，使用全新合成九组smoke并封存实际结果。最终scope来源审计由Ultra history独立完成，代码实现与本节契约由Ultra复审；不能以工程测试代替本月来源闭合。

潜在N增大不改变13份保留/14份全生命周期模型峰值以及dev10/final5阶段模型预算；按真实N重算test候选、有效长度、NPZ身份/指标数组、JSON证据和日志空间，保持每盘1 GiB余量。全量train候选行数仍由base原始4,731,777条决定。第9节磁盘阻断仅在有可核验的分盘峰值预算及实际剩余容量证据后解除；不得因月度人数更多而修改一次测试或接受条件。

### 10.5 Ultra v3追加签审与当前放行条件

**工程实现及冻结证据PASS；RAM/VRAM按下述实测与保守容量账本有条件PASS。正式prepare/dev仍须取得月度历史闭合证书，并以真实R/N、实际证据大小和即时可用资源通过分盘预算。** 本节替代第8–9节中已由新证据解决的工程和资源状态，不把当时未完成的审计改写为已完成。Ultra只核查本地代码及下载证据，未连接服务器、未启动真实数据实验。

v3冻结runner SHA-256为`ec8df470eab5bc3bb39978371c9d086b9a1e834b2be63dc72648616ffa9d1e22`，cache仍为`e328ad066104df75b496450cf169e6ea47846c7c8d8194e38959b7652c914505`。scope的canonical SHA-256为`808016bf27329ad69c54e0b7377ae5fae9a3168a12c1ecbb1f5d275eaff53891`。审阅确认：history/registry/closure严格锁定同一scope及hash，closure人数与source数要求真正的整数且与内容相符；scope与closure SHA贯穿protocol、dev、selection、plan、checkpoint及final证据；旧v2拒绝，旧20,247守卫已移除。模型、召回、负采样、训练预算和选择门槛未变。

证据位于`evidence/cross_v3_frozen_validation_20260924/`。Medium封存结果为**91 tests / 4.141秒通过**，九组合成CPU smoke三轮**1.585秒**；Ultra独立复跑为**91 tests / 4.231秒通过**。Ultra逐项核对receipt中的六个代码/测试文件、unittest log及smoke summary SHA，全部匹配；以绝对路径重新调用`_validate_dev_evidence`、重算selection、加载smoke plan均通过，九份checkpoint的scope绑定逐一一致。合成counts为train4/screen2/confirm2/test2，`test_accessed=false`、无test cache、`status=smoke_only`、`final_evaluation_allowed=false`。Python编译和diff检查通过。直接调用内部证据验证函数须传入已resolve的dev路径；曾用相对路径触发的泛化hash错误已查明为路径包含关系判断，不是证据损坏，正式compare路径已先resolve。

服务器部署记录位于`server_snapshot/next_cross_20260924/server_v3_validation.json`及`server_v3_tests.log`：源码包SHA-256为`5230094f4dfdf556966c0181690719d493687b601a065b03358b765123105133`，94个来源文件核验，**91 tests / 9.275秒通过、exit=0**。JSON中的12.903秒是包含准备的整段wall time，不能写成unittest耗时。这些工程测试不认证真实月度fresh身份。

旧GPU预检可复用的范围经源码核实：与封存v2 runner `1c7ad8d9960de223ff9cb479df4ade63cb448e27f6c914fe467a5da5f7c8ef2f`逐项比较，`RecallPoolBuilder`、旧oracle、factors加载、template/variant构造、evaluator、负采样、batch切分、bootstrap和paired逻辑AST一致；训练函数仅增加scope元数据传递。故第9节候选等价与初始化证据仍支持未变算法，不能改称v3端到端GPU运行。

主代理另提供已完成的v3对象生命周期probe，目录`server_snapshot/next_cross_20260924/gpu_lifecycle_v3/`，`final_seal.json` SHA-256为`e6aeb4f5cf15aa540c51a99ceff1d8cb8aa284e82ff42c7916c15280e7416589`。Ultra核验四个封存文件和当前runner hash，确认仅复用原512条历史prefix、相同候选及两次256 batch；loss与负例trace和原probe完全一致，`targets_calls=0`。本次在优化器、梯度、CPU模板1,260,683,576字节、factors388,006,912字节均保持存活时，实际分配两份独立CPU模型tensor clone，模拟重载与上一份final payload同时存活。观测峰值RSS为**7.461 GiB**，CUDA allocated为**6.643 GiB**、reserved为**9.441 GiB**。这验证的是实际tensor存活集合；没有checkpoint落盘或真实反序列化，没有完整epoch、完整cohort metadata或全部cache驻留，不能声称正式全过程峰值已实测。

正式容量据此采用有额外余量的估算账本：7.462 GiB实测RSS之上，再宽松预留4份模型tensor约4.697 GiB、factors0.362 GiB、全部候选memmap约3 GiB、protocol/cohort/sets/临时数组和allocator 8 GiB、最大818,614名eligible用户下的250行bootstrap索引与值临时矩阵约3.05 GiB，合计约26.6 GiB；其中部分对象故意重复计入。candidate项使用R不超过全量4,731,777行、N不超过818,614人的宽松上界，prepare后替换为真实数。采用**32 GiB进程工作集预算**，不是数学证明或实测峰值；服务器cgroup上限66,571,993,088字节、GPU总量33,796,456,448字节与该预算相容。启动前须确认实际可用RAM至少32 GiB、GPU至少16 GiB且无竞争训练，并保留运行时RSS/VRAM监测。257尾batch与变体额外归一化/gate保留在该余量内，不改batch、精度或算法；无需再运行数百步重复probe。

分盘方案已核对`server_snapshot/next_cross_20260924/capacity_plan/`。以主代理提供的data空闲14,323,101,696字节、system空闲8,641,900,544字节为预算基准，dev置于data、protocol/cache及final置于system：dev的10T加1 GiB最低需求为13,680,577,584字节，另有642,524,112字节容纳checkpoint序列化差额和其他dev证据。final保留dev缓存后的system基线为`8,848,117,064 + 304(R+N) + E`字节，E须计protocol JSON、8R positions、封存证明、NPZ、trace、日志及序列化开销。当前system单凭原空闲量不足。

可行的后续安排是在**九组dev完成并复验之后**，将独立旧`/root/corrected_baseline_20260912/din_best.pth`（提供大小1,260,104,760字节）完整校验后迁至data，并用原路径symlink保留读取。必须记录源/目标SHA、大小、块占用和实际free，先复制/fsync、重读核验、原子完成目标，后原子切换原路径；不得先删除唯一副本。按名义字节预算，迁移后data在9T及1 GiB安全余量之外允许其他dev证据643,102,928字节，system允许`304(R+N)+E <= 1,053,888,240`字节。不能在dev前迁入该资产，也不能只下载九个dev checkpoint后从服务器移除它们：final各入口会重新验证全部九份dev证据。保留九份checkpoint服务器可读性与containment守卫不变。启动和迁移前分别用实测free复算，不能把两盘free相加、把E设零或只保证dev能写下。

月度历史来源的审计解释已与history负责人对齐：Stage A/B原始2026-09-10日志和原始runner证明两次实际九月评估；原代码采用全局`np.random.choice`，不能用后来`default_rng(42)`补成精确10k身份。若完整重建其原validation总体，可对真实九月运行保守覆盖774,585人，明确它是包含集合而不是实际曝光人数。该依据与“八月曾用过”不同，八月来源本身不构成本月排除理由。

已删除的`token_smoke_20260920`可按有限来源推断作`conservatively_covered`，但必须在正式registry封存完整推断链：最早同轮d0b4769/cf82d1e文档已记载固定future-window/CF300及Top5和paired成功；原runner只消费既有base固定evaluation前缀，不构造或重采样base；当时全部可达兼容base的构建清单、data_id和SHA完整且无反证。以future-full前300个dev/test记录及完整future-smoke用户并集覆盖，或采用已封存的更宽四基座并集，均须明列保守超额部分。原manifest与smoke命令缺失继续披露，300参数不能证明实际返回300人，权重体积不能识别base；不得标为精确恢复。history负责人封存月度证书后，主代理仍需调用正式loader并独立核对真实cohort、scope和完整资源条件，才可进入已授权的prepare/dev。

### 10.6 月度闭合证书独立复核

history负责人随后完成`server_snapshot/next_cross_20260924/history_audit/monthly_20260924/`，本轮月度历史前置条件现为**PASS**。Ultra使用冻结v3的`load_historical_users(..., require_closure=True)`独立读取正式证书，17.114秒通过；未修改证书。`historical_users_month_20260924.json` SHA-256为`c11c428479f455f447a24cf2c581e32c2d47a9e7aa2b70f6aa4bbdad76aaf5b5`，closure receipt SHA-256为`7dedeedd636512fb16355dbb678d65e149e11c414350ee92894c88f8a4153ee0`，source registry SHA-256为`2e7f764cb2ce048ed6a2f03938632cd9cbeaf17b3ed5194aaa190dfe3b825487`；三者与固定scope及hash一致，无unresolved来源。

Ultra按封存的身份导出、selection边界及保守覆盖规则独立重建296个来源集合，逐项验证人数和canonical raw-set hash，并集与history/closure/registry一致：266个`recovered`、30个`conservatively_covered`，共**796,912名排除raw用户**。并集canonical SHA-256为`7b47bbedca3f03aadf17fbe7032dde343eeb98979fa3dfecebb651e719f2d2c6`。直接核对closure引用的五份proof SHA，并以已校验SHA为`e71413ee4eead36954650c1ddb99c8334ead9edecee75ab6b9c01012efd9bb06`的完整base仅重算身份与prefix训练资格：eligible共**796,197人**，其中历史合格**778,901人**，全部剩余fresh为**17,296人**。fresh排序raw集合SHA-256为`878750379c1c23082ddbcb1c7d74147318752cfc96735921e596d69e4052d6bc`。原始排除总数可大于当前base eligible人数，因为并集也含不在当前eligible内的历史raw身份；不能将两数直接相减推算fresh。

关键保守来源亦独立复算：deleted token smoke采用更宽的四基座1,199人并集，身份与每个基座覆盖hash一致，相对此前并集新增5人；证书明确两corrected基座不兼容原`targets()`接口，仅保留额外覆盖余量，并披露原命令、manifest和逐用户数组缺失。local future smoke用原seed42、排序后的完整raw用户列表重抽30人，身份与proof完全一致；原文30人、188条正反馈、142条prefix训练、28名评估用户四项精确重现，保守排除全部30人，相对前序并集新增2人。Stage A/B的774,585人全集仍基于真实九月运行及原始随机抽样逻辑，而非八月运行独立贡献；undated offline smoke按可能九月曝光保守纳入。证书限定为项目历史有限来源下的保守闭合，不声称每名被排除者实际看过标签。

上述独立复算没有调用`targets()`、生成候选、执行模型或读取推荐指标。历史与人数条件足以允许主代理执行正式prepare以锁定协议并获得实际`ranker_train_rows=R`；正式dev启动仍须将R及17,296名fresh代入第10.5节预算，并以即时RAM/GPU/free与实际metadata字节复核。无需重复91项合成测试，也无需继续六月缺口审计。

### 10.7 正式prepare及最终启动资源签审

主代理提供`server_snapshot/next_cross_20260924/prepare_v3_receipt.json`，正式prepare exit=0、耗时43.734秒，协议hash为`dcf65b10c266ba02b00d0493815a14b8af9aa4f47551d7c562a2010104409667`。实际**R=604,511、N=17,296**及train/screen/confirm/test四组records hash均与Ultra只读完整base预计算一致，positions数组SHA-256为`b09afbf0d67a51ebd7218185204bdf848756eb4637444c12bbc71df45835387f`。开发拆分100,000/20,000/80,000，四项交叉集合计数均0，scope及closure hash保持一致，`test_future_labels_read=false`。

**最终启动资源签审PASS，允许按已冻结命令启动正式九组dev，无新增审计阻断。** 审批依据为`launch_capacity_receipt_v3_balanced.json`，它保留并补充原`launch_capacity_receipt_v3.json`的单DIN迁移方案。即时检查时，history传输tar、解压证明及107,324,555字节protocol均已落盘；system实际free为7,978,549,248字节，data为14,323,036,160字节，不能再重复扣除这些已有对象。RAM可用60,899,930,112字节，GPU可用33,795,604,480字节且无竞争进程，均超过32 GiB/16 GiB门槛。

192 MiB（201,326,592字节）每盘额外证据预算经输出结构复核可接受：当前evaluator的逐用户文件仅含两列int64身份、一列float32和两列int8，合计22字节/人；dev的27份20k screen与9份80k confirm，即使按不压缩数组计也仅27,720,000字节，final的6份20k screen与两份17,296 test合计3,401,024字节。按真实dev 2,362步及final 18,484步展开现有trace/manifest模板，重复JSON及checkpoint非tensor元数据估计分别约1.74 MB和4.69 MB；模型tensor主体已单独计T，不重复纳入。各50项tensor的容器头、NPY/ZIP附加开销及有限manifest/hash文件可由剩余预算覆盖。一个宽松分配为每盘40 MiB逐用户结果、16 MiB结构/序列化附加、64 MiB日志及64 MiB小型证据归档，共184 MiB，另余8 MiB。该预算不包括另在服务器复制整套模型、候选缓存或完整history树；这些大对象若需新增副本必须单独预算，不能挤占1 GiB安全余量。

balanced方案在九组dev完成并完整复验后迁移两件独立不可变产物，均以完整复制/fsync、SHA及可读性复核、目标原子完成、原路径原子换为symlink的顺序实施；迁移前重新核实源为预期文件且无写入者：

| 来源 | 目标 | 字节数 / SHA-256 |
|---|---|---|
| `/root/corrected_baseline_20260912/din_best.pth` | `/root/autodl-tmp/next_cross_20260924_immutable/corrected_din_best.pth` | 1,260,104,760 / `69724b407c42af8c1b3802f6821228c8178b4623a5c31991090b5f0131e35423` |
| `/root/next_cross_20260924/history_month_v3_20260924.tar` | `/root/autodl-tmp/next_cross_20260924_immutable/history_month_v3_20260924.tar` | 248,166,400 / `fd7f01891d6d20a611b9b62e244a2ac45af4e80acbec04c83e9b808dbdb609a5` |

独立传输tar不属于解压后的运行证明路径；解压history树、protocol、source-chain及九份dev checkpoint保持原样。Ultra同意把该归档包作为第二个迁移对象，现有用户备份移动授权覆盖此无损安排；这不是删除证据或修改冻结协议。两件迁移均尚未执行，本节不把计划写成完成收据。

全部候选含2 MiB固定附加预算共1,659,986,688字节。Ultra逐式核对balanced回执：dev data要求`10T + 1GiB + 192MiB = 13,881,904,176`字节，余**441,131,984字节**；dev完成后data保留9T并迁入上述两件产物、仍扣1 GiB与192 MiB，要求14,129,491,760字节，余**193,544,400字节**；system释放两件源文件后可用9,486,820,408字节，对final的5T峰值、全部候选、1 GiB与192 MiB合计9,238,472,984字节，余**248,347,424字节**。这三个余量均在安全余量和额外证据预算之外。只迁DIN时system仅额外余181,024字节，追加迁移tar可平衡两盘，故正式全流程采用balanced安排。dev完成后的迁移与final入口仍须用实际文件占用、增长和free复检；无胜者不进入final，不需要为了使用预算而执行迁移。

### 10.8 正式运行期间只读增量归档工具审计

主代理报告正式dev已于北京时间2026-09-24 02:36:37启动，监控证据放在封存dev树之外。新增`tools/archive_cross_v3.py`与`tests/test_archive_cross_v3.py`不位于冻结`code/`，不会改变runner的code hash集合。最终工具SHA-256为`2f1bde3d14a81f07de080d1031235648da0bdcc038a0a84a435f332e0fc5ac10`，测试文件为`df27417f466237abcaa051857e1ee635a1a692cc615920c6113e18ada856f71c`。**Ultra风险签审PASS，可由主代理/Low使用既有SSH ControlPath每900秒单实例调用。** 工具本身为单次执行，不建立调度、不训练、不迁移、不关机。

远端执行只有固定Python身份/大小/SHA inventory和固定路径`cat`读取，命令参数经shell quoting；没有远端写入、删除或清理操作。认证沿用既有ControlPath与BatchMode，不把密码、私钥内容或token写入命令/receipt。每组必须已有三epoch的`COMPLETED.json`，其checkpoint SHA同时匹配manifest；只归档10份不会再修改的训练文件：checkpoint、history、manifest、完成标记、三份epoch trace和三份screen NPZ。由于组完成标记早于全局confirm评估，confirm明确排除，不能根据组完成就宣称整轮证据已封存。可选protocol归档仅复制七份初始身份文件，test records仅为已锁定身份，不调用targets或生成候选。

本地下载写入独立`.partial/<run-id>/`，逐文件核对size/SHA与JSON/NPZ/NPY可读性；checkpoint使用`weights_only=True`、CPU加载，检查tensor有限性并交叉核对manifest/history/trace。传输完成后再次读取远端inventory，一致才写本地receipt、fsync并原子发布。已存在副本只验证，不覆盖；漂移、失败副本及失败receipt保留。独占锁防并发，本地root及祖先目录symlink拒绝，测试确认不会写到root之外；异常中若receipt失败或进程终止留下锁，应停止调度并检查进程/partial/receipt，不能自动猜测锁失效或删锁隐藏失败。工具没有把本地备份成功解释为远端删除授权。

Ultra独立运行归档专项**7 tests / 0.071秒全部通过**，覆盖原子发布、幂等不覆盖、hash失败partial保留、漂移拒绝、NPZ/checkpoint读取和有限性、远端二次inventory及锁/symlink。还把同一远端inventory脚本只在本地封存v3 synthetic目录运行，验证真实synthetic checkpoint的受限读取、10份组文件和七份protocol文件均通过；没有连接服务器。Medium封存证据`server_snapshot/next_cross_20260924/archive_tool_validation/`含专项及完整**98 tests / 4.138秒通过**记录；Ultra核对四份seal内文件SHA，seal本身SHA-256为`dcf1da23fb0ecd2542dd6d5b69c669a2b6071af5731e8d2ce02564ca30ac5574`。runner/cache SHA保持第10.5节不变。

增量目录含新增`ARCHIVE_RECEIPT.json`且没有confirm、全局dev manifest/完成证据，它是可验证训练备份，不能直接充当完整dev验证树。全部九组及confirm完成后须另作全局原始证据归档；需要还原运行证据时，将归档receipt留在还原树之外并按原始封存inventory复核。正式final入口仍读取服务器原九份checkpoint与dev证据，本地增量成功不解除这些依赖；不提前删除服务器模型，不据此重开test或改写任何seal。

### 10.9 九组完成后完整归档工具最终签审

**`tools/archive_cross_v3_complete.py`最终审计PASS，可在全局dev完成后由主代理触发；尚未执行真实服务器完整归档。** 最终源码SHA-256为`4066e0206b24f69ca5e1783232876179af96e51e8b5fcb5f88b33c98d7875ca0`，测试文件`tests/test_archive_cross_v3_complete.py`为`42bca7bbca29df7e0f878e5e496ca19de6ae5c96b5c61de7243599b20d5df28f`，共享增量helper仍为第10.8节的`2f1bde3d14a81f07de080d1031235648da0bdcc038a0a84a435f332e0fc5ac10`。冻结runner/cache保持第10.5节哈希，未改实验代码。Medium封存专项**9 tests / 2.807秒**、完整**107 tests / 5.314秒**以及编译、diff检查通过；Ultra独立复跑最终专项为**9 tests / 2.841秒全部通过**。测试在临时目录生成完整九组三epoch合成fixture，不再依赖Git忽略的历史`.pth/.pkl`。验证目录为`server_snapshot/next_cross_20260924/archive_complete_validation/`，四份证据文件与seal逐项一致，seal SHA-256为`060afab5deb74384fbf5d7ed87adcf19fea9d6e29840af7574298f9983d6916a`。

工具先读取全局dev `COMPLETED.json`；只有`status=complete`、`groups=9`、`test_accessed=false`才继续，否则正常返回`not_ready`，不hash或下载仍在写入的树。进入后远端禁用Python bytecode写入，以服务器resolve后的绝对dev路径调用冻结`_validate_dev_evidence`，验证原路径上的code/base/history/scope/trace/cache/checkpoint依赖。远端无evaluate、compare、targets、训练、迁移、删除或关机操作；`--dry-run`在ready后也会做完整官方验证及hash，不适合频繁轮询。

读取protocol任何内容前先只检查路径名，白名单仅含七份初始身份文件和train/screen/confirm三份dev cache各自的sidecar、items、lengths及保留的`.json.building`文件。final缓存、`TEST_STARTED.json`或其他未授权文件均在内容读取/hash前拒绝；inventory遍历到每个文件时再次检查白名单，防止首次扫描后新出现final文件。最终测试用read trap验证预存测试标记不会打开；Ultra另在本地合成fixture首次名称扫描后注入标记，得到`protocol path appeared after gate; refusing to open`，未触发内容读取trap。两项都未连接真实服务器、未读取未完成指标。

完整dev树包含九份checkpoint、三epoch训练证据、seed初始化审计、confirm逐用户数组/指标、全局manifest及完成标记，protocol树包含初始身份与全部三份dev cache payload。冻结runner的`evidence_hashes`排除所有名为`COMPLETED.json`的文件，因此工具严格允许`evidence_hashes`加全局manifest/标记及九份预期组标记；各组标记另外由`verify_group`校验，不使用宽泛的extra忽略规则。所有归档receipt留在原始树之外。

远端audit显式绑定protocol/dev hash、code hashes、scope hash、closure hash及`test_accessed=false`。本地核对这些绑定与下载manifest一致，并检查全部inventory的size/SHA、JSON/NPZ/checkpoint可读性、tensor有限性、初始化state hash、candidate metadata/payload/逐行身份及配对UID/position。下载完成后第二次远端audit/inventory必须与第一次完全相等。本地检查是可携带备份的完整性与身份验证，不能冒称把不存在的`/root/...`路径搬到本地后已通过完整官方验证；完整原环境语义检查依赖所封存、且与下载字节绑定的远端官方validator结果。source/base/history等既有外部依赖不重复传输，完整重放仍需分别保留它们。

匹配的增量checkpoint只有在receipt、size/SHA及受限加载/有限性重新通过后才通过hardlink复用；其他文件从远端读取。hardlink共享同一inode，是两个不可变视图，**不是独立灾备副本**。两目录跨文件系统时失败关闭，不暗中替换为有破坏性的操作。正式目标为`dev_complete`、`protocol_complete`和树外`dev_complete_receipts`，三者及`dev_archive`彼此分离，服务器原始文件不删除。

发布顺序是先rename protocol树、再rename dev树，最后发布树外完成receipt，**不是双目录事务**。任一步中断均保留已发布树、partial及失败证据；未取得完整receipt的后续调用拒绝覆盖，必须先检查残留状态，不能删除唯一证据或手改seal来假装续跑。独占锁异常残留也须先核进程和partial；工具不自动猜测陈旧锁。已有完整封存副本只重验，一致则跳过、漂移则失败。完整归档必须在任何final缓存/测试标记出现前结束；selection/plan放在dev和protocol树之外并单独归档。

### 10.10 已授权后续执行链、阶段触发及失败边界

本节按冻结v3 `compare`、`_verified_selection`、`lock`、`_load_plan`、`final_train`、`final_test`及原协议的一次验收约束复核。**执行链可由主代理在已有用户授权内继续，不需重复请求启动许可；本节是未来阶段的触发条件，不是这些阶段已经执行或通过的声明。** Ultra未读任何未完成真实指标，也未执行服务器操作。

1. **全局dev完成 → 官方验证与完整本地归档。** 必须是九组固定三epoch全部保存、confirm完成、全局marker/manifest均完整且`test_accessed=false`，冻结官方validator验证全部身份与证据。单组完成或增量归档不足以触发最终阶段。第10.9节完整dev/protocol归档须成功封存后，才允许产生final缓存或测试marker；compare可在全局证据通过后运行，其输出不得放入封存dev/protocol树。

2. **compare → lock → 明确分支。** 使用冻结门槛及排他输出创建，lock通过`_verified_selection`重算整个注册决策，并绑定code/base/cohort/scope/closure。两个挑战模型都达标是合法情况：先按confirm平均HR收益、再按NDCG、再偏好`zero_cross`排序；HR差在`1e-12`内时采用NDCG/`zero_cross`规则，最终只锁定一个winner，不能附加`accepted_candidates`必须恰有一个的新门槛。若`no_winner`，plan应为`final_evaluation_allowed=false`、`winner=null`、`models=[raw]`，跳过全部final训练和测试，不消耗fresh标签，也不为final腾空间而实施迁移；封存无胜者结论、selection和plan后进入关机收尾。已有selection/plan应复验其与注册决策一致，不覆盖、删除或换文件名绕过失败。

3. **锁定accepted winner → 资源复查 → 全量成对训练。** 必须保持服务器九份dev checkpoint及完整依赖可读，因为各final入口的`_load_plan`会重新验证selection和dev。完成完整归档后，按第10.7节执行必要的两件不可变资产迁移，源/目标hash与原路径引用保持一致；即时复算两盘实际free、RAM/VRAM及竞争进程，final目录须为空。固定raw加锁定winner、seed42/init424242、三epoch、dim/token256、hist50、batch256、candidates75，使用全量**4,731,777条prefix训练行**并覆盖全部**17,296名fresh用户**的训练历史；screen只作固定训练诊断，不按test选epoch。两模型paired trace、完整行/用户coverage、final checkpoint metadata/hash及重载有限性通过，`final_manifest.json`与`FINAL_TRAIN_COMPLETED.json`匹配后才进入test。非空partial final目录不得自动删除或另起有利运行来绕过失败。

4. **完整final训练 → 同一17,296人一次成对test。** final-test再次核plan、训练证据、candidate receipt、全量positions/coverage及两checkpoint身份，protocol级`TEST_STARTED.json`必须不存在；代码在构建test candidate及评分之前排他创建marker。任何已有marker，包括停留在`started`、已`complete`或不完整状态，都阻止再次评分。即使故障看似早于targets读取，也保留marker、partial及错误，不删marker、不换run目录、不克隆protocol重封存、不改用户子集或另训有利checkpoint；冻结runner没有获审计的自动续接入口。只对已保存逐用户数组重算报告不构成新的模型测试。成功后核两模型完全相同的有序UID/position与候选、N=17,296、结果hash及marker `status=complete`，按冻结最终接受门槛报告`accepted`或`raw_retained`。训练manifest的`test_future_labels_read=false`描述训练阶段来源，测试完成后仍保留该字段不构成矛盾。

5. **最终结果或无胜者结果 → 本地完整封存 → 关机确认。** 保留已归档的source/base/history、原始dev/protocol，再单独封存selection/plan和阶段回执；若进入final，还须归档完整final目录、final训练/test候选的sidecar与payload、原始`TEST_STARTED.json`及results。第10.9节工具刻意只接受dev阶段，不能用它声称final已归档；最终归档路径须逐项验证远端/本地size/SHA、checkpoint受限加载与有限性、NPZ身份、result/marker绑定和一次test状态。确认本地证据可读、无活动训练/归档写入者后停止监控/worker，保留服务器证据，执行已有用户授权的关机并验证实际关机状态或可核验的关机断连结果，不能只凭命令已发送宣称已关机。

共同异常规则：身份、hash、scope、可用资源、证据完整性或marker任一检查失败即停止依赖它的后续阶段，记录原始错误和现场；传输可按工具既有幂等验证机制恢复，实验评分/训练不得借归档恢复自动重跑。主代理可继续已授权的只读诊断与无损修复，涉及改变冻结实验或续接已开始test则必须先完成明确审计，不能把“用户授权跑完”解释为允许绕过一次测试与冻结规则。

### 10.11 两件已批准资产迁移工具最终签审

**`tools/migrate_cross_v3_assets.py`独立审计PASS，仅在第10.10节accepted分支与全部前置门槛通过后执行；尚未连接服务器或实施真实迁移。** 最终源码SHA-256为`bcaca40a81090968868c9b18b8a901f617e8397409316b7f838958a8d2d50e8c`，`tests/test_migrate_cross_v3_assets.py`为`54cf8f74d9c0f173668cc1ff2bd97ab5f6b555508afb3c3821f56fe64642f19a`。Medium专项12项/0.055秒通过，Ultra独立复跑**12项/0.059秒全部通过**，包括目标突现不覆盖、源nlink/ctime变化、原路径切换失败、固定锁阻止换回执目录重试、迁后真实容量不足及完整归档回执绑定。Medium稳定组合119项/5.326秒通过，明确排除了并行开发的final归档模块；先前包含该未完成模块的125项尝试中三项失败留在结果中，不冒称当时全部通过。验证目录`server_snapshot/next_cross_20260924/migration_tool_validation/`的八项源码/测试/冻结代码/说明/日志/结果SHA均独立匹配，`SHA256.json`本身为`113eea1e9633089378146f877c349b5e8f2acf75e11c780746074297ce0b9bf6`。冻结runner/cache未改。

CLI仅允许第10.7节两个固定源/目标、bytes/SHA及固定protocol hash。默认check-only不创建目录、锁、回执或bytecode；执行模式重复所有门槛。官方dev validator确认九组及confirm全局完成，冻结`_load_plan`从已保存数组重算`compare(return_only=True)`并验证accepted lock；它不执行新的模型评分或targets。final候选或TEST标记已出现则拒绝。主代理必须先在本地用第10.9节`verify_bundle`复验完整归档，再把**原始`COMPLETE.json`字节不改**复制到服务器树外路径，同时提供在本地记录的SHA。工具消费该原件及独立SHA，核status/ready/test未访问、dev/protocol/code/scope/closure绑定、dev全部seal与protocol inventory；这证明其消费了经主代理确认的归档回执，不声称服务器能自行复验另一台机器上的本地文件。不能手工构造替代receipt。

源必须为无symlink祖先的唯一regular file，nlink=1且device/inode/size/mtime/ctime/nlink及批准SHA在验证、复制后和切换前一致。固定目标父目录下永久`.asset-migration.started`锁阻止不同execution目录并发绕过；已有目标、partial、锁或执行证据一律拒绝自动重试。复制到排他`.migration-partial`后flush/fsync、重算SHA并验证可读性，再用原子no-replace hardlink发布目标；成功保留的partial和目标共享同一inode，不另占一份内容，也不能当作可随手清理的失败垃圾。源目录先建立精确目标symlink并核验，确认目标已持久化后才原子替换原源路径目录项，再fsync源目录并重读。该切换释放原文件空间但保留完全相同的逻辑路径与字节；没有先删唯一副本的阶段。

容量按**当前实际free**计算，既有dev缓存不再扣第二次。预计回收上限取`min(文件大小,st_blocks*512)`；源/目标必须属于不同文件系统。每件复制前重查剩余迁移字节及data侧1 GiB＋192 MiB余量，system预计free须覆盖`5×1,260,683,576 + 304×(4,731,777+17,296) + 1MiB + 1GiB + 192MiB`；两件切换后以零待回收字节再核真实free才写COMPLETED。若旧文件仍被进程打开或竞争分配导致空间未释放，迁后检查失败并保留已完成切换和FAILED证据，不能进入final。工具只度量磁盘；32 GiB RAM、16 GiB GPU、无竞争训练/源写入者及final目录为空仍由主代理在最终启动前另外记录。

任何复制、fsync、验证或切换异常均保留源/目标/partial/link及永久锁和回执；回执初始化前失败时，固定STARTED锁仍保存预定execution路径。主代理可据现场开展已授权只读诊断，恢复须单独审查，不能删锁、改目录或清空partial来自动重跑。工具本身不连接SSH、不训练、不调用final-test、不关机。该PASS认证受审工具及其阶段约束，不替代实际部署hash校验、check-only结果、执行收据或最终资源复查。

### 10.12 accepted分支最终完整归档工具签审

**`tools/archive_cross_v3_final.py`最终工程审计PASS；仅用于已完成的一次final test，尚未读取或归档真实服务器final结果。** 源码SHA-256为`1b86f389300c00db94aee631d45a7ce9e271daf8f58cc96142c292bce49b6ad9`，测试文件`tests/test_archive_cross_v3_final.py`为`2a579b9ae57aa704f5af9e50d9a8d4cb36967a4a4d498d8173f54f2a3bcb6ead`。Ultra独立专项**8项/2.225秒通过**；Medium最终增量/完整dev/final归档组合**24项/3.753秒通过**。`server_snapshot/next_cross_20260924/archive_final_validation/`内源码、测试、cache、USAGE、实际UTC回执及组合日志六项SHA逐一匹配，`SHA256SUMS.txt`本身为`6fc3aac49b3568dcb04e4695ab89f5eb838184801d23935ed8be1401a5e0b9d9`；helper及冻结runner/cache另外独立核对，均未变。语法和diff格式检查通过。

远端先检查`FINAL_TRAIN_COMPLETED.json`是否存在及protocol级`TEST_STARTED.json`为`complete`，不就绪只返回not_ready；read trap测试确认缺测试marker时不会打开final完成文件或模型内容。ready之后禁用bytecode写入，调用冻结`_load_plan`，其内部`compare(return_only=True)`只重算已封存开发数组，不重新评分、不调用targets、不写selection；另通过`validate_base`只核身份、全量train positions hash及原路径依赖。CLI固定root下的final/protocol/source/selection/plan布局，远端仅做身份/字节读取、hash及验证，没有训练、模型测试、迁移、清理或关机入口。允许两个accepted候选按冻结规则选出一个winner，不额外施加accepted列表长度为1；无胜者分支不适用此工具。

归档保持`final_v3/`、`protocol_v3/`和`control/`三个相对树，后者包含selection、plan及四份明确命名的compare/lock/final训练/test日志，receipt在树外。final固定名单包含真实runner必产`disk_preflight.json`、初始化审计、shared及两模型epoch trace、history/screen NPZ、两组原始checkpoint和两份final包装checkpoint、完成manifest、最终metrics/NPZ/results；protocol包括原始身份、全部三份dev和两份final候选cache、保留的`.building`构建证据与原始TEST marker。训练evidence集合必须精确一致，先检查路径集合再逐文件打开，不允许extra、缺项或树外相对路径。远端inventory及下载后的原始文件保持原路径引用；不会改写manifest使其看似在本地通过原路径验证。

远端plan/protocol/final/marker/results身份绑定以及本地hash/size、可读性、scope/closure、全量4,731,777行、test17,296人及train positions绑定均核查。四份checkpoint分别受限CPU加载与tensor有限性检查，逐一核epoch3、seed42/init424242、scope、训练位置、各自manifest/history/组完成marker/trace；两份top包装的plan/test身份和state tensors与对应训练checkpoint一致，但不假设两文件hash相同或省略其中一份。候选用`load_cache`复验items/lengths payload、logical pool hash、records hash、shape和行数；配对test要求相同有序UID/position/pool_hit，逐用户数组反算metrics、HR/NDCG差值及gained/lost/users并与结果对应。final manifest的全量positions hash不能误用dev的604,511行子集hash；它以远端base和四checkpoint绑定。初始化非gate hash与两模型完整paired trace一致。

测试包含真实冻结runner在临时合成base上运行prepare、三epoch final-train及一次final-test，再把真实产物送入portable `verify_bundle`；为允许小型smoke产物，测试仅替换`_load_plan`授权返回和归档固定人数/hash常量，明确不证明正式accepted选择已发生。其他测试覆盖固定名单、hardlink inode复用、幂等复验、hash失败及远端漂移时保留partial/锁、remote.original与inventory hash、四checkpoint包装身份及not-ready读取边界。测试没有连接服务器或读取未完成真实指标。

下载先写独立partial，匹配的已有protocol文件只有size/SHA复验后才hardlink；共享inode不是独立灾备副本，两视图保持不可变。所有文件复验后第二次完整远端inventory必须完全一致才发布；目标与receipt及复用source三者分离，发布前再拒绝symlink祖先，既有封存目标只验证不覆盖。锁仅在成功且本轮receipt写入成功后移除，任何下载/验证/发布或receipt失败均保留现场；若目录已发布但最终receipt未完成，禁止覆盖，须人工审查残留状态。目标是归档已完成同一次计算，不存在借归档恢复重新执行实验的路径。

**统计复算边界：** 本工具验证已封存95% CI的两个端点有限且有序，并按该CI与已核delta/NDCG重算`accepted/raw_retained`规则；它不独立重跑bootstrap。待完整结果到本地后，最终结果审计仍应从原始配对数组按冻结种子与规则重算开发97.5%及final95% CI，逐值对齐selection/results；不能把此工程归档PASS写成置信区间已独立复算。portable校验也不替代服务器原路径上的完整官方依赖验证，source、exact base五文件、history与dev归档须另行保存并核验，缺少这些不能宣称已完整可重放。关机条件仍是第10.10节的实际本地证据完整、结果审查、无活动写入者及关机状态确认。

### 10.13 已有完整127项验证及exact base五文件归档复核

主代理已在并行开发收敛后完成全库**127项/6.202秒通过**，记录为`server_snapshot/next_cross_20260924/full_repository_ready_20260924.json`，UTC完成时间2026-09-23 20:20:32。Ultra只核既有结果，不重复运行旧测试：日志结尾为127项OK，SHA-256为`470530c536e359c3b8b1192d4c3486c35ee5b8ee31849fedbfa1e637a46d0d48`；回执列出的十个工具/测试/冻结runner/cache文件SHA与当前文件全部一致，回执自身SHA为`b835589e71eecc5ca8fb221b908a62e9fe754bd5590e8edfd653f53f00c5d607`。这关闭了第10.11节曾披露的并行final fixture失败；旧失败记录仍保留。127项不包含后来新增、尚在单独审核的final阶段monitor工具，不能提前宣称覆盖它。

exact base五文件已于北京时间2026-09-24 04:16:47发布到本地`server_snapshot/next_cross_20260924/base_archive/`，合计**2,038,507,950字节**。Ultra独立核目录恰有五份非symlink原文件，逐项重算size/SHA并与冻结protocol的`base_assets`完全相等，data另与`base_data_hash`一致；远端传输前/后inventory记录逐字段一致，assembly四来源的canonical路径/size/SHA也与本地原组装manifest一致。五份字节身份如下：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `data.pkl` | 322,157,387 | `e71413ee4eead36954650c1ddb99c8334ead9edecee75ab6b9c01012efd9bb06` |
| `run_manifest.json` | 1,960 | `a034a78b6641017fd413e42bd4d37c834fb1ede0acfb7c5e8a3f5d562b835ef1` |
| `v2_best.pth` | 418,950,145 | `58301a1ce43a1db609c5cd9a22e7b35f5c01849faf32c3a9ff2d8347f8d311c5` |
| `svd.npy` | 388,007,040 | `51565a6c1788348162aecea6451ad41779bb496c9f484ca450efd2af716b9d4b` |
| `itemcf.pkl` | 909,391,418 | `8fa65999a9574ccc1dd1d0cd5cff97ca21d16db407bdc83e4629c2cd73f240bf` |

原归档程序的只读路径和完成回执亦核阅：复用早前恢复目录内同SHA的data.pkl，并另外下载四文件；先partial排他写入、fsync、逐文件校验，远端二次inventory相等后发布，收据留在`base_archive_plan/`树外。完成回执`COMPLETED_20260923T201120Z.json` SHA为`74bcd9d9d635e465ebd704f12985d132587f3d3e20d6d2f48112051ef4a36646`，assembly回执为`09e48e6cd6b633c13e5ca4cbb06edffd30be3e781829c5335c59cfaa746459e5`。

可读性记录明确限定范围：V2在CPU以`weights_only=True`加载，93个tensor有限；SVD为378,913×256 float32、分块有限性通过；JSON可解析；两份pickle只用pickletools遍历完整语法流并核STOP与文件结尾，不称已经语义反序列化。原始组装manifest没有data_id或encoders_hash，备份不臆造这两字段，也不声称此次重新计算；身份依据仍为冻结protocol及先前正式base审计。Ultra本轮只重新核字节身份与记录，没有再执行模型、targets或pickle语义加载，没有连接服务器；“exact base尚缺本地副本”的前述关机依赖缺口现已关闭，其他阶段及最终统计审计条件不变。

### 10.14 固定final单阶段启动与外部900秒监控工具签审

**`tools/monitor_cross_v3_final.py`工程审计PASS；只批准受审工具在既有阶段条件下使用，未部署或启动任何真实final阶段。** 最终源码SHA-256为`adddce35575b17bb81041e0e261fdbe72cdbcdf6c0440b349995427d9216ca94`，`tests/test_monitor_cross_v3_final.py`为`6546e32fab18dc903b5547ea99470d9b6ca15d7529004742b339d379776fa918`。Medium专项11项/1.051秒，Ultra独立复跑**11项/1.047秒全部通过**。Medium新增工具后的全库138项/6.595秒日志OK；Ultra仅核既有日志和hash，不重复全库测试。`server_snapshot/next_cross_20260924/final_monitor_validation/`中八项源码/测试/冻结代码/USAGE/日志/结果SHA全部一致，`SHA256.json`本身为`aaae4bd7f98da249319b85142ee5ef68dec8c0ab19f83a271d283cbe4dc1e6ac`。语法及diff检查通过，冻结runner/cache未变。

工具默认check-only，只接受`--stage final-train`或`--stage final-test`；`--execute`仅启动所给单阶段一个子进程，不自动串联、重跑、归档、迁移、清理或关机。固定本轮Python、src_v3/code、base、dev、selection、plan、protocol和final路径；参数为dim/token256、hist50、candidates75，训练另固定batch256、三epoch，计划固定seed42/init424242。精确argv已交给真实冻结runner parser验证，不用shell拼接。preflight核两核心SHA、固定路径、accepted选择/lock及protocol hash，调用官方`_load_plan`重验完整dev及从已存数组重算compare；这里没有新的模型评分或targets调用。

训练入口要求final为空且无final/test缓存；test入口要求完整final manifest及匹配训练完成marker、plan/protocol/scope、全量4,731,777行和17,296人历史coverage、test records hash、sealed evidence和两份top checkpoint hash。任何TEST标记（含悬空symlink）拒绝启动；即使marker丢失，既有test候选、results或逐用户/metrics输出也拒绝，避免覆盖一次测试现场。evidence相对路径在内容打开前严格拒绝absolute、`..`、非规范分隔及symlink；canonical JSON hash遵守冻结`ensure_ascii=True`，中文回归与原runner一致。

完整dev本地归档、accepted决定、受审迁移及其实际迁后容量、当前RAM≥32 GiB/GPU≥16 GiB、各盘余量、无竞争训练/迁移/源写入者是**主代理外部综合门槛**。按主代理决定，不增加新的工具许可回执CLI；USAGE明确没有这些实际票据不得execute，STARTED保留`root_external_gate_required=true`，不能把工具check_passed解释为它已核全部资源。Low只执行主代理单独指示的一个阶段；final-train monitor须先退出且`exit.json.status=complete`，训练证据独立复核后，主代理才另行指示final-test。marker提前出现不授权两个阶段并行。

每stage排他创建永久`monitor_final_train_v3`或`monitor_final_test_v3`目录作为一次执行锁，所有STARTED、launch、stdout、snapshot、exit/error位于该外部目录，不放进冻结code/dev/protocol/final证据树。stdout/stderr直接以原始字节写独占日志，包括无效UTF-8，不读取、tail或展示运行中的loss/指标。快照只记录文件名/字节、PID/资源、GPU进程、RAM、disk free及cgroup memory/OOM值，开始、固定900秒间隔和结束各采集；普通主循环最多等待5秒再观察退出，不整段睡900秒。采集命令各有30秒timeout，因此采集期间可能延迟退出观察，不声称任何情况下5秒内发现退出。采集失败被保留并将结果标为`monitoring_failed`，不会让采集全失效的监控冒称成功。

退出码0本身不足：STARTED记录的selection、plan、protocol和plan列明全部code SHA须在退出后仍一致，并验证该stage的完成链。train检查manifest/marker、计划、coverage与checkpoint seal；test另检查TEST marker complete、results hash、scope/cohort/models、checkpoint绑定、配对人数和必需输出文件。回执只包含结构和hash，不输出指标值；`complete`表示该进程阶段具备受核完成证据，不替代四checkpoint完整归档、候选/NPZ交叉验证或统计CI复算，`raw_retained`可以是有效完成状态。

测试通过实际冻结runner在临时CPU合成fixture生成prepare/final-train/final-test产物，再核阶段完成与篡改拒绝；只替换smoke授权边界，不认证正式accepted运行。其余测试覆盖真实parser参数、check-only零写/不spawn、单child及900秒时钟、原字节日志和永久锁、非零/缺marker不报成功、snapshot不读metrics、Unicode hash与树外路径拒绝、丢marker残留test拒绝、采集退化及退出代码漂移。没有服务器连接，没有读取未完成真实指标。

失败或监控异常永久保留目录、marker和已有输出，不自动kill/restart child；monitor自身被中断时子进程可能仍存活，必须先核原PID，不能启动替代实例。异常死亡可能来不及写exit，STARTED目录仍阻止重试。每阶段退出后，主代理须将stdout原字节导出为既有final归档工具需要的root级`final_train_v3.log`或`final_test_v3.log`，核size/SHA并记录对应关系，既有日志不覆盖；整份monitor目录也作为外部证据保存，不能误认为四日志白名单已自动包含所有快照。工具不执行该导出，也不根据任何失败自行关机。
