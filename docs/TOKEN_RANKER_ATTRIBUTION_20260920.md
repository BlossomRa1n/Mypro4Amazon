**Token 精排失败归因复核 · 2026-09-20**

最初复核依据已提交的实验结果、历史评审和实际训练代码；当时没有启动服务器或重新训练。重点是 9 月 20 日凌晨的正式实验，同时解释早期五路 token 的失败是否属于同一问题。

**最终更新：四组实验、全量协议审计及本地证据校验全部完成。** [最终优化总结与结果分析](TOKEN_CONTROLLED_RESULTS_20260920.md) 是当前结论：**SVD 用户均值初始化是本次退化的主要来源；随机用户初始化下，semantic_concat 的测试 HR@5 为 4.585%，高于 DIN 的 4.469%。** 先前把 context/dense 重编码列为首要嫌疑的判断没有得到这四组支持。以下第 1–6 节保留的是验证前的历史证据与假设，不能替代新受控结论。

新 runner 为 `code/run_controlled_token_experiments.py`。第一轮启动后发现训练 RNG 重置位于模型构造前，随后被构造用 seed 覆盖；已在 `din_random` 第 2 轮停止并写 `SUPERSEDED.json`，不作为正式证据。正式结果只取修正版目录 `/root/autodl-tmp/future_window_token_control_v2_20260920`，本地完整包、90 个文件 SHA-256、全量协议重建和统计复算均通过。

验证前的初步结论：早期实验最强的线索是商品分支的提前混合；凌晨两个新模型的共同疑点是 context/dense 的非等价重编码，以及没有隔离的初始化变化。当时只能确认固定成品模型比较落后，不能把损失全部归因于 token 化，也不能认定 Mixer 是主要原因。只切片再拼接的 raw-slice 已通过历史等价验证，本身没有失败。

**1. 已确认：损失发生在候选内排序**

| 正式模型 | 测试 HR@5 | 相对 DIN | 测试 NDCG@5 | 候选命中用户中的 HR@5 |
|---|---:|---:|---:|---:|
| DIN | 4.481% | — | 2.03855% | 38.483% |
| semantic_concat | 4.064% | -9.31% | 1.84478% | 34.902% |
| semantic_rankmixer | 4.043% | -9.77% | 1.80299% | 34.722% |

三者使用同一候选池，候选 HR@75 都是 11.644%。在 100,000 测试用户中，concat 相比 DIN 新增命中 1,401 人、丢失命中 1,818 人，净少 417 人；Mixer 净少 438 人。因此本轮差异不能用召回覆盖率下降解释。

concat−DIN 的 HR 差为 -0.417 个百分点，配对区间 [-0.530, -0.306]；Mixer−DIN 为 -0.438，区间 [-0.553, -0.326]。两者对固定 DIN 的退化有一致的用户层面证据，但区间不包含重训练随机性。

Mixer−concat 只有 -0.021 个百分点，区间 [-0.124, +0.084]；NDCG 差的区间也跨零。当前没有确认 Mixer 带来额外收益，也没有确认它带来额外损害。不能把两个模型各自落后 DIN 当成两次独立证明“token 化无效”。

证据：[正式结果](TOKEN_RANKER_RESULTS_20260920.json)、[配对统计](TOKEN_RANKER_PAIRED_20260920.json)、[验证报告](TOKEN_RANKER_VALIDATION_20260920.md)。HR@5 是至少命中一个未来目标的用户比例，不是 Precision@5。

**2. 第一优先嫌疑：等宽 token 重编码改变了特征通路和尺度**

原 DIN 把品类、品牌和 10 个统计标量直接交给排序 MLP；semantic_concat 先进行如下变换：

- context：品类 256 维 + 品牌 64 维 + 两个商品统计值，经 322→256 线性层、GELU、LayerNorm。
- dense：10 个统计标量经 10→64→256、GELU、Dropout、LayerNorm。
- 两个商品统计值同时出现在 context 和 dense。原始 context/dense 没有另行直连排序头。
- 排序头的首层输入由 1358 维变成 1536 维，虽然后续隐藏层、BatchNorm、PReLU 与 DIN 对齐，整体仍不等价。

这意味着“划分语义 token”的实验同时改变了压缩、非线性、归一化、特征重复和优化路径。语义命名清楚，不保证表示更容易被排序头利用。

其中最具体、可检验的机制是初始化尺度不均衡。商品 SVD 向量先做单位范数归一化；新模型的用户向量是这些商品向量的均值，兴趣向量是它们的 softmax 加权和。因而这三路在初始化时的范数通常不超过 1。另两路 context/dense 经默认 LayerNorm 后，256 维输出范数通常接近 sqrt(256)=16，cross 乘积通常更小。同为 256 维，并没有让六路信号处于相近尺度。

在这种初始化下，投影分支可能更强地影响首层响应和梯度分配，原有 CF 信号的相对权重变小。首层之后的 BatchNorm 归一化的是混合后的神经元输出，并不能自动恢复各输入分支的相对贡献。这是有代码依据的机制假设；本地没有真实激活、梯度或训练后范数，不能据此宣称已测得“CF 被淹没”。

此外，当前数据里的 user_verified_ratio、user_avg_helpful 恒为零，item_rating_number 与 item_click_count 完全相同。所谓 10 个 dense 字段最多只有 7 个不同的变化量，却被扩成一个完整 256 维 token。该变换增加了表示容量，没有增加观测信息；它是否有益需要结果支持。

必须避免两个过度推断：322→256 不保证保真，但有效数据维度可能很低，不能仅凭维数宣布重要信息必然丢失；LayerNorm 去掉的是中间向量的均值和尺度，前面的投影可能把原始数值编码到方向里，不能说所有数值信息都被抹掉。

代码证据：[token_models.py](../code/token_models.py) 77–116、165–188 行；[model_ext.py](../code/model_ext.py) 207–239 行；[baseline_data.py](../code/baseline_data.py) 176–187 行；[run_baseline.py](../code/run_baseline.py) 143–153 行。

**3. 已确认的混杂：初始化和选模没有对齐**

DIN 只用 SVD 初始化商品向量，用户向量随机初始化；新模型同时用训练前缀内商品向量的均值初始化用户向量。这会直接改变 user、user×item 两路信号，也可能改变模型使用用户 ID 的方式。它不是测试集泄漏，但它的收益或伤害没有被单独测量。

仅设置相同 seed 也没有保证共同参数相同。新层的创建及全模型初始化会消耗随机数，因此 attention、品牌、品类等兼容参数需要显式复制。concat 与 Mixer 的现有对比同样没有完全排除这个因素。

另一个差异是 checkpoint 选择：既有 DIN 按 10,000 开发用户、quota/100 候选条件选 epoch；新模型按 100,000 开发用户、RRF/75 候选条件选 epoch。本轮最终评估的候选池一致，选模过程却不一致。这个差异的影响方向和大小均未知。

因此，目前能确定的是“这两个训练完成的模型落后这个 DIN checkpoint”，不能给出 token 变换、用户初始化、随机初始化各自贡献了多少下降的百分比分解。

代码证据：[run_baseline.py](../code/run_baseline.py) 396–405 行；[run_token_experiments.py](../code/run_token_experiments.py) 52–75、285–289 行；[验证限制](TOKEN_RANKER_VALIDATION_20260920.md) 107–121 行。

**4. 训练目标改善没有转化成 Top-5 改善，盲目加容量缺少依据**

| 模型 | 第 1→2→3 轮训练 BPR loss | 第 1→2→3 轮开发 HR@5 |
|---|---|---|
| concat | 0.03208 → 0.01390 → 0.00616 | 4.080% → 4.235% → 4.201% |
| Mixer | 0.02828 → 0.01403 → 0.00630 | 4.073% → 4.179% → 4.185% |

concat 第三轮 loss 继续下降、开发 HR 回落。Mixer 第二到第三轮 HR 只增加 0.006 个百分点，NDCG 却从 1.90671% 降至 1.86612%。这是训练代理目标与实际排序效果脱节的直接现象。

代码中每个正样本配 4 个从训练商品集合均匀抽取的未交互负例，优化 BPR；实际评估要求在召回系统筛出的 75 个较相关候选中选前五。两个区分任务难度和分布不同，因此低训练 loss 不保证真实候选内排序好。这是 DIN 和新模型共有的限制，单靠它不能解释新模型为什么更差；它主要削弱了“再加层、加 epoch 就能补回来”的依据。

现有曲线不能证明模型容量不足，也不足以确认具体的过拟合机制。三轮固定学习率日程并不代表所有结构都已充分调优，但已没有理由把“没训练够”当成默认解释。

本轮 Mixer 只是一个残差 MLP-Mixer block，后面仍保留 concat+MLP。残差系数初始为 1e-3，起步接近未混合的 token。它既没有引入新信息，也没有改变负例和优化目标。它没有救回共同表示的下降，不等于论文 RankMixer、所有 token 交互或 MoE 都已经被否定。

代码证据：[负采样](../code/baseline_data.py) 194–226 行；[训练循环](../code/run_token_experiments.py) 285–340 行；[Mixer](../code/token_models.py) 13–42 行。

**5. 早期五路 token 的塌缩：提前混合是最强线索，但旧报告定因过度**

9 月 12 日工作汇报记录：初版 AUC 0.5105，去 cross LayerNorm 后 0.5364，关用户 warm-start 后 0.5359，去全部 LayerNorm 后 0.5502；拆开 item 分支、删除加权求和和二次交互、恢复 raw item 直连后达到 0.6745。这个组合修改提供了最强的改善线索。

合理解释是：把商品身份、上下文及二次交互提前叠加为一个表示，使下游更难单独利用原始商品信号；拆开并保留直连缓解了问题。但一次修改动了多个因素，不能区分究竟是求和、投影、二次项、尺度还是优化共同造成的。关 warm-start 没改善，也不能推导所有后续模型里 warm-start 都无影响。

旧汇报把残余 AUC 差距全部归因于随机投影，证据不足。256→256 线性投影不必然丢失信息；把相同计算从“token 定义层”挪到“mixer 层”，若计算图不变，模型也不会变好。同期评审还发现旧链路存在训练目标进入历史、AUC 跨用户比较等问题，具体旧实验受何影响需对应快照确认。历史 AUC 不能与本轮未来窗口 HR 直接比较。

尤其不能将这个解释套到凌晨模型：正式维度都是 256，user/item/sequence/cross 的投影实际为 Identity，cross 没有 LayerNorm。当前待定位的是 context/dense 变换及训练混杂。

raw-slice 则已有 CPU/CUDA logits、梯度、连续三次 AdamW 更新和 BatchNorm 状态的等价通过记录。它说明纯粹重新分组不会改变预测，既不应降低效果，也不会凭空增加效果。这与 semantic token 重编码属于不同实验。

历史依据：[工作汇报](召回精排工作汇报_2026-09-12.md) 68–105 行；[同期评审](召回精排优化评审_2026-09-12.md) 24–70、150–158 行；[raw-slice 验收记录](OPTIMIZATION_LOG_20260912.md) 142–144 行。

**6. 需要服务器时，按判别力排序验证**

先做无需完整训练的检查：加载正式 checkpoint 和少量开发样本，测每路 token 的范数/方差、排序头各分支贡献、梯度尺度以及 Mixer 学到的残差系数。它能检验尺度假设是否真实存在于样本及训练后模型，但不能单独证明因果。关闭训练后分支导致分布改变，只能作敏感性诊断，不能替代重训消融。

严格区分结构与用户初始化，最小完整交叉对照是四组：

| 组别 | 结构 | 用户初始化 |
|---|---|---|
| A | DIN / 已验证等价的 raw-slice | 随机 |
| B | DIN / 同一 raw-slice | 训练商品 SVD 均值 |
| C | semantic_concat | 与 A 相同的随机用户向量 |
| D | semantic_concat | 与 B 相同的 SVD 均值 |

固定商品初始化、召回、训练样本和顺序、负样本、训练预算、开发选模池；显式复制所有形状兼容的共同初始参数，并记录参数量和耗时。形状变化的首层也要记录初始化规则。raw-slice 的正确性用权重复制和等价检查确认，不再额外花一轮训练证明“切开再拼接不变”。

- C−A、D−B 检验不同初始化条件下的语义重编码影响。
- B−A、D−C 检验用户初始化影响及其与结构的交互。
- 如果两种初始化下 semantic_concat 均落后，优先分别改变 context、dense；对嫌疑分支单独取消 LN、调整增益或保留原始旁路，每次一个因素。
- 只有表示本身不再退化后，再比较相同 token、共同初始参数的 concat 与 Mixer。开发集稳定的方案再重复种子，用另行预留的确认集做最终结论。

如果只允许两个训练任务，可以先跑 A/C，回答“在 DIN 的初始化条件下，语义重编码是否仍退化”；它无法回答用户初始化与结构的完整交互。旧有 DIN 和旧 D 不能直接替代新的严格控制组。

初步文档归因不需要开服务器；用户现已要求量化各因素贡献，服务器受控验证正在推进。仅靠原有汇总结果无法做到这一点。

**证据边界**

本 checkout 有正式汇总 JSON、配对统计、三轮训练历史、代码与 SHA-256 清单，但没有报告所述 9 月 20 日逐用户 NPZ、真实激活和 checkpoint。本次复核了 JSON 内的指标与相对降幅，未重新计算原始 NPZ 的置信区间。token_models.py 和 run_token_experiments.py 的当前 SHA-256 均匹配正式归档清单。

没有发现足以解释本轮下降的明确实现错误：候选相关 DIN attention 在每个候选上重新计算，BPR 正负合批路径也正确。token 模型移除了候选 verified embedding，但原 DIN 在该数据链路一直取默认 0，这只是常量分支，不能解释为丢失有效购买验证信号。

run_token_experiments.py 顶部“只训练 token/fusion”的说明与实现不符：优化器实际更新全部参数，最终实验报告已正确注明。归因应以实际实现和最终报告为准。

**7. 本次四组执行协议与验收**

四组均从头训练：`din_random`、`din_svd`、`semantic_concat_random`、`semantic_concat_svd`；旧 DIN checkpoint 和旧两组 token 结果均不充当控制组。固定 `data_id=754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158`、RRF/75 候选、权重 `[2.0, 1.0, 0.7, 0.05]`、ItemCF half-life 180 天、4 个负样本、3 epoch、10 万开发用户、10 万测试用户、训练 seed 42、初始化 seed 424242。模型维度、batch 与 workers 继承同一 baseline manifest。

商品表四组显式复制同一 train-only SVD；用户表仅切换同一随机矩阵/同一训练商品 SVD 均值。品牌、品类、DIN attention、所有兼容排序头参数与 buffers 均显式复制，包含首层 bias；仅首层 weight 因 1358/1536 输入不同而不能复制。每组记录参数 SHA-256、参数量、实际训练初始 RNG 指纹及共享参数断言。

训练 seed 在建模完成后重置。相同架构的两种用户初始化共享训练随机流起点；跨架构虽也使用同一起点，token 额外的 dropout 调用意味着不能宣称两种架构的每个 dropout mask 都相同。DataLoader 使用独立的 seed+epoch generator；PrefixDataset 按同一数据 seed、epoch 和样本位置生成相同负样本。

候选池只计算一次并复用；修正后的启动可复用已完成的同 fingerprint 候选缓存，不能复用作废模型。逐用户输出增加 uid/position，配对前核对身份顺序和 pool_hit 一致，保存开发/测试记录 fingerprint。每组都按同一开发 HR@5 选择 3 轮中的最佳 checkpoint；测试指标不参与选模。默认完成评估后删除本次可再训练的 best checkpoint，保留逐用户数组、指标、历史、初始状态审计和完整日志以节省磁盘。

验收必须同时满足 monitor status=complete、exit_code=0、总 `COMPLETED.json`、包含四组的 `results.json`、配置与初始化审计正确。正常启动、监控、读取使用 Low；错误和矛盾才进入诊断。单训练 seed 的用户配对区间不覆盖训练随机性，不能自动推广为多 seed 稳健结论。

启动前，本地临时环境 CPU 与服务器 CUDA 的真实 smoke 均通过：按正式 256 维拓扑验证 226 项 tensor 相等性（含非零 bias 哨兵复制），四组各执行 2 步真实 BPR 前后向，确认首次 forward 的 CPU/CUDA/Python/NumPy RNG、batch 顺序及负例一致。另用真实排序路径核对 DIN/concat 的逐用户指标与 uid/position 数组。证据见 [TOKEN_CONTROLLED_SMOKE_20260920.json](TOKEN_CONTROLLED_SMOKE_20260920.json)；这些合成数据结果不用于质量结论。

正式启动核对：monitor PID 7884、训练 PID 7885，15 分钟 heartbeat；已确认 v2 manifest 的四组/固定条件、同一候选缓存硬链接、第一组 initial_state_audit、源码快照和两端 runner SHA-256 一致。runner SHA-256 为 `a59e98478038b1cb5e2427607a260aca8826295505e38d0823f01cd7898e8be9`。启动时磁盘剩余 3.9 GB、GPU 训练使用约 7.5 GB，无 traceback。应用已设置本任务每 15 分钟自动跟进（`token`）。用户后续明确授权：全部校验、本地归档与分析完成后关机并验证，再提交和推送 GitHub；此顺序全部完成后才暂停自动跟进。禁止追加多 seed 或 context/dense 等实验。

结果验收工具为 [compare_controlled_token_results.py](../tools/compare_controlled_token_results.py)，本地与服务器均可调用。已通过合成 fixture 验证正常输出、HR 并列时选择最早轮次、8 项配对统计和 UID 不一致时拒绝输出；未提前读取或编造正式结果。正式完成后传入上述新 run-dir 与输出路径 `controlled_paired_verified.json`，工具会核对协议、完成标记、历史/选模、共同参数哈希、两种用户初始化、CPU/CUDA RNG 起点和逐用户数组，再给出 HR 与 NDCG 的配对 95% 区间。

关机前还须运行 [audit_controlled_protocol.py](../tools/audit_controlled_protocol.py)：核对源代码/资产 SHA-256、未来窗口及 CF300 配置，重建全部三轮 DataLoader 顺序和有效训练样本的负例，核验四组分散 PrefixDataset 探针，再用两份完整候选缓存重算 pool_hit、核对实际 uid/position 和开发/测试用户不相交。这是冻结代码及配置的确定性重建，不是训练时逐批保存的原始 trace；报告明确保留此证据边界。部分依赖源码于运行中补采，来源和采集时间一并保留，不冒称全部为启动前快照。

[archive_controlled_evidence.py](../tools/archive_controlled_evidence.py) 只有在结果验收与全量协议审计通过后才打包；本地再校验包及所有成员 SHA-256，并实际读取 JSON、NPZ 与两份 10 万用户候选缓存。已通过合成归档测试及本地文件/传输压缩包损坏拒绝测试。所有检查、总结分析及本地可读确认完成前禁止关机；遇到失败保留全部证据。
