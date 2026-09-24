# 下一轮 Cross 结果与收尾分析（2026-09-24）

本报告记录 `next-cross-multiseed-20260924-v3-month-scope` 的正式九组开发结果、独立统计复算和冻结的收尾分支。它只解释已经封存的 screen/confirm 数组，不添加调参或新实验。完整 dev/protocol 双树和 compare/lock 控制证据均已在本地归档。

## 结论

两个 challenger 均未满足六条预注册开发接受门槛，因此冻结选择为 `no_winner`，`winner=null`，锁定计划为 `models=["raw"]`、`final_evaluation_allowed=false`。按协议跳过资产迁移、全量训练和最终 test；17,296 名本月时间范围内未参与评估或选型的 fresh 用户仍未被最终阶段使用。本报告没有 final-test 指标，也不把 no_winner 写成 raw 优越性或模型等效性证明。

## 固定范围与证据

- 协议：`next-cross-multiseed-20260924-v3-month-scope`，manifest SHA-256 `dcf65b10c266ba02b00d0493815a14b8af9aa4f47551d7c562a2010104409667`。
- 月度范围：Asia/Shanghai 的 `[2026-09-01 00:00:00, 2026-09-24 01:54:44)`，范围 SHA-256 `808016bf27329ad69c54e0b7377ae5fae9a3168a12c1ecbb1f5d275eaff53891`；历史前缀训练允许。
- 九组：`raw`、`normalized_gated`、`zero_cross` × seeds 42/43/44，epoch 3；screen 20,000，confirm 80,000。三 seed 的同一用户差值先取均值，再对 80,000 个用户 bootstrap 10,000 次。
- 正式 test cohort 为 17,296 人；有序 raw-ID 集合 SHA-256 `878750379c1c23082ddbcb1c7d74147318752cfc96735921e596d69e4052d6bc`。
- 冻结 runner SHA-256 `ec8df470eab5bc3bb39978371c9d086b9a1e834b2be63dc72648616ffa9d1e22`。完整回执 SHA-256 `1a12e5999e89b78e74ac385d0f060cb5047e438cad46aaa80d098ba4a6ecd17b`。
- 冻结 compare 的 selection 文件 SHA-256 `543d6a877492522146fe31e75849b5380ff408238861263fb5c3e8ca6029c11d`；lock plan 文件 SHA-256 `bdc709d4f976d3b56ce10e04b72ae7f328b568d9790055377c157ab16ed242e2`。

## 独立统计复算

复算脚本为 [recompute_statistics.py](/Users/admin/projects/分布式训练/AI/Mypro4Amazon/server_snapshot/next_cross_20260924/statistical_audit/recompute_statistics.py)，结果回执为 [dev_statistics.json](/Users/admin/projects/分布式训练/AI/Mypro4Amazon/server_snapshot/next_cross_20260924/statistical_audit/dev_statistics.json)，脚本 SHA-256 `9bd34d94debaea8634ac1c40f60a2eb0ff3f1faada2730fe8befa4ed5d23ebab`，结果回执 SHA-256 `69662d9327a9402eb9d2f1c73dac514fa07c20972d1ef487f36786eb859ae1e4`。

独立实现逐项验证了 UID、position、pool_hit 的成对身份，检查二值 hit/pool、NDCG 范围、用户数量、有限值和三 seed 的 cohort 一致性。使用 float64、PCG64、10,000 次用户 bootstrap 和线性插值百分位数；独立分块大小为 100，冻结实现为 250，但随机流与区间端点完全一致。两个 challenger 的全部 challenge 字段与冻结 selection 完全相等，`exact_challenges_equal=true`；独立审计没有模型评分，`model_evaluations=0`。

### 汇总

差值是 challenger − raw；正值才有利于 challenger。CI 是开发阶段 97.5% 双侧区间，端点为 0.0125 和 0.9875。

| challenger | screen HR Δ | screen NDCG Δ | confirm HR Δ | confirm HR 97.5% CI | confirm NDCG Δ | confirm NDCG 97.5% CI | 接受 |
|---|---:|---:|---:|---:|---:|---:|---|
| normalized_gated | -0.0014833 | -0.0011379 | -0.0023208 | [-0.0030917, -0.0015374] | -0.0015169 | [-0.0019154, -0.0011301] | 否 |
| zero_cross | -0.0022833 | -0.0014384 | -0.0015917 | [-0.0023959, -0.0007958] | -0.0011340 | [-0.0015257, -0.0007419] | 否 |

HR 差值换算为百分点分别为 screen `-0.1483` 与 `-0.2283`，confirm `-0.2321` 与 `-0.1592`。两者 confirm HR 区间整体低于零；两者 confirm NDCG 点估计也低于零。

### 各 seed 的成对结果

| challenger | seed | screen HR Δ | screen NDCG Δ | screen gained/lost | confirm HR Δ | confirm NDCG Δ | confirm gained/lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| normalized_gated | 42 | -0.000250 | -0.0004746 | 277 / 282 | -0.001075 | -0.0006891 | 1031 / 1117 |
| normalized_gated | 43 | -0.002750 | -0.0017062 | 260 / 315 | -0.0033875 | -0.0018281 | 992 / 1263 |
| normalized_gated | 44 | -0.001450 | -0.0012328 | 260 / 289 | -0.002500 | -0.0020335 | 954 / 1154 |
| zero_cross | 42 | -0.002150 | -0.0012145 | 285 / 328 | -0.0019875 | -0.0012161 | 1019 / 1178 |
| zero_cross | 43 | -0.003850 | -0.0025386 | 233 / 310 | -0.0013000 | -0.0009257 | 1044 / 1148 |
| zero_cross | 44 | -0.000850 | -0.0005622 | 260 / 277 | -0.0014875 | -0.0012603 | 1029 / 1148 |

两种 challenger 在每个 seed 的 screen 和 confirm HR/NDCG 差值均为负；confirm 中 lost 都多于 gained。confirm HR 的 seed 间标准差为 `0.0011666`（normalized_gated）和 `0.0003554`（zero_cross），但这种 seed 波动没有改变全部 seed 为负的方向。

### 六条门槛

| 门槛 | normalized_gated | zero_cross |
|---|---|---|
| screen 平均 HR > 0 | 失败 | 失败 |
| screen 平均 NDCG ≥ 0 | 失败 | 失败 |
| 每个 confirm seed HR > 0 | 失败 | 失败 |
| confirm 平均 HR ≥ 0.0005 | 失败 | 失败 |
| confirm HR 97.5% CI 下界 > 0 | 失败 | 失败 |
| confirm 平均 NDCG ≥ 0 | 失败 | 失败 |

## 冻结选择与未执行阶段

冻结 compare 和 lock 均以 exit 0 完成，且 lock 重新验证 selection。最终计划绑定了同一 protocol、历史闭合证明、代码与 base 资产、17,296 人 test cohort 和 `test_future_labels_read=false`。由于没有 accepted challenger：

- 未执行迁移；
- 未执行 full final train；
- 未创建或完成 `TEST_STARTED.json`；
- 未生成 final candidate cache、final 用户数组或 final test 结果；
- 仅保留 raw 作为计划中的模型，停止在 dev 结果审查与证据收尾。

该分支保留完整 dev/protocol 双树、九个 checkpoint、screen/confirm 数组、compare/lock 原始日志、selection/plan、base 五项资产、历史闭合证明和监控/退出回执。完整归档回执已验证九组、confirm 完成且 `test_accessed=false`；独立统计复算不重复哈希 checkpoint，而依赖这份完整归档封印。

## 优化总结

本轮固定设计的两个 challenger 在三 seed、两个开发切分和两个指标上都低于 raw；因此没有可以安全晋级的优化方案。结果支持停止本轮变体和保留 raw 基线，不能支持“raw 已被证明更优”这一超出本次门槛的结论，也不能把未执行的 final test 当作缺失的模型比较。

如果未来要继续优化，应先把问题写成新的、独立的协议，再开始新的 cohort 和评估：

1. 先分析已封存的 gained/lost 用户、seed 差异和召回/排序链路，定位负向差值来自覆盖损失、排序损失还是变体 gate；这一步只使用现有证据，不回看 fresh test 标签。
2. 对候选策略、阈值或 gate 做单独预注册，明确新的训练/验证切分、最小收益、CI 和停止条件；不能根据本次负结果事后放宽 `0.0005` 或 CI 门槛。
3. 保留 paired 用户级分析和“三 seed 先平均、再按用户 bootstrap”的统计口径，避免把 seed-user 行误当成独立样本。
4. 继续把候选池、evaluator 和 full artifact 校验作为工程前置；先通过资源与证据门禁，再消耗新的用户 cohort。

这些是后续设计建议，不是本轮新实验、调参或效果声明。

## 局限与证据边界

本报告的真实统计只覆盖保存的 20,000/80,000 用户级数组和冻结 dev 协议。它不估计未使用 fresh cohort 上的效果，不替代 final test，也不证明等效性。独立统计结果的时间戳为 `2026-09-23T23:38:23.826348+00:00`（UTC）；数组清单、每个文件 SHA 和逐字段结果均保存在统计回执中。任何后续关机和外部运行状态以主任务最后的服务器/monitor 证据为准。

## 收尾状态

外部运行证据归档已完成并做第二次远端 inventory 比对：`closure_external/` 共 51 个文件、259,194 字节；回执 `closure_external_receipts/EXTERNAL_ARCHIVE.json`，回执 SHA-256 见本地 state。远端已确认 runner、monitor、归档、迁移进程均不存在，未发现 `TEST_STARTED.json`、`FINAL_TRAIN_COMPLETED.json` 或 final 结果目录。no_winner 分支因此满足安全关机条件；服务器随后按授权关机，独立 SSH 断连结果记录在操作回执中。
