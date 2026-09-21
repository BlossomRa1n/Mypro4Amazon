# 2026-09-21 优化总结与停机验收

## 已完成

1. 在固定 10 万用户子集上完成四种负样本/损失组合：4 随机 BPR、16 随机 BPR、12 随机+4 中等候选 BPR、同一负样本集合的 sampled softmax。
2. 在同一混合负样本策略上完成 full、zero_seq、zero_user、zero_item、zero_context、zero_cross、zero_dense 七组 token 消融。
3. 所有组均完成 3 epoch、screen/test 评估、逐 epoch metrics、用户级配对数组、manifest、日志和 `COMPLETED.json`。
4. 证据包已下载本地并完成 JSON/NPZ 可读性检查；远端与本地 tar SHA-256 相同：`2f162650f43c3e2d7d5bca84693fec99346831c57ab9533d47de1ff07f83a4b8`。

## 决策

- 不进行全量训练：16 随机在 screen 下降、test 上升，未满足“两个候选策略都稳定有效”的条件；混合策略只在一个固定 20k 协议中显示小幅收益，且训练池是 train-corpus hard-negative pool，不是严格 prefix-causal。
- 当前开发方向保留 16 个负样本中的混合构成作为候选；先不要把收益写成全量收益。
- token 消融显示 dense 保留最稳定；sequence/user/item/context 去掉后 test 均下降；cross 去掉后反而上升，说明 cross 当前可能带来噪声，需在真正独立 holdout 或全量前复核。

## 限制

训练池的 ItemCF/category/hot 使用完整 train split，目标在候选排序后才过滤，训练 item dense 统计也使用 train-corpus 统计；因此结果是相对筛选实验。screen/test 各 20k，test 已用于选择判断，不能视为新鲜 holdout，也不能和之前 100k 四组架构实验直接横比。

## 安全关机验收

- 负样本筛选：`/root/autodl-tmp/token_negative_screen_20260921/COMPLETED.json` 存在。
- token 消融：`/root/autodl-tmp/token_ablation_20260921_v2/COMPLETED.json` 存在。
- 本地归档：`evidence/token_evidence_20260921.tgz` 及解压可读副本存在。
- 产物无 NaN/Traceback/失败标记；所有七组消融的用户数组均为 20,000 行。
