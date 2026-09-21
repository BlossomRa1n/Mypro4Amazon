# Token 消融记录（2026-09-21）

## 协议

固定 semantic_concat、随机用户初始化、12 随机 + 4 中等候选负样本、BPR、10 万用户子集、20k screen/test、75 候选、RRF `[2.0,1.0,0.7,0.05]`、ItemCF 半衰期 180 天、seed 42、3 epoch。每次保持 6 个 token 槽位和参数宽度不变，在加入 token type offset 后将一个槽位置零；因此“zero”同时移除了该槽位的内容和该槽位的 type offset。

## 结果

| 变体 | screen HR@5 | screen NDCG@5 | test HR@5 | test NDCG@5 |
|---|---:|---:|---:|---:|
| full | 0.01900 | 0.008509 | 0.01895 | 0.008436 |
| zero_seq | 0.01830 | 0.008205 | 0.01740 | 0.008003 |
| zero_user | 0.01885 | 0.008781 | 0.01725 | 0.007884 |
| zero_item | 0.01865 | 0.008609 | 0.01795 | 0.008176 |
| zero_context | 0.01865 | 0.008421 | 0.01750 | 0.007794 |
| zero_cross | **0.02000** | **0.009408** | **0.01940** | **0.008768** |
| zero_dense | 0.01755 | 0.007932 | 0.01655 | 0.007560 |

相对 full，dense 消融在 screen/test 都明显下降，说明 dense 源在当前实现中贡献最稳定；sequence、user、item、context 消融在 test 均下降，sequence/user/context 的下降更大。cross 消融反而上升（test HR `+0.00045`、NDCG `+0.000332`），所以当前 cross token 可能引入噪声或与其它 token 产生不利交互；这不是“cross 必然无用”的证明，需在全量或严格独立 holdout 上复核。

## 证据边界

该实验与负样本筛选共享 train-corpus hard-negative pool：ItemCF/category/hot 用完整 train split 拟合，item dense 统计也来自 train corpus；它不是严格 prefix-causal。screen/test 各 20k，test 已用于方案筛选，不能视作新鲜 holdout。消融结果只在本协议内支持“dense 保留、cross 需复核”的决策。

服务器产物：`/root/autodl-tmp/token_ablation_20260921_v2`；本地归档在 `evidence/20260921/token_ablation_20260921_v2`，包含 results、所有 epoch metrics、用户级配对数组、manifest、COMPLETED 和日志。
