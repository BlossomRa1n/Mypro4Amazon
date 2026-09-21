# 负样本筛选记录（2026-09-21）

## 协议

- 固定 10 万用户子集，`data_id=d68d1f3a050d1f374b759aea2559fb30ce0bd6cbfe3b690df7d611f6ee426a93`。
- future-window；screen/test 各 20,000 用户；候选池 75；RRF 权重 `[2.0, 1.0, 0.7, 0.05]`；ItemCF 半衰期 180 天；seed 42；初始化 seed 424242；3 epoch；batch 256。
- 四个变体：4 随机+BPR、16 随机+BPR、12 随机+4 个中等候选+BPR、同一负样本集合+sampled softmax。
- 四组共用评估用户、候选池和模型模板；结果来自固定代码和确定性 seed，可事后重建。运行文件未保存逐 batch 负样本 hash，因此不能把它表述为原始 batch trace 证明。

## 结果

| 变体 | screen HR@5 | screen NDCG@5 | test HR@5 | test NDCG@5 |
|---|---:|---:|---:|---:|
| 4 随机+BPR | 0.01885 | 0.008479 | 0.01705 | 0.007473 |
| 16 随机+BPR | 0.01780 | 0.008244 | 0.01835 | 0.008216 |
| 12 随机+4 中等+BPR | 0.01900 | 0.008509 | **0.01895** | **0.008436** |
| 12 随机+4 中等+sampled softmax | 0.01920 | 0.008553 | 0.01795 | 0.008150 |

相对 4 随机基线，混合 BPR 在 screen 为 HR `+0.00015`、NDCG `+0.000030`，在 test 为 HR `+0.00190`、NDCG `+0.000963`。16 随机在 screen 下降、test 上升，稳定性不足，暂不选作全量训练方案。混合 BPR 是当前最合理候选，但提升幅度仍需在同协议下谨慎解释，不能宣称已完成全量收益证明。

## 证据边界

训练池使用完整 train split 拟合的 ItemCF/category/hot（V2 权重为 0），不是严格 prefix-causal：其他训练目标、当前目标对候选排序和统计特征有影响；目标在采样时被过滤，但过滤发生在候选排序之后。它适合相对 hard-negative 筛选，不能称为无目标泄漏的因果验证。训练 item dense 特征也使用 train-corpus 统计量。screen/test 均为 20k，并且 test 已随方案筛选被读取，不是全新 holdout；不可与 100k 受控架构实验直接横比。

服务器产物：`/root/autodl-tmp/token_negative_screen_20260921`，包含 `results.json`、`suite_manifest.json`、每轮 metrics/history、用户级配对数组、`COMPLETED.json` 和完整日志。
