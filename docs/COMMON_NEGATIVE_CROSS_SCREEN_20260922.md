# 共同负例与 cross token 验证（2026-09-22）

## 结论先行

1. 在同一行 16 个随机负例、同一前 4 个子集的配对实验中，16 个负例没有获得统计上可靠的收益。当前应继续使用 4 个负例作为成本较低的基线。
2. raw cross 改成 normalized cross 的结果不稳定：screen 略升，test 略降，不能接受为确定改进。
3. normalized cross 加初始约 0.05 的门控后，test HR@5 从 0.01830 升到 0.02000，配对 bootstrap 95% CI 为 [0.00035, 0.00310]；test NDCG@5 从 0.008319 升到 0.008951，CI 为 [0.000001, 0.001270]。这是本轮最值得继续验证的方向，但仍需要多 seed 和严格时间因果评估后才能作为正式方案。

## 受控条件

- 服务器：`connect.westc.seetacloud.com:54579`，运行目录 `/root/autodl-tmp/common_cross_screen_20260922`。
- `data_id=d68d1f3a050d1f374b759aea2559fb30ce0bd6cbfe3b690df7d611f6ee426a93`。
- future-window 协议；候选池 75；RRF；权重 `[2.0, 1.0, 0.7, 0.05]`；ItemCF 半衰期 180 天。
- 固定 seed 42、初始化 seed 424242、3 epoch、同一训练顺序、同一 screen/test 用户各 20,000。
- 共同随机负例池形状 `(572569, 16)`；4 负例严格使用每行前 4 个，16 负例使用同一行全部 16 个。池 SHA-256：`6bc679ddd1619573d7fa0a00a611339c62bb6811e0bd843ee7bfc9ca2951bae`。
- 五个变体：`random4_common`、`random16_common`、`cross_raw`、`cross_normalized`、`cross_gated_normalized`。
- cross normalized 使用固定 0.32 缩放；gated normalized 以 `sigmoid(-2.944439)≈0.05` 启动。该 0.32 是初始化尺度匹配假设，不是从本轮结果拟合出来的参数。

## 最终指标

| 变体 | screen HR@5 | screen NDCG@5 | test HR@5 | test NDCG@5 |
|---|---:|---:|---:|---:|
| 4 随机 + BPR | 0.01855 | 0.008692 | 0.01770 | 0.007975 |
| 16 随机 + BPR | 0.01890 | 0.009019 | 0.01800 | 0.008207 |
| cross raw | 0.01900 | 0.008509 | 0.01895 | 0.008436 |
| cross normalized | 0.01965 | 0.008893 | 0.01830 | 0.008319 |
| cross normalized + gate | 0.02000 | 0.009014 | 0.02000 | 0.008951 |

候选池本身五组完全相同：screen HR@75=0.05735，test HR@75=0.05360。因此差异来自精排训练与 cross 结构，而不是召回候选变化。

## 配对比较

配对单位是相同用户的最终用户级 `hit5`、`ndcg5`；UID、候选位置和所有数值均通过校验，screen/test 各 20,000 用户。

| 比较（后者减前者） | screen HR 差异（95% CI） | test HR 差异（95% CI） | 判断 |
|---|---:|---:|---|
| 16 随机 − 4 随机 | +0.00035（−0.00115, +0.00180） | +0.00030（−0.00120, +0.00175） | 无确定收益 |
| normalized − raw | +0.00065（−0.00070, +0.00200） | −0.00065（−0.00200, +0.00070） | 不稳定 |
| gated normalized − normalized | +0.00035（−0.00110, +0.00180） | +0.00170（+0.00035, +0.00310） | test 有利，需复验 |

## 解释与限制

- 16 个负例的点估计为正，但提升只有约 0.0003 HR，远小于用户级波动；因此不能据此把训练成本增加约四倍视为值得。
- normalized 只做尺度匹配，单独没有稳定收益；小门控让模型先依赖稳定 token，再逐步使用 cross，test 结果支持这个方向。
- 本轮 cross 变体使用既有 mixed negative 训练池；训练池统计来自完整 train split，仍属于探索性 screen，不能称严格 prefix-causal。test 已参与过历史方向判断，也不是全新 holdout。
- 本轮不保留 checkpoint；保留了 results、每 epoch history、screen/test 用户级 NPZ、manifest、日志、固定负例池和 SHA-256 清单。
- 结论只基于单 seed。正式采纳前应至少用预注册的多 seed 复验 gated normalized，并记录 gate 最终值。

## 证据位置

- `evidence/common_cross_screen_20260922/results.json`
- `evidence/common_cross_screen_20260922/paired_analysis.json`
- `evidence/common_cross_screen_20260922/suite_manifest.json`
- `evidence/common_cross_screen_20260922/train.log`
- `evidence/common_cross_screen_20260922/common_random16.npy`
- `evidence/common_cross_screen_20260922/files.sha256`

