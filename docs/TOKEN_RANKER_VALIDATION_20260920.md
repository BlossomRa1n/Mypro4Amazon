# 精排 Token 验证记录（2026-09-20）

## 目标

固定未来窗口协议、固定候选池和召回配置，只验证 DIN 精排输入 token 化及 RankMixer 融合头对 Top-5 的影响。召回不在本轮调参。

固定配置：

- 候选预算：75
- ItemCF 邻居：300（来源 `/root/autodl-tmp/future_window_cf300_dev_20260919`）
- 融合：RRF
- ItemCF 半衰期：180 天
- 融合权重：`[2.0, 1.0, 0.7, 0.05]`
- 评估：未来窗口，100,000 开发用户 + 100,000 独立测试用户
- 训练：4 个负样本，3 epoch，SVD 仅作为共同 CF warm-start

## 本轮变体

1. `semantic_concat`：六个 256 维语义 token，融合头保持 concat + MLP。
2. `semantic_rankmixer`：完全相同的六个 token，只将融合头替换为一个残差 RankMixer block。

token 顺序固定为：`sequence / user / item / context / cross / dense`。DIN target attention 对每个候选单独计算；没有把候选相关兴趣表示错误缓存为用户常量。

## 代码改动

- `code/token_models.py`
  - 新增 `SemanticTokenDIN`。
  - 新增固定 token 数的 `RankMixerBlock`。
  - 保留原始 `user_emb * item_emb` cross 信号，不对 cross token 做 LayerNorm。
  - concat 变体的最终头与 `DINExtendedModel` 对齐为 `Linear + BatchNorm + PReLU + Dropout`。
- `code/run_token_experiments.py`
  - 新增未来窗口 token 实验入口。
  - 同一候选池上重测现有 DIN baseline。
  - 保存开发/测试 Top-5、NDCG、Recall、候选覆盖和逐用户配对比较。
  - 每个变体重新初始化精排头，只复用共同的 train-only SVD warm-start。
- `tools/auto_shutdown_token_suite.sh`
  - 运行完整套件。
  - 只有写入 `COMPLETED.json` 后才请求关机。
- 监控：`tools/monitor_training.py --poll-seconds 900`。

## Smoke 验证

服务器小样本目录：`/root/autodl-tmp/token_smoke_20260920`，已归档至 `server_snapshot/2026-09-20/token_smoke_20260920/` 后删除服务器临时权重。

- 300 开发用户 + 300 测试用户，1 epoch，2 个训练 step，`token_dim=64`。
- 两个变体均完成前向、BPR、候选重排、Top-5 指标和 paired comparison。
- 该 smoke 只验证链路，不用于模型结论。
- 临时权重占用约 4.7 GiB，正式套件已移除 `last.pth`，只保留每个变体的 best checkpoint。

## 正式运行

- 服务器目录：`/root/autodl-tmp/future_window_token_suite_20260920`
- 托管监控 PID：`48499`（启动时记录）
- 训练子进程：`48502`（启动时记录）
- 监控频率：900 秒
- 启动时间：2026-09-19 17:55 UTC
- 当前状态：运行中，尚未生成 `COMPLETED.json`

第一次正式启动在候选池构建阶段被主动停止：发现 concat 头误用了 LayerNorm/GELU，未进入训练，未产生可用结果；修复后已从空目录重新启动，旧缓存和权重均已删除。

正式结果完成后追加：各变体 screen/test 指标、相对固定 baseline 的 paired HR@5 区间、NDCG 差值、最终保留方案和关机时间。

### 固定 baseline 同池重测

- 开发 100,000 用户：候选池 HR@75 `11.723%`，DIN HR@5 `4.669%`，NDCG@5 `2.12691%`。
- 独立测试 100,000 用户：候选池 HR@75 `11.644%`，DIN HR@5 `4.481%`，NDCG@5 `2.03855%`。
- 两组结果与 2026-09-19 候选预算 75 快照一致，说明本轮候选池和评估协议未漂移。
