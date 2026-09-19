# 精排 Token 验证记录（2026-09-20）

## 目标

固定未来窗口协议、固定候选池和召回配置，比较既有 DIN 与新训练的语义 token 精排模型对 Top-5 的实际表现。召回不在本轮调参。本轮初始化和选 epoch 条件存在差异，因此不是严格的单变量结构消融，详见末尾限制。

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
2. `semantic_rankmixer`：同一套六个 token 构造，在 concat + MLP 前增加一个残差 MLP-Mixer block。类名为 `RankMixerBlock`，本轮不声称复现论文 RankMixer。

token 顺序固定为：`sequence / user / item / context / cross / dense`。DIN target attention 对每个候选单独计算；没有把候选相关兴趣表示错误缓存为用户常量。

## 代码改动

- `code/token_models.py`
  - 新增 `SemanticTokenDIN`。
  - 新增固定 token 数的 `RankMixerBlock`。
  - 保留原始 `user_emb * item_emb` cross 信号，不对 cross token 做 LayerNorm。
  - concat 变体的隐藏层宽度、归一化和激活与 `DINExtendedModel` 对齐为 `Linear + BatchNorm + PReLU + Dropout`；首层输入从 1358 维变为 1536 维，参数量并不相同。
- `code/run_token_experiments.py`
  - 新增未来窗口 token 实验入口。
  - 同一候选池上重测现有 DIN baseline。
  - 保存开发/测试 Top-5、NDCG、Recall、候选覆盖和逐用户配对比较。
  - 每个变体重新初始化模型，用 train-only SVD 初始化商品向量，并用用户训练前缀内商品 SVD 均值初始化用户向量；训练更新全部模型参数。
- `tools/auto_shutdown_token_suite.sh`
  - 运行完整套件。
  - 只有写入 `COMPLETED.json` 后才请求关机。
- 监控：`tools/monitor_training.py --poll-seconds 900`。
- 本地结果捕获：`tools/watch_token_suite_archive.ps1` 每 900 秒等待本次证据包，下载、逐文件校验 SHA-256 后回传归档确认。SCP 使用大写 `-P` 指定端口，并检查原生命令退出码。
- 收尾保护：`tools/finalize_token_archive.py` 仅暂停等待训练的 shell 启动器，训练进程继续运行；每 900 秒确认训练结束后核对完整结果、打包 JSON/NPZ/日志，等待本地确认（最多 900 秒）后恢复原启动器关机。模型 checkpoint 保留原位置。

## Smoke 验证

服务器小样本目录：`/root/autodl-tmp/token_smoke_20260920`，已归档至 `server_snapshot/2026-09-20/token_smoke_20260920/` 后删除服务器临时权重。

- 300 开发用户 + 300 测试用户，1 epoch，2 个训练 step，`token_dim=64`。
- 两个变体均完成前向、BPR、候选重排、Top-5 指标和 paired comparison。
- 该 smoke 只验证链路，不用于模型结论。
- 临时权重占用约 4.7 GiB，正式套件已移除 `last.pth`，只保留每个变体的 best checkpoint。

## 正式运行

- 服务器目录：`/root/autodl-tmp/future_window_token_suite_20260920`
- 托管监控 PID：`48902`
- 启动器 PID：`48903`
- 训练子进程：`48905`
- 监控频率：900 秒
- 启动时间：2026-09-19 18:01 UTC
- 当前状态：`semantic_concat` 已完成 3 epoch，`semantic_rankmixer` 正在第 3 epoch，尚未生成 `COMPLETED.json`。

第一次正式启动在候选池构建阶段被主动停止：发现 concat 头误用了 LayerNorm/GELU，未进入训练，未产生可用结果；修复后已从空目录重新启动，旧缓存和权重均已删除。

正式结果完成后追加：各变体 screen/test 指标、相对固定 baseline 的 paired HR@5 区间、NDCG 差值、最终保留方案和关机时间。

### 固定 baseline 同池重测

- 开发 100,000 用户：候选池 HR@75 `11.723%`，DIN HR@5 `4.669%`，NDCG@5 `2.12691%`。
- 独立测试 100,000 用户：候选池 HR@75 `11.644%`，DIN HR@5 `4.481%`，NDCG@5 `2.03855%`。
- 两组结果与 2026-09-19 候选预算 75 快照一致，说明本轮候选池和评估协议未漂移。

### 已完成的 semantic_concat

- 开发集逐 epoch HR@5：epoch 1 `4.080%`，epoch 2 `4.235%`，epoch 3 `4.201%`；按开发集 HR@5 选择 epoch 2。
- 测试集最终 HR@5 `4.064%`，NDCG@5 `1.84478%`；候选池仍为 HR@75 `11.644%`。
- 该结果低于同池 baseline，暂不作为替代方案；最终结论需等待 `semantic_rankmixer` 的完整 screen/test paired 结果。

## 结果解释限制与下一轮要求

- **初始化不一致**：原 DIN 只用 SVD 初始化商品 embedding；新变体同时初始化用户 embedding。当前对比支持固定模型之间的实测优劣，不支持把全部差异归因于 token 化。
- **选模条件不一致**：既有 baseline 按旧的 10,000 开发用户、quota/100 候选条件选 epoch；新变体按本轮 100,000 开发用户、RRF/75 候选选 epoch。
- **输入不只是等长重排**：context/dense 新增投影和归一化，两项商品质量特征同时出现在 context 和 dense；不能称为纯 raw-slice 对照。
- **MLP-Mixer 实现范围**：沿 token 和通道进行 MLP 混合，仍保留后续 concat + MLP；不是未经核对的论文 RankMixer 复现。
- **统计范围**：只有 seed=42，逐用户 paired 区间不包含重训练随机性，也未做多重比较校正。开发集被用于选 epoch，其区间只作描述；测试集不参与本轮 epoch 选择，但历史上已被多轮实验查看，不应再称为首次使用的全新 holdout。
- **指标含义**：HR@5 是 Top-5 中至少命中一个未来商品的用户比例，不是五个推荐位置的 Precision@5。

下一轮仍冻结召回，先将 DIN / raw-slice concat / semantic concat 的初始化、训练预算、开发集选模和 head 参数预算对齐；再为入围 token 方案加 MLP-Mixer 做直接配对，最后对开发集稳定获益方案复跑多个 seed。新实验不在本轮自动关机后自行启动。

## 收尾改动与验证

- 修复本地归档脚本复用 SSH 小写 `-p` 给 SCP 的端口错误、遗漏 NPZ/mixer 最终指标、未检查下载失败以及下载与立即关机的竞态。
- 已终止本轮旧的 10 秒和 60 秒本地轮询，统一为 900 秒。服务器训练监控原本已是 900 秒。
- 归档程序通过合成文件验证：证据 SHA-256 一致、排除 checkpoint、拒绝不匹配完成标记。模型训练未因此重启。
- 收尾程序首次尝试使用 pidfd 等待进程，在服务器 Python/内核环境不兼容；两次尝试均在发送任何进程信号前退出。最终使用 900 秒进程状态检查，收尾 PID 为 `58211`。
- 新增 `tools/compare_token_results.py`：在每个 split 内核对数组长度、候选命中掩码和指标均值，计算 concat/baseline、mixer/baseline、mixer/concat 的 HR 配对 bootstrap 及 NDCG 配对正态近似区间。6 用户合成数据的已知 gain/loss、NDCG 区间与 6 类错误拒绝路径均通过。
- 本地新版启动器已接入“训练完成 → 同步打包/等待归档 → 关机”，供未来启动直接使用；本次正在运行的旧 shell 文件未热修改，由独立收尾进程保护。
- 归档宽限有 900 秒上限：若客户端离线，服务器保留证据包后仍会关机，避免无限闲置；本轮收尾需实际核对本地归档成功。包内 `monitor_status.json` 是打包时快照，最终关机状态要另行核对。
