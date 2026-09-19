# 精排 Token 验证记录（2026-09-20）

截至 2026-09-20 05:15（北京时间）：本轮两个变体已完成训练、评估和归档，均未超过既有 DIN，保留 baseline。服务器已执行有效关机命令，05:15:20 SSH 返回连接拒绝。本地报告和证据已齐全，下一轮先做初始化/选模一致的 raw-slice 对照。

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
- 当前状态：两个变体均完成 3 epoch 和最终 screen/test 评估，`COMPLETED.json` 为 `complete`；证据包已下载并逐文件验证 SHA-256。

第一次正式启动在候选池构建阶段被主动停止：发现 concat 头误用了 LayerNorm/GELU，未进入训练，未产生可用结果；修复后已从空目录重新启动，旧缓存和权重均已删除。

最终完整指标见 `docs/TOKEN_RANKER_RESULTS_20260920.json`，补充配对统计见 `docs/TOKEN_RANKER_PAIRED_20260920.json`。

### 固定 baseline 同池重测

- 开发 100,000 用户：候选池 HR@75 `11.723%`，DIN HR@5 `4.669%`，NDCG@5 `2.12691%`。
- 独立测试 100,000 用户：候选池 HR@75 `11.644%`，DIN HR@5 `4.481%`，NDCG@5 `2.03855%`。
- 两组结果与 2026-09-19 候选预算 75 快照一致，说明本轮候选池和评估协议未漂移。

### 已完成的 semantic_concat

- 开发集逐 epoch HR@5：epoch 1 `4.080%`，epoch 2 `4.235%`，epoch 3 `4.201%`；按开发集 HR@5 选择 epoch 2。
- 测试集最终 HR@5 `4.064%`，NDCG@5 `1.84478%`；候选池仍为 HR@75 `11.644%`。
- 该结果低于同池 baseline，本轮不作为替代方案。

### 最终完整结果与决定

| 模型 | 开发 HR@5 | 测试 HR@5 | 测试 NDCG@5 | 测试 Recall@5 | 最佳 epoch |
|---|---:|---:|---:|---:|---:|
| 既有 DIN baseline | 4.669% | **4.481%** | **2.03855%** | **2.62890%** | 来源 checkpoint |
| semantic_concat | 4.235% | 4.064% | 1.84478% | 2.39562% | 2 |
| semantic_rankmixer（MLP-Mixer） | 4.185% | 4.043% | 1.80299% | 2.34247% | 3 |

三个模型测试候选 HR@75 均为 11.644%，开发候选 HR@75 均为 11.723%。最终保留既有 DIN；两种新变体本轮均不入选。该决定不等于否定 token 化方向，归因限制见下节。

按 100,000 测试用户直接配对，以下差值和区间单位均为**百分点**，方向为前者减后者：

| 比较 | HR@5 差值 | HR 配对 bootstrap 95% 区间 | 新增命中 / 丢失命中用户 | NDCG@5 差值 | NDCG 配对正态近似 95% 区间 |
|---|---:|---:|---:|---:|---:|
| concat − baseline | -0.417 | [-0.530, -0.306] | 1401 / 1818 | -0.19376 | [-0.25011, -0.13741] |
| mixer − baseline | -0.438 | [-0.553, -0.326] | 1424 / 1862 | -0.23556 | [-0.29222, -0.17890] |
| mixer − concat | -0.021 | [-0.124, +0.084] | 1385 / 1406 | -0.04179 | [-0.09268, +0.00909] |

两种新模型对 baseline 的区间均为负；mixer 对 concat 的测试 HR 和 NDCG 区间均跨零，没有确认额外收益。以上是单 seed 固定模型的用户抽样描述，不含多重比较校正或重训练随机性。

- mixer 开发 HR@5 三个 epoch：4.073%、4.179%、4.185%；按既定 HR 优先规则选 epoch 3。其 epoch 3 NDCG 比 epoch 2 下降，进一步说明没有稳定的整体提升。
- 三轮训练及逐 epoch 开发评估合计：concat 67.66 分钟，mixer 70.69 分钟；不含候选池构建、模型读写和最终重评，不作严格吞吐基准。
- 已逐项核对本地复算与服务器的 gained/lost 和 HR 差值。服务器 NumPy 2.5.2、本地 2.4.6，mixer/baseline 的 bootstrap 下界分别为 -0.552025 与 -0.553 个百分点，端点相差不到 0.001 个百分点；原始输出均保留，结论一致。上表使用本地统一复算结果。
- 正式证据路径：`server_snapshot/2026-09-20/token_suite_final_20260920/`，含完整 JSON、逐用户 NPZ、history、日志与 SHA-256 清单。smoke 结果未混入正式统计。
- 原 DIN 和两个变体 best checkpoint 保留服务器原位置；未因本轮落选删除 checkpoint。本地保存其校验和 `checkpoints_and_code.sha256`。
- 证据包 SHA-256：`69e181016a54bc01a98a8eeb87bf3eafa5d43864f507d8af882c651017f8c52c`。

## 结果解释限制与下一轮要求

- **初始化不一致**：原 DIN 只用 SVD 初始化商品 embedding；新变体同时初始化用户 embedding。当前对比支持固定模型之间的实测优劣，不支持把全部差异归因于 token 化。
- **选模条件不一致**：既有 baseline 按旧的 10,000 开发用户、quota/100 候选条件选 epoch；新变体按本轮 100,000 开发用户、RRF/75 候选选 epoch。
- **输入不只是等长重排**：context/dense 新增投影和归一化，两项商品质量特征同时出现在 context 和 dense；不能称为纯 raw-slice 对照。
- **MLP-Mixer 实现范围**：沿 token 和通道进行 MLP 混合，仍保留后续 concat + MLP；不是未经核对的论文 RankMixer 复现。
- **统计范围**：只有 seed=42，逐用户 paired 区间不包含重训练随机性，也未做多重比较校正。开发集被用于选 epoch，其区间只作描述；测试集不参与本轮 epoch 选择，但历史上已被多轮实验查看，不应再称为首次使用的全新 holdout。
- **指标含义**：HR@5 是 Top-5 中至少命中一个未来商品的用户比例，不是五个推荐位置的 Precision@5。

下一轮仍冻结召回：

1. 统一 DIN / token 模型的初始化策略，并显式复制形状兼容的共同初始参数；仅设置相同 seed 不能保证新增层之后的随机参数相同。训练预算、数据顺序和开发集选模池保持一致，记录各模型参数量及耗时。
2. 添加 raw-slice 保真对照：将原 1358 维输入补零至 1536 维再 reshape 为 6×256。先验证 slice 后直接 concat 的数值等价性，再比较 semantic concat 的投影/归一化是否损失信息。
3. 给入围 token 方案加 MLP-Mixer，直接与相同 token 的 concat 比较；不要同时改变 token 方案与融合结构。
4. 仅对开发集稳定获益的方案重复多个 seed；使用另行预留的确认集评估最终效果，避免反复用当前已看过的测试结果调参。

新实验不在本轮自动关机后自行启动。

## 收尾改动与验证

- 修复本地归档脚本复用 SSH 小写 `-p` 给 SCP 的端口错误、遗漏 NPZ/mixer 最终指标、未检查下载失败以及下载与立即关机的竞态。
- 已终止本轮旧的 10 秒和 60 秒本地轮询，统一为 900 秒。服务器训练监控原本已是 900 秒。
- 归档程序通过合成文件验证：证据 SHA-256 一致、排除 checkpoint、拒绝不匹配完成标记。模型训练未因此重启。
- 收尾程序首次尝试使用 pidfd 等待进程，在服务器 Python/内核环境不兼容；两次尝试均在发送任何进程信号前退出。最终使用 900 秒进程状态检查，收尾 PID 为 `58211`。
- 新增 `tools/compare_token_results.py`：在每个 split 内核对数组长度、候选命中掩码和指标均值，计算 concat/baseline、mixer/baseline、mixer/concat 的 HR 配对 bootstrap 及 NDCG 配对正态近似区间。6 用户合成数据的已知 gain/loss、NDCG 区间与 6 类错误拒绝路径均通过。
- 新版启动器已接入“训练完成 → 同步打包/等待归档 → 关机”，供未来启动直接使用；本次旧 shell 存活期间没有热修改该文件，旧 shell 退出后才同步新版脚本到服务器，并通过 `bash -n`。
- 归档宽限有 900 秒上限：若客户端离线，服务器保留证据包后仍会关机，避免无限闲置；本轮收尾需实际核对本地归档成功。包内 `monitor_status.json` 是打包时快照，最终关机状态要另行核对。
- 2026-09-20 05:13:40（北京时间）归档确认后释放启动器，但旧脚本的 `/sbin/shutdown` 不存在，实际首次关机失败。已改为启动时用 `command -v shutdown` 解析可执行文件；本服务器对应 `/usr/bin/shutdown`。该故障发生在全部训练和归档完成之后，不影响模型结果。
- 05:15 左右执行 `/usr/bin/shutdown -h now` 返回成功；05:15:20 SSH 返回 `Connection refused`，完成关机核验。证据位于本地快照中的 `shutdown_confirmation.log`、`shutdown_followup.log`、`shutdown_retry.log`、`post_shutdown_connection.log`。
- 服务器已收到完整结果、配对统计、checkpoint/code 校验和、报告及更新后的收尾脚本；最后的 SSH 连接拒绝核验是在关机后补写本地报告，远端文档因此不含这一条最终核验。
- 本地旧的 10 秒/60 秒归档 watcher 和本轮 900 秒 watcher 均已退出；没有安排新训练或继续唤醒服务器。
