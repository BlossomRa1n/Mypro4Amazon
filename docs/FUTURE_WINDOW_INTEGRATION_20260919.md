# Future-Window Protocol Integration Log

更新时间：2026-09-19（Asia/Shanghai）

## 目标

将未来窗口 Top-5 评估从测试专用的数据类接入正式 `code/run_baseline.py`，使训练阶段选模、候选评估、DIN 精排和最终结果都使用同一个 80/20 多目标协议。

## 本次改动

- `code/run_baseline.py`
  - 新增 `--protocol {future-window,leave-two-out}`，默认使用 `future-window`；旧协议保留为显式辅助诊断。
  - 新增 `--future-test-users`，固定未来窗口测试用户集合，避免开发用户和测试用户重叠。
  - `prepare()` 按协议实例化 `FutureWindowData` 或 `BenchmarkData`。
  - 将协议和测试用户数加入数据缓存指纹，避免复用旧切分的 `data.pkl`、SVD 或 ItemCF。
  - 将未来后缀目标集合传入 V2/DIN checkpoint 选择和最终 `ranking_metrics()`。
  - 预测 JSON 新增完整 `targets`、`target_count` 和集合交集意义下的 `candidate_hit`。

- `tests/test_baseline.py`
  - 增加 runner target helper 的多目标传递测试。
  - 增加未来目标集合与训练交互不相交的断言。

- `tools/monitor_training.py`
  - 新增训练监控包装器，记录子进程、日志大小、完成标记、心跳和退出码。
  - 训练完成或失败后写入 `monitor_status.json`，避免服务器训练结束后长时间无人发现。

- `code/run_baseline.py` 优化参数
  - 新增 `--fusion-mode rrf`，保留 `quota` 作为基线融合。
  - 新增 `--itemcf-half-life-days`，按查询历史事件距当前的天数衰减 ItemCF 分数。
  - 结果 JSON 记录融合模式和半衰期，便于基线/优化方案对拍。
  - 新增 `--eval-only-run`，允许在不重新训练模型的情况下，复用已锁定 checkpoint 对比召回融合；只在开发集确认收益后才评估测试集。

## 验证结果

- 本地完整测试：22 项通过。
- 本地 `prepare-only`：通过。使用现有五品类数据采样 30 用户，得到 188 条去重正反馈；142 条进入 80% 前缀训练资产；10 个固定测试用户，18 个开发用户。
- 本地 1 轮端到端 smoke：通过。命令使用 `future-window`、30 个采样用户、10 个测试用户、8 维模型、1 个 epoch；训练、SVD、ItemCF、V2、DIN、候选融合、最终 val/test JSON 均完成。
- smoke 结果文件：`baseline_runs/future_prepare_smoke_20260919/results.json`。
- smoke 结果仅用于流程验证：测试集候选池 HR@100 为 0.30，DIN HR@5 为 0.00；用户数和训练轮数过小，不能作为质量结论。

## 当前结论

未来窗口协议已经在本地正式 runner 中打通，但还没有在服务器上完成正式规模重训。旧的单目标结果 `2.081%` 和 `2.433%` 仍只能作为历史辅助结果，不能与未来窗口主指标直接比较。

## 下一步

1. 将本次代码和日志提交并同步到服务器。
2. 服务器先运行同配置 smoke，再运行固定开发集基线。
3. 基线完成后，在同一未来窗口用户集合上重跑入选的 RRF + 随机负样本 DIN 方案。
4. 通过候选覆盖率和条件 DIN HR@5 判断下一轮优先优化召回还是精排。
