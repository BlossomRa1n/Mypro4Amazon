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
  - 新增 `--validation-only`，让开发期对照完全不生成测试指标，避免误用测试结果调参。
  - `--eval-only-run` 现在校验来源 run manifest、数据协议、`data_id` 和两个 checkpoint 的模型配置，并把 checkpoint 来源身份写入结果；不匹配会直接失败。
  - `--validation-only` 的 `COMPLETED.json` 明确标记为 `validation-only`，预测 JSON 在未来窗口协议下使用 `first_target` 表示兼容性的首个目标，完整标签只使用 `targets`。

- `tools/monitor_training.py`
  - 读取完成标记中的状态，区分完整运行和 `validation-only`，避免监控结果把开发集对照误报成最终测试。

- `tools/auto_select_future_window.py`
  - 等待 RRF 开发集对照完成，只读取 `val` 结果；以 DIN HR@5 为主指标、候选池 HR@100 为平局破局指标且要求 DIN NDCG 不回退。
  - 只有开发集满足规则时才自动启动同一 checkpoint 的 100,000 用户最终测试，并再次交给 `monitor_training.py`；拒绝时写入 `AUTO_SELECT.json` 并结束，不触碰测试集调参。
  - 对照链固定使用与基线相同的 `final-users=100000` 开发用户集合；若用户数不同，不允许比较汇总指标。

- 服务器空间恢复
  - 2026-09-19 基线在数据盘满载时留下了不完整的 `din_best.pth.tmp`；已删除该临时文件并保留 epoch-1 continuation checkpoint。`tools/relocate_checkpoint.sh` 会在恢复训练写完 epoch-2 continuation 后将其迁到系统盘，再继续 epoch-3，避免原子保存需要额外 3.6 GiB 时再次失败。

## 验证结果

- 本地完整测试：22 项通过。
- 本地 `prepare-only`：通过。使用现有五品类数据采样 30 用户，得到 188 条去重正反馈；142 条进入 80% 前缀训练资产；10 个固定测试用户，18 个开发用户。
- 本地 1 轮端到端 smoke：通过。命令使用 `future-window`、30 个采样用户、10 个测试用户、8 维模型、1 个 epoch；训练、SVD、ItemCF、V2、DIN、候选融合、最终 val/test JSON 均完成。
- smoke 结果文件：`baseline_runs/future_prepare_smoke_20260919/results.json`。
- smoke 结果仅用于流程验证：测试集候选池 HR@100 为 0.30，DIN HR@5 为 0.00；用户数和训练轮数过小，不能作为质量结论。

## 2026-09-19 服务器正式结果

代码已同步到服务器 `/root/MyPro-Amazon/code` 和 `/root/MyPro-Amazon/tools`，并通过 SHA-256 校验。一次旧的自动链状态文件已清理；基线训练进程没有被中断。训练结束后，监控器写入 `monitor_status.json`，选择器按同一开发用户集合完成对照，最终评估只在开发集门控通过后启动。

服务器正式运行目录和本地归档对应为 `server_snapshot/2026-09-19/`：

- `future_window_baseline_20260919`：五品类、全量数据、未来窗口、100,000 测试用户、100,000 最终评估用户、3 epoch、256 维。测试 DIN HR@5 `0.03468`，NDCG@5 `0.0154037`，Recall@5 `0.0203112`；候选池 HR@100 `0.13132`，DIN 条件 HR@5 `0.26409`。
- `future_window_rrf_dev_20260919`：复用基线 checkpoint，只改变融合为 RRF 并加入 ItemCF 90 天半衰期，validation-only，开发集 100,000 用户。DIN HR@5 `0.04029`，NDCG@5 `0.0179362`；候选池 HR@100 `0.12375`。相对同用户基线，HR@5 增加 `0.00473`，配对 bootstrap 描述性区间为 `[0.00376975, 0.00569]`，不作显著性声明。
- `future_window_rrf_final_20260919`：开发集门控接受后，在未用于选择的测试用户上完成最终评估。测试 DIN HR@5 `0.03842`，NDCG@5 `0.0170563`，Recall@5 `0.0223965`；候选池 HR@100 `0.12275`，DIN 条件 HR@5 `0.31299`。

最终测试相对同一用户协议下的 quota 基线：HR@5 从 `3.468%` 到 `3.842%`（`+0.374` 个百分点，约 `+10.79%`）；NDCG@5 从 `1.5404%` 到 `1.7056%`（约 `+10.73%`）。这是当前未来窗口主指标结论；旧的 leave-two-out 单目标结果 `2.081%` 和 `2.433%` 只能作为历史辅助，不能直接横向比较。

本次链条没有生成新的模型 checkpoint，最终方案复用了基线的 V2/DIN checkpoint，仅改变候选融合和 ItemCF 时间衰减。服务器完成后无训练、选择器或监控进程残留；数据盘约剩 `655 MiB`，原始数据、checkpoint 和结果均保留，后续清理必须避开这些文件。

## 下一步

1. 将本次服务器结果和测试通过记录提交到 Git，并推送远程仓库。
2. 以最终测试的候选池 HR@100 `12.275%` 和 DIN 条件 HR@5 `31.30%` 拆解瓶颈：当前召回覆盖率仍是第一优先级，精排在候选命中条件下已有较好的排序能力。
3. 在固定未来窗口开发集上一次只改变一个变量，优先筛选候选预算、ItemCF 衰减和召回权重；通过门控后才使用测试集确认。
4. 精排结构优化（raw-slice token、RankMixer）必须重新训练并沿用同一未来窗口协议，不能复用旧 leave-two-out checkpoint 得出主指标结论。

## 2026-09-19 wcf2 final

在固定 100,000 个开发用户上，严格配对选择接受 `RRF + ItemCF half-life 180 days + fusion_weights [2.0, 1.0, 0.7, 0.05]`：DIN HR@5 从 `3.556%` 提升到 `4.396%`，配对差值 `+0.840` 个百分点（gained `1856`、lost `1016`，描述性 bootstrap 95% 区间 `[+0.735,+0.946]`）；NDCG@5 从 `1.5748%` 提升到 `1.9801%`。

最终测试使用同一 checkpoint、独立的 100,000 个测试用户，结果为：

- DIN HR@5 `4.121%`，NDCG@5 `1.8663%`，Recall@5 `2.4092%`。
- 候选池 HR@100 `12.284%`，DIN 条件 HR@5 `33.5477%`。
- 相对未来窗口 quota 基线（HR@5 `3.468%`、NDCG@5 `1.5404%`），HR@5 增加 `0.653` 个百分点（约 `18.8%`），NDCG@5 增加 `0.326` 个百分点（约 `21.2%`）。
- 相对已完成的 RRF + half-life 180 默认权重方案（HR@5 `3.857%`），HR@5 增加 `0.264` 个百分点（约 `6.9%`）。

最终目录为 `/root/autodl-tmp/future_window_rrf_final_wcf2_hl180_20260919`，本地快照为 `server_snapshot/2026-09-19/parameter_sweeps/rrf_wcf2_hl180_final/`。本轮没有训练新 checkpoint，只复用未来窗口基线的 V2/DIN 权重；监控器记录 `exit_code=0`，无残留评估进程。

随后对 Category 权重做了一个单变量开发集检查：`[2.0, 1.0, 0.4, 0.05]` 在 100,000 个开发用户上得到 DIN HR@5 `4.382%`、NDCG@5 `1.975%`，低于入选的 Category `0.7`（`4.396%`、`1.980%`），因此拒绝，不进入测试集。结果快照为 `server_snapshot/2026-09-19/parameter_sweeps/rrf_wcf2_wcat04_hl180/`；该次 eval-only 运行生成的约 1 GiB 缓存已从服务器删除，只保留结果、预测和监控文件。

## 2026-09-19 cf_neighbors 验证

在固定的 100,000 开发用户、同一未来窗口协议、RRF 权重 `[2.0, 1.0, 0.7, 0.05]` 和 ItemCF 半衰期 180 天上，只改变 ItemCF 邻居截断数。所有运行复用同一份 V2/DIN checkpoint，验证阶段使用 `validation-only`，通过后才使用独立测试用户。

- `cf_neighbors=200`：开发集 ItemCF HR@100 `12.587%`，候选池 HR@100 `12.744%`，DIN HR@5 `4.388%`，NDCG@5 `1.97730%`。相对当前 `cf_neighbors=100` 入选方案，配对 gained `230`、lost `238`，HR@5 `-0.008` 个百分点，NDCG `-0.00285` 个百分点，拒绝。快照为 `server_snapshot/2026-09-19/parameter_sweeps/rrf_wcf2_hl180_cf200_dev/`。
- `cf_neighbors=300`：开发集 ItemCF HR@100 `12.756%`，候选池 HR@100 `12.904%`，DIN HR@5 `4.397%`，NDCG@5 `1.98105%`。相对 `cf_neighbors=100`，配对 gained `323`、lost `322`，HR@5 增加 `0.001` 个百分点，NDCG 增加 `0.00090` 个百分点；按既定门控通过，但开发集增幅很小。快照为 `server_snapshot/2026-09-19/parameter_sweeps/rrf_wcf2_hl180_cf300_dev/`。
- `cf_neighbors=300` 独立测试：DIN HR@5 `4.193%`，NDCG@5 `1.88455%`，Recall@5 `2.44663%`，候选池 HR@100 `12.888%`，候选命中条件下 DIN HR@5 `32.534%`。相对当前 `cf_neighbors=100` 测试结果（`4.121%` / `1.86634%`），配对 gained `365`、lost `293`，HR@5 增加 `0.072` 个百分点，描述性 bootstrap 95% 区间 `[+0.022, +0.122]`，NDCG 增加 `0.01822` 个百分点。快照为 `server_snapshot/2026-09-19/parameter_sweeps/rrf_wcf2_hl180_cf300_final/`。

因此，当前推荐配置更新为 `cf_neighbors=300`、RRF、ItemCF half-life `180` 天和融合权重 `[2.0, 1.0, 0.7, 0.05]`。这次没有训练新模型，测试运行由监控器托管并以 `exit_code=0` 完成；V2/DIN checkpoint 仍来自 `future_window_baseline_20260919`。

## 2026-09-19 candidate budget 验证

在当前 cf300 配置上把候选预算从 `100` 增加到 `150`，只在固定 100,000 开发用户上验证。候选池 HR@150 从 `12.904%` 提高到 `14.938%`，但纯 DIN HR@5 从 `4.397%` 降到 `3.961%`，NDCG@5 从 `1.98105%` 降到 `1.74441%`。逐用户配对 gained `452`、lost `888`，HR@5 差值 `-0.436` 个百分点，描述性 bootstrap 95% 区间 `[-0.508,-0.364]`；方案拒绝，不进入测试集。完整快照位于 `server_snapshot/2026-09-19/parameter_sweeps/candidates150_cf300_dev/`。

结论：当前候选预算继续保持 `100`。扩大候选池虽然提升召回覆盖，却向 DIN 引入更多难以排序的噪声，端到端 Top-5 反而下降。

## 2026-09-20 candidate budget 75

在 cf300 配置上继续验证候选预算 `75`。固定 100,000 开发用户上，候选池 HR@75 为 `11.723%`，DIN HR@5 为 `4.669%`，NDCG@5 为 `2.12691%`；相对候选预算 100，逐用户 gained `623`、lost `351`，HR@5 增加 `0.272` 个百分点，描述性 bootstrap 95% 区间 `[+0.211,+0.333]`，NDCG 增加 `0.14587` 个百分点，开发门控通过。

独立 100,000 测试用户确认：候选池 HR@75 为 `11.644%`，DIN HR@5 `4.481%`，NDCG@5 `2.03855%`，Recall@5 `2.62890%`，候选命中条件下 DIN HR@5 `38.483%`。相对 cf300/预算 100 测试结果（`4.193%` / `1.88455%`），逐用户 gained `610`、lost `322`，HR@5 增加 `0.288` 个百分点，描述性 bootstrap 95% 区间 `[+0.228,+0.348]`，NDCG 增加 `0.15399` 个百分点。结果快照位于 `server_snapshot/2026-09-19/parameter_sweeps/candidates75_cf300_final/`。

当前推荐配置更新为 `candidates=75`、`cf_neighbors=300`、RRF、ItemCF half-life `180` 天和融合权重 `[2.0, 1.0, 0.7, 0.05]`。这轮仍未训练新模型，继续复用未来窗口基线 V2/DIN checkpoint。
