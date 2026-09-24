# 全量 raw 模型与固定 test100k 最终结果

**结论：冻结的 raw-only 全量一遍训练和唯一 100,000 人 test 已完成，训练、test 的运行与控制归档通过本地 SHA/大小和语义门禁，finalize 与随后新取的远端根目录归档也已通过。最终 HR@5 为 4.520%，NDCG@5 为 0.0204711370。** 这是一个预先锁定模型的绝对验收结果：`structure_gain_accepted=null`，`business_threshold_registered=false`。完整执行和可核验归档不等于证明结构收益、full1 优于 full3，或达到未注册的业务门槛。

本报告于北京时间 **2026-09-25** 在主执行者明确放行已封存 test 后撰写。分析只读本地原指标、原逐用户观测、候选缓存及完成/传输回执，核对既有聚合与身份；没有重新调用模型评分、重建候选、重新读取 future targets、追加 bootstrap、事后分组、模型选择或新实验。最终汇总 closeout、补充恢复包、关机与推送是后续独立门禁，状态由 `OVERNIGHT_CLOSEOUT_20260925.md` 记录；本报告本身不是关机授权。

## 1. 唯一正式 test 的绝对指标

| 指标 | 原始结果 | 分母与含义 |
|---|---:|---|
| HR@5 | **0.0452000000（4.520%，4,520 / 100,000）** | 全部固定 100,000 人；前五至少有一个完整 future 目标。 |
| NDCG@5 | **0.020471137017011642** | 全部固定 100,000 人；IDCG 使用完整 future 唯一目标数，不改成候选内正例数。 |
| pool_hit | **0.1187100000（11.871%，11,871 / 100,000）** | 全部固定 100,000 人；有效候选内至少有一个 future 正例。 |
| 候选 GAUC | **0.8021323507160839** | 候选内同时有正负例的有效用户等权 AUC；原分数并列计 0.5。 |
| AUC 有效人数 / 覆盖率 | **11,871 / 100,000；11.871%** | 本次有效人数恰与 pool_hit 人数相同；不据此改变 HR/NDCG/pool_hit 的全体分母。 |

原始 `raw_metrics.json` 与 `TEST_COMPLETED.json.metrics.raw` 逐字段一致。报告作者以 `allow_pickle=False` 读回封存的 `raw_users.npz`，确认 UID 唯一数为 100,000，有序 UID/position 与原锁定 records 完全相同，候选 IDs/lengths 与原 NPY 缓存逐项相同，实际所有人的有效候选数均为 75，所有保存的数值均有限。原 hit5 合计 4,520、pool_hit 合计 11,871、auc_valid 合计 11,871；原有效 AUC 的平均值与 JSON 完全一致。全部七个 test 附件的实际 SHA 均与 TEST_COMPLETED 的 evidence map 一致。

NDCG 采用冻结实现对保存的 float32 逐用户值求均值，原生聚合与 JSON 完全相同；用 float64 累加得到 0.020471137860715388，两者仅差约 8.44×10⁻¹⁰。报告保留原实现输出，不用另一种累加精度替换正式指标。该核对不重新推导 full-future IDCG。

GAUC 的 0.8021 描述候选有正负例的 11,871 人中的成对排序，并不是全 catalog、真实曝光标签或全用户 GAUC。未召回任何 future 目标的用户仍进入 HR/NDCG 的分母，却不进入本次 GAUC 的有效分母；因此 GAUC 与 4.520% 的 HR 不能脱离候选覆盖和分母直接比较。本报告不另设“高”“低”或“达标”的事后阈值。

## 2. 锁定合同、实际训练与唯一 test 身份

最终模型为 semantic_concat、随机用户初始化、SVD 商品初始化、保留完整 raw cross token；训练 seed42、初始化 seed424242。实际按冻结合同消费 **4,731,777 条合法历史 prefix 行、完整 1 epoch、18,484 次更新**，末 batch 129 条。实际覆盖 **796,197 名用户**，全部固定 test100k 用户的合法历史均被训练消费；这不是要求 test 用户 ID 在训练中从未出现的冷启动验收。

FP32、batch256、embedding/token_dim256、hist50、BPR mixed16、AdamW 学习率 0.001、weight decay 1e−5、clip5 和 epoch 级 cosine T_max=3 保持注册值。实际单个 epoch 的更新日志学习率均为 0.001，scheduler 在 epoch 末执行。负例来源记录为 band11–25：9,166,248；band26–50：9,362,976；candidate fallback：296,755；random：56,882,453，合计 75,708,432，等于 4,731,777×16。来源数量、消费 UID、顺序和末 batch 已由正式门禁核验，不把消费 digest 说成负例 IDs 已被独立重放。

训练候选权重 `[2,0,0.7,0.05]`，test 候选权重 `[2,1,0.7,0.05]`；RRF、75 容量、ItemCF300 和 180 天半衰期不变。训练没有 V2 通道而 test 有 V2 通道，属于本轮冻结的分布差异，未根据最终表现再作修改。Test 排序按原分数降序，分数并列按 item ID 升序；候选由既定召回链路产生，不注入或按 future 正例过滤。

固定 test100k 从 337,749 名合格用户中，以 seed20260924 在排序后的 raw ID 上无放回抽样并排序保存，剩余储备 237,749 人；身份和边界在训练/评分之前封印，未按模型分数、命中率、候选覆盖或 future 商品内容选人。与 fast train100k、screen20k、confirm80k 的用户交集均为零。全量训练可使用各 test 用户评估边界之前的合法历史，训练与 prepared 回执均记录 test future labels 未读取、test pool 未构建。

Stage A/B 的排除集合是依据运行前源码、补丁、启动配置、日志和固定 RNG 调用序列作的**确定性重建**，不是找回原逐次名单。重建并集 19,878 人替代原 774,585 人保守母体；其余已审核来源保留，并补入最新 screen/confirm。原逐次身份数组、RNG 快照和当时精确 NumPy 版本没有保存。本轮仅覆盖用户批准的本月已知来源，六月按批准不排除，也不声明覆盖任何未知、未记录实验。新 test 与旧 17,296 人名单交集为 5,114，两者不是两份相互独立的 test，不作重复计数。

唯一全局 marker 来自原身份 lock 父目录的 `FINAL_TEST100K_STARTED.json`，并非只取 final 工作目录中的同名副本。只读原路径前后 SHA 相同，原始读回字节、全局副本与 session `TEST_STARTED.json` 相同，绑定原 plan、prepared、checkpoint 和有序 records。正式 test 只执行一个 raw 模型；finalize 仅完成回执封印，没有重新启动训练或 test。

## 3. 开发证据支持什么

已封印 raw 诊断在固定 80k confirm、三个 seed 下，早期 screen 规则选出的 best HR@5 平均为 4.687083%，epoch3 last 为 4.159583%；best−last 为 **+0.005275（+0.5275 个百分点）**，95% 配对区间 **[+0.0044208333,+0.0061208333]**。三个 seed 的差值均正，注册开发支持条件通过。这支持在 fast 开发合同下保留按注册 screen 规则选择的早期 checkpoint；不是对 full1 最优的证明。

共同早期停止下的 cross 复验中，zero−raw 的平均 HR 差为 **+0.0000416667（+0.0041667 个百分点）**，97.5% 区间 **[−0.0004500,+0.0005291667]**，三个 seed 一正两负，结果为 `raw_retained`。未晋级不等于等价、非劣效或早期 raw 显著更优。旧固定 epoch3 的 zero−raw 差仍为 −0.0015916667，97.5% 区间 [−0.0023958854,−0.0007958333]；不能用“一个区间跨零、另一个不跨零”证明停止点与结构的交互显著。

Confirm 已参与开发，不能称为新的独立 holdout。两轮注册区间均先对每名用户跨固定三个 seed 的配对差求平均，再对 80,000 人 bootstrap；不是 240,000 个独立用户，区间也不覆盖所有可能训练 seed 的不确定性。两轮区间水平分别为 95% 和 97.5%，不能据其大小另作未经注册的显著性排序。

Fast train100k 恰等于 screen20k∪confirm80k，但 seed42 的 best 在 step591，seed43/44 在 step1181；对应仍有 26,798、9,046、8,943 名 confirm 用户的训练目标尚未被消费。全量一遍则覆盖所有规定合法 prefix，用户、样本数、更新数及暴露范围均不同。**最终 4.520% 不能直接减去开发集的某个绝对数值，解释为全量改进或退化。** 此次没有 full3 或最终 zero 对照，不报告 full1−full3 收益、最终 cross 收益或全量优化的因果收益。

训练退化的现象与停止策略已有可执行证据。现有日志和探针在检查范围内未见数值崩溃，但用户 ID 记忆、共享参数漂移、head 优化和负例目标失配等机制未被独立因果拆分。Zero 消融清零的是含公共 type offset 的额外 cross token，DIN 内部历史注意力交互仍存在，也不能将消融解释为“所有交互都无效”。

## 4. 时间可用性与归档验证的限制

身份隔离不自动证明所有特征严格时序无泄漏。当前 benchmark 复用所有用户合法训练 prefix 聚合的 item 统计、SVD、ItemCF 和 V2；这些资产可能含某条训练行之后的合法 prefix 信息。训练负例排除使用用户完整合法 prefix，属于回顾式负例采样；active catalog、编码与静态品类/brand 也有历史可用性假设。用户 query/history 的时间边界不代表召回底座同时具备逐行或统一全球日历时点的可用性。

原审计显式为 **`strict_per_row_causal=false`、`strict_global_calendar_causal=false`**。本轮沿用这些已披露条件，不声称严格逐行因果、交叉拟合或上线因果收益；“合法历史”只在注册 benchmark 边界下使用。未将 held-out/test 后缀用于训练目标、统计拟合、训练负例过滤或候选选型，是另一条被守卫和来源证据约束的边界，不能与所有时序可用性混为一谈。

本地语义验证可核 checkpoint 可读性/有限值、消费顺序/人数、候选缓存、封存标签/分数、身份、指标及回执一致性，但完整 future IDCG 无法仅从候选 labels 独立重建；本报告没有为消除此限制重新打开 targets。完整负例 IDs 也未从消费 digest 重放。原始远端来源、部署清单和传输真实性由独立来源映射、实际 command/stdout/stderr/exit 与前后 inventory 门禁补充，单个本地 `verified` 字段不替代整条证据链。

## 5. 实际时间、成本与归档

以下北京时间均为 **2026-09-25（UTC+8）**；对应 UTC 为 2026-09-24。阶段监督墙钟、训练计时、下载与本地检查分开报告，不能相加后称为全夜端到端耗时。

| 阶段 | 实际时间或耗时 | 口径 |
|---|---|---|
| 全量训练候选池构建与校验 | 5,073.194 秒，84.55 分钟 | 已封印 fullpool 的 build_and_validation_seconds。 |
| 共同早停 cross 复验 | 02:00:20–02:40:22，约 40.06 分钟 | 监督墙钟；zero 三 seed 纯训练记录约 23.64 分钟。 |
| 最终 raw 全量训练 | 04:11:45–05:21:17，4,171.474 秒，69.52 分钟 | 监督墙钟，含前置校验及完成检查；train_seconds 为 3,720.925 秒，62.02 分钟，不含全部准备/归档。 |
| 训练归档与本地语义校验 | 05:21:40–05:23:27 | run transfer 86.324 秒、control transfer 0.828 秒、本地 verifier 19.232 秒。 |
| 唯一 test 候选构建与评分 | 05:27:36–05:40:57，800.215 秒，13.34 分钟 | 整个 test 监督阶段；没有单独可靠的纯评分耗时，不能全部称为模型推理时间。 |
| Test 归档与本地语义校验 | 05:41:19–05:41:53 | run transfer 6.583 秒、control transfer 0.624 秒、本地 verifier 27.098 秒。 |
| Finalize | 05:44:09 完成，exit0 | 只封印 FINAL_COMPLETED。 |
| Finalize 后的新远端根目录归档 | 05:45:43 验证完成 | 新取得 237 文件、3,367,517 字节，实际传输命令 exit0。 |

最终训练 run 为 8 文件、1,270,293,392 字节，control 为 5 文件、5,526 字节；test run 为 8 文件、80,975,069 字节，control 为 5 文件、2,583 字节。前后远端 inventory 相同，本地文件集合、SHA 与大小精确一致；语义检查分别通过。Fresh-root 归档主要封存来源、计划、回执和控制文件，它不代替独立的训练/test 大资产归档，部分重复文件不能当成新增独立证据数量。

此前 raw 诊断归档 511 文件、8,493,758,940 字节；cross run 39 文件、3,996,425,341 字节，source/control 95 文件、745,209 字节；fullpool run 8 文件、1,476,351,420 字节，source/control 15 文件、77,277 字节。Raw 诊断为取得 last 对照实际跑完三 epoch；早期 checkpoint 位置不能算成本轮已经节省六至十二倍训练算力。夜间最多三轮新优化的授权下实际只新增一轮共同早停 cross 复验，随后执行独立注册的 full/test 验收。窗口为 00:04:49–08:04:49；本轮不因最终指标再增加搜索。没有云账单或精确计费资料，故不报告货币成本。

本地完整测试日志记录 164 项、89.024 秒、OK；运维/恢复守卫另有定向 mock 日志。正式最终 runner 的服务器小型链路为 CPU 三项通过，没有将其写成已完成最终 CUDA tiny smoke。原 raw/cross 的 GPU 证据不能替代这一具体 runner 的 smoke 口径。

## 6. 两个保留的部署失败与恢复范围

首次服务器 CPU 小型链路因缺少 `docs/TRAINING_DIAGNOSTICS_PROTOCOL_20260924.md` 报一个 FileNotFoundError。实际部署清单 `source_manifest.json`（126 项）与 `remote_source_manifest_v2.json`（176 项）证明原 126 项 SHA 零变更、零删除，只增加 50 项 docs，第二次三项 CPU 测试 27.044 秒通过。该结论绑定这两份远端部署清单，不表示当前本地仓库每一项都等于旧部署；本地 cross 归档验证器已有单独审计的加强版本。旧失败日志保留。

首次正式 prepare 因远端遗漏原身份 SHA256_MANIFEST.json 而退出，发生在候选、训练或 test 访问之前。实际修复仅补相同 SHA 的原始 949 字节封印，55 项原文件不变，证据变为 56 项并恢复 28/28 授权 refs；旧失败、旧锁和旧输出保留。独立恢复 prepare 于 03:58:48 exit0，仍绑定同一原授权与计划，不是重置 test 身份或重试正式评分。

原恢复 tar 为 129,638,964 字节，含 827 个源文件及 Git 恢复资料；93 项外部资产共 10,592,920,075 字节已有实际 SHA/大小复核与恢复回执。GB 数据、模型、候选和原 Git bundle 是明确的外部依赖，不能只保存源码压缩包就声称一切可恢复。完整 source mapping 的 149 项来源均纳入来源/归档依赖，legacy src_v2 原源码不能用当前 Git 替代。

原包的本地环境快照与远端科学运行环境分开保存。实际远端为 Python 3.12.3、PyTorch 2.3.0+cu121、CUDA runtime12.1、cuDNN8902，环境补充记录 159 项 Python 包、74 项 conda 包版本/build、glibc2.35 与 driver595.91.07。环境记录不是离线系统镜像或安装包，也不保证跨设备 bitwise GPU 重现。本地归档运行时 Python3.12.14/PyTorch2.14 的资料不能误称远端训练环境。

补充恢复包将在本报告、工作总结和 closeout 冻结之后捕获当时 HEAD、选定源码/文档及 staged/unstaged 状态，并单独做恢复验证。关机后的 closure 文档、最终 commit 与 push 回执不在关机前包的捕获时点内，应另行记录，不追写已封印旧包。

## 7. 证据定位与原字节身份

下表路径默认相对于 `server_snapshot/full_final_test100k_20260925/`；另有前缀的条目按所示路径。SHA 均为文件原字节；canonical hash 另列，不能互换。Checkpoint 的 SHA/可读性依据已通过的正式训练语义门禁；本报告另直接复核 test 七附件的 SHA 和封存数组聚合。

| 证据 | 字节 | SHA-256 |
|---|---:|---|
| `authorization.json`，28 项 refs | 8,457 | `d14f939f9f94f6a256fd361d0b35f2a29b9204325dfeaac008d1e04a85457f5e` |
| `final_plan.json` | 10,476 | `ab8caa81e32142a07095b57c445d5f8afc1c3a6ee626868ed1b379547f9dd893` |
| `../final_gates_20260925/source_mapping_verified.json` | 56,440 | `9312bf4c3e175efeefeb8a86cc87339f40325f75162852b2fd9bd5cdee21b805` |
| `training_run/TRAINING_COMPLETED.json` | 279,893 | `568e9b3e12552b55ee57b751ff688a9b9e22588c302c0d44e3d3b8bf3626de57` |
| `training_run/raw/final.pth` | 1,260,755,308 | `1876241e233adbfddbf531ae9f463e4995ddb12b95f34af49c32160561167bf8` |
| `training_run/raw/trace.json` | 167,044 | `3d782a8614876d06506d3ba2d4aeb8f7f61704cd137b5aa5850bc0f51916bebc` |
| `training_run/raw/initial_state_audit.json` | 6,158 | `6ceefae2b3fc63859b7d69aa8cee0f7de62ad298e53d1713ebbca62ea95bfd52` |
| `training_run/raw/consumed_uids.npy` | 6,369,704 | `71fe691bf6b40d4babc53334fcba85c1c143b959b23faba5e6284b8805cf915b` |
| `training_archive_verified.json` | 4,778 | `456cd997c0e40e9dd5945a3297d2af269433c636efefc2617466bd7abaff7428` |
| `training_run_transfer_verified.json` | 791 | `473bb3b2c4bb8d65990ae109afcbcc00ba97775f7694bf277763703c25ab6a88` |
| `training_control_transfer_verified.json` | 689 | `b6397c53e96c08afa0ec5a24b475aa72a7f1215566cef6d2a1a9dd4ecf043b99` |
| `FINAL_MODELS_PREPARED.json` | 1,285 | `ee300352672ae6e3df8eae649c0d209bca6c9956084dd0ce3f2952583ebe86f9` |
| `seal_models_v1/COMPLETED.json` | 570 | `6441bb1b19066613b26ea0b72143c9a771dc2383fef446e52f5905e05d7e0c90` |
| `GLOBAL_FINAL_TEST100K_STARTED.json` / session 原 marker | 604 | `04c0900e6aef9ae9b8652de08a54a87763b46a03163cb135ad4b34127a0697fa` |
| `global_marker_source_verified.json` | 3,499 | `6fdd9866e789cafbe9e8833c0287b2844d67218916012a3fb0d6ddbd046f87b4` |
| `test_run/TEST_COMPLETED.json` | 1,626 | `933c2c0c8a26a23bfb8183523eaf82049a912741b10a8e66fca7af5934ca9e17` |
| `test_run/raw_metrics.json` | 305 | `2786ff1e6f651755e1955faf5948309e72fbda1c6730ad23c939f0df3d114f56` |
| `test_run/raw_users.npz` | 50,564,712 | `0544e45b906db4a40761bb7274d99a02333e6915dc13a569dbbbde1b5b6d5b61` |
| `test_run/candidate_test_final.json` | 7,503 | `fa879ea1225195d9cdf1bd3315b2e21d9408c1bc0e40929c3b798db56a22ff8f` |
| `test_archive_verified.json` | 5,689 | `44802497b8c12349de047e1dfb1ed0c41e089df82a0b7ab368dde69e4e623d88` |
| `test_run_transfer_verified.json` | 780 | `edd37e383b34f353605cbffb09ccb6b3c442a3c292240b563f05a847fe70fa43` |
| `test_control_transfer_verified.json` | 681 | `c3598b4d850c1ad30551fd7905d3741894581a7ae98ae78b799feabdb4673376` |
| `test_watcher/COMPLETED.json` | 549 | `210b6d47876e305ef3028dd98a0420f9e38087c82b6191dba4b1a8da2295c230` |
| `FINAL_COMPLETED.json` | 668 | `4cfac4fbaab15ed848fc07825592c4f5a5fee93d462ab9dc527d54978b29104b` |
| `finalize_only_v1/COMPLETED.json` | 376 | `b1c6f04954bcdb680944e3952b2708859c8493f7e2f1f42726a78da5263614d0` |
| `final_remote_root_transfer_verified.json` | 675 | `c3eed6a9a9345dd5efed6371e15f4d90cbb49a56c2b64e9ddc3892d1e2fe30ea` |
| `final_remote_root_transfer_verified.inventory.json` | 38,845 | `83d80758ee37802143185986f6cc3849e01bad57bc7ed521af4d7160112cc8b8` |
| `final_remote_root_transfer_execution.json` | 1,776 | `16b7257c0b010543e74800b5761bdd4fffa579e37d970e96697f356d00dcc1b0` |

身份文件位于 `server_snapshot/test100k_feasibility_20260924/locked_test100k/`：test100k_manifest 为 `9416c80d3c2bda822239e19a1ffd5cc8b0497c1aca252225dee26329a9cf4303`，test_records 原字节为 `086e977d4d7bb06f453bae2596dcdddb514796add4439bb5bb5ecdd18d8732fc`，原 seal 为 `30087cca4a9d7e3db166c8bff7d41cd3ef06226b78087a7a2a0989f5e199e188`。最终有序 records canonical hash 为 `7ecf66b7186e6fcb0195dcaee8492d3912fc7d3cca50358c35b8970240bc0f38`，test pool hash 为 `31ca5aae77ba797dd6d314898da372617424a57eadb87ac90296e5f22c76c296`，plan canonical hash 为 `092aafb2a776c1e84d8bec2dcda9ee31312b131dfe7ced9ff2c6bf039fe8015f`，prepared canonical hash 为 `150f35b350bf9bf354f54cb60dc8ac6a68e14fc0e69da57fa4e2a3202faa8228`。

原 `TEST_COMPLETED.local_archive_verified=false` 是远端 test 完成时的历史状态，早于本地归档验证；它保持原字节，没有改成 true。随后真实的 test_archive PASS、finalize 引用该本地门禁的 FINAL_COMPLETED 和新远端根目录 transfer 共同证明后续归档完成。实际命令、退出和日志分别保存在 training/test watcher events、seal_models_v1、marker_capture_v1、finalize_only_v1 与 root transfer execution，不用辅助 mock PASS 代替正式执行。

本报告与 `OVERNIGHT_WORK_SUMMARY_20260925.md` 将供最终 closeout 绑定。绑定之后不修改两报告、operations state、checklist 或 operations 树；后续恢复、关机和 Git 状态另写 closure 及独立回执。原 auth-bound cross 结果、最终计划模板及 28 项授权引用保持原字节。
