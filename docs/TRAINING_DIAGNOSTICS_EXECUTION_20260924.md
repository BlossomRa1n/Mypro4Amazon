# Raw 训练退化诊断执行记录

2026-09-24 22:16:46（上海）启动的正式诊断已完成。三seed训练、best/last六份confirm、六个checkpoint与完整本地归档均已验收；结果见 `TRAINING_DIAGNOSTICS_RESULTS_20260924.md`。后续夜间cross与full准备来自用户另行批准的8小时有界任务，不改变本诊断协议。

## 已完成

1. Ultra 定义并冻结 `raw-deterioration-20260924-v1`，Medium 实现 runner/指标/监控，Ultra 审计身份、时序、候选和配对逻辑。
2. 本地与服务器各 145 项测试通过；服务器另完成真实 adapter 的 CUDA 三 seed 小型 smoke。CPU 同 seed 对照证明插入诊断后训练轨迹和末次权重与旧路径逐项一致。GPU smoke 证明 CUDA 路径可运行，不将 CPU bitwise 结果延伸宣称为全部 GPU 训练完全确定。
3. 固定原 train 100k、screen 20k、confirm 80k 和 604,511 条训练行；新 100k test 完成身份、排除来源、资格和互斥校验，只锁名单、不开放评分。
4. 商品统计从合法 train mask 重算通过，训练目标和历史时间边界审计通过。SVD/V2/ItemCF 与训练负例排除仍保留预注册的 full-prefix 近似，不宣称逐行或全局日历因果。
5. 九份旧 dev checkpoint 的本地/远端 SHA、归档回执绑定及本地 `torch.load` 全通过后，释放服务器重复副本。旧模型保留于本地忽略目录；原始数据、旧 manifest、日志和源码未移除。数据盘可用约 14.31 GB，满足本轮六模型加原子写入与余量。

## 固定运行合同

| 项目 | 值 |
|---|---|
| 方案 | semantic_concat、随机用户初始化、raw cross |
| seeds | 42、43、44；初始化 424242、424243、424244 |
| 优化 | FP32、AdamW、BPR mixed16、3 epoch，与原 raw 保持一致 |
| screen 时点 | step 591、1181、2362、4724、7086 |
| 选模 | 每 seed 按 screen HR、NDCG、较早 step 依次选 best |
| confirm | 三 seed 的 best/last 全部锁定后统一读取；复用开发 confirm |
| test | 新冻结 100,000 人；本轮评分禁止 |
| 后续动作 | 只分析诊断结果；不自动发起新消融、全量训练或 test |

服务器启动的协议 manifest SHA（canonical）为 `c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6`。协议正文 SHA 为 `45b95890f2d8d4f23d108fca0bd336878f7e835814e50b6170a8251e322851e8`。

正式产物：`/root/autodl-tmp/training_diagnostics_20260924_raw_v1`。源码：`/root/training_diagnostics_20260924/src_v2`。监控：`/root/training_diagnostics_20260924/control_v1`。监控进程 PID 2956、训练 PID 2957；启动时点标识比 PID 更可靠，后续应同时核对命令和运行目录。

## 已解释的部署失败

首次服务器测试的 source_v1 包遗漏旧 tests 所依赖的 tools 文件，且一项 CPU 旧路径一致性测试在可见 CUDA 时加载了不同设备的张量。这是测试部署问题；未开始正式训练。保留 `server_tests_v1.log`，source_v2 补齐依赖，服务器 CPU 测试显式隐藏 CUDA，并独立执行 GPU smoke；两者均通过。不修改科学配置解决此问题。

正式 source_v2 与 protocol_v2 分开归档，旧 source_v1/protocol_v1 保留为实施证据。不存在失败产物覆盖或试验结果挑选。

## 监控与收尾

Low 启动正式任务。服务器监督脚本每 900 秒记录 GPU、磁盘、step 与状态，并独立检测进程退出；断开客户端后继续运行。当前 Codex 线程也配置 15 分钟自动续办，负责归档、分析和关机，不自动重试训练。

本地状态与证据目录：`server_snapshot/training_diagnostics_20260924/`，其中 `OPERATIONS_STATE.json` 是后续接管入口；该目录受 Git 忽略，包含真实用户记录、模型和日志，不推送公开仓库。

本诊断自身的完成与归档门槛现已通过：511文件、8,493,758,940字节传输逐SHA/size相等；六checkpoint可读且有限；21份数组与原cohort/candidate一致；九份旧raw训练轨迹精确一致。原诊断全部产物保留于本地；服务器三份不再使用的last重复副本在完整归档及恢复备份验收后迁移为本地保留，raw best仍在服务器供新对照使用。

按预注册screen规则，三个seed选择0.25/0.5/0.5 epoch，confirm HR平均由4.159583%提高至4.687083%，差0.5275个百分点，95%配对区间[0.442083,0.612083]个百分点。这支持避免部署后续退化checkpoint的操作策略，不证明退化的唯一机制，也不是全量或新test收益。

恢复包 `recovery_20260924T174044Z.tar.gz` 已实际解压、全文件校验并恢复Git/worktree，SHA为 `8a78627c67abac8aefd742b6fcbe74bf9ac66c0831041a6937b25053c49a93b5`。本地临时Python环境午夜失效后，改用持久目录 `/Users/admin/.cache/mypro4amazon-diagnostics-venv`；相关测试及正式归档验证重新通过，服务器科学环境未改动。

用户批准的下一阶段固定截止2026-09-25 08:04:49（上海），至多3轮新优化，到点只完成在跑注册轮。第1轮zero_cross于02:00:18启动，三seed各1epoch，与raw共用0.25/0.5/1选模点；完整训练prefix候选缓存于02:11:50启动，四CPU进程构建。test100k仍封存，后续只有cross选择、源/归档、full模型及实际UID消费覆盖全部通过后才可一次验收。完整后续证据归档、结果定稿后再安全关机，并以全新SSH确认、Git提交推送。

03:25状态更新：cross已于02:40:22完成，原始run/source/control全部下载，模型/数组语义、传输及加强版来源校验通过，固定决策为raw_retained。全量pool继续构建；final executor及训练/test自动归档流程均通过审计和smoke/离线测试，等待真实pool、资源和来源门禁。没有追加优化轮，10万test仍未访问。GitHub SSH只读连通性已通过，尚未提交或推送新变更。
