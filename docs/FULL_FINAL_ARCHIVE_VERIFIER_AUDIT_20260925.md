# Full最终归档验证器审计

状态：**代码审计PASS；正式本地归档仍须实际运行验收；不是test授权**。审计只读源码、测试与哈希并编写本文，没有连接服务器、训练模型、重新评分或读取test future targets。

| 对象 | SHA-256 |
|---|---|
| `tools/verify_full_final_archive.py` | `ac523e7fef268d014e0d6d2b4b8431a84fdde243d756f1c643cb71dc3ee8a1bb` |
| `tests/test_verify_full_final_archive.py` | `2e83448b28e1cf75e150ed599b247ced027bc698d0cb4faa8d06abd85ee0a0f7` |
| 冻结final executor，保持未修改 | `ae8756bde5ef024fbafb2078f615d348e4e7a71d51d13afa9a8bf50b284df862` |
| `tools/monitor_final_stage.py`，追加审计PASS | `781d741b42f2f9c6ff7c3abb10d201e4811f7aec1754c3d602a5b7a2d0503207` |
| `tests/test_monitor_final_stage.py` | `1a32f79ead73befb453adc4c6824c16c82043233fce044e62cf073f09040ad0d` |

## 1. 本地映射与来源边界

工具使用显式本地输入，不修改原 `final_plan.json` 字节，也不解析为远端文件去读取plan中的原路径。它重算plan canonical seal并核对executor、固定训练合同、metrics/acceptance与正式人数；对本地diagnostic和old protocol核对原引用SHA及canonical。正式模式固定diagnostic `c82aa0…` 与旧protocol文件 `4afac7…`，smoke须一致。base pickle在反序列化前先匹配原protocol `base_assets.data.pkl` SHA；pickle仍限独立来源映射门禁已验证的可信本地资产。

本地raw完成文件与plan引用SHA一致，实际使用的raw初始审计、best checkpoint与history均在raw evidence seal中核验。test records逐字节匹配原引用，UID/raw UID唯一，顺序hash与plan一致；positions的dtype/shape/逻辑数组SHA及全部行与base合法train_positions相等。只访问UID、合法训练位置及身份边界映射，并把targets置为拒绝；不调用targets或模型评分。

原授权人/授权来源、远端原路径到本地副本的真实性、prepared与global marker原位置、完整源bundle及远端传输真实性仍由主执行者独立映射/传输门禁证明。工具回执明确保留此限制，不能把本地语义验证写成已经替代source/transfer验收。

## 2. 训练归档语义验收

训练目录文件集合必须精确等于所需artifact加完成回执，拒绝缺失、多余文件、symlink、特殊文件、失败marker及任何SHA漂移。JSON/JSONL递归拒绝非有限值。输出验证回执不能写入正在验证的目录、raw控制目录或test模式所引用的训练目录，也不能覆盖已有文件。

每模型检查初始state hash等于封印raw42、完整架构等于raw控制仅改变cross_mode、固定seed/init/epoch/T_max/FP32 metadata，trace与完成/模型manifest相等。seed42原排列和batch sizes独立重建；正式18,484 updates、末129，step日志、mixed16来源总数、实际消费UID数组及hash均复核。消费UID集合必须精确等于 `unique(data.uid[full_positions])`，且包含全部固定test UID，不只验证声明的count。

最终checkpoint CPU读取，所有tensor键、shape、dtype与已封印raw best规格一致且有限；每个文件SHA及embedded manifest相等。双模型消费trace必须相同。逐行负例ID摘要只核格式与封印一致，不重放负例采样或训练，回执明确此边界。

## 3. 一次test证据与指标

test模式另要求显式本地prepared、global once-marker、training run与训练归档回执，并重新执行训练语义验收。prepared canonical、plan文件SHA/plan hash/models、实际training completion SHA、训练回执SHA、实际每模型checkpoint SHA逐项绑定。global marker内容必须完全等于会话 `TEST_STARTED.json`，其prepared_hash与checkpoint hashes必须来自这些实际已验收文件；不能只检查两份自报完成JSON内部相等。

test cache sidecar及payload的SHA、dtype、shape、length、ID/padding和logical hash按原cache实现重读。source完整字典必须等于本地已认证old protocol的base资产/code map与固定100k身份、75容量、四路 `[2,1,0.7,0.05]`、CF300/180天规则，不只验证若干参数。逐用户数组顺序必须等于锁定records，每行候选ID/length/顺序必须等于cache；从已有score和label重算HR、pool_hit、候选AUC有效mask、GAUC及汇总。

raw+zero分支检查同一UID/候选/label配对，重算ΔHR/ΔNDCG、gained/lost及seed20260925的10,000次95%配对bootstrap，固定四项结构验收规则与报告相同。raw单模型不能报告未注册paired比较。NDCG保留已有per-user值，仅验证有限范围和汇总；其完整future IDCG不能从候选label重建，工具不会为重算分母重新打开目标。已有label视为封印观察，不重新生成。

## 4. 修正与测试证据

本次关闭初稿三类缺口：test结果未绑定实际prepared/global marker/训练模型链；cache source只比子集且base pickle未在加载前绑定原字节；回执位置只排除了当前run而未排除其他只读归档。正式固定source anchors和smoke一致性也在pickle前执行。

实现者运行 `tests.test_verify_full_final_archive` 与 `tests.test_monitor_final_stage` 的组合测试，报告4项通过、21.213秒；其中本验证器为2项，monitor为另外2项，monitor后续追加审计见下节。审计者阅读验证器测试而未重新训练。测试使用真实tiny完整CLI产物验收training/test，断言plan bytes前后不变；另从真实双模型tiny评分数组检查配对CI和接受规则。完整重封marker/prepared_hash、paired CI、接受字段、metrics、非有限JSON、缺少初始审计、未封印额外文件和归档内回执位置均拒绝。

实际正式执行仍需主执行者完成所有源/身份映射与传输门禁，再用该冻结版本验证本地完整训练与test归档。test模式会再次CPU读取最终checkpoint与原raw规格，磁盘/时间估算应包含该读回。只有实际回执、完整源与可恢复归档共同通过，才满足finalize或安全关机前的证据要求。

## 5. Final阶段监控器追加审计

独立新monitor与原 `monitor_training_diagnostics.py` 的diff限定为stage选项/记录、对应完成marker及最终模型目录的 `*/steps.jsonl`；旧monitor未修改。`--stage training`必须见 `TRAINING_COMPLETED.json`，`--stage test`必须见 `TEST_COMPLETED.json`，同时要求被监督子进程实际exit0且无 `FAILED.json`，才写control目录的完成回执。仅出现旧通用 `COMPLETED.json` 不能误报成功。

monitor只启动调用方明确提供的一条command，记录command、stage、进程退出状态、磁盘/GPU采样与最新训练step，不自动串联prepare/train/test、不重启、不关机。run与control必须彼此分离且开始时非空拒绝；日志与control状态不写进不可变训练/test归档。默认15分钟采样，子进程退出检测最长约1秒，不等待完整采样间隔。终止时仍需主执行者以具体stage完成文件内容和归档门禁验收；monitor的marker存在检查本身不验证模型/指标语义。

追加代码审计PASS。实现者的两项tiny测试分别覆盖training/test正确marker加exit0成功，以及旧通用marker加exit0仍失败；审计者读代码/diff/测试和SHA，没有另开训练或实验。
