# 训练退化诊断：本地归档验证器独立审计

结论：`tools/verify_training_diagnostics_archive.py` 在下列SHA版本通过独立静态终审，可作为安全关机前的**本地归档完整性与可读性门槛**。它不单独授权关机；远端到本地size/SHA清单、源码/协议/control归档、原cohort精确身份及旧训练轨迹校验仍是独立必过门槛。审计没有连接服务器、打断正式训练、改训练代码或读取test未来标签。

| 对象 | SHA-256 |
|---|---|
| 归档验证器 | `6c3b9ecaec8d7494d74451020899c1a2ee6fefede539f59c6b33b488be323b79` |
| `tests/test_verify_training_diagnostics_archive.py` | `fb1ad97c8b3ccb74761934902f59a2ea923e3e3b49a2ab6bdf02539c3768ddc2` |
| 未改动的冻结训练协议 | `45b95890f2d8d4f23d108fca0bd336878f7e835814e50b6170a8251e322851e8` |

审计同时重查runner、adapter与metrics的SHA，仍分别为实施前审计记录的 `67fe7d71…52ccd944e`、`3c5b743f…fec7ec51`、`5f4d7721…6b00607`，没有因本次归档工具修补改变正式训练。

## 1. 精确归档封印与路径

归档内实际文件集合必须精确等于 `COMPLETED.json.evidence_hashes` 的keys加顶层 `COMPLETED.json`。每份证据重算SHA；源清单若有size则同时核验，没有size则独立记录本地读取的size。缺失文件、未封印附加文件、hash不同、run出现FAILED、完成状态或test标记不合格，均失败而不写成功receipt。

归档树在读取完成标记前拒绝symlink和特殊文件，包括 `COMPLETED.json` 的symlink；sealed relative path不得为绝对路径或含 `..`，解析后必须留在归档root。receipt必须在run目录之外、不能覆盖已有receipt，全部检查成功后才写入。工具只有本地读取和外部receipt写入，没有SSH、删除、关机或训练功能。

顶层COMPLETED自身未纳入自引用hash；工具把其SHA写入receipt，源封印真实性由独立远端传输inventory校验承担。工具不会把一个重新制作的自洽封印当成远端原件证明。

## 2. 模型、协议与正式规模绑定

新协议canonical manifest hash重算通过，protocol ID精确为 `raw-deterioration-20260924-v1`，test_evaluation_allowed必须false，test身份回执future_labels_read必须false，正式test身份人数100,000。smoke需显式allow-smoke且receipt始终 `formal_archive_verified=false`。

run_config核验协议hash、smoke、3epochs、seed列表42/43/44、五个screen点与协议dimensions/history/batch配置。三个seed的初始化seed、FP32、raw、model_config、协议binding一致；正式设置256/256/50/256/75且每epoch2,362步。固定panels的自身hash与两个split hash重新计算，正式512行/512用户、train16负例/screen50负例、整型有效ID与负例唯一性核验。

每个seed的best/last元数据在checkpoint payload、锁定表、seed COMPLETED、全局COMPLETED之间完全一致，6个模型文件逐SHA读取到CPU并检查tensor有限性、参数key集合与初始审计一致。best step独立由history HR/NDCG/较早step词典序重选，last必须为3epoch；checkpoint的screen指标和epoch fraction绑定对应history项。

本工具检查checkpoint可读取和内部参数/元数据一致，不创建模型进行新评分，不恢复训练或声称optimizer状态可用。

## 3. 用户、候选与指标

正式15份screen数组每份20,000用户，6份confirm数组每份80,000用户，候选宽度不超过75。uid、position、candidate_ids、lengths必须整型且符合有效范围；用户唯一、候选有效部分唯一且padding正确，labels与auc_valid必须bool，所有数组数值有限。

全部screen跨时点/seed，以及全部confirm跨best/last/seed的uid、position、candidate_ids、lengths、labels、pool_hit分别要求逐元素一致。候选与用户ID同时必须小于相应model_config cardinality。它验证内部配对一致性；原始固定cohort的精确UID/position顺序另由本地原协议/候选cache对照门槛认证，receipt明确披露这一边界。

逐用户HR、pool_hit、AUC validity与AUC由封存scores/labels独立重算；score tie按item-ID排序，AUC精确tie计0.5。汇总HR/NDCG/pool_hit、GAUC、有效人数和覆盖重新计算。NDCG逐用户值只做有限性/范围与汇总一致性检查，因为完整未来目标集的IDCG分母未在本轮数组中独立保存，工具不重新打开未来标签。

confirm每seed差值、gained/lost，以及先跨seed均值后的用户差值/总均值重新计算；bootstrap登记seed=20260924、10,000次、有限且有序的95%CI及预定四项followup门槛校验。CI不重复bootstrap；与原封印报告一致及远端原件hash验证共同承担已存CI来源校验，不能把它描述成重新独立bootstrap。

## 4. History、训练轨迹与初始审计

五个请求点及其ceil映射、history实际step与epoch fraction校验。全部3B条steps JSONL必须连续，每epoch的batch_rows/epoch与trace逐项对应；正式每epoch604,511行、100,000覆盖用户、2,362batch，负例来源总数=行数×16，stream/order digest格式及seed/global完成记录一致。

初始审计key集合、共同tensor hashes和仅cross_gate_logit白名单校验；history必须包含train/screen面板、embedding/cross/token/BN/梯度等诊断项，batch invariance记录必须passed=true且tolerance等于注册值。工具读取已封印诊断，不重新执行batch评分、不回放负采样、不重新生成初始化tensor；精确旧raw stream对照另有独立门槛。

## 5. 必须修项与拒绝验证

首轮审计发现并由实施代理仅修改归档工具/tests，已逐项复核：

| 漏检点 | 已落实修正 |
|---|---|
| JSONL的parse_constant不能拒绝`1e999`转成Inf | JSON/JSONL共用递归finite parser，同时拒绝NaN/Inf字面量和指数溢出 |
| `int(length)`会静默截断小数length | uid/position/candidate IDs/lengths强制整型再使用 |
| run_config只读取、screen未跨点配对 | 绑定完整settings/seed/point，核对全部15screen的固定身份和候选 |
| 缺少正式人数、训练覆盖、checkpoint配置和诊断字段检查 | 加入上述正式规模、trace-step、模型元数据、history和panel契约 |
| COMPLETED等symlink可指向树外 | 读取前拒绝整树symlink/特殊文件，并限制封印相对路径 |

实施代理报告 `/private/tmp/token-control-local-env/bin/python -m unittest tests.test_verify_training_diagnostics_archive -v` 通过，耗时1.131秒。测试使用真实tiny三seed产物验证成功路径，并验证以下拒绝：默认拒绝smoke、hash篡改、缺文件、重新封印NaN JSON、重新封印JSONL `1e999`、小数length、重新封印run_config漂移、symlink、重新封印screen identity漂移；失败均不写receipt。

本审计独立完整阅读最终实现、对照已封印runner真实产物schema、检查AST与文件SHA；上述tiny运行测试由实施代理执行，本审计没有追加正式训练或评分。

## 6. 独立补充门槛

只读检查 `server_snapshot/training_diagnostics_20260924/verify_source_and_legacy.py`：其设计将本地源码包逐文件SHA与protocol.code_hashes绑定，检查monitor正常完成，21份数组与旧本地cohort及candidate ID/length逐元素比较，9份训练epoch trace与旧raw逐dict比较，适合作为本工具的独立补充。

补充脚本最后修订已只读复核：入口用显式raise拒绝 `not __debug__`，防止Python `-O`/`PYTHONOPTIMIZE`禁用assert后空转写verified；旧protocol文件硬SHA、records canonical hash、candidate sidecar与新protocol引用SHA、候选payload SHA全部重验。旧dev manifest另硬锁canonical hash `9e2fbaf4e8ccb10e62cc3ec356d1f74f299517f898c7dbb2c6037348f616627d` 并重算，9条旧raw trace先比封印file SHA再作逐dict对照。上述必修项已收敛，无剩余代码修改要求；实际完整归档门槛通过仍须等待正式运行结束。

安全关机必须等待本地正式归档验证receipt、独立源码/cohort/旧轨迹回执、远端到本地完整size/SHA及control日志归档全部完成。当前这份代码审计报告不代表正式训练已经完成或已满足关机条件。
