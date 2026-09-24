# Raw 训练退化诊断：实施前独立审计

审计范围：`raw-deterioration-20260924-v1` 的协议、固定100k身份adapter、raw训练runner与指标实现。审计只读取本地源码和已封存身份/来源文件，没有连接服务器、训练模型、生成test候选或访问test未来目标。

结论：在本文记录的源码状态下，没有发现阻止按封印协议启动三seed raw-only开发诊断的问题。此结论是实施前静态与本地证据审计，不是正式训练成功、因果数据认证或早停有效的结论。开跑仍须以服务器实际文件SHA、adapter完整运行与既定小型验证回执为准。

## 1. 锁定版本

| 文件 | 审阅时 SHA-256 |
|---|---|
| `docs/TRAINING_DIAGNOSTICS_PROTOCOL_20260924.md` | `45b95890f2d8d4f23d108fca0bd336878f7e835814e50b6170a8251e322851e8` |
| `code/diagnostic_protocol.py` | `3c5b743fcdbfbd37d2b93a6379bbef8cb441f5ffb116a75836b71375fec7ec51` |
| `code/run_training_diagnostics.py` | `67fe7d717bc21da0720a786fd11087f1b40f8a12ae3dd08a8bb6f7a52ccd944e` |
| `code/diagnostic_metrics.py` | `5f4d7721f0c734ad015c222bc370a6df2d1753e9e7ada2ce1c955f1b76b00607` |

新协议adapter绑定全部code目录Python文件和协议文档SHA。`baseline_data.py`、`future_window_data.py`、`token_models.py`、`run_cross_multiseed.py`、`cross_pool_cache.py` 的当前SHA均与旧封存v3相同；本轮没有偷偷修写旧训练协议或共同模型定义。三个新模块通过独立AST解析检查。

若实施前继续改动以上源码，必须补审实际差异并重新封印，而不是沿用本表声称新版本已审。

## 2. 固定名单与候选证据

adapter硬编码校验旧protocol文件SHA `4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1` 与新test identity lock文件SHA `9416c80d3c2bda822239e19a1ffd5cc8b0497c1aca252225dee26329a9cf4303`，不能仅重算一个新manifest hash就换成员。正式新test count必须为100,000，`formal_evaluation_allowed` 与新协议 `test_evaluation_allowed` 必须为false。

固定开发名单经旧 `_records` 检查人数、顺序hash、UID/raw-ID唯一性；screen、confirm不相交且并集=train，正式20k/80k/100k人数锁定。训练位置逐元素等于full base合法训练行按train UID筛选后的原顺序，并检查其SHA。

新test检查records/排序raw-ID hash、排除并集count/hash、与train身份互斥、UID/raw-ID/边界映射、实际训练行资格、合格母体人数/hash，以及锁定随机算法抽取后的实际有序名单。结合screen/confirm并集=train，可推出新test与全部开发角色互斥。adapter没有将新名单塞进旧 `all_fresh`。

独立遍历并校验本地证据引用：lock内6次、reviewed registry内305次、StageAB独立audit内26次、旧monthly registry内427次 `path+sha256` 引用，全部文件存在、路径留在证据root内且SHA一致。这是引用次数，重复引用未去重，不能相加称为764份独立证据。旧registry本身由reviewed registry的 `inherited_unchanged_from_registry` 精确引用绑定，因此其后续读取有封印来源。

候选cache先绑定sidecar文件SHA，再由原 `load_cache` 校验complete/schema、数组dtype/shape/SHA、有效length、item范围/唯一性/零padding与逻辑pool hash。adapter进一步检查原base与encoder、原source assets、原source code、records顺序/count、budget75、CF300、180天衰减、训练/评估各自RRF权重。旧候选生产源码SHA与新诊断执行源码SHA分开记录，没有以新源码冒充旧cache来源。

test守卫按锁定UID拒绝 `targets` 调用，runner又将开发target访问限定到固定开发记录；本轮无final-test入口。直接读取既有pickle的任意底层数组仍可绕开函数守卫，因此结论也依赖当前已审核调用路径：训练面板只读合法训练目标，screen面板只读授权screen后缀，confirm只在锁定六模型后评分，没有test评分路径。

## 3. 训练与指标实现

三seed初始化来自旧模板，user随机、item复制SVD；raw以外的变体不进入本轮。BPR、AdamW、epoch级cosine、clip、FP32、batch末行规则、数据顺序和混合负例调用均沿用旧raw路径。按真实steps-per-epoch映射五个点；每seed依HR、NDCG、较早step词典序保留best，完整3epoch生成last。没有基于screen结果跳seed、改停止预算或开放额外checkpoint的confirm。

评估以 `preserve_training_state` 保存/恢复Python、NumPy、Torch CPU/CUDA RNG和module.training，强制eval并检查buffer未变；梯度与optimizer/scheduler不被诊断调用修改。原epoch边界scheduler相对评估的顺序差异不会改变下一步学习率或已保存模型参数；CPU原路径一致性测试专门覆盖了这一点。

固定面板依照root已指定的512训练行×16mixed与512screen用户×50random执行，使用独立seed并保存实际正负item。screen单正例已明确为未来边界第一事件，真实75池仍用整个未来窗口。train面板先行均值再用户均值；不能以train mixed16与screen random50的绝对loss差宣称纯过拟合。

候选HR/NDCG沿用旧定义，不注入正例；排序tie按item-ID，AUC原始score tie计0.5，GAUC只对正负标签均存在的用户等权平均并报告有效mask/覆盖。六个confirm数组必须等于manifest原UID/position顺序，best/last与跨seed候选IDs、lengths、labels、pool_hit全部相等后才计算配对统计。

唯一主比较对同一用户先均值三个seed的HR差，再对80k用户10,000次配对bootstrap，95%百分位区间。代码输出的 `supports_early_stopping_followup` 只采用预定三seed方向、+0.0005平均HR、CI下界正、平均NDCG非负四项，失败项明确输出；运行完整性由前置fail-closed检查保证。该字段只有报告含义，不触发全量训练、结构比较或test。

## 4. 审计提出并已复核的修正

| 原缺口 | 已落实修正 |
|---|---|
| confirm仅检查同seed两数组身份相等 | 检查manifest精确顺序/唯一性，并核对全部seed的候选IDs、lengths、labels、pool_hit |
| batch invariance只报差值，没有门槛 | 对固定前8用户首候选单独/整池score作finite与`1e-5+1e-5*abs(reference)`检查，超阈值失败 |
| 全表embedding norm可能被未训练用户主导 | 增加固定train正例的u/v/raw product/完整cross token分位数与独立type-offset norm |
| confirm统计缺预定决策字段 | 增加报告用途的四门槛、失败项和明确decision_scope |
| 新协议未校验禁test标记、未绑定本文 | adapter load验证`test_evaluation_allowed is False`并核验协议文档SHA |
| 时间可用性只有文字说明 | 增加运行时temporal_audit，结果绑定manifest并在load时独立重算 |

## 5. 时间审计的实际能力与限制

新增temporal_audit重算合法train_mask，检查所有既定训练position小于该用户train_ends；仅使用mask内训练iid/rating/ts重算counts、popularity、mean rating、created及active catalog。item平均值比较保留与原pandas浮点聚合相容的明确容差。固定最多1024训练行、前512screen和前512confirm边界作严格timestamp历史检查，不消耗训练RNG；此处不读取test item后缀或targets。

此审计不能证明SVD/CF/V2逐训练行因果，也不能证明全局日历部署可用性。full合法prefix统计、全prefix训练正例排除、静态catalog/metadata假设仍保留。回执显式标记 `strict_per_row_causal=false`、`strict_global_calendar_causal=false`。本轮只在相同近似条件内比较checkpoint，训练3路/评估4路与16负例/75候选也只是固定分布差异，不能直接当作bug。

## 6. 验证与运行边界

实施代理报告命令 `/private/tmp/token-control-local-env/bin/python -m unittest tests.test_training_diagnostics tests.test_monitor_training_diagnostics -v` 的7项测试全部通过，耗时4.908秒。包含AUC tie/无效用户、RNG与BN恢复、best平局与checkpoint读回、三个seed smoke、真实adapter prepare/load/run集成、拒绝test目标、原raw与新诊断seed42各epoch消费trace一致、最终模型全部tensor在CPU上bitwise一致、旧HR/NDCG完全一致，以及supervisor的成功/失败child与缺完成标记。工具输出没有单独保存日志；这些执行结果由实施代理提供，本审计没有另行启动训练，也不将CPU smoke冒称GPU确定性保证。

本审计独立完成源码静态追踪、AST解析、旧共享核心SHA比对、证据引用逐文件SHA校验和协议/实现对照。正式服务器prepare对full数据及所有cache的运行时校验、CUDA smoke和实际环境回执仍由启动流程给出。

磁盘preflight按6份保留checkpoint加1份原子写入峰值、全部score数组及余量预算；formal仅允许CUDA，缺CUDA不静默转CPU。run拒绝覆盖非空目录，失败写回执且无optimizer自动续训。run_config记录Python、NumPy、Torch、CUDA/cuDNN、CPU线程数和GPU型号/容量，且被最终evidence hashes覆盖。

另审阅 `tools/monitor_training_diagnostics.py`：独立control目录、明确child命令、start_new_session、900秒默认状态采样与1秒以内退出检测、GPU探针10秒超时、磁盘/step/GPU/退出码落盘；成功必须同时满足退出0且有COMPLETED、无FAILED。它不重启、不创建额外实验、不关机。服务器启动时仍须对supervisor本身使用nohup/独立会话，且将该tools文件纳入源码归档SHA；code目录hash不会自动覆盖tools。结束后本地size/SHA/JSON/NPZ/checkpoint可读性确认与安全关机由服务器编排执行，不能由本审计静态通过替代。

下一步范围仍只有已授权raw-only训练退化诊断。新test保持身份封存；没有任何本报告结论授权使用其标签或新增实验。
