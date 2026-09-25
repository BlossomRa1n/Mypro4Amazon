# 午后 V2 训练池实验独立审计

审计时间：北京时间2026-09-25约18:13。范围为新协议、旧raw控制的条件复用、新候选builder、ranker runner、monitor与独立验收器。此记录不改写冻结协议或既有结果，也不构成新test、full训练或部署许可。

## 已确认的执行决定

- 新协议 `AFTERNOON_V2_TRAIN_POOL_PROTOCOL_20260925.md` 的SHA为 `e0b5b95aaf2d9b39f55849c41c1733906af1eb8e4ba3de14973f0859c150ac05`。
- 单一处理是训练RRF V2权重0→1，raw/concat、随机用户表与SVD商品表、自然mixed16、行顺序、优化器及原3epoch cosine的第一epoch路径不变。负例IDs和RNG消费可随pool改变，独立保存digest，不强求全部负例流相等。
- 旧fast raw控制可**有条件复用**；完整fast pool构建GO，正式训练代码审查GO仍以full cache、实测分盘预算、封印部署和监督门禁通过为前提。

## 旧基线的实际复核

已独立复核33件旧raw所需产物的文件SHA和大小，原src_v2依赖集合与协议hash逐项相等。三seed第一epoch均604,511行、2,362 updates、末batch95、覆盖100,000用户；旧前三点评估规则仍选seed42 step591、seed43/44 step1181。三份best各1,260,699,318字节。

9份screen与3份confirm数组均使用`allow_pickle=False`读取，核验有限性、全体用户指标均值、跨seed及跨screen点的UID/position/candidate IDs/lengths/labels/pool_hit/AUC有效mask稳定，screen与confirm身份不交叠。此次预检重hash模型但没有重播训练；模型可读性及完整旧归档仍由已封印原archive gate承担。

回执：`server_snapshot/v2_train_alignment_20260925/independent_baseline_reuse.json`，SHA `da9160401bf60b693f76dd8b5c0e7a95277ef59d6f382cc01121e4197b6a3c69`。其条件是新分支保持协议规定共同合同；不是全量full1 checkpoint可作为对照的声明。

## Builder与实际GPU pilot

冻结builder：`code/build_v2_train_pool.py`，SHA `bdffa2816c5adb3a708ddf04910d26848041fd85956b177000bf5b3450a6575c`。

代码审阅确认只从exact SHA的旧依赖加载，独立新输出，不使用旧CPU fork builder；原V2 GPU batch128、外层cache1024行。训练guard禁止targets/evaluation以及train_mask之外iid/ts/rating，V2 user_features需命中exact训练positions且UID一致，history_end不超过当前合法训练位置。底座仍是完整合法prefix train-fit，不把它表述为逐行严格因果。

3个本地synthetic测试仅覆盖guard/cache/失败保留；后续正式GPU pilot才验证了真实V2路径。已独立验收本轮前4096合法positions、4个1024行块、source/sidecar/arrays完整封印、int32/75候选合法ID/去重/padding、逻辑pool hash、GPU显存峰值、2GiB数据盘保留量和注册估时公式。独立回执 `server_snapshot/optimization_20260925_evening/PILOT_INDEPENDENT_VERIFIED.json`，SHA `b3985df4716c58b0a7b2717da801b258927eb0ab1a5435924f2a0982af852419`。

pilot wall约107.08秒；builder初始化约1.97秒、构建与验收约8.16秒；其余约97秒主要是旧输入加载。按注册最慢后三块和1.5倍保守系数，full fast pool约1540.18秒（25.67分钟），另需考虑输入加载；预留3600秒后仍可在20:44:45之前完成预计主要阶段。峰值主机RSS约3.46GB、GPU allocated约1.21GB、reserved约1.70GB。构建full payload预算189,656,008字节，资源回执预计构建后数据盘约3.11GB，超过2GiB保留量。

`scoring_performed=false`在此builder指未做ranker/screen/confirm/test评估；V2召回本身确实执行GPU打分，不能把该字段解释为未计算任何模型分数。pool-only验收的`formal_archive_verified=false`是因为未验ranker训练归档，不是否定实际正式pilot来源。

## Runner与保存语义

冻结runner：`code/run_v2_train_alignment.py`，SHA `e9ac6413968717a3a7544b5ea86bc57e707de5ce5b177407a81eeaf3df036ff0`。

审阅确认旧raw初始参数及buffers逐项hash对齐、原positions/hash、旧固定诊断面板在替换train pool前重建核验、负例总数逐行16、行顺序/batch/UID覆盖等旧不变量匹配，负例消费digest独立。三个固定screen点按HR/NDCG/较早step选best；完整CPU clone包括buffers，后续训练不得修改选中clone，epoch末仅一次原子保存并读回。三seed都完成后才写全局CHECKPOINTS_LOCKED，然后开始新confirm评分。

实现测试还将**相同旧pool**交给新保存路径，与原逐点评估保存路径实际小模型训练比较：seed42 best所有参数/buffers、全部screen数组以及第一epoch完整消费digest精确相等。这个局部等价测试支持CPU clone改变保存时机未改变所选模型，不用正式开发指标决定保存实现。

runner启动与19个阶段快照核算不同filesystem上的data reserve2GiB、system reserve1.5GiB，系统峰值按剩余best加一份atomic保守预算，数组/日志至少250MB或旧数组1.25倍的较大者；已经落盘的其他产物从待写预算扣除。每次save前检查并读回checkpoint。三个seed固定1epoch，不自动恢复、不自动追加种子/epoch。

唯一统计为同用户先跨三个seed平均差，再以固定seed20260925对80k用户bootstrap10,000次，97.5%区间与四项开发门槛。未通过则raw_retained；全部通过仅支持开发候选，不能称新独立holdout或直接推进新test。

### 历史精度记录的限制

旧raw environment记录了torch/CUDA/cuDNN/NumPy、线程及GPU，但未记录TF32开关。已审旧封印源码未见显式TF32/混合精度覆盖。新runner要求same版本、same GPU及线程，并对正式fresh process强制检查matmul TF32=false、cuDNN TF32=true、float32 precision=highest，保存实际值；`historical_tf32_recorded=false`、`historical_tf32_equality_verified=false`明确保留缺失。

这支持在原FP32合同及同环境下开展有条件复用，但不能把历史开关逐项相等描述为直接观测事实。该有界证据缺口必须随结果披露；不得以“已经精确验证旧TF32”为依据作过强机制归因。

## Monitor及独立验收

独立monitor：`tools/monitor_v2_train_alignment.py`，SHA `abff64c635e5199d06911148567666d4288595976c5f37c02aa32befc1a57a50`。仅stdlib；默认900秒状态采样，退出每1秒探测；data/system状态与GPU仅作观测，精确预算由runner硬门禁负责。控制目录与run/pool分开，无restart/shutdown。2个真实子进程测试独立复跑，3.045秒通过。

独立验收器：`tools/verify_v2_train_alignment.py`，当前冻结SHA `63818a9647d260be974120786523a135a365dd9c57e1ef069c07866c4522fdf2`。它不导入新训练/构建模块来验证产物，只按已绑定来源映射本地归档，验完整文件集合、payload、positions、source、初始化/行trace/CPU best、9screen/3confirm配对与bootstrap重算。模型与逐用户指标读回使用额外固定hash的原独立reader `verify_training_diagnostics_archive.py`，SHA `6c3b9ecaec8d7494d74451020899c1a2ee6fefede539f59c6b33b488be323b79`；应与新验收器共同归档。

独立测试 `tests/test_verify_v2_train_alignment.py`，SHA `97d4600403edc474f192d12b7864723107872e463683909b0d08dea84fb86e68`；9tests约4.35秒通过。覆盖真实小模型3seed产物，并对重新封印后的CI、接受决定、初始化声明、epochs、选best step、磁盘预算、confirm位置错配进行拒绝检查，还覆盖pool padding/重复ID/逻辑hash、路径/文件大小/文件集合、AUC mask和JSON非有限值。

完整正式验收尚需fullcache与run真实产物产生，之后必须重新执行此独立验收器、远端传输inventory及恢复验证；本备忘录不能替代尚未生成的真实回执。后续若验收工具因实际schema缺陷必须修复，应新增版本/变更证据，不能修改实验注册门槛来迁就结果。
