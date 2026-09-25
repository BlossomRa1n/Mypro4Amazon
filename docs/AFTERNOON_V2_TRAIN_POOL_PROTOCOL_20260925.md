# 午后单因素开发协议：训练候选池加入 V2

版本：`afternoon-v2-train-pool-20260925-v1`。本文件须在正式新候选池构建和任何新 screen/confirm 评分前，以 SHA-256 绑定到新的运行 manifest。旧协议、结果报告、最终 test100k、operations 目录和冻结恢复证据不改写。

## 1. 授权窗口与本轮问题

本轮来自新的用户授权：北京时间 **2026-09-25 17:44:45 起持续至少三小时，至 20:44:45**。这不是此前夜间授权的延期，也不恢复已暂停的 automation。完成工作包括实现、服务器验证、独立审计、归档和结果解释，不要求持续盲跑训练来消耗时间；若工作提前完成，则继续完成验证和整理。至少三小时不构成强制截止，已注册任务未完成时按原范围继续。

本轮只问：在相同 fast 开发数据、相同 raw ranker 和相同 screen 早停规则下，将训练负例候选池中的 V2 权重由 0 改为 1，是否改善固定评估候选中的排序？已知训练池为三路、评估池为四路，且此前随机负例面板与真实候选排序曲线出现分歧；这些支持开展假设检验，不证明候选池失配就是退化的唯一原因。

最终 test100k 已在前一阶段使用，**不得用旧 test 结果选择本轮 checkpoint、权重、负例配额或后续分支**。本协议不启动新的 test、full 训练或部署；剩余身份集合不能直接视为新的可用 holdout。confirm80k 是已反复使用的开发数据，不称为新的独立确认。

## 2. 唯一处理变量与共同训练合同

| 项目 | 固定规则 |
|---|---|
| 处理变量 | train RRF 权重从 `[2,0,0.7,0.05]` 变为 `[2,1,0.7,0.05]`；只替换新分支的训练 pool |
| 评估候选 | 复用封印 screen/confirm pool，权重仍为 `[2,1,0.7,0.05]`，每人最多75个候选；不重造评估 pool |
| 数据 | 原 train100k 用户对应604,511条合法训练 prefix；train100k = screen20k ∪ confirm80k；positions、UID与历史边界保持原封印 |
| 模型 | 原 `semantic_concat`、`raw` cross；dim/token_dim256、hist50及所有其他配置不变 |
| 初始化 | seed42/43/44；init424242/424243/424244；用户表为原随机初始化，商品表仍复制原 SVD，其他张量与封印 raw 初始审计逐张量一致 |
| 负例 | 原 `TracedPrefixDataset`、mixed16；rank11–25抽2、26–50抽2、不足4从候选 fallback、余下随机补足；过滤完整合法 train_sets 与当前 target，保持去重规则 |
| 更新 | batch256，FP32；AdamW lr0.001、weight decay1e-5、clip5；原 CosineAnnealingLR `T_max=3, eta_min=1e-6`，每完整epoch后step |
| 预算 | 三seed各完整1epoch：2362 updates，最后batch95；无自动续训，无epoch2/3 |
| 行顺序 | 保持原 `default_rng(seed + epoch).permutation` 与 batch 切分；每seed完整消费604,511行，覆盖100,000名训练用户 |

“随机初始化”仅指原用户 embedding 等原随机部分，不能改成商品 embedding 全随机。固定原3epoch cosine的第一epoch路径，不将 scheduler 改成 `T_max=1`。

**负例配对的边界：**原 sampler 先抽候选负例，再用同一行 RNG 抽随机负例。改变 pool 可能改变候选选择、RNG 消费与后续随机负例 IDs。因此两分支保持相同 sampler 算法及配额，负例 IDs/完整消费 digest 允许不同，也预期不同；不声称全部负例流相等。不为追求相等另行固定旧 random 尾段、重放负例或修改 fallback，这会增加另一项算法改动。实际 source counts 与完整负例消费 digest 必须分别保存。

## 3. 旧 raw 控制的复用门禁

控制为 `server_snapshot/training_diagnostics_20260924/run/` 的 fast raw early-best 证据，而非最终 full1 seed42 checkpoint。复用前绑定旧 global COMPLETED、CHECKPOINTS_LOCKED、正式 archive/backup receipt、run_config、panels、三seed initial_state_audit、epoch1 trace、前三点评估数组、best checkpoint及confirm_best数组/metrics的精确SHA。

必须确认以下全部成立：

1. 旧控制来源、旧源码快照及其依赖可按原封印恢复，不能用当前源码替代旧哈希绑定依赖；原数据/encoders/base assets/SVD/V2/ItemCF与positions对应相同底座。
2. 表2中除训练 pool 权重外的共同合同完全一致，包括目标函数、dropout、训练行、batch、初始化、精度、优化器/调度路径、screen规则和评估 pools。
3. 旧raw前三点为step591/1181/2362；按同一规则复算旧early-best仍是seed42=591、43=1181、44=1181，与原best锁和checkpoint SHA一致。只使用原已存指标/数组，不重新读取test。
4. 新分支初始模型张量hash与对应旧raw逐项相同；新epoch1 trace的rows、order_sha256、batch_sizes与covered_users与旧raw一致。新负例digest/source counts单独保存，不与旧raw强制相等。
5. screen/confirm逐用户UID、position、candidate IDs、lengths、labels、pool_hit与AUC有效mask全部与旧控制一致；所有数组有限、checkpoint可读、shape/dtype正确。

任何共同合同变更或来源门禁不通过时，停止复用控制；不得悄悄继续后作“同条件”比较。需要重训控制或改变共同合同，必须先形成新的明确方案；当前结果不能决定如何改对照。新分支始终从原初始化独立训练，不从旧best继续优化。

## 4. 新候选池的构建与时序边界

创建新的独立 builder、wrapper/manifest和输出目录。原 `diagnostic_protocol.load_inputs` 强制旧train权重为0，因此新wrapper必须先完整验证旧输入，再明确覆盖并绑定新训练池；不松动旧loader的校验、不修改冻结旧manifest。新pool source须绑定新协议、builder/runner源码、旧manifest及base assets SHA、records/positions hash、604511×75形状、int32类型、权重、rank/融合规则、半衰期180天、ItemCF300、catalog/encoders。

优先复用冻结 `RecallPoolBuilder` 的单进程GPU路径：V2 user batch128、FP32 full-catalog打分、原top75与四路RRF；外层使用有界1024行 cache块。不得直接把硬编码CPU、fork多进程的 `build_full_prefix_pools.py` 权重改成1后加载GPU。构建与ranker训练顺序运行，不在同一GPU并发争用。

构建只允许访问合法训练行的history及已授权train-fit底座，拒绝 `targets()`/evaluation/future-target调用，并对iid/ts/rating索引和V2 `user_features(uid,pos)` 的合法训练记录/历史边界设守卫。当前目标可以经已拟合底座进入召回结果，sampler按原规则排除；不为此读未来目标过滤。

本轮保持原回顾式 train-fit/full-prefix 基准：V2、ItemCF、SVD和统计由完整合法训练prefix拟合，可能已见到某条当前训练target；完整train_sets还包含该行之后的合法训练正例。**不声称逐行严格因果或全局日历严格因果**。新守卫确保没有越过既定train边界，不消除原有train-fit限制。

cache通过合法ID、去重、长度、零尾padding、记录顺序、文件与逻辑hash验证后才能训练。中断/不完整文件保留为证据且不得重用；不能靠覆盖partial文件实现隐式重试。

## 5. 资源 pilot 与启动门禁

正式pool前先做一个不读取任何screen/confirm/test目标的固定资源pilot：**冻结positions中的前4096条合法训练行**，同GPU、同FP32、同batch128、同外层块1024和同builder。记录初始化耗时、4块耗时、总吞吐、主机/显存峰值、产物字节、磁盘/ inode、完整训练守卫结果和pilot source/hash。pilot只检验正确性与成本，不按推荐指标选择行或参数。

用最后三块中最慢的每行耗时作保守估计，正式pool估时为“初始化 + 最慢每行耗时 × 604511 × 1.5”。这是估算而非保证；同时预留至少60分钟供三seed训练、screen/confirm、bootstrap和归档。若估算不能在20:44:45前完成主要实验，则记录预计完成时间并请主执行者根据持续授权安排继续；不得为了赶时间减少seed、候选数、评估点、合法行或改变精度。若显存、RAM、磁盘或技术正确性不满足，则停止启动正式训练，继续做修复/审计/归档，不自动切另一优化方向。

pool有效负载为 `604511 × (75×4 + 4) = 183771344` bytes，另加npy headers/sidecar；若另存int64 positions，再加4836088 bytes。仅构建 fast pool，不构建4,731,777行 full pool。

资源门禁必须用**当时实测**空间与精确预算，不能引用夜间剩余空间。主执行者在本轮报告的即时快照为数据盘约3.303GB、系统盘约7.223GB可用，精确bytes与对应挂载点须写入正式资源回执。这是当时状态，不是持续保证。

注册分盘布局：新pool/pilot仅放数据盘的新运行目录，数据盘始终至少保留 **2GiB**；checkpoint、原子保存临时文件、评估结果与小型源码/日志放系统盘的新运行目录，系统盘始终至少保留 **1.5GiB**。manifest必须记录两个输出根的绝对路径、实际filesystem/device与逐阶段剩余bytes/inode，不能仅用目录名假设位于不同磁盘。local归档目的地独立核算。

每个best以原实际checkpoint最大bytes约1.2608GB估算，三份best加一次原子替换临时文件保守按 **4份约5.043GB** 计算。系统盘还须给所有screen/confirm数组、源码、logs、manifest及未落盘归档临时内容合计预留 **至少0.25GB或逐项实测预算的较大者**，再加1.5GiB不能动用的余量。预计总需求约6.90GB；只在实际7.223GB等当前空闲值大于精确剩余峰值预算时通过。数据盘新增pool/pilot预计约0.2GB，再加2GiB余量。评估数组以旧对应产物大小并加保守余量，不用压缩比猜测。

若实现采用每个screen best先保存完整CPU state clone（包括所有buffers），epoch末仅原子写一次best，须验证clone独立、selection/BN/state hash完整、与原逐点保存语义一致；此优化可减少实际IO和临时峰值，但**启动预算仍用上述4份保守峰值**，不因预期节省放松资源门禁。每个阶段及每次save前重算“已有文件 + 剩余最坏峰值 + 固定余量”，避免把已有产物重复计入待写预算。超出预算即停，报告需要扩容（主执行者建议至少10GiB）；不删除旧冻结证据，不缩减实验合同。

只可按主执行者已有授权处理已核验的冗余副本，旧冻结证据保持可恢复。记录GPU型号、驱动/CUDA/PyTorch、CPU线程、dtype/TF32开关等与旧控制合同的兼容情况；不得为了速度悄悄改精度或topk规则。

## 6. Screen锁定与唯一confirm比较

每seed只在step **591、1181、2362** 评分screen20k，依次按HR@5、NDCG@5、较早step作字典序选best。591/2362约0.250211685，表述0.25epoch仅为请求点简称。诊断包裹必须保持训练RNG、model mode与BN状态，不能因观测改变训练。若保留原固定train/random面板，则把它作为两分支共用的旧固定面板，并明确其负例不随新pool改变；不拿不同分布面板的绝对loss差解释收益。

三seed训练和best checkpoint全部锁定、SHA及选模验证通过后，才统一读取新分支confirm80k。不得先看一个seed的confirm决定是否继续其余seed。保存9份screen数组、3份confirm数组、3份best及一个全局锁；不得看confirm后换stop、加seed、改pool或复训挑选。

唯一主要比较为新V2-train-pool early-best减旧raw early-best的confirm HR@5。按每位用户先平均三seed差值，再对80,000名用户做10,000次配对bootstrap，seed **20260925**，双侧 **97.5%** 区间（分位点0.0125/0.9875）。采用较保守区间是本轮预先固定的开发筛选门槛，不恢复confirm的新鲜性，也不宣称覆盖所有历史自适应试验的家族错误率。

仅当以下四项同时通过，标记 `supports_v2_train_pool_followup=true`：

1. 三个seed各自的HR差都严格大于0。
2. 三seed平均HR差至少0.0005，即+0.05个百分点。
3. HR配对97.5%区间下界严格大于0。
4. 三seed平均NDCG差非负。

NDCG区间、GAUC、gained/lost、source counts、pool组成和各时点曲线均为描述性诊断，不增加择优接受检验。HR/NDCG分母是全体用户；候选GAUC只在同时有候选正负例者上用户等权。用户跨seed均值不等于240k独立用户，区间不覆盖全部可能训练seed的不确定性。

评估pool始终不变，改善只能报告为该固定候选内的排序改善，不能写为召回覆盖提升。若两方案选出的停止step不同，实际消费UID暴露范围也不同；结果是“新pool + 同一早停规则”的整体训练策略效果，不能唯一归因为V2某种机制或用户ID记忆消除。

## 7. 停止、失败与结果解释

- source/hash/时序/shape/init/order/候选配对/精度门禁失败、OOM、非有限loss/gradient/数组、磁盘余量不足或保存读回失败时立即停止当前正式阶段，保留FAILED和原始日志，不生成完成或晋级状态。
- 技术修复须留下失败原因、代码diff、新源码hash和新attempt目录；只可在不改变注册科学合同的情况下重试。禁止用已有screen/confirm结果决定重试或选择其中更好的一次；已经暴露的开发结果必须写入记录。
- 四项开发门槛未全通过则固定保留旧raw策略，准确写“未通过本轮开发晋级”，不写“等价”或“V2训练pool无效”。
- 即使全通过，也仅支持后续开发候选；不自动授权full、部署或新的最终test。对已消费test100k不做选型、重评分或追加显著性比较。
- 本轮不同时改user embedding学习率、负例数、cross、BN、stop网格或recall权重大小。资源不可行也不悄悄切换减LR；另一个方向须新的预注册，不能成为看本轮confirm后择优的隐藏第二检验。

## 8. 交付与 TODO

- [ ] 主执行者绑定本文件SHA，确认独立新运行目录和旧证据只读边界。
- [ ] 实现新wrapper/manifest、带合法训练守卫的GPU候选builder、raw早停runner及独立验收器；冻结代码SHA。
- [ ] 核对旧raw控制及原源码依赖、底座、positions、eval pool、初始化和共同合同，出baseline-reuse receipt。
- [ ] 服务器与本地实测资源；完成固定4096行pilot、估时和字节预算，输出资源门禁。
- [ ] 构建604511×75新train pool并全量验收；保留pool组成摘要与来源hash。
- [ ] 三seed各1epoch，保存step日志、初始审计、rows/order/batches/负例source/digest、9份screen和3份best。
- [ ] 全局best锁后统一confirm，保存3份逐用户数组及唯一配对统计，逐项输出四门槛。
- [ ] 独立验证checkpoint、数组配对、统计重算、来源链、传输文件集合/size/SHA及可恢复归档。
- [ ] 在新结果文档报告收益或未晋级、成本、限制、失败/修复历史与下一步；旧报告不更新。

本清单是本轮执行计划；实际完成状态用新运行回执和新结果文档记录，不通过改写已经封印的协议来追记结果。
