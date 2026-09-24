# 最小早停cross新runner执行前审计

状态：**代码审计PASS；不是正式运行完成或test授权**。审计对象为独立新增的 `code/run_early_stop_cross.py` 与 `tests/test_early_stop_cross.py`，合同为冻结 `early-stop-zero-cross-20260925-v1`。审计仅本地只读检查执行代码并编写文档；没有连接服务器、运行实验、训练模型或读取test目标。

## 1. 审计版本

| 对象 | SHA-256 |
|---|---|
| 新runner | `cacafb6b0916fd5f934debeb3f1dc73d1677dea5afe0b857fa807d89782f3282` |
| 新tests | `05ccd16b5ff6f68e2498c1a4b182f8ca437241613c1681c0898fe2c17122b4c7` |
| 冻结夜间协议 | `8d6dc4a421c84f09ed4dd626c2fbb144557b00a561cbf1de1be62baa87f6d710` |
| 原diagnostic adapter | `3c5b743fcdbfbd37d2b93a6379bbef8cb441f5ffb116a75836b71375fec7ec51` |
| 原raw diagnostic runner | `67fe7d717bc21da0720a786fd11087f1b40f8a12ae3dd08a8bb6f7a52ccd944e` |
| 原diagnostic metrics | `5f4d7721f0c734ad015c222bc370a6df2d1753e9e7ada2ce1c955f1b76b00607` |
| 原raw diagnostic协议 | `45b95890f2d8d4f23d108fca0bd336878f7e835814e50b6170a8251e322851e8` |

三个原始代码文件与原协议SHA均保持冻结值。新入口从独立只读legacy目录导入原依赖，启动前核对该目录所有Python文件的名称/SHA集合与原protocol `code_hashes`完全一致，不通过修改旧协议或给旧runner添加例外来绕过冻结。

## 2. 核心比较与训练合同

新训练仅三个 `zero_cross`，seed42/43/44、init424242/424243/424244，每个1epoch。原AdamW、FP32、clip5、batch256、mixed16、原RNG/行顺序和 `CosineAnnealingLR(T_max=3)` 保留；正式screen请求0.25/0.5/1，对应591/1181/2362。评价守卫使用原已审计RNG/BN/mode保持、候选有限性及固定panel实现，重建panel须与原实际内容逐字段hash相等。

新模型共同初始tensor审计与原raw一致。epoch1实际消费trace逐字段对照同seed原raw与旧zero；只配置相同不足以通过。raw控制只在共同三个早期点重算best，正式结果必须仍为42→591、43/44→1181并绑定原锁定checkpoint。raw末期last文件不是这个新runner的必要输入，完整归档验收后可由独立存储流程管理。

只保存每个zero的best，磁盘模型原子写入峰值按4个、保留3个估算。全部3个best锁定后才统一confirm；raw复用已有best confirm数组。候选ID/length、UID/position、label和pool_hit均配对相等。主要bootstrap先对每个用户跨3seed平均配对差，再按用户抽样10,000次，seed20260925，固定双侧97.5%分位区间 `[0.0125, 0.9875]`。三个seed HR均正、均值≥0.0005、CI下界>0、NDCG均值非负四项均真才选zero，否则保留raw。输出明确 `final_test_allowed=false`。

## 3. 初稿发现的四个门禁已关闭

1. 旧zero控制不能仅信任自报 `evidence_hashes`。正式版同时重算old dev canonical seal，核对固定canonical `9e2fbaf4e8ccb10e62cc3ec356d1f74f299517f898c7dbb2c6037348f616627d`、文件SHA `1f437772b1e5418d5f09da60b97e437fc5079587f0c55296148ef43a6d9bb5a1`、旧protocol hash以及 `test_accessed=false`，再检查三个trace的实际SHA和逐字段等同。
2. 备份回执不能仅写 `status=verified`。正式版要求其绑定实际raw `COMPLETED.json` SHA及所用archive回执SHA；archive本身须formal且绑定同一raw完成SHA。备份tar/依赖可恢复性仍由独立外部门禁验证，不由这个runner重新解包证明。
3. 不能静默换计算环境。正式版将当前Python版本号、Torch、NumPy、CUDA runtime、cuDNN、Torch计算线程与受raw完成封印保护的 `run_config.json` 对照，任一不同即拒绝。平台字符串与绝对可执行路径无需固定。
4. 新round不得在截止后启动。正式版在入口按aware UTC拒绝 `now >= 2026-09-25T00:04:49Z`，即北京时间08:04:49；manifest记录实际开始与固定deadline。已合规启动的注册轮不在每个seed处重新截断，可以完成原范围。

同时补入新cross协议ID与legacy父hash到checkpoint metadata、去掉无效CLI参数、递归拒绝JSON中包括`1e999`转换成无穷的非有限数。未因统计/诊断字段扩张改变实验内容。

## 4. 验证证据与边界

实现者报告该版本两项本地测试通过，用时5.799秒。审计者已阅读对应测试代码：一个真实tiny原raw+旧zero→新cross subprocess完整链，覆盖独立legacy源码封印导入、真实trace相同、每seed三点、仅3个checkpoint、97.5%CI元数据及非空目录拒绝重跑；另一个覆盖旧manifest篡改、计算环境不一致、deadline边界与备份绑定拒绝。该结果由实现者执行，审计者未重复训练测试，也不把tiny CPU结果当作正式CUDA效果。

正式部署仍须由主执行者完成：实际source bundle与本审计SHA一致、备份可恢复实检、原诊断source/cohort/transfer归档门禁、进程与资源预检、固定窗口预算以及正式run结束后的完整文件SHA/size与checkpoint/数组可读验收。新runner的完成标记及统计晋级字段不能替代这些外部完整性条件；任一外部门禁未过时，不得开放full/test。

Full/test另由 `FULL_FINAL_PLAN_TEMPLATE_20260925.md`定义待锁合同；模板未授权，不属于本次cross代码PASS的自动延伸。最终100k test仍只限已批准身份读取，候选与future目标保持封存。
