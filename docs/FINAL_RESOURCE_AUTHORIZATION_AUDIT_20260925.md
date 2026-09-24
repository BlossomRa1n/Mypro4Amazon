# Final资源授权与引用组装审计

状态：**代码审计PASS；正式资源授权、引用组装、source mapping和prepare的实际门禁须分别完成。** 审计者仅只读本地脚本和归档、运行隔离guard测试，并写本文；未SSH、生成真实授权、部署、训练、构造test候选或评分。审计读回时 `RESOURCE_AUTHORIZATION.json`、`authorization.json` 和 `authorization_inputs/` 均不存在。

| 文件 | SHA-256 |
|---|---|
| `server_snapshot/full_final_test100k_20260925/authorize_resources.py` | `fbefd99fad40f239a6ba5212b8193ebb12ca78f0be19071923815243f591f373` |
| 同目录 `assemble_authorization.py` | `ea834c888b92226a9393c71c3ba44158e3a39ea7d776d7111b140dd4210f6615` |
| 同目录 `test_rootops_guards.py` | `75b0464c6db0a78dd5f5a2ad631cb8447438eaa46af45f40790294644730dbfc` |
| 冻结final executor，仅只读核对兼容性 | `ae8756bde5ef024fbafb2078f615d348e4e7a71d51d13afa9a8bf50b284df862` |

## 1. 资源证据、预算和范围

资源脚本要求full pool全部门禁marker、semantic回执和source回执已通过；source回执必须封印当前semantic回执，二者必须共同指向当前full COMPLETED字节。已完成pool的构造与验证时间必须位于0–10,800秒。资源快照来自所有pool门禁后的固定 `resource_snapshot_all_gates.json`，记录时间不得在未来且必须在20分钟内。

本轮使用固定150分钟预算：计划/来源检查5分钟、full训练及checkpoint 70分钟、训练归档及模型锁定15分钟、test pool及评分10分钟、test归档/分析/git 20分钟、余量30分钟。它不重复计入已经完成的pool构造及pool归档。时间依据为既有raw九个epoch均值469.80219339019254秒及4,731,777/604,511行比例，外推full训练约3,677.351秒（61.289分钟）；该外推及mtime推算的pool时间均不构成新的benchmark或性能保证。

截止固定为 **2026-09-25 08:04:49 Asia/Shanghai（2026-09-25T00:04:49Z）**。资源授权、assembler和冻结executor的prepare与train入口均要求完整9,000秒预算余量。因此实际train入口最晚为 **2026-09-25 05:34:49 Asia/Shanghai**；prepare曾经通过不豁免train实际启动时的期限检查。20分钟freshness检查由资源授权和assembler执行，后续运行前的资源状态仍须主执行者确认。

磁盘预算不再依赖未封印的stat值。脚本先要求raw归档回执为verified且其completion SHA等于当前raw COMPLETED，再以流式SHA核验实际seed42/raw/best.pth等于COMPLETED evidence seal，随后读取大小。资源回执记录raw archive、raw completion与checkpoint SHA，assembler再将这些值绑定当前raw引用。无需加载checkpoint或读取模型内部对象。

现有checkpoint为1,260,699,318字节。预算为两份checkpoint 2,521,398,636字节，加160MiB test数组/池余量167,772,160字节及1GiB保留1,073,741,824字节，合计 **3,762,912,620字节**。既有full pool已分配，不要求删除checkpoint。冻结test路径的数组/池保守大小134,300,000字节低于160MiB。另要求远端root分区空闲至少512MiB、本地归档空间至少8GiB。

资源门禁同时要求cgroup内存上限至少48GiB且当前空闲至少20GiB、CPU quota至少4核、单GPU。GPU型号固定为本次时间外推所依据的 `NVIDIA GeForce RTX 4080 SUPER`，利用率不超过5%、已用显存低于100MiB、总显存至少16,000MiB。上述状态是快照证据，不保证运行期绝无资源竞争。

## 2. 引用、封印与写入顺序

Assembler限定已封cross selection为 `raw_retained`；cross source、full pool semantic/source回执通过，且pool回执与当前full COMPLETED的SHA关系一致。它要求resource状态approved、固定raw-only模型集合、optimization_search_closed=true、两项test访问标志均false、完整9,000秒及精确分项预算、固定deadline、当前snapshot/time/full SHA与resource封印一致，resource review及snapshot均不超过20分钟。

原remote直接引用路径包括raw、cross、full/pilot池、诊断manifest和原test身份锁。需要复制的回执和说明文件排他放置于固定 `authorization_inputs/`，每份副本写后重读SHA；auth保留原remote引用与SHA，local mapping记录对应原本地文件，copy manifest记录实际待上传副本及auth字节。

新增引用包含cross control传输与source实际重验执行回执、full pool四项归档/传输/source回执、cross结果说明、resource review、resource snapshot和time evidence。冻结executor要求REQUIRED_REFS为引用键集合的子集，并对所有引用逐项检查SHA，因此新增引用兼容，未修改冻结executor。独立source mapping工具要求mapping与auth refs键集合精确相同；新增引用也纳入字节核验，不能因不在旧REQUIRED_REFS内而漏检。

Source mapping回执必须写在独立 `server_snapshot/final_gates_20260925/source_mapping_verified.json`，不能放在authorization、mapping或其他输入父树内。Assembler定义了该GATES目录但不自行生成mapping成功回执；仍须实际运行已审mapping工具，再由prepare-only脚本核验完整上传集合及回执。

两脚本使用显式require，不依赖会被Python优化模式移除的assert。输入和输出拒绝symlink及祖先symlink、`..`路径。Assembler在首次mkdir或copy前统一检查全部输出及只读树边界、lexists和冲突，副本用xb、JSON用x，不覆盖既有证据。发生中途失败时保留partial，不能删除后盲重跑。资源脚本只排他创建resource review，assembler只写本地staging/auth/mapping/manifest；均无远端、训练、评分、删除、重试或关机入口。

## 3. 隔离测试与科学边界

审计者独立运行 `PYTHONDONTWRITEBYTECODE=1 /Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python -m unittest discover -s server_snapshot/full_final_test100k_20260925 -p test_rootops_guards.py -v`，4项测试0.010秒通过；同一命令增加 `-O` 后4项0.010秒通过。测试使用临时小字节fixture，不调用main，不触碰真实授权、SSH或训练。覆盖当前资源/来源SHA、raw-only和closed范围、预算/期限/freshness、test隔离标志、raw checkpoint及归档漂移、GPU型号/忙碌状态、输出全量preflight、已有文件/悬空及祖先symlink/路径穿越、assert缺失和xb复制。实现者另报告4项 `-O` 0.012秒通过，二者记录不混同。

本审计不代替实际full pool门禁、资源授权、source mapping、prepare、训练后本地语义/传输归档或模型锁定。后续顺序仍为prepare→train→本地归档→seal-models→一次test→本地归档→finalize；所有阶段须由主执行者按实际证据放行。原test身份锁父目录的全局一次性marker不能因复制身份材料而换路径绕过。

最终raw-only分支只能报告锁定full1模型的绝对test指标；没有full3对照，不能据此声称full1已证明优于full3。cross confirm区间跨零不构成等价证明，不再开放模型、停止点或seed搜索。审计中发现的必须修复项均已关闭，冻结协议、三份诊断代码、cross协议及final模板未更改。
