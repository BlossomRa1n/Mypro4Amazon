# Final closeout与模型准备操作脚本审计

状态：**两helper代码审计PASS；实际封模、test、closeout及关机门禁尚须各自真实执行回执。** 审计者只读本地源码、已公开的来源/恢复元数据，并运行隔离测试；没有SSH、加载正式checkpoint、读取正式test指标/预测、训练、评分、构造候选或生成实际closeout。

| 文件，位于 `server_snapshot/full_final_test100k_20260925/` | 最终SHA-256 |
|---|---|
| `verify_closeout.py` | `4ace223c650078c0d99b0c99ec6e60815752e25cc31636e98fc12b0520963e7e` |
| `test_closeout_guards.py` | `c1b810bfbd012d40b106bda8cec2a4ee3039f9b75c8903b8defe807b9b6e0ec0` |
| `prepare_models_only.py` | `9ba8dfe28bf3c1457efc41400b505f05377b6cb48f206af348be6f6f5b2055b4` |
| `test_prepare_models_only.py` | `ea6b7d75757fc3a1ddaac4f38b2317fa760e1039feaa1d866e8991ee99dcd608` |
| `FINAL_TEST_LAUNCH_RUNBOOK.md` | `9e0333c9fa0fe0d816fdabc7dcb36b84ecb1e767b65c2b9fb7d7a1d1c2a109ca` |
| `CLOSEOUT_VERIFIER_README.md` | `942570c3bfb3957b11f9051c58bb94abc52b9db9aae54bd5e24379c4d17340d7` |
| `closeout_inputs.example.json`，仅示例，须另写实际config | `ffe5e26cf75274fbfb286334203b69169d25dcae49d4eb44441b588af0828a9f` |

## 1. Closeout身份、执行来源与完整归档

Closeout固定本轮实际authorization SHA `d14f939f9f94f6a256fd361d0b35f2a29b9204325dfeaac008d1e04a85457f5e`、final plan文件SHA `ab8caa81e32142a07095b57c445d5f8afc1c3a6ee626868ed1b379547f9dd893`，并复核canonical plan、raw-only完整合同与authorization逐字段绑定。已实际执行的source mapping回执SHA固定为 `9312bf4c3e175efeefeb8a86cc87339f40325f75162852b2fd9bd5cdee21b805`，必须仍包含完整149项来源，防止把files删为空而只保留自述verified；28项auth refs、mapping、source references键集合相等且原remote角色/本地SHA逐项匹配。

训练及test的run/control分别要求原remote路径、固定完成marker、两次inventory相等、本地精确文件集合/SHA/size、文件数/总字节及completion SHA。监督控制目录按冻结watcher校验精确5文件、完整原command、900秒间隔、无重启/关机、status/COMPLETED及全历史status一致。Formal run/control的失败证据不能由解释manifest豁免。

语义回执必须绑定同plan、completion及全归档字节集合；local_inputs键集合固定为training 6项、test 9项，helper源码键集合固定为4项，且当前字节匹配。Watcher的实际events必须包含单次start、每项归档及semantic子进程的真实argv和exit0、最终完成事件；remote_ready中的executor/monitor/plan SHA、stage/control completion和launch逐项等于实际归档。Semantic argv参数集合精确，包含原raw run及test阶段真实training run，不能借缩减回执input集合漏掉执行来源。

prepared锁必须canonical封印，绑定原plan、training completion、训练本地语义回执及唯一raw checkpoint的原remote角色/SHA/size；test completion和final completion继续绑定同prepared、checkpoint、plan和100k身份。raw-only必须 `structure_gain_accepted=null`、无paired字段、`business_threshold_registered=false`。模型tensor语义沿用已经由固定工具、实际argv/exit0证明完成的语义验收，closeout只重hash checkpoint，不重复反序列化模型。

全局test marker位于原身份锁父目录，独立于final remote root。Closeout因此额外要求真实只读下载的marker来源回执：原remote路径由auth test_lock推导、remote/local SHA和size相等、前后稳定、原始readback字节等于本地marker、returncode0及read_bytes请求匹配。再要求global marker等于test run的TEST_STARTED并绑定唯一checkpoint、prepared/plan、records及test目录。此回执须由真实读回生成，不能合成标志来冒充原路径存在。

最后要求以FINAL_COMPLETED为marker的新final remote root完整归档，固定archive工具实际argv/exit0、输出receipt/inventory SHA及stdout最终JSON与receipt一致。归档中的auth、plan、prepared、final、两control树、所有auth复制输入、已上传semantic回执和部署source manifest字节均重新绑定；旧snapshot保持原字节，不能以旧snapshot代替final归档。

## 2. 观测、错误解释和输出保全

Closeout只在各阶段证据完整后，用 `allow_pickle=False` 读已封存test数组和consumed UID。它检查100k唯一UID/position顺序、数值有限、候选数组shape、全部100k历史训练UID已覆盖、HR/NDCG/pool_hit均值和candidate GAUC有效人数/覆盖/均值。它不读取未来targets、不评分、不重建候选、不添加bootstrap。更深入的候选顺序、模型tensor和指标语义依赖前述固定语义工具的实际完成证据；full-future IDCG不能仅从候选labels重新证明。

解释manifest要求明确reviewed、unresolved为空、scan roots精确且所有解释均绑定原失败文件SHA和resolution evidence。除FAILED/FAILURE文件名，脚本也检查运营JSON的failed状态及非零returncode/exit_code，并显式注册历史错误日志，避免“没有FAILED文件”等同“从未失败”。科学文本正确性和未知错误的判读仍由主执行者审查，脚本不会自动生成豁免。

必须保留并解释两次部署缺件：

- **服务器CPU fixture首次失败。** `server_cpu_tests.log` SHA `a88c5500f6ec232bfb2b6b8cb714f0131d34e5ac532ff1fa06f7cc72262b2799`，3项0.650秒、1error，缺 `docs/TRAINING_DIAGNOSTICS_PROTOCOL_20260924.md`；补传docs后 `server_cpu_tests_v2.log` SHA `05ff0d45cd961cdb5618fc07859d3d2f1a5b12f73357ad1d3b4c1b832d260162`，3项27.044秒全部通过。两日志记录的executor和tests SHA相同。来源manifest由126项增至176项，恰新增50项docs、原126项零删零改；不能把此过程写成唯一新增一个文件。
- **正式prepare缺身份seal。** 保留原prepare_FAILURE及非零exit，绑定真实identity seal修复与sameauth prepare recovery回执。脚本独立比较远端before/after证据集合，要求唯一新增949字节固定seal、55→56原文件不变；重hash原本地55文件冻结树和全部首次prepare日志，拒绝旧归档被回写。具体实际恢复事实见 `FINAL_PREPARE_IDENTITY_SEAL_RECOVERY_AUDIT_20260925.md`。

分析报告、work summary和资源解释以SHA绑定，但非空文本本身不证明分析正确。Closeout限制字段明确不授权关机；实际恢复/可恢复依赖门禁和主执行者关机GO仍独立。

所有输入拒绝symlink、祖先symlink、非绝对路径和`..`。输出必须不存在且位于所有输入树、父目录、动态来源父目录和code/tools之外；推荐新的 `server_snapshot/closeout_gates_20260925/`，不能写入原source mapping gates父树。首次发布前复核所有捕获依赖SHA/size，exclusive写单个CLOSEOUT回执，不改任何输入。运行时所列归档与报告必须已静止，后续修改需要独立新验收，不能用既有回执覆盖新字节。

## 3. Model seal-only helper

Helper只接受显式 `--go-seal-models-only`，固定本轮auth/plan SHA，并先核已实际完成的训练、semantic、两个transfer、control和watcher执行来源。它调用上表固定closeout helper的纯training provenance函数，不读取test结果。要求当前训练归档字节与semantic evidence精确一致，checkpoint SHA匹配，失败记录不被豁免。

远端preflight和上传前再次检查固定executor、原plan、training completion/evidence/checkpoint SHA、训练monitor完成、无失败/活跃final进程、无prepared/final/test目录及原全局test marker。只向固定final root以xb上传已经本地验收的训练semantic回执，核实际payload、上传返回path/SHA/size及本地上传前后字节；随后唯一执行冻结executor的 `seal-models`。没有prepare-plan、train、test、marker创建、自动重试或关机入口。

所有输出使用新 `seal_models_v1` 目录及排他日志。每次remote调用保留command、脚本SHA、非payload请求、stdout/stderr原始字节和退出记录；超时保留捕获的partial stdout/stderr，明确remote_outcome_unknown，不能假定远端已取消。下载prepared时先以xb保存原始字节和下载SHA/size，再解析/验收；坏JSON或合同不符仍保留下载证据。最终prepared验证canonical、固定raw-only、原plan/training/receipt/checkpoint四项完整path/SHA/size、test未访问标志；成功只表示模型已准备，不能直接启动test。

运行手册已改用现有 `/Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python`，因为纯training gate导入需要NumPy，普通python3不具备此依赖。手册将test的独立GO、新鲜资源/当前状态检查、唯一monitor command、原全局marker读回及watcher观察分开说明；手册本身不是执行授权。

## 4. 独立离线验证与结论

审计者用上述venv、`PYTHONDONTWRITEBYTECODE=1`运行unittest discover：

- `test_closeout_guards.py`：最终8项，**0.214秒PASS**。覆盖路径/JSON重复键与非有限值、transfer稳定/覆盖/同size篡改、raw-only重封漂移、缺语义input/helper、观测身份/有限性/metric/coverage、原路径marker来源、真实已有身份修复差异与旧本地树只读重hash、未知故障及resolution SHA拒绝。
- `test_prepare_models_only.py`：最终4项，**0.034秒PASS**。包含完整mock成功、非零seal退出、timeout partial、坏下载原文保留和已有输出不覆盖，以及显式GO、exact prepared refs和安全路径。所有网络/正式命令均mock；测试打印的“Prepared model seal verified”是fixture输出，不是实际生产封模。

实现者另报告8项closeout 0.209秒和4项seal 0.032秒通过；不与审计者独立执行混同。两脚本没有实际closeout结果，审计PASS不能替代正式阶段成功。审计结束后再次重hash本轮全部28项auth引用仍匹配，冻结cross文档、final模板、协议及科学代码未改。
