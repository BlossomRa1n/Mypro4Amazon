# 最终本地源码恢复补充包审计

状态：**代码审计PASS；尚未执行正式补充包，不是完整恢复门禁或关机授权。** 审计范围是本地helper源码、测试及运行说明，审计者只执行临时Git仓库内的隔离synthetic tests，没有SSH、训练、正式实验、评分、test未来targets或正式test指标读取。原恢复包和冻结授权引用未改。

| 文件，均位于 `server_snapshot/final_recovery_bundle_20260925/` | SHA-256 |
|---|---|
| `build_final_recovery.py` | `8dddc7e47703b37fe65237cc10e4cfe1a3697036875e47b8a123c5ba24d0725a` |
| `test_build_final_recovery.py` | `7789c6b7daa46ef31c9b14fede4d0de471034064c0df39f3d43ecf1d8ed3da03` |
| `README.md` | `ce2a0f8f47d17215775c2616b76604dd6f98cfd7d4230fd55f924bf0b5d891ca` |

## 1. 覆盖及执行边界

Helper仅接受本地plan或execute。Plan只报告源码规模与估算，不创建正式包；execute要求最终报告完成声明、报告精确相对路径/SHA、至少一份明确传入的证据回执路径/SHA，以及helper目录下全新输出子目录。报告必须属于实际收录的源码文档集合。失败保留现场和FAILED回执，不自动重试或覆盖旧目录。

收录集合来自Git tracked与nonignored untracked，并限定为合格根目录文本/configuration文件和code/docs/tests/tools/archive内合格文本后缀。收录已删除路径以复原删除状态。生成证据树、server_snapshot、数据模型格式、环境和credential路径均排除或拒绝；路径及祖先拒绝symlink和父目录穿越，文本拒绝NUL、疑似secret内容和单文件超过16MiB。ignored helper的源码、测试、README作为明确的窄例外收录，不泛化到server_snapshot。

最终报告、显式回执、原证据index、旧恢复回执都进入新包；原大资产与旧Git bundle仍是外部依赖。Helper不推断“全部final receipts”，也不检查传入回执的全部领域语义。正式执行前，主执行者必须锁定完整回执清单并核已通过的训练、test、来源、传输、控制、checkpoint、marker、final、fresh-root及恢复依赖门禁。仅传入任意一份JSON不能证明全部证据可恢复。

## 2. 原包与恢复验证

旧恢复archive及restore receipt按index中的SHA只读核验，旧restore receipt必须为verified且绑定旧archive。Helper仅从旧tar提取唯一regular的 `recovery/repository.bundle` 到新验证目录，执行Git bundle verify；捕获的全部refs和HEAD必须已存在于此旧bundle，否则停止，不能冒用陈旧历史。原包不会被回写，Git历史也不会被新建正式commit改变。

新补充包保留工作树字节/SHA/size/mode、捕获HEAD/refs/status、限定路径的binary staged/unstaged patch、引用映射及manifest。打包后只允许canonical、唯一、regular、无link且大小受限的成员解压到新目录，再核成员集合及SHA/size/mode。恢复时clone旧bundle且不checkout全历史资产，明确将HEAD detached到捕获commit并核 `rev-parse HEAD`，加载该commit到index，仅checkout选中历史源码，依次应用staged与unstaged patch，再覆盖选中untracked文件。

复原验证核所有选中文件的SHA/size/mode、删除状态、精确staged/unstaged diff，并在末尾再次核实际HEAD。成功回执的 `restore_verification.captured_head_verified` 记录真实恢复HEAD。发布VERIFIED前，重新核当前repo snapshot、源码inventory、报告、全部显式回执、index、旧archive和旧restore receipt；漂移则停止并保留失败证据。

此验证证明捕获HEAD与所选源码、index状态可以复原，不证明恢复clone具有原branch attachment或全部原refs命名；clone可把非默认分支置于remote refs。原refs名称/OID被记录且其bundle来源已核验。历史大资产不落入恢复工作树；完整大资产恢复仍依赖旧archive、index记录及最终门禁。回执中的 `original_evidence_unchanged` 不应外推为helper重新逐字节核验了index列出的所有大资产，helper仅重核自身实际依赖的旧包/回执及明确输入。

## 3. 审计修复与独立验证

首版存在真实HEAD证明缺口：`git clone --no-checkout` 后仅read-tree捕获commit，可能使index/文件正确而clone的HEAD仍指向旧bundle默认分支。实现者修订为 `git update-ref --no-deref HEAD <captured-head>`，并在恢复前后核实际HEAD。新增case构建默认master与captured分支分别指向不同commit的bundle，验证恢复HEAD、detached形式、对应源码及原source snapshot均正确，同时README明确refs恢复范围。

审计者独立运行：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 server_snapshot/final_recovery_bundle_20260925/test_build_final_recovery.py
```

**13项synthetic tests，3.355秒，全部PASS。** 覆盖staged/unstaged/untracked/删除/可执行mode roundtrip、原证据不变、HEAD多分支恢复、plan不创建包、报告完成与SHA门禁、secret抑制、source/output symlink、陈旧Git历史、恶意tar路径/link/重复项、payload篡改及捕获期source/receipt漂移。实现者另报告13项3.373秒PASS。所有测试使用临时合成仓库，不是正式项目补充包或正式恢复门禁已经通过的事实。

## 4. 正式执行条件

正式包应在最终结果报告及工作总结完成、全部最终回执已固定后，且在任何新Git commit与关机前执行。输出选择全新子目录，命令绑定本审源码SHA和完整显式证据清单；执行后须独立检查新archive、manifest、VERIFIED、真实captured HEAD、所保留验证目录及外部依赖的本地存在性、SHA/size和可读性。完整closeout、无active任务及主执行者明确关机放行仍是独立条件。
