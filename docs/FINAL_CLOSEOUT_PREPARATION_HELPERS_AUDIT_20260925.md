# 收尾说明与恢复角色绑定辅助程序审计

结论：**在 Root 已完成正式 test 本地语义门禁、finalize 与 fresh-root 归档，并停止 operations 写入的前提下，两项本地辅助程序的受限职责代码审计通过。** 它们只在独立 preflight 目录创建新文件，不执行训练、评分、打包或关机，不替代正式 closeout、补充恢复实际执行与 Root 授权。

审计范围为 `server_snapshot/closeout_preflight_20260925/` 内以下原字节：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `prepare_closeout_explanations.py` | 11,579 | `a0e4bf5fa052451ca11d6a768dba7685bd13c8936a58dcb1395aaa6b2d891d65` |
| `complete_final_recovery_roles.py` | 12,622 | `5792e30273539ac8b8820e737406850f2ff7861def4b5141559807a0f886135b` |
| `test_closeout_preparation.py` | 5,400 | `9e63855b43a14c753f5cf2ec78ef25eb6b25e7880734ec1134fcfabce3a06b2c` |

## 1. 收尾说明

输出必须是独立 preflight 目录的全新直属子文件，路径及父路径拒绝 symlink、`..`、覆盖。Final 模式要求显式 Root 复核和 writers finished 标志、实际 FINAL_COMPLETED、test watcher、fresh-root transfer 与 transfer execution 终态。它重核已知失败、修复与来源证据的 SHA/大小，并重新扫描 operations 与 final remote-root。

已知三份 prepare 错误记录归属同一次缺原 identity seal 的失败；修复只补原 949 字节封印，原 55 项不变并由独立 prepare recovery 通过。首次 CPU 缺协议文档与后续三项通过也分别保留。两份实际部署清单证明原 126 项不变、增加 50 项 docs，不把当前本地加强后的 cross 验证器误说成旧远端相同版本。

未知本地失败、内容改变的归档副本、正式阶段失败不能通过自动解释消除。字节相同的已知远端原路径归档副本可单独列明。扫描识别失败文件名及 operational JSON 顶层非零 returncode/exit_code；不解析正式 test_run 内指标 JSON。原 CPU 错误日志靠显式历史来源纳入，不能仅靠 FAILED 文件名搜索。正式 closeout 还会独立扫描全部绑定树，因此此 helper 的 reviewed 状态不是 standalone 全量验收。

实际 `closeout_explanations_reviewed.json` 已读回为 reviewed、unresolved=[]、四项 resolved；原字节 SHA `b1e77c97bb9ad0c2ae6943fa4f3428b2e1ed12c7f1ea1411af7a444e1a341b8a`，17,112 字节。选择此独立文件的 config 必须在 operations 冻结前定好，closeout 后不得复制文件回 operations 或改变 config。

## 2. 恢复角色绑定

角色生成器固定 v2 plan、report path addendum 及原 recovery builder 的 SHA；complete 要求真实 closeout schema/status、准确报告/summary SHA 以及 closeout 绑定的 reviewed explanations。原固定 small receipts 漂移拒绝，显式 pending_finalization 的 watcher history 才取实际最终字节；缺失角色拒绝发布。

Eligible small receipts 先过原 builder 的大小/文本/敏感内容检查，最后再次核 SHA/大小。生成器补入实际 closeout 的依赖和 JSON 显式本地 path+SHA 引用，relative path 必须唯一落到 repo 或当前回执目录；歧义与缺失拒绝。原远端绝对路径保留为来源，不冒充本地。生成的 packaging argv 仅供 Root 审核，脚本不调用它；ready_for_root_packaging_review 不是恢复 PASS。

审计发现并修复过一个实际阻断：旧生成器未将 `.pyc` 归为 binary，而 operations/fresh-root 已有 26 个 bytecode 文件，closeout 全量绑定后会被错误送入文本 NUL 检查。实现者将 `.pyc/.pyo` 及明确编译二进制加入统一 external-only 分类；所有原文件保留，以旧 SHA/实际大小为 external evidence 引用，不读取、复制进 small text 或删除。递归发现的 binary/large 显式引用也保留身份与来源，避免直接跳过造成清单缺项。重复相同身份去重，大小不符或冲突拒绝。

GB 资产、原 tar/Git bundle、旧大 protocol 和 binary 仅继承已封印 SHA 并核现存大小，生成器不重新 hash；因此它不独立排除 closeout 之后同大小的大文件字节变化。Root 必须保持已验归档不可变并留存完整外部树。小回执的递归覆盖只处理显式 path+SHA，最终完整性仍依赖 Root 审定角色集、已有 mapping/transfer 清单和实际 builder restore PASS。源码选择由原 builder 管理，不承诺恢复全部 Git branch/ref 名称或离线环境镜像。

## 3. 隔离检查与限制

审计者以 `PYTHONDONTWRITEBYTECODE=1 python3 server_snapshot/closeout_preflight_20260925/test_closeout_preparation.py -v` 独立执行，**5 个 synthetic tests、0.021 秒、OK**。覆盖已知同字节失败副本、未知/变更/正式阶段失败拒绝、无明确 review 时不输出、引用漂移/symlink 拒绝、`.pyc/.pyo` 外部身份保留及禁止 binary read。未运行包含模型训练的 tiny fixture，没有 SSH、正式训练/test、future targets 访问、重新评分、打包或关机。

这五项不是完整真实角色图/打包的端到端测试。真实 complete 仍可能因历史引用歧义、过大文本、缺件或敏感扫描而安全拒绝，届时应保留输出并针对缺口审定修正，不能靠删除原证据或降低正式门禁解决。实际 closeout 和恢复结果由后续真实回执决定。
