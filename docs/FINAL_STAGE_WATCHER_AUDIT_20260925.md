# 最终训练/test归档观察器审计

状态：**代码审计PASS；不是训练、test、模型封印或关机授权，也不表示正式阶段已执行。** 审计仅阅读本地代码并运行离线测试，没有SSH、模型载入、训练、评分或读取未来目标。

| 文件 | SHA-256 |
|---|---|
| `server_snapshot/full_final_test100k_20260925/watch_final_stage.py` | `bd6542522f0460e4572d67d6715b476820337c8a4caf4ff6462b3f2489d166b1` |
| 同目录 `test_watch_final_stage.py` | `b8af34386a119814c7ef2cefda4433200f14911224936695844248b48c7904e8` |

该工具只观察显式选择的training或test阶段。它要求本地已锁formal plan，核对canonical seal、executor SHA和原plan字节；使用已有Unix SSH control socket。每次只读探测绑定远端executor、monitor、plan的字节SHA，检查失败证据、stage、精确launch command/路径/900秒策略和实际exit0。必须同时看到对应TRAINING_COMPLETED或TEST_COMPLETED以及control COMPLETED，且最终status完全一致，才进入归档。

归档分别调用已审冻结 `archive_verified_remote_tree.py` 处理run和control树，使用不同输出与回执，不把live控制日志混进已封模型树。工具再核对inventory来源、完成marker、精确文件集合及逐文件SHA/size；control文件集必须精确为launch、status、status history、training log与COMPLETED。历史状态不得包含失败/错误stage/nonzero exit，最后一条必须等于完成回执；归档后的完成内容须等于实际观察到的完成内容。

随后使用固定SHA的 `verify_full_final_archive.py` 验证本地真实模型/数组与准备链，并核对semantic回执的formal状态、stage、plan hash/文件SHA、verifier SHA及对应完成SHA。test时显式传入prepared、global once-marker、training run及其归档回执，不自动生成它们。成功summary明确保留独立root来源/授权真实性门禁，不能用watcher完成代替该门禁。

所有输入输出拒绝symlink及父目录穿越，远端路径要求规范绝对路径；现有输出、inventory、日志、失败回执及watcher目录均拒绝覆盖。审计发现并关闭了仅保护具体输入文件而遗漏其父归档树的问题：输出还必须与raw、full positions、base、old protocol、diagnostic、test身份父树、已有training run及code/tools源码树互不嵌套，避免误指output-root后破坏既有sealed archive。默认operational目录仍可正常使用。

轮询默认30秒且不得超过30秒，每次SSH进程最长25秒，总等待默认24小时；本地归档/语义子进程各有可配置、正且有限的默认6小时上限。进程失败或超时停止本地helper进程组并保留已写日志/partial文件，不向远端训练/test进程发信号，不重试、不删除。进程命令、退出和阶段状态写入exclusive本地events日志。外部信号也进入保全失败路径。

这类操作上限与夜间授权是不同边界：watcher可以在授权截止后完成已经开始的轮之归档，但不会创建新训练/test轮。按成功回执继续seal、test或finalize仍由root在其各自门禁通过后显式执行；工具没有这些动作，也没有关机逻辑。

审计者独立运行 `/Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python -m unittest discover -s server_snapshot/full_final_test100k_20260925 -p test_watch_final_stage.py -v`，8项测试0.547秒通过。测试覆盖training/test两条mock完整观察→run transfer→control transfer→semantic链、缺少完成回执、非exit0（包括bool冒充int）、来源/plan漂移、错误命令、历史失败、传输字节篡改、unsafe/已有输出、父归档保护，以及真实本地短子进程的成功/失败/超时并保留日志。实现者在最终existing-socket启动守卫加入后另报告8项0.548秒通过，审计者核对其日志及最终代码SHA。测试无SSH/真实训练/评分，不替代正式阶段实际回执。
