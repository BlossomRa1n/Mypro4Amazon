# 2026-09-25 夜间实验收尾与关机记录

**正式训练、唯一10万人test、本地归档、最终分析和恢复检查均完成。服务器已执行一次关机，随后两次全新SSH均返回Connection refused。** 本记录在冻结的最终报告和工作总结之外新增，不回写已封印证据。时间均为北京时间2026-09-25。

## 结果与预算

- Raw开发诊断支持保留早期checkpoint：三个seed固定80k confirm平均HR@5从epoch3的4.159583%提高到4.687083%，差+0.5275个百分点。它是可执行的停止策略证据，不是退化单一机制的因果证明。
- 共同早停cross消融未支持去cross晋级：zero−raw平均HR差+0.0041667个百分点，97.5%区间跨零，保留raw。不能据此声称结构等价或raw早期显著更好。
- 锁定raw模型全量4,731,777行完整训练一遍，最终100,000人test：HR@5=4.520%，NDCG@5=0.0204711370，候选命中率11.871%。候选GAUC=0.802132，仅对11,871名候选内同时有正负例的用户有效。
- 最终只有raw一个模型，没有full3或zero最终对照；未注册业务阈值。完整prefix统计和静态catalog仍非严格逐行/全局日历因果。完整边界见`FULL_FINAL_TEST100K_RESULTS_20260925.md`。
- 用户批准窗口00:04:49–08:04:49，最多3轮新优化，实际新增1轮共同早停cross复验；未为用满预算继续搜索。训练、test与关机均在窗口内完成。

## 最终完整性与恢复

科学closeout实际退出0、stderr为空，绑定975项依赖，四份历史错误记录均有已核验说明，没有未解释的正式训练/test失败。关机前Root对全部975项再次计算SHA/大小，均与closeout一致；两份冻结报告保持原字节。模型有限值、实际训练UID覆盖、负例来源数量、训练顺序、候选、配对身份、逐用户指标、来源/传输清单及本地可读性由已封印各层验证器核验。

补充恢复包`server_snapshot/final_recovery_bundle_20260925/sealed_001/final_source_recovery.tar.gz`实际大小7,007,118字节，SHA-256为`ddc19f61a85ed1dcef966a25e442344a108c4b8f88d940780ebacc17fd85f97a`。生成和实际解压/恢复检查耗时6.154秒，222个选定源码文件、捕获HEAD、Git bundle及staged/unstaged差异均核验通过，原证据与源码未改变。

恢复以原129,638,964字节恢复包、Git bundle和完整本地大资产归档为外部依赖，不声称新7MB压缩包独立包含全部模型数据，也不恢复所有Git分支附着或离线系统镜像。恢复角色完整清单包括203个必需角色；实际保留1,042个文本回执参数、460项外部引用，以及12份原字节保留但不递归解释历史archive成员的来源JSON。该包捕获于关机之前，不包含本收尾文档和后续提交；这些由最终Git提交/推送及独立回执保留。

## 一次本地恢复清单准备失败

首次恢复角色生成在原secret-shape检查处退出1，未生成角色完成文件或打包目录。原因是80k历史confirm身份名单副本中的raw用户ID包含云密钥样式子串；没有输出这些ID值。原名单已被批准作为外部数据资产，副本却被当成小文本回执。修复仅为精确路径、SHA和大小的外部身份分类，原SECRET检测器、builder和科学材料不变。

随后只读预检一次汇总历史清单的目录/成员语义：明确的旧下载root、run/remote_root枚举、同字节证据副本及源码文档内部archive成员不能都当作新本地回执路径。按独立审定的有限映射处理，203个必需角色和实际closeout依赖优先，未知/漂移仍阻断。最终修补版本8项mock通过，正式v2角色生成退出0，随后原builder实际恢复PASS。v1错误、stdout/stderr、归因及v2成功证据全部保留；没有通过删除文件或降低科学门禁消除错误。

## 关机及独立核验

关机前两次只读预检没有发现训练、评分、候选构建、监控或归档任务；GPU空闲，内存OOM/配额事件为零。Root在科学closeout、完整本地归档、最终分析、恢复PASS和最后SHA复核之后单独发布安全关机授权。

`/usr/bin/shutdown -h now`仅执行一次。06:27:29原SSH连接断开，返回255、输出为空；这里不将其写成退出0，也没有盲目重试。06:28:03和06:29:21分别使用禁用旧socket、ControlMaster与ControlPersist的全新SSH连接，两次均返回`Connection refused`。满足约定的关机后SSH不可达核验；未取得平台物理电源/账单遥测，故不声称直接证明物理断电。

| 收尾凭据 | SHA-256 |
|---|---|
| `server_snapshot/closeout_gates_20260925/CLOSEOUT_VERIFIED.json` | `fabc663463b07038b11b5f7d532d08aed80ccce156a3f3a7c796f63686ca3e4a` |
| `server_snapshot/final_recovery_bundle_20260925/sealed_001/VERIFIED.json` | `38b4e62a050335dfec1a95b600ffafeae96c804b1036f5fbeb6a254e9fea1a95` |
| `server_snapshot/closeout_preflight_20260925/roles_v1_resolution_verified.json` | `b1572df2dfc5a073b2e8b7c5dce97eb371b485de00e5b919b97b22e47cc4d5c0` |
| `server_snapshot/shutdown_closeout_20260925/ROOT_SAFE_TO_SHUTDOWN.json` | `7df5c82d8e624383fca86179b65bebe01beb7fd38a0b354db7f8085b588b76f0` |
| `server_snapshot/shutdown_closeout_20260925/shutdown_execution_v1/SHUTDOWN_VERIFIED.json` | `28690d9e4b329f8259be85b18abfb6729d1cbe87de27017bbecb60e12e1f5cb1` |

本收尾记录随随后项目代码、测试和文档一起提交；提交SHA、真实push输出、远端HEAD核验和自动续办暂停状态保存在独立的`server_snapshot/git_closeout_20260925/`，避免提交内容自引用自己的SHA。原数据、模型、用户身份及服务器凭据不加入Git。
