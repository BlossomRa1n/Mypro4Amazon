# 最终恢复依赖闭集只读审计

结论：**截至本次审查，未发现已完成科学范围缺少必需的大型本地数据或模型依赖；最终train/test/fresh-root及恢复补充包仍待实际完成，不能据此宣布最终恢复或关机门禁通过。** 本审只读元数据和源码，依赖原93项在北京时间2026-09-25 04:22的已记录SHA/size复核，不重复GB级SHA。没有SSH、模型加载、实验、评分、正式test指标/预测或未来targets读取。

审查来源：`server_snapshot/closeout_preflight_20260925/existing_evidence_index.json`（SHA `f1429d166212d5ca12c232ea6d152f641bf5116fdee40c5358f32d85c4e283a2`）、`final_recovery_receipt_roles_plan.json`（SHA `2a9688e86b36410253c50f651762b4c929d60539561d0641e567b7b3fe2d5575`）、`server_snapshot/final_gates_20260925/source_mapping_verified.json`（SHA `9312bf4c3e175efeefeb8a86cc87339f40325f75162852b2fd9bd5cdee21b805`），以及它们绑定的原恢复preflight、外资产清单和既有transfer inventories。角色计划是待主执行者审核的计划，不是新验证PASS。

## 已覆盖的必要范围

| 必要依赖 | 已有本地来源及覆盖 |
|---|---|
| 基础数据与编码器 | `next_cross_20260924/base_archive/data.pkl`、`run_manifest.json`；编码器不是遗漏的独立下载资产，数据对象与旧协议绑定data_id/encoders_hash。只审源码和manifest，未反序列化数据。 |
| 召回与初始化资产 | 同目录 `v2_best.pth`、`svd.npy`、`itemcf.pkl`，与前两项一起在原93项中。冻结runner直接使用这些已存资产，不需要重新取得原始Amazon数据才能恢复本次封印运行输入。 |
| 旧协议、候选和身份 | `protocol_complete`的19项及`locked_test100k`的5项、原identity_evidence tar，均属于原93项；58,867,570-byte旧protocol作为外部payload保留，不能强塞进16MiB上限的small receipt supplement。 |
| Raw科学运行与legacy源码 | 原transfer的511项：run 60、remote_root 451；包含独立 `src_v2`、diagnostic protocol、控制和运行证据。恢复legacy时用此实际封印树，不能用当前Git工作树代替。 |
| Cross及fullpool | Cross run/control inventories为39/95项；fullpool run/control为8/15项，另保留pilot tar与解压来源。各自原语义和transfer回执已存在。 |
| Git历史及当前源码 | 原129,638,964-byte恢复tar内Git bundle为63,189,581 bytes，并有既有restore回执；后续源码/报告/receipt变动由已审补充helper捕获，仍以原tar作为外部依赖。 |
| 最终新增模型与test | 应由最终training/test run与control、prepared/marker/final及fresh-root完整归档覆盖；本审时尚为pending，等待实际完成的语义/transfer/closeout与补充恢复回执。 |

本次仅stat核验：source_mapping的149项全部存在且大小与回执一致，149/149可归入原93、raw511、cross/fullpool inventories或小receipt角色，无未分类必要路径。Raw511、cross39/95、fullpool8/15的全部文件也存在且大小一致；非final必需角色没有缺失路径。此次未重算这些payload的SHA，原SHA有效性依赖列出的既有门禁和主执行者要求的最终漂移检查。集合之间有重叠，不能简单相加计作唯一文件数或唯一备份字节。

## 仅需补齐或明确的两类小资料

1. **把 `server_snapshot/full_final_test100k_20260925/local_archive_runtime.json` 纳入最终receipt角色。** 它记录本地归档校验使用的非symlink Python实际路径、NumPy/PyTorch版本及venv site-packages的PYTHONPATH说明。closeout会绑定operations树，但源码补充helper排除该树，因此显式保留此小JSON有实际恢复用途。
2. **若交付承诺包括重建同远端软件栈，关机前补一份小的只读环境回执。** 已有raw run_config和final_plan记录Python3.12.3、PyTorch2.3.0+cu121、NumPy2.5.2、CUDA12.1、cuDNN8902、线程、Linux/glibc及GPU名称；未找到完整远端已安装package name/version/build或GPU driver记录。冻结import链还用pandas、scipy、scikit-learn、tqdm。可由主执行者只读获取包元数据、Python/OS/glibc和driver后放独立sibling目录并绑定，不需复制整个环境或GB级安装包，不导出环境变量/凭据，也不开展运行实验。

原恢复tar中实际存在 `recovery/environment_freeze.txt`，352 bytes，SHA `eacd12f67abca1e5c4569e9b2ca896b7a5a3825f62a8296926ae2df3815835ae`；它记录本地PyTorch2.14.0等包。原recovery_preflight已明确这不是完成训练的远端环境，不能把该freeze当远端2.3.0+cu121的完整环境锁。未备份wheels、容器镜像、驱动二进制或整机镜像，意味着当前方案不承诺完全离线重装、逐位GPU结果复现或任意未来平台兼容；科学证据、源码和已封印输入模型恢复与这些更强承诺应分开。

## 恢复时的范围与路径

原93清单中每项都有绝对local path/SHA/size，角色计划的group级 `original_assets_and_raw.local_path=null` 表示由该清单逐项定位，不是漏掉整组资产。所有外部payload须继续留在本地，回执本身不能恢复GB内容。尤其应保留raw完整remote_root，使 `remote_root/src_v2` 可恢复到原 `/root/training_diagnostics_20260924/src_v2`；原base/protocol路径和28项authorization refs按已存mapping恢复，不能修改冻结manifest路径来伪装一致。

本次没有重跑clone/恢复或重新验证checkpoint可读性，也没有验证还未产生的最终文件。最终两报告冻结、完整closeout、补充包restore以及外部payload门禁通过后，主执行者才可独立决定关机。无需因本次范围审计机械复制额外无关文件、扩大原始数据收集或重复已有大型证据。

## 环境补充后续核验

运营执行者于UTC2026-09-24 20:57:43–20:57:44（北京时间2026-09-25 04:57:43–04:57:44）完成只读远端环境捕获。审计者随后回读 `server_snapshot/closeout_preflight_20260925/remote_environment_capture/execution_receipt.json`，SHA `c034a93b93e113adade6569ca3e262be7b90e86a67fc881369d4b822097217d9`、2549 bytes，真实exit0、stderr空；原stdout为18,409 bytes，SHA `6bc29b73128f6d6cd592089aac5d86ad835133889dd641412726d6c67b151bc5`，小文件字节重核一致。

原文含159个installed package name/version与74个conda name/version/build、Python3.12.3、glibc2.35、Linux/kernel/machine及GPU driver595.91.07。冻结运行相关依赖包括torch2.3.0+cu121、NumPy2.5.2、pandas3.0.5、scipy1.18.1、scikit-learn1.9.0、tqdm4.66.2。真实命令仅读取这些元数据与nvidia-smi的name/driver，未输出环境变量、source URLs或direct_url，也未更改环境。

此补充消除了上述远端包版本/driver资料缺口。后续独立 `final_recovery_receipt_roles_plan_v2.json`（SHA `b064e30974f9eff883c52f42f3188fd3afe34f9aa888724c1d69d080d2947c8c`）已新增execution receipt、原stdout/stderr及local_archive_runtime四个明确角色；审计核原199角色零删零变、原449引用边保持，仅新增4角色与2引用边。V2共203角色、127现有小凭据、64pending，仍待主执行者最终审核，不是完成或打包授权。环境资料不是离线安装包、容器或整机镜像，也没有开展环境重建实验，原更强复现范围限制不变。
