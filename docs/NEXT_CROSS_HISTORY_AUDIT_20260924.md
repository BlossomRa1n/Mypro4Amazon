# 本月历史用户独立审计（2026-09-24）

本次按用户最新授权，只排除 **2026-09-01 00:00:00 至 2026-09-24 01:54:44（Asia/Shanghai，左闭右开）**期间已用于评估或选型的用户。历史训练前缀仍可使用。采用明确记录的保守超集后，本月身份排除审计闭合：排除并集为 **796,912 个 raw user ID**，当前全量 future-window 基座中满足训练及评估条件且不在该并集内的用户为 **17,296 人**。这是动态 `all_fresh` 的身份容量，不是新模型结果。

审计从原始来源重新按月组装，没有把旧 all-history partial 文件改名或直接改成 complete。旧 `source_registry_partial_20260924.json`、`historical_users_partial_20260924.json` 和旧阻断凭证原样保留；其中六月来源缺口已经超出本轮授权范围，不再阻断本月实验。

## 1. 冻结口径与身份边界

```json
{
  "schema_version": 1,
  "kind": "month_to_freeze",
  "timezone": "Asia/Shanghai",
  "start_inclusive": "2026-09-01T00:00:00+08:00",
  "end_exclusive": "2026-09-24T01:54:44+08:00",
  "exposure_policy": "evaluation_or_selection",
  "training_prefix_allowed": true
}
```

该对象按 sorted keys、`ensure_ascii=True`、紧凑分隔符计算的 SHA-256：

`808016bf27329ad69c54e0b7377ae5fae9a3168a12c1ecbb1f5d275eaff53891`

current full base data ID 为 `754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158`；原始 `data.pkl` SHA-256 为 `e71413ee4eead36954650c1ddb99c8334ead9edecee75ab6b9c01012efd9bb06`。该基座有 828,005 名非保留 raw 用户，其中 796,197 名同时满足当前训练和评估资格。

容量审计只保留 raw 用户身份、训练索引和窗口索引边界，没有调用 `targets()`，没有生成候选或运行模型，没有计算任何新指标，也没有查看新 fresh-test 的未来标签内容。被动 pickle 读取器丢弃其余状态；这是代码级读取约束，不是密码学封存。

## 2. 来源闭合结果

登记表共有 **296 个来源条目**，由下列来源组成。`conservatively_covered` 表示为确保覆盖可能多排除一些用户，不表示超集内每人都实际在本月接受过评估。

| 来源 | 登记数量 | 身份证据与处理 |
| --- | ---: | --- |
| 本月带 UID 的逐用户 NPZ | 239 | 原 NPZ SHA、数据身份与原编码映射；只读 UID，不读指标或标签数组 |
| 本月缺 UID 的逐用户 NPZ | 25 | 原选择边界 + 原 cache 指纹 + suite manifest，保守覆盖固定记录集合 |
| 本月预测 JSON / 用户 CSV | 27 | 重新核对原文件 SHA，身份提取仅保留 `user_id` 字段 |
| 九月 Stage A/B 及同一旧验证全集 | 1 | 真实九月日志 + 原评估代码 + 完整旧验证集合重建；保守覆盖 774,585 用户 |
| 九月 negative/common-cross 的 100k 基座 | 1 | 整个已用于本月实验的 100,000 用户基座作为保守超集 |
| 可能属于九月的 undated offline smoke | 1 | 原日志与每品类前 10,000 行身份重建；保守覆盖全部 3,034 原始用户 |
| 已删除的 `token_smoke_20260920` | 1 | 同轮原始文档、原代码接口、项目基座构建清单组成的有限历史推断；保守候选并集 1,199 用户 |
| 本地 `future_prepare_smoke_20260919` | 1 | 原 seed42 抽样规则和原始配置重建，188/142/28 三项计数完全吻合；覆盖全部 30 抽样用户 |

全面 NPZ 盘点比旧恢复清单补出 **124 个** `evaluations/*/users.npz` 来源。两个 `fresh_validation` 输出具有相同的 100,000 raw 用户，其中 **2,918 人**不在原 partial 并集中。共 **111 个** corrected-record hashes 已与原指标元数据及重建记录精确核对；其余 13 个融合 NPZ 没有 records hash，但保存的 UID 仍已用原数据映射恢复。

### 25 份缺 UID NPZ

这些文件来自初始 token suite、control 和 control_smoke4。原 suite manifest 没有 eval_records hash，但原 cache 文件名对包含 records、data_id、V2、ItemCF 和召回配置的完整 payload 做指纹。使用当时的 `fingerprint`（普通 `json.dumps` 分隔符，不能改成紧凑分隔符）重建后，以下四种 cache 指纹均在原九月 22 日 inventory 中匹配：

- 100k screen：`2e6e630b051374e923117f25ecebd3b7b621af622316c57dab1de98e5bb57dbd`
- 100k test：`278152944aeabd9f98df38f0b8e4a5f9adc21d6eebdf9e6c3be499fe80880d42`
- 300 screen：`f0fdcd56f5343fc07fd917bfe8ec0fff0adaabb8914d3c7dc5cdda03f18ac71a`
- 300 test：`ab6aedd158a6e823c90ff575e07241f460b80e1c8d23bc37828533bf28873d82`

该证据足以保守覆盖所用用户，但不构成缺 UID 指标数组每行与身份之间的独立配对证明。本次没有利用这些数组重新计算模型指标。

## 3. 为什么只排除本月仍留下 17,296 人

原 `stageAB_summary.txt` 的内部时间将两阶段实际运行定位到 **2026-09-10 10:09:57—12:24:44**。原 `train_din_ext.py` 在筛选后的旧验证用户里使用全局 `np.random.choice` 抽取每次 10,000 用户；没有原 RNG 轨迹，不能假装精确重建每次抽样。

因此排除其完整验证用户超集。原五品类 CSV 保留 rating 1/2/4/5、按 `(user,item,timestamp)` 去重，随后历史 LEFT JOIN 中 review JSONL 的重复键会增加行数。按原键重数重建后得到：

| 原始统计 | 精确重建值 |
| --- | ---: |
| CSV 去重交互 | 7,969,371 |
| 合并后交互 | 8,009,352 |
| 训练交互 | 6,117,035 |
| 训练用户 | 831,494 |
| 验证正反馈交互 | 1,701,196 |
| 验证 raw 用户 | 774,585 |

重建验证集合与 2026-08-28 旧预测 CSV 用户集合完全一致，双向差集均为零。**八月 CSV 只作为身份复原旁证；真正让这个超集进入本月排除的是九月 10 日的实际 Stage A/B 评估。** 不能把 774,585 人写成“本月实际全部评估过”。

undated offline smoke 不以文件 mtime 推断运行月份，作为可能属于九月的来源保守纳入；其 46,708 原始去重交互、46,752 合并交互、3,034 训练用户及 2,887 验证用户与原日志吻合。全部 3,034 输入用户的超集已经纳入。

## 4. 删除 smoke 的有限证据推断与限制

`token_smoke_20260920` 的原 manifest、启动命令和旧 Windows 逐用户档案没有恢复到。本次不声称恢复了这些文件。

保守闭合采用以下可核查推断链，并经另一名独立 Ultra 审计者复核：

1. 最初的 `cf82d1e` 文档已同时记录“固定 future-window / CF300”、该 smoke 完成两变体 Top-5 与 paired comparison；协议语境不是事后补写。
2. 同时封存的原 runner 必须传入 `--base-run`，只读取已有 data、SVD 和 checkpoint；不创建新基座、不重采样评估用户。它从基座的固定 `evaluation()` 前缀取记录。
3. 原 `TokenSuite.targets()` 无条件调用 `self.data.targets(records)`；原 corrected `BenchmarkData` 没有该方法，不能完成所记载的成功评估。兼容的当时项目构建是全量 future 基座、小型 future smoke 及同身份的副本/链接；100k future 基座在九月 21 日才构建。
4. 原项目构建历史、九月 22 日 inventory、恢复的 run manifests 和删除记录共同界定本次项目内的构建清单；未发现相反证据或另一个遗漏的兼容构建。该结论不要求证明任意未知外部路径绝不存在。
5. 仍额外纳入 corrected 全量前缀及 corrected smoke 作为宽松缓冲；四个已知基座覆盖并集为 1,199 用户，较前一阶段并集多排除 5 人。

旧 `evaluation()` 会静默截断请求人数，故不能仅凭“请求 300/300”排除 197 开发 / 100 测试的小型 future 基座；本次覆盖其全部 300 原始用户。checkpoint 的约 4.7 GiB 体积仅是背景，不作为身份推断证据。该条目严格标为 `conservatively_covered`，不标为 exact recovered。

另一个旧 Windows 本地 30 用户 smoke 按原 `load_data` 的 raw 用户排序与 `default_rng(42).choice(...,30,replace=False)` 重建。原文的 **188 条正反馈首次交互、142 条训练交互、28 名评估用户（18 开发+10测试）**全部精确吻合；保守纳入全部 30 用户，新增 2 个 raw ID。

## 5. 交付与机器契约

审计目录（Git 忽略）：

`server_snapshot/next_cross_20260924/history_audit/monthly_20260924/`

核心引用全部为相对路径，可将整个目录原样携带到服务器。原始大证据保存在 `proofs/*.gz`，其压缩文件 SHA 是可携带依赖的校验值；解压后仍可核对原来源记录中的 SHA。所有身份文件留在被忽略目录，不随 Git 提交上传。

| 文件 | SHA-256 |
| --- | --- |
| `historical_users_month_20260924.json` | `c11c428479f455f447a24cf2c581e32c2d47a9e7aa2b70f6aa4bbdad76aaf5b5` |
| `source_registry_month_20260924.json` | `2e7f764cb2ce048ed6a2f03938632cd9cbeaf17b3ed5194aaa190dfe3b825487` |
| `closure_receipt_month_20260924.json` | `7dedeedd636512fb16355dbb678d65e149e11c414350ee92894c88f8a4153ee0` |
| `fresh_capacity_month_20260924.json` | `e9f19f3b4cdd61d24bcdb6a2322548ec360de10213e90084af4b96370c284799` |

receipt 为 `status=complete`、`scope_complete=true`、`unresolved_sources=[]`；history、registry、receipt 的 scope 与 scope hash 相同。source_count、raw_union_count 和排序 raw-ID 并集 canonical hash均显式绑定。所有保守条目附带实际存在且有 SHA 的 supporting_sources。

本报告只签认本月历史身份排除及容量，**不替代正式 prepare、资源预检、开发训练、选型门控或最终测试**。正式实验若采用该清单，应使用全部 17,296 名 fresh 合格用户，不应再静默截断至一个请求人数。

## 6. 独立复核与服务器交付

独立复核者逐项核对了 296 来源的用户数、排序 raw-ID hash 及依赖 SHA，重新合并得到 796,912 用户；仅使用身份和边界重算当前基座，合格历史用户为 778,901 人，fresh 为 17,296 人。四基座 1,199 用户并集和本地 30 用户的 188/142/28 计数也独立复算一致。本地正式 `load_historical_users(..., require_closure=True)` 校验通过。

可携带归档为 `history_month_v3_20260924.tar`，**248,166,400 字节**，SHA-256 为 `fd7f01891d6d20a611b9b62e244a2ac45af4e80acbec04c83e9b808dbdb609a5`。大证据已在包内逐文件 gzip，外层 tar 不重复压缩。服务器源包保留于 `/root/next_cross_20260924/history_month_v3_20260924.tar`，只解压一份至 `/root/next_cross_20260924/history_month_v3/`，两者均位于 system 盘。

服务器逐项核对 **35 个 payload 文件及其 bundle manifest**的大小与 SHA 后，以 `src_v3` runner 执行严格 history loader，**31.16 秒通过**。服务器 runner SHA 为 `ec8df470eab5bc3bb39978371c9d086b9a1e834b2be63dc72648616ffa9d1e22`。该验证只检查证据和身份 schema，未执行 prepare、候选生成、评估或训练。

正式 prepare 可使用：

`/root/next_cross_20260924/history_month_v3/historical_users_month_20260924.json`

机器交付回执：服务器 `/root/next_cross_20260924/history_month_v3_server_verification.json`；本地 `server_snapshot/next_cross_20260924/history_audit/history_month_v3_server_verification.json`。本节为上传完成后的追加记录，已封存包内报告保留上传前版本，不覆写或改变原归档校验值。
