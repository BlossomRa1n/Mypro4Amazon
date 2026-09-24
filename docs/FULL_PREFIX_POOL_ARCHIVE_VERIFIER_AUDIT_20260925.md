# Full训练候选池归档验证器审计

状态：**代码审计PASS；正式full/pilot本地归档仍须实际执行验收；不是test授权**。本次审计只读本地源码、测试与SHA并编写本文，没有连接服务器、构造候选、训练、重新评分或读取未来目标。

| 对象 | SHA-256 |
|---|---|
| `tools/verify_full_prefix_pool_archive.py` | `53c388ad768d4c124b94fb426a4374370c8634575e69ca32cc781b95eb62d75a` |
| `tests/test_verify_full_prefix_pool_archive.py` | `10c0784aa922044eac24ea3a52f6a13c0dbf9db9904c70090b28667905c5fb89` |
| 冻结builder，执行时传入的注册SHA | `b6534421c6c65d794205a5a90844cd5a11343685a870c04f545d86dc5a6aa0d3` |

## 1. 来源与读取边界

正式模式先固定diagnostic canonical `c82aa0…` 和old protocol文件SHA `4afac7…`，校验两份manifest自封印，再检查完整legacy Python文件集合与已封印code hashes；仅在通过后导入legacy。builder只校验字节SHA，不导入、不执行；调用方必须将 `--builder-sha256` 设为上表已审冻结值，full和pilot完成回执须同时绑定该值。

old protocol记录的base assets全部先做SHA检查，`data.pkl`再额外绑定 `base_data_hash` 后才通过冻结 `load_data` 读取。验证器不加载V2模型，不调用候选生成器；`targets` 与 `evaluation` 被设为拒绝。它只使用UID、合法训练边界及训练区间的时间/item信息重建合法位置，并核对训练item计数。读取base pickle本身以此前已认证、字节完全一致的本地资产为前提。

## 2. 完整训练池与pilot校验

正式full必须覆盖全部4,731,777个合法训练前缀，位置为严格递增int64，实际内容与重建合法位置逐项相等；pilot必须为注册的均匀抽样位置，正式至少2,048行。两个cache的source完整字典必须分别匹配冻结base/code、records顺序hash、75容量、四路权重 `[2,0,0.7,0.05]`、CF300/180天及padding约定。

冻结 `load_cache` 检查每个array文件SHA、int32 dtype、shape/lengths、有效ID范围、行内唯一性、零尾padding和全池logical hash。验证器再绑定cache sidecar SHA、positions SHA、pool/records hash以及完成回执。full/pilot目录的文件集合必须精确等于所需完成、开始、磁盘回执、building锁、positions、sidecar与payload，拒绝多余/缺失文件、symlink、特殊文件及FAILED marker。

`CONSTRUCTION_STARTED.json` 中除status外的所有字段须原样匹配完成回执；磁盘回执必须等于完成回执内嵌记录，且payload按每行 `75×4+4+8` 字节、至少1MiB metadata和1GiB reserve独立重算。记录的串行/并行检查须报告完整128行（tiny为可用行数）顺序一致，但验证器不重跑候选生成。

所有pilot行与full对应位置逐项比较；旧训练池位置、sidecar及payload另按原封印验证，再独立比较均匀128行与full对应行。它检验完整归档内部一致性与这两组独立重叠，不宣称独立重新生成了4.7M候选。

## 3. 证据保全与测试

所有入口拒绝symlink及父目录穿越，回执不可覆盖已有路径，也不可放进full、pilot、old protocol、base或legacy只读输入树。工具没有SSH、删除、重写原来源路径或关机行为。正式源映射/传输真实性以及资源与授权期限仍由主执行者独立验收。

实现者报告已用 `/Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python -m unittest tests.test_verify_full_prefix_pool_archive -v` 运行1项测试，24.269秒，OK；已确认上表code/test哈希后未再修改。该测试复用真实tiny完整CLI链，构建full及pilot并完成本地验收，同时覆盖positions payload追加字节、磁盘记录篡改与未显式允许smoke的拒绝。审计者阅读了测试及其调用路径并独立核对SHA，没有重复运行包含训练的fixture。

最终生产门禁必须使用冻结builder SHA和实际完整本地full/pilot归档得到正式回执。只有实际语义回执、原始来源/传输证据及可恢复性共同通过，才能作为后续final准备条件；本代码审计PASS不替代这些条件。
