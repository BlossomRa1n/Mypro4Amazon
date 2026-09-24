# 全量合法前缀候选池审计

状态：**代码审计PASS；正式pilot/full产物仍须实际验收；不授权test**。本审计只读代码、测试与封印依赖，没有连接服务器、运行训练或构造test候选。

| 审计对象 | SHA-256 |
|---|---|
| `tools/build_full_prefix_pools.py` | `b6534421c6c65d794205a5a90844cd5a11343685a870c04f545d86dc5a6aa0d3` |
| `tests/test_full_prefix_pools.py` | `2791ab2d120d46c5c1e46f69b084ea9ec0cb909eb805f24e9ff5ad66e204c565` |

正式入口固定诊断manifest canonical hash为 `c82aa0baca95e24793ce2b0b26ea2c70c468fb389a61c8cc1c278b37bdc8c3e6`，旧protocol文件SHA为 `4afac7eb2089c273f026e2f39bdb1eafeb408dae500b29d54868972515198ee1`。同时重算manifest本身、旧protocol、全部base资产与独立legacy目录Python文件集合的SHA；构造最后一个block再次核验。自封重写manifest不能替代原注册来源。原三个diagnostic代码文件及原诊断/夜间协议SHA均与冻结值相同。

全量行严格等于base合法 `train_positions`：4,731,777条、升序、唯一、合法用户prefix边界及首事件时间约束；item counts从train mask复算。正式base data ID固定为 `754d426bc41ff19907bd31b3f139e3150d09cc13a746924fecc42e562a92b158`。构建时禁止 `targets`/`evaluation`，并为iid/ts/rating数组包裹合法train mask访问守卫。V2资产仍校验SHA，但权重为零，所以不加载模型或执行V2推理。操作不读取test未来目标、不计算候选命中覆盖、不创建test池。

候选算法直接调用冻结 `RecallPoolBuilder`，75容量、权重 `[2,0,0.7,0.05]`、CF300、180天半衰期、原category/hot和tie/order规则。worker范围1–4，CPU fork前禁止CUDA初始化并将Torch/BLAS线程限制为1。最多一个1024行block及四个subchunk在途；有序map返回后检查sequence、完整records envelope和候选行数。构造前比较均匀抽取至多128行的串行/并行完整ID顺序；正式模式另从原604,511行缓存均匀取128行，要求冻结算法重建值和旧缓存的ID/顺序逐项一致。

旧缓存 `source.code_hashes` 保留原protocol map，保证候选格式与旧来源兼容；实际执行的diagnostic冻结源码map、wrapper SHA和protocol文件SHA单独写进构造回执，二者不能混称。`PositionRecords`惰性读取原行轴，不物化数百万tuple。pilot与full有不同mode、selected_rows与文件名，pilot不能冒充full。

新增磁盘门禁在创建positions或cache的磁盘payload前检查输出目录所在文件系统：每行items 75×4字节、length4字节、position8字节，正式净payload合计 **1,476,314,424字节**，再加1MiB头部/JSON余量与1GiB安全余量。临时文件在同一文件系统rename发布，payload不重复分配。`DISK_PREFLIGHT.json`记录st_dev、实际free和需求；不足即失败。该检查不是空间预留，主执行者仍需将并行cross/checkpoint写入计入同一文件系统动态余量。

原 `build_cache` 使用有界memmap，完成后逐项复核shape/dtype/length/ID/padding、文件SHA与logical hash，最后发布sidecar；positions独占临时文件后rename，完成回执最后写入。输出已存在、失败中断或内容漂移时拒绝覆盖，保留现场。

实现者报告14项组合测试通过；审计者读取本文件对应测试代码但未重新运行训练/实验。覆盖真实非smoke原候选算法串并行相等、future守卫、worker异常不发布完成sidecar、重排/漏行/mask/asset变动拒绝、错序worker envelope、独立冻结smoke CLI、重封正式manifest拒绝和低磁盘空间拒绝。tiny CPU测试不能证明正式耗时；正式2048行pilot、128原缓存重叠、实际source/cache读回及资源/deadline门禁仍需主执行者验收。
