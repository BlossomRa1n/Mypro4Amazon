# 夜间cross归档验证器终审

状态：**代码审计PASS；实际正式归档仍需运行验收；不构成test授权**。审计只读源码与测试，没有重跑训练、连接服务器或读取test目标。

| 对象 | SHA-256 |
|---|---|
| `tools/verify_early_stop_cross_archive.py` | `2a7394f01f5375c67841d0f76453ac4a44a4d0a3eb7f9e84e28d20ffa179ca9d` |
| `tests/test_verify_early_stop_cross_archive.py` | `75402f6d8f8f6854a5f1217d60dacb9f026c912216a63a9b91db1967c8a1c2fb` |

验证器拒绝失败marker、symlink、特殊文件、目录越界、缺失或多余未封印文件，逐文件核对SHA和需要的size/JSON有限性。正式模式锁定原diagnostic canonical、夜间文档SHA、原raw归档完成/协议/证据及old dev固定canonical和文件SHA；source bundle与manifest精确覆盖。它复核原raw/zero共同初始化、三个epoch1消费trace、原raw早期best点、zero三点screen选择、3个checkpoint/9个screen数组/3个confirm数组及所有pairing和指标。

主要统计由逐用户原始数组独立复算：先跨3seed按用户平均，再用seed20260925做10,000次bootstrap、97.5%分位区间；四项晋级条件来自复算值，不能信任自报selection。NDCG不重新打开未来目标分母，核验范围为有限边界、配对、汇总和差值区间；原cohort UID/顺序认证与远端传输校验保留为独立外部门槛。

终审要求的两个缺口已经关闭：

1. 回执同时输出相等的 `completion_sha256` 与 `cross_completion_sha256`，后者直接满足独立final executor的cross归档引用接口，无需手工改写已验收回执。
2. 每seed先通过raw evidence封印及raw lock核对对应raw best SHA和embedded manifest，在CPU提取所有tensor的shape/dtype后释放raw payload，再检查zero best的全部规格完全相同。原有zero tensor键集合、有限性和metadata校验仍保留；重封checkpoint错误shape/dtype不再能通过。

实现者报告最终真实tiny fixture测试1项通过，约6.24秒；审计者阅读了测试而未重复执行。测试实际生成raw、old zero及新cross，核验有效归档；完整重封错误shape和错误dtype后均在规格门槛拒绝，另覆盖bootstrap区间、selection、初始化、epoch合同、未封印文件、失败marker及配对身份篡改。实现者此前报告的15项组合测试不能替代本次正式数据验收，也不被写作审计者亲自执行。

复核时新cross runner仍为 `cacafb6b0916fd5f934debeb3f1dc73d1677dea5afe0b857fa807d89782f3282`，原三个diagnostic代码仍为冻结SHA。该工具不训练、不重新评分、不读test，只在全部本地检查通过后排他写验证回执。主执行者仍须完成实际正式归档、cohort/source/transfer、可读恢复和资源/期限门禁；代码PASS不是full/test开测凭据。
