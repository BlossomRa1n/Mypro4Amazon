# 传输与来源映射辅助门禁审计

状态：**四个工具代码审计PASS；cross加强版来源门禁已实际通过；fullpool来源门禁与final映射工具仍须正式输入齐备后实际运行。** 本审计仅检查本地来源和测试，不连接服务器、不训练、不评分，不打开test未来目标；没有修改正在执行或已完成的传输现场。

| 工具/测试 | SHA-256 |
|---|---|
| `tools/archive_verified_remote_tree.py` | `ef060ca3060243a199fc4bde944626d519c3b70b4930cb75ad0dcb1c8ebd9730` |
| `tools/verify_early_cross_sources.py` | `6171429b2bee7b28512edd45ecb730106946eb1681b78ac844e5654411923a74` |
| `tools/verify_final_source_mapping.py` | `c044c439b2b3a962b4141235cab0353e6ad48ff4b792365a6b489bd174227c11` |
| `tests/test_source_verifier_guards.py` | `c191c269d4e87d869d3d833ed707804166beb6e7fccb05add9d95ece827e60c1` |
| `tools/verify_fullpool_sources.py` | `8279ed9c60b359ec15ed2fc441c6fbd3f7ed1a99281bbe9bde89870e73d9c075` |
| `tests/test_verify_fullpool_sources.py` | `df8008706b015ef77f4580f23f935510ba40a1caa844dfedce3de345a3abd99b` |

## 1. 字节传输门禁

`archive_verified_remote_tree.py` 在远端只读枚举已完成树，拒绝symlink和特殊文件，为全部常规文件记录SHA与size，再用不带delete的rsync下载至不存在的新目录。本地文件集合必须与初次inventory精确一致，并逐文件匹配SHA/size；之后重新生成远端inventory，要求与第一次完全相等。成功回执绑定inventory SHA、completion SHA、两个根目录、文件数及总字节数；cross、training、test完成文件分别有对应SHA别名。

本地输出/回执路径拒绝symlink、父目录穿越及已存在路径，回执不能放在传输树中；失败保留partial证据并另记failed文件。工具不删除、不关机，也不声称验证了模型或指标语义；回执的 `local_payload_semantics_verified=false` 是正确边界。远端完成marker的成功status只确定可归档前提，仍需相应语义验证器检查模型/数组和运行合同。

## 2. Cross来源加强版

初稿暴露的三个缺口现已关闭：输出回执可进入已封只读树；源manifest自报runner/monitor SHA而未固定已审来源；控制传输回执缺少与实际完成marker、路径及inventory的绑定。

新版本固定6项source集合及逐项注册SHA，包含cross runner、协议、monitor、raw归档回执、恢复备份绑定和测试源码，分别验证本地src与远端归档src。冻结61个legacy依赖逐文件匹配固定diagnostic canonical下的code hashes；原old protocol文件固定SHA，从其cohort记录与受封candidate sidecar/payload恢复原顺序身份。

两个transfer回执均核验inventory实际SHA、remote/local roots、completion marker、completion实际SHA、filecount/bytes、完整文件集合/size以及三个成功布尔字段。source/control树还重新逐文件hash；run的manifest和12数组另外绑定原completion seal，模型完整字节和语义由独立archive验证器验收。12数组须精确为每seed三screen一confirm；在打开前检查SHA，再逐元素比较原UID/position、候选ID及lengths。

控制树的 `launch.json` 被其inventory保护，实际command的runner、参数集合、run/control路径、禁止自动重启/关机等均绑定固定角色。控制COMPLETED须exit0、真实完成marker、无失败，且最终status与COMPLETED内容相等。回执输出禁止进入cross run、remote_root、src、raw树和原protocol树，并在写入前再次检查安全位置。

实际 `source_cohort_verified_v2.json` 已存在且报告6 sources、61 dependencies、12 arrays和monitor exit0。实现者报告使用上述冻结新源码实际执行成功；审计者读取了实际回执及所有修复源码，并独立运行下面的对抗测试。旧回执未覆盖，新旧回执SHA同为 `b53e48b40ddb5d6c6a26aa8158eeec5be3977d70e3babc995f63ec4d7651ae8d`，因为输出字段为保持兼容而未变化。该相同SHA不单独证明使用了新验证器；新验证器SHA、执行记录及本文审计共同记录版本来源。

版本执行绑定现已补齐：`server_snapshot/early_stop_cross_20260925/source_cohort_verified_v2_execution.json`，SHA `183e29ae7cdf18927763c4f20373999663fc36457c8996b57122f13b0f22a077`。它记录UTC2026-09-24 19:16:45.340557至19:16:47.286644的一次实际独立只读重验、完整command、上述新verifier SHA、exit0、stdout及空stderr，产出独立 `source_cohort_verified_v2_recheck.json`，其SHA仍为 `b53e48…`，并记录原/旧v2回执字节保持不变。时间明确属于这次重验，没有倒填此前执行时间。审计者已读回该执行记录并核对文件SHA。

## 3. Final原始来源映射

工具要求formal authorized且smoke=false的授权，mapping键集合精确等于授权refs，每份本地副本实际字节必须等于原remote ref SHA。它固定diagnostic canonical、old protocol文件SHA、三个test身份文件SHA及已审executor SHA；检查完整legacy文件集合、base assets和身份receipt列出的证据字节。只做test身份文件字节校验，没有future target调用或candidate-hit统计。

所有输入路径安全检查、SHA和size记录后，输出仍保留授权的原remote path文本和对应本地路径，不改原authorization或plan。authorization/mapping在结束前重新核对SHA，防止映射期间改动。输出回执不能置于legacy/base/evidence树，也不能置于old protocol、executor、authorization、mapping及任一mapped文件父目录树内；正式调用须使用独立的外部gates目录。

来源真实性仍依赖主执行者已审授权及其原remote→local映射；本工具不会凭“status=authorized”认证人类授权本身，也不替代资源预算、真正训练/test归档或全量pool语义门禁。正式auth/map未落定前，只能说代码通过，不能写成final来源已经通过。

## 4. 测试与结论边界

审计者独立运行 `/Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python -m unittest tests.test_source_verifier_guards -v`，7项测试在0.045秒通过。测试不训练/评分，覆盖已有/只读输出、symlink和穿越、输入symlink、source缺失/多余/重封SHA、transfer roots/marker/hash/totals/flags、控制payload篡改、launch每个参数变更、配对身份不变但数组payload改变，以及总数正确但seed分布错误等拒绝。

本次审计不重跑远端transfer，也不修改任何冻结实验协议、模型选择或原回执。正式cross的语义archive、两transfer和加强版来源均有独立实际证据；正式final仍必须在其auth/map、full/pilot及其余前置条件齐备后逐门运行并保留结果。

## 5. Fullpool来源与pilot传输追加审计

`verify_fullpool_sources.py` 代码审计PASS。它固定builder、builder test、语义验证器、diagnostic/old protocol及先前独立捕获的pilot完成SHA；full/source-control两个transfer均按原remote/local role和completion marker绑定，并重新逐文件核对完整集合、size和SHA。原本地源码、预先归档src和远端归档src三方均匹配冻结builder/test。

执行来源进一步固定 `supervise_full.py` SHA `e8c67a0118c49e0f2fd01563457c2a535b22905a591cee71f9ac60c822303c0a`，本地已审副本与远端归档副本必须一致。该监督器只做联合磁盘/期限预核、启动固定4worker CPU builder、900秒状态采样，并要求exit0且存在COMPLETED；没有自动重试、test读取或关机。来源门禁要求launch命令及策略完全匹配原捕获记录；最终status须精确含9个规定字段，COMPLETED只可额外含status和peak RSS，投影后逐字段完全一致，不能用空status或删字段蒙混通过。

正式语义回执须formal=true、smoke=false并绑定已审verifier；其full completion SHA及全部文件SHA/size map必须等于实际full归档与transfer inventory。full和pilot的完成角色、原输出目录、4worker、完整行数/2,048pilot行数、冻结代码/来源map及隔离布尔值均核验。它不通过导入builder或生成候选来复查。

pilot由现有tar与解包目录双重认证：实际COMPLETED须等于此前捕获的固定SHA `1bd80ad658977dcedde5b5d459756df9be5b653acb13f392debf7ad7c9a1fbc3` 和语义回执；解包回执绑定tar SHA；逐tar member仅允许固定单层根目录和常规文件，拒绝穿越、链接、重复、额外条目。每份payload的tar流、实际解包文件及语义map SHA/size必须相等，覆盖数量精确一致。

输出路径安全检查禁止修改full、remote、src、pilot解包树及工具/测试目录；只有所有准备检查通过后才新建回执目录并写exclusive输出。正式调用为 `--repo <绝对repo路径> --receipt <外置回执路径>`，archive根层新回执可用；不能把代码PASS或已验pilot当成full实际PASS。

审计者独立运行 `/Users/admin/.cache/mypro4amazon-diagnostics-venv/bin/python -m unittest tests.test_verify_fullpool_sources -v`，最终7项测试0.047秒通过。包含实际已封pilot的tar/解包/先前完成核对；transfer角色、marker、seal、flags和等大小payload篡改；launch参数/worker/policy；空或缺字段status、值漂移；本地与远端同改而重封的监督器；unsafe tar及不安全回执；准备未齐不得留下输出目录。全部测试无模型、候选生成或targets。正式full/source/control和语义回执齐备后仍必须运行此门禁，本文未先行宣称其实际通过。
