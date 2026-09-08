# 精排模型 V2 方案（方案 B）

> Transformer 替换 DIN，多视角 Query + 特征 Token 化 + TokenMixer 融合
> 待讨论，未定稿

---

## 一、整体架构

```
                        ┌──────────────┐
                        │  User Features │  (256d)
                        │  Item Features │  (592d)
                        │  Dense Stats   │  (10d)
                        │  Seq Features  │  [B,50,323] (K) + [B,50,256] (V)
                        └──────┬───────┘
                               │
        ┌──────────────────────┼──────────────────────────────┐
        │                      │                              │
   Branch 1              Branch 2+3+5        Branch 4            (全部 token 进)
   Sequence              User/Item/Dense      FM Cross            TokenMixer
   Transformer           MLP 编码             (raw embedding)
        │                      │                  │                   │
   seq_out [B,256]     user_out [B,256]    cross_out [B,256]         │
                       item_out [B,256]     = user_emb ⊙ item_emb    │
                       dense_out [B,256]                             │
        │                      │                              │
        └──────────────────────┼──────────────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │   RankMixer       │  (MLP-Mixer: Token Mixing + Channel Mixing)
                     │   [B, 5, 256]     │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │   FFN             │
                     │   → logit → 点击概率│
                     └───────────────────┘
```

---

## 二、Branch 1：序列建模（Perceiver Resampler 替代 DIN）

### 2.1 当前 DIN 的做法

```
历史 50 item ──→ 和目标 item 算标量注意力权重 ──→ 加权求和 ──→ 1 个 256d 向量
                ↑
          仅看"历史 item ↔ 目标 item"单点关系
          历史 item 之间无交互、品牌/时间信号只参与注意力分数计算
```

### 2.2 V2 Perceiver Resampler 方案

核心思路：借鉴 Flamingo / Perceiver IO 的 Resampler 结构，将序列建模拆为**压缩**和**查询**两个独立阶段。

```
═══════════════════════════════════════════════════════════════
阶段一：压缩 (per-user, 只跑一次, 所有候选共享)
═══════════════════════════════════════════════════════════════

Step 1: 序列表示（无需 1155→256 投影）

  原 DIN 的 1155d 由 9 个部分拼接而成:
    ┌─ 纯历史信号 (323d) ────────────────────┐
    │ hist_item_emb       256d                │
    │ hist_brand_emb       64d                │
    │ hist_rating           1d                │
    │ hist_delta            1d                │
    │ hist_verified         1d                │
    ├─ 候选依赖信号 (832d) ────────────────────┤
    │ target_item_emb     256d  ← 和 hist_item_emb 来自同一 embedding 空间  │
    │ diff (hist-target)   256d  ← hist-target, 无新信息                   │
    │ prod (hist*target)   256d  ← hist*target, 无新信息                   │
    │ target_brand_emb     64d  ← 目标品牌                                 │
    └─────────────────────────────────────────┘

  删除 832d 候选依赖信号 — 理由:
    · diff/prod 是 DIN 为浅层 MLP 做的特征工程, Transformer 的 Q·K^T 自己会建模交互
    · target_item_emb/brand_emb 的信息已在阶段二的 Q1 [item+cat+brand]_emb 中
    · 删除后序列不再依赖候选 → 阶段一可缓存所有候选共享

  保留 323d 纯历史信号 — 使用方式:
    K = hist_item_emb ⊕ hist_brand_emb ⊕ [rating, delta, verified] = 323d
    V = hist_item_emb                                              = 256d
    辅助信号(brand/rating/delta/verified)只参与注意力计算, 不输出

  → Seq_K [B, 50, 323], Seq_V [B, 50, 256]   ← 纯历史, 候选无关, 可缓存

Step 2: Latent Slots (可学习记忆槽, L=16)
  16 个随机初始化的查询向量，每个带独立的可学习 slot embedding
  Latents [B, 16, 256]  ← 压缩目标：把 50 条历史压进 16 个槽

Step 3: Cross-Attention (Latent → Seq)
  Q = Latents [B, 16, 256]
  K = Seq_K [B, 50, 323]        ← 含 brand + rating + delta + verified
  V = Seq_V [B, 50, 256]        ← 仅 item_emb, 纯语义
  → [B, 16, 256]  ← 16 个槽各自从历史中提取不同信息

Step 4: Self-Attention (Latent 内交互)
  16 个槽互相看 ← 去重、互补、形成结构化记忆
  → [B, 16, 256]

Step 5: + FFN (per slot, 可选, 共享权重)
  → compressed_memory [B, 16, 256]     ← ⭐ 缓存！候选共享

═══════════════════════════════════════════════════════════════
阶段二：查询 (per-candidate)
═══════════════════════════════════════════════════════════════

Step 6: 构建 Query Tokens（2 个）
  Q1 (候选相似度): [item_emb, cat_emb, brand_emb] → Linear(576→256) → [B, 256]
  Q2 (偏好匹配):   user_emb ⊙ item_emb                                → [B, 256]

Step 7: Cross-Attention (Query → Memory)
  Q = [Q1, Q2] [B, 2, 256]
  K,V = compressed_memory [B, 16, 256]
  → [B, 2, 256]

Step 8: Sum Pooling
  [B, 2, 256] → sum over query dim → [B, 256]

Output: seq_out [B, 256]
```

### 2.3 和旧 Transformer 方案的结构对比

| | 旧方案 (Self-Attn + Cross-Attn) | 新方案 (Perceiver Resampler) |
|---|---|---|
| 历史 item 交互 | Self-Attn 53² = 2809 对 | Latent Self-Attn 16² = 256 对 |
| 候选依赖部分 | Q 混入 Self-Attn → 全部 53² 污染 | Q 独立在阶段二 → 与压缩完全解耦 |
| 每候选成本 | ~470 对 (Q↔Seq + Cross-Attn) | **~32 对 (Q↔Memory)** |
| 500 候选 | 2500 + 500×470 = **237.5K 对** | 1056 + 500×32 = **17K 对** |
| Q3 Learnable Probe | 1 个独立探针 | **融入 16 个 Latent Slots** — 探针军团 |
| 信息瓶颈 | 无（直接看 50 item） | 有（通过 16 槽查看历史） |

### 2.4 为什么 Query 从 3 个减到 2 个

Q3 (隐式探针) 的核心价值——"让模型自己学会该问什么"——现在由 16 个 Latent Slots 在压缩阶段完成。每个槽是一个可学习的"问题模板"，16 个槽比 1 个探针的表达力强得多：

```
Q3 (旧): 1 个可学习向量 → "还有什么我没注意到的?"
          → 单点查询，信息有限

Latents (新): 16 个带位置编码的可学习槽 → 特化成 16 种不同的"问题类型"
          → slot_0: "最近购买了什么?"
          → slot_3: "品牌忠诚度如何?"
          → slot_7: "复购周期是多长?"
          → ... 训练中自发分化
```

Q1 和 Q2 保留——它们是候选依赖的，必须在阶段二运行。只需要 2 个 query token 去查压缩后的记忆。

### 2.5 Latent Slots 的差异化设计

```
Latents [16, 256]: 随机初始化 N(0, 0.02)
Slot Embedding [16, 256]: 可学习，每个槽独有的位置编码
  → Latents + Slot Embedding → 16 个互不相同的初始"视角"

+ 差异化正则 (可选):
  L_div = -Σ_i≠j cosine_sim(slot_i, slot_j) → 惩罚槽之间的相似性
  系数 λ=0.001, 防 16 个槽退化成同一模式
```

### 2.6 推理性能：Perceiver 如何解决候选依赖问题

```
阶段一 (per user, 一次, 所有候选共享):
  K/V 分路:          K=[B,50,323], V=[B,50,256]           | 无需降维投影
  Cross-Attn(L→S):  Q=[B,16,256] attn K=[B,50,323], V=[B,50,256] | 16×50=800 对
  Self-Attn(L↔L):  16²=256 对                                      | 16×16
  缓存: compressed_memory [B,16,256]                              | 总: ~1056 对

阶段二 (per candidate):
  Q1 构建:          Linear(576→256)              | 一次小 Linear
  Q2 构建:          user_emb ⊙ item_emb          | 逐元素乘
  Cross-Attn(Q→M): [B,2,256] attn [B,16,256]   | 2×16=32 对
                                                 | 总: ~32 对

每候选成本: 32 对 ← 比旧 Transformer 方案(~470 对) 快 ~15×
              ← 比朴素 Self-Attn(2809 对) 快 ~90×
```

| 对比 | 朴素 Self-Attn | 旧方案(缓存后) | **Perceiver(新)** |
|---|---|---|---|
| 每候选注意力对 | 2809 | ~470 | **~32** |
| 500 候选总对 | 1.4M | 237.5K | **17K** |
| 500 候选估计延迟 | ~140ms | ~25ms | **~2ms** |

与 DIN 一样，Branch 1 的本质就是"以候选为中心去读历史"——候选依赖性是目标注意力模型的固有属性，不是 V2 引入的设计缺陷。改进点在于：DIN 只能做这个，而 V2 在增加了历史 item 间交互的前提下仅增加了 ~9× 常数倍。

---

## 三、Branch 2/3/4/5：特征编码与交互

### 3.1 User MLP（简化）

与 Item 不同，User 只有一个 embedding 来源（`user_emb`），不需要分组编码。6 个 dense 统计标量已独立为 `dense_out` 进入 TokenMixer——这里不再重复。

```
Raw: user_emb [B, 256]

→ Linear(256→512) → PReLU → Dropout
→ Linear(512→256) → LayerNorm
→ user_out [B, 256]
```

| 对比 | Item | User |
|---|---|---|
| Embedding 来源 | 4 个（item + cat + brand + verified） | 1 个（user_id） |
| 需要分组编码 | ✅ 本体 vs 上下文有天然结构 | ❌ 单一来源，无结构可分 |
| Dense 标量 | 已移除 → 仅由 dense_out 承载 | 已移除 → 仅由 dense_out 承载 |

> 两层 MLP 256→512→256 的作用从"特征编码"变为"特征精炼"：给 user_emb 增加非线性，使其更好地适配 TokenMixer 的融合空间。

### 3.2 Item MLP（分组编码 + 二阶交叉）

**设计动机**：Item 有 4 个不同来源的 embedding——item 本体(256)、品类(256)、品牌(64)、口碑(16)。全 concat 进一个 Linear，品类和品牌的多重语义被稀释；item_emb 和 cat_emb 之间无显式交互，依赖 MLP 隐式学习组合效应。

改进为**本体/上下文分组 → 独立编码 → 显式交叉**：

```
Item Features: item_emb(256) + cat_emb(256) + brand_emb(64) + verified_emb(16)

  Group A: item_core — "这个商品是什么"（本体）
    item_emb(256) → Linear(256→256) → LayerNorm → item_core [B, 256]

  Group B: item_ctx — "商品处于什么品类/品牌/口碑上下文"（环境）
    cat_emb(256) + brand_emb(64) + verified_emb(16) = 336d
    → Linear(336→256) → LayerNorm → item_ctx [B, 256]

  Group C: item_cross — "本体和上下文的关系"
    item_core ⊙ item_ctx → [B, 256]
    回答：同一品类/品牌下，这个单品是典型代表还是异类？

  Fusion (可学习权重加和):
    item_out = w1 * item_core + w2 * item_ctx + w3 * item_cross  → [B, 256]
    初始: w = [1.0, 1.0, 0.5]（交叉项初始低权重，防噪声主导早期训练）
```

| 改动点 | 原方案 | 改进后 | 收益 |
|---|---|---|---|
| verified_emb | 4d，占输入 0.7% | 16d，与 cat/brand 同组 | 复购口碑信号不再被淹没 |
| item-cat 交互 | 无，靠 MLP 隐式学 | `item_core ⊙ item_ctx` 显式 | 梯度路径更短，收敛更快 |
| 4 个 item dense 标量 | 混入 584d 输入 | **移除**，仅由 dense_out 承载 | 单一数据源，消除冗余 |

### 3.3 FM 二阶交叉（DeepFM 风格）

```
cross_out = user_emb ⊙ item_emb   (Hadamard, FM 隐向量逐元素乘积)
→ [B, 256]
```

**为什么用原始 embedding 而非 user_out ⊙ item_out**：

| 选项 | 问题 |
|---|---|
| `user_out ⊙ item_out`（NFM 风格） | user_out/item_out 各有一个独立 token 进 TokenMixer，Token Mixing 自然会学它们的交互——cross_out 提供的信息与 Token Mixing 重复 |
| `user_emb ⊙ item_emb`（纯 FM） | 从 embedding 层直取协同过滤信号，与 Branch 2/3 的 Deep 分支走不同路径，最终在 TokenMixer 中 **FM 信号 + Deep 信号互补融合** |

这与 DeepFM 的设计完全一致：FM 分支和 Deep 分支**共享同一套 embedding 输入**，走不同的处理路径，在最后融合。Q2 也使用了 `user_emb ⊙ item_emb`，但用途不同——Q2 用它去查询历史序列，cross_out 直接进入 TokenMixer 提供全局交互信号。

### 3.4 Dense 统计特征 MLP ⭐ 新增

**设计动机**：dense 标量在 Item MLP 584d 输入中占比 0.7%，在 User MLP 262d 输入中占比 2.3%——极易被 embedding 维度淹没。改进后 User/Item MLP 输入中已**完全移除**这些标量，10 个统计特征**仅由** `dense_out` 这单一通道进入 TokenMixer，杜绝信号冗余和数据来源歧义。

```
Raw [B, 10]
  用户侧 (6): click_count_norm, time_span_norm, user_avg_rating_norm,
             user_std_rating_norm, user_verified_ratio, user_avg_helpful_norm
  商品侧 (4): item_click_count_norm, created_at_ts_norm,
             item_avg_rating_norm, item_rating_number_norm

→ Linear(10→64) → PReLU → Dropout
→ Linear(64→256) → LayerNorm
→ dense_out [B, 256]
```

> 注意：User MLP 和 Item MLP 输入中已**完全移除**各自的 dense 标量。10 个统计特征**仅由** `dense_out` 这单一通道进入 TokenMixer——杜绝同一信号从两个路径进入融合层造成的来源歧义。

### 3.5 品类偏好 token？（不加，已讨论）

- Branch 1 的 Q1 已提供候选-历史的**品类匹配**
- Branch 3 的 item_out 已提供候选的**品类身份**
- Branch 2 的 user_emb 在训练中隐式编码**品类偏好**
- 独立品类偏好 token 提供的增量信息（Top-N 偏好分布、兴趣熵）在单品类场景下区分度为零；扩多品类后可考虑**扩展 User MLP 输入**而非独立 token

---

## 四、Fusion：RankMixer + FFN

### 4.1 为什么选择 RankMixer

参考字节跳动 2024 年 RankMixer 论文，在抖音/头条精排线上替代复杂特征交叉结构。方案对比：

| | 硬切分 13×64d | 多头 Self-Attention | **RankMixer (选用)** |
|---|---|---|---|
| 子空间分解 | 硬边界切割 | 可学习 QKV 投影 | token 本身即语义单元，无需分解 |
| Token 交互 | 13 碎片 Self-Attn | 5 token Q·K^T | Token Mixing MLP(5→20→5) |
| 对异质 token | 差（碎片无语义） | 差（Q·K 假设同一空间） | **好**（学习独立混合权重） |
| 参数 | 中 | ~530K | **~525K** |
| 工业验证 | 无 | 通用 | ✅ 字节精排线上 |
| 复杂度 | O(T²) | O(T²) | **O(T) 线性** |

**选 RankMixer 的核心原因**：我们的 5 个 token 来自 5 个不同分支，异质性强——`seq_out` 和 `dense_out` 做点积没有直观语义。MLP-Mixer 的 Token Mixing 给每对 token 学独立的混合权重，不假设 token 处于同一投影空间。

### 4.2 RankMixer 结构

```
RankMixer Block × N (建议 N=1~2):

  Input: [B, 5, 256]

  ┌─ Token Mixing ─────────────────────────┐
  │  [B, 5, 256] → permute → [B, 256, 5]   │
  │  → MLP(5→20→5) → permute → [B, 5, 256]│
  │  + Residual + LayerNorm                 │
  │  作用: 5 个 token 之间交换信息, 学到"seq 该关注 user 的多少"等混合权重 │
  └─────────────────────────────────────────┘

  ┌─ Channel Mixing ────────────────────────┐
  │  [B, 5, 256]                            │
  │  → MLP(256→1024→256)                    │
  │  + Residual + LayerNorm                 │
  │  作用: 每个 token 内部做非线性精炼 (共享权重)│
  └─────────────────────────────────────────┘

  Output: [B, 5, 256]
```

**参数拆解**（1 层）：

| 模块 | 计算 |
|---|---|
| Token Mixing | 5×20 + 20×5 = **200** |
| Channel Mixing | 256×1024 + 1024×256 ≈ **524K** |
| 2× LayerNorm | 2×256 = **512** |
| **合计** | **≈ 525K**（对比 Self-Attn ≈ 530K，几乎一致） |

### 4.3 Sparse MoE（限定在 RankMixer 之后、FFN 之前）

**为什么放在这里，而不是 Branch 1 内部**：

| 位置 | 门控输入 | 问题 |
|---|---|---|
| Branch 1 sum pooling 后 | 仅序列 256d | 不知道 user 是谁、item 是什么——盲目路由 |
| **RankMixer 之后 (选用)** | 5 token 拼成的 **1280d** | 看到全部信息，能做有意义的路由决策 |

1280d 里包含了所有 token 的完整表示，门控网络有足够信息判断："这个 user-item 对属于哪种类型、该由哪个专家打分"。

```
RankMixer Output [B, 5, 256]
  → Flatten → [B, 1280]           ← 5 token 完整信息

  ┌─ Sparse MoE ───────────────────────────────────┐
  │  Gate: Linear(1280 → N) → Softmax → Top-K (K=2) │
  │  Expert_i: Linear(1280 → 512), i = 1..N         │
  │  Output: Σ gate_i · Expert_i(x)                 │
  │  + Load Balancing Loss (辅助, λ=0.01)            │
  └────────────────────────────────────────────────┘
       ↓
  PReLU → Dropout(0.1) → Linear(512 → 1) → logit [B]
  → sigmoid → 点击概率
```

**参数（N=4, K=2）**：

| 模块 | 参数量 |
|---|---|
| Gate: Linear(1280→4) | 1280×4 + 4 = **5.1K** |
| 4 × Expert: Linear(1280→512) | 4 × (1280×512+512) ≈ **2.6M** |
| 总计 | **≈ 2.6M**（MoE 部分，不含 RankMixer ~525K） |

**N 和 K 的建议**：
- N=4 专家，K=2（Top-2 路由）：每个样本由 2 个专家联合打分，负载均衡且保持多样性
- 论文对冲：专家可能自发特化——专家 0 关注"老用户+品质商品"，专家 1 关注"新用户+热门品"等
- Load Balancing Loss 防止某专家退化：`λ · Σ(N·f_i - 1)²`，其中 f_i 是专家 i 的使用频率

### 4.4 和原版"切分"方案的精神对照

| 原版思路 | RankMixer 等价实现 |
|---|---|
| "把粗粒度表示拆成细粒度子空间" | Channel Mixing 的 MLP(256→1024→256) — 瓶颈扩展等效于子空间拆分 |
| "token 之间交换信息" | Token Mixing 的 MLP(5→20→5) — 学到的权重等效于注意力 |
| "序列 token 特殊处理" | 不再特殊——5 token 在 Mixer 中权重均等，序列信号不再被压缩 |

**核心改进**：原版把 3 个 token 搅碎后硬切成 13 片，RankMixer 保留 5 个语义完整的 token，通过可学习的 MLP 做交叉。语义边界由模型决定，不由 `64` 这个数字决定。

---

## 五、维度汇总表

（和你朋友给的原版对比）

| 模块 | 原版（朋友） | 修正版（本方案） | 修正原因 |
|---|---|---|---|
| User 原始维度 | `[B, 32] → 832` | **`[B, 256]`**（纯 user_emb） | dense 标量已移入独立 `dense_out` token |
| User MLP | — | `256→512→256`，特征精炼 | User 单一 embedding 源，无需分组 |
| Item 原始维度 | `[B, 1155] → MLP → 256` | **`[B, 592]`**（纯 embedding，分两组） | 1155 是序列维度；dense 标量已移入 `dense_out` |
| Item MLP | — | 分组编码：本体 + 上下文 + 显式交叉 | 利用 item/cat/brand/verified 多源结构 |
| verified_emb | `4d` | **`16d`** | 4d 被淹没；16d 匹配其信息量上限 |
| Seq 每位置维度 | `[B, 50, 1155]` | **`K=[B,50,323]` + `V=[B,50,256]`**（纯历史，移除 target 侧 832d） | diff/prod/target 侧信号由 Q1/Q2 动态激活 |
| 序列架构 | — | **Perceiver Resampler**：16 槽压缩 → 2 Query 查询 | 两阶段解耦，推理加速 ~90× |
| Latent Slots | — | L=16，带可学习 Slot Embedding | 替代旧 Q3，自主分化成多视角探针 |
| Query 数量 | 3 | **2**（Q1 候选相似度 + Q2 偏好匹配） | Q3 职责并入 Latent Slots |
| Q1 | user_emb | `[item, cat, brand]_emb` → 256 | 以候选为中心，带品类品牌 |
| Q2 | item_emb | `user x item` | 交互信息更丰富 |
| 融合 Token 数 | — | **5**（seq/user/item/cross/dense） | dense 统计特征独立编码，防信号淹没 |
| 融合方式 | TokenMixer + FFN | RankMixer + Sparse MoE + FFN | — |
| Perceiver 阶段一 | — | Cross-Attn (16 latents × 50 seq) + Latent Self-Attn | 压缩历史，候选无关，每用户一次 |
| 每候选成本 | O(50) | O(32) + O(308) 缓存摊销 | Perceiver 后候选推理极轻 |
| 推理方式 | 单候选逐条 | 用户级 batch 打包 (复用 `inference_full.py` 策略) | 500 候选一次 forward，~10ms |

---

## 六、训练策略

### 6.1 Loss 设计

```
主 Loss: L_bpr = -log(sigmoid(pos_score - neg_score)).mean()

辅助 Loss (必须):
  MoE Load Balancing:
    L_balance = N * Σ(f_i - 1/N)²    (N=4 专家, f_i=fraction of tokens routed to expert i)
    λ_balance = 0.01
    防马太效应 — 早期如果某专家"运气好"被多选 → 梯度更多 → 越来越好 → 其他专家退化

辅助 Loss (可选, 先不加):
  Slot Diversity:
    L_div = -Σ_{i≠j} cosine_sim(slot_i, slot_j) / (L*(L-1))
    λ_div = 0.001
    跑完第一版看 16 个 slot 的相似度分布, 自发分化(<0.3)就不加

总 Loss: L = L_bpr + 0.01 * L_balance (+ 0.001 * L_div if needed)
```

### 6.2 训练配置

| 项目 | 配置 | 理由 |
|---|---|---|
| 优化器 | **AdamW** (lr=1e-3, weight_decay=1e-4) | Transformer 类标准选择，decoupled decay |
| 调度器 | Linear Warmup 500 steps → CosineDecay | Warmup 防冷启动梯度爆炸 |
| Batch Size | 512~1024 | 取决于 5060Ti 8GB，AMP 混合精度 |
| Epochs | 15~20, Early Stop patience=5 | 监控验证集 Recall@20 |
| 精度 | AMP (GradScaler) | 与当前一致 |
| Gradient Clipping | max_norm=1.0 | MoE 门控容易梯度爆炸 |
| Neg Ratio | 4, ItemCF Hard Negative | 复用当前 DINExtendedDataset |

### 6.3 ALS 预训练 + 分阶段训练

```
Phase 0: ALS 预训练 (离线, 一次, 复用 als_init.py)
  → user_embedding, item_embedding 的初始值
  → category_embedding, brand_embedding, verified_embedding 随机初始化
  → Latent Slots / Slot Embedding 随机初始化 N(0, 0.02)

Phase 1: 冻结 Embedding (epoch 1~3)
  lr=1e-3, 只训练 Transformer / RankMixer / MoE / MLP
  目的: 上层结构在稳定 embedding 基础上先学会融合

Phase 2: 全参数微调 (epoch 4~15)
  lr=5e-4, 解冻全部参数
  CosineDecay to 1e-6
  Early Stop: Recall@20 连续 5 epoch 不提升 → 停止

Phase 3: Best Checkpoint
  监控: 验证集 Recall@20
  保存: din_v2_best.pth
```

### 6.4 位置编码

| 对象 | 方案 |
|---|---|
| 50 条历史序列 | 可学习 Position Embedding [50, 256]，加到 Seq_K |
| 16 个 Latent Slot | 可学习 Slot Embedding [16, 256]，加到 Latents |

50 长度下可学习编码和正弦编码效果一致，选择实现更简单的可学习方式。

---

## 七、所有决策总结

- [x] **Token 设计**：5 类（seq / user / item / cross / dense）
- [x] **品类 token**：不加（Branch 1/2/3 已覆盖）
- [x] **序列建模**：Perceiver Resampler，K=[50,323] + V=[50,256]，不投影
- [x] **Item 编码**：分组编码（本体 + 上下文 + 显式交叉）
- [x] **FM 交叉**：user_emb ⊙ item_emb（DeepFM 风格，原始 embedding）
- [x] **Dense 特征**：独立 MLP → dense_out，User/Item MLP 中已完全移除
- [x] **TokenMixer**：RankMixer (MLP-Mixer)，N=1~2 层
- [x] **Sparse MoE**：RankMixer 之后 Flatten [1280] 处，N=4/Top-2
- [x] **位置编码**：可学习 Position Embedding + Slot Embedding
- [x] **Loss**：BPR pairwise + ItemCF Hard Negative + MoE Load Balancing
- [x] **训练**：ALS 预训练 → 冻结 3 epoch → 全参数微调，AdamW + Warmup + CosineDecay
- [x] **推理**：用户级 batch 打包，阶段一缓存跨候选复用

---

## 八、和现有 DIN 的对比

| | DIN (当前) | Transformer V2 (本方案) |
|---|---|---|
| 序列建模 | 单向 target→history，标量注意力 | Perceiver Resampler：压缩→查询两阶段 |
| 历史交互 | 无（各历史 item 独立） | Latent Self-Attn：16 槽去重互补 |
| 隐式探针 | 无 | 16 个可学习 Latent Slots，自发分化 |
| 查询视角 | 1 个 target item | 2 Query（候选相似度 + 偏好匹配） |
| 候选依赖成本 | O(50) per candidate | **O(32) per candidate**（Perceiver 缓存后） |
| 品牌/品类 | 仅参与注意力分数计算 | Q1 显式编码 + Latent 槽可自主学习品牌模式 |
| 特征融合 | concat → MLP | RankMixer + Sparse MoE + FFN |
| 计算复杂度 | O(50) per candidate | 压缩 O(1056) + 候选 O(32) |
| 推理延迟 | ≈0.1ms per candidate | 压缩 ≈0.5ms + 候选 ≈0.02ms (Perceiver 缓存) |