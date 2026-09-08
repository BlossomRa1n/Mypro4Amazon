import os

curr_path = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(curr_path)

# ============================================================
# Amazon Reviews 2023 — 数据集 & 路径配置
# ============================================================

DATA_PATH = '/root/autodl-tmp/amazon_data'  # 服务器 autodl-tmp 数据盘 (当前工作环境)
# DATA_PATH = os.path.join(ROOT_PATH, 'amazon_reviews')  # 本地默认路径

# --- 三档规模控制 ---
# 档位 1 (本地验证): offline=True → 只看 10000 条
# 档位 2 (5060Ti): 5 品类全量 (MI + OP + CDs_and_Vinyl + Video_Games + Toys_and_Games)
# 档位 3 (服务器): 追加更多品类到列表中
AMAZON_CATEGORIES = [
    'Musical_Instruments',   # 最小品类, 品牌丰富
    'Office_Products',       # 品牌丰富
    'CDs_and_Vinyl',         # 替换 All_Beauty (后者 5-core 仅 ~240 用户, 空壳)
    'Video_Games',           # 现有主力品类
    'Toys_and_Games',        # 品牌丰富, 与 Video_Games 用户重叠
]
# 注意: EXT_CATEGORIES 应与 AMAZON_CATEGORIES 保持一致, 区别仅在于数据格式 (raw JSONL vs rating CSV)

OFFLINE_MODE = False         # True: 每品类只读10000条; False: 全量
AMAZON_SAMPLE_USERS = None    # 本地验证: 1000; 5060Ti: None
AMAZON_MIN_USER_INTERACTIONS = 0   # Sampled 评估已对齐论文，不需要稠密过滤

USER_DATA_PATH = os.path.join(ROOT_PATH, 'user_data')
MODEL_PATH = os.path.join(USER_DATA_PATH, 'model_data')
TMP_PATH = os.path.join(USER_DATA_PATH, 'tmp_data')
RESULT_PATH = os.path.join(ROOT_PATH, 'prediction_result')

ITEMCF_SIM_PKL = os.path.join(MODEL_PATH, 'itemcf_i2i_sim.pkl')

ENCODER_PKL = os.path.join(TMP_PATH, 'id_encoders.pkl')

# --- 模型超参数 ---
EMBED_DIM = 256
HIDDEN_DIMS = [512, 384, 256, 128]    # 加深 MLP: 更好融合序列 + 统计特征
HIST_LEN = 50
BATCH_SIZE = 1024
NUM_WORKERS = 12              # DataLoader 线程数 (云服务器 16 vCPU: 12)
NUM_EPOCHS = 20               # 稠密双品类: 20 轮
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_NEGATIVES = 4
TEMPERATURE = 0.1

# --- Two-Tower V2 (SASRec + InfoNCE) 配置 ---
V2_MODEL_FILE = os.path.join(MODEL_PATH, 'two_tower_v2_model.pth')
V2_EMBED_PKL = os.path.join(MODEL_PATH, 'item_embeddings_v2.pkl')
SASREC_NUM_HEADS = 2
SASREC_NUM_BLOCKS = 2
SASREC_DROPOUT = 0.1
INFONCE_TEMPERATURE = 0.07
NUM_HARD_NEGATIVES = 3
# --- Phase 2 特征接线开关 (用户塔) ---
V2_USE_BRAND_PREF = True      # 品牌偏好向量: hist_brands → brand_embedding 平均池化 → 拼 MLP (Phase2 开)
V2_USE_TIME_DECAY = True      # 时间衰减: hist_time_deltas → 距今时间 exp 衰减 → 加权池化 hist_vec (Phase2 开)
V2_TIME_DECAY_LAMBDA = 0.3    # 时间衰减系数 (越大越激进地只关注近期交互)
V2_NUM_EPOCHS = 20           # 恢复正式训练
V2_LEARNING_RATE = 5e-4
V2_BATCH_SIZE = 2048
# InfoNCE 不能梯度累积! 每个 batch 的 softmax 必须看到所有 in-batch negatives
# 32GB 显存 → 2048 batch, 2047 个负样本, 比 1024 大一倍
V2_GRADIENT_ACCUM_STEPS = 1

# --- Round 3+ 负采样实验开关 (默认关闭, 与现有 8.66% 基线完全一致) ---
NUM_EXPLICIT_NEGATIVES = 0        # 显式负样本 (1-2星) 追加为困难负样本数量 (Round 3+ 负结果 -0.47pp, 已关)
USE_SAME_CATEGORY_HARD_NEG = True  # 同品类难负: ItemCF 相似里只保留同品类 item
DYNAMIC_HARD_NEG_EVERY = 0        # ANCE 式动态难负: 每 N epoch 用当前 embedding 重挖 (0=关闭; Phase1 负增益-0.13pp 已回退)

# --- MIND 多兴趣网络配置 ---
MIND_MODEL_FILE = os.path.join(MODEL_PATH, 'mind_model.pth')
MIND_EMBED_PKL = os.path.join(MODEL_PATH, 'item_embeddings_mind.pkl')
MIND_NUM_INTERESTS = 3
MIND_ROUTING_ITERS = 3
MIND_TEMPERATURE = 0.07
MIND_NUM_EPOCHS = 15
MIND_LEARNING_RATE = 5e-4
MIND_BATCH_SIZE = 512

# --- HSTU 第 5 路召回配置 (与 V2 同超参, 仅用户塔不同: pointwise 传导 vs softmax 注意力) ---
HSTU_MODEL_FILE = os.path.join(MODEL_PATH, 'hstu_model.pth')
HSTU_BEST_FILE = os.path.join(MODEL_PATH, 'hstu_best.pth')
HSTU_EMBED_PKL = os.path.join(MODEL_PATH, 'item_embeddings_hstu.pkl')
HSTU_NUM_BLOCKS = 2          # HSTU block 层数 (对齐 SASREC_NUM_BLOCKS=2)
HSTU_NUM_EPOCHS = 20         # 复用 V2_NUM_EPOCHS
HSTU_LEARNING_RATE = 5e-4    # 复用 V2_LEARNING_RATE
HSTU_BATCH_SIZE = 2048       # 复用 V2_BATCH_SIZE

# --- DIN 精排模型配置 ---
DIN_HIDDEN_DIMS = [256, 128, 64]
DIN_DROPOUT = 0.1
DIN_NEG_RATIO = 4              # BPR: 每正样本配 N 个负样本
# (embed_dim → EMBED_DIM, lr → LEARNING_RATE, batch → BATCH_SIZE,
#  weight_decay → WEIGHT_DECAY, num_epochs → NUM_EPOCHS)

# --- 多路召回融合配置 ---
RECALL_NUM = 100
RECALL_WEIGHTS = {
    'itemcf': 1.5,        # ItemCF — grid search 最优, 单路最强 (10.62%)
    'v2_sasrec': 1.0,     # V2 SASRec (单路 8.66%, 权重下调)
    'category': 0.7,
    'hot': 0.05,          # 兜底
}

# --- 最终推荐数量 ---
FINAL_RECOMMEND_NUM = 5

# --- 评估 & Early Stop 配置 ---
EVAL_SPLIT_RATIO = 0.8       # 每个用户前80%交互作训练集，后20%作验证集 (时序分割)
EARLY_STOP_PATIENCE = 5      # 连续 N 个 epoch 验证指标不提升就停止训练
EVAL_K = 100                  # 评估用的 Top-K (Recall@20 / NDCG@20)
EVAL_MAX_USERS = 10000        # 双品类稠密用户多，3K 更稳
EVAL_NUM_NEGATIVES = 100     # Sampled metrics: 每用户 100 个随机负样本 (与 SASRec 论文对齐)
EVAL_FULL_RANK = True        # 全库评估: Sampled NDCG@20 与全库 HR@100 反相关 (见记忆 val-metric-mismatch), 切回全库
SKIP_EVAL = False            # 本地验证: True (只训练不评估，快速验证pipeline)
CHECKPOINT_DIR = os.path.join(MODEL_PATH, 'checkpoints')

# --- AMP 混合精度 & 梯度累积 ---
USE_AMP = True               # 本地验证: False (兼容CPU); 服务器: True
GRADIENT_ACCUM_STEPS = 4     # 本地验证: 2; 服务器: 4 (BPR 用, V2 不用)

# Best checkpoint 路径 (按验证集 Recall 选出的最佳模型)
V2_BEST_FILE = os.path.join(MODEL_PATH, 'two_tower_v2_best.pth')

# --- ALS 预训练 Item Embedding ---
USE_ALS_INIT = True             # ALS 预训练替代随机初始化 (核心性能提升)
ALS_CACHE = os.path.join(MODEL_PATH, 'als_item_embeddings.pkl')
ALS_FACTORS = None              # None = 使用 EMBED_DIM; 或指定维度
ALS_ITERATIONS = 15             # ALS 迭代次数
ALS_ALPHA = 10.0                # 置信度参数 (10.0 适合隐式反馈)
ALS_FIX_EMBEDDINGS = False      # True=冻结 item_embedding 不训练
ALS_METHOD = 'svd'             # 'svd'=截断SVD(分钟级) | 'lightgcn'=SVD+K层图卷积(已证负-0.64pp) | 'als'=implicit ALS(全量~54h,慎用)
SVD_CACHE = os.path.join(MODEL_PATH, 'svd_item_embeddings.pkl')
LIGHTGCN_CACHE = os.path.join(MODEL_PATH, 'lightgcn_item_embeddings.pkl')
LIGHTGCN_NUM_LAYERS = 3         # LightGCN 传播层数 (2=item-item共现, 3=标准推荐设置; 0=退化为纯SVD)

# ============================================================
# Extended DIN (Raw JSONL) Configuration
# ============================================================
EXT_CATEGORIES = [
    'Musical_Instruments',
    'Office_Products',
    'CDs_and_Vinyl',
    'Video_Games',
    'Toys_and_Games',
]  # raw JSONL (与 AMAZON_CATEGORIES 一致)
EXT_MIN_USER_INTER = 5           # 用户最少交互数 (5-core)
EXT_MIN_ITEM_INTER = 5           # 商品最少交互数 (5-core)
EXT_ENCODER_PKL = os.path.join(TMP_PATH, 'id_encoders_ext.pkl')

DIN_BRAND_EMBED_DIM = 64         # Brand Embedding 维度
DIN_NEG_RATIO = 4                # 每正样本配 N 个负样本 (BPR)
DIN_EXT_MODEL_FILE = os.path.join(MODEL_PATH, 'din_ext_model.pth')
DIN_EXT_BEST_FILE = os.path.join(MODEL_PATH, 'din_ext_best.pth')
