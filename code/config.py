import os

curr_path = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(curr_path)

# ============================================================
# Amazon Reviews 2023 — 数据集 & 路径配置
# ============================================================

DATA_PATH = os.path.join(ROOT_PATH, 'amazon_reviews')

# --- 三档规模控制 ---
# 档位 1 (本地验证): offline=True → 只看 10000 条
# 档位 2 (5060Ti): 3 品类全量 ~175 万条正反馈
# 档位 3 (服务器): 追加更多品类到列表中
AMAZON_CATEGORIES = [
    'All_Beauty',             # 极稀疏, 5-core 后仅 253 用户 — 几乎不贡献用户但品类 Embedding 多一个值
    'Video_Games',            # ~92K 用户, ~23K 商品, 人均 ~8
    'CDs_and_Vinyl',          # ~123K 用户, ~88K 商品, 人均 ~12 — 最稠密
]
# 三品类混合: 品类 Embedding(3 个值有区分度) + 品类召回恢复有效
# SASRec 论文对标: Beauty + Games (2014 版), 此处用 2023 版近似

OFFLINE_MODE = False         # True: 每品类只读10000条; False: 全量
AMAZON_SAMPLE_USERS = None    # 本地验证: 1000; 5060Ti: None
AMAZON_MIN_USER_INTERACTIONS = 0   # Sampled 评估已对齐论文，不需要稠密过滤

USER_DATA_PATH = os.path.join(ROOT_PATH, 'user_data')
MODEL_PATH = os.path.join(USER_DATA_PATH, 'model_data')
TMP_PATH = os.path.join(USER_DATA_PATH, 'tmp_data')
RESULT_PATH = os.path.join(ROOT_PATH, 'prediction_result')

ITEMCF_SIM_PKL = os.path.join(MODEL_PATH, 'itemcf_i2i_sim.pkl')

DEEP_MODEL_FILE = os.path.join(MODEL_PATH, 'two_tower_model.pth')
EMBED_PKL = os.path.join(MODEL_PATH, 'item_embeddings.pkl')
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
INFONCE_TEMPERATURE = 0.3
NUM_HARD_NEGATIVES = 0
V2_NUM_EPOCHS = 20           # 恢复正式训练
V2_LEARNING_RATE = 5e-4
V2_BATCH_SIZE = 2048
# InfoNCE 不能梯度累积! 每个 batch 的 softmax 必须看到所有 in-batch negatives
# 32GB 显存 → 2048 batch, 2047 个负样本, 比 1024 大一倍
V2_GRADIENT_ACCUM_STEPS = 1

# --- MIND 多兴趣网络配置 ---
MIND_MODEL_FILE = os.path.join(MODEL_PATH, 'mind_model.pth')
MIND_EMBED_PKL = os.path.join(MODEL_PATH, 'item_embeddings_mind.pkl')
MIND_NUM_INTERESTS = 3
MIND_ROUTING_ITERS = 3
MIND_TEMPERATURE = 0.07
MIND_NUM_EPOCHS = 15
MIND_LEARNING_RATE = 5e-4
MIND_BATCH_SIZE = 512

# --- DIN 精排模型配置 ---
DIN_MODEL_FILE = os.path.join(MODEL_PATH, 'din_model.pth')
DIN_EMBED_DIM = 256
DIN_HIDDEN_DIMS = [256, 128, 64]
DIN_DROPOUT = 0.1
DIN_NUM_EPOCHS = 10          # 本地验证: 3; 正式训练: 10+
DIN_LEARNING_RATE = 1e-3
DIN_BATCH_SIZE = 1024
DIN_WEIGHT_DECAY = 1e-5
DIN_POS_WEIGHT = 4.0         # BCE 正样本权重: 负样本/正样本 ≈ 4, 防止"一律说不买"

# --- 多路召回融合配置 ---
RECALL_NUM = 50
RECALL_WEIGHTS = {
    'itemcf': 1.0,        # ItemCF — 主力通道
    'v2_sasrec': 0.0,     # V2 双塔关闭 (NDCG≈0.0056, 接近随机)
    'v1_bpr': 0.0,        # V1 BPR 双塔关闭
    'category': 0.5,      # 3 品类有区分度
    'hot': 0.1,           # 兜底
}

# --- 最终推荐数量 ---
FINAL_RECOMMEND_NUM = 5

# --- 评估 & Early Stop 配置 ---
EVAL_SPLIT_RATIO = 0.8       # 每个用户前80%交互作训练集，后20%作验证集 (时序分割)
EARLY_STOP_PATIENCE = 5      # 连续 N 个 epoch 验证指标不提升就停止训练
EVAL_K = 20                  # 评估用的 Top-K (Recall@20 / NDCG@20)
EVAL_MAX_USERS = 3000        # 双品类稠密用户多，3K 更稳
EVAL_NUM_NEGATIVES = 100     # Sampled metrics: 每用户 100 个随机负样本 (与 SASRec 论文对齐)
EVAL_FULL_RANK = False       # 消融实验后切回 Sampled 评估
SKIP_EVAL = False            # 本地验证: True (只训练不评估，快速验证pipeline)
CHECKPOINT_DIR = os.path.join(MODEL_PATH, 'checkpoints')

# --- AMP 混合精度 & 梯度累积 ---
USE_AMP = True               # 本地验证: False (兼容CPU); 服务器: True
GRADIENT_ACCUM_STEPS = 4     # 本地验证: 2; 服务器: 4 (BPR 用, V2 不用)

# Best checkpoint 路径 (按验证集 Recall 选出的最佳模型)
DEEP_BEST_FILE = os.path.join(MODEL_PATH, 'two_tower_best.pth')
V2_BEST_FILE = os.path.join(MODEL_PATH, 'two_tower_v2_best.pth')
DIN_BEST_FILE = os.path.join(MODEL_PATH, 'din_best.pth')

# --- ALS 预训练 Item Embedding ---
USE_ALS_INIT = True             # ALS 预训练替代随机初始化 (核心性能提升)
ALS_CACHE = os.path.join(MODEL_PATH, 'als_item_embeddings.pkl')
ALS_FACTORS = None              # None = 使用 EMBED_DIM; 或指定维度
ALS_ITERATIONS = 15             # ALS 迭代次数
ALS_ALPHA = 10.0                # 置信度参数 (10.0 适合隐式反馈)
ALS_FIX_EMBEDDINGS = False      # True=冻结 item_embedding 不训练

# ============================================================
# Extended DIN (All_Beauty Raw) Configuration
# ============================================================
EXT_CATEGORIES = ['All_Beauty']
EXT_MIN_USER_INTER = 5           # 用户最少交互数 (5-core)
EXT_MIN_ITEM_INTER = 5           # 商品最少交互数 (5-core)
EXT_ENCODER_PKL = os.path.join(TMP_PATH, 'id_encoders_ext.pkl')

DIN_BRAND_EMBED_DIM = 64         # Brand Embedding 维度
DIN_NEG_RATIO = 4                # 每正样本配 N 个负样本 (BPR)
DIN_EXT_MODEL_FILE = os.path.join(MODEL_PATH, 'din_ext_model.pth')
DIN_EXT_BEST_FILE = os.path.join(MODEL_PATH, 'din_ext_best.pth')
