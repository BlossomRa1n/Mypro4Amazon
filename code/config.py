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
    'Musical_Instruments',   # ~25万 条正反馈,  ~1.4万用户,  ~1.0万商品
    'Office_Products',       # ~50万 条正反馈,  ~8.0万用户,  ~2.5万商品
    'All_Beauty',            # ~100万条正反馈,  ~9.0万用户,  ~3.5万商品
]
# 服务器可追加: 'Video_Games', 'Toys_and_Games', 'CDs_and_Vinyl' 等

OFFLINE_MODE = False         # True: 每品类只读10000条; False: 全量
AMAZON_SAMPLE_USERS = None    # 本地验证: 1000; 5060Ti: None

USER_DATA_PATH = os.path.join(ROOT_PATH, 'user_data')
MODEL_PATH = os.path.join(USER_DATA_PATH, 'model_data')
TMP_PATH = os.path.join(USER_DATA_PATH, 'tmp_data')
RESULT_PATH = os.path.join(ROOT_PATH, 'prediction_result')

ITEMCF_SIM_PKL = os.path.join(MODEL_PATH, 'itemcf_i2i_sim.pkl')

DEEP_MODEL_FILE = os.path.join(MODEL_PATH, 'two_tower_model.pth')
EMBED_PKL = os.path.join(MODEL_PATH, 'item_embeddings.pkl')
ENCODER_PKL = os.path.join(TMP_PATH, 'id_encoders.pkl')

# --- 模型超参数 ---
EMBED_DIM = 64
HIDDEN_DIMS = [256, 128]
HIST_LEN = 50
BATCH_SIZE = 1024
NUM_WORKERS = 8               # DataLoader 线程数 (云服务器 16 vCPU: 8; 本地: 0-2)
NUM_EPOCHS = 10               # 本地验证: 3; 正式训练: 10+
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
INFONCE_TEMPERATURE = 0.2
NUM_HARD_NEGATIVES = 0
V2_NUM_EPOCHS = 15           # 本地验证: 3; 正式训练: 15+
V2_LEARNING_RATE = 5e-4
V2_BATCH_SIZE = 512

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
DIN_EMBED_DIM = 64
DIN_HIDDEN_DIMS = [256, 128, 64]
DIN_DROPOUT = 0.1
DIN_NUM_EPOCHS = 10          # 本地验证: 3; 正式训练: 10+
DIN_LEARNING_RATE = 1e-3
DIN_BATCH_SIZE = 1024
DIN_WEIGHT_DECAY = 1e-5

# --- 多路召回融合配置 ---
RECALL_NUM = 50
RECALL_WEIGHTS = {
    'itemcf': 1.0,
    'embedding': 1.0,
    'category': 0.5,
    'hot': 0.1,
}

# --- 最终推荐数量 ---
FINAL_RECOMMEND_NUM = 5

# --- 评估 & Early Stop 配置 ---
EVAL_SPLIT_RATIO = 0.8       # 每个用户前80%交互作训练集，后20%作验证集 (时序分割)
EARLY_STOP_PATIENCE = 5      # 连续 N 个 epoch 验证指标不提升就停止训练
EVAL_K = 20                  # 评估用的 Top-K (Recall@20 / NDCG@20)
EVAL_MAX_USERS = 3000        # 评估采样用户数 (正式训练建议 3000)
SKIP_EVAL = False            # 本地验证: True (只训练不评估，快速验证pipeline)
CHECKPOINT_DIR = os.path.join(MODEL_PATH, 'checkpoints')

# --- AMP 混合精度 & 梯度累积 ---
USE_AMP = True               # 本地验证: False (兼容CPU); 服务器: True
GRADIENT_ACCUM_STEPS = 4     # 本地验证: 2; 服务器: 4
                             # V2_BATCH_SIZE=512 + accum=4 → 等效 batch=2048
                             # InfoNCE 负样本池 = 2048-1, 比 512 大 4 倍

# Best checkpoint 路径 (按验证集 Recall 选出的最佳模型)
DEEP_BEST_FILE = os.path.join(MODEL_PATH, 'two_tower_best.pth')
V2_BEST_FILE = os.path.join(MODEL_PATH, 'two_tower_v2_best.pth')
DIN_BEST_FILE = os.path.join(MODEL_PATH, 'din_best.pth')
