"""
ALS 预训练 Item Embedding:
用 Alternating Least Squares 在用户-物品交互矩阵上学习物品向量,
替代随机初始化, 给双塔模型一个高起点的 embedding。

原理:
  ALS 直接拟合隐式反馈矩阵中的非零元素 (用户买过→偏好高, 未交互→不确定),
  相比 SVD 不需要填 0 (引入噪声), 比随机初始化有几十倍的信号提升。

全量 implicit ALS 成本 ~54h (n_users·k³ 主导), 不可接受 → 默认改用截断 SVD
(ALS_METHOD='svd', randomized, 分钟级 warm-start), 或用 LightGCN 图卷积
(ALS_METHOD='lightgcn', SVD 初始化 + K 层传播注入高阶协同, 分钟级)。

依赖: implicit (ALS_METHOD='als') / scikit-learn (ALS_METHOD='svd'/'lightgcn')
"""

import os, pickle

# Fix OpenBLAS/OpenMP conflict: must be set BEFORE numpy/scipy import
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp
import torch
import os, pickle

import config


def build_user_item_matrix(click_df, user_le, item_le):
    """
    从 click_df (正样本) 构建 CSR 稀疏矩阵。
    只包含正样本 (rating ≥ 4 或 click_label == 1)。

    返回:
      matrix: scipy.sparse.csr_matrix (num_users × num_items), dtype=float32
      confidence: 每非零元素的置信度 (点击次数 → 多击加权)
    """
    num_users = len(user_le.classes_)
    num_items = len(item_le.classes_)

    # 只取正样本
    if 'click_label' in click_df.columns:
        pos = click_df[click_df['click_label'] == 1]
    else:
        pos = click_df

    # 编码用户/物品
    user_idx = user_le.transform(pos['user_id'].values)
    # Amazon 列名可能不同
    if 'click_article_id' in pos.columns:
        item_idx = item_le.transform(pos['click_article_id'].values)
    elif 'parent_asin' in pos.columns:
        item_idx = item_le.transform(pos['parent_asin'].values)
    else:
        raise KeyError("click_df 缺少 click_article_id / parent_asin 列")

    # 构建稀疏矩阵 (data=1 表示交互存在)
    data = np.ones(len(pos), dtype=np.float32)
    matrix = sp.csr_matrix((data, (user_idx, item_idx)),
                          shape=(num_users, num_items), dtype=np.float32)

    return matrix


def train_als_and_save(click_df, user_le, item_le,
                        factors=None, iterations=15, alpha=10.0):
    """
    训练 item embedding 预训练向量并缓存到磁盘 (缓存存在则直接加载)。
    求解器由 config.ALS_METHOD 控制: 'svd' (默认, 分钟级) 或 'als' (implicit, 全量 ~54h)。

    参数:
      click_df: 正样本交互 DataFrame
      user_le, item_le: LabelEncoder
      factors: 隐因子维度 (默认 config.EMBED_DIM)
      iterations: ALS 迭代次数 (SVD 模式复用为 randomized power iteration 次数)
      alpha: 置信度参数 (仅 ALS 模式; 大值 → 高权重交互, 默认 10.0)

    返回:
      item_factors: np.ndarray (num_items, factors), L2 归一化
    """
    if factors is None:
        factors = config.EMBED_DIM

    method = getattr(config, 'ALS_METHOD', 'als')
    if method == 'svd':
        cache_path = getattr(config, 'SVD_CACHE',
                             os.path.join(config.MODEL_PATH, 'svd_item_embeddings.pkl'))
    elif method == 'lightgcn':
        cache_path = getattr(config, 'LIGHTGCN_CACHE',
                             os.path.join(config.MODEL_PATH, 'lightgcn_item_embeddings.pkl'))
    else:
        cache_path = getattr(config, 'ALS_CACHE',
                             os.path.join(config.MODEL_PATH, 'als_item_embeddings.pkl'))

    if os.path.exists(cache_path):
        print(f">>> Loading {method.upper()} item factors from {cache_path}...")
        with open(cache_path, 'rb') as f:
            item_factors = pickle.load(f)
        return item_factors

    print(">>> Building user-item matrix...")
    matrix = build_user_item_matrix(click_df, user_le, item_le)
    print(f"    Matrix: {matrix.shape[0]:,} users × {matrix.shape[1]:,} items, "
          f"{matrix.nnz:,} non-zeros ({matrix.nnz/matrix.shape[0]/matrix.shape[1]*100:.3f}%)")

    if method == 'svd':
        item_factors = _fit_svd(matrix, factors)
    elif method == 'lightgcn':
        item_factors = _fit_lightgcn(matrix, factors,
                                     getattr(config, 'LIGHTGCN_NUM_LAYERS', 3))
    else:
        item_factors = _fit_als(matrix, factors, iterations, alpha)

    if item_factors is None:
        return None

    # L2 归一化 (与双塔输出的归一化一致, 点积 = 余弦相似度)
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    item_factors = item_factors / norms

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(item_factors, f)
    print(f">>> {method.upper()} item factors saved to {cache_path}, shape={item_factors.shape}")

    return item_factors


def _fit_als(matrix, factors, iterations, alpha):
    """implicit ALS 求解 item factors。全量 ~54h (n_users·k³ 主导), 慎用。"""
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        print("[ALS] implicit 未安装, 跳过。安装: pip install implicit")
        return None

    print(f">>> Training ALS (factors={factors}, iters={iterations}, alpha={alpha})...")
    als_model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        alpha=alpha,
        random_state=42,
        use_gpu=False,
        num_threads=16,           # 利用服务器 16 核
    )
    als_model.fit(matrix)
    return als_model.item_factors.astype(np.float32)


def _fit_svd(matrix, factors):
    """
    截断 SVD (randomized) 求解右奇异向量 V 作 item embedding。
    成本 O(nnz·k·n_iter) ≈ 1e10 flops, 分钟级; 维度 256 天然匹配 EMBED_DIM。
    """
    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError:
        print("[SVD] scikit-learn 未安装, 跳过。")
        return None

    n_iter = getattr(config, 'ALS_ITERATIONS', 15)  # 复用 ALS_ITERATIONS 作 power iteration 次数
    print(f">>> Training TruncatedSVD (n_components={factors}, randomized, n_iter={n_iter})...")
    svd = TruncatedSVD(n_components=factors, algorithm='randomized',
                       n_iter=n_iter, random_state=42)
    svd.fit(matrix)
    # components_ 形状 (factors, num_items) → 转置为 (num_items, factors) = 右奇异向量 V
    return svd.components_.T.astype(np.float32)


def _fit_lightgcn(matrix, factors, num_layers=3):
    """
    LightGCN 图卷积 warm-start (SVD 初始化 + K 层传播):
    在 user-item 二分图上做 K 层归一化图卷积, 注入高阶协同信号
    (item-user-item 共现 = ItemCF 的图结构), 取各层平均作为 item embedding。

    相比 SVD (1-hop 最优低秩分解, 只捕获"共同用户的物品相似"):
    - 2-hop = item-user-item → item-item 共现 (ItemCF 的核心信号)
    - 3-hop = user-item-user-item → user-user 兴趣相似
    E^(0) 用 SVD 因子初始化 (保留 1-hop 最优信号), 传播在其上叠加高阶结构,
    严格 ≥ 纯 SVD (num_layers=0 时退化为纯 SVD)。

    成本 O(nnz·dim·K) ≈ 分钟级, 与 SVD 同量级。
    """
    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError:
        print("[LightGCN] scikit-learn 未安装, 跳过。")
        return None

    num_users, num_items = matrix.shape
    dim = factors
    n_iter = getattr(config, 'ALS_ITERATIONS', 15)

    # --- 1. 归一化邻接: R_norm[u,i] = 1 / sqrt(d_u · d_i) (LightGCN 对称归一化) ---
    user_deg = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    item_deg = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    user_deg = np.maximum(user_deg, 1e-8)
    item_deg = np.maximum(item_deg, 1e-8)
    R = matrix.tocoo()
    norm_data = (1.0 / np.sqrt(user_deg[R.row] * item_deg[R.col])).astype(np.float32)
    R_norm = sp.csr_matrix((norm_data, (R.row, R.col)), shape=(num_users, num_items))

    # --- 2. E^(0) = SVD 因子 (1-hop 最优低秩信号) ---
    print(f">>> LightGCN: fitting TruncatedSVD base (n_components={dim}, n_iter={n_iter})...")
    svd = TruncatedSVD(n_components=dim, algorithm='randomized',
                       n_iter=n_iter, random_state=42)
    U_sigma = svd.fit_transform(matrix).astype(np.float32)   # (num_users, dim) ≈ U·Σ
    V = svd.components_.T.astype(np.float32)                 # (num_items, dim) 右奇异向量
    # L2 归一化 (U 带奇异值权重, 需与单位范数的 V 对齐尺度)
    E_user = U_sigma / (np.linalg.norm(U_sigma, axis=1, keepdims=True) + 1e-8)
    E_item = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    # --- 3. K 层传播, 各层累加 (LightGCN 最终 = 各层平均) ---
    print(f">>> LightGCN: propagating {num_layers} layers over "
          f"{num_users:,}×{num_items:,} bipartite graph ({matrix.nnz:,} edges)...")
    E_user_all = E_user
    E_item_all = E_item
    for layer in range(1, num_layers + 1):
        E_user = R_norm @ E_item          # (num_users, dim): 用户聚合其邻居物品
        E_item = R_norm.T @ E_user        # (num_items, dim): 物品聚合其邻居用户
        E_user_all = E_user_all + E_user
        E_item_all = E_item_all + E_item
        print(f"    Layer {layer}/{num_layers} done")

    item_factors = (E_item_all / (num_layers + 1)).astype(np.float32)
    return item_factors


def _get_item_embed_param(model):
    """自适应获取模型的 item_embedding 参数 (兼容双塔和精排)"""
    # 双塔系列: ItemTower 的 item_embedding
    if hasattr(model, 'item_tower') and hasattr(model.item_tower, 'item_embedding'):
        return model.item_tower.item_embedding.weight
    # DINExtendedModel: 直接挂在 model 上
    if hasattr(model, 'item_embedding'):
        return model.item_embedding.weight
    return None


def init_model_with_als(model, click_df, user_le, item_le, fix_embeddings=False):
    """
    用 ALS 预训练向量初始化模型的 item_embedding。
    自适应兼容 TwoTowerV2Model / MINDModel / DINExtendedModel。

    参数:
      model: 任意包含 item_embedding 的模型
      click_df: 正样本交互 DataFrame
      user_le, item_le: LabelEncoder
      fix_embeddings: True → 冻结 item_embedding 不训练 (仅精排时用)
    """
    item_factors = train_als_and_save(click_df, user_le, item_le)
    if item_factors is None:
        print("[ALS] 预训练失败, 使用随机初始化。")
        return model, item_factors is not None

    # 自适应获取 item_embedding 权重
    embed_weight = _get_item_embed_param(model)
    if embed_weight is None:
        print("[ALS] 找不到 item_embedding 参数, 跳过。")
        return model, False

    num_items_model, embed_dim_model = embed_weight.shape

    if item_factors.shape[1] != embed_dim_model:
        print(f"[ALS] 维度不匹配: ALS={item_factors.shape[1]}, Model={embed_dim_model}, 跳过。")
        return model, False

    # 填充对齐 (ALS 可能比 item_le 少一些 padding)
    n_als = min(item_factors.shape[0], num_items_model)
    n_embed = min(item_factors.shape[1], embed_dim_model)

    with torch.no_grad():
        embed_weight.data[:n_als, :n_embed] = (
            torch.FloatTensor(item_factors[:n_als, :n_embed])
        )

        if fix_embeddings:
            embed_weight.requires_grad_(False)
            print("[ALS] Item embedding 已冻结 (不参与训练)")

    print(f"[ALS] 初始化完成: {n_als:,} items × {n_embed} dims 从 ALS 载入")

    return model, True
