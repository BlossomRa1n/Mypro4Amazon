"""
ALS 预训练 Item Embedding:
用 Alternating Least Squares 在用户-物品交互矩阵上学习物品向量,
替代随机初始化, 给双塔模型一个高起点的 embedding。

原理:
  ALS 直接拟合隐式反馈矩阵中的非零元素 (用户买过→偏好高, 未交互→不确定),
  相比 SVD 不需要填 0 (引入噪声), 比随机初始化有几十倍的信号提升。

依赖: pip install implicit
"""

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
    训练 ALS 并缓存 item_factors 到磁盘。
    如果缓存存在则直接加载。

    参数:
      click_df: 正样本交互 DataFrame
      user_le, item_le: LabelEncoder
      factors: ALS 隐因子维度 (默认 config.EMBED_DIM)
      iterations: ALS 迭代次数
      alpha: 置信度参数 (大值 → 高权重交互, 默认 10.0 适合二值交互)

    返回:
      item_factors: np.ndarray (num_items, factors), L2 归一化
    """
    if factors is None:
        factors = config.EMBED_DIM

    cache_path = getattr(config, 'ALS_CACHE',
                         os.path.join(config.MODEL_PATH, 'als_item_embeddings.pkl'))

    if os.path.exists(cache_path):
        print(f">>> Loading ALS item factors from {cache_path}...")
        with open(cache_path, 'rb') as f:
            item_factors = pickle.load(f)
        return item_factors

    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        print("[ALS] implicit 未安装, 跳过。安装: pip install implicit")
        return None

    print(f">>> Building user-item matrix for ALS...")
    matrix = build_user_item_matrix(click_df, user_le, item_le)
    print(f"    Matrix: {matrix.shape[0]:,} users × {matrix.shape[1]:,} items, "
          f"{matrix.nnz:,} non-zeros ({matrix.nnz/matrix.shape[0]/matrix.shape[1]*100:.3f}%)")

    print(f">>> Training ALS (factors={factors}, iters={iterations}, alpha={alpha})...")
    als_model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        alpha=alpha,
        random_state=42,
        use_gpu=False,           # ALS 在 CPU 上反而更快 (IMPLICIT 库的限制)
        num_threads=8,
    )
    als_model.fit(matrix)

    item_factors = als_model.item_factors.astype(np.float32)

    # L2 归一化 (与双塔输出的归一化一致, 点积 = 余弦相似度)
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    item_factors = item_factors / norms

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(item_factors, f)
    print(f">>> ALS item factors saved to {cache_path}, shape={item_factors.shape}")

    return item_factors


def _get_item_embed_param(model):
    """自适应获取模型的 item_embedding 参数 (兼容双塔和 DIN)"""
    # 双塔系列: ItemTower 的 item_embedding
    if hasattr(model, 'item_tower') and hasattr(model.item_tower, 'item_embedding'):
        return model.item_tower.item_embedding.weight
    # DINModel: 直接挂在 model 上
    if hasattr(model, 'item_embedding'):
        return model.item_embedding.weight
    return None


def init_model_with_als(model, click_df, user_le, item_le, fix_embeddings=False):
    """
    用 ALS 预训练向量初始化模型的 item_embedding。
    自适应兼容 TwoTowerModel / TwoTowerV2Model / MINDModel / DINModel。

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
