import pandas as pd
import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils import reduce_mem
import config


# ============================================================
# Amazon Reviews 2023 — 多品类数据加载
# ============================================================

def load_amazon_reviews(data_path, categories, offline=False):
    """
    加载一个或多个 Amazon 品类的 benchmark CSV，合并为统一 DataFrame。

    Rating 利用策略:
      - 5 分 → click_label=1 (正样本), click_weight=1.0
      - 4 分 → click_label=1 (正样本), click_weight=0.5
      - 3 分 → 丢弃 (中性评分，信号弱)
      - 1-2分 → click_label=0 (显式负样本), click_weight=0.8

    数据来源: Hugging Face McAuley-Lab/Amazon-Reviews-2023
    下载: benchmark/5core/rating_only/<Category>.csv
    """
    RATING_MAP = {5.0: (1, 1.0), 4.0: (1, 0.5), 2.0: (0, 0.8), 1.0: (0, 0.8)}

    all_ratings = []

    for cat in categories:
        ratings_path = os.path.join(data_path, f'{cat}.csv')
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"Amazon reviews CSV not found: {ratings_path}\n"
                f"Download from: https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/"
                f"resolve/main/benchmark/5core/rating_only/{cat}.csv"
            )

        if offline:
            print(f">>> [Offline] Loading partial {cat}...")
            df = pd.read_csv(ratings_path, nrows=10000)
        else:
            print(f">>> Loading {cat}...")
            df = pd.read_csv(ratings_path)

        # Rating → label & weight. 3分丢弃
        if 'rating' in df.columns:
            df['click_label'], df['click_weight'] = zip(*df['rating'].map(
                lambda r: RATING_MAP.get(r, (None, None))
            ))
            df = df.dropna(subset=['click_label']).copy()
            df['click_label'] = df['click_label'].astype('int8')
            df['click_weight'] = df['click_weight'].astype('float32')

        df = df.rename(columns={
            'parent_asin': 'click_article_id',
            'timestamp': 'click_timestamp'
        })

        all_ratings.append(df)

    ratings = pd.concat(all_ratings, ignore_index=True)
    ratings = ratings.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])

    # 可选：限制用户数量 (只对正样本用户采样)
    if config.AMAZON_SAMPLE_USERS is not None and not offline:
        pos_users = ratings[ratings['click_label'] == 1]['user_id'].unique()
        if len(pos_users) > config.AMAZON_SAMPLE_USERS:
            sampled = np.random.choice(pos_users, size=config.AMAZON_SAMPLE_USERS, replace=False)
            ratings = ratings[ratings['user_id'].isin(sampled)]
            print(f">>> Sampled {config.AMAZON_SAMPLE_USERS} positive users")

    # 可选：只保留交互数≥阈值的稠密用户 (Embedding 需要足够梯度信号)
    min_inter = getattr(config, 'AMAZON_MIN_USER_INTERACTIONS', None)
    if min_inter is not None and min_inter > 0:
        user_pos_counts = ratings[ratings['click_label'] == 1].groupby('user_id').size()
        dense_users = user_pos_counts[user_pos_counts >= min_inter].index
        ratings = ratings[ratings['user_id'].isin(dense_users)]
        print(f">>> Dense user filter: ≥{min_inter} positive interactions → "
              f"{len(dense_users):,} users retained")

    ratings = reduce_mem(ratings)

    pos = ratings[ratings['click_label'] == 1]
    neg = ratings[ratings['click_label'] == 0]
    print(f">>> Total: {len(ratings):,} interactions")
    print(f"    正样本 (4-5分): {len(pos):,} ({len(pos)/len(ratings)*100:.0f}%) | "
          f"显式负样本 (1-2分): {len(neg):,} ({len(neg)/len(ratings)*100:.0f}%)")
    print(f"    用户: {pos['user_id'].nunique():,} | 物品: {ratings['click_article_id'].nunique():,} "
          f"| 品类: {', '.join(categories)}")
    return ratings


def load_amazon_products(data_path, categories):
    """
    从多个品类的 Amazon Reviews CSV 构建物品元数据。
    每个商品的 category_id 设为其来源品类名，品类 Embedding 从此有意义。

    article_id  ← parent_asin
    category_id ← 品类名称 (如 'Office_Products', 'Video_Games')
    created_at_ts ← 每个 item 的最早评论时间戳
    """
    all_products = []

    for cat in categories:
        ratings_path = os.path.join(data_path, f'{cat}.csv')
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Amazon reviews CSV not found: {ratings_path}")

        ratings = pd.read_csv(ratings_path, usecols=['parent_asin', 'timestamp'])

        products = ratings.groupby('parent_asin').agg(
            created_at_ts=('timestamp', 'min'),
        ).reset_index()

        products = products.rename(columns={'parent_asin': 'article_id'})
        products['category_id'] = cat
        all_products.append(products)

    products = pd.concat(all_products, ignore_index=True)
    # 同一个 parent_asin 可能出现在多个品类？去重保留首次出现的 category
    products = products.drop_duplicates('article_id', keep='first')

    products = reduce_mem(products)
    print(f">>> Products: {len(products)} items across {len(categories)} categories")
    return products


# ============================================================
# 统一数据加载入口
# ============================================================

def get_all_click_df(data_path, offline=False):
    """加载全部点击日志 (含正样本 + 显式负样本)"""
    return load_amazon_reviews(data_path, config.AMAZON_CATEGORIES, offline=offline)


def get_positive_click_df(click_df):
    """只取正样本 (4-5 分)，用于构建用户序列和 ItemCF"""
    return click_df[click_df['click_label'] == 1].copy()


def get_user_item_time(click_df):
    print(">>> Constructing User-Item-Time dictionary...")
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = (
        click_df.groupby('user_id')[['click_article_id', 'click_timestamp']]
        .apply(lambda x: make_item_time_pair(x))
        .reset_index()
        .rename(columns={0: 'item_time_list'})
    )

    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict


def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


def load_articles(data_path):
    """加载物品元数据"""
    return load_amazon_products(data_path, config.AMAZON_CATEGORIES)


def build_encoders(click_df, articles_df, encoder_path):
    """构建 LabelEncoder 映射 raw ID -> 连续整数"""
    if os.path.exists(encoder_path):
        print(f">>> Loading encoders from {encoder_path}...")
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        return encoders

    print(">>> Building ID encoders...")
    encoders = {}

    user_le = LabelEncoder()
    all_users = click_df['user_id'].unique()
    user_le.fit(all_users)
    encoders['user_id'] = user_le
    encoders['raw_to_idx'] = {raw: idx for idx, raw in enumerate(user_le.classes_)}

    item_le = LabelEncoder()
    all_items = np.union1d(
        click_df['click_article_id'].unique(),
        articles_df['article_id'].unique()
    )
    item_le.fit(all_items)
    encoders['item_id'] = item_le

    if 'category_id' in articles_df.columns:
        cat_le = LabelEncoder()
        all_cats = articles_df['category_id'].unique()
        cat_le.fit(all_cats)
        encoders['category_id'] = cat_le

    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f">>> Encoders saved to {encoder_path}")

    return encoders


def build_item_features(articles_df, click_df, encoders):
    """构建物品特征 (index=encoded item_idx)"""
    print(">>> Building item features...")
    item_le = encoders['item_id']

    articles_df = articles_df.copy()
    articles_df['item_idx'] = item_le.transform(articles_df['article_id'])

    if 'category_id' in articles_df.columns and 'category_id' in encoders:
        cat_le = encoders['category_id']
        articles_df['category_idx'] = cat_le.transform(articles_df['category_id'])
    else:
        articles_df['category_idx'] = 0

    if 'created_at_ts' in articles_df.columns:
        ts = articles_df['created_at_ts'].astype(float)
        articles_df['created_at_ts_norm'] = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8)
    else:
        articles_df['created_at_ts_norm'] = 0.0

    item_click_count = click_df.groupby('click_article_id').size().reset_index(name='item_click_count')
    articles_df = articles_df.merge(item_click_count, left_on='article_id', right_on='click_article_id', how='left')
    articles_df['item_click_count'] = articles_df['item_click_count'].fillna(0)
    icc = articles_df['item_click_count'].astype(float)
    articles_df['item_click_count_norm'] = (icc - icc.min()) / (icc.max() - icc.min() + 1e-8)

    feature_cols = ['item_idx', 'category_idx', 'item_click_count_norm', 'created_at_ts_norm']
    item_features = articles_df[feature_cols].set_index('item_idx').sort_index()

    all_item_indices = np.arange(len(item_le.classes_))
    full_features = pd.DataFrame(index=all_item_indices)
    full_features = full_features.join(item_features)
    full_features = full_features.fillna(0)

    return full_features


def build_user_features(click_df, encoders, hist_len=50):
    """构建用户特征"""
    print(">>> Building user features...")
    item_le = encoders['item_id']
    click_df = click_df.copy()
    click_df['item_idx'] = item_le.transform(click_df['click_article_id'])

    click_df = click_df.sort_values('click_timestamp')

    user_hist = click_df.groupby('user_id')['item_idx'].apply(list).reset_index()
    user_hist.columns = ['user_id', 'hist_items']

    user_stats = click_df.groupby('user_id').agg(
        click_count=('click_article_id', 'count'),
        time_span=('click_timestamp', lambda x: x.max() - x.min())
    ).reset_index()

    user_features = user_hist.merge(user_stats, on='user_id', how='left')

    user_features['hist_items_trunc'] = user_features['hist_items'].apply(
        lambda x: x[-hist_len:] if len(x) > hist_len else x
    )
    user_features['hist_len'] = user_features['hist_items_trunc'].apply(len)

    max_click = user_features['click_count'].max()
    user_features['click_count_norm'] = user_features['click_count'] / (max_click + 1e-8)
    max_span = user_features['time_span'].max()
    user_features['time_span_norm'] = user_features['time_span'] / (max_span + 1e-8)

    return user_features


def build_enhanced_user_features(click_df, articles_df, encoders, hist_len=50):
    """
    增强版用户特征工程: 新增用户兴趣画像特征
    1. 品类偏好分布 (top3 品类及偏好强度)
    2. 活跃度分层 (高/中/低活跃用户)
    3. 兴趣集中度 (熵，衡量兴趣是否集中)
    """
    print(">>> Building enhanced user features with interest profile...")

    user_features = build_user_features(click_df, encoders, hist_len)

    item_le = encoders['item_id']
    click_df = click_df.copy()
    click_df['item_idx'] = item_le.transform(click_df['click_article_id'])

    if 'category_id' in articles_df.columns:
        item_category = dict(zip(
            item_le.transform(articles_df['article_id']),
            articles_df['category_id']
        ))

        user_cat_pref = {}
        user_interest_entropy = {}

        for uid, group in click_df.groupby('user_id'):
            cat_counts = {}
            for item_idx in group['item_idx']:
                cat = item_category.get(item_idx)
                if cat is not None:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

            total = sum(cat_counts.values())
            if total > 0:
                for cat in cat_counts:
                    cat_counts[cat] /= total

                top3_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:3]
                user_cat_pref[uid] = top3_cats

                entropy = -sum(p * np.log(p + 1e-8) for p in cat_counts.values() if p > 0)
                max_entropy = np.log(len(cat_counts)) if len(cat_counts) > 1 else 1.0
                user_interest_entropy[uid] = entropy / (max_entropy + 1e-8)
            else:
                user_cat_pref[uid] = []
                user_interest_entropy[uid] = 0.0

        user_features['interest_entropy'] = user_features['user_id'].map(
            lambda x: user_interest_entropy.get(x, 0.0)
        )

        click_count_median = user_features['click_count'].median()
        user_features['activity_level'] = user_features['click_count'].apply(
            lambda x: 2 if x >= click_count_median * 2 else (1 if x >= click_count_median else 0)
        )
        user_features['activity_level_norm'] = user_features['activity_level'] / 2.0
        user_features['interest_entropy_norm'] = (
            user_features['interest_entropy'] - user_features['interest_entropy'].min()
        ) / (user_features['interest_entropy'].max() - user_features['interest_entropy'].min() + 1e-8)

    return user_features


def build_hard_negative_index(i2i_sim, encoders, num_hard_negatives=4):
    """Hard Negative Mining: 从 ItemCF 相似但用户未点击的 item 中挖掘困难负样本"""
    print(">>> Building hard negative index from ItemCF similarity...")
    item_le = encoders['item_id']

    # 预建 raw→idx 映射 (O(1) 替代 transform() 的 O(N) 扫)
    raw_to_enc = {cls: i for i, cls in enumerate(item_le.classes_)}
    import heapq

    hard_neg_index = {}
    for raw_item, sim_items in tqdm(i2i_sim.items(), desc="HardNeg Index"):
        if raw_item not in raw_to_enc:
            continue
        item_idx = raw_to_enc[raw_item]

        # heapq.nlargest 取 Top-K, 比全量 sorted() 快 10-50 倍
        import heapq
        top_k = heapq.nlargest(
            num_hard_negatives * 3, sim_items.items(),
            key=lambda x: x[1]
        )
        hard_negs = []
        for sim_item, _ in top_k:
            enc = raw_to_enc.get(sim_item)
            if enc is not None:
                hard_negs.append(enc)
            if len(hard_negs) >= num_hard_negatives:
                break
        hard_neg_index[item_idx] = hard_negs

    print(f">>> Built hard negative index for {len(hard_neg_index):,} items")
    return hard_neg_index


def build_same_category_hard_neg_index(item_features, num_items, num_hard_negatives=4):
    """同品类难负: 每个 item 的难负 = 同品类下最热门的其他 item (天然难区分)。
    item_features: DataFrame, index=item_idx, 含 category_idx + item_click_count_norm
    """
    print(">>> Building same-category hard negative index...")
    from collections import defaultdict
    idx_arr = np.arange(num_items)
    cat_arr = item_features['category_idx'].reindex(idx_arr, fill_value=0).to_numpy(dtype=np.int64)
    pop_arr = item_features['item_click_count_norm'].reindex(idx_arr, fill_value=0).to_numpy(dtype=np.float32)

    cat_items = defaultdict(list)
    for idx, cat in zip(idx_arr, cat_arr):
        cat_items[int(cat)].append(idx)

    hard_neg_index = {}
    for cat, items in cat_items.items():
        items_sorted = sorted(items, key=lambda i: -pop_arr[i])
        m = len(items_sorted)
        for item in items_sorted:
            negs = []
            for cand in items_sorted:
                if cand != item:
                    negs.append(cand)
                if len(negs) >= num_hard_negatives:
                    break
            hard_neg_index[item] = negs

    print(f">>> Built same-category hard negative index for {len(hard_neg_index):,} items")
    return hard_neg_index


# ============================================================
# PyTorch Dataset 类
# ============================================================

class TwoTowerDataset(Dataset):
    def __init__(self, click_df, user_features, item_features, encoders, num_items,
                 hist_len=50, num_negatives=4):
        self.user_le = encoders['user_id']
        self.item_le = encoders['item_id']
        self.raw_to_idx = encoders.get('raw_to_idx', None)

        # 只用正样本做 pairwise 训练
        click_df = click_df[click_df['click_label'] == 1].copy()
        click_df['user_idx'] = self.user_le.transform(click_df['user_id'])
        click_df['item_idx'] = self.item_le.transform(click_df['click_article_id'])

        self.interactions = click_df[['user_idx', 'item_idx']].values
        self.weights = click_df['click_weight'].values.astype(np.float32)

        self.num_items = num_items
        self.hist_len = hist_len
        self.num_negatives = num_negatives

        # --- 预构建 idx_to_raw 反向映射 (只做一次，不在 __getitem__ 里重复建) ---
        if self.raw_to_idx is not None:
            self.idx_to_raw = {v: k for k, v in self.raw_to_idx.items()}
        else:
            self.idx_to_raw = None

        # --- 构建 raw_id -> user_features 行的快速查找 ---
        user_feat_idx = {}
        for i, row in user_features.iterrows():
            user_feat_idx[row['user_id']] = i
        self._user_feat_rows = user_features.iloc  # 整行访问

        # --- 预构建 user 特征数组 (按 user_idx 索引) ---
        num_users = len(self.user_le.classes_)
        self.user_hist_arr = [None] * num_users
        self.user_click_cnt_arr = np.zeros(num_users, dtype=np.float32)
        self.user_time_span_arr = np.zeros(num_users, dtype=np.float32)

        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            if self.raw_to_idx is not None:
                uid_idx = self.raw_to_idx.get(raw_uid)
            elif raw_uid in self.user_le.classes_:
                uid_idx = self.user_le.transform([raw_uid])[0]
            else:
                uid_idx = None
            if uid_idx is not None:
                self.user_hist_arr[uid_idx] = row['hist_items_trunc']
                self.user_click_cnt_arr[uid_idx] = row['click_count_norm']
                self.user_time_span_arr[uid_idx] = row['time_span_norm']

        # --- 预构建 item 特征数组 (按 item_idx 索引) ---
        num_all_items = len(self.item_le.classes_)
        self.item_cat_arr = np.zeros(num_all_items, dtype=np.int64)
        self.item_click_arr = np.zeros(num_all_items, dtype=np.float32)
        self.item_created_arr = np.zeros(num_all_items, dtype=np.float32)
        for idx in item_features.index:
            self.item_cat_arr[idx] = int(item_features.loc[idx].get('category_idx', 0))
            self.item_click_arr[idx] = float(item_features.loc[idx].get('item_click_count_norm', 0))
            self.item_created_arr[idx] = float(item_features.loc[idx].get('created_at_ts_norm', 0))

        self.item_popularity = np.ones(num_items, dtype=np.float32)
        item_counts = click_df['item_idx'].value_counts()
        for idx, cnt in item_counts.items():
            self.item_popularity[idx] = cnt
        self.item_popularity = self.item_popularity / self.item_popularity.sum()

    def __len__(self):
        return len(self.interactions)

    def _get_user_data(self, user_idx):
        hist_items = self.user_hist_arr[user_idx]
        if hist_items is None:
            hist_items = []
        hist_len_actual = len(hist_items)

        if hist_len_actual < self.hist_len:
            padded = hist_items + [0] * (self.hist_len - hist_len_actual)
        else:
            padded = hist_items[-self.hist_len:]
            hist_len_actual = self.hist_len

        return {
            'hist_items': torch.LongTensor(padded),
            'hist_len': torch.tensor(hist_len_actual, dtype=torch.long),
            'click_count': torch.tensor(self.user_click_cnt_arr[user_idx], dtype=torch.float32),
            'time_span': torch.tensor(self.user_time_span_arr[user_idx], dtype=torch.float32),
        }

    def _get_item_data(self, item_idx):
        return {
            'category_id': torch.tensor(self.item_cat_arr[item_idx], dtype=torch.long),
            'item_click_count': torch.tensor(self.item_click_arr[item_idx], dtype=torch.float32),
            'created_at_ts': torch.tensor(self.item_created_arr[item_idx], dtype=torch.float32),
        }

    def __getitem__(self, idx):
        user_idx, pos_item_idx = self.interactions[idx]

        user_data = self._get_user_data(user_idx)
        pos_item_data = self._get_item_data(pos_item_idx)

        neg_item_indices = np.random.choice(
            self.num_items, size=self.num_negatives, replace=False, p=self.item_popularity
        )
        neg_item_idx = neg_item_indices[0]
        neg_item_data = self._get_item_data(neg_item_idx)

        return {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            'hist_items': user_data['hist_items'],
            'hist_len': user_data['hist_len'],
            'click_count': user_data['click_count'],
            'time_span': user_data['time_span'],
            'pos_item_id': torch.tensor(pos_item_idx, dtype=torch.long),
            'pos_category_id': pos_item_data['category_id'],
            'pos_item_click_count': pos_item_data['item_click_count'],
            'pos_created_at_ts': pos_item_data['created_at_ts'],
            'neg_item_id': torch.tensor(neg_item_idx, dtype=torch.long),
            'neg_category_id': neg_item_data['category_id'],
            'neg_item_click_count': neg_item_data['item_click_count'],
            'neg_created_at_ts': neg_item_data['created_at_ts'],
            'click_weight': torch.tensor(self.weights[idx], dtype=torch.float32),
        }


class TwoTowerV2Dataset(Dataset):
    """增强版双塔数据集: InfoNCE (In-batch negatives) + Hard Negative Mining
    + 扩展特征: brand/quality (item) + user_stats (user)
    """
    def __init__(self, click_df, user_features, item_features, encoders, num_items,
                 hist_len=50, hard_neg_index=None, num_hard_negatives=4, num_brands=0,
                 num_explicit_negatives=0):
        self.user_le = encoders['user_id']
        self.item_le = encoders['item_id']
        self.raw_to_idx = encoders.get('raw_to_idx', None)

        self.num_explicit_negatives = num_explicit_negatives
        # 显式负样本 (1-2星) 映射: user_idx -> [item_idx, ...] (过滤正样本前捕获)
        self.user_explicit_negs = {}
        if num_explicit_negatives > 0 and 'click_label' in click_df.columns:
            neg_df = click_df[click_df['click_label'] == 0].copy()
            neg_df['user_idx'] = self.user_le.transform(neg_df['user_id'])
            neg_df['item_idx'] = self.item_le.transform(neg_df['click_article_id'])
            self.user_explicit_negs = neg_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

        # 只用正样本做对比学习
        click_df = click_df[click_df['click_label'] == 1].copy()
        click_df['user_idx'] = self.user_le.transform(click_df['user_id'])
        click_df['item_idx'] = self.item_le.transform(click_df['click_article_id'])

        self.interactions = click_df[['user_idx', 'item_idx']].values
        self.weights = click_df['click_weight'].values.astype(np.float32)

        self.num_items = num_items
        self.hist_len = hist_len
        self.hard_neg_index = hard_neg_index or {}
        self.num_hard_negatives = num_hard_negatives
        self.num_brands = num_brands

        # --- 预构建数组 (只做一次，__getitem__ 纯数组下标) ---
        if self.raw_to_idx is not None:
            self.idx_to_raw = {v: k for k, v in self.raw_to_idx.items()}
        else:
            self.idx_to_raw = None

        num_users = len(self.user_le.classes_)
        self.user_hist_arr = [None] * num_users
        self.user_hist_brands_arr = [None] * num_users   # Phase 2: 品牌偏好
        self.user_hist_time_arr = [None] * num_users     # Phase 2: 时间衰减
        self.user_click_cnt_arr = np.zeros(num_users, dtype=np.float32)
        self.user_time_span_arr = np.zeros(num_users, dtype=np.float32)
        self.user_avg_rating_arr = np.zeros(num_users, dtype=np.float32)
        self.user_std_rating_arr = np.zeros(num_users, dtype=np.float32)

        # --- 向量化散射: raw user_id -> encoded uid_idx (原 iterrows 83 万行) ---
        if self.raw_to_idx is not None:
            uid_idx = user_features['user_id'].map(self.raw_to_idx)
        else:
            raw_to_idx = {raw: idx for idx, raw in enumerate(self.user_le.classes_)}
            uid_idx = user_features['user_id'].map(raw_to_idx)
        valid = uid_idx.notna()
        uid_arr = uid_idx[valid].astype(np.int64).to_numpy()
        uf_valid = user_features.loc[valid]

        # 标量特征: fancy-indexing 一次性散射 (等价于原逐行赋值)
        # 注: rating 两列原循环用 .get(col, 0) 缺列->0; click/time 用 row[...] 缺列->KeyError, 精确复现
        def _ucol(df, name, dtype):
            return df[name].to_numpy(dtype=dtype) if name in df.columns else np.zeros(len(df), dtype=dtype)

        self.user_click_cnt_arr[uid_arr] = uf_valid['click_count_norm'].to_numpy(dtype=np.float32)
        self.user_time_span_arr[uid_arr] = uf_valid['time_span_norm'].to_numpy(dtype=np.float32)
        self.user_avg_rating_arr[uid_arr] = _ucol(uf_valid, 'user_avg_rating_norm', np.float32)
        self.user_std_rating_arr[uid_arr] = _ucol(uf_valid, 'user_std_rating_norm', np.float32)

        # hist_items_trunc / hist_brands / hist_time_deltas 变长 list (缺列默认 None, __getitem__ 兜底)
        def _ucol_list(df, name):
            return df[name].tolist() if name in df.columns else [None] * len(df)

        brands_list = _ucol_list(uf_valid, 'hist_brands')
        times_list = _ucol_list(uf_valid, 'hist_time_deltas')
        for u, h, b, t in zip(uid_arr, uf_valid['hist_items_trunc'].tolist(), brands_list, times_list):
            self.user_hist_arr[u] = h
            self.user_hist_brands_arr[u] = b
            self.user_hist_time_arr[u] = t

        # --- 向量化散射: item_features.index 即 item_idx (0..N-1) ---
        # 原循环 .loc[idx].get(col, 0) 的等价: 缺列->0, gap index 留 0 → reindex(fill_value=0)
        num_all_items = len(self.item_le.classes_)
        _idx = np.arange(num_all_items)

        def _col(name, dtype):
            if name in item_features.columns:
                return item_features[name].reindex(_idx, fill_value=0).to_numpy(dtype=dtype)
            return np.zeros(num_all_items, dtype=dtype)

        self.item_cat_arr = _col('category_idx', np.int64)
        self.item_brand_arr = _col('brand_idx', np.int64)
        self.item_click_arr = _col('item_click_count_norm', np.float32)
        self.item_created_arr = _col('created_at_ts_norm', np.float32)
        self.item_avg_rating_arr = _col('item_avg_rating_norm', np.float32)
        self.item_rating_num_arr = _col('item_rating_number_norm', np.float32)

    def __len__(self):
        return len(self.interactions)

    def _get_user_data(self, user_idx):
        hist_items = self.user_hist_arr[user_idx]
        if hist_items is None:
            hist_items = []
        hist_brands = self.user_hist_brands_arr[user_idx] or []
        hist_times = self.user_hist_time_arr[user_idx] or []
        hist_len_actual = len(hist_items)

        def _pad(seq, pad_val):
            seq = seq[-self.hist_len:] if len(seq) > self.hist_len else seq
            return seq + [pad_val] * (self.hist_len - len(seq))

        padded = _pad(hist_items, 0)
        padded_brands = _pad(hist_brands, 0)
        padded_times = _pad(hist_times, 0.0)
        hist_len_actual = min(hist_len_actual, self.hist_len)

        return {
            'hist_items': torch.LongTensor(padded),
            'hist_brands': torch.LongTensor(padded_brands),
            'hist_time_deltas': torch.FloatTensor(padded_times),
            'hist_len': torch.tensor(hist_len_actual, dtype=torch.long),
            'click_count': torch.tensor(self.user_click_cnt_arr[user_idx], dtype=torch.float32),
            'time_span': torch.tensor(self.user_time_span_arr[user_idx], dtype=torch.float32),
            'user_avg_rating': torch.tensor(self.user_avg_rating_arr[user_idx], dtype=torch.float32),
            'user_std_rating': torch.tensor(self.user_std_rating_arr[user_idx], dtype=torch.float32),
        }

    def _get_item_data(self, item_idx):
        return {
            'category_id': torch.tensor(self.item_cat_arr[item_idx], dtype=torch.long),
            'brand_id': torch.tensor(self.item_brand_arr[item_idx], dtype=torch.long),
            'item_click_count': torch.tensor(self.item_click_arr[item_idx], dtype=torch.float32),
            'created_at_ts': torch.tensor(self.item_created_arr[item_idx], dtype=torch.float32),
            'item_avg_rating': torch.tensor(self.item_avg_rating_arr[item_idx], dtype=torch.float32),
            'item_rating_number': torch.tensor(self.item_rating_num_arr[item_idx], dtype=torch.float32),
        }

    def __getitem__(self, idx):
        user_idx, pos_item_idx = self.interactions[idx]

        user_data = self._get_user_data(user_idx)
        pos_item_data = self._get_item_data(pos_item_idx)

        batch = {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            'hist_items': user_data['hist_items'],
            'hist_brands': user_data['hist_brands'],
            'hist_time_deltas': user_data['hist_time_deltas'],
            'hist_len': user_data['hist_len'],
            'click_count': user_data['click_count'],
            'time_span': user_data['time_span'],
            'user_avg_rating': user_data['user_avg_rating'],
            'user_std_rating': user_data['user_std_rating'],
            'pos_item_id': torch.tensor(pos_item_idx, dtype=torch.long),
            'pos_category_id': pos_item_data['category_id'],
            'pos_brand_id': pos_item_data['brand_id'],
            'pos_item_click_count': pos_item_data['item_click_count'],
            'pos_created_at_ts': pos_item_data['created_at_ts'],
            'pos_item_avg_rating': pos_item_data['item_avg_rating'],
            'pos_item_rating_number': pos_item_data['item_rating_number'],
            'click_weight': torch.tensor(self.weights[idx], dtype=torch.float32),
        }

        if self.num_hard_negatives > 0:
            if pos_item_idx in self.hard_neg_index:
                hard_negs = self.hard_neg_index[pos_item_idx][:self.num_hard_negatives]
                while len(hard_negs) < self.num_hard_negatives:
                    hard_negs.append(np.random.randint(1, self.num_items))
            else:
                hard_negs = np.random.randint(1, self.num_items, size=self.num_hard_negatives).tolist()

            # 显式负样本 (1-2星) 追加为额外困难负样本 (固定长度, 缺则随机补齐)
            if self.num_explicit_negatives > 0:
                expl = self.user_explicit_negs.get(user_idx, [])
                chosen = []
                if expl:
                    k = min(len(expl), self.num_explicit_negatives)
                    chosen = [int(x) for x in np.random.choice(expl, size=k, replace=False)]
                while len(chosen) < self.num_explicit_negatives:
                    chosen.append(int(np.random.randint(1, self.num_items)))
                hard_negs.extend(chosen)

            hard_neg_data = [self._get_item_data(neg_idx) for neg_idx in hard_negs]
            batch['hard_neg_ids'] = torch.LongTensor(hard_negs)
            batch['hard_neg_category_ids'] = torch.stack([d['category_id'] for d in hard_neg_data])
            batch['hard_neg_brand_ids'] = torch.stack([d['brand_id'] for d in hard_neg_data])
            batch['hard_neg_item_click_count'] = torch.stack([d['item_click_count'] for d in hard_neg_data])
            batch['hard_neg_created_at_ts'] = torch.stack([d['created_at_ts'] for d in hard_neg_data])
            batch['hard_neg_item_avg_rating'] = torch.stack([d['item_avg_rating'] for d in hard_neg_data])
            batch['hard_neg_item_rating_number'] = torch.stack([d['item_rating_number'] for d in hard_neg_data])

        return batch


class DINDataset(Dataset):
    """DIN 精排数据集 (BPR pairwise): 每样本 = (用户, 正样本item, 负样本item)"""
    def __init__(self, click_df, user_features, item_features, encoders,
                 hist_len=50, neg_ratio=4, hard_neg_index=None, num_hard_negatives=4):
        self.user_le = encoders['user_id']
        self.item_le = encoders['item_id']
        self.raw_to_idx = encoders.get('raw_to_idx', None)

        click_df = click_df.copy()
        click_df['user_idx'] = self.user_le.transform(click_df['user_id'])
        click_df['item_idx'] = self.item_le.transform(click_df['click_article_id'])

        pos_df = click_df[click_df['click_label'] == 1]
        neg_df = click_df[click_df['click_label'] == 0]
        user_pos_items = pos_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

        # 先建 user history 字典
        uid_to_hist = {}
        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            if self.raw_to_idx is not None:
                uid_idx = self.raw_to_idx.get(raw_uid)
            elif raw_uid in self.user_le.classes_:
                uid_idx = self.user_le.transform([raw_uid])[0]
            else:
                uid_idx = None
            if uid_idx is not None:
                uid_to_hist[uid_idx] = row['hist_items_trunc']

        # 构建负样本池: 显式负样本 (1-2分) → list of (uid, iid)
        explicit_negs = []
        for _, row in neg_df.iterrows():
            explicit_negs.append((row['user_idx'], row['item_idx']))

        # BPR pairwise 样本: (uid, pos_iid, neg_iid)
        self.pairs = []
        rng = np.random.default_rng(42)

        # 每个正样本配 neg_ratio 个负样本
        for _, row in pos_df.iterrows():
            uid = row['user_idx']
            pos_iid = row['item_idx']
            pos_set = user_pos_items.get(uid, set())
            hist = uid_to_hist.get(uid, [])

            # 收集该用户的所有负样本候选
            neg_candidates = []

            # 1) Hard negatives: ItemCF 相似但未点击
            if hard_neg_index is not None and hist:
                seen_hn = set()
                for h in hist:
                    if h > 0 and h in hard_neg_index:
                        for hn in hard_neg_index[h][:num_hard_negatives]:
                            if hn not in seen_hn and hn not in pos_set:
                                seen_hn.add(hn)
                                neg_candidates.append(hn)

            # 2) 不够补随机
            while len(neg_candidates) < neg_ratio * 3:
                r = rng.integers(1, len(self.item_le.classes_))
                if r not in pos_set:
                    neg_candidates.append(r)

            # 打乱取 neg_ratio 个
            rng.shuffle(neg_candidates)
            for neg_iid in neg_candidates[:neg_ratio]:
                self.pairs.append((uid, pos_iid, int(neg_iid)))

        print(f">>> DINDataset pairwise: {len(self.pairs):,} pairs")
        self.hist_len = hist_len

        # --- 预构建数组 ---
        if self.raw_to_idx is not None:
            self.idx_to_raw = {v: k for k, v in self.raw_to_idx.items()}
        else:
            self.idx_to_raw = None

        num_users = len(self.user_le.classes_)
        self.user_hist_arr = [None] * num_users
        self.user_click_cnt_arr = np.zeros(num_users, dtype=np.float32)
        self.user_time_span_arr = np.zeros(num_users, dtype=np.float32)

        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            if self.raw_to_idx is not None:
                uid_idx = self.raw_to_idx.get(raw_uid)
            elif raw_uid in self.user_le.classes_:
                uid_idx = self.user_le.transform([raw_uid])[0]
            else:
                uid_idx = None
            if uid_idx is not None:
                self.user_hist_arr[uid_idx] = row['hist_items_trunc']
                self.user_click_cnt_arr[uid_idx] = row['click_count_norm']
                self.user_time_span_arr[uid_idx] = row['time_span_norm']

        num_all_items = len(self.item_le.classes_)
        self.item_cat_arr = np.zeros(num_all_items, dtype=np.int64)
        self.item_click_arr = np.zeros(num_all_items, dtype=np.float32)
        self.item_created_arr = np.zeros(num_all_items, dtype=np.float32)
        for idx in item_features.index:
            self.item_cat_arr[idx] = int(item_features.loc[idx].get('category_idx', 0))
            self.item_click_arr[idx] = float(item_features.loc[idx].get('item_click_count_norm', 0))
            self.item_created_arr[idx] = float(item_features.loc[idx].get('created_at_ts_norm', 0))

    def __len__(self):
        return len(self.pairs)

    def _get_user_data(self, user_idx):
        hist_items = self.user_hist_arr[user_idx]
        if hist_items is None:
            hist_items = []
        hist_len_actual = len(hist_items)

        if hist_len_actual < self.hist_len:
            padded = hist_items + [0] * (self.hist_len - hist_len_actual)
        else:
            padded = hist_items[-self.hist_len:]
            hist_len_actual = self.hist_len

        return {
            'hist_items': torch.LongTensor(padded),
            'hist_len': torch.tensor(hist_len_actual, dtype=torch.long),
            'click_count': torch.tensor(self.user_click_cnt_arr[user_idx], dtype=torch.float32),
            'time_span': torch.tensor(self.user_time_span_arr[user_idx], dtype=torch.float32),
        }

    def _get_item_data(self, item_idx):
        return {
            'category_id': torch.tensor(self.item_cat_arr[item_idx], dtype=torch.long),
            'item_click_count': torch.tensor(self.item_click_arr[item_idx], dtype=torch.float32),
            'created_at_ts': torch.tensor(self.item_created_arr[item_idx], dtype=torch.float32),
        }

    def __getitem__(self, idx):
        user_idx, pos_item_idx, neg_item_idx = self.pairs[idx]

        user_data = self._get_user_data(user_idx)
        pos_data = self._get_item_data(pos_item_idx)
        neg_data = self._get_item_data(neg_item_idx)

        return {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            'hist_items': user_data['hist_items'],
            'hist_len': user_data['hist_len'],
            'click_count': user_data['click_count'],
            'time_span': user_data['time_span'],

            'pos_item_id': torch.tensor(pos_item_idx, dtype=torch.long),
            'pos_category_id': pos_data['category_id'],
            'pos_item_click_count': pos_data['item_click_count'],
            'pos_created_at_ts': pos_data['created_at_ts'],

            'neg_item_id': torch.tensor(neg_item_idx, dtype=torch.long),
            'neg_category_id': neg_data['category_id'],
            'neg_item_click_count': neg_data['item_click_count'],
            'neg_created_at_ts': neg_data['created_at_ts'],
        }
