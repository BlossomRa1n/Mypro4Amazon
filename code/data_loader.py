import pandas as pd
import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
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
        if 'rating' in df.columns:
            df = df.drop(columns=['rating'])

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
    category_id ← 品类名称 (如 'Office_Products', 'All_Beauty')
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

    hard_neg_index = {}
    for raw_item, sim_items in i2i_sim.items():
        if raw_item not in item_le.classes_:
            continue
        item_idx = item_le.transform([raw_item])[0]
        sorted_sims = sorted(sim_items.items(), key=lambda x: -x[1])
        hard_negs = []
        for sim_item, sim_score in sorted_sims[:num_hard_negatives * 3]:
            if sim_item in item_le.classes_:
                hard_negs.append(item_le.transform([sim_item])[0])
            if len(hard_negs) >= num_hard_negatives:
                break
        hard_neg_index[item_idx] = hard_negs

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

        self.user_features = user_features
        self.item_features = item_features
        self.num_items = num_items
        self.hist_len = hist_len
        self.num_negatives = num_negatives

        self.user_hist_dict = {}
        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            if self.raw_to_idx is not None:
                uid_idx = self.raw_to_idx.get(raw_uid)
            elif raw_uid in self.user_le.classes_:
                uid_idx = self.user_le.transform([raw_uid])[0]
            else:
                uid_idx = None
            if uid_idx is not None:
                self.user_hist_dict[uid_idx] = row['hist_items_trunc']

        self.item_popularity = np.ones(num_items, dtype=np.float32)
        item_counts = click_df['item_idx'].value_counts()
        for idx, cnt in item_counts.items():
            self.item_popularity[idx] = cnt
        self.item_popularity = self.item_popularity / self.item_popularity.sum()

    def __len__(self):
        return len(self.interactions)

    def _get_user_data(self, user_idx):
        hist_items = self.user_hist_dict.get(user_idx, [])
        hist_len_actual = len(hist_items)

        if hist_len_actual < self.hist_len:
            padded = hist_items + [0] * (self.hist_len - hist_len_actual)
        else:
            padded = hist_items[-self.hist_len:]
            hist_len_actual = self.hist_len

        if self.raw_to_idx is not None:
            idx_to_raw = {v: k for k, v in self.raw_to_idx.items()}
            raw_uid = idx_to_raw.get(user_idx)
        else:
            raw_uid = self.user_le.inverse_transform([user_idx])[0]

        user_row = self.user_features[self.user_features['user_id'] == raw_uid]
        if len(user_row) > 0:
            click_count_norm = user_row.iloc[0]['click_count_norm']
            time_span_norm = user_row.iloc[0]['time_span_norm']
        else:
            click_count_norm = 0.0
            time_span_norm = 0.0

        return {
            'hist_items': torch.LongTensor(padded),
            'hist_len': torch.tensor(hist_len_actual, dtype=torch.long),
            'click_count': torch.tensor(click_count_norm, dtype=torch.float32),
            'time_span': torch.tensor(time_span_norm, dtype=torch.float32),
        }

    def _get_item_data(self, item_idx):
        if item_idx in self.item_features.index:
            row = self.item_features.loc[item_idx]
            return {
                'category_id': torch.tensor(int(row.get('category_idx', 0)), dtype=torch.long),
                'item_click_count': torch.tensor(float(row.get('item_click_count_norm', 0)), dtype=torch.float32),
                'created_at_ts': torch.tensor(float(row.get('created_at_ts_norm', 0)), dtype=torch.float32),
            }
        return {
            'category_id': torch.tensor(0, dtype=torch.long),
            'item_click_count': torch.tensor(0.0, dtype=torch.float32),
            'created_at_ts': torch.tensor(0.0, dtype=torch.float32),
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
    """增强版双塔数据集: InfoNCE (In-batch negatives) + Hard Negative Mining"""
    def __init__(self, click_df, user_features, item_features, encoders, num_items,
                 hist_len=50, hard_neg_index=None, num_hard_negatives=4):
        self.user_le = encoders['user_id']
        self.item_le = encoders['item_id']
        self.raw_to_idx = encoders.get('raw_to_idx', None)

        # 只用正样本做对比学习
        click_df = click_df[click_df['click_label'] == 1].copy()
        click_df['user_idx'] = self.user_le.transform(click_df['user_id'])
        click_df['item_idx'] = self.item_le.transform(click_df['click_article_id'])

        self.interactions = click_df[['user_idx', 'item_idx']].values
        self.weights = click_df['click_weight'].values.astype(np.float32)

        self.user_features = user_features
        self.item_features = item_features
        self.num_items = num_items
        self.hist_len = hist_len
        self.hard_neg_index = hard_neg_index or {}
        self.num_hard_negatives = num_hard_negatives

        self.user_hist_dict = {}
        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            if self.raw_to_idx is not None:
                uid_idx = self.raw_to_idx.get(raw_uid)
            elif raw_uid in self.user_le.classes_:
                uid_idx = self.user_le.transform([raw_uid])[0]
            else:
                uid_idx = None
            if uid_idx is not None:
                self.user_hist_dict[uid_idx] = row['hist_items_trunc']

    def __len__(self):
        return len(self.interactions)

    def _get_user_data(self, user_idx):
        hist_items = self.user_hist_dict.get(user_idx, [])
        hist_len_actual = len(hist_items)

        if hist_len_actual < self.hist_len:
            padded = hist_items + [0] * (self.hist_len - hist_len_actual)
        else:
            padded = hist_items[-self.hist_len:]
            hist_len_actual = self.hist_len

        if self.raw_to_idx is not None:
            idx_to_raw = {v: k for k, v in self.raw_to_idx.items()}
            raw_uid = idx_to_raw.get(user_idx)
        else:
            raw_uid = self.user_le.inverse_transform([user_idx])[0]

        user_row = self.user_features[self.user_features['user_id'] == raw_uid]
        if len(user_row) > 0:
            click_count_norm = user_row.iloc[0]['click_count_norm']
            time_span_norm = user_row.iloc[0]['time_span_norm']
        else:
            click_count_norm = 0.0
            time_span_norm = 0.0

        return {
            'hist_items': torch.LongTensor(padded),
            'hist_len': torch.tensor(hist_len_actual, dtype=torch.long),
            'click_count': torch.tensor(click_count_norm, dtype=torch.float32),
            'time_span': torch.tensor(time_span_norm, dtype=torch.float32),
        }

    def _get_item_data(self, item_idx):
        if item_idx in self.item_features.index:
            row = self.item_features.loc[item_idx]
            return {
                'category_id': torch.tensor(int(row.get('category_idx', 0)), dtype=torch.long),
                'item_click_count': torch.tensor(float(row.get('item_click_count_norm', 0)), dtype=torch.float32),
                'created_at_ts': torch.tensor(float(row.get('created_at_ts_norm', 0)), dtype=torch.float32),
            }
        return {
            'category_id': torch.tensor(0, dtype=torch.long),
            'item_click_count': torch.tensor(0.0, dtype=torch.float32),
            'created_at_ts': torch.tensor(0.0, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        user_idx, pos_item_idx = self.interactions[idx]

        user_data = self._get_user_data(user_idx)
        pos_item_data = self._get_item_data(pos_item_idx)

        batch = {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            'hist_items': user_data['hist_items'],
            'hist_len': user_data['hist_len'],
            'click_count': user_data['click_count'],
            'time_span': user_data['time_span'],
            'pos_item_id': torch.tensor(pos_item_idx, dtype=torch.long),
            'pos_category_id': pos_item_data['category_id'],
            'pos_item_click_count': pos_item_data['item_click_count'],
            'pos_created_at_ts': pos_item_data['created_at_ts'],
            'click_weight': torch.tensor(self.weights[idx], dtype=torch.float32),
        }

        if self.num_hard_negatives > 0:
            if pos_item_idx in self.hard_neg_index:
                hard_negs = self.hard_neg_index[pos_item_idx][:self.num_hard_negatives]
                while len(hard_negs) < self.num_hard_negatives:
                    hard_negs.append(np.random.randint(1, self.num_items))
            else:
                hard_negs = np.random.randint(1, self.num_items, size=self.num_hard_negatives).tolist()
            hard_neg_data = [self._get_item_data(neg_idx) for neg_idx in hard_negs]
            batch['hard_neg_ids'] = torch.LongTensor(hard_negs)
            batch['hard_neg_category_ids'] = torch.stack([d['category_id'] for d in hard_neg_data])
            batch['hard_neg_item_click_count'] = torch.stack([d['item_click_count'] for d in hard_neg_data])
            batch['hard_neg_created_at_ts'] = torch.stack([d['created_at_ts'] for d in hard_neg_data])

        return batch


class DINDataset(Dataset):
    """DIN 精排数据集: 正样本 (4-5分) + 显式负样本 (1-2分) + 随机负样本"""
    def __init__(self, click_df, user_features, item_features, encoders,
                 hist_len=50, neg_ratio=4):
        self.user_le = encoders['user_id']
        self.item_le = encoders['item_id']
        self.raw_to_idx = encoders.get('raw_to_idx', None)

        click_df = click_df.copy()
        click_df['user_idx'] = self.user_le.transform(click_df['user_id'])
        click_df['item_idx'] = self.item_le.transform(click_df['click_article_id'])

        pos_df = click_df[click_df['click_label'] == 1]
        neg_df = click_df[click_df['click_label'] == 0]
        user_pos_items = pos_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

        self.samples = []  # (user_idx, item_idx, label, weight)
        # 正样本 (4-5分, 按评分强度加权)
        for _, row in pos_df.iterrows():
            self.samples.append((row['user_idx'], row['item_idx'], 1.0, row['click_weight']))
        # 显式负样本 (1-2分)
        for _, row in neg_df.iterrows():
            self.samples.append((row['user_idx'], row['item_idx'], 0.0, row['click_weight']))
        # 随机负样本 (每正样本配 neg_ratio 个)
        neg_count = len(pos_df) * neg_ratio
        rng = np.random.default_rng(42)
        while neg_count > 0:
            rand_uidx = rng.integers(0, len(self.user_le.classes_), size=min(neg_count, 50000))
            rand_iidx = rng.integers(1, len(self.item_le.classes_), size=min(neg_count, 50000))
            for u, i in zip(rand_uidx, rand_iidx):
                pos_set = user_pos_items.get(u, set())
                if i not in pos_set:
                    self.samples.append((u, i, 0.0, 0.5))
                    neg_count -= 1
                    if neg_count <= 0:
                        break

        self.user_features = user_features
        self.item_features = item_features
        self.hist_len = hist_len

        self.user_hist_dict = {}
        for _, row in user_features.iterrows():
            raw_uid = row['user_id']
            if self.raw_to_idx is not None:
                uid_idx = self.raw_to_idx.get(raw_uid)
            elif raw_uid in self.user_le.classes_:
                uid_idx = self.user_le.transform([raw_uid])[0]
            else:
                uid_idx = None
            if uid_idx is not None:
                self.user_hist_dict[uid_idx] = row['hist_items_trunc']

    def __len__(self):
        return len(self.samples)

    def _get_user_data(self, user_idx):
        hist_items = self.user_hist_dict.get(user_idx, [])
        hist_len_actual = len(hist_items)

        if hist_len_actual < self.hist_len:
            padded = hist_items + [0] * (self.hist_len - hist_len_actual)
        else:
            padded = hist_items[-self.hist_len:]
            hist_len_actual = self.hist_len

        if self.raw_to_idx is not None:
            idx_to_raw = {v: k for k, v in self.raw_to_idx.items()}
            raw_uid = idx_to_raw.get(user_idx)
        else:
            raw_uid = self.user_le.inverse_transform([user_idx])[0]

        user_row = self.user_features[self.user_features['user_id'] == raw_uid]
        if len(user_row) > 0:
            click_count_norm = user_row.iloc[0]['click_count_norm']
            time_span_norm = user_row.iloc[0]['time_span_norm']
        else:
            click_count_norm = 0.0
            time_span_norm = 0.0

        return {
            'hist_items': torch.LongTensor(padded),
            'hist_len': torch.tensor(hist_len_actual, dtype=torch.long),
            'click_count': torch.tensor(click_count_norm, dtype=torch.float32),
            'time_span': torch.tensor(time_span_norm, dtype=torch.float32),
        }

    def _get_item_data(self, item_idx):
        if item_idx in self.item_features.index:
            row = self.item_features.loc[item_idx]
            return {
                'category_id': torch.tensor(int(row.get('category_idx', 0)), dtype=torch.long),
                'item_click_count': torch.tensor(float(row.get('item_click_count_norm', 0)), dtype=torch.float32),
                'created_at_ts': torch.tensor(float(row.get('created_at_ts_norm', 0)), dtype=torch.float32),
            }
        return {
            'category_id': torch.tensor(0, dtype=torch.long),
            'item_click_count': torch.tensor(0.0, dtype=torch.float32),
            'created_at_ts': torch.tensor(0.0, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        user_idx, item_idx, label, weight = self.samples[idx]

        user_data = self._get_user_data(user_idx)
        item_data = self._get_item_data(item_idx)

        return {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            'hist_items': user_data['hist_items'],
            'hist_len': user_data['hist_len'],
            'click_count': user_data['click_count'],
            'time_span': user_data['time_span'],
            'item_id': torch.tensor(item_idx, dtype=torch.long),
            'category_id': item_data['category_id'],
            'item_click_count': item_data['item_click_count'],
            'created_at_ts': item_data['created_at_ts'],
            'label': torch.tensor(label, dtype=torch.float32),
            'click_weight': torch.tensor(weight, dtype=torch.float32),
        }
