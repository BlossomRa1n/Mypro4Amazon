"""
诊断脚本：逐通道检查召回质量
"""
import pickle, os, sys, numpy as np, pandas as pd
sys.path.insert(0, 'code')
import config
from data_loader import get_all_click_df, load_articles, build_encoders, build_user_features, get_user_item_time, get_item_topk_click

print("=" * 60)
print("Pipeline Diagnostic")
print("=" * 60)

# 1. 数据加载
click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
articles_df = load_articles(config.DATA_PATH)
encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)

# 2. 时序分割测试集（留一法）
click_df = click_df.sort_values('click_timestamp')
test_df = click_df.groupby('user_id').tail(1)
test_set = dict(zip(test_df['user_id'], test_df['click_article_id']))
print(f"\nUsers: {click_df['user_id'].nunique()}, Items: {click_df['click_article_id'].nunique()}")
print(f"Test users: {len(test_set)}")

# 3. 随机基线
print("\n--- Baseline: Random ---")
for k in [5, 20, 50]:
    total, hits = 0, 0
    for uid in test_set:
        rand_items = np.random.choice(len(encoders['item_id'].classes_), k, replace=False)
        raw_items = encoders['item_id'].inverse_transform(rand_items)
        if test_set[uid] in raw_items:
            hits += 1
        total += 1
    print(f"  K={k}: HR={hits/total:.4f} ({hits/total*100:.2f}%)")

# 4. 热门基线
print("\n--- Baseline: Global Popularity ---")
pop_items = click_df['click_article_id'].value_counts().index.tolist()
for k in [5, 20, 50]:
    hits = sum(1 for uid, test_item in test_set.items() if test_item in pop_items[:k])
    print(f"  K={k}: HR={hits/len(test_set):.4f} ({hits/len(test_set)*100:.2f}%)")

# 5. 模型文件状态
print("\n--- Model Files ---")
for path, label in [
    (config.V2_MODEL_FILE, "V2 model"), (config.DEEP_MODEL_FILE, "V1 model"),
    (config.DIN_MODEL_FILE, "DIN model"), (config.ITEMCF_SIM_PKL, "ItemCF sim"),
    (config.V2_EMBED_PKL, "V2 embeddings"), (config.EMBED_PKL, "V1 embeddings")
]:
    ok = os.path.exists(path)
    size = os.path.getsize(path) / 1024**2 if ok else 0
    print(f"  {'[OK]' if ok else '[MISS]'} {label:20s} {size:8.1f} MB  {path}")

# 6. 检查 embedding 和 encoder 的 item 数量是否一致
print("\n--- Consistency Checks ---")
if os.path.exists(config.V2_EMBED_PKL):
    emb = pickle.load(open(config.V2_EMBED_PKL, 'rb'))
    print(f"  Embeddings shape: {emb.shape}, Encoder items: {len(encoders['item_id'].classes_)}")
    if emb.shape[0] != len(encoders['item_id'].classes_):
        print("  [WARN] Shape mismatch!")

import torch
if os.path.exists(config.V2_MODEL_FILE):
    ckpt = torch.load(config.V2_MODEL_FILE, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    print(f"  V2 model: num_users={cfg['num_users']}, num_items={cfg['num_items']}")
    print(f"  Encoder:  num_users={len(encoders['user_id'].classes_)}, num_items={len(encoders['item_id'].classes_)}")

# 7. 抽样测试：对 5 个用户手动跑嵌入召回
print("\n--- Sample Embedding Recall Test ---")
if os.path.exists(config.V2_MODEL_FILE) and os.path.exists(config.V2_EMBED_PKL):
    from model import TwoTowerV2Model
    ckpt = torch.load(config.V2_MODEL_FILE, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = TwoTowerV2Model(
        num_users=cfg['num_users'], num_items=cfg['num_items'],
        num_categories=cfg['num_categories'], embed_dim=cfg['embed_dim'],
        hidden_dims=cfg['hidden_dims'], hist_len=cfg['hist_len'],
        temperature=cfg.get('temperature', 0.07),
        num_heads=cfg.get('num_heads', 2), num_blocks=cfg.get('num_blocks', 2))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    emb = pickle.load(open(config.V2_EMBED_PKL, 'rb'))
    emb_tensor = torch.FloatTensor(emb)

    user_features = build_user_features(click_df, encoders, hist_len=config.HIST_LEN)
    user_le, item_le = encoders['user_id'], encoders['item_id']

    sample_users = list(test_set.keys())[:5]
    for raw_uid in sample_users:
        if raw_uid not in user_le.classes_:
            print(f"  User {raw_uid}: NOT in encoder!")
            continue
        user_idx = user_le.transform([raw_uid])[0]
        row = user_features[user_features['user_id'] == raw_uid]
        if len(row) == 0:
            print(f"  User {raw_uid}: no features!")
            continue
        row = row.iloc[0]
        hist = row['hist_items_trunc']
        hist_len_val = min(len(hist), config.HIST_LEN)
        padded = hist[-config.HIST_LEN:] if len(hist) > config.HIST_LEN else hist + [0] * (config.HIST_LEN - len(hist))

        batch = {
            'user_id': torch.LongTensor([user_idx]),
            'hist_items': torch.LongTensor([padded]),
            'hist_len': torch.LongTensor([hist_len_val]),
            'click_count': torch.FloatTensor([row['click_count_norm']]),
            'time_span': torch.FloatTensor([row['time_span_norm']]),
        }
        with torch.no_grad():
            user_vec = model.get_user_embedding(batch)
            scores = torch.matmul(user_vec, emb_tensor.t()).squeeze(0).numpy()

        # 过滤看过的电影
        for h in hist:
            if 0 <= h < len(scores):
                scores[h] = -1e9
        top5 = np.argsort(scores)[::-1][:5]
        top5_raw = [item_le.inverse_transform([int(i)])[0] for i in top5]
        top5_scores = [float(scores[int(i)]) for i in top5]
        test_item = test_set[raw_uid]
        print(f"  User {raw_uid}: top5={top5_raw}, test={test_item}, hit={test_item in top5_raw}")
        print(f"    Scores (top3): {top5_scores[:3]}, (min/max all): {scores[scores > -1e8].min():.4f}/{scores[scores > -1e8].max():.4f}")

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)
