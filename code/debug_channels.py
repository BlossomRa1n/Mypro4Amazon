"""分别测试每条召回通道的质量"""
import sys, pickle, os, numpy as np, pandas as pd
sys.path.insert(0, 'code')
import config
from data_loader import get_all_click_df, load_articles, build_encoders, build_user_features, get_user_item_time, get_item_topk_click
from recall_fusion import itemcf_recall, category_preference_recall, hot_recall
import torch
from model import TwoTowerV2Model

click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
articles_df = load_articles(config.DATA_PATH)
encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)

# 时序分割
click_df = click_df.sort_values('click_timestamp')
split = {}
for _, grp in click_df.groupby('user_id'):
    grp = grp.sort_values('click_timestamp')
    n = max(1, int(len(grp) * 0.8))
    train = grp.iloc[:n]
    test = grp.iloc[n:]
    if len(test) > 0:
        split[grp['user_id'].iloc[0]] = (train, test)

print(f"Valid users: {len(split)}")

# 构建训练用的 click_df（不含测试交互）
train_rows = [v[0] for v in split.values()]
train_click = pd.concat(train_rows)
test_sets = {uid: set(v[1]['click_article_id']) for uid, v in split.items()}

user_item_time_dict = get_user_item_time(train_click)
item_topk_click = get_item_topk_click(train_click, k=50)
with open(config.ITEMCF_SIM_PKL, 'rb') as f:
    i2i_sim = pickle.load(f)

# 逐通道测试
def hr_at_k(recall_dict, test_sets, ks=[5, 20, 50]):
    results = {}
    for k in ks:
        hits, total = 0, 0
        for uid, gt in test_sets.items():
            if uid in recall_dict:
                recs = list(recall_dict[uid].keys()) if isinstance(recall_dict[uid], dict) else [x[0] for x in recall_dict[uid]]
                if set(recs[:k]) & gt:
                    hits += 1
            total += 1
        results[k] = hits / total if total else 0
    return results

target = list(test_sets.keys())

print("\n--- Channel 1: ItemCF ---")
r1 = itemcf_recall(target, user_item_time_dict, i2i_sim, item_topk_click, sim_item_topk=10, recall_item_num=50)
for k, v in hr_at_k(r1, test_sets).items():
    print(f"  K={k:2d}: HR={v:.4f} ({v*100:.2f}%)")

print("\n--- Channel 3: Category Preference ---")
r3 = category_preference_recall(target, train_click, articles_df, item_topk_click, recall_item_num=20, weight=0.5)
for k, v in hr_at_k(r3, test_sets).items():
    print(f"  K={k:2d}: HR={v:.4f} ({v*100:.2f}%)")

print("\n--- Channel 4: Hot (fallback) ---")
r4 = hot_recall(target, item_topk_click, recall_item_num=5, weight=0.1)
for k, v in hr_at_k(r4, test_sets).items():
    print(f"  K={k:2d}: HR={v:.4f} ({v*100:.2f}%)")

print("\n--- Channel 2: Embedding (V2) ---")
if os.path.exists(config.V2_MODEL_FILE) and os.path.exists(config.V2_EMBED_PKL):
    ckpt = torch.load(config.V2_MODEL_FILE, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = TwoTowerV2Model(
        num_users=cfg['num_users'], num_items=cfg['num_items'], num_categories=cfg['num_categories'],
        embed_dim=cfg['embed_dim'], hidden_dims=cfg['hidden_dims'], hist_len=cfg['hist_len'],
        temperature=cfg.get('temperature', 0.07), num_heads=cfg.get('num_heads', 2),
        num_blocks=cfg.get('num_blocks', 2)).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    emb = pickle.load(open(config.V2_EMBED_PKL, 'rb'))
    emb_t = torch.FloatTensor(emb)

    # Check if embeddings are garbage
    print(f"  Embedding stats: mean={emb.mean():.4f}, std={emb.std():.4f}, has_nan={np.isnan(emb).any()}")
    if np.isnan(emb).any():
        print("  [FAIL] Embeddings contain NaN! This explains the bad results.")
    else:
        # Quick batch recall test
        from tqdm import tqdm
        user_features = build_user_features(train_click, encoders, hist_len=config.HIST_LEN)
        user_le, item_le = encoders['user_id'], encoders['item_id']
        r2 = {}
        hits_any = 0
        for raw_uid in tqdm(target[:200], desc="V2 Recall (sample 200)"):
            if raw_uid not in user_le.classes_:
                continue
            row = user_features[user_features['user_id'] == raw_uid]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            hist = row['hist_items_trunc']
            hl = min(len(hist), config.HIST_LEN)
            padded = hist[-config.HIST_LEN:] + [0] * max(0, config.HIST_LEN - hl)
            uidx = user_le.transform([raw_uid])[0]
            batch = {
                'user_id': torch.LongTensor([uidx]),
                'hist_items': torch.LongTensor([padded]),
                'hist_len': torch.LongTensor([hl]),
                'click_count': torch.FloatTensor([row['click_count_norm']]),
                'time_span': torch.FloatTensor([row['time_span_norm']]),
            }
            with torch.no_grad():
                uv = model.get_user_embedding(batch)
                scores = torch.matmul(uv, emb_t.t()).squeeze(0).numpy()
            for h in hist:
                if 0 <= h < len(scores):
                    scores[h] = -1e9
            top50 = np.argsort(scores)[::-1][:50]
            r2[raw_uid] = {item_le.inverse_transform([int(i)])[0]: float(scores[i]) for i in top50}
            if test_sets[raw_uid] & set(r2[raw_uid].keys()):
                hits_any += 1
        print(f"  V2 sample HR@50 (200 users): {hits_any/200:.4f}")
else:
    print("  V2 model or embeddings missing")

print("\n--- Merged (ItemCF + Category + Hot, no embedding) ---")
from recall_fusion import merge_recall_results
merged = merge_recall_results([r1, r3, r4], [1.0, 0.5, 0.1])
for k, v in hr_at_k(merged, test_sets).items():
    print(f"  K={k:2d}: HR={v:.4f} ({v*100:.2f}%)")

print("\nDone.")
