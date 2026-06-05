import numpy as np
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

import config
from data_loader import get_all_click_df, get_user_item_time, get_item_topk_click, load_articles
from itemcf import itemcf_sim, item_based_recommend


def itemcf_recall(target_users, user_item_time_dict, i2i_sim, item_topk_click,
                  sim_item_topk=10, recall_item_num=50):
    print(">>> [Recall Channel 1] ItemCF Recall...")
    user_recall_dict = {}
    for user in tqdm(target_users, desc="ItemCF Recall"):
        topk_items = item_based_recommend(
            user, user_item_time_dict, i2i_sim,
            sim_item_topk=sim_item_topk,
            recall_item_num=recall_item_num,
            item_topk_click=item_topk_click
        )
        user_recall_dict[user] = {item: score for item, score in topk_items}
    return user_recall_dict


def embedding_recall(target_users, user_recall_dict, click_df, user_features,
                     item_features, encoders, all_item_vecs, item_topk_click,
                     hist_len=50, recall_item_num=50, weight=1.0,
                     model_path=None, channel_label="Embedding"):
    """
    双塔向量召回: 用训练好的双塔模型计算用户-物品相似度。
    支持传入 model_path 和 all_item_vecs 实现多模型并行召回。
    """
    import torch
    from model import TwoTowerV2Model

    # 自动检测模型路径
    if model_path is None:
        V2_BEST = os.path.join(config.MODEL_PATH, 'two_tower_v2_best.pth')
        V1_BEST = os.path.join(config.MODEL_PATH, 'two_tower_best.pth')
        for candidate in [(V2_BEST,), (config.V2_MODEL_FILE,), (V1_BEST,), (config.DEEP_MODEL_FILE,)]:
            if os.path.exists(candidate[0]):
                model_path = candidate[0]; break

    if model_path is None or not os.path.exists(model_path):
        print(f"    [{channel_label}] No model file found, skipping.")
        return user_recall_dict

    # 判断 V2/V1
    use_v2 = 'v2' in os.path.basename(model_path).lower()

    print(f"    [{channel_label}] Loading: {os.path.basename(model_path)} (v2={use_v2})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = checkpoint['config']

    if use_v2:
        model = TwoTowerV2Model(
            num_users=model_cfg['num_users'],
            num_items=model_cfg['num_items'],
            num_categories=model_cfg['num_categories'],
            embed_dim=model_cfg['embed_dim'],
            hidden_dims=model_cfg['hidden_dims'],
            hist_len=model_cfg['hist_len'],
            temperature=model_cfg.get('temperature', 0.07),
            num_heads=model_cfg.get('num_heads', 2),
            num_blocks=model_cfg.get('num_blocks', 2),
        ).to(device)
    else:
        from model import TwoTowerModel
        model = TwoTowerModel(
            num_users=model_cfg['num_users'],
            num_items=model_cfg['num_items'],
            num_categories=model_cfg['num_categories'],
            embed_dim=model_cfg['embed_dim'],
            hidden_dims=model_cfg['hidden_dims'],
            hist_len=model_cfg['hist_len'],
            temperature=model_cfg.get('temperature', 0.1),
        ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    user_le = encoders['user_id']
    item_le = encoders['item_id']
    raw_to_idx = encoders.get('raw_to_idx', None)

    all_item_vecs_tensor = torch.FloatTensor(all_item_vecs).to(device)

    # --- 预构建 idx → raw_id 映射 (避免循环里逐条 inverse_transform) ---
    idx_to_raw_item = item_le.classes_.tolist()

    # --- 预构建用户特征字典 (O(1) 查询, 替代扫 DataFrame) ---
    user_feat_dict = {}
    for _, feat_row in user_features.iterrows():
        uid = feat_row['user_id']
        user_feat_dict[uid] = {
            'hist': feat_row['hist_items_trunc'],
            'click_norm': feat_row['click_count_norm'],
            'span_norm': feat_row['time_span_norm'],
        }

    # 预取出所有有效用户 (只做一次, O(N) 但是 N=273K 而不是 N=273K×222K)
    valid_users = []
    for raw_uid in target_users:
        if raw_uid not in user_le.classes_:
            continue
        feat = user_feat_dict.get(raw_uid)
        if feat is None:
            continue
        valid_users.append((raw_uid, feat))

    # 分 batch 向量化推理
    batch_size = 256
    for start in tqdm(range(0, len(valid_users), batch_size), desc="Embedding Recall"):
        chunk = valid_users[start:start + batch_size]

        # 组装 batch 张量
        user_ids, histories, hist_lens, click_counts, time_spans = [], [], [], [], []
        for raw_uid, feat in chunk:
            if raw_to_idx is not None:
                uidx = raw_to_idx.get(raw_uid, 0)
            else:
                uidx = user_le.transform([raw_uid])[0]
            user_ids.append(uidx)

            hist_items = feat['hist']
            hl = len(hist_items)
            if hl > hist_len:
                hist_items = hist_items[-hist_len:]
                hl = hist_len
            padded = hist_items + [0] * (hist_len - hl)
            histories.append(padded)
            hist_lens.append(hl)
            click_counts.append(feat['click_norm'])
            time_spans.append(feat['span_norm'])

        user_batch = {
            'user_id': torch.LongTensor(user_ids).to(device),
            'hist_items': torch.LongTensor(histories).to(device),
            'hist_len': torch.LongTensor(hist_lens).to(device),
            'click_count': torch.FloatTensor(click_counts).to(device),
            'time_span': torch.FloatTensor(time_spans).to(device),
        }

        with torch.no_grad():
            user_vecs = model.get_user_embedding(user_batch)  # (B, D)

        # GPU 上一次性完成: matmul → mask history → topk (省掉 CPU 传输 179MB/批)
        scores = torch.matmul(user_vecs, all_item_vecs_tensor.t())  # (B, num_items)

        # 批量 mask 掉用户历史: 向量化 scatter, 比逐行 for 快
        for i, (raw_uid, feat) in enumerate(chunk):
            hist_set = feat['hist']
            for h in hist_set:
                if 0 <= h < scores.shape[1]:
                    scores[i, h] = -1e9

        # GPU topk (比 CPU argsort 快 10-50 倍)
        top_scores, top_indices = torch.topk(scores, recall_item_num, dim=1)

        top_indices = top_indices.cpu().numpy()
        top_scores = top_scores.cpu().numpy()

        for i, (raw_uid, _) in enumerate(chunk):
            for j in range(recall_item_num):
                idx = int(top_indices[i, j])
                raw_item_id = idx_to_raw_item[idx]  # O(1) list lookup
                sc = float(top_scores[i, j]) * weight
                if raw_uid not in user_recall_dict:
                    user_recall_dict[raw_uid] = {}
                user_recall_dict[raw_uid][raw_item_id] = (
                    user_recall_dict[raw_uid].get(raw_item_id, 0) + sc
                )

    return user_recall_dict


def category_preference_recall(target_users, click_df, articles_df, item_topk_click,
                               recall_item_num=20, weight=0.5):
    """
    品类偏好召回: 根据用户历史点击的品类偏好，推荐该品类下的热门文章
    解决冷启动和兴趣延续问题
    """
    print(">>> [Recall Channel 3] Category Preference Recall...")

    if 'category_id' not in articles_df.columns:
        print("    No category_id in articles, skipping.")
        return {}

    item_category = dict(zip(articles_df['article_id'], articles_df['category_id']))

    category_items = defaultdict(list)
    for item_id, count in click_df['click_article_id'].value_counts().items():
        cat = item_category.get(item_id)
        if cat is not None:
            category_items[cat].append((item_id, count))

    for cat in category_items:
        category_items[cat].sort(key=lambda x: x[1], reverse=True)

    user_category_pref = defaultdict(lambda: defaultdict(float))
    # 向量化: 用 groupby + size 替代 iterrows() 扫 155 万行
    click_df_enc = click_df.copy()
    click_df_enc['cat'] = click_df_enc['click_article_id'].map(item_category)
    cat_counts = click_df_enc.dropna(subset=['cat']).groupby(['user_id', 'cat']).size()
    user_totals = cat_counts.groupby('user_id').sum()  # 预计算每个用户的总数
    for (uid, cat), cnt in tqdm(cat_counts.items(), desc="Building category pref"):
        user_category_pref[uid][cat] = float(cnt / user_totals[uid])

    user_recall_dict = {}
    for user in tqdm(target_users, desc="Category Recall"):
        if user not in user_category_pref:
            continue
        recall_items = {}
        for cat, pref_score in sorted(user_category_pref[user].items(), key=lambda x: -x[1])[:3]:
            for item_id, _ in category_items.get(cat, [])[:recall_item_num]:
                recall_items[item_id] = recall_items.get(item_id, 0) + pref_score * weight
        user_recall_dict[user] = recall_items

    return user_recall_dict


def hot_recall(target_users, item_topk_click, recall_item_num=5, weight=0.1):
    """
    热门召回: 兜底策略，保证每个用户都有推荐结果
    """
    print(">>> [Recall Channel 4] Hot Item Recall (fallback)...")
    user_recall_dict = {}
    hot_items = item_topk_click[:recall_item_num]
    for user in target_users:
        recall_items = {}
        for rank, item_id in enumerate(hot_items):
            recall_items[item_id] = weight * (1.0 - rank / len(hot_items))
        user_recall_dict[user] = recall_items
    return user_recall_dict


def merge_recall_results(recall_channels, weights=None, final_recall_num=50):
    """
    多路召回融合: 每路独立排序取 Top-N，保证每一路都有代表进最终候选池。
    避免单路分数尺度过大淹没其他路。

    策略:
      - 每条路按 weight 分配名额 (至少 3, 保证各路都有代表)
      - 从每路按分配名额取分数最高的商品 → 合并 → 统一去重排序取 final_recall_num
    """
    print(">>> Merging multi-channel recall results...")
    if weights is None:
        weights = [1.0] * len(recall_channels)

    n_channels = len(recall_channels)
    total_weight = sum(w for w in weights if w > 0)

    # 计算每条路的分配名额 (至少 3)
    floor = 3
    quotas = []
    remaining = final_recall_num - floor * n_channels
    remaining = max(remaining, final_recall_num // 4)  # 前几路如果数量少, 至少不至于全挤掉
    for w in weights:
        if w <= 0 or total_weight == 0:
            quotas.append(0)
        else:
            quota = max(floor, int(final_recall_num * w / total_weight))
            quotas.append(quota)

    # 从每路取 Top-quota 个
    merged = defaultdict(dict)
    for channel, weight, quota in zip(recall_channels, weights, quotas):
        if weight <= 0:
            continue
        for user, items in channel.items():
            if isinstance(items, dict):
                # 按分数降序取前 quota 个
                top_items = list(items.items())
                top_items.sort(key=lambda x: -x[1])
                for item_id, score in top_items[:quota]:
                    merged[user][item_id] = merged[user].get(item_id, 0) + score * weight

    result = {}
    for user, items in merged.items():
        result[user] = dict(sorted(items.items(), key=lambda x: -x[1])[:final_recall_num])

    return result


def multi_channel_recall(target_users, click_df, user_item_time_dict, i2i_sim,
                         item_topk_click, articles_df, user_features, item_features,
                         encoders, all_item_vecs, hist_len=50,
                         final_recall_num=50):
    """
    多路召回主函数: 融合 ItemCF + 双塔向量 + 品类偏好 + 热门
    """
    print("=" * 60)
    print("Starting Multi-Channel Recall...")
    print("=" * 60)

    itemcf_recall_dict = itemcf_recall(
        target_users, user_item_time_dict, i2i_sim, item_topk_click,
        sim_item_topk=10, recall_item_num=final_recall_num
    )

    # 双塔 V2 (SASRec + InfoNCE) — 主力
    v2_embed_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_best.pth')
    if not os.path.exists(v2_embed_path):
        v2_embed_path = config.V2_MODEL_FILE
    v2_vecs = all_item_vecs  # 由 inference_full.py 预加载

    v2_recall_dict = embedding_recall(
        target_users, {}, click_df, user_features,
        item_features, encoders, v2_vecs, item_topk_click,
        hist_len=hist_len, recall_item_num=final_recall_num, weight=1.0,
        model_path=v2_embed_path, channel_label="V2 SASRec"
    )

    # 双塔 V1 (BPR) — 辅助
    v1_embed_path = os.path.join(config.MODEL_PATH, 'two_tower_best.pth')
    if not os.path.exists(v1_embed_path):
        v1_embed_path = config.DEEP_MODEL_FILE
    # 加载 V1 embeddings (独立文件)
    v1_vecs = None
    v1_pkl = os.path.join(config.MODEL_PATH, 'item_embeddings.pkl')
    if os.path.exists(v1_pkl):
        with open(v1_pkl, 'rb') as f:
            v1_vecs = pickle.load(f)
        v1_recall_dict = embedding_recall(
            target_users, {}, click_df, user_features,
            item_features, encoders, v1_vecs, item_topk_click,
            hist_len=hist_len, recall_item_num=final_recall_num, weight=1.0,
            model_path=v1_embed_path, channel_label="V1 BPR"
        )
    else:
        print("    [V1 BPR] No item_embeddings.pkl found, skipping V1 recall")
        v1_recall_dict = {}

    category_recall_dict = category_preference_recall(
        target_users, click_df, articles_df, item_topk_click,
        recall_item_num=20, weight=0.5
    )

    hot_recall_dict = hot_recall(
        target_users, item_topk_click, recall_item_num=5, weight=0.1
    )

    channels = [itemcf_recall_dict, v2_recall_dict, v1_recall_dict, category_recall_dict, hot_recall_dict]
    weights = [
        config.RECALL_WEIGHTS.get('itemcf', 1.0),
        config.RECALL_WEIGHTS.get('v2_sasrec', 0.0),
        config.RECALL_WEIGHTS.get('v1_bpr', 0.0),
        config.RECALL_WEIGHTS.get('category', 0.5),
        config.RECALL_WEIGHTS.get('hot', 0.1),
    ]
    print(f"    Weights: ItemCF={weights[0]:.1f}, V2={weights[1]:.1f}, V1={weights[2]:.1f}, Cat={weights[3]:.1f}, Hot={weights[4]:.1f}")

    merged = merge_recall_results(channels, weights, final_recall_num=final_recall_num)

    final_result = {}
    for user in target_users:
        items = merged.get(user, {})
        sorted_items = sorted(items.items(), key=lambda x: -x[1])[:final_recall_num]
        final_result[user] = sorted_items

    print(f">>> Multi-channel recall done. {len(final_result)} users, "
          f"avg {np.mean([len(v) for v in final_result.values()]):.1f} items/user")
    return final_result
