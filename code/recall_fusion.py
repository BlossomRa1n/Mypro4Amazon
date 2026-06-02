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
                     hist_len=50, recall_item_num=50, weight=1.0):
    """
    双塔向量召回: 用训练好的双塔模型计算用户-物品相似度
    """
    import torch
    from model import TwoTowerV2Model

    print(">>> [Recall Channel 2] Two-Tower Embedding Recall...")

    # 优先加载 V2 best → V2 → V1 best → V1
    V2_BEST = os.path.join(config.MODEL_PATH, 'two_tower_v2_best.pth')
    V1_BEST = os.path.join(config.MODEL_PATH, 'two_tower_best.pth')

    model_path = None; use_v2 = True
    for candidate, is_v2 in [(V2_BEST, True), (config.V2_MODEL_FILE, True),
                              (V1_BEST, False), (config.DEEP_MODEL_FILE, False)]:
        if os.path.exists(candidate):
            model_path = candidate; use_v2 = is_v2; break

    if model_path is None:
        print("    No model file found, skipping embedding recall.")
        return user_recall_dict

    print(f"    Loading: {model_path} (v2={use_v2})")

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

    # 预取出所有有效用户的数据（避免循环中逐行查 DataFrame）
    valid_users = []
    for raw_uid in target_users:
        if raw_uid not in user_le.classes_:
            continue
        row = user_features[user_features['user_id'] == raw_uid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        valid_users.append((raw_uid, row))

    # 分 batch 向量化推理，比逐用户循环快 ~10 倍
    batch_size = 256
    for start in tqdm(range(0, len(valid_users), batch_size), desc="Embedding Recall"):
        chunk = valid_users[start:start + batch_size]
        B = len(chunk)

        # 组装 batch 张量
        user_ids, histories, hist_lens, click_counts, time_spans = [], [], [], [], []
        for raw_uid, row in chunk:
            if raw_to_idx is not None:
                uidx = raw_to_idx.get(raw_uid, 0)
            else:
                uidx = user_le.transform([raw_uid])[0]
            user_ids.append(uidx)

            hist_items = row['hist_items_trunc']
            hl = len(hist_items)
            if hl > hist_len:
                hist_items = hist_items[-hist_len:]
                hl = hist_len
            padded = hist_items + [0] * (hist_len - hl)
            histories.append(padded)
            hist_lens.append(hl)
            click_counts.append(row['click_count_norm'])
            time_spans.append(row['time_span_norm'])

        user_batch = {
            'user_id': torch.LongTensor(user_ids).to(device),
            'hist_items': torch.LongTensor(histories).to(device),
            'hist_len': torch.LongTensor(hist_lens).to(device),
            'click_count': torch.FloatTensor(click_counts).to(device),
            'time_span': torch.FloatTensor(time_spans).to(device),
        }

        with torch.no_grad():
            user_vecs = model.get_user_embedding(user_batch)  # (B, D)

        scores = torch.matmul(user_vecs, all_item_vecs_tensor.t())  # (B, num_items)
        scores = scores.cpu().numpy()

        for i, (raw_uid, row) in enumerate(chunk):
            hist_set = set(row['hist_items_trunc'])
            s = scores[i]
            for h in hist_set:
                if 0 <= h < len(s):
                    s[h] = -1e9
            top_indices = np.argsort(s)[::-1][:recall_item_num]
            for idx in top_indices:
                raw_item_id = item_le.inverse_transform([int(idx)])[0]
                sc = float(s[idx]) * weight
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
    for _, row in tqdm(click_df.iterrows(), total=len(click_df), desc="Building category pref"):
        uid = row['user_id']
        item_id = row['click_article_id']
        cat = item_category.get(item_id)
        if cat is not None:
            user_category_pref[uid][cat] += 1.0

    for uid in user_category_pref:
        total = sum(user_category_pref[uid].values())
        for cat in user_category_pref[uid]:
            user_category_pref[uid][cat] /= total

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


def merge_recall_results(recall_channels, weights=None):
    """
    多路召回融合: 加权合并各通道的召回结果
    支持自定义各通道权重
    """
    print(">>> Merging multi-channel recall results...")
    if weights is None:
        weights = [1.0] * len(recall_channels)

    merged = defaultdict(lambda: defaultdict(float))

    for channel, weight in zip(recall_channels, weights):
        for user, items in channel.items():
            for item, score in items.items():
                merged[user][item] += score * weight

    result = {}
    for user, items in merged.items():
        result[user] = dict(sorted(items.items(), key=lambda x: -x[1]))

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

    embedding_recall_dict = embedding_recall(
        target_users, itemcf_recall_dict, click_df, user_features,
        item_features, encoders, all_item_vecs, item_topk_click,
        hist_len=hist_len, recall_item_num=final_recall_num, weight=1.0
    )

    category_recall_dict = category_preference_recall(
        target_users, click_df, articles_df, item_topk_click,
        recall_item_num=20, weight=0.5
    )

    hot_recall_dict = hot_recall(
        target_users, item_topk_click, recall_item_num=5, weight=0.1
    )

    channels = [itemcf_recall_dict, embedding_recall_dict, category_recall_dict, hot_recall_dict]
    weights = [1.0, 1.0, 0.5, 0.1]

    merged = merge_recall_results(channels, weights)

    final_result = {}
    for user in target_users:
        items = merged.get(user, {})
        sorted_items = sorted(items.items(), key=lambda x: -x[1])[:final_recall_num]
        final_result[user] = sorted_items

    print(f">>> Multi-channel recall done. {len(final_result)} users, "
          f"avg {np.mean([len(v) for v in final_result.values()]):.1f} items/user")
    return final_result
