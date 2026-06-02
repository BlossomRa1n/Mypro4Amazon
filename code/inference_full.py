"""
完整推理流程: 多路召回 + DIN 精排 (batch 推理) + 提交生成

DIN 推理优化: 每用户所有候选打包成单 batch，N 次 forward → 1 次 forward
"""
import os, pickle, time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from data_loader import (
    get_all_click_df, get_positive_click_df, load_articles,
    build_encoders, build_user_features, build_enhanced_user_features,
    build_item_features, get_user_item_time, get_item_topk_click
)
from model import TwoTowerV2Model, DINModel
from recall_fusion import multi_channel_recall
from submit import save_submission


def load_model_with_fallback(device):
    """按优先级尝试加载模型: best → default → None"""
    candidates = [
        (config.DIN_BEST_FILE,   "DIN best"),
        (config.DIN_MODEL_FILE,  "DIN default"),
    ]
    for path, label in candidates:
        if os.path.exists(path):
            print(f"    Loading DIN from: {path}")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            cfg = ckpt['config']
            model = DINModel(
                num_users=cfg['num_users'], num_items=cfg['num_items'],
                num_categories=cfg['num_categories'], embed_dim=cfg['embed_dim'],
                hidden_dims=cfg['hidden_dims'], hist_len=cfg['hist_len'],
                dropout=cfg['dropout'],
            ).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f"    Loaded epoch {ckpt.get('epoch', '?')}, metrics: {ckpt.get('metrics', {})}")
            return model
    print("    No DIN model found, using recall scores only")
    return None


def din_rerank_batch(user_batch, item_indices, item_features_df, din_model, device):
    """
    对单个用户的全部候选商品做批量 DIN 推理。
    user_batch: 用户侧特征 dict (不含 item_id)
    item_indices: [N] 候选物品的内部 idx
    返回: [N] 点击概率 numpy array
    """
    N = len(item_indices)
    # 用户特征复制 N 份
    batch = {
        'user_id': user_batch['user_id'].repeat(N),
        'hist_items': user_batch['hist_items'].repeat(N, 1),
        'hist_len': user_batch['hist_len'].repeat(N),
        'click_count': user_batch['click_count'].repeat(N),
        'time_span': user_batch['time_span'].repeat(N),
        'item_id': torch.LongTensor(item_indices).to(device),
    }

    # 物品特征: 逐个取
    cat_ids, click_cnts, created_ats = [], [], []
    for idx in item_indices:
        if idx in item_features_df.index:
            row = item_features_df.loc[idx]
            cat_ids.append(int(row.get('category_idx', 0)))
            click_cnts.append(float(row.get('item_click_count_norm', 0)))
            created_ats.append(float(row.get('created_at_ts_norm', 0)))
        else:
            cat_ids.append(0); click_cnts.append(0.0); created_ats.append(0.0)

    batch['category_id'] = torch.LongTensor(cat_ids).to(device)
    batch['item_click_count'] = torch.FloatTensor(click_cnts).to(device)
    batch['created_at_ts'] = torch.FloatTensor(created_ats).to(device)

    with torch.no_grad():
        logits = din_model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def _build_user_batch(user_idx, row, hist_len, device):
    """构造单用户的基础 batch (不含 item 信息)"""
    hist = row['hist_items_trunc']
    hl = len(hist)
    if hl > hist_len:
        hist = hist[-hist_len:]; hl = hist_len
    padded = hist + [0] * (hist_len - hl)

    return {
        'user_id': torch.LongTensor([user_idx]).to(device),
        'hist_items': torch.LongTensor([padded]).to(device),
        'hist_len': torch.LongTensor([hl]).to(device),
        'click_count': torch.FloatTensor([row['click_count_norm']]).to(device),
        'time_span': torch.FloatTensor([row['time_span_norm']]).to(device),
    }, hl


def inference():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ================================================================
    # Step 1: 加载数据
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 1: Loading data...")
    print("=" * 60)
    full_click = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    click_df = get_positive_click_df(full_click)
    articles_df = load_articles(config.DATA_PATH)
    encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)
    user_features = build_enhanced_user_features(click_df, articles_df, encoders, hist_len=config.HIST_LEN)
    item_features = build_item_features(articles_df, click_df, encoders)

    user_le = encoders['user_id']
    item_le = encoders['item_id']
    num_users = len(user_le.classes_)
    num_items = len(item_le.classes_)
    num_categories = len(encoders.get('category_id', __import__('sklearn').preprocessing.LabelEncoder()).classes_) if 'category_id' in encoders else 1

    target_users = click_df['user_id'].unique()
    print(f">>> Target users: {len(target_users):,} | Items: {num_items:,} | Categories: {num_categories}")

    # ================================================================
    # Step 2: ItemCF
    # ================================================================
    print("\nStep 2: Building ItemCF similarity...")
    user_item_time_dict = get_user_item_time(click_df)
    item_topk_click = get_item_topk_click(click_df, k=50)

    if os.path.exists(config.ITEMCF_SIM_PKL):
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        print(f">>> Loaded cached ItemCF similarity")
    else:
        from itemcf import itemcf_sim
        i2i_sim = itemcf_sim(user_item_time_dict)
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        with open(config.ITEMCF_SIM_PKL, 'wb') as f:
            pickle.dump(i2i_sim, f)
        print(f">>> Computed and cached ItemCF similarity")

    # ================================================================
    # Step 3: Load embeddings
    # ================================================================
    print("\nStep 3: Loading item embeddings...")
    all_item_vecs = None
    for path, label in [(config.V2_EMBED_PKL, "V2"), (config.EMBED_PKL, "V1")]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                all_item_vecs = pickle.load(f)
            print(f">>> Using {label} embeddings, shape: {all_item_vecs.shape}")
            break
    if all_item_vecs is None:
        print(">>> No embeddings found, skipping embedding recall")

    # ================================================================
    # Step 4: Multi-Channel Recall
    # ================================================================
    print("\nStep 4: Multi-Channel Recall...")
    if all_item_vecs is not None:
        recall_results = multi_channel_recall(
            target_users, click_df, user_item_time_dict, i2i_sim,
            item_topk_click, articles_df, user_features, item_features,
            encoders, all_item_vecs, hist_len=config.HIST_LEN,
            final_recall_num=config.RECALL_NUM
        )
    else:
        print(">>> No embeddings, using ItemCF only...")
        recall_results = {}
        for user in tqdm(target_users, desc="ItemCF Only"):
            from itemcf import item_based_recommend
            topk = item_based_recommend(
                user, user_item_time_dict, i2i_sim,
                sim_item_topk=10, recall_item_num=config.RECALL_NUM,
                item_topk_click=item_topk_click
            )
            recall_results[user] = topk

    # ================================================================
    # Step 5: Load DIN model
    # ================================================================
    print("\nStep 5: Loading DIN model...")
    din_model = load_model_with_fallback(device)

    # ================================================================
    # Step 6: DIN Batch Reranking + Final Ranking
    # ================================================================
    print("\nStep 6: Final ranking (DIN batch inference)...")
    user_recall_items_dict = {}
    hist_len = config.HIST_LEN

    for raw_user_id in tqdm(target_users, desc="DIN Rerank"):
        recalled_items = recall_results.get(raw_user_id, [])
        if isinstance(recalled_items, dict):
            recalled_items = list(recalled_items.items())

        # ---- 无召回或空候选: 热门兜底 ----
        if len(recalled_items) == 0:
            topk = item_topk_click[:config.FINAL_RECOMMEND_NUM]
            user_recall_items_dict[raw_user_id] = [(iid, -999.0) for iid in topk]
            continue

        # ---- 无 DIN 或用户不在编码器中: 召回分排序 ----
        if din_model is None or raw_user_id not in user_le.classes_:
            sorted_items = sorted(recalled_items, key=lambda x: -x[1])[:config.FINAL_RECOMMEND_NUM]
            user_recall_items_dict[raw_user_id] = sorted_items
            continue

        # ---- 有 DIN: 准备 batch 推理 ----
        user_idx = user_le.transform([raw_user_id])[0]
        user_row = user_features[user_features['user_id'] == raw_user_id]
        if len(user_row) == 0:
            sorted_items = sorted(recalled_items, key=lambda x: -x[1])[:config.FINAL_RECOMMEND_NUM]
            user_recall_items_dict[raw_user_id] = sorted_items
            continue

        row = user_row.iloc[0]
        user_batch, hl = _build_user_batch(user_idx, row, hist_len, device)

        # 分离有效候选 (DIN 认识) 和无效候选 (只能保留召回分)
        valid_candidates = []  # [(recall_score, item_idx)]
        fallback_scores = []   # [(item_idx, recall_score)] — DIN 不认识
        for item_id, recall_score in recalled_items:
            if item_id not in item_le.classes_:
                fallback_scores.append((item_id, recall_score))
                continue
            item_idx = item_le.transform([item_id])[0]
            if item_idx not in item_features.index:
                fallback_scores.append((item_id, recall_score))
                continue
            valid_candidates.append((recall_score, item_idx, item_id))

        # Batch inference on all valid candidates at once
        if valid_candidates:
            recall_scores = np.array([c[0] for c in valid_candidates])
            item_indices = [c[1] for c in valid_candidates]
            item_ids = [c[2] for c in valid_candidates]

            probs = din_rerank_batch(user_batch, item_indices, item_features, din_model, device)

            alpha = 0.7
            final_scores = alpha * probs + (1 - alpha) * recall_scores / (np.abs(recall_scores) + 1.0)
            din_scores = list(zip(item_ids, final_scores))
        else:
            din_scores = []

        # 合并 fallback
        if fallback_scores:
            din_scores.extend([(iid, rs) for iid, rs in fallback_scores])

        din_scores.sort(key=lambda x: -x[1])
        user_recall_items_dict[raw_user_id] = din_scores[:config.FINAL_RECOMMEND_NUM]

        # ---- 补齐 ----
        if len(user_recall_items_dict[raw_user_id]) < config.FINAL_RECOMMEND_NUM:
            existing = set(x[0] for x in user_recall_items_dict[raw_user_id])
            for iid in item_topk_click:
                if iid not in existing:
                    user_recall_items_dict.setdefault(raw_user_id, []).append((iid, -999.0))
                    existing.add(iid)
                if len(user_recall_items_dict[raw_user_id]) >= config.FINAL_RECOMMEND_NUM:
                    break

    # ================================================================
    # Step 7: Save
    # ================================================================
    print("\nStep 7: Saving submission...")
    save_submission(user_recall_items_dict, config.RESULT_PATH, file_name='result_full_pipeline.csv')
    print(f">>> Full pipeline complete! Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    inference()
