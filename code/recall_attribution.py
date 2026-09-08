"""
P0 召回失败归因 — 判定低召回是"可达性(数据稀疏)"还是"排序(模型质量)"问题。

背景: 召回天花板 HR@100 只有 10%, 最终 HR@5 0.35%。两个互斥病因:
  A. 可达性 — 验证正样本在 ItemCF 图里根本走不到 (历史 item 与正样本无共现边)
     → 数据稀疏, 模型帮不上, 只能靠品类/内容/热度泛化
  B. 排序质量 — 正样本可达, 但 ItemCF / 向量召回把它排在 top-100 之外
     → 修 ItemCF 参数 (sim_item_topk) 或模型训练 (过拟合/坍缩)

两大部分:
  Part 1 (全量 val, 纯 ItemCF 图结构, 无模型, 秒级~分钟级):
    对每个 (用户, 验证正样本) 分类:
      - 已在历史:    正样本在用户训练历史里 (重复购买, ItemCF 显式排除)
      - 可达(top10): 正样本是某历史 item 的 top-10 相似邻居 (ItemCF 能走到)
      - 可达(超top10): 有共现边, 但被 sim_item_topk=10 截断 (扩 topk 可救)
      - 不可达:      历史 item 与正样本完全无共现边 (数据稀疏)
  Part 2 (采样 SAMPLE_USERS 用户, 复用 recall_fusion 真实四路召回):
    各路独立 HR@native-size + 并集 HR + 各路独有命中, 定位主力/拖后腿通道。

判读:
  - 不可达 + 超top10 占比高 → 病因 A (数据稀疏), 优先扩 sim_item_topk / 品类热度泛化
  - 可达(top10) 占比高但 ItemCF/V2 HR 低 → 病因 B (排序), 修模型训练
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle, time
import numpy as np
import torch
from tqdm import tqdm

import config
from data_loader import (get_all_click_df, load_articles,
                         get_user_item_time, get_item_topk_click)
from data_loader_ext import (load_raw_meta, build_extended_encoders,
                             build_extended_item_features,
                             build_extended_user_features, merge_jsonl_features)
from evaluate import split_train_val
from recall_fusion import (itemcf_recall, embedding_recall,
                           category_preference_recall, hot_recall)

RUN_CHANNEL_BREAKDOWN = True   # Part 2 开关
SAMPLE_USERS = 100000          # Part 2 采样用户数


def part1_reachability(val_gt, user_item_time_dict, i2i_sim, sim_item_topk=10):
    """全量 val 可达性归因 (纯图结构, 无模型)。"""
    print("\n" + "=" * 60)
    print("Part 1: 可达性归因 (ItemCF 图结构)")
    print("=" * 60)

    print(">>> Precomputing top-10 neighbors per item (一次, O(1) 查询)...")
    item_top10 = {}
    for item, nbrs in tqdm(i2i_sim.items(), desc="top-10 neighbors"):
        top10 = set()
        for j, _ in sorted(nbrs.items(), key=lambda x: -x[1])[:sim_item_topk]:
            top10.add(j)
        item_top10[item] = top10

    stats = {'in_history': 0, 'reach_top10': 0, 'reach_beyond': 0, 'unreachable': 0}
    n_pairs = 0
    n_users_with_hist = 0
    n_users_any_reach = 0
    hist_len_dist = {}  # 历史长度 → (用户数, 可达用户数)

    for user, gt_items in tqdm(val_gt.items(), desc="Reachability scan"):
        hist = user_item_time_dict.get(user, [])
        hist_items = [it for it, _ in hist]
        hist_set = set(hist_items)
        if not hist_items:
            continue
        n_users_with_hist += 1
        hl = len(hist_items)
        any_reach = False
        for gt in gt_items:
            n_pairs += 1
            if gt in hist_set:
                stats['in_history'] += 1
                continue
            in_top10 = False
            has_edge = False
            for h in hist_items:
                if gt in item_top10.get(h, ()):
                    in_top10 = True
                    break
                if gt in i2i_sim.get(h, {}):
                    has_edge = True
            if in_top10:
                stats['reach_top10'] += 1
                any_reach = True
            elif has_edge:
                stats['reach_beyond'] += 1
                any_reach = True
            else:
                stats['unreachable'] += 1
        if any_reach:
            n_users_any_reach += 1
        hist_len_dist[hl] = hist_len_dist.get(hl, [0, 0])
        hist_len_dist[hl][0] += 1
        if any_reach:
            hist_len_dist[hl][1] += 1

    print(f"\n>>> 验证 (user, gt) 样本对总数: {n_pairs:,}")
    labels = {'in_history':   '已在历史(重复购买)',
              'reach_top10':  '可达 (top-10 内)',
              'reach_beyond': '可达 (超出 top-10)',
              'unreachable':  '不可达 (无共现边)'}
    for k in ['in_history', 'reach_top10', 'reach_beyond', 'unreachable']:
        v = stats[k]
        print(f"    {labels[k]:26s}: {v:8,} ({v/max(n_pairs,1)*100:5.1f}%)")

    print(f"\n>>> 用户级可达率: {n_users_any_reach:,}/{n_users_with_hist:,} "
          f"({n_users_any_reach/max(n_users_with_hist,1)*100:.1f}%) 至少一个正样本可达(top10)")

    print("\n>>> 历史长度 → 用户可达率 (看稀疏度如何影响可达):")
    for hl in sorted(hist_len_dist):
        u, r = hist_len_dist[hl]
        print(f"    hist_len={hl:3d}: {r:6,}/{u:6,} ({r/max(u,1)*100:5.1f}%)")

    reachable = (stats['reach_top10'] + stats['reach_beyond']) / max(n_pairs, 1)
    print("\n>>> 判读:")
    print(f"    数据稀疏侧 (不可达 + 超top10) = {(1 - reachable)*100:.1f}%")
    print(f"    ItemCF 图可走到 (top10)        = {stats['reach_top10']/max(n_pairs,1)*100:.1f}%")
    return stats


def part2_channels(val_gt, sample_users, user_item_time_dict, i2i_sim,
                   item_topk_click, click_df, train_click, raw_meta):
    """采样跑真实四路召回, 各路独立命中 + 并集 + 独有命中。"""
    print("\n" + "=" * 60)
    print(f"Part 2: 通道命中拆解 (采样 {len(sample_users):,} 用户)")
    print("=" * 60)

    print(">>> Building encoders & features (与 inference 一致)...")
    encoders = build_extended_encoders(click_df, raw_meta, config.EXT_ENCODER_PKL)
    user_features = build_extended_user_features(
        train_click, raw_meta, encoders, hist_len=config.HIST_LEN)
    # 归一化 patch (embedding_recall 需要 user_avg/std_rating_norm)
    if 'hist_items_trunc' not in user_features.columns:
        user_features['hist_items_trunc'] = user_features['hist_items']
        user_features['user_avg_rating_norm'] = user_features['user_avg_rating'] / 5.0
        s = user_features['user_std_rating'].astype(float)
        user_features['user_std_rating_norm'] = (s - s.min()) / (s.max() - s.min() + 1e-8)
    if 'user_avg_rating_norm' not in user_features.columns:
        user_rating_stats = train_click.groupby('user_id')['rating'].agg(['mean', 'std']).fillna(0)
        user_rating_stats.columns = ['user_avg_rating', 'user_std_rating']
        user_rating_stats = user_rating_stats.reset_index()
        user_rating_stats['user_avg_rating_norm'] = user_rating_stats['user_avg_rating'] / 5.0
        user_rating_stats['user_std_rating_norm'] = (user_rating_stats['user_std_rating'] / 2.0).clip(0, 1)
        user_features = user_features.merge(
            user_rating_stats[['user_id', 'user_avg_rating_norm', 'user_std_rating_norm']],
            on='user_id', how='left')
        user_features['user_avg_rating_norm'] = user_features['user_avg_rating_norm'].fillna(0)
        user_features['user_std_rating_norm'] = user_features['user_std_rating_norm'].fillna(0)
    item_features = build_extended_item_features(train_click, raw_meta, encoders)
    articles_df = load_articles(config.DATA_PATH)

    all_item_vecs = None
    if os.path.exists(config.V2_EMBED_PKL):
        with open(config.V2_EMBED_PKL, 'rb') as f:
            all_item_vecs = pickle.load(f)
        print(f">>> Loaded V2 embeddings: {all_item_vecs.shape}")

    # --- 四路召回 (只对采样用户) ---
    print("\n>>> [Ch1] ItemCF Recall...")
    itemcf_dict = itemcf_recall(sample_users, user_item_time_dict, i2i_sim,
                                item_topk_click, sim_item_topk=10,
                                recall_item_num=config.RECALL_NUM)

    if all_item_vecs is not None:
        print(">>> [Ch2] V2 Embedding Recall...")
        v2_dict = embedding_recall(
            sample_users, {}, train_click, user_features, item_features,
            encoders, all_item_vecs, item_topk_click,
            hist_len=config.HIST_LEN, recall_item_num=config.RECALL_NUM, weight=1.0)
    else:
        print(">>> [Ch2] V2 skipped (no embeddings)")
        v2_dict = {}

    print(">>> [Ch3] Category Preference Recall...")
    cat_dict = category_preference_recall(
        sample_users, train_click, articles_df, item_topk_click,
        recall_item_num=20, weight=0.5)

    print(">>> [Ch4] Hot Recall...")
    hot_dict = hot_recall(sample_users, item_topk_click, recall_item_num=5, weight=0.1)

    # --- 命中统计 ---
    def _pool(recall_dict, u):
        items = recall_dict.get(u, {})
        return set(items.keys()) if isinstance(items, dict) else set(i for i, _ in items)

    n = len(sample_users)
    pools = {'ItemCF': itemcf_dict, 'V2': v2_dict, 'Category': cat_dict, 'Hot': hot_dict}
    hit_sets = {name: set() for name in pools}
    union_hit = set()

    for u in sample_users:
        gt = val_gt.get(u, set())
        if not gt:
            continue
        for name, d in pools.items():
            if gt & _pool(d, u):
                hit_sets[name].add(u)
        # 并集
        if gt & set().union(*[_pool(d, u) for d in pools.values()]):
            union_hit.add(u)

    print("\n>>> 各通道独立命中 (native recall 大小):")
    for name in ['ItemCF', 'V2', 'Category', 'Hot']:
        h = len(hit_sets[name])
        print(f"    {name:10s}: HR={h/max(n,1)*100:5.2f}% ({h:,}/{n:,})")

    print(f"\n>>> 并集 (四路合并, 对应召回天花板): "
          f"{len(union_hit)/max(n,1)*100:.2f}% ({len(union_hit):,}/{n:,})")

    # 独有命中 (只有这一路命中, 其他三路都没命中 → 该路不可替代)
    print("\n>>> 各路独有命中 (仅该路命中, 其余路都漏):")
    for name in ['ItemCF', 'V2', 'Category', 'Hot']:
        others = set().union(*[hit_sets[o] for o in pools if o != name])
        unique = hit_sets[name] - others
        print(f"    {name:10s}: {len(unique):,} 用户 ({len(unique)/max(n,1)*100:5.2f}%)")

    # 可达(top10) 用户的 ItemCF 命中率 (关键: 图能走到, ItemCF 到底排进 top100 没有?)
    return hit_sets, union_hit


def main():
    t0 = time.time()
    print(">>> Loading data (与 inference_full.py 一致)...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)
    click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)

    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    # ItemCF 只用正样本
    user_item_time_dict = get_user_item_time(
        train_click[train_click['click_label'] == 1])

    if os.path.exists(config.ITEMCF_SIM_PKL):
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        print(">>> Loaded cached ItemCF similarity")
    else:
        from itemcf import itemcf_sim
        i2i_sim = itemcf_sim(user_item_time_dict)
        with open(config.ITEMCF_SIM_PKL, 'wb') as f:
            pickle.dump(i2i_sim, f)

    val_gt = val_click.groupby('user_id')['click_article_id'].apply(set).to_dict()
    print(f">>> val_gt: {len(val_gt):,} users")

    # ===== Part 1 =====
    part1_reachability(val_gt, user_item_time_dict, i2i_sim, sim_item_topk=10)

    # ===== Part 2 =====
    if RUN_CHANNEL_BREAKDOWN:
        item_topk_click = get_item_topk_click(click_df, k=200)
        all_val_users = list(val_gt.keys())
        rng = np.random.default_rng(42)
        if len(all_val_users) > SAMPLE_USERS:
            sample_users = rng.choice(all_val_users, SAMPLE_USERS,
                                      replace=False).tolist()
        else:
            sample_users = all_val_users
        part2_channels(val_gt, sample_users, user_item_time_dict, i2i_sim,
                       item_topk_click, click_df, train_click, raw_meta)

    print(f"\n>>> Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
