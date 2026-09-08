"""
EDA: Amazon Video_Games 探索性分析 (面试用)
输出 8 张核心图表到 prediction_result/eda/
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import config
from data_loader import get_all_click_df, get_item_topk_click
from data_loader_ext import load_raw_meta, build_extended_encoders, merge_jsonl_features

OUT = os.path.join(config.RESULT_PATH, 'eda')
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
click_df = merge_jsonl_features(click_df, config.DATA_PATH, config.AMAZON_CATEGORIES)
raw_meta = load_raw_meta(config.DATA_PATH, config.AMAZON_CATEGORIES)

# ── 1. Rating Distribution ──
fig, ax = plt.subplots(figsize=(6, 4))
rating_counts = click_df['rating'].value_counts().sort_index()
ax.bar(rating_counts.index, rating_counts.values, color='#2196F3')
ax.set_title('Rating Distribution (Video_Games)', fontsize=12)
ax.set_xlabel('Rating'); ax.set_ylabel('Count')
for x, y in rating_counts.items():
    ax.text(x, y, f'{y:,}\n({y/len(click_df)*100:.1f}%)', ha='center', va='bottom', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/01_rating_dist.png', dpi=150); plt.close()

# ── 2. Interactions per User ──
user_ints = click_df.groupby('user_id').size()
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.clip(user_ints, 1, 50), bins=50, color='#FF9800', edgecolor='white')
for pct_val, label in [(95, 'P95'), (50, 'median')]:
    pct = np.percentile(user_ints, pct_val)
    ax.axvline(pct, color='red', linestyle='--', alpha=0.7)
    ax.text(pct, ax.get_ylim()[1]*0.95, f'{label}: {pct:.0f}', rotation=90, va='top', fontsize=9, color='red')
ax.set_title(f'Interactions per User (n={len(user_ints):,})', fontsize=12)
ax.set_xlabel('# Interactions (clipped at 50)'); ax.set_ylabel('# Users')
fig.tight_layout(); fig.savefig(f'{OUT}/02_user_interactions.png', dpi=150); plt.close()

# ── 3. Interactions per Item (Pareto) ──
item_ints = click_df.groupby('click_article_id').size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(item_ints)+1), item_ints.values, color='#4CAF50', linewidth=0.8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title(f'Item Popularity (Pareto) — {len(item_ints):,} items', fontsize=12)
ax.set_xlabel('Item Rank (log)'); ax.set_ylabel('# Interactions (log)')
fig.tight_layout(); fig.savefig(f'{OUT}/03_item_pareto.png', dpi=150); plt.close()

# ── 4. Time span per user ──
time_spans = click_df.groupby('user_id')['click_timestamp'].agg(lambda x: (x.max() - x.min()) / 86400_000)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.clip(time_spans, 0, 365*3), bins=50, color='#9C27B0', edgecolor='white')
ax.set_title(f'User Activity Time Span (days)', fontsize=12)
ax.set_xlabel('Days from first to last interaction (clipped)'); ax.set_ylabel('# Users')
fig.tight_layout(); fig.savefig(f'{OUT}/04_time_span.png', dpi=150); plt.close()

# ── 5. Top-20 Categories ──
if os.path.exists(config.EXT_ENCODER_PKL):
    with open(config.EXT_ENCODER_PKL, 'rb') as f:
        encoders = pickle.load(f)
    cat_counts = click_df['click_article_id'].map(lambda aid: raw_meta.set_index('parent_asin').get(aid, {}).get('main_category', 'Unknown') if isinstance(raw_meta.set_index('parent_asin').get(aid, {}), dict) else 'Unknown').value_counts().head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(cat_counts)), cat_counts.values, color='#E91E63')
    ax.set_yticks(range(len(cat_counts)))
    ax.set_yticklabels(cat_counts.index)
    ax.set_title('Top-20 Categories', fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(f'{OUT}/05_categories.png', dpi=150); plt.close()

# ── 6. Verified Purchase Rate ──
if 'verified_purchase' in click_df.columns:
    verified_rate = click_df.groupby('user_id')['verified_purchase'].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(verified_rate, bins=40, color='#00BCD4', edgecolor='white')
    ax.axvline(verified_rate.median(), color='red', linestyle='--', alpha=0.7, label=f'median={verified_rate.median():.1f}%')
    ax.set_title('User Verified Purchase Rate (%)', fontsize=12)
    ax.set_xlabel('% Verified'); ax.set_ylabel('# Users')
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{OUT}/06_verified_rate.png', dpi=150); plt.close()

# ── 7. Helpful Votes vs Rating ──
if 'helpful_vote' in click_df.columns and 'rating' in click_df.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    by_rating = click_df.groupby('rating')['helpful_vote'].mean()
    ax.bar(by_rating.index, by_rating.values, color='#FF5722')
    ax.set_title('Average Helpful Votes by Rating', fontsize=12)
    ax.set_xlabel('Rating'); ax.set_ylabel('Avg Helpful Votes')
    fig.tight_layout(); fig.savefig(f'{OUT}/07_helpful_vs_rating.png', dpi=150); plt.close()

# ── 8. Summary Stats ──
with open(f'{OUT}/summary.txt', 'w') as f:
    f.write(f"=== Video_Games EDA Summary ===\n")
    f.write(f"Interactions: {len(click_df):,}\n")
    f.write(f"Users: {click_df['user_id'].nunique():,}\n")
    f.write(f"Items: {click_df['click_article_id'].nunique():,}\n")
    f.write(f"Avg interactions/user: {user_ints.mean():.1f}\n")
    f.write(f"Median interactions/user: {user_ints.median():.0f}\n")
    f.write(f"Avg interactions/item: {item_ints.mean():.1f}\n")
    f.write(f"Median interactions/item: {item_ints.median():.0f}\n")
    f.write(f"Brands: {raw_meta['brand'].nunique():,}\n")
    f.write(f"Categories: {raw_meta['main_category'].nunique():,}\n")
    f.write(f"Avg rating: {click_df['rating'].mean():.2f}\n")
    verify = click_df.get('verified_purchase', pd.Series([0]*len(click_df)))
    f.write(f"Verified rate: {verify.mean()*100:.1f}%\n")

print(f"\nDone! 图表在: {OUT}/")
for fn in sorted(os.listdir(OUT)):
    print(f"  {OUT}/{fn}")
