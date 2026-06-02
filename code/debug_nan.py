"""快速诊断 NaN 来源 — 用实际训练参数 """
import torch, numpy as np, sys
sys.path.insert(0, 'code')
import config
from data_loader import get_all_click_df, load_articles, build_encoders, build_user_features, build_item_features, TwoTowerV2Dataset
from model import TwoTowerV2Model
from torch.utils.data import DataLoader
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Using device: {device}")

# 1. 数据
click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
articles_df = load_articles(config.DATA_PATH)
encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)
num_users, num_items = len(encoders['user_id'].classes_), len(encoders['item_id'].classes_)
num_categories = 1 if 'category_id' not in encoders else len(encoders['category_id'].classes_)
user_features = build_user_features(click_df, encoders, hist_len=config.HIST_LEN)
item_features = build_item_features(articles_df, click_df, encoders)

dataset = TwoTowerV2Dataset(
    click_df[click_df['click_article_id'].notna()], user_features, item_features, encoders,
    num_items=num_items, hist_len=config.HIST_LEN, hard_neg_index=None, num_hard_negatives=4
)

# 用训练 batch_size
B = min(config.V2_BATCH_SIZE, 512)
print(f"batch_size={B}, num_items={num_items}, num_users={num_users}")
batch = next(iter(DataLoader(dataset, batch_size=B, shuffle=True)))
batch = {k: v.to(device) for k, v in batch.items()}

# 2. 温度扫描
print("\n--- Temperature Scan (forward only) ---")
model = TwoTowerV2Model(
    num_users=num_users, num_items=num_items, num_categories=num_categories,
    embed_dim=config.EMBED_DIM, hidden_dims=config.HIDDEN_DIMS, hist_len=config.HIST_LEN,
    temperature=0.07, num_heads=config.SASREC_NUM_HEADS,
    num_blocks=config.SASREC_NUM_BLOCKS, dropout=0.0, num_hard_negatives=0
).to(device)

for temp in [0.5, 0.3, 0.2, 0.1, 0.07, 0.05]:
    model.temperature = temp
    with torch.no_grad():
        u, p = model(batch)
    loss = model.compute_infonce_loss(u, p)
    print(f"  temp={temp:.2f}: loss={loss.item():.6f}, nan={torch.isnan(loss).item()}")

# 3. 模拟 1 步训练
print("\n--- 1 Training Step (with backprop) ---")
model2 = TwoTowerV2Model(
    num_users=num_users, num_items=num_items, num_categories=num_categories,
    embed_dim=config.EMBED_DIM, hidden_dims=config.HIDDEN_DIMS, hist_len=config.HIST_LEN,
    temperature=config.INFONCE_TEMPERATURE, num_heads=config.SASREC_NUM_HEADS,
    num_blocks=config.SASREC_NUM_BLOCKS, dropout=config.SASREC_DROPOUT,
    num_hard_negatives=config.NUM_HARD_NEGATIVES
).to(device).train()

optimizer = optim.AdamW(model2.parameters(), lr=config.V2_LEARNING_RATE)

u, p = model2(batch)
loss = model2.compute_infonce_loss(u, p)
print(f"  Forward: loss={loss.item():.6f}, nan={torch.isnan(loss).item()}")

if torch.isnan(loss):
    print("  [FAIL] Forward NaN — problems in model init or embedding layers")
else:
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=5.0)
    nan_names = [name for name, p in model2.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    if nan_names:
        print(f"  [FAIL] Gradient NaN in: {nan_names[:5]}...")
    else:
        print("  Gradients OK")
        optimizer.step()
        with torch.no_grad():
            u2, p2 = model2(batch)
            loss2 = model2.compute_infonce_loss(u2, p2)
            print(f"  After step: loss={loss2.item():.6f}, nan={torch.isnan(loss2).item()}")

print("\nDone.")
