"""
DIN 精排模型训练脚本 (Deep Interest Network + BCE loss)

新增: 时序验证 + AUC 评估 + Early Stop + 每 epoch 存档
"""
import os, time, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import (
    get_all_click_df, get_positive_click_df, load_articles,
    build_encoders, build_item_features, build_user_features,
    DINDataset
)
from model import DINModel
from evaluate import split_train_val, evaluate_din


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])
    return result


def _save_checkpoint(model, optimizer, scheduler, model_cfg, filepath, epoch, metrics, best_auc, patience, save_optimizer=True):
    data = {
        'model_state_dict': model.state_dict(),
        'config': model_cfg,
        'epoch': epoch,
        'metrics': metrics,
        'best_auc': best_auc,
        'patience_counter': patience,
    }
    if save_optimizer:
        data['optimizer_state_dict'] = optimizer.state_dict()
        data['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(data, filepath)


def _find_latest_checkpoint(prefix):
    import glob, re
    pattern = os.path.join(config.CHECKPOINT_DIR, f'{prefix}_epoch*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    def extract_epoch(p):
        m = re.search(r'epoch(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=extract_epoch)
    print(f">>> Found checkpoint: {os.path.basename(files[-1])}")
    return files[-1]


def _load_checkpoint(filepath, model_class, optimizer, scheduler, device):
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    model = model_class(**model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return model, model_cfg, ckpt.get('epoch', 0), ckpt.get('best_auc', 0.5), ckpt.get('patience_counter', 0), ckpt.get('metrics', {})


def train():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ================================================================
    # Step 1: 加载数据 & 时序分割
    # ================================================================
    print("Step 1: Loading data...")
    click_df = get_all_click_df(config.DATA_PATH, offline=config.OFFLINE_MODE)
    articles_df = load_articles(config.DATA_PATH)

    print("\n--- Temporal Train/Val Split ---")
    train_click, val_click = split_train_val(click_df, config.EVAL_SPLIT_RATIO)

    # ================================================================
    # Step 2: 构建编码器 & 特征
    # ================================================================
    print("\nStep 2: Building encoders & features...")
    encoders = build_encoders(click_df, articles_df, config.ENCODER_PKL)
    num_users = len(encoders['user_id'].classes_)
    num_items = len(encoders['item_id'].classes_)
    num_categories = len(encoders.get('category_id', __import__('sklearn').preprocessing.LabelEncoder()).classes_) if 'category_id' in encoders else 1
    print(f">>> num_users={num_users}, num_items={num_items}, num_categories={num_categories}")

    item_features = build_item_features(articles_df, click_df, encoders)
    user_features = build_user_features(get_positive_click_df(train_click), encoders, hist_len=config.HIST_LEN)

    # ================================================================
    # Step 3: 加载 ItemCF → Hard Negative Index → 构建 DIN Dataset
    # ================================================================
    print("\nStep 3: Loading ItemCF for hard negative mining...")
    hard_neg_index = None
    if os.path.exists(config.ITEMCF_SIM_PKL):
        import pickle
        from data_loader import build_hard_negative_index
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        hard_neg_index = build_hard_negative_index(
            i2i_sim, encoders, num_hard_negatives=4
        )
        print(f">>> Loaded ItemCF hard negatives for {len(hard_neg_index)} items")
    else:
        print(">>> No ItemCF cache found, using random negatives for DIN")

    print("Building DIN dataset (pointwise)...")
    dataset = DINDataset(
        train_click, user_features, item_features, encoders,
        hist_len=config.HIST_LEN, neg_ratio=4,
        hard_neg_index=hard_neg_index, num_hard_negatives=4
    )
    print(f">>> DIN dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset, batch_size=config.DIN_BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, persistent_workers=True
    )

    # ================================================================
    # Step 4: 初始化模型
    # ================================================================
    print("\nStep 4: Initializing DIN model...")
    model_cfg = {
        'num_users': num_users,
        'num_items': num_items,
        'num_categories': num_categories,
        'embed_dim': config.DIN_EMBED_DIM,
        'hidden_dims': config.DIN_HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'dropout': config.DIN_DROPOUT,
    }
    model = DINModel(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Total parameters: {total_params:,}")

    # ---- ALS 预训练初始化 (BPR/V2 有, DIN 也得有) ----
    if getattr(config, 'USE_ALS_INIT', False):
        from als_init import init_model_with_als
        pos_click = get_positive_click_df(click_df)
        model, als_ok = init_model_with_als(
            model, pos_click, encoders['user_id'], encoders['item_id'],
            fix_embeddings=getattr(config, 'ALS_FIX_EMBEDDINGS', False)
        )
        if als_ok:
            print(">>> ALS pre-training loaded — DIN starts from collaborative-filtering quality")
        else:
            print(">>> ALS unavailable for DIN, falling back to random init")
    # ----------------------------------

    optimizer = optim.Adam(model.parameters(), lr=config.DIN_LEARNING_RATE, weight_decay=config.DIN_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ---- Resume ----
    start_epoch, best_auc, patience_counter = 0, 0.5, 0
    history = []
    # ---- Resume (优先含优化器的 latest.pth, 回退到 epoch checkpoint) ----
    resume_path = os.path.join(config.MODEL_PATH, 'din_latest.pth')
    if not os.path.exists(resume_path):
        resume_path = _find_latest_checkpoint('din')
    if resume_path:
        model, model_cfg, start_epoch, best_auc, patience_counter, prev_metrics = \
            _load_checkpoint(resume_path, DINModel, optimizer, scheduler, device)
        print(f">>> Resumed from epoch {start_epoch}, best AUC={best_auc:.4f}")
        history_path = os.path.join(config.MODEL_PATH, 'din_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)[:start_epoch]

    # ================================================================
    # Step 5: 训练循环
    # ================================================================
    print(f"\nStep 5: Training DIN from epoch {start_epoch+1}...")
    best_epoch = start_epoch

    for epoch in range(start_epoch, config.DIN_NUM_EPOCHS):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.DIN_NUM_EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # BPR pairwise: 拆分 user / pos / neg
            user_keys = ['user_id', 'hist_items', 'hist_len', 'click_count', 'time_span']
            user_batch = {k: batch[k] for k in user_keys}
            pos_batch = {k.replace('pos_', 'item_'): batch[f'pos_{k}']
                         for k in ['item_id','category_id','item_click_count','created_at_ts']}
            neg_batch = {k.replace('neg_', 'item_'): batch[f'neg_{k}']
                         for k in ['item_id','category_id','item_click_count','created_at_ts']}

            pos_score, neg_score = model.forward_pairwise(user_batch, pos_batch, neg_batch)
            loss = model.compute_bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # BPR 准确率: 正样本分数 > 负样本分数的比例
            correct = (pos_score > neg_score).float().mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct:.4f}")

        scheduler.step()
        train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{config.DIN_NUM_EPOCHS} | "
              f"Train Loss={train_loss:.4f}")

        # ---- 验证 ----
        if config.SKIP_EVAL:
            metrics = {'auc': 0.5, 'pos_mean': 0.5, 'n_users': 0}
        else:
            print("  Running validation...")
            metrics = evaluate_din(
                model, val_click, user_features, item_features,
                encoders, device, max_users=config.EVAL_MAX_USERS
            )

        auc = metrics['auc']
        pos_mean = metrics['pos_mean']
        print(f"  → Val AUC={auc:.4f} | Pos Mean={pos_mean:.4f} | "
              f"Users={metrics['n_users']}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_auc': auc, 'val_pos_mean': pos_mean})

        # ---- 每 epoch 存档 (仅权重, 不存优化器以节省磁盘) ----
        epoch_path = os.path.join(config.CHECKPOINT_DIR, f'din_epoch{epoch+1:02d}.pth')
        _save_checkpoint(model, optimizer, scheduler, model_cfg,
                        epoch_path, epoch + 1, metrics, best_auc, patience_counter,
                        save_optimizer=False)

        # ---- 更新最新存档 (含优化器, 用于断点恢复) ----
        latest_path = os.path.join(config.MODEL_PATH, 'din_latest.pth')
        _save_checkpoint(model, optimizer, scheduler, model_cfg,
                        latest_path, epoch + 1, metrics, best_auc, patience_counter,
                        save_optimizer=True)

        # ---- 更新最佳 (基于 AUC) ----
        is_better = config.SKIP_EVAL or auc > best_auc
        if is_better:
            best_auc = auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = os.path.join(config.MODEL_PATH, 'din_best.pth')
            _save_checkpoint(model, optimizer, scheduler, model_cfg,
                            best_path, epoch + 1, metrics, best_auc, patience_counter,
                            save_optimizer=True)
            print(f"  ★ New best! AUC={auc:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs "
                  f"(best AUC={best_auc:.4f} at epoch {best_epoch})")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop triggered after {epoch+1} epochs")
            break

    # ---- 保存训练历史 ----
    history_path = os.path.join(config.MODEL_PATH, 'din_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n>>> Training history saved to {history_path}")

    print(f"\n{'='*60}")
    print(f"DIN training complete!")
    print(f"  Best epoch: {best_epoch}, Best AUC={best_auc:.4f}")
    print(f"  Total epochs run: {len(history)}")
    print(f"  All checkpoints: {config.CHECKPOINT_DIR}/")
    print(f"{'='*60}")
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    train()
