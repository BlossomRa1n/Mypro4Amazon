"""
TwoTowerV2 训练脚本 (SASRec 用户塔 + InfoNCE loss + Hard Negative Mining)

新增: 时序验证 + Recall@K 评估 + Early Stop + 每 epoch 存档
"""
import os, time, pickle, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import (
    get_all_click_df, get_positive_click_df, load_articles,
    build_encoders, build_item_features, build_user_features,
    build_enhanced_user_features, build_hard_negative_index,
    TwoTowerV2Dataset
)
from model import TwoTowerV2Model
from evaluate import split_train_val, evaluate_two_tower, compute_item_embeddings


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])
    return result


def _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg, filepath, epoch, metrics, best_recall, patience):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'config': model_cfg,
        'epoch': epoch,
        'metrics': metrics,
        'best_recall': best_recall,
        'patience_counter': patience,
    }, filepath)


def _find_latest_checkpoint(prefix):
    """在 checkpoints 目录下找最新存档"""
    import glob
    pattern = os.path.join(config.CHECKPOINT_DIR, f'{prefix}_epoch*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    # 按 epoch 号排序 (文件名含 epoch05)
    def extract_epoch(p):
        import re
        m = re.search(r'epoch(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=extract_epoch)
    latest = files[-1]
    print(f">>> Found checkpoint: {os.path.basename(latest)}")
    return latest


def _load_checkpoint(filepath, model_class, optimizer, scheduler, device):
    """加载存档，恢复模型/优化器/scheduler 状态"""
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model_cfg = ckpt['config']
    model = model_class(**model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt.get('scheduler_state_dict', {}))
    start_epoch = ckpt.get('epoch', 0)
    best_recall = ckpt.get('best_recall', 0.0)
    patience_counter = ckpt.get('patience_counter', 0)
    metrics = ckpt.get('metrics', {})
    scaler_state = ckpt.get('scaler_state_dict')
    return model, model_cfg, start_epoch, best_recall, patience_counter, metrics, scaler_state


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
    user_features = build_enhanced_user_features(get_positive_click_df(train_click), articles_df, encoders, hist_len=config.HIST_LEN)

    # Hard Negative Index
    hard_neg_index = None
    if config.NUM_HARD_NEGATIVES > 0 and os.path.exists(config.ITEMCF_SIM_PKL):
        print(">>> Loading ItemCF similarity for hard negative mining...")
        with open(config.ITEMCF_SIM_PKL, 'rb') as f:
            i2i_sim = pickle.load(f)
        hard_neg_index = build_hard_negative_index(i2i_sim, encoders, config.NUM_HARD_NEGATIVES)

    # ================================================================
    # Step 3: 构建 Dataset / DataLoader (仅训练集)
    # ================================================================
    print("\nStep 3: Building V2 training dataset...")
    dataset = TwoTowerV2Dataset(
        train_click, user_features, item_features, encoders,
        num_items=num_items, hist_len=config.HIST_LEN,
        hard_neg_index=hard_neg_index, num_hard_negatives=config.NUM_HARD_NEGATIVES
    )
    dataloader = DataLoader(
        dataset, batch_size=config.V2_BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, persistent_workers=True
    )

    # ================================================================
    # Step 4: 初始化模型
    # ================================================================
    print("\nStep 4: Initializing TwoTowerV2 (SASRec + InfoNCE)...")
    model_cfg = {
        'num_users': num_users,
        'num_items': num_items,
        'num_categories': num_categories,
        'embed_dim': config.EMBED_DIM,
        'hidden_dims': config.HIDDEN_DIMS,
        'hist_len': config.HIST_LEN,
        'temperature': config.INFONCE_TEMPERATURE,
        'num_heads': config.SASREC_NUM_HEADS,
        'num_blocks': config.SASREC_NUM_BLOCKS,
        'dropout': config.SASREC_DROPOUT,
        'num_hard_negatives': config.NUM_HARD_NEGATIVES,
    }
    model = TwoTowerV2Model(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Total parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.V2_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.V2_NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ---- Resume: 自动检测最新 checkpoint 并恢复训练 ----
    start_epoch = 0
    best_recall = 0.0
    patience_counter = 0
    history = []
    resume_path = _find_latest_checkpoint('two_tower_v2')
    if resume_path:
        model, model_cfg, start_epoch, best_recall, patience_counter, prev_metrics, scaler_state = \
            _load_checkpoint(resume_path, TwoTowerV2Model, optimizer, scheduler, device)
        # 恢复 AMP scaler
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        print(f">>> Resumed from epoch {start_epoch}, best HR@{config.EVAL_K}={best_recall:.4f}, "
              f"patience={patience_counter}")
        # 从 history.json 恢复历史记录
        history_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            history = history[:start_epoch]  # 截断到 resume epoch

    # ================================================================
    # Step 5: 训练循环
    # ================================================================
    print(f"\nStep 5: Training from epoch {start_epoch+1}...")
    print(f"    AMP={config.USE_AMP}, GradAccum={config.GRADIENT_ACCUM_STEPS}, "
          f"Effective BS={config.V2_BATCH_SIZE * config.GRADIENT_ACCUM_STEPS}")

    best_epoch = start_epoch if start_epoch > 0 else -1

    for epoch in range(start_epoch, config.V2_NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.V2_NUM_EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            # AMP 自动混合精度: forward 用 fp16, loss 自动 scale
            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                user_vec, pos_item_vec = model(batch)

                hard_neg_vecs = None
                if config.NUM_HARD_NEGATIVES > 0 and 'hard_neg_ids' in batch and batch['hard_neg_ids'].numel() > 0:
                    bs = batch['hard_neg_ids'].shape[0]
                    nh = batch['hard_neg_ids'].shape[1]
                    hard_neg_batch = {
                        'item_id': batch['hard_neg_ids'].reshape(-1),
                        'category_id': batch['hard_neg_category_ids'].reshape(-1),
                        'item_click_count': batch['hard_neg_item_click_count'].reshape(-1),
                        'created_at_ts': batch['hard_neg_created_at_ts'].reshape(-1),
                    }
                    hard_neg_vecs = model.item_tower(
                        hard_neg_batch['item_id'], hard_neg_batch['category_id'],
                        hard_neg_batch['item_click_count'], hard_neg_batch['created_at_ts'],
                    )
                    hard_neg_vecs = hard_neg_vecs.reshape(bs, nh, -1)

                loss = model.compute_infonce_loss(user_vec, pos_item_vec, hard_neg_vecs,
                                                  click_weight=batch.get('click_weight'))
                # 梯度累积：loss 除以累积步数
                loss = loss / config.GRADIENT_ACCUM_STEPS

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # AMP: GradScaler 自动处理 loss 缩放
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积：每 accum_steps 个 batch 才更新一次
            if (batch_idx + 1) % config.GRADIENT_ACCUM_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config.GRADIENT_ACCUM_STEPS  # 恢复原始 loss 用于显示
            num_batches += 1
            eff_batch = config.V2_BATCH_SIZE * config.GRADIENT_ACCUM_STEPS
            pbar.set_postfix(loss=f"{loss.item() * config.GRADIENT_ACCUM_STEPS:.4f}",
                             eff_bs=f"{eff_batch}")

        scheduler.step()
        train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{config.V2_NUM_EPOCHS} | "
              f"Train Loss={train_loss:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        # ---- 验证 ----
        if config.SKIP_EVAL:
            metrics = {'hr': 0.0, 'ndcg': 0.0, 'n_users': 0}
        else:
            print("  Running validation...")
            metrics = evaluate_two_tower(
                model, val_click, train_click, user_features, item_features,
                encoders, device, k=config.EVAL_K, max_users=config.EVAL_MAX_USERS
            )

        hr = metrics['hr']
        ndcg = metrics['ndcg']
        print(f"  → Val HR@{config.EVAL_K}={hr:.4f} ({hr*100:.1f}%) | "
              f"NDCG@{config.EVAL_K}={ndcg:.4f} | "
              f"Users={metrics['n_users']}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_hr': hr, 'val_ndcg': ndcg})

        # ---- 每 epoch 存档 ----
        epoch_path = os.path.join(config.CHECKPOINT_DIR, f'two_tower_v2_epoch{epoch+1:02d}.pth')
        _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                        epoch_path, epoch + 1, metrics, best_recall, patience_counter)
        print(f"  → Saved: {epoch_path}")

        # ---- 更新最佳 ----
        is_better = config.SKIP_EVAL or hr > best_recall
        if is_better:
            best_recall = hr
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_best.pth')
            _save_checkpoint(model, optimizer, scheduler, scaler, model_cfg,
                            best_path, epoch + 1, metrics, best_recall, patience_counter)
            print(f"  ★ New best! HR@{config.EVAL_K}={hr:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs "
                  f"(best HR@{config.EVAL_K}={best_recall:.4f} at epoch {best_epoch})")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop triggered after {epoch+1} epochs")
            break

    # ---- 保存训练历史 ----
    history_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n>>> Training history saved to {history_path}")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best epoch: {best_epoch}, Best HR@{config.EVAL_K}={best_recall:.4f}")
    print(f"  Total epochs run: {len(history)}")
    print(f"  All checkpoints: {config.CHECKPOINT_DIR}/")
    print(f"{'='*60}")

    # ================================================================
    # Step 6: 最佳模型 → 预计算 item embeddings
    # ================================================================
    best_path = os.path.join(config.MODEL_PATH, 'two_tower_v2_best.pth')
    if not os.path.exists(best_path):
        print("[WARN] No best model saved, using last checkpoint.")
        best_path = os.path.join(config.CHECKPOINT_DIR,
                                 f'two_tower_v2_epoch{len(history):02d}.pth')

    print("\nStep 6: Loading best model for item embeddings...")
    # 只加载模型权重，不需要优化器状态
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = TwoTowerV2Model(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    best_ep = ckpt.get('epoch', -1)
    best_metrics = ckpt.get('metrics', {})

    print("  Pre-computing item embeddings...")
    all_item_vecs = compute_item_embeddings(
        model, num_items, item_features, device, batch_size=2048
    )

    all_item_vecs = all_item_vecs.cpu().numpy()
    with open(config.V2_EMBED_PKL, 'wb') as f:
        pickle.dump(all_item_vecs, f)
    print(f">>> Item embeddings saved to {config.V2_EMBED_PKL}, shape={all_item_vecs.shape}")

    # 也保存一份到 config 中 inference_full.py 可能用到的路径
    from evaluate import _build_item_batch_for_indices
    print(f"\nAll done! Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    train()
