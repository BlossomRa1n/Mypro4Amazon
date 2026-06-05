"""快速诊断: 检查训练前后的 embedding 分布"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config
import pickle, numpy as np
from collections import defaultdict

def diagnose(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict']
    
    print(f"\n=== Diagnose: {model_path} ===")
    
    # 1. 检查 item_embedding
    for key in ['item_tower.item_embedding.weight', 
                'user_tower.item_embedding.weight',
                'user_tower.item_embedding_for_hist.weight']:
        if key in sd:
            w = sd[key].cpu().numpy()
            norms = np.linalg.norm(w, axis=1)
            print(f"  {key}:")
            print(f"    shape={w.shape}, mean_norm={norms.mean():.4f}, std_norm={norms.std():.4f}")
            print(f"    min_norm={norms.min():.4f}, max_norm={norms.max():.4f}")
            # 检查方差
            var = np.var(w, axis=0).mean()
            print(f"    avg_dim_variance={var:.6f}")
    
    # 2. 检查所有 embedding 的余弦相似度 (抽样)
    for key in ['item_tower.item_embedding.weight']:
        if key in sd:
            w = sd[key].cpu().numpy()
            # 随机取 100 个向量, 算 pairwise cosine similarity
            idx = np.random.choice(len(w), min(100, len(w)), replace=False)
            sample = w[idx]
            sample = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-8)
            sim = np.dot(sample, sample.T)
            # 去掉对角线
            mask = ~np.eye(len(sample), dtype=bool)
            avg_sim = sim[mask].mean()
            print(f"  Average pairwise cosine similarity (100 samples): {avg_sim:.4f}")
            if avg_sim > 0.9:
                print(f"  ⚠️ COLLAPSED! All items have nearly identical embeddings.")
            elif avg_sim < 0.05:
                print(f"  ✅ Diverse embeddings (near-orthogonal).")
    
    # 3. 检查 user_embedding
    for key in ['user_tower.user_embedding.weight']:
        if key in sd:
            w = sd[key].cpu().numpy()
            norms = np.linalg.norm(w, axis=1)
            print(f"  {key}:")
            print(f"    mean_norm={norms.mean():.4f}, std_norm={norms.std():.4f}")
            var = np.var(w, axis=0).mean()
            print(f"    avg_dim_variance={var:.6f}")

if __name__ == "__main__":
    for p in [config.V2_BEST_FILE, config.V2_MODEL_FILE, 
              config.DEEP_BEST_FILE, config.DEEP_MODEL_FILE]:
        if os.path.exists(p):
            diagnose(p)
