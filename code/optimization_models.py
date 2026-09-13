"""Parameter-compatible DIN feature slices and cached-user inference."""
import torch

from baseline_data import collate
from model_ext import DINExtendedModel
from run_baseline import to_device


class RawSliceDIN(DINExtendedModel):
    def raw_slices(self, user, batch):
        size, device = user["user_emb"].shape[0], user["user_emb"].device
        item = self.item_embedding(batch["item_id"])
        brand = self.brand_embedding(batch["brand_id"])
        verified = batch.get("verified", torch.zeros(size, dtype=torch.long, device=device))
        interest = self.attention_layer(
            user["hist_emb"], item, hist_brand_emb=user["hist_brand_emb"],
            target_brand_emb=brand, hist_ratings=user["hist_ratings"],
            hist_deltas=user["hist_deltas"], hist_verified=user["hist_verified"],
            hist_mask=user["hist_mask"])
        dense_user = [user[key] for key in ("click_count", "time_span", "user_avg_rating",
                      "user_std_rating", "user_verified_ratio", "user_avg_helpful")]
        dense_item = [batch.get(key, torch.zeros(size, device=device)).float().unsqueeze(-1)
                      for key in ("item_click_count", "created_at_ts", "item_avg_rating", "item_rating_number")]
        # Preserve original order, widths and raw values, including constant branches.
        return (user["user_emb"], item, self.category_embedding(batch["category_id"]),
                brand, self.verified_embedding(verified), interest,
                torch.cat(dense_user + dense_item, dim=-1), user["user_emb"] * item)

    def _score_item(self, user_enc, batch):
        return self.mlp(torch.cat(self.raw_slices(user_enc, batch), dim=-1)).squeeze(-1)

    def _score(self, batch):
        return self._score_item(self._encode_user(batch), batch)


def score_din_cached(model, data, records, pools, device, microbatch=512):
    """Cache only candidate-independent tensors; target attention remains per item."""
    model.eval()
    ranked = []
    with torch.inference_mode():
        for offset in range(0, len(records), 16):
            chunk = records[offset:offset + 16]
            parts = [list(pool) for pool in pools[offset:offset + 16]]
            sizes = [len(part) for part in parts]
            ids = [iid for part in parts for iid in part]
            if not ids:
                ranked.extend([[] for _ in chunk])
                continue
            users = to_device(collate([data.user_features(uid, pos) for uid, pos in chunk]), device)
            encoded = model._encode_user(users)
            rows = torch.repeat_interleave(torch.arange(len(chunk), device=device),
                                           torch.as_tensor(sizes, device=device))
            scores = []
            for start in range(0, len(ids), microbatch):
                index = rows[start:start + microbatch]
                user = {key: value[index] if value is not None else None for key, value in encoded.items()}
                items = {key: torch.as_tensor(value, device=device) for key, value in
                         data.item_features(ids[start:start + microbatch]).items()}
                scores.extend(model._score_item(user, items).float().cpu().tolist())
            cursor = 0
            for part in parts:
                values = scores[cursor:cursor + len(part)]
                order = sorted(range(len(part)), key=lambda i: (-values[i], part[i]))
                ranked.append([part[i] for i in order])
                cursor += len(part)
    return ranked
