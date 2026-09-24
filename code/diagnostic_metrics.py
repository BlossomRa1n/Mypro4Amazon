"""Non-invasive development diagnostics; candidate AUC is a negative-pool proxy."""
from contextlib import contextmanager
import math
import random
import numpy as np
import torch
from baseline_data import collate


@contextmanager
def preserve_training_state(model=None):
    """Diagnostics cannot advance training RNG or update BatchNorm statistics."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    flags = [(module, module.training) for module in model.modules()] if model is not None else []
    buffers = {key: value.detach().clone() for key, value in model.named_buffers()} if model is not None else {}
    try:
        if model is not None: model.eval()
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None: torch.cuda.set_rng_state_all(state[3])
        if model is not None:
            changed = []
            for key, value in model.named_buffers():
                if not torch.equal(value, buffers[key]):
                    value.copy_(buffers[key])
                    changed.append(key)
            for module, flag in flags: module.training = flag
            if changed: raise AssertionError(f"diagnostic mutated model buffers: {changed}")


def pair_auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative): return 0.0, False
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0) + 0.5 * (delta == 0)).mean()), True


def score_candidates(data, model, uid, position, ids):
    if not len(ids): return np.empty(0, dtype=np.float32)
    device = next(model.parameters()).device
    user = {key: value.to(device) for key, value in collate([data.user_features(int(uid), int(position))]).items()}
    encoded = model._encode_user(user)
    encoded = {key: value.expand(len(ids), *value.shape[1:]) if torch.is_tensor(value) and value.shape[0] == 1 else value for key, value in encoded.items()}
    items = {key: torch.as_tensor(value, device=device) for key, value in data.item_features(ids).items()}
    scores = model._score_item(encoded, items).float().detach().cpu().numpy().reshape(-1)
    if not np.isfinite(scores).all(): raise FloatingPointError('nonfinite candidate score')
    return scores


def evaluate_candidates(data, model, records, pools):
    if len(records) != len(pools): raise ValueError('candidate row count mismatch')
    width = max((len(pool) for pool in pools), default=0)
    n = len(records)
    arrays = dict(uid=np.asarray([x[0] for x in records], dtype=np.int64), position=np.asarray([x[1] for x in records], dtype=np.int64),
                  candidate_ids=np.zeros((n, width), dtype=np.int32), scores=np.zeros((n, width), dtype=np.float32),
                  labels=np.zeros((n, width), dtype=bool), lengths=np.zeros(n, dtype=np.int32),
                  hit5=np.zeros(n, dtype=np.int8), ndcg5=np.zeros(n, dtype=np.float32), pool_hit=np.zeros(n, dtype=np.int8),
                  auc=np.zeros(n, dtype=np.float64), auc_valid=np.zeros(n, dtype=bool))
    with preserve_training_state(model), torch.inference_mode():
        for index, ((uid, position, _), pool) in enumerate(zip(records, pools)):
            ids = list(map(int, pool))
            if len(set(ids)) != len(ids) or any(x < 2 or x >= len(data.items) for x in ids): raise ValueError('invalid candidate IDs')
            scores = score_candidates(data, model, uid, position, ids)
            targets = set(map(int, data.targets([(int(uid), int(position))])[0]))
            labels = np.asarray([item in targets for item in ids], dtype=bool)
            rank = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:5]
            size = len(ids)
            arrays['candidate_ids'][index, :size] = ids; arrays['scores'][index, :size] = scores
            arrays['labels'][index, :size] = labels; arrays['lengths'][index] = size
            arrays['hit5'][index] = any(labels[i] for i in rank); arrays['pool_hit'][index] = labels.any()
            arrays['ndcg5'][index] = sum(1 / math.log2(j + 2) for j, i in enumerate(rank) if labels[i]) / max(sum(1 / math.log2(j + 2) for j in range(min(5, len(targets)))), 1e-12)
            arrays['auc'][index], arrays['auc_valid'][index] = pair_auc(scores, labels)
    mask = arrays['auc_valid']
    metrics = {key: float(arrays[source].mean()) if n else 0.0 for key, source in [('hr5', 'hit5'), ('ndcg5', 'ndcg5'), ('pool_hit', 'pool_hit')]}
    metrics.update(users=n, gauc=float(arrays['auc'][mask].mean()) if mask.any() else 0.0, auc_valid_users=int(mask.sum()), auc_coverage=float(mask.mean()) if n else 0.0,
                   auc_semantics='user-equal AUC of existing candidate positives versus current negative proxy; raw-score ties count 0.5')
    return metrics, arrays


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'), map(float, np.quantile(array, [0, .25, .5, .75, .95, 1])))) if array.size else {}


def parameter_summary(model):
    result = {}
    with torch.no_grad():
        for name in ('user_embedding', 'item_embedding'):
            result[name + '_norm'] = quantiles(getattr(model, name).weight.detach().float().norm(dim=-1).cpu().numpy())
        result['batchnorm'] = {name: {'running_mean': quantiles(module.running_mean.cpu().numpy()), 'running_var': quantiles(module.running_var.cpu().numpy()), 'batches': int(module.num_batches_tracked)} for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
        result['parameter_l2'] = math.sqrt(sum(float(p.detach().float().square().sum()) for p in model.parameters()))
        result['gradient_l2_after_clip'] = math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in model.parameters() if p.grad is not None))
    return result
