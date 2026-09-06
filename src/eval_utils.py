"""
Shared ranking-evaluation utilities (Phase 3), used by every model so all
comparisons are apples-to-apples.

All metrics operate on a `scores` matrix of shape (n_eval_users, n_items)
(higher = more recommended) plus the ground-truth held-out item per user,
and a `seen` set per user (train items) to mask out from ranking.
"""
import numpy as np


def mask_seen(scores, seen_items_per_user):
    """Set scores of already-seen (train) items to -inf so they can't be
    'recommended' again -- standard practice, otherwise metrics are inflated
    by the model just re-surfacing items the user already rated."""
    scores = scores.copy()
    for row, seen in enumerate(seen_items_per_user):
        if len(seen):
            scores[row, seen] = -np.inf
    return scores


def topk_indices(scores, k):
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-row_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def precision_recall_ndcg_at_k(scores, ground_truth_items, k):
    """ground_truth_items: array of shape (n_users,) - single held-out item
    per user (leave-one-out style, matches our Phase 0 split)."""
    top_k = topk_indices(scores, k)
    hits = (top_k == ground_truth_items[:, None])
    hit_any = hits.any(axis=1).astype(float)

    precision = hit_any / k
    recall = hit_any  # single relevant item per user -> recall is 0/1

    # NDCG: rank position of the hit (if any)
    hit_rank = np.where(hits.any(axis=1), hits.argmax(axis=1), -1)
    ndcg = np.zeros(len(ground_truth_items))
    valid = hit_rank >= 0
    ndcg[valid] = 1.0 / np.log2(hit_rank[valid] + 2)

    # MRR
    mrr = np.zeros(len(ground_truth_items))
    mrr[valid] = 1.0 / (hit_rank[valid] + 1)

    return {
        f"precision@{k}": precision.mean(),
        f"recall@{k}": recall.mean(),
        f"ndcg@{k}": ndcg.mean(),
        f"mrr@{k}": mrr.mean(),
        f"hit_rate@{k}": hit_any.mean(),
    }


def coverage_at_k(scores, k, n_items):
    top_k = topk_indices(scores, k)
    unique_recommended = np.unique(top_k)
    return len(unique_recommended) / n_items


def popularity_bias_at_k(scores, k, item_pop_rank):
    """Average popularity RANK (0=most popular) of recommended items.
    Lower = model just recommends blockbusters; higher = more long-tail."""
    top_k = topk_indices(scores, k)
    ranks = item_pop_rank[top_k]
    return ranks.mean()


def intra_list_diversity_at_k(scores, k, item_features):
    """1 - average pairwise cosine similarity within each user's top-k list,
    using item content features (e.g. TF-IDF tag vectors)."""
    top_k = topk_indices(scores, k)
    feats = item_features[top_k]  # (n_users, k, d)
    norms = np.linalg.norm(feats, axis=2, keepdims=True) + 1e-9
    feats = feats / norms
    sims = np.matmul(feats, feats.transpose(0, 2, 1))  # (n_users, k, k)
    iu = np.triu_indices(k, k=1)
    pair_sims = sims[:, iu[0], iu[1]]
    return float(1 - pair_sims.mean())


def evaluate_model(scores, ground_truth_items, seen_items_per_user, item_pop_rank,
                    item_features=None, ks=(5, 10, 20)):
    scores = mask_seen(scores, seen_items_per_user)
    results = {}
    for k in ks:
        results.update(precision_recall_ndcg_at_k(scores, ground_truth_items, k))
        results[f"coverage@{k}"] = coverage_at_k(scores, k, scores.shape[1])
        results[f"pop_bias@{k}"] = popularity_bias_at_k(scores, k, item_pop_rank)
        if item_features is not None:
            results[f"diversity@{k}"] = intra_list_diversity_at_k(scores, k, item_features)
    return results
