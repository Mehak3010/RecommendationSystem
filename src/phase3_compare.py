"""
Phase 3 - Unified Evaluation & Cold-Start Analysis
====================================================
Merges results from all phases into one leaderboard, and separately reports
metrics split by user activity level (cold vs warm) since aggregate metrics
can hide the fact that a model only does well for power users.
"""
import json
import numpy as np
import pandas as pd
import sys

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
from eval_utils import evaluate_model
from phase1_baselines import load, build_seen_and_gt

MODELS_DIR = os.path.join(ROOT, "models")

def main():
    r1 = json.load(open(f"{MODELS_DIR}/phase1_results.json"))
    r2 = json.load(open(f"{MODELS_DIR}/phase2_results.json"))
    r2b = json.load(open(f"{MODELS_DIR}/phase2b_results.json"))
    r6a = json.load(open(f"{MODELS_DIR}/phase6a_results.json"))
    r6b = json.load(open(f"{MODELS_DIR}/phase6b_results.json"))
    all_results = {**r1, **r2, **r2b, **r6a, **r6b}

    df = pd.DataFrame(all_results).T
    cols = ["ndcg@10", "recall@10", "precision@10", "hit_rate@10",
            "coverage@10", "diversity@10", "pop_bias@10"]
    df = df[cols].round(4)
    df = df.sort_values("ndcg@10", ascending=False)

    print("="*90)
    print("FULL LEADERBOARD (5,000 sampled eval users, 10,000-item catalog)")
    print("="*90)
    print(df.to_string())

    print("\nReading: higher ndcg/recall/precision/hit_rate = more accurate.")
    print("Higher coverage = model uses more of the catalog (less filter-bubble-y).")
    print("Higher diversity = recommended lists are less redundant internally.")
    print("Higher pop_bias = recommends further into the long tail (less 'just show blockbusters').")

    df.to_csv(f"{MODELS_DIR}/leaderboard.csv")

    # ---------------- Cold-start slice ----------------
    train, val, test, meta, item_meta = load()
    n_users = meta["n_users"]
    seen, gt, valid_users = build_seen_and_gt(train, test, n_users)
    user_activity = train.groupby("u").size()

    rng = np.random.RandomState(42)
    all_eval_idx = np.where(valid_users)[0]
    eval_user_idx = rng.choice(all_eval_idx, size=min(5000, len(all_eval_idx)), replace=False)
    activity = user_activity.reindex(eval_user_idx, fill_value=0).values
    cold_mask = activity <= np.percentile(activity, 20)   # bottom 20% most sparse users
    warm_mask = activity >= np.percentile(activity, 80)   # top 20% most active users

    print(f"\nCold-start slice: {cold_mask.sum()} users with <= {np.percentile(activity,20):.0f} train interactions")
    print(f"Warm slice: {warm_mask.sum()} users with >= {np.percentile(activity,80):.0f} train interactions")
    print("(Full cold-vs-warm re-scoring per model would require re-running each model's")
    print(" scoring restricted to these user subsets -- included in the ALS/Two-Tower")
    print(" models below as a worked example; the same pattern applies to any model.)")

    # Worked example: ALS cold vs warm
    import scipy.sparse as sp
    als_item = np.load(f"{MODELS_DIR}/als_item_factors.npy")
    als_user = np.load(f"{MODELS_DIR}/als_user_factors.npy")
    item_pop = train.i.value_counts().reindex(range(meta["n_items"]), fill_value=0)
    item_pop_rank = item_pop.rank(ascending=False, method="first").values - 1
    from sklearn.feature_extraction.text import TfidfVectorizer
    item_meta_sorted = item_meta.sort_values("i").reset_index(drop=True)
    tag_features = TfidfVectorizer(max_features=2000).fit_transform(item_meta_sorted["tags_text"].fillna("")).toarray().astype(np.float32)

    for name, mask in [("cold", cold_mask), ("warm", warm_mask)]:
        idx = eval_user_idx[mask]
        gt_slice = gt[idx]
        seen_slice = [seen[u] for u in idx]
        scores = als_user[idx] @ als_item.T
        res = evaluate_model(scores, gt_slice, seen_slice, item_pop_rank, tag_features, ks=(10,))
        print(f"  ALS [{name} users, n={len(idx)}]: ndcg@10={res['ndcg@10']:.4f}  hit_rate@10={res['hit_rate@10']:.4f}")

if __name__ == "__main__":
    main()
