"""
Phase 1 - Classical Baselines
=============================
Popularity | Content-based (TF-IDF over genre tags) | ALS (implicit) |
Matrix Factorization (SGD via PyTorch, implemented from scratch)

All models are evaluated identically using eval_utils.evaluate_model against
the held-out (leave-one-out) test interaction per user.
"""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import implicit
import sys, time, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
from eval_utils import evaluate_model

DATA = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "phase1_results.json")

def load():
    train = pd.read_csv(f"{DATA}/train.csv")
    val = pd.read_csv(f"{DATA}/val.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    meta = json.load(open(f"{DATA}/mappings.json"))
    item_meta = pd.read_csv(f"{DATA}/item_meta.csv")
    return train, val, test, meta, item_meta


def build_seen_and_gt(train, test, n_users):
    seen = [np.array([], dtype=int)] * n_users
    seen_map = train.groupby("u")["i"].apply(lambda s: s.values).to_dict()
    for u, items in seen_map.items():
        seen[u] = items
    gt = np.full(n_users, -1, dtype=int)
    for row in test.itertuples():
        gt[row.u] = row.i
    valid_users = gt >= 0
    return seen, gt, valid_users


def main():
    train, val, test, meta, item_meta = load()
    n_users, n_items = meta["n_users"], meta["n_items"]
    print(f"n_users={n_users} n_items={n_items} train={len(train):,}")

    seen, gt, valid_users = build_seen_and_gt(train, test, n_users)
    all_eval_idx = np.where(valid_users)[0]
    # Subsample evaluation users to keep the dense (n_eval_users x n_items)
    # scoring matrices memory-friendly; 5k users is plenty for stable
    # aggregate ranking metrics on a 10k-item catalog.
    rng = np.random.RandomState(42)
    EVAL_SAMPLE = min(5000, len(all_eval_idx))
    eval_user_idx = rng.choice(all_eval_idx, size=EVAL_SAMPLE, replace=False)
    gt_eval = gt[eval_user_idx]
    seen_eval = [seen[u] for u in eval_user_idx]
    print(f"Evaluating on {len(eval_user_idx):,} sampled users (out of {len(all_eval_idx):,} eligible)")

    item_pop = train.i.value_counts().reindex(range(n_items), fill_value=0)
    item_pop_rank = item_pop.rank(ascending=False, method="first").values - 1  # 0 = most popular

    tfidf = TfidfVectorizer(max_features=2000)
    item_meta = item_meta.sort_values("i").reset_index(drop=True)
    tag_features = tfidf.fit_transform(item_meta["tags_text"].fillna("")).toarray().astype(np.float32)

    all_results = {}

    # ---------- 1. Popularity baseline ----------
    print("\n[1/4] Popularity baseline...")
    pop_scores_row = (-item_pop_rank).astype(np.float32)  # higher score = more popular
    pop_scores = np.broadcast_to(pop_scores_row, (len(eval_user_idx), n_items)).copy()
    all_results["popularity"] = evaluate_model(pop_scores, gt_eval, seen_eval, item_pop_rank, tag_features)

    # ---------- 2. Content-based (user profile = avg TF-IDF of liked items) ----------
    print("[2/4] Content-based (TF-IDF)...")
    liked = train[train.label == 1]
    user_profile = np.zeros((n_users, tag_features.shape[1]), dtype=np.float32)
    sum_feats = liked.groupby("u")["i"].apply(lambda idxs: tag_features[idxs].sum(axis=0))
    for u, vec in sum_feats.items():
        user_profile[u] = vec
    norms = np.linalg.norm(user_profile, axis=1, keepdims=True) + 1e-9
    user_profile = user_profile / norms
    content_scores = user_profile[eval_user_idx] @ tag_features.T
    all_results["content_based"] = evaluate_model(content_scores, gt_eval, seen_eval, item_pop_rank, tag_features)

    # ---------- 3. ALS (implicit library) ----------
    print("[3/4] ALS (implicit)...")
    conf = 1 + 15 * train.label.values  # confidence weighting for implicit feedback
    user_item = sp.csr_matrix((conf, (train.u.values, train.i.values)), shape=(n_users, n_items))
    als = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, random_state=42)
    t0 = time.time()
    als.fit(user_item)
    print(f"  ALS trained in {time.time()-t0:.1f}s")
    als_scores = als.user_factors[eval_user_idx] @ als.item_factors.T
    all_results["als"] = evaluate_model(als_scores, gt_eval, seen_eval, item_pop_rank, tag_features)
    np.save(os.path.join(MODELS_DIR, "als_item_factors.npy"), als.item_factors)
    np.save(os.path.join(MODELS_DIR, "als_user_factors.npy"), als.user_factors)

    # ---------- 4. Matrix Factorization from scratch (PyTorch, SGD + BPR-style loss) ----------
    print("[4/4] Matrix Factorization (PyTorch, from scratch)...")
    device = "cpu"
    EMB_DIM = 64
    class MF(nn.Module):
        def __init__(self, n_users, n_items, dim):
            super().__init__()
            self.U = nn.Embedding(n_users, dim)
            self.I = nn.Embedding(n_items, dim)
            self.u_bias = nn.Embedding(n_users, 1)
            self.i_bias = nn.Embedding(n_items, 1)
            nn.init.normal_(self.U.weight, std=0.05)
            nn.init.normal_(self.I.weight, std=0.05)
            nn.init.zeros_(self.u_bias.weight)
            nn.init.zeros_(self.i_bias.weight)

        def score(self, u, i):
            return (self.U(u) * self.I(i)).sum(-1) + self.u_bias(u).squeeze(-1) + self.i_bias(i).squeeze(-1)

    mf = MF(n_users, n_items, EMB_DIM).to(device)
    opt = torch.optim.Adam(mf.parameters(), lr=0.01, weight_decay=1e-6)

    pos = train[train.label == 1][["u", "i"]].values
    u_pos_t = torch.tensor(pos[:, 0], dtype=torch.long)
    i_pos_t = torch.tensor(pos[:, 1], dtype=torch.long)

    n_pos = len(pos)
    batch_size = 8192
    n_epochs = 3
    steps_per_epoch = n_pos // batch_size
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_pos)
        total_loss = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step*batch_size:(step+1)*batch_size]
            u_b = u_pos_t[idx]
            i_pos_b = i_pos_t[idx]
            i_neg_b = torch.randint(0, n_items, (len(idx),))  # random negative sampling (BPR)

            pos_score = mf.score(u_b, i_pos_b)
            neg_score = mf.score(u_b, i_neg_b)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
            loss = loss + 1e-5 * (mf.U(u_b).pow(2).sum() + mf.I(i_pos_b).pow(2).sum())

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  epoch {epoch+1}/{n_epochs} loss={total_loss/steps_per_epoch:.4f}")
    print(f"  MF trained in {time.time()-t0:.1f}s")

    with torch.no_grad():
        U = mf.U.weight.numpy()
        I = mf.I.weight.numpy()
        ub = mf.u_bias.weight.numpy().squeeze(-1)
        ib = mf.i_bias.weight.numpy().squeeze(-1)
    mf_scores = U[eval_user_idx] @ I.T + ub[eval_user_idx, None] + ib[None, :]
    all_results["mf_bpr_pytorch"] = evaluate_model(mf_scores, gt_eval, seen_eval, item_pop_rank, tag_features)
    torch.save(mf.state_dict(), os.path.join(MODELS_DIR, "mf_bpr.pt"))

    # ---------- Report ----------
    print("\n" + "="*70)
    df_results = pd.DataFrame(all_results).T
    cols = ["ndcg@10", "recall@10", "precision@10", "hit_rate@10", "coverage@10", "diversity@10", "pop_bias@10"]
    print(df_results[cols].round(4).to_string())
    json.dump(all_results, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
