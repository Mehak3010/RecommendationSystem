"""
Phase 6a - LightGCN (Graph Neural Network recommender), from scratch
======================================================================
He et al. 2020. No feature transform, no nonlinearity -- LightGCN's whole
point is that for collaborative filtering, simplified neighborhood
aggregation (E_{k+1} = D^-1/2 A D^-1/2 E_k) outperforms heavier GCN
variants. Implemented here with a torch sparse COO adjacency (no
torch-geometric dependency, which isn't installable in this sandbox).

Compute note: full propagation over the whole user-item graph on every
mini-batch step is expensive on CPU at this graph size (63k nodes, ~11.7M
directed edges). We use large batches and a capped step budget to keep
training time reasonable; see README for the honest performance-vs-compute
tradeoff discussion.
"""
import json, time, sys
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

sys.path.append("/home/claude/recsys_project/src")
from eval_utils import evaluate_model
from phase1_baselines import load, build_seen_and_gt

MODELS_DIR = "/home/claude/recsys_project/models"
N_LAYERS = 3
DIM = 64


def build_norm_adj(train_pos, n_users, n_items):
    u = train_pos[:, 0]
    i = train_pos[:, 1] + n_users  # item nodes offset after user nodes
    n_nodes = n_users + n_items
    rows = np.concatenate([u, i])
    cols = np.concatenate([i, u])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

    indices = torch.tensor(np.vstack([A_norm.row, A_norm.col]), dtype=torch.long)
    values = torch.tensor(A_norm.data, dtype=torch.float32)
    A_t = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()
    return A_t


def main():
    train, val, test, meta, item_meta = load()
    n_users, n_items = meta["n_users"], meta["n_items"]
    seen, gt, valid_users = build_seen_and_gt(train, test, n_users)

    pos = train[train.label == 1][["u", "i"]].values
    print(f"Building normalized adjacency over {n_users + n_items:,} nodes, {len(pos):,} positive edges...")
    A_hat = build_norm_adj(pos, n_users, n_items)
    print(f"  adjacency nnz = {A_hat._nnz():,}")

    E0 = torch.nn.Parameter(torch.randn(n_users + n_items, DIM) * 0.05)
    opt = torch.optim.Adam([E0], lr=0.01)

    def propagate(E):
        embs = [E]
        h = E
        for _ in range(N_LAYERS):
            h = torch.sparse.mm(A_hat, h)
            embs.append(h)
        return torch.stack(embs, dim=0).mean(dim=0)

    # quick timing check
    t0 = time.time()
    _ = propagate(E0)
    print(f"  single propagation pass: {time.time()-t0:.2f}s")

    N_STEPS = 60
    BATCH = 16384
    t0 = time.time()
    for step in range(N_STEPS):
        idx = np.random.randint(0, len(pos), BATCH)
        u_b = torch.tensor(pos[idx, 0], dtype=torch.long)
        i_pos_b = torch.tensor(pos[idx, 1] + n_users, dtype=torch.long)
        i_neg_b = torch.tensor(np.random.randint(0, n_items, BATCH) + n_users, dtype=torch.long)

        final_emb = propagate(E0)
        u_e = final_emb[u_b]
        pos_e = final_emb[i_pos_b]
        neg_e = final_emb[i_neg_b]
        pos_score = (u_e * pos_e).sum(-1)
        neg_score = (u_e * neg_e).sum(-1)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        loss = loss + 1e-4 * (u_e.pow(2).sum() + pos_e.pow(2).sum()) / BATCH

        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0 or step == N_STEPS - 1:
            print(f"  step {step+1}/{N_STEPS} loss={loss.item():.4f} ({time.time()-t0:.1f}s elapsed)")

    print(f"LightGCN trained in {time.time()-t0:.1f}s total")

    with torch.no_grad():
        final_emb = propagate(E0).numpy()
    user_emb = final_emb[:n_users]
    item_emb = final_emb[n_users:]
    np.save(f"{MODELS_DIR}/lightgcn_user_emb.npy", user_emb)
    np.save(f"{MODELS_DIR}/lightgcn_item_emb.npy", item_emb)

    # ---- evaluate ----
    rng = np.random.RandomState(42)
    all_eval_idx = np.where(valid_users)[0]
    eval_user_idx = rng.choice(all_eval_idx, size=min(5000, len(all_eval_idx)), replace=False)
    gt_eval = gt[eval_user_idx]
    seen_eval = [seen[u] for u in eval_user_idx]
    item_pop = train.i.value_counts().reindex(range(n_items), fill_value=0)
    item_pop_rank = item_pop.rank(ascending=False, method="first").values - 1
    item_meta_sorted = item_meta.sort_values("i").reset_index(drop=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tag_features = TfidfVectorizer(max_features=2000).fit_transform(item_meta_sorted["tags_text"].fillna("")).toarray().astype(np.float32)

    scores = user_emb[eval_user_idx] @ item_emb.T
    results = {"lightgcn": evaluate_model(scores, gt_eval, seen_eval, item_pop_rank, tag_features)}
    print("\n" + "="*70)
    df = pd.DataFrame(results).T
    print(df[["ndcg@10", "recall@10", "hit_rate@10", "coverage@10", "diversity@10", "pop_bias@10"]].round(4).to_string())
    json.dump(results, open(f"{MODELS_DIR}/phase6a_results.json", "w"), indent=2)

if __name__ == "__main__":
    main()
