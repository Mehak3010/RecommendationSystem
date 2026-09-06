"""
Phase 2 - Deep Learning Models
===============================
1. NCF   - concat(user_emb, item_emb) -> MLP -> score
2. Two-Tower - separate user/item MLP encoders (item tower fuses TF-IDF
   content features, so it's a multimodal fusion tower), scored via dot
   product. This is the retrieval architecture used in most production
   systems (YouTube DNN, Pinterest PinnerSage, etc.) since the two towers
   can be indexed independently for fast ANN retrieval (see Phase 5).

Both trained with BPR pairwise loss + random negative sampling, same
protocol as the Phase 1 MF baseline for a fair comparison.
"""
import json, time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
from eval_utils import evaluate_model
from phase1_baselines import load, build_seen_and_gt

MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_PATH = f"{MODELS_DIR}/phase2_results.json"

def train_pairwise(model, score_fn, pos_pairs, n_items, n_epochs=3, batch_size=8192, lr=0.005, reg=1e-6, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    u_t = torch.tensor(pos_pairs[:, 0], dtype=torch.long)
    i_t = torch.tensor(pos_pairs[:, 1], dtype=torch.long)
    n = len(pos_pairs)
    steps = n // batch_size
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total = 0.0
        for step in range(steps):
            idx = perm[step*batch_size:(step+1)*batch_size]
            u_b, ip_b = u_t[idx], i_t[idx]
            in_b = torch.randint(0, n_items, (len(idx),))
            pos_score = score_fn(u_b, ip_b)
            neg_score = score_fn(u_b, in_b)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"  [{tag}] epoch {epoch+1}/{n_epochs} loss={total/steps:.4f}")
    print(f"  [{tag}] trained in {time.time()-t0:.1f}s")


class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, hidden=(64, 32, 16)):
        super().__init__()
        self.U = nn.Embedding(n_users, emb_dim)
        self.I = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.U.weight, std=0.05)
        nn.init.normal_(self.I.weight, std=0.05)
        layers = []
        in_dim = emb_dim * 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def score(self, u, i):
        x = torch.cat([self.U(u), self.I(i)], dim=-1)
        return self.mlp(x).squeeze(-1)


class TwoTower(nn.Module):
    """User tower: user_id embedding -> MLP.
    Item tower: item_id embedding + TF-IDF content vector -> MLP.
    Score = dot product of the two tower outputs (enables independent
    ANN indexing of the item tower at serving time)."""
    def __init__(self, n_users, n_items, content_dim, emb_dim=32, tower_dim=32):
        super().__init__()
        self.U = nn.Embedding(n_users, emb_dim)
        self.I = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.U.weight, std=0.05)
        nn.init.normal_(self.I.weight, std=0.05)
        self.user_tower = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, tower_dim))
        self.item_tower = nn.Sequential(nn.Linear(emb_dim + content_dim, 64), nn.ReLU(), nn.Linear(64, tower_dim))
        self.content = None  # set externally: (n_items, content_dim) tensor

    def user_vec(self, u):
        return self.user_tower(self.U(u))

    def item_vec(self, i):
        c = self.content[i]
        return self.item_tower(torch.cat([self.I(i), c], dim=-1))

    def score(self, u, i):
        return (self.user_vec(u) * self.item_vec(i)).sum(-1)

    def all_item_vecs(self, n_items, device):
        idx = torch.arange(n_items, device=device)
        with torch.no_grad():
            return self.item_vec(idx)

    def all_user_vecs(self, user_idx, device):
        idx = torch.tensor(user_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            return self.user_vec(idx)


def main():
    train, val, test, meta, item_meta = load()
    n_users, n_items = meta["n_users"], meta["n_items"]
    seen, gt, valid_users = build_seen_and_gt(train, test, n_users)

    rng = np.random.RandomState(42)
    all_eval_idx = np.where(valid_users)[0]
    eval_user_idx = rng.choice(all_eval_idx, size=min(5000, len(all_eval_idx)), replace=False)
    gt_eval = gt[eval_user_idx]
    seen_eval = [seen[u] for u in eval_user_idx]

    item_pop = train.i.value_counts().reindex(range(n_items), fill_value=0)
    item_pop_rank = item_pop.rank(ascending=False, method="first").values - 1

    item_meta = item_meta.sort_values("i").reset_index(drop=True)
    tfidf = TfidfVectorizer(max_features=2000)
    tag_features = tfidf.fit_transform(item_meta["tags_text"].fillna("")).toarray().astype(np.float32)
    # reduce to a manageable content dim for the item tower
    svd = TruncatedSVD(n_components=32, random_state=42)
    content_reduced = svd.fit_transform(tag_features).astype(np.float32)

    pos_pairs = train[train.label == 1][["u", "i"]].values
    all_results = {}

    # ---------- NCF ----------
    print("\n[1/2] Training NCF...")
    ncf = NCF(n_users, n_items, emb_dim=32)
    train_pairwise(ncf, ncf.score, pos_pairs, n_items, n_epochs=3, tag="NCF")
    with torch.no_grad():
        eu = torch.tensor(eval_user_idx, dtype=torch.long)
        all_i = torch.arange(n_items, dtype=torch.long)
        # score in chunks to keep memory sane
        scores = np.zeros((len(eval_user_idx), n_items), dtype=np.float32)
        for start in range(0, len(eval_user_idx), 200):
            chunk = eu[start:start+200]
            u_rep = chunk.repeat_interleave(n_items)
            i_rep = all_i.repeat(len(chunk))
            s = ncf.score(u_rep, i_rep).view(len(chunk), n_items).numpy()
            scores[start:start+200] = s
    all_results["ncf"] = evaluate_model(scores, gt_eval, seen_eval, item_pop_rank, tag_features)
    torch.save(ncf.state_dict(), f"{MODELS_DIR}/ncf.pt")

    # ---------- Two-Tower ----------
    print("\n[2/2] Training Two-Tower...")
    tt = TwoTower(n_users, n_items, content_dim=32, emb_dim=32, tower_dim=32)
    tt.content = torch.tensor(content_reduced)
    train_pairwise(tt, tt.score, pos_pairs, n_items, n_epochs=3, tag="TwoTower")
    with torch.no_grad():
        item_vecs = tt.all_item_vecs(n_items, "cpu").numpy()
        user_vecs = tt.all_user_vecs(eval_user_idx, "cpu").numpy()
    tt_scores = user_vecs @ item_vecs.T
    all_results["two_tower"] = evaluate_model(tt_scores, gt_eval, seen_eval, item_pop_rank, tag_features)
    torch.save(tt.state_dict(), f"{MODELS_DIR}/two_tower.pt")
    np.save(f"{MODELS_DIR}/two_tower_item_vecs.npy", item_vecs)  # for Phase 5 ANN index

    # ---------- Report ----------
    print("\n" + "="*70)
    df = pd.DataFrame(all_results).T
    cols = ["ndcg@10", "recall@10", "precision@10", "hit_rate@10", "coverage@10", "diversity@10", "pop_bias@10"]
    print(df[cols].round(4).to_string())
    json.dump(all_results, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
