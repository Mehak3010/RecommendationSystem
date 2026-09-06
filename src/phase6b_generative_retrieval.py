"""
Phase 6b - Semantic-ID Generative Retrieval (simplified TIGER-style)
=======================================================================
The 2026 trend this demonstrates: instead of a fixed item catalog looked
up via ANN, represent each item as a learned discrete "semantic ID" token
(derived by clustering item embeddings), and train a generative model to
predict the next semantic token directly -- the approach behind TIGER and
the generative-retrieval component of LinkedIn's 2026 feed-ranking rebuild.

SIMPLIFICATION (documented): the full published approach uses multi-level
residual quantization (RQ-VAE) to give each item a short *sequence* of
tokens (finer-grained, near-unique per item). Doing that with correct,
efficient multi-level batched scoring across a 10k-item catalog is a
non-trivial systems problem on its own. Here we implement a single-level
version: items are clustered into K semantic tokens (coarse "genre-like"
clusters learned from embeddings, not hand-labeled), a transformer predicts
the user's next semantic token (i.e. "what kind of book next"), and a
secondary signal (ALS score) ranks items within the predicted cluster. This
keeps the core idea -- generate a semantic ID, don't just search a fixed
index -- correctly implemented and fast enough to train and evaluate here,
while being upfront that it's a reduced-scale version of the full method.
"""
import json, time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

sys.path.append("/home/claude/recsys_project/src")
from eval_utils import evaluate_model
from phase1_baselines import load, build_seen_and_gt

MODELS_DIR = "/home/claude/recsys_project/models"
K_CLUSTERS = 256
MAX_LEN = 50
PAD = 0


def main():
    train, val, test, meta, item_meta = load()
    n_users, n_items = meta["n_users"], meta["n_items"]
    seen, gt, valid_users = build_seen_and_gt(train, test, n_users)

    # ---- Step 1: build semantic IDs by clustering item embeddings ----
    item_vecs = np.load(f"{MODELS_DIR}/two_tower_item_vecs.npy")  # reuse Phase 2 item tower output
    print(f"Clustering {n_items} items into {K_CLUSTERS} semantic tokens...")
    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=5)
    item_cluster = km.fit_predict(item_vecs)  # (n_items,) values in [0, K_CLUSTERS)
    cluster_sizes = pd.Series(item_cluster).value_counts()
    print(f"  cluster sizes: min={cluster_sizes.min()} max={cluster_sizes.max()} mean={cluster_sizes.mean():.1f}")

    # ---- Step 2: build proxy-ordered semantic-token sequences per user ----
    seqs_items = train.groupby("u", sort=False)["i"].apply(list).to_dict()
    seqs_tokens = {u: [item_cluster[i] + 1 for i in items] for u, items in seqs_items.items()}  # +1 for pad=0

    # ---- Step 3: tiny causal transformer over the semantic-token vocab (K+1) ----
    class SemanticIDTransformer(nn.Module):
        def __init__(self, vocab, d_model=48, n_heads=2, n_layers=2, max_len=MAX_LEN):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
            self.pos_emb = nn.Embedding(max_len, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
                                                dropout=0.2, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.out = nn.Linear(d_model, vocab)
            self.max_len = max_len

        def forward(self, seqs):
            B, L = seqs.shape
            pos = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, L)
            x = self.tok_emb(seqs) + self.pos_emb(pos)
            pad_mask = (seqs == PAD)
            causal = torch.triu(torch.ones(L, L, device=seqs.device), diagonal=1).bool()
            h = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
            return self.out(h)  # (B, L, vocab) -- vocab is small (257), full softmax is cheap now

    vocab = K_CLUSTERS + 1
    model = SemanticIDTransformer(vocab)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def make_batch(user_ids):
        B = len(user_ids)
        seq_batch = np.zeros((B, MAX_LEN), dtype=np.int64)
        target_batch = np.zeros((B, MAX_LEN), dtype=np.int64)
        for row, u in enumerate(user_ids):
            s = seqs_tokens.get(u, [])
            s = s[-(MAX_LEN+1):]
            if len(s) < 2:
                continue
            inp, tgt = s[:-1][-MAX_LEN:], s[1:][-MAX_LEN:]
            L = len(inp)
            seq_batch[row, -L:] = inp
            target_batch[row, -L:] = tgt
        return torch.tensor(seq_batch), torch.tensor(target_batch)

    all_users = np.array(list(seqs_tokens.keys()))
    print("Training semantic-ID transformer (full softmax over 257 tokens -- cheap)...")
    t0 = time.time()
    n_epochs, batch_size = 3, 256
    for epoch in range(n_epochs):
        perm = np.random.permutation(all_users)
        total, nb = 0.0, 0
        for start in range(0, len(perm), batch_size):
            batch_users = perm[start:start+batch_size]
            seq_in, tgt = make_batch(batch_users)
            logits = model(seq_in)  # (B, L, vocab) -- full softmax, vocab=257 so this is cheap
            loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        print(f"  epoch {epoch+1}/{n_epochs} loss={total/nb:.4f}")
    print(f"Trained in {time.time()-t0:.1f}s")

    # ---- Step 4: evaluate -- predict next semantic token, rank items within
    # that token's cluster using ALS as the secondary/fine-grained ranker ----
    als_item = np.load(f"{MODELS_DIR}/als_item_factors.npy")
    als_user = np.load(f"{MODELS_DIR}/als_user_factors.npy")

    rng = np.random.RandomState(42)
    all_eval_idx = np.where(valid_users)[0]
    eval_user_idx = rng.choice(all_eval_idx, size=min(5000, len(all_eval_idx)), replace=False)
    gt_eval = gt[eval_user_idx]
    seen_eval = [seen[u] for u in eval_user_idx]

    model.eval()
    n_eval = len(eval_user_idx)
    scores = np.zeros((n_eval, n_items), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_eval, 256):
            chunk = eval_user_idx[start:start+256]
            seq_in, _ = make_batch(chunk)
            logits = model(seq_in)
            last_logits = logits[:, -1, :]  # (B, vocab) - predicted distribution over next semantic token
            cluster_logprob = torch.log_softmax(last_logits, dim=-1).numpy()  # (B, K+1)
            # score(item) = P(its cluster | history) [dominant term] + small ALS tiebreak within cluster
            for row in range(len(chunk)):
                token_logprob_per_item = cluster_logprob[row, item_cluster + 1]  # (n_items,)
                als_tiebreak = 0.01 * (als_item @ als_user[chunk[row]])
                scores[start+row] = token_logprob_per_item + als_tiebreak

    item_pop = train.i.value_counts().reindex(range(n_items), fill_value=0)
    item_pop_rank = item_pop.rank(ascending=False, method="first").values - 1
    item_meta_sorted = item_meta.sort_values("i").reset_index(drop=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tag_features = TfidfVectorizer(max_features=2000).fit_transform(item_meta_sorted["tags_text"].fillna("")).toarray().astype(np.float32)

    results = {"semantic_id_generative": evaluate_model(scores, gt_eval, seen_eval, item_pop_rank, tag_features)}
    print("\n" + "="*70)
    df = pd.DataFrame(results).T
    print(df[["ndcg@10", "recall@10", "hit_rate@10", "coverage@10", "diversity@10", "pop_bias@10"]].round(4).to_string())
    json.dump(results, open(f"{MODELS_DIR}/phase6b_results.json", "w"), indent=2)
    np.save(f"{MODELS_DIR}/item_semantic_clusters.npy", item_cluster)

if __name__ == "__main__":
    main()
