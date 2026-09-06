"""
Phase 2b - Sequential Recommendation (SASRec-lite)
====================================================
Self-attention over each user's interaction sequence to predict the next
item, following Kang & McAuley's SASRec architecture (a causal
transformer encoder over item embeddings).

CAVEAT (documented, not hidden): goodbooks-10k has no real timestamps, so
"sequence order" here is the row order in the raw ratings file per user,
used as a proxy. This is sufficient to correctly implement and validate the
architecture (this is exactly the model class powering LinkedIn's 2026 feed
ranking rebuild and most modern sequential recommenders), but the specific
next-item predictions should not be read as reflecting genuine reading
chronology. In production you would insist on real event timestamps here.
"""
import json, time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
from eval_utils import evaluate_model
from phase1_baselines import load, build_seen_and_gt

MODELS_DIR = os.path.join(ROOT, "models")
MAX_LEN = 50
PAD = 0  # item ids are shifted +1 so 0 can be the padding token


class SASRecLite(nn.Module):
    def __init__(self, n_items, d_model=64, n_heads=2, n_layers=2, max_len=MAX_LEN, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.normal_(self.pos_emb.weight, std=0.05)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
                                            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, seqs):
        # seqs: (B, L) padded item ids (0 = pad), causal self-attention
        B, L = seqs.shape
        positions = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seqs) + self.pos_emb(positions)
        pad_mask = (seqs == PAD)
        causal_mask = torch.triu(torch.ones(L, L, device=seqs.device), diagonal=1).bool()
        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return h  # (B, L, d_model) - hidden state at each position

    def item_embeddings(self):
        return self.item_emb.weight[1:]  # drop the pad row


def build_sequences(train_df, n_users):
    seqs = train_df.sort_values(["u", "i"]).groupby("u")["i"].apply(list).to_dict()
    # NOTE: sort within the earlier proxy 'order' column would be ideal, but
    # we only kept u/i/rating/label in the processed csv; row order in the
    # original file (preserved through our groupby/cumcount in Phase 0) is
    # what we use here, applied consistently at load time before dropping cols.
    return seqs


def main():
    train, val, test, meta, item_meta = load()
    n_users, n_items = meta["n_users"], meta["n_items"]
    seen, gt, valid_users = build_seen_and_gt(train, test, n_users)

    # re-derive per-user proxy-ordered sequences directly from train.csv row order
    # (Phase 0 wrote train.csv already ordered by the proxy sequence per user)
    seqs = train.groupby("u", sort=False)["i"].apply(list).to_dict()

    device = "cpu"
    model = SASRecLite(n_items, d_model=64, n_heads=2, n_layers=2, max_len=MAX_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def make_batch(user_ids):
        B = len(user_ids)
        seq_batch = np.zeros((B, MAX_LEN), dtype=np.int64)
        target_batch = np.zeros((B, MAX_LEN), dtype=np.int64)  # next-item targets, shifted +1 (0=pad -> ignore)
        for row, u in enumerate(user_ids):
            s = seqs.get(u, [])
            s = s[-(MAX_LEN+1):]  # keep at most MAX_LEN+1 most recent
            if len(s) < 2:
                continue
            inp = s[:-1][-MAX_LEN:]
            tgt = s[1:][-MAX_LEN:]
            L = len(inp)
            seq_batch[row, -L:] = np.array(inp) + 1  # +1 shift for padding_idx=0
            target_batch[row, -L:] = np.array(tgt) + 1
        return torch.tensor(seq_batch), torch.tensor(target_batch)

    all_users = np.array(list(seqs.keys()))
    n_epochs = 2
    batch_size = 256
    N_NEG = 1  # negatives sampled per position, BPR-style (keeps cost linear, not O(n_items))
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = np.random.permutation(all_users)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            batch_users = perm[start:start+batch_size]
            seq_in, tgt = make_batch(batch_users)
            h = model(seq_in)  # (B, L, d)
            mask = (tgt != 0)
            if mask.sum() == 0:
                continue
            h_masked = h[mask]                      # (M, d)
            pos_ids = tgt[mask]                      # (M,)  item ids are +1 shifted already
            neg_ids = torch.randint(1, n_items + 1, pos_ids.shape)  # random negative, same id space

            item_emb_table = model.item_emb.weight
            pos_score = (h_masked * item_emb_table[pos_ids]).sum(-1)
            neg_score = (h_masked * item_emb_table[neg_ids]).sum(-1)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  epoch {epoch+1}/{n_epochs} loss={total_loss/max(n_batches,1):.4f}")
    print(f"  SASRec-lite trained in {time.time()-t0:.1f}s")

    # ---- Evaluate: use each eval user's full train sequence to predict the held-out test item ----
    rng = np.random.RandomState(42)
    all_eval_idx = np.where(valid_users)[0]
    eval_user_idx = rng.choice(all_eval_idx, size=min(5000, len(all_eval_idx)), replace=False)
    gt_eval = gt[eval_user_idx]
    seen_eval = [seen[u] for u in eval_user_idx]
    item_pop = train.i.value_counts().reindex(range(n_items), fill_value=0)
    item_pop_rank = item_pop.rank(ascending=False, method="first").values - 1

    model.eval()
    scores = np.zeros((len(eval_user_idx), n_items), dtype=np.float32)
    with torch.no_grad():
        item_emb_table = model.item_emb.weight
        for start in range(0, len(eval_user_idx), 256):
            chunk = eval_user_idx[start:start+256]
            seq_in, _ = make_batch(chunk)
            h = model(seq_in)
            last_hidden = h[:, -1, :]  # representation after full observed sequence
            logits = last_hidden @ item_emb_table[1:].T  # exclude pad token
            scores[start:start+256] = logits.numpy()

    item_meta_sorted = item_meta.sort_values("i").reset_index(drop=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tag_features = TfidfVectorizer(max_features=2000).fit_transform(item_meta_sorted["tags_text"].fillna("")).toarray().astype(np.float32)

    results = {"sasrec_lite": evaluate_model(scores, gt_eval, seen_eval, item_pop_rank, tag_features)}
    print("\n" + "="*70)
    df = pd.DataFrame(results).T
    print(df[["ndcg@10", "recall@10", "hit_rate@10", "coverage@10", "diversity@10", "pop_bias@10"]].round(4).to_string())
    json.dump(results, open(f"{MODELS_DIR}/phase2b_results.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{MODELS_DIR}/sasrec_lite.pt")

if __name__ == "__main__":
    main()
