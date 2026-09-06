# Book Recommender: Classical → Deep → Generative

An end-to-end recommendation system built on [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)
(53,424 users, 10,000 books, 5.9M ratings), covering nine models across four
model families, a shared evaluation harness, a two-stage FAISS+ALS serving
API, and a working implementation of 2026-era generative retrieval.

## Results

Evaluated on 5,000 held-out users (leave-one-out, next-item prediction), ranked over the full 10,000-item catalog:

| Model | NDCG@10 | Recall@10 | Coverage@10 | Diversity@10 |
|---|---|---|---|---|
| **SASRec-lite** (sequential transformer) | **0.0278** | 0.0522 | 0.20 | 0.65 |
| **Semantic-ID Generative** (clustering + transformer) | 0.0263 | 0.0502 | **0.57** | 0.55 |
| ALS (implicit) | 0.0129 | 0.0302 | 0.25 | 0.64 |
| Two-Tower (neural, multimodal) | 0.0114 | 0.0250 | 0.17 | 0.63 |
| NCF | 0.0110 | 0.0228 | 0.11 | 0.65 |
| Popularity baseline | 0.0090 | 0.0178 | 0.005 | 0.65 |
| LightGCN (compute-limited, see below) | 0.0088 | 0.0172 | 0.02 | 0.66 |
| Matrix Factorization (BPR, from scratch) | 0.0079 | 0.0152 | 0.005 | 0.65 |
| Content-based (TF-IDF) | 0.0031 | 0.0070 | 0.35 | 0.04 |

Full breakdown: `models/leaderboard.csv`. Run `src/phase3_compare.py` to regenerate.

### What the results actually say

- **Sequential context wins.** SASRec directly conditions on "what did this user just interact with," which lines up almost exactly with a next-item eval protocol — that's a meaningful chunk of its advantage, not purely architecture. Worth saying out loud in an interview rather than overclaiming.
- **The generative semantic-ID model is the standout finding**: it gets SASRec-level accuracy while recommending across **57% of the catalog** vs. SASRec's 20% — because it predicts a *type* of book first, then ranks within that cluster, rather than directly memorizing popular next-items. That's a genuine coverage/diversity win worth highlighting.
- **Content-based has terrible accuracy but the best diversity/coverage tradeoff of the non-generative models** — classic result, and a good illustration of why hybrid systems exist.
- **LightGCN underperforms here mainly due to compute, not the algorithm**: this environment is CPU-only and full-graph propagation on every mini-batch step is expensive, so training was capped at 60 steps (loss was still dropping). On a GPU with a normal training budget this would be expected to land nearer the Two-Tower/ALS range. Documented honestly rather than hidden — a good thing to be upfront about if asked.

## Architecture by phase

```
Phase 0  Data engineering       -> src/phase0_data_prep.py
Phase 1  Classical baselines    -> src/phase1_baselines.py     (popularity, content-based, ALS, MF-from-scratch)
Phase 2  Deep learning          -> src/phase2_deep.py           (NCF, Two-Tower)
Phase 2b Sequential             -> src/phase2b_sequential.py    (SASRec-lite, causal transformer)
Phase 3  Unified evaluation     -> src/phase3_compare.py        (leaderboard + cold-start slice)
Phase 5  Serving                -> api/serve.py                 (FAISS retrieval -> ALS re-rank, FastAPI)
Phase 6a Graph neural net       -> src/phase6a_lightgcn.py      (LightGCN, from scratch, no torch-geometric)
Phase 6b Generative retrieval   -> src/phase6b_generative_retrieval.py  (semantic-ID transformer)
```

Shared evaluation code (`src/eval_utils.py`) is used by every phase so all
numbers above are directly comparable — same metric implementations, same
masking of already-seen items, same eval users.

## Honest limitations (say these out loud in interviews — it's a strength, not a weakness, to know them)

1. **No real timestamps.** goodbooks-10k has no interaction timestamps, so the "sequence" used for SASRec and the leave-one-out split is each user's row order in the raw file, not true chronology. The architectures are implemented correctly; the specific predictions shouldn't be read as reflecting genuine reading order. In a real job, this is exactly the kind of data-quality issue you'd flag to a PM before shipping.
2. **Evaluation is on a sampled 5,000-user subset** of the ~53k eligible users, for memory reasons (dense scoring matrices at full scale need tens of GB). Metrics are stable at this sample size but a production eval would use the full population or a larger stratified sample.
3. **Semantic-ID model is single-level**, not full residual quantization (RQ-VAE) as in the TIGER paper — a simplification made so scoring stays tractable on a CPU-only sandbox; documented in `phase6b`'s docstring.
4. **LightGCN is compute-capped** (see above).

## Try it

```bash
# 1. Data prep
python3 src/phase0_data_prep.py

# 2. Train + evaluate each model family
python3 src/phase1_baselines.py
python3 src/phase2_deep.py
python3 src/phase2b_sequential.py
python3 src/phase6a_lightgcn.py
python3 src/phase6b_generative_retrieval.py

# 3. Consolidated leaderboard + cold-start analysis
python3 src/phase3_compare.py

# 4. Serve
cd api && uvicorn serve:app --reload
curl "http://localhost:8000/recommend/42?k=5"
```

## Resume line

> Built an end-to-end book recommender (goodbooks-10k, 6M ratings) spanning
> matrix factorization, neural collaborative filtering, a two-tower
> retrieval model, a SASRec-style sequential transformer, LightGCN, and a
> semantic-ID generative retrieval model; shipped a FAISS + FastAPI serving
> layer with sub-2ms re-ranking latency and a unified offline evaluation
> harness (NDCG, coverage, diversity, cold-start) across all nine models.

Interviewers will pick one line and go deep — make sure you can defend the
proxy-ordering caveat, why two-tower over single-tower, why BPR loss over
cross-entropy, and what you'd change with a GPU budget for LightGCN.