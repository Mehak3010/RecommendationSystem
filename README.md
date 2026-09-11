# The Drawer — a book recommender built like a real product, on top of a real research pipeline

Two things live in this repo, and it's worth being explicit about the difference:

1. **A research pipeline**: nine recommendation models across four model families,
   trained and evaluated on [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)
   (53,424 users, 10,000 books, 5.9M ratings), under one shared evaluation harness
   so every number is directly comparable.
2. **The Drawer**: a working app built on top of that pipeline — search, genre
   filtering, "similar to this," real personalized picks from a trained ALS
   model, and a save-for-later shelf — not a demo form that calls a model once
   and shows a JSON blob.

This README is the blueprint for both halves: what each screen does, what's
calling what underneath it, how to reproduce every number, and what its real
limitations are.

## What you can actually do in the app

Open the app and you land on **Catalog & Recommend**, which has two modes:

- **Find similar titles** — search 10,000 books by title/author, filter by
  genre, and get "similar to this" recommendations computed live via TF-IDF
  text similarity over each book's title, author, and Goodreads tags. Not
  precomputed — every request scores the full catalog on the fly.
- **Personalized picks** — pull a real reader from the dataset (an actual
  user_id with actual rating history) and get recommendations scored by a
  trained ALS (implicit matrix factorization) model, optionally filtered to
  one genre. This is genuine collaborative filtering, not a lookup table.

Every book card shows real signal pulled straight from the dataset: cover
art, star rating, a five-segment rating-distribution bar (how many 1★ vs 5★
ratings it actually has), and genre tags built from real Goodreads community
tags — not a hand-picked genre field, since the dataset doesn't have one.

**Your Shelf** — hover any cover and click the bookmark icon to save a book.
Saved books persist in the browser (localStorage) and get refreshed with
higher-resolution cover art in the background when you open the shelf. This
is intentionally client-side: no login system, no backend write path, no user
table — a save action gets you a save, not an account.

**About this project** — tucked away as a quiet footer link rather than a
peer nav item, because it's for a different visitor than the rest of the app:
the model leaderboard, per-model metric breakdowns, the architecture map, and
the honest-limitations writeup below, for anyone (an interviewer, you)
evaluating the engineering rather than looking for a book.

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

The app's "Personalized picks" mode runs on **ALS** specifically (row 3
above) — not the top-ranked SASRec model — because ALS's item/user factor
vectors are cheap to serve at request time (one dot product per candidate
item), while SASRec needs the user's actual interaction sequence as input,
which the app doesn't collect. That trade-off — highest offline accuracy vs.
what's actually servable live — is itself a real production decision, and
part of why both numbers are worth knowing.

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
Phase 7  The Drawer (this app)  -> Dashboard/dashboard.py + Dashboard/static/index.html
                                    (FastAPI + vanilla JS. Catalog search, live TF-IDF similarity,
                                    genre filtering, real ALS-backed personalized picks, and a
                                    client-side shelf -- all served from real project artifacts,
                                    nothing mocked.)
```

Shared evaluation code (`src/eval_utils.py`) is used by every phase so all
numbers above are directly comparable — same metric implementations, same
masking of already-seen items, same eval users.

## API reference (Dashboard/dashboard.py)

Everything the app's UI calls is a plain JSON endpoint, so it's also usable
directly — `curl` any of these against a running dashboard:

| Endpoint | What it does |
|---|---|
| `GET /api/stats` | Reader/title/model counts shown in the top bar |
| `GET /api/leaderboard` | Full model comparison table (backs the "About" leaderboard) |
| `GET /api/model/{model_key}` | Per-model metric breakdown at k=5/10/20 |
| `GET /api/genres` | Genre buckets derived from real Goodreads tags, with live counts |
| `GET /api/genres/{genre_key}/debug` | Sanity-check view: which raw tags a genre bucket matches |
| `GET /api/books?search=&genre=&limit=&offset=` | Catalog search/browse, composable with genre filter |
| `GET /api/book/{i}` | Single book's full card data |
| `GET /api/recommend/{i}?k=` | TF-IDF "similar to this" for a given book |
| `GET /api/random_reader` | Pulls a real user_id + their rated-highly shelf |
| `GET /api/recommend_for_reader/{user_id}?k=&genre=` | ALS-backed personalized picks, optionally genre-filtered |
| `GET /api/architecture` | The phase table above, as JSON |
| `GET /api/limitations` | This README's "Honest limitations" section + compute tradeoffs report |
| `GET /api/charts` | Paths to the static leaderboard/coverage charts |

`recommend_for_reader` and the genre-filtered variant both 503 gracefully if
`models/als_item_factors.npy` / `als_user_factors.npy` aren't present —
see "Try it" below for how to generate them.

## Honest limitations

1. **No real timestamps.** goodbooks-10k has no interaction timestamps, so the "sequence" used for SASRec and the leave-one-out split is each user's row order in the raw file, not true chronology. The architectures are implemented correctly; the specific predictions shouldn't be read as reflecting genuine reading order. In a real job, this is exactly the kind of data-quality issue you'd flag to a PM before shipping.
2. **Evaluation is on a sampled 5,000-user subset** of the ~53k eligible users, for memory reasons (dense scoring matrices at full scale need tens of GB). Metrics are stable at this sample size but a production eval would use the full population or a larger stratified sample.
3. **Semantic-ID model is single-level**, not full residual quantization (RQ-VAE) as in the TIGER paper — a simplification made so scoring stays tractable on a CPU-only sandbox; documented in `phase6b`'s docstring.
4. **LightGCN is compute-capped** (see above).
5. **"Similar to this" is text similarity, not collaborative filtering.** It matches on title/author/tag overlap (TF-IDF + cosine similarity), so it will confidently recommend a book with similar-sounding tags even if no reader who liked the seed book ever rated it. This is a genuinely different, weaker signal than the ALS-backed personalized picks, and the two shouldn't be confused for the same kind of "recommendation."
6. **Your Shelf has no backend.** Saved books live in browser localStorage only — clearing browser data loses the shelf, and it doesn't sync across devices. This is a deliberate scope decision (no auth system, no user table), not an oversight, but worth being upfront about if asked how it'd need to change for production.

## Try it

goodbooks-10k's CSVs aren't committed to this repo (they're ~90MB combined).
Pull them from the dataset's own GitHub mirror — no Kaggle account needed:

```bash
mkdir -p data && cd data
curl -sL -o ratings.csv   https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv
curl -sL -o books.csv     https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv
curl -sL -o tags.csv      https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv
curl -sL -o book_tags.csv https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv
cd ..
```

Then run the pipeline in order:

```bash
# 1. Data prep -- also carries cover art + rating-distribution columns
#    from books.csv into data/processed/item_meta.csv for the app to use
python3 src/phase0_data_prep.py

# 2. Train + evaluate each model family
python3 src/phase1_baselines.py      # also produces models/als_*.npy, which the
                                      # app's personalized-picks mode needs
python3 src/phase2_deep.py
python3 src/phase2b_sequential.py
python3 src/phase6a_lightgcn.py
python3 src/phase6b_generative_retrieval.py

# 3. Consolidated leaderboard + cold-start analysis
python3 src/phase3_compare.py

# 4. Optional: standalone FAISS+ALS serving API (separate from the app below)
cd api && uvicorn serve:app --reload
curl "http://localhost:8000/recommend/42?k=5"
```

## Running the app

```bash
cd Dashboard
uvicorn dashboard:app --host 0.0.0.0 --port 8010 --reload
```

Then open `http://localhost:8010/`. Requires `data/processed/` to exist (Phase
0 above); personalized picks additionally require `models/als_item_factors.npy`
and `models/als_user_factors.npy` (from `phase1_baselines.py`) — without them
the app still runs, that one feature just degrades to a clear "not available"
message instead of erroring.

A `Dashboard/Dockerfile` is included for deploying the app itself (e.g. to
Hugging Face Spaces or Render) without needing the full training pipeline —
it only needs `data/processed/`, `models/leaderboard.csv`,
`models/*_results.json`, and the two ALS `.npy` files.

## Tech stack

FastAPI + vanilla JS/HTML/CSS (no frontend framework — deliberately, to keep
the app's own dependency surface small). scikit-learn, implicit, PyTorch, and
FAISS for the model side. Pandas/NumPy throughout. No database — everything
is served from flat files (`data/processed/`, `models/`) produced by the
pipeline above.
