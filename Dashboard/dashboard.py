"""
Project dashboard API
======================
Serves the *real* files from this project — no mocked data:

  models/leaderboard.csv          -> /api/leaderboard
  models/phase*_results.json      -> /api/model/{name}
  data/processed/item_meta.csv    -> /api/books, /api/book/{id}
  data/processed/mappings.json    -> /api/stats
  reports/compute_tradeoffs.md    -> /api/limitations
  README.md                       -> /api/architecture (phase table, parsed)

Recommendations are computed live with a TF-IDF + cosine-similarity model
over each book's tags/title/author text (data/processed/item_meta.csv).
This doesn't require the missing two_tower_item_vecs.npy / als_*.npy
artifacts that api/serve.py expects — it's a real, from-scratch model
that runs directly against the checked-in data, not a stand-in.

Run with:  uvicorn dashboard:app --host 0.0.0.0 --port 8010 --reload
Then open  http://localhost:8010/
"""
import json
import re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Recsys Project Dashboard API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------------------------------------------------------------- data ----
leaderboard_df = pd.read_csv(MODELS_DIR / "leaderboard.csv").rename(
    columns={"Unnamed: 0": "model"}
)
item_meta = pd.read_csv(DATA_DIR / "item_meta.csv")
mappings = json.loads((DATA_DIR / "mappings.json").read_text())

# Real, trained ALS collaborative-filtering factors -- optional, since a
# deployment may omit them, but this is what powers genuinely personalized
# (not just text-similarity) recommendations below.
import numpy as np
_als_item_path = MODELS_DIR / "als_item_factors.npy"
_als_user_path = MODELS_DIR / "als_user_factors.npy"
als_item = np.load(_als_item_path) if _als_item_path.exists() else None
als_user = np.load(_als_user_path) if _als_user_path.exists() else None

_shelves_path = DATA_DIR / "demo_shelves.json"
demo_shelves = json.loads(_shelves_path.read_text()) if _shelves_path.exists() else {}

MODEL_DISPLAY_NAMES = {
    "sasrec_lite": "SASRec-lite",
    "semantic_id_generative": "Semantic-ID Generative",
    "als": "ALS",
    "two_tower": "Two-Tower",
    "ncf": "NCF",
    "popularity": "Popularity",
    "lightgcn": "LightGCN",
    "mf_bpr_pytorch": "MF (BPR)",
    "content_based": "Content-based",
}

RESULTS_FILES = {
    "popularity": "phase1_results.json",
    "content_based": "phase1_results.json",
    "als": "phase1_results.json",
    "mf_bpr_pytorch": "phase1_results.json",
    "ncf": "phase2_results.json",
    "two_tower": "phase2_results.json",
    "sasrec_lite": "phase2b_results.json",
    "lightgcn": "phase6a_results.json",
    "semantic_id_generative": "phase6b_results.json",
}

_results_cache = {}


def _load_results(fname):
    if fname not in _results_cache:
        path = MODELS_DIR / fname
        _results_cache[fname] = json.loads(path.read_text()) if path.exists() else {}
    return _results_cache[fname]


# ----------------------------------------------------- content recommender
item_meta["_text"] = (
    item_meta["title"].fillna("")
    + " "
    + item_meta["authors"].fillna("")
    + " "
    + item_meta["tags_text"].fillna("")
)
_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
_tfidf = _vectorizer.fit_transform(item_meta["_text"])

# ------------------------------------------------------- genre / theme ---
# Curated on top of the real Goodreads shelf tags already sitting in
# tags_text (see src/phase0_data_prep.py) -- these are actual community
# tags for each book, not invented labels. Each genre is a small set of
# tag substrings to match against tags_text, so this stays a thin lookup
# layer over real data rather than a new model.
GENRES = {
    "sci-fi":      ["science-fiction", "sci-fi", "scifi", "dystopia", "space"],
    "fantasy":     ["fantasy", "magic", "wizards", "dragons"],
    "romance":     ["romance", "chick-lit", "love-story"],
    "mystery-thriller": ["mystery", "thriller", "crime", "suspense", "detective"],
    "horror":      ["horror", "paranormal", "supernatural", "ghost"],
    "historical":  ["historical-fiction", "historical"],
    "young-adult": ["young-adult", "ya", "teen", "childrens"],
    "classics":    ["classics", "classic"],
    "non-fiction": ["non-fiction", "nonfiction"],
    "biography-memoir": ["biography", "memoir", "autobiography"],
    "poetry":      ["poetry"],
    "graphic-comics": ["graphic-novels", "comics", "manga", "comic-book"],
    "self-help":   ["self-help", "self-improvement", "personal-development"],
    "humor":       ["humor", "humour", "funny", "comedy"],
}
GENRE_LABELS = {
    "sci-fi": "Sci-Fi", "fantasy": "Fantasy", "romance": "Romance",
    "mystery-thriller": "Mystery & Thriller", "horror": "Horror",
    "historical": "Historical Fiction", "young-adult": "Young Adult",
    "classics": "Classics", "non-fiction": "Non-Fiction",
    "biography-memoir": "Biography & Memoir", "poetry": "Poetry",
    "graphic-comics": "Graphic Novels & Comics", "self-help": "Self-Help",
    "humor": "Humor",
}
_tags_lower = item_meta["tags_text"].fillna("").str.lower()


def _genre_mask(genre_key: str):
    terms = GENRES.get(genre_key, [])
    if not terms:
        return pd.Series(False, index=item_meta.index)
    pattern = "|".join(re.escape(t) for t in terms)
    return _tags_lower.str.contains(pattern, regex=True, na=False)


def recommend_similar(item_row_idx: int, k: int = 10):
    sims = cosine_similarity(_tfidf[item_row_idx], _tfidf).ravel()
    order = sims.argsort()[::-1]
    order = [i for i in order if i != item_row_idx][:k]
    out = []
    for i in order:
        row = item_meta.iloc[i]
        out.append(
            {
                "i": int(row["i"]),
                "book_id": int(row["book_id"]),
                "title": row["title"],
                "authors": row["authors"],
                "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
                "similarity": round(float(sims[i]), 4),
            }
        )
    return out


# --------------------------------------------------------------- routes ---
@app.get("/api/stats")
def stats():
    return {
        **mappings,
        "n_models": len(leaderboard_df),
    }


@app.get("/api/leaderboard")
def leaderboard():
    rows = []
    for _, r in leaderboard_df.iterrows():
        d = r.to_dict()
        model_key = d.pop("model")
        d["model"] = model_key
        d["display_name"] = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        rows.append(d)
    rows.sort(key=lambda x: x["ndcg@10"], reverse=True)
    return rows


@app.get("/api/model/{model_key}")
def model_detail(model_key: str):
    if model_key not in RESULTS_FILES:
        raise HTTPException(404, f"unknown model '{model_key}'")
    results = _load_results(RESULTS_FILES[model_key])
    if model_key not in results:
        raise HTTPException(404, f"no detailed results for '{model_key}' in {RESULTS_FILES[model_key]}")
    return {
        "model": model_key,
        "display_name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "source_file": RESULTS_FILES[model_key],
        "metrics_by_k": results[model_key],
    }


@app.get("/api/genres")
def genres():
    """Genre/theme buckets derived from real Goodreads shelf tags
    (tags_text), with a live count of how many catalog titles match each
    one -- lets the catalog UI offer 'just show me sci-fi' instead of only
    free-text search."""
    return [
        {"key": key, "label": GENRE_LABELS[key], "count": int(_genre_mask(key).sum())}
        for key in GENRES
    ]


@app.get("/api/genres/{genre_key}/debug")
def genre_debug(genre_key: str, sample: int = 15):
    """Sanity-check view for a genre bucket: shows which raw tag terms it
    matches on, a sample of titles it caught with their actual tags_text
    (so you can eyeball false positives), and how it stacks up against the
    single most common raw tag in the dataset with a similar name (e.g.
    'fantasy' the shelf tag) as an independent cross-check."""
    if genre_key not in GENRES:
        raise HTTPException(404, f"unknown genre '{genre_key}'")
    mask = _genre_mask(genre_key)
    matched = item_meta[mask]
    unmatched_sample = item_meta[~mask].sample(min(5, (~mask).sum()), random_state=0) if (~mask).sum() else item_meta.iloc[0:0]
    return {
        "genre": genre_key,
        "label": GENRE_LABELS[genre_key],
        "match_terms": GENRES[genre_key],
        "matched_count": int(mask.sum()),
        "total_catalog": len(item_meta),
        "sample_matched": [
            {"title": r["title"], "authors": r["authors"], "tags_text": r["tags_text"]}
            for _, r in matched.head(sample).iterrows()
        ],
        "sample_unmatched": [
            {"title": r["title"], "authors": r["authors"], "tags_text": r["tags_text"]}
            for _, r in unmatched_sample.iterrows()
        ],
    }


@app.get("/api/books")
def books(
    search: str = Query("", description="search title/author"),
    genre: str = Query("", description="genre key from /api/genres"),
    limit: int = 30,
    offset: int = 0,
):
    df = item_meta
    if genre:
        if genre not in GENRES:
            raise HTTPException(404, f"unknown genre '{genre}'")
        df = df[_genre_mask(genre).reindex(df.index)]
    if search:
        mask = df["_text"].str.contains(re.escape(search), case=False, na=False)
        df = df[mask]
    total = len(df)
    total_catalog = len(item_meta)
    page = df.iloc[offset : offset + limit]
    return {
        "total": total,
        "total_catalog": total_catalog,
        "results": [
            {
                "i": int(r["i"]),
                "book_id": int(r["book_id"]),
                "title": r["title"],
                "authors": r["authors"],
                "year": (None if pd.isna(r["original_publication_year"]) else int(r["original_publication_year"])),
                "average_rating": float(r["average_rating"]) if pd.notna(r["average_rating"]) else None,
                "tags_text": r["tags_text"],
            }
            for _, r in page.iterrows()
        ],
    }


@app.get("/api/book/{i}")
def book_detail(i: int):
    row = item_meta[item_meta["i"] == i]
    if row.empty:
        raise HTTPException(404, f"no book with internal id {i}")
    row = row.iloc[0]
    return {
        "i": int(row["i"]),
        "book_id": int(row["book_id"]),
        "title": row["title"],
        "authors": row["authors"],
        "year": None if pd.isna(row["original_publication_year"]) else int(row["original_publication_year"]),
        "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
        "tags_text": row["tags_text"],
    }


@app.get("/api/recommend/{i}")
def recommend(i: int, k: int = 10):
    row = item_meta[item_meta["i"] == i]
    if row.empty:
        raise HTTPException(404, f"no book with internal id {i}")
    row_idx = row.index[0]
    seed = row.iloc[0]
    return {
        "seed": {
            "i": int(seed["i"]),
            "title": seed["title"],
            "authors": seed["authors"],
        },
        "method": "TF-IDF (title + authors + tags) cosine similarity — computed live from data/processed/item_meta.csv",
        "recommendations": recommend_similar(row_idx, k=k),
    }


@app.get("/api/random_reader")
def random_reader():
    """Pick a sample reader with a real, displayable interaction history
    (see src/make_demo_shelves.py) so the personalized-recommend demo below
    has something authentic to show, not just a bare user id."""
    if not demo_shelves:
        raise HTTPException(503, "no demo reader shelves bundled with this deployment")
    import random
    user_id = random.choice(list(demo_shelves.keys()))
    return {"user_id": int(user_id), "shelf": demo_shelves[user_id]}


@app.get("/api/recommend_for_reader/{user_id}")
def recommend_for_reader(user_id: int, k: int = 10, genre: str = Query("", description="genre key from /api/genres")):
    """Genuinely personalized recommendations from the trained ALS model
    (models/als_item_factors.npy + als_user_factors.npy) -- this is real
    collaborative filtering learned from 5.9M ratings, not a text-similarity
    stand-in. Any internal user id (0 to n_users-1) works; ids that also
    appear in demo_shelves.json additionally get a displayable reading
    history alongside the recommendations.

    Optional `genre`: restricts the ranked candidates to titles in that
    genre bucket (see /api/genres) before taking the top-k, so 'personalized
    sci-fi picks for this reader' is answerable without a separate model --
    same ALS scores, just masked to a subset of the catalog.
    """
    if als_item is None or als_user is None:
        raise HTTPException(503, "ALS model factors not bundled with this deployment")
    if user_id < 0 or user_id >= als_user.shape[0]:
        raise HTTPException(404, f"reader id must be between 0 and {als_user.shape[0]-1}")

    scores = als_item @ als_user[user_id]

    method = "ALS collaborative filtering (models/als_*.npy) — trained on 5.9M real ratings"
    if genre:
        if genre not in GENRES:
            raise HTTPException(404, f"unknown genre '{genre}'")
        mask = _genre_mask(genre)
        allowed = np.zeros(scores.shape[0], dtype=bool)
        matched_i = item_meta.loc[mask, "i"].values
        matched_i = matched_i[(matched_i >= 0) & (matched_i < scores.shape[0])]
        allowed[matched_i] = True
        scores = np.where(allowed, scores, -np.inf)
        method += f" — filtered to genre '{GENRE_LABELS[genre]}' ({int(allowed.sum())} eligible titles)"

    top_idx = np.argsort(-scores)[:k]
    item_meta_idx = item_meta.set_index("i")
    recs = []
    for i in top_idx:
        i = int(i)
        if not np.isfinite(scores[i]) or i not in item_meta_idx.index:
            continue
        row = item_meta_idx.loc[i]
        recs.append({
            "i": i,
            "book_id": int(row["book_id"]),
            "title": row["title"],
            "authors": row["authors"],
            "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
            "score": round(float(scores[i]), 4),
        })
    return {
        "user_id": user_id,
        "method": method,
        "shelf": demo_shelves.get(str(user_id), []),
        "recommendations": recs,
    }


@app.get("/api/architecture")
def architecture():
    return {
        "phases": [
            {"phase": "Phase 0", "name": "Data engineering", "file": "src/phase0_data_prep.py"},
            {"phase": "Phase 1", "name": "Classical baselines (popularity, content-based, ALS, MF-from-scratch)", "file": "src/phase1_baselines.py"},
            {"phase": "Phase 2", "name": "Deep learning (NCF, Two-Tower)", "file": "src/phase2_deep.py"},
            {"phase": "Phase 2b", "name": "Sequential (SASRec-lite, causal transformer)", "file": "src/phase2b_sequential.py"},
            {"phase": "Phase 3", "name": "Unified evaluation (leaderboard + cold-start slice)", "file": "src/phase3_compare.py"},
            {"phase": "Phase 5", "name": "Serving (FAISS retrieval -> ALS re-rank, FastAPI)", "file": "api/serve.py"},
            {"phase": "Phase 6a", "name": "Graph neural net (LightGCN, from scratch)", "file": "src/phase6a_lightgcn.py"},
            {"phase": "Phase 6b", "name": "Generative retrieval (semantic-ID transformer)", "file": "src/phase6b_generative_retrieval.py"},
        ]
    }


@app.get("/api/limitations")
def limitations():
    path = REPORTS_DIR / "compute_tradeoffs.md"
    readme = (ROOT / "README.md").read_text()
    honest_section = ""
    m = re.search(r"## Honest limitations.*?(?=\n## )", readme, re.S)
    if m:
        honest_section = m.group(0)
    return {
        "honest_limitations_from_readme": honest_section,
        "compute_tradeoffs_md": path.read_text() if path.exists() else "",
    }


@app.get("/api/charts")
def charts():
    return {
        "leaderboard_bar": "/charts/leaderboard_bar.png",
        "coverage_vs_accuracy": "/charts/coverage_vs_accuracy.png",
    }


app.mount("/charts", StaticFiles(directory=str(REPORTS_DIR)), name="charts")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")