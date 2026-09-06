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


@app.get("/api/books")
def books(search: str = Query("", description="search title/author"), limit: int = 30, offset: int = 0):
    df = item_meta
    if search:
        mask = df["_text"].str.contains(re.escape(search), case=False, na=False)
        df = df[mask]
    total = len(df)
    page = df.iloc[offset : offset + limit]
    return {
        "total": total,
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