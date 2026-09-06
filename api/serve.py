"""
Phase 5 - Serving
==================
Two-stage architecture (the industry-standard pattern):
  Stage 1 (Retrieval): FAISS approximate nearest-neighbor search over the
    Two-Tower item embeddings to pull ~200 candidates in milliseconds from
    a catalog that could be millions of items (exhaustive scoring doesn't
    scale; ANN does).
  Stage 2 (Re-ranking): the ALS model's item factors are used to re-score
    just those 200 candidates against the user's ALS vector, since a more
    expensive/accurate model is affordable on a small candidate set but not
    on the full catalog.

Run with:  uvicorn serve:app --host 0.0.0.0 --port 8000  (from this directory)
"""
import numpy as np
import faiss
import json
import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data", "processed")

app = FastAPI(title="Book Recommender API", version="0.1")

# ---- Load artifacts once at startup ----
item_vecs = np.load(f"{MODELS_DIR}/two_tower_item_vecs.npy").astype("float32")
als_item = np.load(f"{MODELS_DIR}/als_item_factors.npy").astype("float32")
als_user = np.load(f"{MODELS_DIR}/als_user_factors.npy").astype("float32")

import pandas as pd
item_meta = pd.read_csv(f"{DATA_DIR}/item_meta.csv").sort_values("i").reset_index(drop=True)
train = pd.read_csv(f"{DATA_DIR}/train.csv")
user_seen = train.groupby("u")["i"].apply(set).to_dict()

d = item_vecs.shape[1]
index = faiss.IndexFlatIP(d)   # exact inner-product search here; swap for
                                # IndexIVFFlat/IndexHNSWFlat for million-scale catalogs
index.add(item_vecs)


class RecommendResponse(BaseModel):
    user_id: int
    candidates_retrieved: int
    latency_ms: float
    recommendations: list


@app.get("/health")
def health():
    return {"status": "ok", "n_items": item_vecs.shape[0], "n_users": als_user.shape[0]}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, k: int = 10, retrieve_k: int = 200):
    if user_id < 0 or user_id >= als_user.shape[0]:
        raise HTTPException(status_code=404, detail="unknown user_id (must be an internal reindexed id from Phase 0)")

    t0 = time.time()

    # Stage 1: retrieval. We don't have a user tower query vector cached
    # per user here, so we approximate the query as the ALS user vector
    # projected into the two-tower item space via a cheap similarity proxy:
    # in a real system you'd cache each user's two-tower query vector and
    # search with it directly. For this demo we retrieve using the item
    # index seeded from the user's own liked items' centroid.
    liked = list(user_seen.get(user_id, []))[:20]
    if liked:
        query = item_vecs[liked].mean(axis=0, keepdims=True)
    else:
        query = item_vecs.mean(axis=0, keepdims=True)  # cold-start fallback: catalog centroid
    _, candidate_ids = index.search(query, retrieve_k)
    candidate_ids = candidate_ids[0]

    # Stage 2: re-rank candidates with ALS
    seen = user_seen.get(user_id, set())
    candidate_ids = [c for c in candidate_ids if c not in seen]
    cand_scores = als_item[candidate_ids] @ als_user[user_id]
    order = np.argsort(-cand_scores)[:k]
    top_items = [candidate_ids[i] for i in order]

    recs = []
    for i in top_items:
        row = item_meta.iloc[i]
        recs.append({
            "item_idx": int(i),
            "title": row["title"],
            "authors": row["authors"],
            "score": float(als_item[i] @ als_user[user_id]),
        })

    latency_ms = (time.time() - t0) * 1000
    return RecommendResponse(
        user_id=user_id,
        candidates_retrieved=len(candidate_ids),
        latency_ms=round(latency_ms, 2),
        recommendations=recs,
    )
