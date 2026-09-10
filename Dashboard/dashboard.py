"""
Project dashboard API
=====================

Serves the real files from this project.

Run with:

    uvicorn dashboard:app --host 0.0.0.0 --port 8010 --reload
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================
# Paths
# ==============================================================

ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

REPORTS_DIR = ROOT / "reports"

STATIC_DIR = (
    Path(__file__).resolve().parent
    / "static"
)


# ==============================================================
# App
# ==============================================================

app = FastAPI(
    title="Recsys Project Dashboard API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================
# Data
# ==============================================================

leaderboard_df = pd.read_csv(
    MODELS_DIR / "leaderboard.csv"
).rename(
    columns={
        "Unnamed: 0": "model"
    }
)

item_meta = pd.read_csv(
    DATA_DIR / "item_meta.csv"
)

mappings = json.loads(
    (
        DATA_DIR / "mappings.json"
    ).read_text()
)


# ==============================================================
# Cover availability
# ==============================================================

_HAS_COVER = (
    "small_image_url"
    in item_meta.columns
)

_HAS_BIGGER_COVER = (
    "image_url"
    in item_meta.columns
)

_HAS_ISBN = (
    "isbn13"
    in item_meta.columns
)

_HAS_GOOGLE_COVER = (
    "google_cover_url"
    in item_meta.columns
)

_HAS_GOOGLE_LARGE_COVER = (
    "google_cover_url_large"
    in item_meta.columns
)

_HAS_RATING_DIST = all(
    column in item_meta.columns
    for column in [
        "ratings_1",
        "ratings_2",
        "ratings_3",
        "ratings_4",
        "ratings_5",
    ]
)


# ==============================================================
# Cover helpers
# ==============================================================

def _unique_urls(urls: list[str]) -> list[str]:
    """
    Remove duplicate URLs while preserving order.
    """

    result = []
    seen = set()

    for url in urls:

        if not url:
            continue

        url = str(url)

        if url in seen:
            continue

        seen.add(url)
        result.append(url)

    return result


def _google_shelf_url(url: str) -> str:
    """
    Upgrade a Google Books image URL for the larger Shelf card.

    Google URLs often contain zoom=1 or zoom=2.
    """

    if not url:
        return url

    upgraded = str(url)

    upgraded = upgraded.replace(
        "zoom=1",
        "zoom=3"
    )

    upgraded = upgraded.replace(
        "zoom=2",
        "zoom=3"
    )

    upgraded = upgraded.replace(
        "w=128",
        "w=400"
    )

    return upgraded


def _cover_urls(row) -> list[str]:
    """
    Catalog cover candidates.

    Catalog thumbnails are small, so reliability is prioritized.
    """

    urls = []

    # Google Books
    if (
        _HAS_GOOGLE_COVER
        and pd.notna(
            row.get("google_cover_url")
        )
    ):
        urls.append(
            str(row["google_cover_url"])
        )

    # Open Library large
    if (
        _HAS_ISBN
        and pd.notna(
            row.get("isbn13")
        )
    ):
        isbn = str(
            int(row["isbn13"])
        )

        urls.append(
            "https://covers.openlibrary.org/"
            f"b/isbn/{isbn}-L.jpg?default=false"
        )

    # Goodreads medium
    if (
        _HAS_BIGGER_COVER
        and pd.notna(
            row.get("image_url")
        )
    ):
        urls.append(
            str(row["image_url"])
        )

    # Goodreads small
    if (
        _HAS_COVER
        and pd.notna(
            row.get("small_image_url")
        )
    ):
        urls.append(
            str(row["small_image_url"])
        )

    return _unique_urls(urls)


def _shelf_cover_urls(row) -> list[str]:
    """
    High-resolution cover candidates for the Shelf.

    IMPORTANT:
    Shelf cards are large, so Open Library's -L image is tried FIRST.
    Google Books and Goodreads are fallbacks only.

    Priority:
        1. Open Library Large
        2. Google Books large
        3. Google Books normal/upgraded
        4. Goodreads medium
        5. Goodreads small
    """

    urls = []

    # ----------------------------------------------------------
    # 1. OPEN LIBRARY LARGE — FIRST
    # ----------------------------------------------------------

    if (
        _HAS_ISBN
        and pd.notna(row.get("isbn13"))
    ):
        isbn = str(int(row["isbn13"]))

        urls.append(
            f"https://covers.openlibrary.org/"
            f"b/isbn/{isbn}-L.jpg?default=false"
        )

    # ----------------------------------------------------------
    # 2. GOOGLE BOOKS LARGE
    # ----------------------------------------------------------

    if (
        _HAS_GOOGLE_LARGE_COVER
        and pd.notna(
            row.get("google_cover_url_large")
        )
    ):
        urls.append(
            str(
                row["google_cover_url_large"]
            )
        )

    # ----------------------------------------------------------
    # 3. GOOGLE BOOKS NORMAL / ZOOMED
    # ----------------------------------------------------------

    if (
        _HAS_GOOGLE_COVER
        and pd.notna(
            row.get("google_cover_url")
        )
    ):
        google_url = str(
            row["google_cover_url"]
        )

        urls.append(
            _google_shelf_url(
                google_url
            )
        )

        urls.append(
            google_url
        )

    # ----------------------------------------------------------
    # 4. GOODREADS MEDIUM
    # ----------------------------------------------------------

    if (
        _HAS_BIGGER_COVER
        and pd.notna(
            row.get("image_url")
        )
    ):
        urls.append(
            str(
                row["image_url"]
            )
        )

    # ----------------------------------------------------------
    # 5. GOODREADS SMALL — LAST RESORT
    # ----------------------------------------------------------

    if (
        _HAS_COVER
        and pd.notna(
            row.get("small_image_url")
        )
    ):
        urls.append(
            str(
                row["small_image_url"]
            )
        )

    return _unique_urls(urls)

# ==============================================================
# Book card
# ==============================================================

def _book_card(row) -> dict:

    cover_urls = _cover_urls(row)

    shelf_cover_urls = (
        _shelf_cover_urls(row)
    )

    isbn13 = None

    if (
        _HAS_ISBN
        and pd.notna(
            row.get("isbn13")
        )
    ):
        isbn13 = str(
            int(row["isbn13"])
        )

    card = {
        "i": int(row["i"]),

        "book_id": int(
            row["book_id"]
        ),

        "title": row["title"],

        "authors": row["authors"],

        "year": (
            int(
                row[
                    "original_publication_year"
                ]
            )
            if pd.notna(
                row.get(
                    "original_publication_year"
                )
            )
            else None
        ),

        "average_rating": (
            float(
                row["average_rating"]
            )
            if pd.notna(
                row.get(
                    "average_rating"
                )
            )
            else None
        ),

        "isbn13": isbn13,

        # Catalog
        "cover_urls": cover_urls,

        # Shelf
        "shelf_cover_urls":
            shelf_cover_urls,

        # Explicit Google large URL
        "google_cover_url_large": (
            str(
                row[
                    "google_cover_url_large"
                ]
            )
            if (
                _HAS_GOOGLE_LARGE_COVER
                and pd.notna(
                    row.get(
                        "google_cover_url_large"
                    )
                )
            )
            else None
        ),

        # Backward compatibility
        "cover_url": (
            cover_urls[0]
            if cover_urls
            else None
        ),
    }

    if _HAS_RATING_DIST:

        distribution = [
            int(
                row[
                    f"ratings_{n}"
                ]
            )
            for n in range(1, 6)
        ]

        total = (
            sum(distribution)
            or 1
        )

        card["rating_count"] = total

        card["rating_dist_pct"] = [
            round(
                100 * value / total,
                1
            )
            for value in distribution
        ]

    else:

        card["rating_count"] = (
            int(
                row["ratings_count"]
            )
            if (
                "ratings_count"
                in row
                and pd.notna(
                    row.get(
                        "ratings_count"
                    )
                )
            )
            else None
        )

        card[
            "rating_dist_pct"
        ] = None

    return card


# ==============================================================
# ALS
# ==============================================================

als_item_path = (
    MODELS_DIR
    / "als_item_factors.npy"
)

als_user_path = (
    MODELS_DIR
    / "als_user_factors.npy"
)

als_item = (
    np.load(als_item_path)
    if als_item_path.exists()
    else None
)

als_user = (
    np.load(als_user_path)
    if als_user_path.exists()
    else None
)


# ==============================================================
# Demo shelves
# ==============================================================

shelves_path = (
    DATA_DIR
    / "demo_shelves.json"
)

demo_shelves = (
    json.loads(
        shelves_path.read_text()
    )
    if shelves_path.exists()
    else {}
)


# ==============================================================
# Model names
# ==============================================================

MODEL_DISPLAY_NAMES = {

    "sasrec_lite":
        "SASRec-lite",

    "semantic_id_generative":
        "Semantic-ID Generative",

    "als":
        "ALS",

    "two_tower":
        "Two-Tower",

    "ncf":
        "NCF",

    "popularity":
        "Popularity",

    "lightgcn":
        "LightGCN",

    "mf_bpr_pytorch":
        "MF (BPR)",

    "content_based":
        "Content-based",
}


RESULTS_FILES = {

    "popularity":
        "phase1_results.json",

    "content_based":
        "phase1_results.json",

    "als":
        "phase1_results.json",

    "mf_bpr_pytorch":
        "phase1_results.json",

    "ncf":
        "phase2_results.json",

    "two_tower":
        "phase2_results.json",

    "sasrec_lite":
        "phase2b_results.json",

    "lightgcn":
        "phase6a_results.json",

    "semantic_id_generative":
        "phase6b_results.json",
}


_results_cache = {}


def _load_results(fname):

    if fname not in _results_cache:

        path = (
            MODELS_DIR
            / fname
        )

        _results_cache[fname] = (
            json.loads(
                path.read_text()
            )
            if path.exists()
            else {}
        )

    return _results_cache[fname]


# ==============================================================
# TF-IDF
# ==============================================================

item_meta["_text"] = (
    item_meta["title"].fillna("")
    + " "
    + item_meta["authors"].fillna("")
    + " "
    + item_meta["tags_text"].fillna("")
)

_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

_tfidf = (
    _vectorizer.fit_transform(
        item_meta["_text"]
    )
)


# ==============================================================
# Genres
# ==============================================================

GENRES = {

    "sci-fi": [
        "science-fiction",
        "sci-fi",
        "scifi",
        "dystopia",
        "space",
    ],

    "fantasy": [
        "fantasy",
        "magic",
        "wizards",
        "dragons",
    ],

    "romance": [
        "romance",
        "chick-lit",
        "love-story",
    ],

    "mystery-thriller": [
        "mystery",
        "thriller",
        "crime",
        "suspense",
        "detective",
    ],

    "horror": [
        "horror",
        "paranormal",
        "supernatural",
        "ghost",
    ],

    "historical": [
        "historical-fiction",
        "historical",
    ],

    "young-adult": [
        "young-adult",
        "ya",
        "teen",
        "childrens",
    ],

    "classics": [
        "classics",
        "classic",
    ],

    "non-fiction": [
        "non-fiction",
        "nonfiction",
    ],

    "biography-memoir": [
        "biography",
        "memoir",
        "autobiography",
    ],

    "poetry": [
        "poetry",
    ],

    "graphic-comics": [
        "graphic-novels",
        "comics",
        "manga",
        "comic-book",
    ],

    "self-help": [
        "self-help",
        "self-improvement",
        "personal-development",
    ],

    "humor": [
        "humor",
        "humour",
        "funny",
        "comedy",
    ],
}


GENRE_LABELS = {

    "sci-fi":
        "Sci-Fi",

    "fantasy":
        "Fantasy",

    "romance":
        "Romance",

    "mystery-thriller":
        "Mystery & Thriller",

    "horror":
        "Horror",

    "historical":
        "Historical Fiction",

    "young-adult":
        "Young Adult",

    "classics":
        "Classics",

    "non-fiction":
        "Non-Fiction",

    "biography-memoir":
        "Biography & Memoir",

    "poetry":
        "Poetry",

    "graphic-comics":
        "Graphic Novels & Comics",

    "self-help":
        "Self-Help",

    "humor":
        "Humor",
}


_tags_lower = (
    item_meta[
        "tags_text"
    ]
    .fillna("")
    .str.lower()
)


def _genre_mask(
    genre_key: str
):

    terms = GENRES.get(
        genre_key,
        []
    )

    if not terms:

        return pd.Series(
            False,
            index=item_meta.index
        )

    pattern = "|".join(
        re.escape(term)
        for term in terms
    )

    return _tags_lower.str.contains(
        pattern,
        regex=True,
        na=False
    )


# ==============================================================
# Similarity
# ==============================================================

def recommend_similar(
    item_row_idx: int,
    k: int = 10
):

    similarities = (
        cosine_similarity(
            _tfidf[item_row_idx],
            _tfidf
        ).ravel()
    )

    order = (
        similarities
        .argsort()[::-1]
    )

    order = [
        i
        for i in order
        if i != item_row_idx
    ][:k]

    result = []

    for i in order:

        row = item_meta.iloc[i]

        card = _book_card(row)

        card["similarity"] = round(
            float(
                similarities[i]
            ),
            4
        )

        result.append(card)

    return result


# ==============================================================
# API: stats
# ==============================================================

@app.get("/api/stats")
def stats():

    return {
        **mappings,
        "n_models":
            len(leaderboard_df),
    }


# ==============================================================
# API: leaderboard
# ==============================================================

@app.get("/api/leaderboard")
def leaderboard():

    rows = []

    for _, row in (
        leaderboard_df.iterrows()
    ):

        data = row.to_dict()

        model_key = data.pop(
            "model"
        )

        data["model"] = model_key

        data["display_name"] = (
            MODEL_DISPLAY_NAMES.get(
                model_key,
                model_key
            )
        )

        rows.append(data)

    rows.sort(
        key=lambda x:
            x["ndcg@10"],
        reverse=True
    )

    return rows


# ==============================================================
# API: model
# ==============================================================

@app.get("/api/model/{model_key}")
def model_detail(
    model_key: str
):

    if model_key not in RESULTS_FILES:

        raise HTTPException(
            404,
            f"unknown model '{model_key}'"
        )

    results = _load_results(
        RESULTS_FILES[model_key]
    )

    if model_key not in results:

        raise HTTPException(
            404,
            "no detailed results for "
            f"'{model_key}' in "
            f"{RESULTS_FILES[model_key]}"
        )

    return {

        "model":
            model_key,

        "display_name":
            MODEL_DISPLAY_NAMES.get(
                model_key,
                model_key
            ),

        "source_file":
            RESULTS_FILES[model_key],

        "metrics_by_k":
            results[model_key],
    }


# ==============================================================
# API: genres
# ==============================================================

@app.get("/api/genres")
def genres():

    return [

        {
            "key": key,

            "label":
                GENRE_LABELS[key],

            "count":
                int(
                    _genre_mask(
                        key
                    ).sum()
                ),
        }

        for key in GENRES
    ]


# ==============================================================
# API: books
# ==============================================================

@app.get("/api/books")
def books(

    search: str = Query(
        "",
        description=
            "search title/author"
    ),

    genre: str = Query(
        "",
        description=
            "genre key from /api/genres"
    ),

    limit: int = 30,

    offset: int = 0,
):

    df = item_meta

    if genre:

        if genre not in GENRES:

            raise HTTPException(
                404,
                f"unknown genre '{genre}'"
            )

        df = df[
            _genre_mask(
                genre
            ).reindex(
                df.index
            )
        ]

    if search:

        mask = (
            df["_text"]
            .str.contains(
                re.escape(search),
                case=False,
                na=False
            )
        )

        df = df[mask]

    total = len(df)

    total_catalog = len(
        item_meta
    )

    page = df.iloc[
        offset:
        offset + limit
    ]

    return {

        "total":
            total,

        "total_catalog":
            total_catalog,

        "results": [

            {
                **_book_card(row),

                "tags_text":
                    row["tags_text"],
            }

            for _, row
            in page.iterrows()
        ],
    }


# ==============================================================
# API: book detail
# ==============================================================

@app.get("/api/book/{i}")
def book_detail(i: int):

    row = item_meta[
        item_meta["i"] == i
    ]

    if row.empty:

        raise HTTPException(
            404,
            f"no book with internal id {i}"
        )

    row = row.iloc[0]

    return {
        **_book_card(row),
        "tags_text":
            row["tags_text"],
    }


# ==============================================================
# API: recommend
# ==============================================================

@app.get("/api/recommend/{i}")
def recommend(
    i: int,
    k: int = 10
):

    row = item_meta[
        item_meta["i"] == i
    ]

    if row.empty:

        raise HTTPException(
            404,
            f"no book with internal id {i}"
        )

    row_idx = row.index[0]

    seed = row.iloc[0]

    return {

        "seed": {

            "i":
                int(seed["i"]),

            "title":
                seed["title"],

            "authors":
                seed["authors"],
        },

        "method":
            "TF-IDF "
            "(title + authors + tags) "
            "cosine similarity — "
            "computed live from "
            "data/processed/item_meta.csv",

        "recommendations":
            recommend_similar(
                row_idx,
                k=k
            ),
    }


# ==============================================================
# API: random reader
# ==============================================================

@app.get("/api/random_reader")
def random_reader():

    if not demo_shelves:

        raise HTTPException(
            503,
            "no demo reader shelves "
            "bundled with this deployment"
        )

    import random

    user_id = random.choice(
        list(
            demo_shelves.keys()
        )
    )

    return {

        "user_id":
            int(user_id),

        "shelf":
            demo_shelves[user_id],
    }


# ==============================================================
# API: personalized recommendations
# ==============================================================

@app.get(
    "/api/recommend_for_reader/{user_id}"
)
def recommend_for_reader(

    user_id: int,

    k: int = 10,

    genre: str = Query(
        "",
        description=
            "genre key from /api/genres"
    ),
):

    if (
        als_item is None
        or als_user is None
    ):

        raise HTTPException(
            503,
            "ALS model factors "
            "not bundled with this deployment"
        )

    if (
        user_id < 0
        or user_id >= als_user.shape[0]
    ):

        raise HTTPException(
            404,
            "reader id must be between "
            f"0 and "
            f"{als_user.shape[0] - 1}"
        )

    scores = (
        als_item
        @ als_user[user_id]
    )

    method = (
        "ALS collaborative filtering "
        "(models/als_*.npy) — "
        "trained on 5.9M real ratings"
    )

    if genre:

        if genre not in GENRES:

            raise HTTPException(
                404,
                f"unknown genre '{genre}'"
            )

        mask = _genre_mask(
            genre
        )

        allowed = np.zeros(
            scores.shape[0],
            dtype=bool
        )

        matched_i = (
            item_meta.loc[
                mask,
                "i"
            ].values
        )

        matched_i = matched_i[
            (
                matched_i >= 0
            )
            & (
                matched_i
                < scores.shape[0]
            )
        ]

        allowed[
            matched_i
        ] = True

        scores = np.where(
            allowed,
            scores,
            -np.inf
        )

        method += (
            " — filtered to genre "
            f"'{GENRE_LABELS[genre]}' "
            f"({int(allowed.sum())} "
            "eligible titles)"
        )

    top_idx = np.argsort(
        -scores
    )[:k]

    item_meta_idx = (
        item_meta
        .set_index("i")
    )

    recommendations = []

    for i in top_idx:

        i = int(i)

        if (
            not np.isfinite(
                scores[i]
            )
            or i not in item_meta_idx.index
        ):
            continue

        row = item_meta_idx.loc[i]

        row = row.copy()

        row["i"] = i

        card = _book_card(row)

        card["score"] = round(
            float(scores[i]),
            4
        )

        recommendations.append(
            card
        )

    return {

        "user_id":
            user_id,

        "method":
            method,

        "shelf":
            demo_shelves.get(
                str(user_id),
                []
            ),

        "recommendations":
            recommendations,
    }


# ==============================================================
# API: architecture
# ==============================================================

@app.get("/api/architecture")
def architecture():

    return {

        "phases": [

            {
                "phase": "Phase 0",
                "name":
                    "Data engineering",
                "file":
                    "src/phase0_data_prep.py",
            },

            {
                "phase": "Phase 1",
                "name":
                    "Classical baselines "
                    "(popularity, "
                    "content-based, ALS, "
                    "MF-from-scratch)",
                "file":
                    "src/phase1_baselines.py",
            },

            {
                "phase": "Phase 2",
                "name":
                    "Deep learning "
                    "(NCF, Two-Tower)",
                "file":
                    "src/phase2_deep.py",
            },

            {
                "phase": "Phase 2b",
                "name":
                    "Sequential "
                    "(SASRec-lite, "
                    "causal transformer)",
                "file":
                    "src/phase2b_sequential.py",
            },

            {
                "phase": "Phase 3",
                "name":
                    "Unified evaluation "
                    "(leaderboard + "
                    "cold-start slice)",
                "file":
                    "src/phase3_compare.py",
            },

            {
                "phase": "Phase 5",
                "name":
                    "Serving "
                    "(FAISS retrieval "
                    "-> ALS re-rank, "
                    "FastAPI)",
                "file":
                    "api/serve.py",
            },

            {
                "phase": "Phase 6a",
                "name":
                    "Graph neural net "
                    "(LightGCN, "
                    "from scratch)",
                "file":
                    "src/phase6a_lightgcn.py",
            },

            {
                "phase": "Phase 6b",
                "name":
                    "Generative retrieval "
                    "(semantic-ID "
                    "transformer)",
                "file":
                    "src/phase6b_generative_retrieval.py",
            },
        ]
    }


# ==============================================================
# API: limitations
# ==============================================================

@app.get("/api/limitations")
def limitations():

    path = (
        REPORTS_DIR
        / "compute_tradeoffs.md"
    )

    readme = (
        ROOT / "README.md"
    ).read_text()

    honest_section = ""

    match = re.search(
        r"## Honest limitations.*?"
        r"(?=\n## )",
        readme,
        re.S
    )

    if match:

        honest_section = (
            match.group(0)
        )

    return {

        "honest_limitations_from_readme":
            honest_section,

        "compute_tradeoffs_md":
            path.read_text()
            if path.exists()
            else "",
    }


# ==============================================================
# API: charts
# ==============================================================

@app.get("/api/charts")
def charts():

    return {

        "leaderboard_bar":
            "/charts/leaderboard_bar.png",

        "coverage_vs_accuracy":
            "/charts/coverage_vs_accuracy.png",
    }


# ==============================================================
# Static files
# ==============================================================

app.mount(
    "/charts",
    StaticFiles(
        directory=str(
            REPORTS_DIR
        )
    ),
    name="charts"
)

app.mount(
    "/",
    StaticFiles(
        directory=str(
            STATIC_DIR
        ),
        html=True
    ),
    name="static"
)