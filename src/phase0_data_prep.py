"""
Phase 0 - Data Engineering
==========================
Loads goodbooks-10k, filters sparse users/items, builds content features
from genre tags, and creates a leakage-aware train/val/test split.

NOTE on splitting strategy: goodbooks-10k has no interaction timestamps.
A true recommender system split must be time-based (train on the past,
predict the future) to avoid leakage. Since we don't have real timestamps,
we use each user's *row order in the raw file* as a proxy for interaction
order (this is a documented limitation - in a production setting you would
insist on real event timestamps). We do a per-user "leave-last-2-out" split:
last interaction -> test, second-to-last -> validation, rest -> train.
This still respects the core principle (no future interaction leaks into
training) even if "future" here is only a proxy ordering.
"""
import pandas as pd
import numpy as np
from scipy import sparse
import json
import os

# Portable path: this file lives at <repo_root>/src/phase0_data_prep.py,
# so the repo root is one directory up from this file's own location --
# works regardless of which machine or OS this is run on, or what the
# current working directory is when you invoke `python3 src/phase0_data_prep.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_USER_INTERACTIONS = 5   # drop users with fewer than this many ratings
MIN_ITEM_INTERACTIONS = 5   # drop books with fewer than this many ratings
IMPLICIT_THRESHOLD = 4      # rating >= 4 counts as a positive implicit signal

def main():
    print("Loading raw ratings...")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    books = pd.read_csv(f"{DATA_DIR}/books.csv")
    tags = pd.read_csv(f"{DATA_DIR}/tags.csv")
    book_tags = pd.read_csv(f"{DATA_DIR}/book_tags.csv")

    print(f"Raw ratings: {len(ratings):,} | users: {ratings.user_id.nunique():,} | books: {ratings.book_id.nunique():,}")

    # --- Filter sparse users/items (a couple of passes since filtering one
    # side can push the other below threshold) ---
    for _ in range(3):
        item_counts = ratings.book_id.value_counts()
        keep_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        ratings = ratings[ratings.book_id.isin(keep_items)]

        user_counts = ratings.user_id.value_counts()
        keep_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        ratings = ratings[ratings.user_id.isin(keep_users)]

    print(f"After filtering: {len(ratings):,} ratings | users: {ratings.user_id.nunique():,} | books: {ratings.book_id.nunique():,}")

    # --- Re-index to contiguous integer ids (needed for embedding matrices) ---
    user_ids = sorted(ratings.user_id.unique())
    item_ids = sorted(ratings.book_id.unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {b: i for i, b in enumerate(item_ids)}
    ratings["u"] = ratings.user_id.map(user2idx)
    ratings["i"] = ratings.book_id.map(item2idx)

    n_users, n_items = len(user_ids), len(item_ids)
    print(f"n_users={n_users}, n_items={n_items}")

    # --- Proxy ordering: row order within each user's ratings ---
    ratings["order"] = ratings.groupby("u").cumcount()
    ratings["n_for_user"] = ratings.groupby("u")["u"].transform("count")

    is_test = ratings["order"] == (ratings["n_for_user"] - 1)
    is_val = ratings["order"] == (ratings["n_for_user"] - 2)
    is_train = ~(is_test | is_val)

    train = ratings[is_train].copy()
    val = ratings[is_val].copy()
    test = ratings[is_test].copy()

    print(f"Split sizes -> train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")

    # --- Implicit feedback label (rating >= threshold = positive) ---
    for df in (train, val, test):
        df["label"] = (df["rating"] >= IMPLICIT_THRESHOLD).astype(int)

    # --- Content features: genre/tag bag for each book (top tags only) ---
    top_tags = book_tags.merge(tags, on="tag_id")
    # goodreads_book_id -> book_id mapping
    gb2book = books.set_index("goodreads_book_id")["book_id"].to_dict()
    top_tags["book_id"] = top_tags["goodreads_book_id"].map(gb2book)
    top_tags = top_tags.dropna(subset=["book_id"])
    top_tags["book_id"] = top_tags["book_id"].astype(int)
    top_tags = top_tags[top_tags.book_id.isin(item2idx)]
    top_tags["i"] = top_tags["book_id"].map(item2idx)

    # keep top 5 tags per book by count for a lightweight content signal
    top_tags = top_tags.sort_values("count", ascending=False)
    top_tags_per_book = top_tags.groupby("i")["tag_name"].apply(lambda s: " ".join(s.head(5))).to_dict()

    item_meta = books[books.book_id.isin(item2idx)].copy()
    item_meta["i"] = item_meta.book_id.map(item2idx)
    item_meta["tags_text"] = item_meta["i"].map(top_tags_per_book).fillna("")
    # small_image_url / ratings_1..5 already exist in books.csv (same download
    # you already have) -- keeping them through here is what lets the
    # dashboard show real cover art and a real rating-distribution bar
    # instead of plain text rows, with no new data source.
    item_meta = item_meta[[
        "i", "book_id", "title", "authors", "original_publication_year",
        "average_rating", "tags_text", "small_image_url",
        "ratings_count", "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5",
    ]]
    item_meta = item_meta.sort_values("i")

    # --- Save everything ---
    train[["u", "i", "rating", "label"]].to_csv(f"{OUT_DIR}/train.csv", index=False)
    val[["u", "i", "rating", "label"]].to_csv(f"{OUT_DIR}/val.csv", index=False)
    test[["u", "i", "rating", "label"]].to_csv(f"{OUT_DIR}/test.csv", index=False)
    item_meta.to_csv(f"{OUT_DIR}/item_meta.csv", index=False)

    with open(f"{OUT_DIR}/mappings.json", "w") as f:
        json.dump({
            "n_users": n_users,
            "n_items": n_items,
            "min_user_interactions": MIN_USER_INTERACTIONS,
            "min_item_interactions": MIN_ITEM_INTERACTIONS,
            "implicit_threshold": IMPLICIT_THRESHOLD,
        }, f, indent=2)

    print("Saved processed data to", OUT_DIR)
    print("\nPositive-rate in train (implicit label=1):", train["label"].mean().round(3))
    print("Item popularity Gini-ish check (top 1% items' share of interactions):")
    item_pop = train.i.value_counts()
    top1pct = max(1, int(0.01 * len(item_pop)))
    print(f"  top 1% of items ({top1pct} items) account for {item_pop.head(top1pct).sum() / len(train):.1%} of interactions")

if __name__ == "__main__":
    main()