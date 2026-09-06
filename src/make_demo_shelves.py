"""
Generates a small demo lookup file: for ~300 sample readers, their liked
book titles (rating >= 4). This lets the deployed dashboard show a real
"reader's shelf" next to personalized recommendations without needing to
ship the full ~80MB train.csv to production -- this file is a few hundred KB.

Run once, locally, after phase0 + phase1 have produced train.csv and the
ALS factors:  python src/make_demo_shelves.py
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
DATA_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")

N_SAMPLE_READERS = 300
MIN_INTERACTIONS, MAX_INTERACTIONS = 10, 40  # a nice demo-able range: not cold, not overwhelming

def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    item_meta = pd.read_csv(os.path.join(DATA_DIR, "item_meta.csv")).set_index("i")

    counts = train.groupby("u").size()
    eligible = counts[(counts >= MIN_INTERACTIONS) & (counts <= MAX_INTERACTIONS)].index.values
    rng = np.random.RandomState(7)
    sample_users = rng.choice(eligible, size=min(N_SAMPLE_READERS, len(eligible)), replace=False)

    liked = train[(train.label == 1) & (train.u.isin(sample_users))]
    shelves = {}
    for u, grp in liked.groupby("u"):
        titles = []
        for i in grp["i"].head(6):
            if i in item_meta.index:
                row = item_meta.loc[i]
                titles.append({"title": row["title"], "authors": row["authors"]})
        if titles:
            shelves[str(int(u))] = titles

    out_path = os.path.join(DATA_DIR, "demo_shelves.json")
    json.dump(shelves, open(out_path, "w"), indent=1)
    print(f"Saved {len(shelves)} demo reader shelves to {out_path}")
    print(f"Sample reader ids you can try: {list(shelves.keys())[:10]}")

if __name__ == "__main__":
    main()
