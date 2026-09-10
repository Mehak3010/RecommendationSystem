"""
Adds `isbn13` and `image_url` (the Goodreads "medium" cover, larger than
the `small_image_url` already in item_meta.csv) so the dashboard can build
a proper cover fallback chain: Open Library large (via ISBN) -> Goodreads
medium -> Goodreads small -> plain placeholder.

Why this is needed: item_meta.csv only carries `small_image_url`
(~50x75px). Any UI element bigger than that -- like the "Your Shelf" grid
cards -- was stretching that tiny source image, which is what caused the
pixelation. Open Library's covers API can serve real large (300-600px+)
images when it has the book indexed by ISBN, which is what actually fixes
the problem rather than just picking a slightly-less-small source.

Run once:  python src/add_cover_data.py
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "data", "processed")


def clean_isbn13(val):
    """isbn13 comes in as a float (e.g. 9.780439e+12) because pandas infers
    a numeric dtype for the column -- convert back to a clean 13-digit
    string, or None if missing/malformed."""
    if pd.isna(val):
        return None
    try:
        s = str(int(round(float(val))))
        return s if len(s) == 13 else None
    except (ValueError, TypeError):
        return None


def main():
    books = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))
    item_meta = pd.read_csv(os.path.join(OUT_DIR, "item_meta.csv"))

    books["isbn13_clean"] = books["isbn13"].map(clean_isbn13)
    lookup = books.set_index("book_id")[["isbn13_clean", "image_url"]]

    item_meta["isbn13"] = item_meta["book_id"].map(lookup["isbn13_clean"])
    item_meta["image_url"] = item_meta["book_id"].map(lookup["image_url"])

    item_meta.to_csv(os.path.join(OUT_DIR, "item_meta.csv"), index=False)

    n_isbn = item_meta["isbn13"].notna().sum()
    n_img = item_meta["image_url"].notna().sum()
    print(f"Added isbn13 to {n_isbn}/{len(item_meta)} books ({n_isbn/len(item_meta):.1%})")
    print(f"Added image_url to {n_img}/{len(item_meta)} books ({n_img/len(item_meta):.1%})")
    print("item_meta.csv columns now:", list(item_meta.columns))

if __name__ == "__main__":
    main()