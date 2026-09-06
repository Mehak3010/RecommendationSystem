import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(MODELS_DIR, "leaderboard.csv")).rename(columns={"Unnamed: 0": "model"})
df = df.sort_values("ndcg@10", ascending=False)

DISPLAY = {
    "sasrec_lite": "SASRec-lite", "semantic_id_generative": "Semantic-ID Generative",
    "als": "ALS", "two_tower": "Two-Tower", "ncf": "NCF", "popularity": "Popularity",
    "lightgcn": "LightGCN", "mf_bpr_pytorch": "MF (BPR)", "content_based": "Content-based",
}
df["label"] = df["model"].map(lambda m: DISPLAY.get(m, m))
highlight = {"sasrec_lite", "semantic_id_generative"}
colors = ["#B8863B" if m in highlight else "#3B5D50" for m in df["model"]]

# --- Chart 1: leaderboard bar ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(df["label"][::-1], df["ndcg@10"][::-1], color=colors[::-1])
ax.set_xlabel("NDCG@10")
ax.set_title("Model Leaderboard (NDCG@10, 5,000 held-out users)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "leaderboard_bar.png"), dpi=150)
plt.close()

# --- Chart 2: coverage vs accuracy scatter ---
fig, ax = plt.subplots(figsize=(7, 6))
for _, row in df.iterrows():
    c = "#B8863B" if row["model"] in highlight else "#3B5D50"
    ax.scatter(row["coverage@10"], row["ndcg@10"], s=90, color=c, zorder=3)
    ax.annotate(row["label"], (row["coverage@10"], row["ndcg@10"]),
                textcoords="offset points", xytext=(7, 4), fontsize=9)
ax.set_xlabel("Coverage@10 (fraction of catalog ever recommended)")
ax.set_ylabel("NDCG@10")
ax.set_title("Accuracy vs. Catalog Coverage")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "coverage_vs_accuracy.png"), dpi=150)
plt.close()

print("saved charts")
