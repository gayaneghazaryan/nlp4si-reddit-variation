from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import PROCESSED_DIR


STEM = "reddit_ra_la_balanced"

POLITE_TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train__polite.csv"
BASE_TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train.csv"

TABLE_OUT = Path("reports/tables/politeness_aggregate_stats.csv")
FIG_DIR = Path("reports/figures/politeness")

TOP_K_STRATEGIES = 10


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    base = pd.read_csv(BASE_TRAIN_PATH)[["utterance_id", "subreddit"]]
    feats = pd.read_csv(POLITE_TRAIN_PATH)

    df = base.merge(feats, on="utterance_id", how="inner")

    print(df.columns)

    # Identify politeness columns
    polite_cols = [c for c in df.columns if c.startswith("polite_")]
    summary_cols = ["polite_sum", "polite_nonzero"]

    strategy_cols = [
        c for c in polite_cols
        if c not in summary_cols
    ]

    # -----------------------------
    # Aggregate statistics
    # -----------------------------
    agg = (
        df.groupby("subreddit")[summary_cols + strategy_cols]
        .agg(["mean", "median", "std"])
        .round(4)
    )

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TABLE_OUT)

    print("[INFO] Politeness aggregate statistics saved to:", TABLE_OUT)
    print(agg[summary_cols].head())

    # -----------------------------
    # Select top-K strategies by mean difference
    # -----------------------------
    means = (
        df.groupby("subreddit")[strategy_cols]
        .mean()
    )

    if means.shape[0] != 2:
        raise ValueError("Expected exactly two subreddits for comparison.")

    diff = (means.iloc[0] - means.iloc[1]).abs()
    top_strategies = diff.sort_values(ascending=False).head(TOP_K_STRATEGIES).index.tolist()

    print("[INFO] Top politeness strategies by mean difference:")
    for s in top_strategies:
        print(" ", s)

    # -----------------------------
    # Plots
    # -----------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # (1) Overall politeness load
    for col in summary_cols:
        plt.figure()
        df.boxplot(column=col, by="subreddit")
        plt.title(f"{col} by subreddit")
        plt.suptitle("")
        plt.ylabel(col)
        plt.tight_layout()

        out = FIG_DIR / f"{col}_by_subreddit.png"
        plt.savefig(out)
        plt.close()
        print("[INFO] Saved plot:", out)

    # (2) Individual strategy plots (top-K only)
    for col in top_strategies:
        plt.figure()
        df.boxplot(column=col, by="subreddit")
        plt.title(f"{col} by subreddit")
        plt.suptitle("")
        plt.ylabel(col)
        plt.tight_layout()

        out = FIG_DIR / f"{col}_by_subreddit.png"
        plt.savefig(out)
        plt.close()
        print("[INFO] Saved plot:", out)


if __name__ == "__main__":
    main()
