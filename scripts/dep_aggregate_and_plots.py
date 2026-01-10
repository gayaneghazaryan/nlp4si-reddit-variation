#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import PROCESSED_DIR


STEM = "reddit_ra_la_balanced"

DEP_TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train__dep.csv"
BASE_TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train.csv"

TABLE_OUT = Path("reports/tables/dep_aggregate_stats.csv")
FIG_DIR = Path("reports/figures/dep")

DEP_PLOT_FEATURES = [
    "dep_avg_tree_depth",
    "dep_max_tree_depth",
    "dep_avg_dep_distance",
    "dep_advcl_rate",
    "dep_ccomp_rate",
    "dep_xcomp_rate",
    "dep_cc_rate",
    "dep_conj_rate",
    "dep_nmod_rate",
    "dep_compound_rate",
]


def main():
    # Load labels and dependency features
    base = pd.read_csv(BASE_TRAIN_PATH)[["utterance_id", "subreddit"]]
    feats = pd.read_csv(DEP_TRAIN_PATH)

    df = base.merge(feats, on="utterance_id", how="inner")

    available = [c for c in DEP_PLOT_FEATURES if c in df.columns]
    if not available:
        raise ValueError("No dependency features found to aggregate.")

    # -----------------------------
    # Aggregate statistics
    # -----------------------------
    agg = (
        df.groupby("subreddit")[available]
        .agg(["mean", "median", "std"])
        .round(4)
    )

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TABLE_OUT)

    print("[INFO] Dependency aggregate statistics saved to:", TABLE_OUT)
    print(agg)

    # -----------------------------
    # Boxplots
    # -----------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for feat in available:
        plt.figure()
        df.boxplot(column=feat, by="subreddit")
        plt.title(f"{feat} by subreddit")
        plt.suptitle("")
        plt.ylabel(feat)
        plt.tight_layout()

        out_path = FIG_DIR / f"{feat}_by_subreddit.png"
        plt.savefig(out_path)
        plt.close()

        print("[INFO] Saved plot:", out_path)


if __name__ == "__main__":
    main()
