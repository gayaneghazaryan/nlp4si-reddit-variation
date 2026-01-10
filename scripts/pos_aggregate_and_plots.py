#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import PROCESSED_DIR


STEM = "reddit_ra_la_balanced"

POS_TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train__pos.csv"
BASE_TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train.csv"

TABLE_OUT = Path("reports/tables/pos_aggregate_stats.csv")
FIG_DIR = Path("reports/figures/pos")

# Plot a focused subset (report-friendly, interpretable)
PLOT_FEATURES = [
    "pos_PRON_rate",
    "pos_first_person_rate",
    "pos_second_person_rate",
    "pos_noun_verb_ratio",
    "pos_AUX_rate",
    "pos_modal_rate",
    "pos_past_tense_rate",
    "pos_pres_tense_rate",
]


def main():
    # Load labels from base, features from feature-only file
    base = pd.read_csv(BASE_TRAIN_PATH)[["utterance_id", "subreddit"]]
    feats = pd.read_csv(POS_TRAIN_PATH)

    df = base.merge(feats, on="utterance_id", how="inner")

    # -----------------------------
    # Aggregate statistics
    # -----------------------------
    available = [c for c in PLOT_FEATURES if c in df.columns]
    if not available:
        raise ValueError(
            f"None of the requested POS features were found. "
            f"Have: {df.columns.tolist()}"
        )

    agg = (
        df.groupby("subreddit")[available]
        .agg(["mean", "median", "std"])
        .round(4)
    )

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TABLE_OUT)

    print("[INFO] POS aggregate statistics saved to:", TABLE_OUT)
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
