#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import PROCESSED_DIR


STEM = "reddit_ra_la_balanced"

NER_TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train__ner.csv"
BASE_TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train.csv"

TABLE_OUT = Path("reports/tables/ner_aggregate_stats.csv")
FIG_DIR = Path("reports/figures/ner")

PLOT_FEATURES = [
    "ner_entity_rate",
    "ner_PERSON_rate",
    "ner_ORG_rate",
    "ner_GPE_rate",
    "ner_DATE_rate",
    "ner_MONEY_rate",
]


def main():
    base = pd.read_csv(BASE_TRAIN_PATH)[["utterance_id", "subreddit"]]
    feats = pd.read_csv(NER_TRAIN_PATH)

    df = base.merge(feats, on="utterance_id", how="inner")

    available = [c for c in PLOT_FEATURES if c in df.columns]
    if not available:
        raise ValueError("No NER features found to aggregate/plot.")

    # Aggregate stats
    agg = (
        df.groupby("subreddit")[available]
        .agg(["mean", "median", "std"])
        .round(4)
    )

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TABLE_OUT)

    print("[INFO] NER aggregate statistics saved to:", TABLE_OUT)
    print(agg)

    # Boxplots
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
