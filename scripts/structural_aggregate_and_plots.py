import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.paths import PROCESSED_DIR

STEM = "reddit_ra_la_balanced"
TRAIN_PATH = PROCESSED_DIR / f"{STEM}_train__struct.csv"

TABLE_OUT = Path("reports/tables/structural_aggregate_stats.csv")
FIG_DIR = Path("reports/figures")

STRUCTURAL_FEATURES = [
    "char_len",
    "word_len",
    "sent_len",
    "avg_word_len",
    "num_qmarks",
    "digit_count",
    "uppercase_word_count",
]


def main():
    df = pd.read_csv(TRAIN_PATH)

    # -----------------------------
    # Aggregate statistics
    # -----------------------------
    agg = (
        df.groupby("subreddit")[STRUCTURAL_FEATURES]
        .agg(["mean", "median", "std"])
        .round(2)
    )

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TABLE_OUT)

    print("[INFO] Aggregate statistics:")
    print(agg)
    print(f"[INFO] Saved table to {TABLE_OUT}")

    # -----------------------------
    # Boxplots
    # -----------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for feat in STRUCTURAL_FEATURES:
        plt.figure()
        df.boxplot(column=feat, by="subreddit")
        plt.title(f"{feat} by subreddit")
        plt.suptitle("")
        plt.ylabel(feat)
        plt.tight_layout()

        out_path = FIG_DIR / f"{feat}_by_subreddit.png"
        plt.savefig(out_path)
        plt.close()

        print(f"[INFO] Saved plot to {out_path}")


if __name__ == "__main__":
    main()
