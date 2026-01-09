from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.utils.paths import PROCESSED_DIR


@dataclass(frozen=True)
class FeatureMergeResult:
    df: pd.DataFrame
    feature_files: List[Path]


def load_base(stem: str, split: str, label_col: str = "subreddit") -> pd.DataFrame:
    """
    Loads the base split file and keeps only the columns needed for training.
    """
    base_path = PROCESSED_DIR / f"{stem}_{split}.csv"
    df = pd.read_csv(base_path)

    required = {"utterance_id", label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Base file {base_path} is missing columns: {missing}")

    return df[["utterance_id", label_col]].copy()


def find_feature_files(stem: str, split: str) -> List[Path]:
    """
    Finds all feature files for a given stem/split following:
      {stem}_{split}__*.csv
    """
    pattern = f"{stem}_{split}__*.csv"
    files = sorted(PROCESSED_DIR.glob(pattern))
    return files


def merge_all_features(
    stem: str,
    split: str,
    label_col: str = "subreddit",
    join_key: str = "utterance_id",
    how: str = "inner",
) -> FeatureMergeResult:
    """
    Loads the base file and merges all discovered feature files on join_key.
    """
    base = load_base(stem, split, label_col=label_col)

    feature_files = find_feature_files(stem, split)
    if not feature_files:
        raise FileNotFoundError(
            f"No feature files found for split={split}. Expected pattern: "
            f"{stem}_{split}__*.csv in {PROCESSED_DIR}"
        )

    merged = base
    seen_feature_cols = set(merged.columns)

    for fp in feature_files:
        fdf = pd.read_csv(fp)

        if join_key not in fdf.columns:
            raise ValueError(f"Feature file {fp} missing join key '{join_key}'")

        # Ensure no duplicate feature column names across feature sets
        overlap = (set(fdf.columns) & seen_feature_cols) - {join_key}
        if overlap:
            raise ValueError(
                f"Duplicate columns detected when merging {fp}: {sorted(overlap)}. "
                f"Rename features to keep them unique across feature sets."
            )

        merged = merged.merge(fdf, on=join_key, how=how)
        seen_feature_cols |= set(fdf.columns)

    return FeatureMergeResult(df=merged, feature_files=feature_files)
