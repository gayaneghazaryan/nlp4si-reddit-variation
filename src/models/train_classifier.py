#!/usr/bin/env python3
"""
Train a feature-based subreddit classifier with train/val/test splits.

Supports:
  A) Training on ONE feature file per split:
     --feature_set struct
     expects files: {stem}_{split}__struct.csv

  B) Training on ALL available feature files per split (merged automatically):
     --feature_set all
     expects files: {stem}_{split}__*.csv

Base files are required for labels:
  {stem}_{split}.csv must contain: utterance_id, subreddit

Examples:
  # Train only on structural features
  python3 scripts/30_train_classifier.py --stem reddit_ra_la_balanced --feature_set struct

  # Train on all extracted feature sets (struct + lexical + ...)
  python3 scripts/30_train_classifier.py --stem reddit_ra_la_balanced --feature_set all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

from src.utils.paths import PROCESSED_DIR


LABEL_COL = "subreddit"
ID_COL = "utterance_id"


def load_base(stem: str, split: str) -> pd.DataFrame:
    base_path = PROCESSED_DIR / f"{stem}_{split}.csv"
    df = pd.read_csv(base_path)

    required = {ID_COL, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Base file {base_path} missing columns: {missing}")

    return df[[ID_COL, LABEL_COL]].copy()


def load_feature_file(stem: str, split: str, feature_set: str) -> pd.DataFrame:
    """
    Loads one feature-only file:
      {stem}_{split}__{feature_set}.csv
    """
    feat_path = PROCESSED_DIR / f"{stem}_{split}__{feature_set}.csv"
    df = pd.read_csv(feat_path)

    if ID_COL not in df.columns:
        raise ValueError(f"Feature file {feat_path} missing '{ID_COL}' column")

    return df


def find_feature_files(stem: str, split: str) -> List[Path]:
    """
    Finds all feature-only files matching:
      {stem}_{split}__*.csv
    """
    return sorted(PROCESSED_DIR.glob(f"{stem}_{split}__*.csv"))


def merge_features_for_split(
    stem: str,
    split: str,
    feature_set: str,
) -> Tuple[pd.DataFrame, List[Path]]:
    """
    Returns:
      merged dataframe with [utterance_id, subreddit] + feature columns
      list of feature files used
    """
    base = load_base(stem, split)

    used_files: List[Path] = []

    if feature_set == "all":
        files = find_feature_files(stem, split)
        if not files:
            raise FileNotFoundError(
                f"No feature files found for {stem}_{split}__*.csv in {PROCESSED_DIR}"
            )

        merged = base
        seen_cols = set(merged.columns)

        for fp in files:
            fdf = pd.read_csv(fp)
            if ID_COL not in fdf.columns:
                raise ValueError(f"Feature file {fp} missing '{ID_COL}'")

            overlap = (set(fdf.columns) & seen_cols) - {ID_COL}
            if overlap:
                raise ValueError(
                    f"Duplicate columns when merging {fp.name}: {sorted(overlap)}. "
                    f"Rename features to keep them unique."
                )

            merged = merged.merge(fdf, on=ID_COL, how="inner")
            seen_cols |= set(fdf.columns)
            used_files.append(fp)
        


        return merged, used_files

    # feature_set is one specific set name (struct, lexical, pragmatic)
    fdf = load_feature_file(stem, split, feature_set)
    merged = base.merge(fdf, on=ID_COL, how="inner")
    used_files.append(PROCESSED_DIR / f"{stem}_{split}__{feature_set}.csv")
    return merged, used_files


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[LABEL_COL]
    X = df.drop(columns=[ID_COL, LABEL_COL])
    return X, y


def train_and_eval(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed: int,
):
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=3000, random_state=seed)),
    ])

    model.fit(X_train, y_train)

    def _report(split_name: str, X, y):
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        print(f"\n=== {split_name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y, pred))
        print("Confusion matrix:")
        print(confusion_matrix(y, pred))

    _report("Validation", X_val, y_val)
    _report("Test", X_test, y_test)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", required=True, help="Dataset stem, e.g. reddit_ra_la_balanced")
    parser.add_argument(
        "--feature_set",
        required=True,
        help="Either a single set name (e.g., struct) or 'all' to merge all feature files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="If set, save the trained model under data/processed/models/",
    )
    args = parser.parse_args()

    stem = args.stem
    feature_set = args.feature_set
    seed = args.seed

    # Load & merge for each split
    train_df, train_files = merge_features_for_split(stem, "train", feature_set)
    val_df, val_files = merge_features_for_split(stem, "val", feature_set)
    test_df, test_files = merge_features_for_split(stem, "test", feature_set)

    print("[INFO] Feature files used (train):")
    for fp in train_files:
        print(f"  - {fp.name}")

    # Build matrices
    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)
    X_test, y_test = split_X_y(test_df)

    if X_train.shape[1] == 0:
        raise ValueError("No feature columns found after merging. Check your feature files.")

    print(f"[INFO] Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # Train + evaluate
    model = train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, seed=seed)

    if args.save_model:
        models_dir = PROCESSED_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{stem}__{feature_set}__logreg.joblib"
        dump(model, model_path)
        print(f"[INFO] Saved model to {model_path}")


if __name__ == "__main__":
    main()
