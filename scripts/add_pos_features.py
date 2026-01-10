#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from src.utils.paths import PROCESSED_DIR
from src.features.pos_features import add_pos_features


STEM = "reddit_ra_la_balanced"         
SPACY_MODEL = "en_core_web_sm"


def main():
    for split in ["train", "val", "test"]:
        base_path = PROCESSED_DIR / f"{STEM}_{split}.csv"
        out_path = PROCESSED_DIR / f"{STEM}_{split}__pos.csv"

        print(f"[INFO] Loading base split: {base_path}")
        df = pd.read_csv(base_path)

        if "utterance_id" not in df.columns:
            raise ValueError(f"{base_path} missing required column: utterance_id")
        if "text" not in df.columns:
            raise ValueError(f"{base_path} missing required column: text")

        # Extract POS features
        df_with = add_pos_features(
            df,
            text_col="text",
            spacy_model=SPACY_MODEL,
            batch_size=256,
            n_process=1,
        )

        # Utterance_id + pos_* features (feature-only file)
        pos_cols = [c for c in df_with.columns if c.startswith("pos_")]
        df_out = df_with[["utterance_id"] + pos_cols]

        df_out.to_csv(out_path, index=False)
        print(f"[INFO] Wrote POS features to {out_path} ({len(df_out)} rows, {len(pos_cols)} features)")


if __name__ == "__main__":
    main()
