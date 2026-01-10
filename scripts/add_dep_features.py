#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from src.utils.paths import PROCESSED_DIR
from src.features.dep_features import add_dep_features


STEM = "reddit_ra_la_balanced"
SPACY_MODEL = "en_core_web_sm"


def main():
    for split in ["train", "val", "test"]:
        base_path = PROCESSED_DIR / f"{STEM}_{split}.csv"
        out_path = PROCESSED_DIR / f"{STEM}_{split}__dep.csv"

        print(f"[INFO] Loading {base_path}")
        df = pd.read_csv(base_path)

        df_with = add_dep_features(
            df,
            text_col="text",
            spacy_model=SPACY_MODEL,
            batch_size=128,   # parser is slower → smaller batch
            n_process=1,
        )

        dep_cols = [c for c in df_with.columns if c.startswith("dep_")]
        df_out = df_with[["utterance_id"] + dep_cols]

        df_out.to_csv(out_path, index=False)
        print(f"[INFO] Wrote dependency features to {out_path} ({len(dep_cols)} features)")


if __name__ == "__main__":
    main()
