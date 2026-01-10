from __future__ import annotations

import argparse
import pandas as pd

from src.utils.paths import PROCESSED_DIR
from src.config import OUTPUT_FILENAME
from src.features.politeness_features import add_politeness_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", type=str, default=OUTPUT_FILENAME.replace(".csv", ""))
    ap.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    ap.add_argument("--strategy_collection", type=str, default="politeness_api")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    stem = args.stem

    for split in ["train", "val", "test"]:
        in_path = PROCESSED_DIR / f"{stem}_{split}.csv"
        out_path = PROCESSED_DIR / f"{stem}_{split}__polite.csv"

        df = pd.read_csv(in_path)
        df_polite = add_politeness_features(
            df,
            text_col="text",
            utterance_id_col="utterance_id",
            speaker_col="speaker_id",
            convo_col="conversation_id",
            spacy_model=args.spacy_model,
            strategy_collection=args.strategy_collection,
            verbose=args.verbose,
        )
        df_polite.to_csv(out_path, index=False)
        print(f"[INFO] Wrote {len(df_polite)} rows with politeness features to {out_path}")


if __name__ == "__main__":
    main()
