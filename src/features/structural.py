from __future__ import annotations
from src.utils.paths import PROCESSED_DIR
from src.config import OUTPUT_FILENAME
import re
from typing import Dict, Any
import pandas as pd

_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_ALLCAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")

def extract_structural_features(text: str) -> Dict[str, Any]:
    t = "" if text is None else str(text)

    newline_count = t.count("\n")
    t_stripped = t.strip()

    #Basic lengths
    char_len = len(t_stripped)
    tokens = t_stripped.split()
    word_len = len(tokens)

    avg_word_len = (sum(len(tok) for tok in tokens) / word_len) if word_len > 0 else 0.0

    #Sentence count - splits on punctuation
    if not t_stripped:
        sent_len = 0
    else:
        sent_len = len([s for s in _SENT_SPLIT_RE.split(t_stripped) if s.strip()]) or 1
    
    #Punctuation / formatting
    num_qmarks = t_stripped.count("?")
    has_question = int(num_qmarks > 0)
    num_exclaims = t_stripped.count("!")

    #digits and caps
    digit_count = sum(ch.isdigit() for ch in t_stripped)
    uppercase_word_count = len(_ALLCAPS_WORD_RE.findall(t_stripped))

    return {
        "char_len": char_len,
        "word_len": word_len,
        "sent_len": sent_len,
        "avg_word_len": avg_word_len,
        "num_qmarks": num_qmarks,
        "has_question": has_question,
        "num_exclaims": num_exclaims,
        "newline_count": newline_count,
        "digit_count": digit_count,
        "uppercase_word_count": uppercase_word_count,        
    }

def add_structural_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    feats = df[text_col].apply(extract_structural_features).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


def main():
    base_path = PROCESSED_DIR / OUTPUT_FILENAME
    stem = base_path.stem

    for split in ["train", "val", "test"]:
        in_path = PROCESSED_DIR / f"{stem}_{split}.csv"

        # Feature-only output
        feat_path = PROCESSED_DIR / f"{stem}_{split}__struct.csv"

        df = pd.read_csv(in_path)

        # Compute features
        df_feat = add_structural_features(df, text_col="text")

        # Key + features 
        feature_cols = [
            "utterance_id", 
            "char_len", "word_len", "sent_len", "avg_word_len",
            "num_qmarks", "has_question", "num_exclaims",
            "newline_count", "digit_count", "uppercase_word_count",
        ]
        df_feat_out = df_feat[feature_cols]

        df_feat_out.to_csv(feat_path, index=False)
        print(f"[INFO] Wrote {len(df_feat_out)} rows of structural features to {feat_path}")
  

if __name__ == "__main__":
    main()