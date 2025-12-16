from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from convokit import Corpus, download

from ..config import SUBREDDITS, OUTPUT_FILENAME, DELETED_MARKERS
from ..utils.paths import PROCESSED_DIR, CONVOKIT_DIR

def is_valid_text(text: Any) -> bool:
    """Return True if the text is available. """

    if text is None:
        return False
    text = str(text).strip()
    return text not in DELETED_MARKERS and len(text) > 0

def get_speaker_id(utt) -> str | None:
    """
    Get a speaker id from a ConvoKit utterance, 
    handling older and newer representations 
    """

    if hasattr(utt, "user") and utt.user not in ("[deleted]", None):
        return utt.user

    speaker = getattr(utt, "speaker", None)
    if speaker is not None:
        sid = getattr(speaker, "id", None)
        if sid not in ("[deleted]", None):
            return sid
    
    return None

def corpus_to_rows(corpus: Corpus, default_subreddit: str) -> List[Dict[str, Any]]:
    """
    Convert a ConvoKit Corpus into cleaned rows for one subreddit.
    Each row has:
        - utterance_id
        - conversation_id
        - speaker_id
        - text
        - subreddit
    """
    rows: List[Dict[str, Any]] = []

    for utt in corpus.iter_utterances():
        text = utt.text

        if not is_valid_text(text):
            continue

        text = str(text).replace("\n", " ").strip()

        meta = utt.meta or {}

        row = {
            "utterance_id": utt.id, 
            "conversation_id": getattr(utt, "root", None)
                                or getattr(utt, "conversation_id", None)
                                or utt.id, 
            "speaker_id": get_speaker_id(utt), 
            "text": text, 
            "subreddit": default_subreddit, 
        }
        rows.append(row)

    return rows

def make_balanced_by_minimum(df: pd.DataFrame, label_col: str, seed: int = 42) -> pd.DataFrame:
    """
    Downsample all classes in 'label_col' to the size of the smallest class.
    Returns a new, shuffled DataFrame.
    """
    counts = df[label_col].value_counts()
    print("[INFO] Class distribution before balancing:")
    print(counts)

    min_count = counts.min()
    print(f"[INFO] Smallest class size: {min_count}")

    balanced_parts = []
    for label, count in counts.items():
        df_label = df[df[label_col] == label]

        if count > min_count:
            df_label_balanced = df_label.sample(n=min_count, random_state=seed)
        else:
            df_label_balanced = df_label

        balanced_parts.append(df_label_balanced)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print("[INFO] Class distribution after balancing:")
    print(balanced_df[label_col].value_counts())

    return balanced_df


def load_and_clean_subreddit(subreddit: str) -> pd.DataFrame:
    """
    Download one subreddit corpus via ConvoKit and return a cleaned DataFrame
    """

    print(f"[INFO] Downloading/loading ConvoKit corpus for r/{subreddit}...")
    filename = download(f"subreddit-{subreddit}",data_dir=str(CONVOKIT_DIR))
    corpus = Corpus(filename=filename)

    rows = corpus_to_rows(corpus, default_subreddit=subreddit)
    df = pd.DataFrame(rows)
    print(f"[INFO] r/{subreddit}: kept {len(df)} cleaned utterances.")
    return df

def load_and_clean_all_subreddits() -> pd.DataFrame:
    """
    Load, clean, and merge all subreddits listed in config.SUBREDDITS.
    Then balance the dataset by downsampling all subreddits to the size of
    the smallest one, and save the balanced CSV under
    PROCESSED_DIR / OUTPUT_FILENAME.

    Returns the balanced DataFrame.
    """
    dfs = []

    for subreddit in SUBREDDITS:
        df_sub = load_and_clean_subreddit(subreddit)
        dfs.append(df_sub)

    if not dfs:
        print("[WARN] No data collected from any subreddit.")
        return pd.DataFrame(
            columns=["utterance_id", "conversation_id", "speaker_id", "text", "subreddit", "timestamp"]
        )

    merged = pd.concat(dfs, ignore_index=True)

    balanced = make_balanced_by_minimum(merged, label_col="subreddit", seed=42)

    output_path = PROCESSED_DIR / OUTPUT_FILENAME
    balanced.to_csv(output_path, index=False)

    print(
        f"[INFO] Saved {len(balanced)} balanced cleaned utterances "
        f"from {len(SUBREDDITS)} subreddits to {output_path}"
    )
    return balanced


if __name__ == "__main__":
    load_and_clean_all_subreddits()