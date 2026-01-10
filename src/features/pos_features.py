from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import spacy


# Simple person pronoun sets (domain-general)
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
THIRD_PERSON = {
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def extract_pos_features_from_doc(doc) -> Dict[str, Any]:
    """
    Extract POS-based features from a spaCy Doc.

    Modal verbs are detected using the Penn Treebank tag:
        token.tag_ == "MD"
    """
    tokens = [t for t in doc if not t.is_space]
    n_tok = len(tokens)

    # POS counts
    pos_counts: Dict[str, int] = {}
    for t in tokens:
        pos = t.pos_
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    feats: Dict[str, Any] = {}

    # Core POS tag rates
    core_pos = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "AUX", "ADP", "DET", "CCONJ", "SCONJ", "NUM", "PROPN"]
    for p in core_pos:
        feats[f"pos_{p}_rate"] = _safe_div(pos_counts.get(p, 0), n_tok)

    # Information density / narrativity proxies
    feats["pos_noun_verb_ratio"] = _safe_div(
        pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0),
        pos_counts.get("VERB", 0),
    )

    # Pronoun person rates
    lower_tokens = [t.text.lower() for t in tokens]
    first = sum(1 for w in lower_tokens if w in FIRST_PERSON)
    second = sum(1 for w in lower_tokens if w in SECOND_PERSON)
    third = sum(1 for w in lower_tokens if w in THIRD_PERSON)

    feats["pos_first_person_rate"] = _safe_div(first, n_tok)
    feats["pos_second_person_rate"] = _safe_div(second, n_tok)
    feats["pos_third_person_rate"] = _safe_div(third, n_tok)

    # Modal verbs via Penn Treebank tag (MD)
    modals = sum(1 for t in tokens if t.tag_ == "MD")
    feats["pos_modal_rate"] = _safe_div(modals, n_tok)

    # Tense proxy via morphology (when available)
    past = 0
    pres = 0
    for t in tokens:
        if t.pos_ in {"VERB", "AUX"}:
            morph = t.morph
            if "Tense=Past" in morph:
                past += 1
            if "Tense=Pres" in morph:
                pres += 1

    feats["pos_past_tense_rate"] = _safe_div(past, n_tok)
    feats["pos_pres_tense_rate"] = _safe_div(pres, n_tok)

    # Token count as reference/debug feature 
    feats["pos_token_count"] = n_tok

    return feats


def add_pos_features(
    df: pd.DataFrame,
    text_col: str = "text",
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 256,
    n_process: int = 1,
) -> pd.DataFrame:
    """
    Adds POS/morphosyntactic features using spaCy (tagger + lemmatizer).
    Uses nlp.pipe for efficient batching.
    """

    nlp = spacy.load(spacy_model, disable=["ner", "parser"])
    texts: List[str] = ["" if pd.isna(x) else str(x) for x in df[text_col].tolist()]

    rows: List[Dict[str, Any]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        rows.append(extract_pos_features_from_doc(doc))

    feats = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
