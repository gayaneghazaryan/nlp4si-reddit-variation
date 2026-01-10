from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


NER_TYPES = ["PERSON", "ORG", "GPE", "DATE", "MONEY"]


def extract_ner_features_from_doc(doc) -> Dict[str, Any]:
    """
    Extract minimal NER features from a spaCy Doc.
    We use rates per token to stay length-robust and domain-general.
    """
    tokens = [t for t in doc if not t.is_space]
    n_tok = len(tokens)

    ents = list(doc.ents) if getattr(doc, "ents", None) is not None else []
    n_ent = len(ents)

    feats: Dict[str, Any] = {}
    feats["ner_token_count"] = n_tok
    feats["ner_entity_count"] = n_ent
    feats["ner_entity_rate"] = _safe_div(n_ent, n_tok)

    # entity-type counts and rates
    type_counts = {t: 0 for t in NER_TYPES}
    for e in ents:
        if e.label_ in type_counts:
            type_counts[e.label_] += 1

    for t in NER_TYPES:
        feats[f"ner_{t}_count"] = type_counts[t]
        feats[f"ner_{t}_rate"] = _safe_div(type_counts[t], n_tok)

    return feats


def add_ner_features(
    df: pd.DataFrame,
    text_col: str = "text",
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 128,
    n_process: int = 1,
) -> pd.DataFrame:
    """
    Adds minimal NER features using spaCy.

    Notes:
      - NER is slower than POS; keep batch_size modest.
      - We disable the dependency parser to save time; NER does not require it.
    """
    import spacy

    # Keep NER enabled; disable parser for speed.
    nlp = spacy.load(spacy_model, disable=["parser"])

    texts: List[str] = ["" if pd.isna(x) else str(x) for x in df[text_col].tolist()]

    rows: List[Dict[str, Any]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        rows.append(extract_ner_features_from_doc(doc))

    feats = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
