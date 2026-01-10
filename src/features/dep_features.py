from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _token_depth(token) -> int:
    """Depth of token in dependency tree (root depth = 0)."""
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
    return depth


def extract_dep_features_from_doc(doc) -> Dict[str, Any]:
    """
    Extract dependency-based syntactic features from a spaCy Doc.
    """
    tokens = [t for t in doc if not t.is_space]
    n_tok = len(tokens)

    feats: Dict[str, Any] = {}

    # -----------------------------
    # Dependency distance & depth
    # -----------------------------
    dep_distances = []
    depths = []

    for t in tokens:
        if t.head is not None:
            dep_distances.append(abs(t.i - t.head.i))
            depths.append(_token_depth(t))

    feats["dep_avg_dep_distance"] = _safe_div(sum(dep_distances), len(dep_distances))
    feats["dep_max_dep_distance"] = max(dep_distances) if dep_distances else 0.0
    feats["dep_avg_tree_depth"] = _safe_div(sum(depths), len(depths))
    feats["dep_max_tree_depth"] = max(depths) if depths else 0.0

    # -----------------------------
    # Dependency label rates
    # -----------------------------
    dep_labels = [
        "advcl", "ccomp", "xcomp", "acl",
        "cc", "conj",
        "nmod", "compound", "amod",
    ]

    dep_counts = {lbl: 0 for lbl in dep_labels}
    for t in tokens:
        if t.dep_ in dep_counts:
            dep_counts[t.dep_] += 1

    for lbl in dep_labels:
        feats[f"dep_{lbl}_rate"] = _safe_div(dep_counts[lbl], n_tok)

    # Reference count
    feats["dep_token_count"] = n_tok

    return feats


def add_dep_features(
    df: pd.DataFrame,
    text_col: str = "text",
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 128,
    n_process: int = 1,
) -> pd.DataFrame:
    """
    Adds dependency-based features using spaCy (parser required).
    """
    import spacy

    nlp = spacy.load(spacy_model, disable=["ner"])  # parser ON
    texts: List[str] = ["" if pd.isna(x) else str(x) for x in df[text_col].tolist()]

    rows: List[Dict[str, Any]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        rows.append(extract_dep_features_from_doc(doc))

    feats = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
