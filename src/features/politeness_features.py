from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# ConvoKit politeness extraction requires parsed dependency info in utterance meta ("parsed")
# We provide it via TextParser, then run PolitenessStrategies.
from convokit import Corpus, Utterance, Speaker
from convokit import TextParser, PolitenessStrategies


def _safe_str(x: Any) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def _build_corpus_from_df(
    df: pd.DataFrame,
    utterance_id_col: str = "utterance_id",
    text_col: str = "text",
    speaker_col: Optional[str] = "speaker_id",
    convo_col: Optional[str] = "conversation_id",
) -> Corpus:
    """
    Build a lightweight ConvoKit Corpus from a DataFrame.

    Notes:
      - speaker_id can be missing; we fall back to a dummy speaker.
      - conversation_id can be missing; we fall back to utterance_id.
    """
    speakers: Dict[str, Speaker] = {}
    utterances: List[Utterance] = []

    for _, row in df.iterrows():
        utt_id = _safe_str(row.get(utterance_id_col))
        text = _safe_str(row.get(text_col)).strip()

        # ConvoKit prefers non-empty; keep empty as-is but it'll yield sparse features.
        spk_id = _safe_str(row.get(speaker_col)) if speaker_col else ""
        if not spk_id:
            spk_id = "__no_speaker__"
        if spk_id not in speakers:
            speakers[spk_id] = Speaker(id=spk_id)

        convo_id = _safe_str(row.get(convo_col)) if convo_col else ""
        if not convo_id:
            convo_id = utt_id

        utt = Utterance(
            id=utt_id,
            speaker=speakers[spk_id],
            conversation_id=convo_id,
            reply_to=None,
            text=text,
        )
        utterances.append(utt)

    return Corpus(utterances=utterances)


def _extract_polite_rows(corpus: Corpus, strategy_attr: str = "politeness_strategies") -> pd.DataFrame:
    """
    After running PolitenessStrategies, each utterance has a meta field
    (default name: "politeness_strategies") containing a dict of strategy features.
    We export these into a DataFrame keyed by utterance_id.
    """
    rows: List[Dict[str, Any]] = []
    all_keys: set[str] = set()

    # first pass: collect keys
    for utt in corpus.iter_utterances():
        feats = utt.meta.get(strategy_attr, {}) or {}
        if isinstance(feats, dict):
            all_keys.update(feats.keys())

    # second pass: make rows with consistent columns
    keys_sorted = sorted(all_keys)
    for utt in corpus.iter_utterances():
        feats = utt.meta.get(strategy_attr, {}) or {}
        if not isinstance(feats, dict):
            feats = {}

        row: Dict[str, Any] = {"utterance_id": utt.id}
        # prefix columns to avoid collisions with other feature groups
        for k in keys_sorted:
            row[f"polite_{k}"] = float(feats.get(k, 0.0) or 0.0)

        # handy summaries (robust even if values are counts or 0/1)
        vals = [row[f"polite_{k}"] for k in keys_sorted]
        row["polite_sum"] = float(sum(vals))
        row["polite_nonzero"] = float(sum(1 for v in vals if v != 0.0))
        rows.append(row)

    return pd.DataFrame(rows)


def add_politeness_features(
    df: pd.DataFrame,
    text_col: str = "text",
    utterance_id_col: str = "utterance_id",
    speaker_col: str = "speaker_id",
    convo_col: str = "conversation_id",
    spacy_model: str = "en_core_web_sm",
    strategy_collection: str = "politeness_api",
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Add ConvoKit politeness strategy features.

    Pipeline:
      1) Build a Corpus from df
      2) TextParser -> creates utterance.meta["parsed"] (dependency parses)
      3) PolitenessStrategies -> creates utterance.meta["politeness_strategies"]
      4) Export to DataFrame and merge back on utterance_id

    ConvoKit notes:
      - PolitenessStrategies.transform requires parses in the "parsed" field by default. :contentReference[oaicite:1]{index=1}
      - TextParser is the standard way to create that parse metadata. :contentReference[oaicite:2]{index=2}
    """
    import spacy

    # Build Corpus
    corpus = _build_corpus_from_df(
        df,
        utterance_id_col=utterance_id_col,
        text_col=text_col,
        speaker_col=speaker_col,
        convo_col=convo_col,
    )

    # Parse
    nlp = spacy.load(spacy_model)  # needs parser enabled for dependency parses
    parser = TextParser(output_field="parsed", spacy_nlp=nlp, verbosity=verbose)
    corpus = parser.transform(corpus)

    # Politeness strategies
    ps = PolitenessStrategies(
        parse_attribute_name="parsed",
        strategy_attribute_name="politeness_strategies",
        strategy_collection=strategy_collection,
        verbose=verbose,
    )
    corpus = ps.transform(corpus, markers=False)

    polite_df = _extract_polite_rows(corpus, strategy_attr="politeness_strategies")

    base = df[[utterance_id_col]].copy()
    out = base.merge(polite_df, left_on=utterance_id_col, right_on="utterance_id", how="left")

    # fill missing (should not happen, but safe)
    for c in out.columns:
        if c.startswith("polite_") or c in {"polite_sum", "polite_nonzero"}:
            out[c] = out[c].fillna(0.0)

    return out
