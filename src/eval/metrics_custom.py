"""
metrics_custom.py — Deterministic retrieval metrics for the 2×2 benchmark.

All metrics operate at DOCUMENT level (filename strings), not text_unit level.
Evidence from all 4 systems is mapped to document filenames via source_map.py.

Metric definitions (from main.tex, Section IV):
  Precision@k = |top-k doc_ids ∩ gold_doc_ids| / k
  Recall@k    = |top-k doc_ids ∩ gold_doc_ids| / |gold_doc_ids|
  MRR         = 1 / rank_of_first_relevant_document
  TemporalAcc = |{d ∈ top-k : d ∈ gold_doc_ids AND V(d, tq)=1}| / k
                denominator is always k (fixed retrieval depth)

Graph Gain (eq.7):  Δ_G(c) = A_TG(c) - A_TV(c)  (answer accuracy per category)
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT     = Path(__file__).resolve().parents[2]
META_JSON = ROOT / "output" / "metadata.json"

_meta: dict | None = None

def _get_meta() -> dict:
    global _meta
    if _meta is None:
        _meta = json.load(open(META_JSON, encoding="utf-8"))
    return _meta


# ── Core retrieval metrics (document-level) ────────────────────

def precision_at_k(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
    k: int = 5,
) -> float:
    """
    P@k = |top-k retrieved ∩ gold| / k
    denominator is always k (even if fewer than k results returned).
    """
    top_k = retrieved_doc_ids[:k]
    gold  = set(gold_doc_ids)
    hits  = sum(1 for d in top_k if d in gold)
    return hits / k


def recall_at_k(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
    k: int = 5,
) -> float:
    """
    R@k = |top-k retrieved ∩ gold| / |gold|
    Returns 0.0 if gold is empty.
    """
    if not gold_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    gold  = set(gold_doc_ids)
    hits  = len(top_k & gold)
    return hits / len(gold)


def mrr(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
) -> float:
    """
    MRR = 1 / rank of first relevant document.
    Returns 0.0 if no relevant document found in the retrieved list.
    """
    gold = set(gold_doc_ids)
    for rank, d in enumerate(retrieved_doc_ids, start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def temporal_accuracy(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
    tq: Optional[str],
    k: int = 5,
    meta: Optional[dict] = None,
) -> float | None:
    """
    TemporalAcc = |{d ∈ top-k : d ∈ gold AND V(d, tq)=1}| / k
    denominator = k (fixed retrieval depth, as defined in paper).

    d is temporally valid when valid_from <= tq <= valid_to.
    valid_to="" means open-ended (valid until replaced).
    """
    if tq is None:
        return None
    if meta is None:
        meta = _get_meta()

    gold  = set(gold_doc_ids)
    top_k = retrieved_doc_ids[:k]

    try:
        tq_dt = datetime.strptime(tq, "%d/%m/%Y")
    except ValueError as exc:
        raise ValueError(f"Invalid tq format {tq!r}; expected dd/mm/YYYY") from exc

    count = 0
    for doc_id in top_k:
        if doc_id not in gold:
            continue
        entry  = meta.get(doc_id, {})
        vf_str = entry.get("valid_from", "")
        vt_str = entry.get("valid_to", "")
        try:
            vf_dt = datetime.strptime(vf_str, "%d/%m/%Y") if vf_str else datetime.min
            vt_dt = datetime.strptime(vt_str, "%d/%m/%Y") if vt_str else datetime.max
            if vf_dt <= tq_dt <= vt_dt:
                count += 1
        except ValueError as exc:
            raise ValueError(f"Invalid validity metadata for {doc_id!r}") from exc

    return count / k


# ── Graph Gain ────────────────────────────────────────────────

def graph_gain(
    tg_rag_scores: list[float],
    tv_rag_scores: list[float],
) -> float:
    """
    Δ_G = mean(A_TG) - mean(A_TV)
    Operates on per-query answer accuracy scores (0–1 floats from RAGAS).
    """
    if not tg_rag_scores or not tv_rag_scores:
        return float("nan")
    return sum(tg_rag_scores) / len(tg_rag_scores) - \
           sum(tv_rag_scores) / len(tv_rag_scores)


# ── Aggregate over dataset ────────────────────────────────────

def compute_all_metrics(
    results: list[dict],
    k: int = 5,
    meta: Optional[dict] = None,
) -> dict:
    """
    Compute mean P@k, R@k, MRR, TemporalAcc over a list of per-query result dicts.

    Each dict must have:
      retrieved_doc_ids : list[str]   — in ranked order
      gold_doc_ids      : list[str]
      tq                : str | None  — dd/mm/yyyy
      latency_s         : float
      ctx_tokens        : int

    Returns dict with mean values and per-category breakdown if 'category' present.
    """
    if meta is None:
        meta = _get_meta()

    p_list, r_list, mrr_list, ta_list = [], [], [], []
    latencies, tokens = [], []
    answer_scores = []
    cat_scores: dict[str, dict[str, list]] = {}

    for r in results:
        ret  = r.get("retrieved_doc_ids", [])
        gold = r.get("gold_doc_ids", [])
        tq   = r.get("tq")
        cat  = r.get("category", "all")

        p   = precision_at_k(ret, gold, k)
        rc  = recall_at_k(ret, gold, k)
        m   = mrr(ret, gold)
        ta  = temporal_accuracy(ret, gold, tq, k, meta)

        p_list.append(p);  r_list.append(rc)
        mrr_list.append(m); ta_list.append(ta)
        latencies.append(r.get("latency_s", 0.0))
        tokens.append(r.get("ctx_tokens", 0))
        if r.get("answer_accuracy") is not None:
            answer_scores.append(float(r["answer_accuracy"]))

        if cat not in cat_scores:
            cat_scores[cat] = {"p": [], "r": [], "mrr": [], "ta": []}
        cat_scores[cat]["p"].append(p)
        cat_scores[cat]["r"].append(rc)
        cat_scores[cat]["mrr"].append(m)
        cat_scores[cat]["ta"].append(ta)

    def mean(lst):
        valid = [x for x in lst if x is not None]
        return sum(valid) / len(valid) if valid else 0.0

    def mean_optional(lst):
        valid = [x for x in lst if x is not None]
        return sum(valid) / len(valid) if valid else None

    out = {
        "n":            len(results),
        "precision_at_k": mean(p_list),
        "recall_at_k":    mean(r_list),
        "mrr":            mean(mrr_list),
        "temporal_accuracy": mean_optional(ta_list),
        "retrieval_latency_s": mean(latencies),
        "ctx_tokens_mean":     mean(tokens),
        "answer_accuracy":     mean(answer_scores) if answer_scores else None,
        "per_category": {
            cat: {
                "precision_at_k":   mean(v["p"]),
                "recall_at_k":      mean(v["r"]),
                "mrr":              mean(v["mrr"]),
                "temporal_accuracy": mean_optional(v["ta"]),
                "n": len(v["p"]),
            }
            for cat, v in cat_scores.items()
        },
    }
    return out


# ── LaTeX table helpers ───────────────────────────────────────

def results_to_latex_row(system_name: str, metrics: dict) -> str:
    """Format one row of Table V (overall performance)."""
    def fmt(value, digits=3):
        return "--" if value is None else f"{value:.{digits}f}"
    return (
        f"{system_name:<22} & "
        f"{fmt(metrics['precision_at_k'])} & "
        f"{fmt(metrics['recall_at_k'])} & "
        f"{fmt(metrics['mrr'])} & "
        f"{fmt(metrics['temporal_accuracy'])} & "
        f"{fmt(metrics.get('answer_accuracy'))} & "
        f"{fmt(metrics['retrieval_latency_s'], 2)} " + chr(92) * 2
    )


def graph_gain_to_latex(
    tv_rag_cat: dict,
    tg_rag_cat: dict,
    categories: list[str] = ("factual", "temporal", "relational", "multi_hop", "historical"),
) -> str:
    """Format Table VI (graph gain per category)."""
    lines = []
    for cat in categories:
        tv_acc = tv_rag_cat.get(cat, {}).get("answer_accuracy", 0.0)
        tg_acc = tg_rag_cat.get(cat, {}).get("answer_accuracy", 0.0)
        delta  = tg_acc - tv_acc
        lines.append(
            f"{cat.capitalize():<12} & {tv_acc:.3f} & {tg_acc:.3f} & "
            f"{'+' if delta >= 0 else ''}{delta:.3f} \\\\"
        )
    return "\n".join(lines)
