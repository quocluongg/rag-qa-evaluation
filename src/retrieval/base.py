"""
base.py — Abstract base class and Evidence dataclass for all 4 retrievers.

Evidence is always at text_unit level; source_map maps to document_ids for metrics.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Evidence:
    """
    A single retrieved piece of evidence.
    Retrieved at text_unit granularity; mapped to document level for metrics.
    """
    text_unit_id: str           # ID from text_units.parquet
    filename:     str           # source document filename (e.g. CTDT_AIoT_2025.md)
    content:      str           # text content of the text unit
    score:        float = 0.0   # cosine similarity or relevance score
    valid_from:   str  = ""     # from metadata.json
    valid_to:     str  = ""     # from metadata.json
    entity_ids:   list = field(default_factory=list)

    @property
    def document_id(self) -> str:
        """Alias: filename used as document_id throughout evaluation."""
        return self.filename


@dataclass
class RetrievalResult:
    """Output of a retriever for a single query."""
    query:      str
    tq:         Optional[str]        # query time; None for non-temporal systems
    evidence:   list[Evidence]       # ordered top-k evidence
    latency_s:  float = 0.0          # retrieval-only latency (seconds)
    ctx_tokens: int   = 0            # total tokens in returned context

    @property
    def document_ids(self) -> list[str]:
        """Unique document filenames in ranked order (deduped, order-preserving)."""
        seen: set[str] = set()
        out: list[str] = []
        for ev in self.evidence:
            if ev.filename not in seen:
                seen.add(ev.filename)
                out.append(ev.filename)
        return out


class BaseRetriever(ABC):
    """Common interface for all 4 retrieval systems."""

    name: str = "base"          # override in subclasses

    def __init__(self, k: int = 5):
        self.k = k

    @abstractmethod
    def retrieve(self, query: str, tq: Optional[str] = None) -> RetrievalResult:
        """
        Parameters
        ----------
        query : natural-language query string
        tq    : query time in dd/mm/yyyy format; None for non-temporal retrievers

        Returns
        -------
        RetrievalResult with top-k Evidence objects
        """
        ...

    def batch_retrieve(
        self, queries: list[str], tqs: list[Optional[str]]
    ) -> list[RetrievalResult]:
        return [self.retrieve(q, t) for q, t in zip(queries, tqs)]
