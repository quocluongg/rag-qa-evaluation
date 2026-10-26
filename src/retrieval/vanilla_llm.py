"""
vanilla_llm.py — Baseline 1: Pure LLM (Parametric memory only, No Retrieval).
"""
from __future__ import annotations
from typing import Optional
from .base import BaseRetriever, Evidence, RetrievalResult

class VanillaLLMRetriever(BaseRetriever):
    """
    Dummy retriever for System 1 (LLM thuần).
    Returns zero evidence chunks, simulating generation without retrieval.
    """
    name = "vanilla_llm"

    def __init__(self, k: int = 0):
        super().__init__(k=0)

    def retrieve(self, query: str, tq: Optional[str] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            tq=tq,
            evidence=[],
            latency_s=0.0,
            ctx_tokens=0,
        )
