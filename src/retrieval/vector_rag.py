"""
vector_rag.py — Vector RAG (no temporal filtering).

Retrieves top-k text_units by cosine similarity to query embedding,
using the LanceDB vector index built by reembed_bge_m3.py.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

from .base import BaseRetriever, Evidence, RetrievalResult
from .source_map import get_source_map

ROOT      = Path(__file__).resolve().parents[2]
LANCEDB_DIR = ROOT / "output" / "lancedb"

# LanceDB table name written by reembed_bge_m3.py
LANCEDB_TABLE = "text_unit_text"


class VectorRAG(BaseRetriever):
    """
    Semantic vector retrieval over all text_units (no temporal constraint).
    Baseline system in the 2×2 comparison.
    """
    name = "vector_rag"

    def __init__(self, k: int = 5, embed_model: str = "BAAI/bge-m3"):
        super().__init__(k)
        self._embed_model_name = embed_model
        self._embedder = None
        self._db = None
        self._table = None
        self._smap = None

    def _lazy_init(self):
        if self._db is not None:
            return
        import lancedb
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self._embed_model_name)
        self._db = lancedb.connect(str(LANCEDB_DIR))
        self._table = self._db.open_table(LANCEDB_TABLE)
        self._smap = get_source_map()

    def _embed(self, text: str) -> list[float]:
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def retrieve(self, query: str, tq: Optional[str] = None) -> RetrievalResult:
        self._lazy_init()
        t0 = time.perf_counter()

        q_vec = self._embed(query)
        # LanceDB ANN search — returns pandas DataFrame
        results = (
            self._table
            .search(q_vec)
            .limit(self.k)
            .to_pandas()
        )

        latency = time.perf_counter() - t0
        evidence = self._rows_to_evidence(results)
        ctx_tokens = sum(ev.content.split().__len__() for ev in evidence)  # approx

        return RetrievalResult(
            query=query,
            tq=None,
            evidence=evidence,
            latency_s=latency,
            ctx_tokens=ctx_tokens,
        )

    def _rows_to_evidence(self, df) -> list[Evidence]:
        smap = self._smap
        out: list[Evidence] = []
        for _, row in df.iterrows():
            tu_id = str(row.get("id", row.get("text_unit_id", "")))
            fname = smap.tu_to_filename(tu_id) or row.get("filename", "")
            vf, vt = smap.get_validity(fname) if fname else ("", "")
            score = float(row.get("_distance", row.get("score", 0.0)))
            # LanceDB returns L2 distance; convert to similarity proxy
            score = 1.0 - score if score > 0 else score
            out.append(Evidence(
                text_unit_id=tu_id,
                filename=fname,
                content=str(row.get("text", "")),
                score=score,
                valid_from=vf,
                valid_to=vt,
            ))
        return out
