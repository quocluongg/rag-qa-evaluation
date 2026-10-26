"""
vector_rag.py — Vector RAG (no temporal filtering).

Retrieves top-k text_units by cosine similarity to query embedding,
using the LanceDB vector index rebuilt from output/text_units.parquet.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

from .base import BaseRetriever, Evidence, RetrievalResult
from .index_utils import ensure_text_unit_index, load_embedder
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

    def __init__(
        self,
        k: int = 5,
        embed_model: str = "BAAI/bge-m3",
        embedder=None,
        index_dir: Optional[Path] = None,
    ):
        super().__init__(k)
        self._embed_model_name = embed_model
        self._embedder = embedder
        self._index_dir = Path(index_dir) if index_dir is not None else LANCEDB_DIR
        self._table = None
        self._smap = None
        self._tu_text: dict[str, str] = {}
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        self._embedder = self._embedder or load_embedder(self._embed_model_name)
        self._smap = get_source_map()
        self._tu_text = dict(zip(self._smap.tu_df["id"].astype(str), self._smap.tu_df["text"].astype(str)))
        self._table = ensure_text_unit_index(self._embedder, db_dir=self._index_dir)
        self._initialized = True

    def ensure_ready(self):
        """Load the embedding model and verify/rebuild the LanceDB index."""
        self._lazy_init()

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
            distance = row.get("_distance")
            if distance is not None and distance == distance:
                score = 1.0 - float(distance)
            else:
                score = float(row.get("score", 0.0) or 0.0)
            content = self._tu_text.get(tu_id)
            if not content:
                table_text = row.get("text", "")
                content = "" if table_text is None or table_text != table_text else str(table_text)
            out.append(Evidence(
                text_unit_id=tu_id,
                filename=fname,
                content=content,
                score=score,
                valid_from=vf,
                valid_to=vt,
            ))
        return out
