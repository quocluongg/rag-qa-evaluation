"""
temporal_vector_rag.py — Temporal Vector RAG.

Extends VectorRAG with pre-retrieval temporal filtering:
only text_units from documents valid at query time tq are searched.

Temporal validity: V(d_i, t_q) = 1 iff t_start <= t_q <= t_end  (eq.2 in main.tex)
"""
from __future__ import annotations
import time
from typing import Optional

from .base import BaseRetriever, Evidence, RetrievalResult
from .source_map import get_source_map
from .vector_rag import VectorRAG, LANCEDB_DIR, LANCEDB_TABLE


class TemporalVectorRAG(VectorRAG):
    """
    Temporal filtering + vector retrieval.
    Primary vector-side system in the 2×2 comparison.

    Strategy: pre-filter the LanceDB table to only rows whose source document
    is temporally valid at tq, then run ANN search on the filtered set.
    """
    name = "temporal_vector_rag"

    def retrieve(self, query: str, tq: Optional[str] = None) -> RetrievalResult:
        self._lazy_init()
        smap = self._smap

        if tq is None:
            # Degrade gracefully to non-temporal if tq missing
            return super().retrieve(query, tq)

        t0 = time.perf_counter()

        # Build set of valid filenames at tq
        valid_files = smap.valid_filenames_at(tq)

        q_vec = self._embed(query)

        # Retrieve more candidates then filter by temporal validity
        # (LanceDB does not support metadata pre-filter natively in all versions;
        #  over-fetch and post-filter is reliable across versions)
        OVERSAMPLE = 5
        candidates = (
            self._table
            .search(q_vec)
            .limit(self.k * OVERSAMPLE)
            .to_pandas()
        )

        # Post-filter: keep only text_units from valid documents
        filtered_rows = []
        for _, row in candidates.iterrows():
            tu_id = str(row.get("id", row.get("text_unit_id", "")))
            fname = smap.tu_to_filename(tu_id) or row.get("filename", "")
            if fname in valid_files:
                filtered_rows.append(row)
            if len(filtered_rows) >= self.k:
                break

        latency = time.perf_counter() - t0

        # Convert to Evidence
        evidence: list[Evidence] = []
        for row in filtered_rows:
            tu_id = str(row.get("id", row.get("text_unit_id", "")))
            fname = smap.tu_to_filename(tu_id) or row.get("filename", "")
            vf, vt = smap.get_validity(fname)
            score = float(row.get("_distance", row.get("score", 0.0)))
            score = 1.0 - score if score > 0 else score
            evidence.append(Evidence(
                text_unit_id=tu_id,
                filename=fname,
                content=str(row.get("text", "")),
                score=score,
                valid_from=vf,
                valid_to=vt,
            ))

        ctx_tokens = sum(len(ev.content.split()) for ev in evidence)

        return RetrievalResult(
            query=query,
            tq=tq,
            evidence=evidence,
            latency_s=latency,
            ctx_tokens=ctx_tokens,
        )
