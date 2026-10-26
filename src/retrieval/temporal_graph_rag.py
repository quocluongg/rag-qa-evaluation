"""
temporal_graph_rag.py — Temporal GraphRAG.
Primary graph-side system in the 2×2 comparison.

Extends GraphRAG by applying temporal pruning before retrieval:
only entities/text_units from temporally valid documents are considered.

Temporal pruning (eq.3 in main.tex):
  G_tq = (V_tq, E_tq)
  V_tq: entities whose text_unit_ids include ≥1 unit from a valid document
  E_tq: relationships where both endpoints are in V_tq
  Pruning creates a query-specific view; original graph is not modified.
"""
from __future__ import annotations
from typing import Optional

from .base import RetrievalResult
from .graph_rag import GraphRAG
from .source_map import get_source_map


class TemporalGraphRAG(GraphRAG):
    """
    Temporal GraphRAG: prune graph view to G_tq before entity lookup and BFS.
    """
    name = "temporal_graph_rag"

    def retrieve(self, query: str, tq: Optional[str] = None) -> RetrievalResult:
        self._lazy_init()
        smap = get_source_map()
        import time
        t0 = time.perf_counter()

        if tq is None:
            # Degrade to non-temporal graph RAG if tq missing
            return super().retrieve(query, tq=None)

        # Temporal pruning: set of valid filenames at tq
        valid_filenames = smap.valid_filenames_at(tq)

        # Restrict entity lookup: build a masked entity index
        # Only consider entities whose text_units come from valid documents
        import numpy as np
        q_vec = self._embed(query)

        # Build valid entity set: entities with ≥1 text_unit from a valid doc
        valid_entity_ids: set[str] = set()
        for _, row in self._tu_df.iterrows():
            fname = smap.tu_to_filename(row["id"])
            if fname not in valid_filenames:
                continue
            eids = row["entity_ids"]
            if eids is None:
                continue
            eids_iter = eids.tolist() if hasattr(eids, "tolist") else list(eids)
            valid_entity_ids.update(eids_iter)

        if not valid_entity_ids:
            # No valid evidence at tq — return empty result
            from .base import Evidence, RetrievalResult
            return RetrievalResult(
                query=query, tq=tq, evidence=[], latency_s=0.0, ctx_tokens=0
            )

        # Filter entity index to valid entities
        orig_vecs = self._entity_vecs
        orig_ids  = self._entity_ids_list
        mask = [i for i, eid in enumerate(orig_ids) if eid in valid_entity_ids]
        if not mask:
            return super().retrieve(query, tq=None, valid_filenames=valid_filenames)

        filtered_vecs = orig_vecs[mask]
        filtered_ids  = [orig_ids[i] for i in mask]

        # Cosine similarity in filtered entity space
        sims  = filtered_vecs @ q_vec
        top_i = np.argsort(sims)[::-1][:10]  # ENTITY_TOP_K
        seed_ids = [filtered_ids[i] for i in top_i]

        # BFS on full graph but restrict to valid_entity_ids
        expanded_raw = self._bfs_expand(seed_ids)
        expanded_ids = expanded_raw & valid_entity_ids

        # Collect text_units restricted to valid documents
        tu_ids = self._collect_tu_ids(expanded_ids, valid_filenames)

        if not tu_ids:
            tu_ids = self._collect_tu_ids(set(seed_ids), valid_filenames)

        # Re-rank and build result (same as GraphRAG)
        ranked = self._rerank_tus(tu_ids, q_vec)[: self.k]
        latency = time.perf_counter() - t0

        from .base import Evidence, RetrievalResult
        evidence: list[Evidence] = []
        for tu_id, score in ranked:
            fname = smap.tu_to_filename(tu_id) or ""
            vf, vt = smap.get_validity(fname)
            tu_row  = self._tu_df[self._tu_df["id"] == tu_id]
            content = str(tu_row["text"].iloc[0]) if not tu_row.empty else ""
            eids_raw = tu_row["entity_ids"].iloc[0] if not tu_row.empty else []
            eids = eids_raw.tolist() if hasattr(eids_raw, "tolist") else list(eids_raw or [])
            evidence.append(Evidence(
                text_unit_id=tu_id,
                filename=fname,
                content=content,
                score=float(score),
                valid_from=vf,
                valid_to=vt,
                entity_ids=eids,
            ))

        ctx_tokens = sum(len(ev.content.split()) for ev in evidence)
        return RetrievalResult(
            query=query, tq=tq, evidence=evidence,
            latency_s=latency, ctx_tokens=ctx_tokens,
        )
