"""
graph_rag.py — Custom graph-aware retriever (no temporal filtering).
Baseline graph-side system in the 2×2 comparison.

Algorithm (described in paper subsection "Custom graph-aware retriever"):
  1. Entity lookup : embed query → cosine sim with entity embeddings → top-K seed entities.
  2. Subgraph expand: BFS depth-1 on relationships from seed entities.
  3. Text unit collect: gather all text_unit_ids from seed + neighbour entities.
  4. Re-rank : cosine(query, text_unit) → top-k text_units.
  5. Source map : text_unit_id → filename (document-level for metrics).

Entity embeddings: use pre-computed entity description embeddings if available
in LanceDB; otherwise embed entity names+descriptions on the fly.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseRetriever, Evidence, RetrievalResult
from .source_map import get_source_map

ROOT         = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = ROOT / "output"
ENTITIES_PQ  = OUTPUT_DIR / "entities.parquet"
RELS_PQ      = OUTPUT_DIR / "relationships.parquet"

# Number of seed entities from entity lookup
ENTITY_TOP_K = 10
# BFS depth
BFS_DEPTH    = 1


class GraphRAG(BaseRetriever):
    """
    Custom graph-aware retriever over GraphRAG parquet index.
    No temporal filtering applied.
    """
    name = "graph_rag"

    def __init__(self, k: int = 5, embed_model: str = "BAAI/bge-m3"):
        super().__init__(k)
        self._embed_model_name = embed_model
        self._embedder = None
        self._entities_df: pd.DataFrame | None = None
        self._rels_df:     pd.DataFrame | None = None
        self._tu_df:       pd.DataFrame | None = None
        self._smap = None

        # Pre-computed entity embeddings matrix (lazy)
        self._entity_vecs: np.ndarray | None = None
        self._entity_ids_list: list[str] = []

    def _lazy_init(self):
        if self._entities_df is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._embedder     = SentenceTransformer(self._embed_model_name)
        self._entities_df  = pd.read_parquet(ENTITIES_PQ)
        self._rels_df      = pd.read_parquet(RELS_PQ)
        self._smap         = get_source_map()
        self._tu_df        = self._smap.tu_df

        # Pre-compute entity embeddings from name + description
        self._build_entity_index()

    def _build_entity_index(self):
        """Embed entity (name + description) for fast cosine lookup."""
        ent_texts = []
        ent_ids   = []
        for _, row in self._entities_df.iterrows():
            name = str(row.get("name", "") or "")
            desc = str(row.get("description", "") or "")
            ent_texts.append(f"{name}. {desc}"[:512])
            ent_ids.append(row["id"])
        if ent_texts:
            vecs = self._embedder.encode(
                ent_texts, normalize_embeddings=True,
                show_progress_bar=False, batch_size=64
            )
            self._entity_vecs      = vecs
            self._entity_ids_list  = ent_ids

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(text, normalize_embeddings=True)

    def _entity_lookup(self, q_vec: np.ndarray) -> list[str]:
        """Return top-ENTITY_TOP_K entity ids by cosine similarity."""
        if self._entity_vecs is None or len(self._entity_vecs) == 0:
            return []
        sims  = self._entity_vecs @ q_vec
        top_i = np.argsort(sims)[::-1][:ENTITY_TOP_K]
        return [self._entity_ids_list[i] for i in top_i]

    def _bfs_expand(self, seed_entity_ids: list[str]) -> set[str]:
        """BFS depth-1: collect seed + their 1-hop neighbours via relationships."""
        all_entity_ids = set(seed_entity_ids)
        seed_set       = set(seed_entity_ids)

        # Build adjacency: entity_id → {neighbour entity_ids}
        # relationships.parquet columns: source, target (entity names or ids)
        # We need to match by entity id — check column names
        rels = self._rels_df

        # Try common column name patterns
        src_col = "source" if "source" in rels.columns else "source_id"
        tgt_col = "target" if "target" in rels.columns else "target_id"

        if src_col in rels.columns and tgt_col in rels.columns:
            for _, rel in rels.iterrows():
                s = str(rel[src_col])
                t = str(rel[tgt_col])
                if s in seed_set:
                    all_entity_ids.add(t)
                elif t in seed_set:
                    all_entity_ids.add(s)

        return all_entity_ids

    def _collect_tu_ids(
        self, entity_ids: set[str], valid_filenames: set[str] | None = None
    ) -> list[str]:
        """
        Collect text_unit_ids from all entities in entity_ids.
        If valid_filenames is given, restrict to text_units from those docs.
        """
        tu_ids: set[str] = set()
        for _, row in self._tu_df.iterrows():
            if valid_filenames:
                fname = self._smap.tu_to_filename(row["id"])
                if fname not in valid_filenames:
                    continue
            eids = row["entity_ids"]
            if eids is None:
                continue
            eids_set = set(eids.tolist() if hasattr(eids, "tolist") else list(eids))
            if eids_set & entity_ids:
                tu_ids.add(row["id"])
        return list(tu_ids)

    def _rerank_tus(
        self, tu_ids: list[str], q_vec: np.ndarray
    ) -> list[tuple[str, float]]:
        """Embed text_units and return [(tu_id, score)] sorted by score desc."""
        rows = self._tu_df[self._tu_df["id"].isin(set(tu_ids))]
        if rows.empty:
            return []
        texts  = rows["text"].tolist()
        t_vecs = self._embedder.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=32
        )
        sims = t_vecs @ q_vec
        pairs = list(zip(rows["id"].tolist(), sims.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def retrieve(
        self, query: str, tq: Optional[str] = None,
        valid_filenames: set[str] | None = None
    ) -> RetrievalResult:
        self._lazy_init()
        smap = self._smap
        t0   = time.perf_counter()

        q_vec = self._embed(query)

        # Step 1: entity lookup
        seed_ids = self._entity_lookup(q_vec)

        # Step 2: BFS expand
        expanded_ids = self._bfs_expand(seed_ids)

        # Step 3: collect text units
        tu_ids = self._collect_tu_ids(expanded_ids, valid_filenames)

        if not tu_ids:
            # Fallback: return evidence from seed entities only (no neighbours)
            tu_ids = self._collect_tu_ids(set(seed_ids), valid_filenames)

        # Step 4: re-rank
        ranked = self._rerank_tus(tu_ids, q_vec)[: self.k]

        latency = time.perf_counter() - t0

        # Step 5: build Evidence objects
        evidence: list[Evidence] = []
        for tu_id, score in ranked:
            fname = smap.tu_to_filename(tu_id) or ""
            vf, vt = smap.get_validity(fname)
            tu_row = self._tu_df[self._tu_df["id"] == tu_id]
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
            query=query,
            tq=None,
            evidence=evidence,
            latency_s=latency,
            ctx_tokens=ctx_tokens,
        )
