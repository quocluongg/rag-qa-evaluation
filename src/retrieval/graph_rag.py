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
from .index_utils import encode_texts, ensure_text_unit_index, load_embedder
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
        self._index_dir = Path(index_dir) if index_dir is not None else None
        self._entities_df: pd.DataFrame | None = None
        self._rels_df:     pd.DataFrame | None = None
        self._tu_df:       pd.DataFrame | None = None
        self._smap = None

        # Pre-computed entity embeddings matrix (lazy)
        self._entity_vecs: np.ndarray | None = None
        self._entity_ids_list: list[str] = []
        self._entity_id_to_name: dict[str, str] = {}
        self._entity_name_to_id: dict[str, str] = {}

        # Pre-computed text-unit embeddings loaded from LanceDB (lazy)
        self._tu_vecs: np.ndarray | None = None
        self._tu_id_to_row: dict[str, int] = {}

    def _lazy_init(self):
        if self._entities_df is not None:
            return
        self._embedder     = self._embedder or load_embedder(self._embed_model_name)
        self._entities_df  = pd.read_parquet(ENTITIES_PQ)
        self._rels_df      = pd.read_parquet(RELS_PQ)
        self._smap         = get_source_map()
        self._tu_df        = self._smap.tu_df

        title_col = "title" if "title" in self._entities_df.columns else "name"
        for _, row in self._entities_df.iterrows():
            entity_id = str(row["id"])
            entity_name = str(row.get(title_col, "") or "")
            if entity_name:
                self._entity_id_to_name[entity_id] = entity_name
                self._entity_name_to_id.setdefault(entity_name, entity_id)

        # Pre-compute entity embeddings from name + description
        self._build_entity_index()

    def _build_entity_index(self):
        """Embed entity (name + description) for fast cosine lookup."""
        ent_texts = []
        ent_ids   = []
        title_col = "title" if "title" in self._entities_df.columns else "name"
        for _, row in self._entities_df.iterrows():
            name = str(row.get(title_col, "") or "")
            desc = str(row.get("description", "") or "")
            ent_texts.append(f"{name}. {desc}"[:512])
            ent_ids.append(row["id"])
        if ent_texts:
            vecs = encode_texts(
                self._embedder,
                ent_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            self._entity_vecs      = np.asarray(vecs, dtype=np.float32)
            self._entity_ids_list  = ent_ids

    def _build_text_index(self) -> None:
        """Load pre-computed text-unit vectors so reranking needs no GPU encode."""
        if self._tu_vecs is not None:
            return
        table = ensure_text_unit_index(self._embedder, db_dir=self._index_dir)
        df = table.to_pandas()
        if "id" not in df.columns or "vector" not in df.columns or df.empty:
            return
        ids = df["id"].astype(str).tolist()
        vectors = [
            np.asarray(vector, dtype=np.float32)
            for vector in df["vector"].tolist()
        ]
        if not vectors:
            return
        self._tu_vecs = np.vstack(vectors)
        self._tu_id_to_row = {tu_id: i for i, tu_id in enumerate(ids)}

    def _embed(self, text: str) -> np.ndarray:
        vec = encode_texts(self._embedder, text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def _entity_lookup(self, q_vec: np.ndarray) -> list[str]:
        """Return top-ENTITY_TOP_K entity ids by cosine similarity."""
        if self._entity_vecs is None or len(self._entity_vecs) == 0:
            return []
        sims  = self._entity_vecs @ q_vec
        top_i = np.argsort(sims)[::-1][:ENTITY_TOP_K]
        return [self._entity_ids_list[i] for i in top_i]

    def _bfs_expand(self, seed_entity_ids: list[str]) -> set[str]:
        """BFS depth-1: collect seed + their 1-hop neighbours via relationships."""
        expanded_ids = set(seed_entity_ids)
        seed_names = {
            self._entity_id_to_name.get(entity_id, entity_id)
            for entity_id in seed_entity_ids
        }

        rels = self._rels_df
        src_col = "source" if "source" in rels.columns else "source_id"
        tgt_col = "target" if "target" in rels.columns else "target_id"

        if src_col in rels.columns and tgt_col in rels.columns:
            for _, rel in rels.iterrows():
                source_name = str(rel[src_col])
                target_name = str(rel[tgt_col])
                if source_name in seed_names:
                    neighbour_id = self._entity_name_to_id.get(target_name)
                    if neighbour_id:
                        expanded_ids.add(neighbour_id)
                elif target_name in seed_names:
                    neighbour_id = self._entity_name_to_id.get(source_name)
                    if neighbour_id:
                        expanded_ids.add(neighbour_id)

        return expanded_ids

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
        """Rank text_units by cosine similarity, reusing LanceDB vectors."""
        rows = self._tu_df[self._tu_df["id"].isin(set(tu_ids))]
        if rows.empty:
            return []

        ids = rows["id"].astype(str).tolist()
        self._build_text_index()
        sims = None
        if self._tu_vecs is not None and len(self._tu_id_to_row) > 0:
            row_indices = [self._tu_id_to_row.get(tu_id) for tu_id in ids]
            if all(index is not None for index in row_indices):
                sims = self._tu_vecs[row_indices] @ q_vec

        if sims is None:
            texts = rows["text"].fillna("").astype(str).tolist()
            t_vecs = encode_texts(
                self._embedder,
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=8,
            )
            sims = np.asarray(t_vecs, dtype=np.float32) @ q_vec

        pairs = list(zip(ids, np.asarray(sims, dtype=np.float32).tolist()))
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
