"""
source_map.py — Map graph evidence (entity_ids / text_unit_ids) to document_ids.

Used by all 4 retrieval systems to normalise evidence to document-level
so that P@k / R@k / MRR can be computed consistently (document-level, not chunk-level).

Key join:
  text_units.parquet: id (tu_id), document_id (doc hash), entity_ids[], relationship_ids[]
  documents.parquet : id (doc hash), title (filename like CTDT_AIoT_2025.md)
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

ROOT         = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = ROOT / "output"
TU_PARQUET   = OUTPUT_DIR / "text_units.parquet"
DOCS_PARQUET = OUTPUT_DIR / "documents.parquet"
META_JSON    = OUTPUT_DIR / "metadata.json"


class SourceMap:
    """
    Builds lookup tables from the GraphRAG index so every retrieval system
    can map its raw evidence back to source document filenames.

    Attributes
    ----------
    tu_df        : full text_units DataFrame (id, document_id, entity_ids, ...)
    docs_df      : full documents DataFrame  (id, title, ...)
    meta         : dict filename → {doc_id, valid_from, valid_to, ...}
    hash_to_file : doc_hash → filename (e.g. "CTDT_AIoT_2025.md")
    file_to_hash : filename → doc_hash
    tu_to_file   : text_unit_id → filename
    """

    def __init__(self):
        self.tu_df   = pd.read_parquet(TU_PARQUET)
        self.docs_df = pd.read_parquet(DOCS_PARQUET)
        self.meta    = json.load(open(META_JSON, encoding="utf-8"))

        # hash ↔ filename
        self.hash_to_file: dict[str, str] = dict(
            zip(self.docs_df["id"], self.docs_df["title"])
        )
        self.file_to_hash: dict[str, str] = {v: k for k, v in self.hash_to_file.items()}

        # text_unit_id → filename  (direct lookup, O(1) after build)
        self._tu_to_file: dict[str, str] = {}
        for _, row in self.tu_df.iterrows():
            fname = self.hash_to_file.get(row["document_id"], "")
            if fname:
                self._tu_to_file[row["id"]] = fname

        # entity_id → set of filenames
        self._entity_to_files: dict[str, set[str]] = {}
        for _, row in self.tu_df.iterrows():
            fname = self.hash_to_file.get(row["document_id"], "")
            if not fname:
                continue
            eids = row["entity_ids"]
            if eids is None:
                continue
            for eid in (eids.tolist() if hasattr(eids, "tolist") else list(eids)):
                self._entity_to_files.setdefault(eid, set()).add(fname)

    # ── text_unit → document ──────────────────────────────────

    def tu_to_filename(self, tu_id: str) -> str | None:
        """Return filename for a text_unit_id, or None if not found."""
        return self._tu_to_file.get(tu_id)

    def tu_ids_to_filenames(self, tu_ids: Sequence[str]) -> list[str]:
        """Unique filenames for a list of text_unit_ids (order-preserving, deduped)."""
        seen: set[str] = set()
        out: list[str] = []
        for tid in tu_ids:
            fn = self._tu_to_file.get(tid)
            if fn and fn not in seen:
                seen.add(fn)
                out.append(fn)
        return out

    # ── entity → document ────────────────────────────────────

    def entity_ids_to_filenames(self, entity_ids: Sequence[str]) -> list[str]:
        """Unique filenames for a list of entity_ids."""
        seen: set[str] = set()
        out: list[str] = []
        for eid in entity_ids:
            for fn in self._entity_to_files.get(eid, set()):
                if fn not in seen:
                    seen.add(fn)
                    out.append(fn)
        return out

    # ── metadata helpers ──────────────────────────────────────

    def get_validity(self, filename: str) -> tuple[str, str]:
        """Return (valid_from, valid_to) strings for a document filename."""
        entry = self.meta.get(filename, {})
        return entry.get("valid_from", ""), entry.get("valid_to", "")

    def is_temporally_valid(self, filename: str, tq: str) -> bool:
        """
        Check V(d, tq) = 1.
        Dates are in dd/mm/yyyy format (from metadata.json).
        valid_to="" means open-ended (valid until replaced).
        """
        from datetime import datetime
        vf_str, vt_str = self.get_validity(filename)
        try:
            tq_dt = datetime.strptime(tq, "%d/%m/%Y")
            vf_dt = datetime.strptime(vf_str, "%d/%m/%Y") if vf_str else datetime.min
            vt_dt = datetime.strptime(vt_str, "%d/%m/%Y") if vt_str else datetime.max
            return vf_dt <= tq_dt <= vt_dt
        except ValueError:
            return False  # invalid metadata must not pass temporal filtering

    def valid_filenames_at(self, tq: str) -> set[str]:
        """Return set of filenames that are temporally valid at tq."""
        return {fn for fn in self.meta if self.is_temporally_valid(fn, tq)}


# ── Singleton accessor ────────────────────────────────────────

_source_map: SourceMap | None = None

def get_source_map() -> SourceMap:
    global _source_map
    if _source_map is None:
        _source_map = SourceMap()
    return _source_map
