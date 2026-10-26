import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.graph_rag import GraphRAG
from retrieval.index_utils import ensure_text_unit_index, table_is_readable
from retrieval.vector_rag import VectorRAG
from testset_utils import gold_docs_for_row, repair_gold_doc_ids


class FakeEmbedder:
    dim = 16

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        rows = []
        for text in items:
            seed = sum((i + 1) * ord(ch) for i, ch in enumerate(str(text)[:128]))
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            rows.append(vec)
        arr = np.vstack(rows)
        return arr[0] if single else arr


class FakeSourceMap:
    def tu_ids_to_filenames(self, tu_ids):
        mapping = {"tu-1": "a.md", "tu-2": "b.md"}
        return [mapping[tu_id] for tu_id in tu_ids if tu_id in mapping]


def test_repair_gold_doc_ids_from_text_units():
    rows = [
        {"qid": "Q1", "gold_doc_ids": "", "gold_tu_ids": "tu-1|tu-2"},
        {"qid": "Q2", "gold_doc_ids": "NONE", "gold_tu_ids": "NONE"},
        {"qid": "Q3", "gold_doc_ids": "c.md", "gold_tu_ids": "tu-1"},
    ]
    repaired = repair_gold_doc_ids(rows, FakeSourceMap())
    assert repaired == 1
    assert rows[0]["gold_doc_ids"] == "a.md|b.md"
    assert rows[1]["gold_doc_ids"] == "NONE"
    assert rows[2]["gold_doc_ids"] == "c.md"
    assert gold_docs_for_row({"gold_doc_ids": "", "gold_tu_ids": "tu-2"}, FakeSourceMap()) == ["b.md"]


def test_lancedb_index_is_built_and_readable(tmp_path):
    embedder = FakeEmbedder()
    db_dir = tmp_path / "lancedb"

    table = ensure_text_unit_index(embedder, db_dir=db_dir)
    assert table.count_rows() == 376

    import lancedb

    db = lancedb.connect(str(db_dir))
    assert table_is_readable(db, "text_unit_text", expected_dim=16)

    retriever = VectorRAG(k=3, embedder=embedder, index_dir=db_dir)
    result = retriever.retrieve("Ngành Công nghệ vi mạch bán dẫn")
    assert len(result.evidence) == 3
    assert all(ev.filename.endswith(".md") for ev in result.evidence)
    assert all(ev.content for ev in result.evidence)


def test_graph_bfs_maps_relationship_names_to_entity_ids():
    retriever = GraphRAG(embedder=FakeEmbedder())
    retriever._entities_df = pd.DataFrame(
        [
            {"id": "e1", "title": "HỌC VIỆN", "description": ""},
            {"id": "e2", "title": "BỘ QUẢN LÝ", "description": ""},
        ]
    )
    retriever._rels_df = pd.DataFrame(
        [{"source": "HỌC VIỆN", "target": "BỘ QUẢN LÝ"}]
    )
    retriever._entity_id_to_name = {"e1": "HỌC VIỆN", "e2": "BỘ QUẢN LÝ"}
    retriever._entity_name_to_id = {"HỌC VIỆN": "e1", "BỘ QUẢN LÝ": "e2"}

    assert retriever._bfs_expand(["e1"]) == {"e1", "e2"}
