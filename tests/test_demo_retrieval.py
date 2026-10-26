from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_engine import GraphRetriever, SourceIndex, VectorRetriever


class FakeBackend:
    name = "fake"
    dim = 2

    def __init__(self, query_vector):
        self.query_vector = np.asarray(query_vector, dtype=np.float32)

    def encode_query(self, text):
        return self.query_vector

    def encode_documents(self, texts):
        raise NotImplementedError


def make_source_index():
    meta = {
        "a.md": {"valid_from": "01/01/2024", "valid_to": "31/12/2024", "title": "Tài liệu A"},
        "b.md": {"valid_from": "01/01/2025", "valid_to": "", "title": "Tài liệu B"},
    }
    docs_df = pd.DataFrame(
        [
            {"id": "doc-a", "title": "a.md"},
            {"id": "doc-b", "title": "b.md"},
        ]
    )
    tu_df = pd.DataFrame(
        [
            {
                "id": "tu-a",
                "document_id": "doc-a",
                "text": "Học viện có quy định cũ.",
                "entity_ids": np.array(["e1"], dtype=object),
                "n_tokens": 10,
            },
            {
                "id": "tu-b",
                "document_id": "doc-b",
                "text": "Bộ quản lý học viện ban hành quy định mới.",
                "entity_ids": np.array(["e2"], dtype=object),
                "n_tokens": 12,
            },
        ]
    )
    entities_df = pd.DataFrame(
        [
            {"id": "e1", "title": "HỌC VIỆN", "description": "Học viện công lập."},
            {"id": "e2", "title": "BỘ QUẢN LÝ", "description": "Bộ chủ quản."},
        ]
    )
    rels_df = pd.DataFrame(
        [
            {"id": "r1", "source": "HỌC VIỆN", "target": "BỘ QUẢN LÝ", "description": "trực thuộc"},
        ]
    )
    return SourceIndex.from_frames(meta, tu_df, docs_df, entities_df, rels_df)


def make_versioned_source_index():
    meta = {
        "a.md": {
            "valid_from": "01/01/2024",
            "valid_to": "31/12/2024",
            "title": "Quy định cũ",
            "doc_id": "A",
            "replaced_by": "B",
        },
        "b.md": {
            "valid_from": "01/01/2025",
            "valid_to": "",
            "title": "Quy định mới",
            "doc_id": "B",
            "replaces": "A",
        },
    }
    docs_df = pd.DataFrame(
        [{"id": "doc-a", "title": "a.md"}, {"id": "doc-b", "title": "b.md"}]
    )
    tu_df = pd.DataFrame(
        [
            {
                "id": "tu-a",
                "document_id": "doc-a",
                "text": "Quy định cũ về học bổng.",
                "entity_ids": np.array(["e1"], dtype=object),
                "n_tokens": 8,
            },
            {
                "id": "tu-b",
                "document_id": "doc-b",
                "text": "Quy định mới về học bổng.",
                "entity_ids": np.array(["e2"], dtype=object),
                "n_tokens": 8,
            },
        ]
    )
    entities_df = pd.DataFrame(
        [
            {"id": "e1", "title": "HỌC BỔNG CŨ", "description": "Quy định cũ."},
            {"id": "e2", "title": "HỌC BỔNG MỚI", "description": "Quy định mới."},
        ]
    )
    rels_df = pd.DataFrame(
        [{"id": "r1", "source": "HỌC BỔNG CŨ", "target": "HỌC BỔNG MỚI"}]
    )
    return SourceIndex.from_frames(meta, tu_df, docs_df, entities_df, rels_df)


def test_source_index_resolves_version_chain():
    si = make_versioned_source_index()
    assert si.replaces_map["b.md"] == {"a.md"}
    assert si.replaced_by_map["a.md"] == {"b.md"}
    assert si.predecessors("b.md") == {"a.md"}


def test_temporal_vector_retriever_adds_previous_version():
    chunks = pd.DataFrame(
        [
            {
                "id": "c1",
                "filename": "a.md",
                "text": "Quy định cũ về học bổng.",
                "valid_from": "01/01/2024",
                "valid_to": "31/12/2024",
                "origin": "test",
            },
            {
                "id": "c2",
                "filename": "b.md",
                "text": "Quy định mới về học bổng.",
                "valid_from": "01/01/2025",
                "valid_to": "",
                "origin": "test",
            },
        ]
    )
    vectors = np.asarray([[0.95, 0.05], [1.0, 0.0]], dtype=np.float32)
    retriever = VectorRetriever(
        chunks,
        vectors,
        FakeBackend([1.0, 0.0]),
        k=1,
        temporal=True,
        source_index=make_versioned_source_index(),
    )
    result = retriever.retrieve("học bổng", "15/10/2025")
    assert result.evidence[0]["filename"] == "b.md"
    assert result.version_evidence[0]["filename"] == "a.md"
    assert result.version_evidence[0]["version_note"] == "phiên bản trước của b.md"
    assert result.trace["version_evidence_count"] == 1
    assert set(result.document_ids) == {"a.md", "b.md"}


def test_temporal_graph_retriever_adds_previous_version():
    si = make_versioned_source_index()
    retriever = GraphRetriever(
        si,
        FakeBackend([0.0, 1.0]),
        ["e1", "e2"],
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        ["tu-a", "tu-b"],
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        k=1,
        temporal=True,
    )
    result = retriever.retrieve("học bổng", "15/10/2025")
    assert result.evidence[0]["text_unit_id"] == "tu-b"
    assert result.version_evidence[0]["text_unit_id"] == "tu-a"
    assert result.trace["version_evidence_count"] == 1


def test_source_index_maps_valid_filenames():
    si = make_source_index()
    assert si.valid_filenames_at("15/10/2024") == {"a.md"}
    assert si.valid_filenames_at("15/10/2025") == {"b.md"}
    assert si.valid_filenames_at("15/10/2023") == set()


def test_source_index_expands_graph_by_entity_title():
    si = make_source_index()
    assert si.adjacency["e1"] == {"e2"}
    assert si.adjacency["e2"] == {"e1"}
    assert si.expand_entity_ids({"e1"}) == {"e1", "e2"}


def test_vector_retriever_ranks_by_cosine_similarity():
    chunks = pd.DataFrame(
        [
            {
                "id": "c1",
                "filename": "a.md",
                "text": "Quy định cũ.",
                "valid_from": "01/01/2024",
                "valid_to": "31/12/2024",
                "origin": "test",
            },
            {
                "id": "c2",
                "filename": "b.md",
                "text": "Quy định mới.",
                "valid_from": "01/01/2025",
                "valid_to": "",
                "origin": "test",
            },
        ]
    )
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    retriever = VectorRetriever(chunks, vectors, FakeBackend([1.0, 0.0]), k=1)
    result = retriever.retrieve("quy định")
    assert len(result.evidence) == 1
    assert result.evidence[0]["filename"] == "a.md"
    assert result.trace["index_size"] == 2


def test_vector_retriever_filters_by_query_time():
    chunks = pd.DataFrame(
        [
            {
                "id": "c1",
                "filename": "a.md",
                "text": "Quy định cũ.",
                "valid_from": "01/01/2024",
                "valid_to": "31/12/2024",
                "origin": "test",
            },
            {
                "id": "c2",
                "filename": "b.md",
                "text": "Quy định mới.",
                "valid_from": "01/01/2025",
                "valid_to": "",
                "origin": "test",
            },
        ]
    )
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    retriever = VectorRetriever(chunks, vectors, FakeBackend([1.0, 0.0]), k=1, temporal=True)
    result = retriever.retrieve("quy định", "15/10/2025")
    assert len(result.evidence) == 1
    assert result.evidence[0]["filename"] == "b.md"
    assert result.trace["valid_chunk_count"] == 1


def test_graph_retriever_expands_to_neighbour_evidence():
    si = make_source_index()
    entity_ids = ["e1", "e2"]
    entity_vectors = np.asarray([[1.0, 0.0], [0.2, 0.98]], dtype=np.float32)
    tu_ids = ["tu-a", "tu-b"]
    tu_vectors = np.asarray([[0.1, 0.99], [0.99, 0.1]], dtype=np.float32)
    retriever = GraphRetriever(
        si,
        FakeBackend([1.0, 0.0]),
        entity_ids,
        entity_vectors,
        tu_ids,
        tu_vectors,
        k=2,
    )
    result = retriever.retrieve("quy định mới")
    assert result.trace["seed_entity_titles"] == ["HỌC VIỆN"]
    assert result.trace["expanded_entity_count"] == 2
    assert {ev["text_unit_id"] for ev in result.evidence} == {"tu-a", "tu-b"}
    assert result.evidence[0]["text_unit_id"] == "tu-b"


def test_temporal_graph_retriever_prunes_invalid_entities():
    si = make_source_index()
    entity_ids = ["e1", "e2"]
    entity_vectors = np.asarray([[1.0, 0.0], [0.2, 0.98]], dtype=np.float32)
    tu_ids = ["tu-a", "tu-b"]
    tu_vectors = np.asarray([[0.1, 0.99], [0.99, 0.1]], dtype=np.float32)
    retriever = GraphRetriever(
        si,
        FakeBackend([1.0, 0.0]),
        entity_ids,
        entity_vectors,
        tu_ids,
        tu_vectors,
        k=2,
        temporal=True,
    )
    result = retriever.retrieve("quy định", "15/10/2025")
    assert result.trace["valid_doc_count"] == 1
    assert result.trace["valid_entity_count"] == 1
    assert {ev["filename"] for ev in result.evidence} == {"b.md"}
