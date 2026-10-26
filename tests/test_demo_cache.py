from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_core import AnswerCache, build_or_load_embeddings, normalize_rows


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def encode_documents(self, texts):
        self.calls += 1
        return normalize_rows([[float(i + 1), 1.0] for i, _ in enumerate(texts)])


def test_answer_cache_round_trip(tmp_path):
    cache = AnswerCache(tmp_path)
    config = {"system": "graph_rag", "query": "abc", "model": "m", "k": 5}
    key = cache.key(config)
    assert cache.load(key) is None
    cache.save(key, {"answer": "ok", "config": config})
    assert cache.load(key)["answer"] == "ok"


def test_embedding_cache_reuses_vectors(tmp_path):
    backend = FakeBackend()
    cache_path = tmp_path / "vectors.npz"
    ids = ["a", "b"]
    texts = ["một", "hai"]
    first = build_or_load_embeddings(backend, ids, texts, cache_path)
    second = build_or_load_embeddings(backend, ids, texts, cache_path)
    assert backend.calls == 1
    assert first.shape == (2, 2)
    assert (first == second).all()


def test_embedding_cache_invalidates_when_texts_change(tmp_path):
    backend = FakeBackend()
    cache_path = tmp_path / "vectors.npz"
    build_or_load_embeddings(backend, ["a"], ["một"], cache_path)
    build_or_load_embeddings(backend, ["a"], ["một hai"], cache_path)
    assert backend.calls == 2


def test_normalize_rows_returns_unit_vectors():
    vectors = normalize_rows([[3.0, 4.0]])
    assert abs(float((vectors**2).sum() ** 0.5) - 1.0) < 1e-6
