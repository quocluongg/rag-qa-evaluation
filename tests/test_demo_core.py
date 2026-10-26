from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_core import (
    build_context,
    build_prompt,
    cache_key,
    entity_embedding_text,
    is_valid_at,
    normalize_tq,
    record_to_markdown,
)


def test_normalize_tq_accepts_dd_mm_yyyy():
    assert normalize_tq("15/10/2024") == "15/10/2024"


def test_normalize_tq_accepts_iso():
    assert normalize_tq("2024-10-15") == "15/10/2024"


def test_normalize_tq_returns_none_for_empty():
    assert normalize_tq("") is None
    assert normalize_tq(None) is None


def test_normalize_tq_rejects_invalid():
    with pytest.raises(ValueError):
        normalize_tq("15-10-2024")


def test_is_valid_at_respects_both_bounds():
    assert is_valid_at("01/01/2024", "31/12/2024", "15/10/2024")
    assert not is_valid_at("01/01/2025", "31/12/2025", "15/10/2024")
    assert not is_valid_at("01/01/2023", "31/12/2023", "15/10/2024")


def test_is_valid_at_handles_open_ended_validity():
    assert is_valid_at("01/09/2026", "", "25/10/2026")
    assert not is_valid_at("", "", "25/10/2026")


def test_build_context_numbers_evidence_and_shows_validity():
    evidence = [
        {
            "filename": "QD_542HV_QCHBong_2026.md",
            "valid_from": "01/09/2026",
            "valid_to": "",
            "content": "Quy định học bổng mới.",
        },
        {
            "filename": "Dh_2026.md",
            "valid_from": "01/01/2026",
            "valid_to": "",
            "content": "Đề án tuyển sinh đại học 2026.",
        },
    ]
    context = build_context(evidence)
    assert "[1] QD_542HV_QCHBong_2026.md" in context
    assert "[2] Dh_2026.md" in context
    assert "01/09/2026 - nay" in context
    assert "Quy định học bổng mới." in context


def test_build_prompt_contains_question_time_and_citation_rule():
    evidence = [
        {
            "filename": "QD_542HV_QCHBong_2026.md",
            "valid_from": "01/09/2026",
            "valid_to": "",
            "content": "Quy định học bổng mới.",
        }
    ]
    prompt = build_prompt("Học bổng năm 2026 thay đổi gì?", "01/10/2026", evidence)
    assert "Học bổng năm 2026 thay đổi gì?" in prompt
    assert "01/10/2026" in prompt
    assert "[1]" in prompt
    assert "Không đủ thông tin" in prompt
    assert "Không bịa" in prompt


def test_cache_key_is_stable_and_config_sensitive():
    base = {
        "system": "temporal_graph_rag",
        "model": "gemini-3.5-flash-lite",
        "query": "Học bổng 2026?",
        "tq": "01/10/2026",
        "k": 5,
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "embedding_backend": "bge-m3",
        "prompt_version": "v1",
    }
    assert cache_key(base) == cache_key(dict(base))
    changed = dict(base, model="gemma-4-31b-it")
    assert cache_key(base) != cache_key(changed)


def test_entity_embedding_text_uses_title_and_description():
    text = entity_embedding_text("HỌC VIỆN PTIT", "Đơn vị sự nghiệp công lập.")
    assert "HỌC VIỆN PTIT" in text
    assert "Đơn vị sự nghiệp công lập." in text


def test_record_to_markdown_contains_config_metrics_and_evidence():
    record = {
        "timestamp": "2026-10-25T10:00:00Z",
        "query": "Học bổng năm 2026 thay đổi gì?",
        "answer": "Quy định mới thay đổi mức học bổng [1].",
        "cache_hit": False,
        "error": None,
        "config": {
            "system": "temporal_graph_rag",
            "model": "gemini-3.5-flash-lite",
            "k": 5,
            "tq": "01/10/2026",
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "embedding_backend": "bge-m3",
            "safe_mode": True,
        },
        "retrieval": {
            "latency_s": 0.12,
            "ctx_words": 240,
            "document_ids": ["QD_542HV_QCHBong_2026.md"],
            "evidence": [
                {
                    "rank": 1,
                    "filename": "QD_542HV_QCHBong_2026.md",
                    "valid_from": "01/09/2026",
                    "valid_to": "",
                    "score": 0.81,
                    "content": "Quy định học bổng mới.",
                }
            ],
            "trace": {"seed_entity_titles": ["HỌC VIỆN PTIT"]},
        },
        "generation": {
            "latency_s": 1.4,
            "prompt_tokens": 300,
            "completion_tokens": 40,
            "total_tokens": 340,
            "thoughts_token_count": 0,
            "finish_reason": "STOP",
            "model_version": "gemini-3.5-flash-lite",
            "attempts": 1,
            "cache_hit": False,
            "error": None,
        },
        "prompt": "Câu hỏi: Học bổng năm 2026 thay đổi gì?",
    }
    md = record_to_markdown(record)
    assert "# Báo cáo chạy GraphRAG" in md
    assert "gemini-3.5-flash-lite" in md
    assert "0.12" in md
    assert "QD_542HV_QCHBong_2026.md" in md
    assert "Quy định học bổng mới." in md
    assert "340" in md
