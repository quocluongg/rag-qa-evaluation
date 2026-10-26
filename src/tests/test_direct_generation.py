from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_testset_direct import (
    CATEGORIES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TOTAL_REQUESTS,
    DEFAULT_SIZE,
    build_context,
    prompt_for,
)


def test_default_protocol_budget_and_distribution():
    assert DEFAULT_SIZE == 90
    assert DEFAULT_BATCH_SIZE == 10
    assert DEFAULT_MAX_ATTEMPTS == 5
    assert DEFAULT_MAX_TOTAL_REQUESTS == 450
    assert DEFAULT_SIZE * DEFAULT_MAX_ATTEMPTS == DEFAULT_MAX_TOTAL_REQUESTS
    assert len(CATEGORIES) == 5
    assert DEFAULT_SIZE // len(CATEGORIES) == 18


def test_context_and_prompt_are_source_grounded():
    units = [
        {
            "text_unit_id": "tu-1",
            "document_id": "doc-1",
            "doc_id": "doc-1",
            "filename": "a.md",
            "valid_from": "2025-01-01",
            "valid_to": "",
            "text": "Chuong trinh AIoT co noi dung ve IoT.",
        },
        {
            "text_unit_id": "tu-2",
            "document_id": "doc-2",
            "doc_id": "doc-2",
            "filename": "b.md",
            "valid_from": "",
            "valid_to": "",
            "text": "INT1340 co hoc phan tien quyet INT1332.",
        },
    ]
    import random

    contexts = build_context("multi-hop", 0, units, random.Random(42))
    assert len(contexts) == 2
    prompt = prompt_for("multi-hop", contexts)
    assert "Use ONLY the supplied source context" in prompt
    assert '"user_input"' in prompt
    assert "text_unit_id=tu-" in prompt
