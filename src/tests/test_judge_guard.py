import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_3systems import (
    aggregate_judge_scores,
    build_judge_prompt,
    factual_consistency_guard,
)


def test_factual_guard_rejects_wrong_course_code():
    cap, missing = factual_consistency_guard(
        "factual_single_hop",
        "Ngành An toàn thông tin có mã xét tuyển ELE16246.",
        "Ngành An toàn thông tin có mã xét tuyển ELE1426.",
    )
    assert cap == 0.0
    assert "ELE16246" in missing


def test_factual_guard_accepts_equivalent_decimal_format():
    cap, missing = factual_consistency_guard(
        "factual_single_hop",
        "Điểm chuẩn là 26,59.",
        "Điểm chuẩn là 26.59.",
    )
    assert cap is None
    assert missing == []


def test_factual_guard_gives_partial_credit_for_one_of_two_parts():
    cap, missing = factual_consistency_guard(
        "factual_single_hop",
        "Môn BAS1203 có 3 tín chỉ.",
        "Môn BAS1203.",
    )
    assert cap == 0.5
    assert "3 tín chỉ" in missing


def test_multihop_guard_only_catches_total_mismatch():
    cap, missing = factual_consistency_guard(
        "multi_hop_relational",
        "Môn BAS1203 có 3 tín chỉ; môn ELE13105 có 3 tín chỉ.",
        "Môn BAS1203 có 3 tín chỉ.",
    )
    assert cap is None
    assert missing == []

    cap, missing = factual_consistency_guard(
        "multi_hop_relational",
        "Môn BAS1203 có 3 tín chỉ; môn ELE13105 có 3 tín chỉ.",
        "Tài liệu không đủ thông tin.",
    )
    assert cap == 0.0
    assert "BAS1203" in missing


def test_factual_guard_skips_unanswerable_and_global_rows():
    assert factual_consistency_guard("unanswerable_negative", "Tài liệu không cung cấp thông tin.", "Không có thông tin.") == (None, [])
    assert factual_consistency_guard("global_synthesis", "Có 3 nhóm ngành, 120 tín chỉ.", "Có 3 nhóm ngành.") == (None, [])


def test_aggregate_judge_scores_reports_mean_std_and_agreement():
    result = aggregate_judge_scores([
        {"correctness": 1.0, "faithfulness": 1.0, "refusal_appropriate": True, "rationale": "a"},
        {"correctness": 0.5, "faithfulness": 1.0, "refusal_appropriate": True, "rationale": "b"},
        {"correctness": 1.0, "faithfulness": 0.5, "refusal_appropriate": False, "rationale": "c"},
    ])
    assert result["answer_accuracy"] == pytest.approx(0.833333, rel=1e-5)
    assert result["answer_accuracy_std"] > 0
    assert result["faithfulness"] == pytest.approx(0.833333, rel=1e-5)
    assert result["refusal_appropriate"] is True
    assert result["judge_agreement"] == pytest.approx(2 / 3)


def test_judge_prompt_separates_correctness_from_context():
    prompt = build_judge_prompt(
        question="Câu hỏi?",
        category="factual_single_hop",
        reference_answer="A",
        context="Bằng chứng A",
        answer="A",
    )
    assert "CHỈ DÙNG CHO FAITHFULNESS" in prompt
    assert "KHÔNG dùng ngữ cảnh" in prompt
    assert "không suy đoán" in prompt.lower()
