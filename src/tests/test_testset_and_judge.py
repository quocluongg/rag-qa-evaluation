import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_3systems import build_judge_prompt

TEST_SET = ROOT / "data" / "scientific_testset_50.csv"


def load_testset():
    return pd.read_csv(TEST_SET, dtype=str).fillna("").set_index("qid")


def split_ids(value):
    return [item for item in value.split("|") if item and item != "NONE"]


def test_q015_disambiguates_course_and_sums_credits():
    row = load_testset().loc["Q015"]
    assert "INT1341" in row["question"]
    assert "ELE14157" not in row["question"]
    assert "6" in row["reference_answer"]
    assert "af997aff09bf" in row["gold_tu_ids"]
    assert "90223fbed11e" in row["gold_tu_ids"]


def test_q017_award_and_scholarship_fund_are_both_answered():
    row = load_testset().loc["Q017"]
    assert "quỹ học bổng" in row["question"]
    assert "giấy khen" in row["question"]
    assert "0,15" in row["reference_answer"]
    assert "542/QĐ-HV" in row["reference_answer"]
    assert row["category"] == "multi_hop_relational"
    assert len(split_ids(row["gold_tu_ids"])) == 2


def test_q025_admission_phone_and_cutoff_are_both_answered():
    row = load_testset().loc["Q025"]
    assert "điện thoại" in row["question"]
    assert "thạc sĩ" in row["question"]
    assert "028.38297220" in row["reference_answer"]
    assert "26.59" in row["reference_answer"]
    assert row["category"] == "multi_hop_relational"
    assert len(split_ids(row["gold_tu_ids"])) == 2


def test_replacement_rows_are_multihop_with_two_gold_tus():
    df = load_testset()
    for qid in ["Q016", "Q017", "Q018", "Q024", "Q025"]:
        row = df.loc[qid]
        assert row["category"] == "multi_hop_relational", qid
        assert len(split_ids(row["gold_tu_ids"])) == 2, qid


def test_q016_aerospace_electives_and_cleanroom_credits_are_both_answered():
    row = load_testset().loc["Q016"]
    assert "Kỹ thuật Truyền thông hàng không vũ trụ" in row["question"]
    assert "Công nghệ phòng sạch" in row["reference_answer"]
    assert "3 tín chỉ" in row["reference_answer"]
    assert "2 tín chỉ" in row["reference_answer"]
    assert "5 tín chỉ" in row["reference_answer"]
    assert row["category"] == "multi_hop_relational"
    assert len(split_ids(row["gold_tu_ids"])) == 2


def test_q018_notice_signer_and_cutoff_are_both_answered():
    row = load_testset().loc["Q018"]
    assert "209/TB-HV" in row["question"]
    assert "Quản trị kinh doanh" in row["question"]
    assert "Trần Quang Anh" in row["reference_answer"]
    assert "25.15" in row["reference_answer"]
    assert row["category"] == "multi_hop_relational"
    assert len(split_ids(row["gold_tu_ids"])) == 2


def test_q024_iot_semester_and_replacement_are_both_answered():
    row = load_testset().loc["Q024"]
    assert "TEL14102" in row["question"]
    assert "học kỳ 8" in row["reference_answer"]
    assert "TEL1464" in row["reference_answer"]
    assert "từ 2 tín chỉ" in row["reference_answer"]
    assert "3 tín chỉ" in row["reference_answer"]
    assert row["category"] == "multi_hop_relational"
    assert len(split_ids(row["gold_tu_ids"])) == 2


def test_category_counts_after_testset_cleanup():
    counts = load_testset()["category"].value_counts().to_dict()
    assert counts == {
        "factual_single_hop": 13,
        "multi_hop_relational": 13,
        "global_synthesis": 12,
        "unanswerable_negative": 12,
    }


def test_multihop_rows_have_two_gold_evidence_units():
    df = load_testset()
    for qid, row in df[df["category"] == "multi_hop_relational"].iterrows():
        assert len(split_ids(row["gold_tu_ids"])) == 2, qid


def test_gold_tus_exist_and_match_gold_docs():
    tu_df = pd.read_parquet(ROOT / "output" / "text_units.parquet").set_index("id")
    doc_df = pd.read_parquet(ROOT / "output" / "documents.parquet").set_index("id")

    for qid, row in load_testset().iterrows():
        tu_ids = split_ids(row["gold_tu_ids"])
        doc_ids = set(split_ids(row["gold_doc_ids"]))
        for tu_id in tu_ids:
            assert tu_id in tu_df.index, (qid, tu_id)
        actual = {doc_df.loc[tu_df.loc[tu_id, "document_id"], "title"] for tu_id in tu_ids}
        assert actual == doc_ids, (qid, actual, doc_ids)


def test_multihop_judge_prompt_requires_partial_credit_for_one_subanswer():
    prompt = build_judge_prompt(
        question="Câu hỏi hai vế?",
        category="multi_hop_relational",
        reference_answer="Vế một là A; vế hai là B.",
        context="Bằng chứng A và B.",
        answer="A.",
    )
    assert "chỉ đúng một vế" in prompt
    assert "0.5" in prompt
    assert "không suy đoán" in prompt.lower()


def test_single_hop_judge_prompt_does_not_add_multihop_rule():
    prompt = build_judge_prompt(
        question="Câu hỏi một vế?",
        category="factual_single_hop",
        reference_answer="A.",
        context="Bằng chứng A.",
        answer="A.",
    )
    assert "chỉ đúng một vế" not in prompt


def test_raw_testset_matches_corrected_csv():
    csv_df = load_testset()
    raw_path = ROOT / "data" / "scientific_testset_50.raw.jsonl"
    raw_rows = {
        row["qid"]: row
        for row in (
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    assert set(raw_rows) == set(csv_df.index)
    for qid in csv_df.index:
        for field in ["category", "question", "reference_answer", "gold_doc_ids", "gold_tu_ids"]:
            assert raw_rows[qid][field] == csv_df.loc[qid, field], (qid, field)
