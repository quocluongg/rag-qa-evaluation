import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from expand_multihop_testset import FIELDS, V2_OVERRIDES, build_expanded_rows

TEST_SET = ROOT / "data" / "scientific_testset_v2.csv"
BASE_SET = ROOT / "data" / "scientific_testset_50.csv"
RAW_SET = ROOT / "data" / "scientific_testset_v2.raw.jsonl"


def load_rows(path=TEST_SET):
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def split_ids(value):
    return [item for item in value.split("|") if item and item != "NONE"]


def test_v2_has_thirty_multihop_questions():
    counts = pd.Series([r["category"] for r in load_rows()]).value_counts().to_dict()
    assert counts == {
        "factual_single_hop": 13,
        "multi_hop_relational": 30,
        "global_synthesis": 12,
        "unanswerable_negative": 12,
    }


def test_v2_preserves_the_original_fifty_rows_except_documented_overrides():
    base = load_rows(BASE_SET)
    v2 = {r["qid"]: r for r in load_rows()}
    for row in base:
        if row["qid"] in V2_OVERRIDES:
            continue
        assert v2[row["qid"]] == row


def test_v2_overrides_disambiguate_known_questions():
    rows = {r["qid"]: r for r in load_rows()}
    assert "Công nghệ vi mạch bán dẫn" in rows["Q002"]["question"]
    assert "An toàn thông tin" in rows["Q003"]["question"]
    assert "Công nghệ Kỹ thuật Điện - Điện tử" in rows["Q005"]["question"]
    assert "Công nghệ thông tin" in rows["Q007"]["question"]
    assert "thạc sĩ năm 2024" in rows["Q009"]["question"]
    assert "Công nghệ vi mạch bán dẫn" in rows["Q011"]["question"]
    assert "Truyền thông đa phương tiện" in rows["Q013"]["question"]


def test_new_rows_use_two_distinct_docs_and_two_gold_tus():
    for row in load_rows():
        if row["qid"] < "Q051":
            continue
        docs = split_ids(row["gold_doc_ids"])
        tus = split_ids(row["gold_tu_ids"])
        assert row["category"] == "multi_hop_relational", row["qid"]
        assert len(docs) == 2 and len(set(docs)) == 2, row["qid"]
        assert len(tus) == 2 and len(set(tus)) == 2, row["qid"]


def test_new_row_evidence_contains_both_credit_values():
    for row in load_rows():
        if row["qid"] < "Q051":
            continue
        values = []
        for line in row["evidence_snippet"].splitlines():
            match = re.search(r"\|\s*([A-Z]{2,}\d{4,})\s*\|\s*(\d+)\s*\|", line)
            if match:
                values.append(int(match.group(2)))
        assert len(values) == 2, (row["qid"], values)
        total = sum(values)
        if "tổng" in row["reference_answer"].lower():
            assert f"{total} tín chỉ" in row["reference_answer"], (row["qid"], total)
        for value in values:
            assert str(value) in row["reference_answer"], (row["qid"], value)
            assert f"| {value} |" in row["evidence_snippet"], (row["qid"], value)


def test_new_questions_do_not_leak_course_codes():
    for row in load_rows():
        if row["qid"] < "Q051":
            continue
        assert not re.search(r"[A-Z]{2,}\d{4,}", row["question"]), row["qid"]
        assert not re.search(r"[A-Z]{2,}\d{4,}", row["reference_answer"]), row["qid"]


def test_v2_gold_tus_exist_and_match_gold_docs():
    tu_df = pd.read_parquet(ROOT / "output" / "text_units.parquet").set_index("id")
    doc_df = pd.read_parquet(ROOT / "output" / "documents.parquet").set_index("id")
    for row in load_rows():
        tus = split_ids(row["gold_tu_ids"])
        docs = set(split_ids(row["gold_doc_ids"]))
        for tu_id in tus:
            assert tu_id in tu_df.index, (row["qid"], tu_id)
        actual = {doc_df.loc[tu_df.loc[tu_id, "document_id"], "title"] for tu_id in tus}
        assert actual == docs, (row["qid"], actual, docs)


def test_v2_raw_jsonl_matches_csv():
    csv_rows = {r["qid"]: r for r in load_rows()}
    raw_rows = {
        r["qid"]: r
        for r in (
            json.loads(line)
            for line in RAW_SET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    assert set(raw_rows) == set(csv_rows)
    for qid, row in csv_rows.items():
        for field in FIELDS:
            assert raw_rows[qid][field] == row[field], (qid, field)


def test_expansion_is_deterministic():
    generated = build_expanded_rows(BASE_SET)
    assert len(generated) == 67
    assert [r["qid"] for r in generated] == [r["qid"] for r in load_rows()]
    for actual, stored in zip(generated, load_rows()):
        assert actual["question"] == stored["question"], actual["qid"]
        assert actual["gold_tu_ids"] == stored["gold_tu_ids"], actual["qid"]
