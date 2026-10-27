import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analyze_errors import (  # noqa: E402
    analyze,
    classify_record,
    error_counts,
    main,
    markdown_report,
    parse_serialized_ids,
    per_question,
    representative_cases,
    retrieval_diagnostics,
)


def make_record(
    qid,
    system,
    accuracy,
    answer,
    retrieved_docs="['doc_a.md']",
    gold_docs="['doc_a.md']",
    category="multi_hop_relational",
    answer_is_refusal=None,
    reference="Mã môn INT1303.",
):
    record = {
        "qid": qid,
        "system": system,
        "category": category,
        "question": f"Question {qid}?",
        "reference_answer": reference,
        "answer": answer,
        "answer_accuracy": accuracy,
        "gold_doc_ids": gold_docs,
        "gold_tu_ids": "[]",
        "retrieved_doc_ids": retrieved_docs,
        "retrieved_tu_ids": "[]",
        "judge_rationale": "mock rationale",
    }
    if answer_is_refusal is not None:
        record["answer_is_refusal"] = answer_is_refusal
    return record


def test_parse_serialized_ids_handles_python_repr_lists():
    assert parse_serialized_ids("['a.md', 'b.md']") == ["a.md", "b.md"]
    assert parse_serialized_ids([]) == []
    assert parse_serialized_ids("a.md|b.md") == ["a.md", "b.md"]


def test_classify_record_priority():
    assert classify_record(make_record("Q001", "vector_rag", 1.0, "INT1303"))["error_type"] == "correct"

    refusal = classify_record(make_record(
        "Q002", "vector_rag", 0.0,
        "Tài liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này.",
        answer_is_refusal=True,
    ))
    assert refusal["error_type"] == "lazy_refusal"

    miss = classify_record(make_record(
        "Q003", "vector_rag", 0.0, "INT1303", retrieved_docs="['other.md']"
    ))
    assert miss["error_type"] == "retrieval_miss"
    assert miss["retrieval_hit"] is False

    partial_retrieval = classify_record(make_record(
        "Q008", "vector_rag", 0.0,
        "Tài liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này.",
        retrieved_docs="['doc_a.md']",
        gold_docs="['doc_a.md', 'doc_b.md']",
        answer_is_refusal=True,
    ))
    assert partial_retrieval["error_type"] == "retrieval_miss"
    assert partial_retrieval["retrieval_coverage"] == 0.5

    partial = classify_record(make_record("Q004", "vector_rag", 0.5, "INT1303"))
    assert partial["error_type"] == "partial_answer"

    wrong = classify_record(make_record("Q005", "vector_rag", 0.0, "INT9999"))
    assert wrong["error_type"] == "wrong_answer"
    assert "INT1303" in wrong["missing_tokens"]

    vanilla = classify_record(make_record("Q006", "vanilla_llm", 0.0, "INT9999", retrieved_docs="[]"))
    assert vanilla["error_type"] == "wrong_answer"

    unjudged = classify_record(make_record("Q007", "graph_rag", None, "INT1303"))
    assert unjudged["error_type"] == "unjudged"


def test_counts_questions_and_cases():
    records = [
        make_record("Q001", "vector_rag", 1.0, "INT1303"),
        make_record("Q002", "vector_rag", 0.0, "Tài liệu không có thông tin.", answer_is_refusal=True),
        make_record("Q003", "vector_rag", 0.0, "INT9999", retrieved_docs="['other.md']"),
        make_record("Q004", "vector_rag", 0.5, "INT1303"),
        make_record("Q005", "vector_rag", 0.0, "INT9999"),
        make_record("Q006", "vanilla_llm", 0.0, "INT9999", retrieved_docs="[]"),
        make_record(
            "Q008", "vector_rag", 0.0,
            "Tài liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này.",
            retrieved_docs="['doc_a.md']",
            gold_docs="['doc_a.md', 'doc_b.md']",
            answer_is_refusal=True,
        ),
    ]
    frame = analyze(records, "multi_hop_relational")
    counts = error_counts(frame)
    assert counts.loc["vector_rag", "correct"] == 1
    assert counts.loc["vector_rag", "lazy_refusal"] == 1
    assert counts.loc["vector_rag", "retrieval_miss"] == 2
    assert counts.loc["vector_rag", "partial_answer"] == 1
    assert counts.loc["vector_rag", "wrong_answer"] == 1
    assert counts.loc["vanilla_llm", "wrong_answer"] == 1

    questions = per_question(frame)
    assert bool(questions.loc["Q002", "all_systems_fail"]) is True
    assert bool(questions.loc["Q001", "all_systems_correct"]) is True

    diagnostics = retrieval_diagnostics(frame)
    assert diagnostics.loc[diagnostics["System"] == "Vector RAG", "Retrieval miss"].iloc[0] == 2
    assert diagnostics.loc[diagnostics["System"] == "Vector RAG", "Mean gold coverage"].iloc[0] == 0.75

    cases = representative_cases(frame, examples=1)
    assert "vector_rag:retrieval_miss" in cases
    assert cases["vector_rag:retrieval_miss"][0]["qid"] == "Q003"


def test_markdown_and_cli_outputs(tmp_path):
    records = [
        make_record("Q001", "vector_rag", 1.0, "INT1303"),
        make_record("Q002", "vector_rag", 0.0, "Tài liệu không có thông tin.", answer_is_refusal=True),
        make_record("Q003", "vanilla_llm", 0.0, "INT9999", retrieved_docs="[]"),
    ]
    frame = analyze(records, "multi_hop_relational")
    report = markdown_report(
        frame,
        error_counts(frame),
        frame.groupby(["system", "error_type"])["answer_accuracy"].mean().unstack(),
        per_question(frame),
        retrieval_diagnostics(frame),
        representative_cases(frame, examples=1),
        Path("input.jsonl"),
        "multi_hop_relational",
    )
    assert "# Error Analysis" in report
    assert "Lazy refusal" in report
    assert "Q002" in report

    input_path = tmp_path / "judged.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    sys.argv = [
        "analyze_errors.py",
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--examples", "1",
    ]
    assert main() == 0
    assert (output_dir / "error_analysis.md").exists()
    assert (output_dir / "error_cases.csv").exists()
    assert (output_dir / "error_summary.csv").exists()
