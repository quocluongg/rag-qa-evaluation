"""Qualitative error analysis for the multi-hop RAG benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from testset_utils import split_ids
from text_utils import answer_contains_token, extract_codes_and_numbers, looks_like_refusal

SYSTEM_ORDER = ["vanilla_llm", "vector_rag", "graph_rag"]
SYSTEM_LABELS = {
    "vanilla_llm": "Vanilla LLM",
    "vector_rag": "Vector RAG",
    "graph_rag": "GraphRAG",
}
ERROR_ORDER = [
    "correct",
    "lazy_refusal",
    "retrieval_miss",
    "partial_answer",
    "wrong_answer",
    "unjudged",
]
ERROR_LABELS = {
    "correct": "Correct",
    "lazy_refusal": "Lazy refusal",
    "retrieval_miss": "Retrieval miss",
    "partial_answer": "Partial answer",
    "wrong_answer": "Wrong answer",
    "unjudged": "Unjudged",
}


def parse_serialized_ids(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
    return split_ids(text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def answer_is_refusal(record: dict[str, Any]) -> bool:
    if "answer_is_refusal" in record:
        return bool(record["answer_is_refusal"])
    return looks_like_refusal(str(record.get("answer", "")))


def classify_record(record: dict[str, Any]) -> dict[str, Any]:
    system = str(record.get("system", ""))
    accuracy = as_float(record.get("answer_accuracy"))
    refusal = answer_is_refusal(record)
    answer = str(record.get("answer", ""))
    reference = str(record.get("reference_answer", ""))

    gold_docs = parse_serialized_ids(record.get("gold_doc_ids"))
    gold_tus = parse_serialized_ids(record.get("gold_tu_ids"))
    retrieved_docs = parse_serialized_ids(record.get("retrieved_doc_ids"))
    retrieved_tus = parse_serialized_ids(record.get("retrieved_tu_ids"))

    gold_units = gold_docs or gold_tus
    retrieved_units = retrieved_docs if gold_docs else retrieved_tus
    coverage = (
        len(set(gold_units) & set(retrieved_units)) / len(gold_units)
        if gold_units else 0.0
    )
    doc_hit = bool(gold_docs) and any(doc in set(retrieved_docs) for doc in gold_docs)
    tu_hit = bool(gold_tus) and any(tu in set(retrieved_tus) for tu in gold_tus)
    retrieval_hit = coverage > 0
    full_retrieval_hit = coverage >= 1.0

    missing_tokens = [
        token for token in extract_codes_and_numbers(reference)
        if not answer_contains_token(answer, token)
    ]

    if math.isnan(accuracy):
        error_type = "unjudged"
    elif accuracy >= 0.99:
        error_type = "correct"
    elif system != "vanilla_llm" and not full_retrieval_hit:
        error_type = "retrieval_miss"
    elif refusal:
        error_type = "lazy_refusal"
    elif accuracy >= 0.25:
        error_type = "partial_answer"
    else:
        error_type = "wrong_answer"

    row = {
        "qid": record.get("qid", ""),
        "system": system,
        "system_label": SYSTEM_LABELS.get(system, system),
        "category": record.get("category", ""),
        "question": record.get("question", ""),
        "reference_answer": reference,
        "answer_accuracy": accuracy,
        "answer_accuracy_std": as_float(record.get("answer_accuracy_std")),
        "judge_agreement": as_float(record.get("judge_agreement")),
        "refusal": refusal,
        "retrieval_hit": retrieval_hit,
        "full_retrieval_hit": full_retrieval_hit,
        "retrieval_coverage": coverage,
        "gold_doc_hit": doc_hit,
        "gold_tu_hit": tu_hit,
        "error_type": error_type,
        "error_label": ERROR_LABELS[error_type],
        "missing_tokens": "|".join(missing_tokens),
        "factual_guard_cap": record.get("factual_guard_cap", ""),
        "factual_guard_missing": record.get("factual_guard_missing", ""),
        "answer_snippet": " ".join(answer.split())[:300],
        "judge_rationale_snippet": " ".join(
            str(record.get("judge_rationale", "")).split()
        )[:300],
    }
    return row


def analyze(records: list[dict[str, Any]], category: str) -> pd.DataFrame:
    rows = [classify_record(record) for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if category.lower() != "all":
        frame = frame[frame["category"] == category].copy()
    return frame


def error_counts(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame.groupby(["system", "error_type"]).size().unstack(fill_value=0)
        .reindex(index=[s for s in SYSTEM_ORDER if s in set(frame["system"])], fill_value=0)
        .reindex(columns=ERROR_ORDER, fill_value=0)
    )
    counts["Total"] = counts.sum(axis=1)
    return counts


def error_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["system", "error_type"])["answer_accuracy"].mean().unstack()
        .reindex(index=[s for s in SYSTEM_ORDER if s in set(frame["system"])])
        .reindex(columns=ERROR_ORDER)
    )


def per_question(frame: pd.DataFrame) -> pd.DataFrame:
    accuracy = frame.pivot_table(
        index="qid", columns="system", values="answer_accuracy", aggfunc="mean"
    )
    accuracy = accuracy.reindex(columns=[s for s in SYSTEM_ORDER if s in accuracy.columns])
    questions = frame.drop_duplicates("qid").set_index("qid")["question"]
    accuracy.insert(0, "question", questions.reindex(accuracy.index))
    accuracy["all_systems_fail"] = (accuracy.drop(columns="question").max(axis=1) < 0.5)
    accuracy["all_systems_correct"] = (accuracy.drop(columns=["question", "all_systems_fail"]).min(axis=1) >= 0.99)
    return accuracy.sort_index()


def retrieval_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rag = frame[frame["system"] != "vanilla_llm"]
    if rag.empty:
        return pd.DataFrame()
    rows = []
    for system, group in rag.groupby("system"):
        rows.append({
            "System": SYSTEM_LABELS.get(system, system),
            "Questions": len(group),
            "Gold doc hit": round(float(group["gold_doc_hit"].mean()), 4),
            "Gold TU hit": round(float(group["gold_tu_hit"].mean()), 4),
            "Mean gold coverage": round(float(group["retrieval_coverage"].mean()), 4),
            "Full coverage": round(float(group["full_retrieval_hit"].mean()), 4),
            "Retrieval miss": int((group["error_type"] == "retrieval_miss").sum()),
        })
    return pd.DataFrame(rows)


def representative_cases(frame: pd.DataFrame, examples: int) -> dict[str, list[dict[str, Any]]]:
    cases: dict[str, list[dict[str, Any]]] = {}
    for system in [s for s in SYSTEM_ORDER if s in set(frame["system"])]:
        for error_type in ERROR_ORDER:
            if error_type == "correct":
                continue
            sub = frame[(frame["system"] == system) & (frame["error_type"] == error_type)]
            if sub.empty:
                continue
            if error_type in {"partial_answer", "wrong_answer"}:
                sub = sub.sort_values(["retrieval_hit", "answer_accuracy"], ascending=[False, True])
            else:
                sub = sub.sort_values(["answer_accuracy", "qid"])
            cases[f"{system}:{error_type}"] = sub.head(examples).to_dict("records")
    return cases


def markdown_report(
    frame: pd.DataFrame,
    counts: pd.DataFrame,
    accuracy: pd.DataFrame,
    questions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    cases: dict[str, list[dict[str, Any]]],
    input_path: Path,
    category: str,
) -> str:
    lines = [
        "# Error Analysis",
        "",
        f"- Input: `{input_path}`",
        f"- Category: `{category}`",
        f"- Records: {len(frame)}",
        f"- Questions: {frame['qid'].nunique()}",
        f"- Systems: {', '.join(SYSTEM_LABELS.get(s, s) for s in SYSTEM_ORDER if s in set(frame['system']))}",
        "",
        "## Error taxonomy",
        "",
        counts.rename(columns=ERROR_LABELS).to_markdown(),
        "",
        "## Mean judge accuracy by error type",
        "",
        accuracy.rename(columns=ERROR_LABELS).round(3).to_markdown(),
        "",
    ]

    if not diagnostics.empty:
        lines += ["## Retrieval diagnostics", "", diagnostics.to_markdown(index=False), ""]

    failures = questions[questions["all_systems_fail"]]
    lines += [
        "## Questions where all systems fail",
        "",
    ]
    if failures.empty:
        lines.append("None.")
    else:
        for qid, row in failures.iterrows():
            scores = ", ".join(
                f"{SYSTEM_LABELS.get(system, system)}={row[system]:.2f}"
                for system in SYSTEM_ORDER
                if system in row and pd.notna(row[system])
            )
            lines.append(f"- **{qid}**: {scores} — {str(row['question'])[:220]}")
    lines.append("")

    lines += ["## Representative cases", ""]
    for key, records in cases.items():
        system, error_type = key.split(":", 1)
        lines += [f"### {SYSTEM_LABELS.get(system, system)} — {ERROR_LABELS[error_type]}", ""]
        for record in records:
            lines += [
                f"**{record['qid']}** (score={record['answer_accuracy']:.2f}, "
                f"coverage={record['retrieval_coverage']:.2f}, "
                f"missing=`{record['missing_tokens'] or 'none'}`)",
                "",
                f"- Question: {str(record['question'])[:400]}",
                f"- Reference: {str(record['reference_answer'])[:300]}",
                f"- Answer: {record['answer_snippet']}",
                f"- Judge: {record['judge_rationale_snippet']}",
                "",
            ]

    lines += ["## Recommendations", ""]
    totals = counts.sum()
    if totals.get("lazy_refusal", 0):
        lines.append(
            f"- Còn {int(totals['lazy_refusal'])} lượt từ chối trên câu hỏi có đáp án; "
            "kiểm tra prompt grounding và ngưỡng refusal."
        )
    if totals.get("retrieval_miss", 0):
        lines.append(
            f"- Có {int(totals['retrieval_miss'])} lượt retrieval miss; "
            "tăng top-k, thử hybrid retrieval hoặc query expansion cho câu hỏi nhiều vế."
        )
    if totals.get("partial_answer", 0):
        lines.append(
            f"- Có {int(totals['partial_answer'])} lượt trả lời đúng một phần; "
            "prompt cần yêu cầu trả lời đủ từng vế và nêu rõ tổng/so sánh."
        )
    if totals.get("wrong_answer", 0):
        lines.append(
            f"- Có {int(totals['wrong_answer'])} lượt trả lời sai dù đã truy hồi; "
            "tập trung vào trích xuất mã/tín chỉ và kiểm tra chéo nhiều đoạn."
        )
    if totals.get("unjudged", 0):
        lines.append(f"- Có {int(totals['unjudged'])} record chưa có điểm judge; chạy lại bước judge.")
    if not failures.empty:
        lines.append(
            f"- {len(failures)} câu hỏi mà cả ba hệ thống đều sai; nên xem lại độ rõ ràng của câu hỏi "
            "và evidence/gold trước khi kết luận về mô hình."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/judged_benchmark.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--category", default="multi_hop_relational")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return 1

    records = read_jsonl(args.input)
    frame = analyze(records, args.category)
    if frame.empty:
        print(f"No records for category={args.category!r} in {args.input}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = error_counts(frame)
    accuracy = error_accuracy(frame)
    questions = per_question(frame)
    diagnostics = retrieval_diagnostics(frame)
    cases = representative_cases(frame, args.examples)

    frame.to_csv(args.output_dir / "error_cases.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(args.output_dir / "error_summary.csv", encoding="utf-8-sig")
    questions.to_csv(args.output_dir / "error_by_question.csv", encoding="utf-8-sig")
    if not diagnostics.empty:
        diagnostics.to_csv(args.output_dir / "error_retrieval_diagnostics.csv", index=False, encoding="utf-8-sig")

    report = markdown_report(
        frame, counts, accuracy, questions, diagnostics, cases, args.input, args.category
    )
    (args.output_dir / "error_analysis.md").write_text(report, encoding="utf-8")
    print(counts.rename(columns=ERROR_LABELS).to_string())
    print(f"\nWrote {args.output_dir / 'error_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
