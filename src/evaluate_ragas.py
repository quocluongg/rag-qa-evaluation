"""Evaluate benchmark JSONL and export machine-readable summaries.

`--judge` enables a paired, structured-output judge using Gemini 3.1 Flash-Lite.
Without `--judge`, retrieval metrics are still computed and answer_accuracy stays
N/A rather than being fabricated.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Evaluate four-system benchmark JSONL")
parser.add_argument("--input", default=str(ROOT / "results" / "run.jsonl"))
parser.add_argument("--output-dir", default=str(ROOT / "results"))
parser.add_argument("--judge", action="store_true")
parser.add_argument("--model", default=os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite"))
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

from eval.metrics_custom import compute_all_metrics, graph_gain_to_latex, results_to_latex_row

records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
if not records:
    raise ValueError("Input run JSONL is empty")
required = {"system", "qid", "retrieved_doc_ids", "gold_doc_ids", "category"}
missing = required - set(records[0])
if missing:
    raise ValueError(f"Run records missing fields: {sorted(missing)}")

if args.judge:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is required with --judge")
    from pydantic import BaseModel, Field
    from langchain_google_genai import ChatGoogleGenerativeAI

    class JudgeScore(BaseModel):
        score: float = Field(description="0 for incorrect/unsupported, 1 for fully correct and supported")
        rationale: str = Field(description="Short Vietnamese rationale grounded in the reference answer and evidence")

    judge = ChatGoogleGenerativeAI(model=args.model, temperature=0.0, google_api_key=api_key).with_structured_output(JudgeScore)
    judged_path = Path(args.output_dir) / "judged.jsonl"
    judged_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.resume and judged_path.exists():
        existing = {(r.get("system"), r.get("qid")): r for r in (json.loads(line) for line in judged_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    else:
        judged_path.write_text("", encoding="utf-8")
    updated = []
    for record in records:
        key = (record.get("system"), record.get("qid"))
        if key in existing:
            record.update({"answer_accuracy": existing[key].get("answer_accuracy"), "judge_rationale": existing[key].get("judge_rationale", "")})
            updated.append(record)
            continue
        prompt = f"""Chấm câu trả lời RAG tiếng Việt theo reference answer và evidence. Chỉ dùng thông tin trong evidence.\nQuestion: {record.get('user_input', '')}\nReference answer: {record.get('reference_answer', '')}\nGenerated answer: {record.get('answer', '')}\nEvidence:\n{record.get('context', '')}\nScore 1 nếu đúng đầy đủ và được evidence hỗ trợ; score 0 nếu sai hoặc không được hỗ trợ; dùng số thực ở giữa cho partial correctness."""
        result = judge.invoke(prompt)
        record["answer_accuracy"] = max(0.0, min(1.0, float(result.score)))
        record["judge_rationale"] = result.rationale
        with judged_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        updated.append(record)
        time.sleep(0.5)
    records = updated
    input_records_path = judged_path
else:
    input_records_path = Path(args.input)

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
summary_rows = []
metrics_by_system = {}
for system in sorted({r["system"] for r in records}):
    subset = [r for r in records if r["system"] == system]
    metrics = compute_all_metrics(subset, k=5)
    metrics_by_system[system] = metrics
    summary_rows.append({
        "system": system,
        "n": metrics["n"],
        "precision_at_5": metrics["precision_at_k"],
        "recall_at_5": metrics["recall_at_k"],
        "mrr": metrics["mrr"],
        "temporal_accuracy": metrics["temporal_accuracy"],
        "answer_accuracy": metrics["answer_accuracy"],
        "retrieval_latency_s": metrics["retrieval_latency_s"],
        "ctx_tokens_mean": metrics["ctx_tokens_mean"],
    })

with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = list(summary_rows[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary_rows)

with (out_dir / "summary_latex_rows.tex").open("w", encoding="utf-8") as handle:
    for system in sorted(metrics_by_system):
        handle.write(results_to_latex_row(system, metrics_by_system[system]) + "\n")

# Compute category-level graph gain only where both systems have judged scores.
if "temporal_vector_rag" in {r["system"] for r in records} and "temporal_graph_rag" in {r["system"] for r in records}:
    tv = [r for r in records if r["system"] == "temporal_vector_rag" and r.get("answer_accuracy") is not None]
    tg = [r for r in records if r["system"] == "temporal_graph_rag" and r.get("answer_accuracy") is not None]
    tv_cat = {}
    tg_cat = {}
    for r in tv:
        tv_cat.setdefault(r["category"], []).append(r["answer_accuracy"])
    for r in tg:
        tg_cat.setdefault(r["category"], []).append(r["answer_accuracy"])
    tv_metrics = {c: {"answer_accuracy": sum(v) / len(v)} for c, v in tv_cat.items() if v}
    tg_metrics = {c: {"answer_accuracy": sum(v) / len(v)} for c, v in tg_cat.items() if v}
    (out_dir / "graph_gain_latex_rows.tex").write_text(graph_gain_to_latex(tv_metrics, tg_metrics) + "\n", encoding="utf-8")

print(f"Evaluated {len(records)} records. Summary: {out_dir / 'summary.csv'}")
print(f"Input records: {input_records_path}")
