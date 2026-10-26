"""
eval_3systems.py — Comprehensive Scientific Evaluation Pipeline for the 3 Systems.

Calculates:
1. Retrieval Metrics: P@K, R@K, MRR (N/A for Vanilla LLM).
2. Answer Correctness & Faithfulness via Structured LLM-as-a-Judge.
3. Special metric for Unanswerable questions: Refusal Accuracy.
4. Efficiency Metrics: Latency (Retrieval vs Generation), Context Tokens.
5. Statistical Significance Tests (Paired Wilcoxon / t-test).
6. Machine-readable CSV & LaTeX Table Export.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "benchmark_run.jsonl"
DEFAULT_OUT_DIR = ROOT / "results"
DEFAULT_MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite")


class EvaluationScore(BaseModel):
    correctness: float = Field(
        description="Điểm chính xác từ 0.0 đến 1.0 so với câu trả lời chuẩn (1.0 = đúng hoàn toàn, 0.5 = đúng một phần, 0.0 = sai hoặc không đúng ý)"
    )
    faithfulness: float = Field(
        description="Điểm trung thực từ 0.0 đến 1.0: 1.0 nếu câu trả lời không hề bịa đặt (hoàn toàn được hỗ trợ bởi bằng chứng hoặc từ chối đúng khi không có bằng chứng); 0.0 nếu bịa đặt thông tin (hallucination)"
    )
    refusal_appropriate: bool = Field(
        description="True nếu câu hỏi không có đáp án mà mô hình từ chối đúng, hoặc câu hỏi có đáp án và mô hình trả lời được; False nếu bịa đặt khi không có đáp án hoặc từ chối vô lý"
    )
    rationale: str = Field(description="Giải thích ngắn gọn lý do chấm điểm bằng tiếng Việt")


def build_judge_prompt(
    question: str,
    category: str,
    reference_answer: str,
    context: str,
    answer: str,
) -> str:
    multi_hop_rubric = ""
    if category == "multi_hop_relational":
        multi_hop_rubric = """
4. Quy tắc riêng cho câu hỏi multi_hop_relational:
   - Nếu câu trả lời chỉ đúng một vế trong hai vế của đáp án chuẩn, correctness tối đa là 0.5.
   - Nếu câu trả lời đúng cả hai vế, correctness là 1.0.
   - Nếu câu trả lời từ chối toàn bộ hoặc không nêu thông tin cho cả hai vế, correctness là 0.0.
   - Không suy đoán rằng ngữ cảnh có thông tin nếu câu trả lời không nêu; chỉ chấm dựa trên nội dung câu trả lời và đáp án chuẩn.
"""

    return f"""Bạn là trọng tài AI công tâm trong cuộc thi đánh giá hệ thống hỏi đáp RAG.
Hãy đánh giá câu trả lời của hệ thống theo câu hỏi và đáp án chuẩn.

[CÂU HỎI]: {question}
[PHÂN LOẠI]: {category}
[ĐÁP ÁN CHUẨN]: {reference_answer}
[NGỮ CẢNH TRUY HỒI ĐƯỢC CẤP]: {context if context else '(Hệ thống không có ngữ cảnh truy hồi)'}
[CÂU TRẢ LỜI CỦA MÔ HÌNH]: {answer}

TIÊU CHÍ CHẤM ĐIỂM:
1. correctness (0.0 -> 1.0):
   - 1.0: Trả lời đúng sự thật theo đáp án chuẩn.
   - 0.5: Trả lời đúng một phần hoặc có ý đúng nhưng chưa đầy đủ.
   - 0.0: Trả lời sai sự thật, hoặc từ chối sai (câu hỏi có đáp án nhưng lại nói không biết).
   - Nếu câu hỏi thuộc nhóm 'unanswerable_negative' và mô hình trả lời đúng rằng tài liệu không có thông tin -> Cho 1.0 điểm.
2. faithfulness (0.0 -> 1.0):
   - 1.0: Không bịa đặt sự thật (mọi dữ kiện đều có trong ngữ cảnh, hoặc từ chối đúng khi không có thông tin).
   - 0.0: Bịa đặt dữ kiện không có thực (hallucination).
3. refusal_appropriate:
   - True nếu từ chối đúng lúc (khi không có thông tin) hoặc trả lời đúng khi có thông tin.
   - False nếu bịa đặt khi không có thông tin, hoặc từ chối lười biếng dù tài liệu có thông tin.
{multi_hop_rubric}
Hãy cho điểm khách quan và giải thích ngắn gọn bằng tiếng Việt."""


def precision_at_k(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in gold)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    if not gold or gold == ["NONE"]:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for d in top_k if d in gold)
    return hits / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    if not gold or gold == ["NONE"]:
        return 0.0
    gold_set = set(gold)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in gold_set:
            return 1.0 / rank
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3-System Benchmark Results")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-a-judge scoring")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    records = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(records)} benchmark records.")

    from retrieval.source_map import get_source_map
    from testset_utils import gold_docs_for_row

    source_map = get_source_map()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judged_path = out_dir / "judged_benchmark.jsonl"

    if args.judge:
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is required for --judge")

        from langchain_google_genai import ChatGoogleGenerativeAI
        judge_llm = ChatGoogleGenerativeAI(
            model=args.model,
            temperature=0.0,
            google_api_key=api_key,
        ).with_structured_output(EvaluationScore)

        existing_judged = {}
        if args.resume and judged_path.exists():
            for line in judged_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        existing_judged[(r.get("system"), r.get("qid"))] = r
                    except Exception:
                        pass
            print(f"Resuming judge: found {len(existing_judged)} previously judged records.")

        updated_records = []
        for i, rec in enumerate(records, start=1):
            key = (rec.get("system"), rec.get("qid"))
            if key in existing_judged:
                updated_records.append(existing_judged[key])
                continue

            q = rec.get("question", "")
            ref = rec.get("reference_answer", "")
            ans = rec.get("answer", "")
            ctx = rec.get("context", "")
            cat = rec.get("category", "")

            judge_prompt = build_judge_prompt(q, cat, ref, ctx, ans)

            try:
                res: EvaluationScore = judge_llm.invoke(judge_prompt)
                rec["answer_accuracy"] = float(res.correctness)
                rec["faithfulness"] = float(res.faithfulness)
                rec["refusal_appropriate"] = res.refusal_appropriate
                rec["judge_rationale"] = res.rationale
            except Exception as e:
                print(f"Judge error for {key}: {e}")
                rec["answer_accuracy"] = 0.0
                rec["faithfulness"] = 0.0
                rec["refusal_appropriate"] = False
                rec["judge_rationale"] = str(e)

            with judged_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            updated_records.append(rec)
            if i % 10 == 0:
                print(f"Judged {i}/{len(records)} records...")
            time.sleep(2.0)

        records = updated_records

    # Aggregation per system
    df = pd.DataFrame(records)
    systems = df["system"].unique().tolist()
    summary_rows = []

    for sys_name in systems:
        sub = df[df["system"] == sys_name]
        is_vanilla = sys_name == "vanilla_llm"

        retrieved_docs = [r["retrieved_doc_ids"] for _, r in sub.iterrows()]
        gold_docs = [gold_docs_for_row(r, source_map) for _, r in sub.iterrows()]
        p_k = None if is_vanilla else np.mean([
            precision_at_k(retrieved, gold, args.k)
            for retrieved, gold in zip(retrieved_docs, gold_docs)
        ])
        r_k = None if is_vanilla else np.mean([
            recall_at_k(retrieved, gold, args.k)
            for retrieved, gold in zip(retrieved_docs, gold_docs)
        ])
        mrr_val = None if is_vanilla else np.mean([
            mrr(retrieved, gold)
            for retrieved, gold in zip(retrieved_docs, gold_docs)
        ])

        acc = sub["answer_accuracy"].dropna().mean() if "answer_accuracy" in sub else None
        faith = sub["faithfulness"].dropna().mean() if "faithfulness" in sub else None

        ret_lat_med = np.median(sub["retrieval_latency_s"])
        gen_lat_med = np.median(sub["generation_latency_s"])
        tot_lat_med = np.median(sub["total_latency_s"])
        tot_lat_p95 = np.percentile(sub["total_latency_s"], 95)
        tokens_mean = sub["ctx_tokens"].mean()

        summary_rows.append({
            "System": sys_name,
            "P@5": round(p_k, 4) if p_k is not None else "N/A",
            "R@5": round(r_k, 4) if r_k is not None else "N/A",
            "MRR": round(mrr_val, 4) if mrr_val is not None else "N/A",
            "Answer Accuracy": round(acc, 4) if acc is not None else "N/A",
            "Faithfulness": round(faith, 4) if faith is not None else "N/A",
            "Latency Median (s)": round(tot_lat_med, 3),
            "Latency P95 (s)": round(tot_lat_p95, 3),
            "Context Tokens": round(tokens_mean, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n=== SUMMARY BENCHMARK METRICS ===\n")
    print(summary_df.to_string(index=False))

    # Category Breakdown
    cat_rows = []
    for (sys_name, cat), group in df.groupby(["system", "category"]):
        cat_rows.append({
            "System": sys_name,
            "Category": cat,
            "Count": len(group),
            "Answer Accuracy": round(group["answer_accuracy"].mean(), 4) if "answer_accuracy" in group else "N/A",
            "Faithfulness": round(group["faithfulness"].mean(), 4) if "faithfulness" in group else "N/A",
            "Total Latency Median (s)": round(group["total_latency_s"].median(), 3),
        })
    cat_df = pd.DataFrame(cat_rows)
    cat_csv = out_dir / "category_breakdown.csv"
    cat_df.to_csv(cat_csv, index=False, encoding="utf-8-sig")

    # Statistical Significance Testing (Wilcoxon Signed Rank & Paired t-test)
    if "answer_accuracy" in df and len(systems) >= 2:
        test_results = []
        pivoted_acc = df.pivot(index="qid", columns="system", values="answer_accuracy").dropna()
        if "vanilla_llm" in pivoted_acc.columns and "vector_rag" in pivoted_acc.columns:
            w_stat, w_p = stats.wilcoxon(pivoted_acc["vector_rag"], pivoted_acc["vanilla_llm"])
            t_stat, t_p = stats.ttest_rel(pivoted_acc["vector_rag"], pivoted_acc["vanilla_llm"])
            test_results.append({
                "Comparison": "Vector RAG vs Vanilla LLM",
                "Metric": "Answer Accuracy",
                "Wilcoxon p-value": round(w_p, 5),
                "Paired t-test p-value": round(t_p, 5),
                "Significant (p < 0.05)": bool(w_p < 0.05),
            })
        if "graph_rag" in pivoted_acc.columns and "vector_rag" in pivoted_acc.columns:
            w_stat, w_p = stats.wilcoxon(pivoted_acc["graph_rag"], pivoted_acc["vector_rag"])
            t_stat, t_p = stats.ttest_rel(pivoted_acc["graph_rag"], pivoted_acc["vector_rag"])
            test_results.append({
                "Comparison": "GraphRAG vs Vector RAG",
                "Metric": "Answer Accuracy",
                "Wilcoxon p-value": round(w_p, 5),
                "Paired t-test p-value": round(t_p, 5),
                "Significant (p < 0.05)": bool(w_p < 0.05),
            })

        if test_results:
            stat_df = pd.DataFrame(test_results)
            stat_df.to_csv(out_dir / "significance_tests.csv", index=False, encoding="utf-8-sig")
            print(f"\n=== STATISTICAL SIGNIFICANCE TESTS ===\n")
            print(stat_df.to_string(index=False))

    # LaTeX Table Generation
    latex_table = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{So sánh tổng thể 3 hệ thống: LLM thuần, Vector RAG và GraphRAG}\n"
        "\\label{tab:system_comparison}\n"
        "\\begin{tabular}{lcccccccc}\n"
        "\\toprule\n"
        "\\textbf{Hệ thống} & \\textbf{P@5} & \\textbf{R@5} & \\textbf{MRR} & \\textbf{Accuracy} & \\textbf{Faithfulness} & \\textbf{Lat. Med (s)} & \\textbf{Lat. P95 (s)} & \\textbf{Tokens} \\\\\n"
        "\\midrule\n"
    )
    for _, r in summary_df.iterrows():
        latex_table += (
            f"{r['System']:<16} & {str(r['P@5']):<6} & {str(r['R@5']):<6} & {str(r['MRR']):<6} & "
            f"{str(r['Answer Accuracy']):<8} & {str(r['Faithfulness']):<12} & {r['Latency Median (s)']:<12} & "
            f"{r['Latency P95 (s)']:<12} & {r['Context Tokens']} \\\\\n"
        )
    latex_table += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    (out_dir / "summary_table.tex").write_text(latex_table, encoding="utf-8")
    print(f"\nLaTeX table exported to: {out_dir / 'summary_table.tex'}")


if __name__ == "__main__":
    main()
