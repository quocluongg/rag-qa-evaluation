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
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats_utils import bootstrap_ci, holm_correction, mcnemar_test, paired_wilcoxon
from text_utils import answer_contains_token, extract_codes_and_numbers

DEFAULT_INPUT = ROOT / "results" / "benchmark_run.jsonl"
DEFAULT_OUT_DIR = ROOT / "results"
DEFAULT_MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite")
DEFAULT_K = 8
JUDGE_VERSION = "v3-repeats-grounded-guard2"


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
[CÂU TRẢ LỜI CỦA MÔ HÌNH]: {answer}
[NGỮ CẢNH TRUY HỒI - CHỈ DÙNG CHO FAITHFULNESS]: {context if context else '(Hệ thống không có ngữ cảnh truy hồi)'}

TIÊU CHÍ CHẤM ĐIỂM:
1. correctness (0.0 -> 1.0) — CHỈ so sánh CÂU TRẢ LỜI với ĐÁP ÁN CHUẨN:
   - 1.0: Trả lời đúng và đầy đủ mọi vế của đáp án chuẩn.
   - 0.5: Trả lời đúng một phần, đúng ý nhưng thiếu vế, hoặc sai một chi tiết không trọng yếu.
   - 0.0: Trả lời sai sự thật, nêu sai mã/số liệu quan trọng, hoặc từ chối sai (câu hỏi có đáp án nhưng lại nói không biết).
   - Nếu câu hỏi thuộc nhóm 'unanswerable_negative' và mô hình trả lời đúng rằng tài liệu không có thông tin -> Cho 1.0 điểm.
   - KHÔNG dùng ngữ cảnh để suy ra đáp án còn thiếu. Ngữ cảnh chỉ dùng cho mục 2.
   - Chấp nhận các cách diễn đạt tương đương và khác biệt định dạng số (ví dụ 26,59 và 26.59; 8.48.01.01 và 8480101).
2. faithfulness (0.0 -> 1.0) — so sánh CÂU TRẢ LỜI với NGỮ CẢNH:
   - 1.0: Không bịa đặt sự thật (mọi dữ kiện đều có trong ngữ cảnh, hoặc từ chối đúng khi không có thông tin).
   - 0.5: Có dữ kiện đúng nhưng thêm một chi tiết không được ngữ cảnh hỗ trợ.
   - 0.0: Bịa đặt dữ kiện không có thực (hallucination).
3. refusal_appropriate:
   - True nếu từ chối đúng lúc (khi không có thông tin) hoặc trả lời đúng khi có thông tin.
   - False nếu bịa đặt khi không có thông tin, hoặc từ chối lười biếng dù tài liệu có thông tin.
{multi_hop_rubric}
Hãy cho điểm khách quan, không suy đoán ngoài đáp án chuẩn, và giải thích ngắn gọn bằng tiếng Việt."""


def factual_consistency_guard(
    category: str,
    reference_answer: str,
    answer: str,
) -> tuple[float | None, list[str]]:
    """Cap correctness when a factual reference token is missing from the answer.

    The LLM judge can be fooled by a plausible but wrong code (e.g. ELE1426 vs
    ELE16246).  For single-hop and multi-hop rows this deterministic guard
    checks course codes, decimal values, dates and x-credit phrases.
    """
    if category not in ("factual_single_hop", "multi_hop_relational"):
        return None, []
    tokens = list(dict.fromkeys(extract_codes_and_numbers(reference_answer)))
    if not tokens:
        return None, []
    matched = [t for t in tokens if answer_contains_token(answer, t)]
    missing = [t for t in tokens if t not in matched]

    if category == "multi_hop_relational":
        return (0.0, missing) if not matched else (None, [])

    if len(tokens) > 2:
        codes = [t for t in tokens if re.fullmatch(r"[A-Z]{2,}\d{3,}", t)]
        others = [t for t in tokens if t not in codes]
        tokens = ([codes[0]] if codes else []) + ([others[0]] if others else [])
        matched = [t for t in tokens if answer_contains_token(answer, t)]
        missing = [t for t in tokens if t not in matched]
    if not missing:
        return None, []
    return (0.0 if not matched else 0.5), missing


def aggregate_judge_scores(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated judge calls into means, standard deviations and agreement."""
    if not repeats:
        return {
            "answer_accuracy": 0.0,
            "answer_accuracy_std": float("nan"),
            "faithfulness": 0.0,
            "faithfulness_std": float("nan"),
            "refusal_appropriate": False,
            "refusal_agreement": 0.0,
            "judge_rationale": "Judge failed for all repeats.",
        }
    correctness = np.asarray([float(r["correctness"]) for r in repeats], dtype=float)
    faithfulness = np.asarray([float(r["faithfulness"]) for r in repeats], dtype=float)
    refusals = np.asarray([bool(r["refusal_appropriate"]) for r in repeats], dtype=bool)
    rounded = np.round(correctness, 2)
    agreement = float(np.mean(rounded == rounded[0]))
    return {
        "answer_accuracy": float(correctness.mean()),
        "answer_accuracy_std": float(correctness.std(ddof=0)),
        "faithfulness": float(faithfulness.mean()),
        "faithfulness_std": float(faithfulness.std(ddof=0)),
        "refusal_appropriate": bool(refusals.mean() >= 0.5),
        "refusal_agreement": float(max(refusals.mean(), 1 - refusals.mean())),
        "judge_rationale": str(repeats[0].get("rationale", "")),
        "judge_agreement": agreement,
    }


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


def format_ci(mean: float, lo: float, hi: float) -> str:
    if any(math.isnan(x) for x in (mean, lo, hi)):
        return "N/A"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def escape_latex(text: Any) -> str:
    return str(text).replace("_", "\\_")


def paired_comparisons(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """McNemar + Wilcoxon for every system pair, overall and per category."""
    pairs = [
        ("vector_rag", "vanilla_llm", "Vector RAG vs Vanilla LLM"),
        ("graph_rag", "vanilla_llm", "GraphRAG vs Vanilla LLM"),
        ("graph_rag", "vector_rag", "GraphRAG vs Vector RAG"),
    ]
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", df)]
    for cat in sorted(df["category"].dropna().unique()):
        groups.append((str(cat), df[df["category"] == cat]))

    rows: list[dict[str, Any]] = []
    for label, sub in groups:
        pivot = sub.pivot_table(
            index="qid", columns="system", values="answer_accuracy", aggfunc="mean"
        )
        for sys_a, sys_b, comparison in pairs:
            if sys_a not in pivot.columns or sys_b not in pivot.columns:
                continue
            paired = pivot[[sys_a, sys_b]].dropna()
            if len(paired) < 2:
                continue
            mc = mcnemar_test(paired[sys_a] >= threshold, paired[sys_b] >= threshold)
            wil = paired_wilcoxon(paired[sys_a], paired[sys_b])
            rows.append({
                "Category": label,
                "Comparison": comparison,
                "N pairs": int(len(paired)),
                "Mean A": float(paired[sys_a].mean()),
                "Mean B": float(paired[sys_b].mean()),
                "Mean Diff (A-B)": float((paired[sys_a] - paired[sys_b]).mean()),
                "McNemar b (A correct, B wrong)": int(mc["b"]),
                "McNemar c (A wrong, B correct)": int(mc["c"]),
                "McNemar p": mc["p_value"],
                "Wilcoxon p": wil,
                "Correct Threshold": threshold,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["McNemar p (Holm)"] = holm_correction(result["McNemar p"].tolist())
        result["Wilcoxon p (Holm)"] = holm_correction(result["Wilcoxon p"].tolist())
        result["Significant after Holm (McNemar)"] = result["McNemar p (Holm)"] < 0.05
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3-System Benchmark Results")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-a-judge scoring")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--judge-repeats", type=int, default=3, help="Number of independent judge calls per answer")
    args = parser.parse_args()
    args.judge_repeats = max(1, args.judge_repeats)

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

        existing_judged: dict[tuple[str, str], dict[str, Any]] = {}
        if args.resume and judged_path.exists():
            lines = [line for line in judged_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for line in lines:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if (
                    r.get("judge_version") == JUDGE_VERSION
                    and int(r.get("judge_repeats", 0)) >= args.judge_repeats
                ):
                    existing_judged[(r.get("system"), r.get("qid"))] = r
            print(f"Resuming judge: found {len(existing_judged)} valid judged records.")
            stale = len(lines) - len(existing_judged)
            if stale > 0:
                backup = judged_path.with_suffix(judged_path.suffix + ".bak")
                backup.write_text(judged_path.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"Discarded {stale} stale judged rows (judge prompt/version mismatch). Backup: {backup}")

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
            repeats: list[dict[str, Any]] = []
            for _ in range(args.judge_repeats):
                try:
                    res: EvaluationScore = judge_llm.invoke(judge_prompt)
                    repeats.append({
                        "correctness": float(res.correctness),
                        "faithfulness": float(res.faithfulness),
                        "refusal_appropriate": bool(res.refusal_appropriate),
                        "rationale": res.rationale,
                    })
                except Exception as e:
                    print(f"Judge error for {key}: {e}")
                time.sleep(2.0)

            aggregated = aggregate_judge_scores(repeats)
            rec.update(aggregated)
            rec["judge_repeats"] = len(repeats)
            rec["judge_version"] = JUDGE_VERSION
            rec["judge_scores"] = repeats

            cap, missing = factual_consistency_guard(cat, ref, ans)
            rec["factual_guard_cap"] = cap
            rec["factual_guard_missing"] = "|".join(missing)
            if cap is not None:
                rec["answer_accuracy"] = min(float(rec["answer_accuracy"]), cap)

            with judged_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            updated_records.append(rec)
            if i % 10 == 0:
                print(f"Judged {i}/{len(records)} records...")

        judged_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in updated_records) + "\n",
            encoding="utf-8",
        )
        records = updated_records

    # Aggregation per system
    k = int(records[0].get("top_k", args.k)) if records else args.k
    if k != args.k:
        print(f"Using top_k={k} from benchmark records (CLI --k={args.k}).")

    df = pd.DataFrame(records)
    systems = df["system"].unique().tolist()
    summary_rows = []

    for sys_name in systems:
        sub = df[df["system"] == sys_name]
        is_vanilla = sys_name == "vanilla_llm"

        retrieved_docs = [r["retrieved_doc_ids"] for _, r in sub.iterrows()]
        gold_docs = [gold_docs_for_row(r, source_map) for _, r in sub.iterrows()]
        p_k = None if is_vanilla else np.mean([
            precision_at_k(retrieved, gold, k)
            for retrieved, gold in zip(retrieved_docs, gold_docs)
        ])
        r_k = None if is_vanilla else np.mean([
            recall_at_k(retrieved, gold, k)
            for retrieved, gold in zip(retrieved_docs, gold_docs)
        ])
        mrr_val = None if is_vanilla else np.mean([
            mrr(retrieved, gold)
            for retrieved, gold in zip(retrieved_docs, gold_docs)
        ])

        acc_vals = pd.to_numeric(sub.get("answer_accuracy"), errors="coerce").dropna().tolist() if "answer_accuracy" in sub else []
        faith_vals = pd.to_numeric(sub.get("faithfulness"), errors="coerce").dropna().tolist() if "faithfulness" in sub else []
        acc, acc_lo, acc_hi = bootstrap_ci(acc_vals)
        faith, faith_lo, faith_hi = bootstrap_ci(faith_vals)
        judge_sd = pd.to_numeric(sub.get("answer_accuracy_std"), errors="coerce").mean() if "answer_accuracy_std" in sub else float("nan")
        refusal_rate = sub["answer_is_refusal"].mean() if "answer_is_refusal" in sub else float("nan")

        ret_lat_med = np.median(sub["retrieval_latency_s"])
        gen_lat_med = np.median(sub["generation_latency_s"])
        tot_lat_med = np.median(sub["total_latency_s"])
        tot_lat_p95 = np.percentile(sub["total_latency_s"], 95)
        tokens_mean = sub["ctx_tokens"].mean()

        summary_rows.append({
            "System": sys_name,
            f"P@{k}": round(p_k, 4) if p_k is not None else "N/A",
            f"R@{k}": round(r_k, 4) if r_k is not None else "N/A",
            "MRR": round(mrr_val, 4) if mrr_val is not None else "N/A",
            "Answer Accuracy": round(acc, 4) if not math.isnan(acc) else "N/A",
            "Accuracy 95% CI": format_ci(acc, acc_lo, acc_hi),
            "Judge Score SD": round(judge_sd, 4) if not math.isnan(judge_sd) else "N/A",
            "Faithfulness": round(faith, 4) if not math.isnan(faith) else "N/A",
            "Faithfulness 95% CI": format_ci(faith, faith_lo, faith_hi),
            "Refusal Rate": round(refusal_rate, 4) if not math.isnan(refusal_rate) else "N/A",
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
        acc_vals = pd.to_numeric(group.get("answer_accuracy"), errors="coerce").dropna().tolist() if "answer_accuracy" in group else []
        faith_vals = pd.to_numeric(group.get("faithfulness"), errors="coerce").dropna().tolist() if "faithfulness" in group else []
        acc, acc_lo, acc_hi = bootstrap_ci(acc_vals)
        faith, faith_lo, faith_hi = bootstrap_ci(faith_vals)
        if sys_name == "vanilla_llm" or cat == "unanswerable_negative":
            recall_k = "N/A"
        else:
            retrieved = [r["retrieved_doc_ids"] for _, r in group.iterrows()]
            gold = [gold_docs_for_row(r, source_map) for _, r in group.iterrows()]
            recall_k = round(float(np.mean([recall_at_k(r, g, k) for r, g in zip(retrieved, gold)])), 4)
        judge_sd = pd.to_numeric(group.get("answer_accuracy_std"), errors="coerce").mean() if "answer_accuracy_std" in group else float("nan")
        cat_rows.append({
            "System": sys_name,
            "Category": cat,
            "Count": len(group),
            "Answer Accuracy": round(acc, 4) if not math.isnan(acc) else "N/A",
            "Accuracy 95% CI": format_ci(acc, acc_lo, acc_hi),
            "Faithfulness": round(faith, 4) if not math.isnan(faith) else "N/A",
            f"Retrieval Recall@{k}": recall_k,
            "Refusal Rate": round(float(group["answer_is_refusal"].mean()), 4) if "answer_is_refusal" in group else "N/A",
            "Judge Score SD": round(judge_sd, 4) if not math.isnan(judge_sd) else "N/A",
        })
    cat_df = pd.DataFrame(cat_rows)
    cat_csv = out_dir / "category_breakdown.csv"
    cat_df.to_csv(cat_csv, index=False, encoding="utf-8-sig")

    # Judge reliability and factual-guard audit
    rel_rows = []
    for sys_name, group in df.groupby("system"):
        rel_rows.append({
            "System": sys_name,
            "Judge Repeats (mean)": round(float(group["judge_repeats"].mean()), 2) if "judge_repeats" in group else "N/A",
            "Correctness SD (mean)": round(float(pd.to_numeric(group.get("answer_accuracy_std"), errors="coerce").mean()), 4) if "answer_accuracy_std" in group else "N/A",
            "Faithfulness SD (mean)": round(float(pd.to_numeric(group.get("faithfulness_std"), errors="coerce").mean()), 4) if "faithfulness_std" in group else "N/A",
            "Judge Agreement": round(float(pd.to_numeric(group.get("judge_agreement"), errors="coerce").mean()), 4) if "judge_agreement" in group else "N/A",
        })
    rel_df = pd.DataFrame(rel_rows)
    rel_df.to_csv(out_dir / "judge_reliability.csv", index=False, encoding="utf-8-sig")

    if "factual_guard_cap" in df:
        guard = df[df["factual_guard_cap"].notna()]
        guard_rows = []
        for sys_name in systems:
            sub_guard = guard[guard["system"] == sys_name]
            guard_rows.append({
                "System": sys_name,
                "Guarded Count": len(sub_guard),
                "Capped to 0.0": int((pd.to_numeric(sub_guard["factual_guard_cap"], errors="coerce") == 0.0).sum()),
                "Capped to 0.5": int((pd.to_numeric(sub_guard["factual_guard_cap"], errors="coerce") == 0.5).sum()),
            })
        pd.DataFrame(guard_rows).to_csv(out_dir / "factual_guard_summary.csv", index=False, encoding="utf-8-sig")

    # Statistical Significance Testing (paired McNemar + Wilcoxon, Holm-corrected)
    sig_df = paired_comparisons(df) if "answer_accuracy" in df else pd.DataFrame()
    if not sig_df.empty:
        sig_df.to_csv(out_dir / "significance_by_category.csv", index=False, encoding="utf-8-sig")
        overall = sig_df[sig_df["Category"] == "ALL"]
        overall.to_csv(out_dir / "significance_tests.csv", index=False, encoding="utf-8-sig")
        print(f"\n=== STATISTICAL SIGNIFICANCE TESTS (Holm-corrected) ===\n")
        show_cols = [
            "Category", "Comparison", "N pairs", "Mean Diff (A-B)",
            "McNemar p", "McNemar p (Holm)", "Wilcoxon p (Holm)",
            "Significant after Holm (McNemar)",
        ]
        print(sig_df[show_cols].to_string(index=False))

    # LaTeX Table Generation
    latex_table = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{So sánh tổng thể 3 hệ thống: LLM thuần, Vector RAG và GraphRAG}\n"
        "\\label{tab:system_comparison}\n"
        "\\begin{tabular}{lcccccccc}\n"
        "\\toprule\n"
        f"\\textbf{{Hệ thống}} & \\textbf{{P@{k}}} & \\textbf{{R@{k}}} & \\textbf{{MRR}} & \\textbf{{Accuracy [95\\% CI]}} & \\textbf{{Faithfulness [95\\% CI]}} & \\textbf{{Lat. Med (s)}} & \\textbf{{Lat. P95 (s)}} & \\textbf{{Tokens}} \\\\\n"
        "\\midrule\n"
    )
    for _, r in summary_df.iterrows():
        latex_table += (
            f"{escape_latex(r['System']):<16} & {str(r[f'P@{k}']):<6} & {str(r[f'R@{k}']):<6} & {str(r['MRR']):<6} & "
            f"{str(r['Accuracy 95% CI']):<22} & {str(r['Faithfulness 95% CI']):<22} & "
            f"{r['Latency Median (s)']:<12} & {r['Latency P95 (s)']:<12} & {r['Context Tokens']} \\\\\n"
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
