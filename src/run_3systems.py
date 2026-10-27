"""
run_3systems.py — Controlled Benchmark Runner comparing 3 Systems:
  1. Vanilla LLM  (Zero-shot, Parametric memory only, No Retrieval)
  2. Vector RAG   (Dense Embedding BAAI/bge-m3 + LanceDB ANN)
  3. GraphRAG     (Microsoft GraphRAG Entity-Relation BFS + Re-ranking)

Features:
- Streaming JSONL record output with instant checkpointing
- Full resume support (--resume) across disconnections
- Detailed latency profiling: retrieval_latency_s vs generation_latency_s
- Token consumption tracking
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_utils import looks_like_refusal

DEFAULT_TESTSET = ROOT / "data" / "scientific_testset_v2.csv"
DEFAULT_OUTPUT = ROOT / "results" / "benchmark_run.jsonl"
DEFAULT_MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite")
DEFAULT_K = 8
PROMPT_VERSION = "v2-grounded-reasoning"


def prompt_vanilla(query: str) -> str:
    return (
        "Bạn là trợ lý AI học thuật. Hãy trả lời câu hỏi tiếng Việt sau đây một cách chính xác dựa trên kiến thức của bạn. "
        "Nếu bạn không biết hoặc thông tin không chắc chắn, hãy trả lời rõ: "
        "'Tôi không có đủ thông tin để trả lời câu hỏi này.' Không bịa đặt thông tin.\n\n"
        f"Câu hỏi: {query}\n\nTrả lời:"
    )


def prompt_rag(query: str, context: str) -> str:
    return (
        "Bạn là trợ lý AI học thuật. Hãy trả lời câu hỏi tiếng Việt sau đây CHỈ dựa trên các đoạn bằng chứng được cung cấp bên dưới.\n\n"
        "QUY TẮC TRẢ LỜI:\n"
        "1. Nếu câu hỏi có nhiều vế, phải trả lời ĐẦY ĐỦ từng vế, không bỏ sót.\n"
        "2. Nếu thông tin nằm rải rác ở nhiều đoạn, hãy tổng hợp tất cả dữ kiện liên quan trước khi trả lời.\n"
        "3. Được phép suy luận logic từ các dữ kiện CÓ TRONG bằng chứng (ví dụ: cộng/so sánh số tín chỉ, đối chiếu mốc thời gian). "
        "Không được dùng kiến thức bên ngoài bằng chứng và không được bịa số liệu, mã, tên.\n"
        "4. Chỉ từ chối khi bằng chứng KHÔNG chứa thông tin cho TẤT CẢ các vế. Nếu chỉ thiếu một phần, hãy trả lời phần có bằng chứng và nói rõ phần nào còn thiếu.\n"
        "5. Trả lời súc tích, nêu rõ con số, mã và tên liên quan.\n\n"
        f"Câu hỏi: {query}\n\n"
        f"Bằng chứng được cung cấp:\n{context}\n\n"
        "Trả lời:"
    )


def load_completed_records(out_path: Path, prompt_version: str, top_k: int) -> set[tuple[str, str]]:
    """Load resume state, discarding rows produced by an older prompt/config."""
    if not out_path.exists():
        return set()
    raw = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid = [
        r for r in raw
        if r.get("prompt_version") == prompt_version and r.get("top_k") == top_k
    ]
    stale = len(raw) - len(valid)
    if stale:
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        backup.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw), encoding="utf-8")
        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for r in valid:
            dedup[(r.get("system", ""), r.get("qid", ""))] = r
        valid = list(dedup.values())
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in valid) + ("\n" if valid else ""),
            encoding="utf-8",
        )
        print(f"Discarded {stale} stale records (prompt/config mismatch). Backup: {backup}")
    return {(r.get("system", ""), r.get("qid", "")) for r in valid}


def extract_text(resp: Any) -> str:
    content = resp.content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content).strip()
    return str(content).strip()


def main():
    parser = argparse.ArgumentParser(description="Run 3-System RAG Benchmark")
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K context units for RAG systems")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is required for answer generation")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from retrieval.vanilla_llm import VanillaLLMRetriever
    from retrieval.vector_rag import VectorRAG
    from retrieval.graph_rag import GraphRAG
    from retrieval.source_map import get_source_map
    from testset_utils import repair_gold_doc_ids, split_ids

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[tuple[str, str]] = set()
    if args.resume:
        completed = load_completed_records(out_path, PROMPT_VERSION, args.k)
        print(f"Resume mode: loaded {len(completed)} completed (system, qid) evaluations.")

    with open(args.testset, encoding="utf-8", newline="") as f:
        test_rows = list(csv.DictReader(f))
    repaired = repair_gold_doc_ids(test_rows, get_source_map())
    print(f"Loaded testset: {len(test_rows)} queries from {args.testset}")
    if repaired:
        print(f"Recovered gold_doc_ids for {repaired} queries from gold_tu_ids.")

    # Shared generation LLM with zero temperature for deterministic, fair comparison
    answer_llm = ChatGoogleGenerativeAI(
        model=args.model,
        temperature=0.0,
        google_api_key=api_key,
    )

    systems = {
        "vanilla_llm": VanillaLLMRetriever(k=0),
        "vector_rag": VectorRAG(k=args.k),
        "graph_rag": GraphRAG(k=args.k),
    }

    manifest = {
        "model": args.model,
        "temperature": 0.0,
        "top_k": args.k,
        "prompt_version": PROMPT_VERSION,
        "testset": str(args.testset),
        "total_queries": len(test_rows),
        "systems": list(systems.keys()),
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (out_path.parent / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total_ops = len(test_rows) * len(systems)
    curr_op = len(completed)

    for row in test_rows:
        qid = row.get("qid", "").strip()
        query = row.get("question", "").strip()
        cat = row.get("category", "general").strip()
        ref_ans = row.get("reference_answer", "").strip()
        gold_docs = split_ids(row.get("gold_doc_ids"))
        gold_tus = split_ids(row.get("gold_tu_ids"))

        for sys_name, retriever in systems.items():
            if (sys_name, qid) in completed:
                continue

            curr_op += 1
            print(f"[{curr_op}/{total_ops}] System: {sys_name:<12} | QID: {qid} ({cat})")

            # 1. Retrieval Phase
            t_ret_start = time.perf_counter()
            ret_result = retriever.retrieve(query)
            retrieval_latency = ret_result.latency_s if ret_result.latency_s > 0 else (time.perf_counter() - t_ret_start)

            # 2. Context Building & Generation Phase
            if sys_name == "vanilla_llm":
                prompt = prompt_vanilla(query)
                context_str = ""
                ctx_tokens = 0
                retrieved_docs = []
                retrieved_tus = []
                evidence_list = []
            else:
                context_str = "\n\n".join(
                    f"[{i+1}] Tài liệu: {ev.filename}\n{ev.content}"
                    for i, ev in enumerate(ret_result.evidence)
                )
                prompt = prompt_rag(query, context_str)
                ctx_tokens = ret_result.ctx_tokens
                retrieved_docs = ret_result.document_ids
                retrieved_tus = [ev.text_unit_id for ev in ret_result.evidence]
                evidence_list = [
                    {"filename": ev.filename, "text_unit_id": ev.text_unit_id, "score": ev.score}
                    for ev in ret_result.evidence
                ]

            t_gen_start = time.perf_counter()
            try:
                raw_resp = answer_llm.invoke(prompt)
                ans = extract_text(raw_resp)
            except Exception as e:
                print(f"Generation error for {sys_name} on {qid}: {e}")
                ans = f"Error during generation: {str(e)}"
            generation_latency = time.perf_counter() - t_gen_start

            record = {
                "system": sys_name,
                "qid": qid,
                "category": cat,
                "question": query,
                "reference_answer": ref_ans,
                "gold_doc_ids": gold_docs,
                "gold_tu_ids": gold_tus,
                "retrieved_doc_ids": retrieved_docs,
                "retrieved_tu_ids": retrieved_tus,
                "evidence": evidence_list,
                "context": context_str,
                "answer": ans,
                "retrieval_latency_s": retrieval_latency if sys_name != "vanilla_llm" else 0.0,
                "generation_latency_s": generation_latency,
                "total_latency_s": (retrieval_latency if sys_name != "vanilla_llm" else 0.0) + generation_latency,
                "ctx_tokens": ctx_tokens,
                "prompt_version": PROMPT_VERSION,
                "top_k": args.k,
                "answer_is_refusal": looks_like_refusal(ans),
                "answer_accuracy": None,
                "faithfulness": None,
            }

            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed.add((sys_name, qid))
            time.sleep(2.0)

    print(f"\nAll benchmark runs completed! Output saved to: {out_path}")


if __name__ == "__main__":
    main()
