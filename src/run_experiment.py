"""Run the four retrieval systems on the frozen 50-query testset.

This runner writes one JSONL record per (system, query), so an interrupted run
can be resumed safely with --resume. Answer generation uses the same model for
all systems; answer judging is handled separately by evaluate_ragas.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTSET = ROOT / "data" / "testset_v1.csv"
DEFAULT_OUTPUT = ROOT / "results" / "run.jsonl"
MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite")

parser = argparse.ArgumentParser(description="Run four-system Temporal GraphRAG benchmark")
parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
parser.add_argument("--model", default=MODEL)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

if args.k <= 0:
    raise ValueError("--k must be positive")
api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY is required for answer generation")

from langchain_google_genai import ChatGoogleGenerativeAI
from retrieval.graph_rag import GraphRAG
from retrieval.temporal_graph_rag import TemporalGraphRAG
from retrieval.temporal_vector_rag import TemporalVectorRAG
from retrieval.vector_rag import VectorRAG


def normalize_tq(value: str | None) -> str | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    raise ValueError(f"Invalid tq {value!r}; use dd/mm/YYYY or YYYY-MM-DD")


def model_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except Exception:
        return None


def answer_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content).strip()
    return str(content).strip()


def build_prompt(query: str, context: str) -> str:
    return f"""Answer the Vietnamese question using only the evidence below. If the evidence is insufficient, say so explicitly. Do not invent facts. Return only the answer in Vietnamese.\n\nQuestion: {query}\n\nEvidence:\n{context}"""


def row_value(row: dict, *names: str) -> str:
    for name in names:
        if row.get(name):
            return str(row[name]).strip()
    return ""

with open(args.testset, encoding="utf-8", newline="") as handle:
    test_rows = list(csv.DictReader(handle))
if len(test_rows) != 50:
    raise ValueError(f"Frozen testset must contain exactly 50 rows, found {len(test_rows)}")
required_categories = {"factual", "temporal", "relational", "multi_hop", "historical"}
counts = {category: sum(row.get("category", "").strip() == category for row in test_rows) for category in required_categories}
if counts != {category: 10 for category in required_categories}:
    raise ValueError(f"Frozen testset must contain 10 rows per category, found {counts}")

# Load one model and one instance of each retriever to preserve identical LLM settings.
answer_llm = ChatGoogleGenerativeAI(model=args.model, temperature=0.0, google_api_key=api_key)
retrievers = {
    "vector_rag": VectorRAG(k=args.k),
    "temporal_vector_rag": TemporalVectorRAG(k=args.k),
    "graph_rag": GraphRAG(k=args.k),
    "temporal_graph_rag": TemporalGraphRAG(k=args.k),
}

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
completed: set[tuple[str, str]] = set()
if args.resume and out_path.exists():
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            completed.add((record.get("system", ""), record.get("qid", "")))
else:
    out_path.write_text("", encoding="utf-8")

manifest = {
    "git_commit": git_commit(),
    "testset": str(Path(args.testset).resolve()),
    "testset_sha256": hashlib.sha256(Path(args.testset).read_bytes()).hexdigest(),
    "model": args.model,
    "temperature": 0.0,
    "top_k": args.k,
    "n_queries": len(test_rows),
    "category_counts": counts,
    "started_at_utc": datetime.utcnow().isoformat() + "Z",
    "packages": {name: model_version(name) for name in ["ragas", "langchain-google-genai", "sentence-transformers", "lancedb"]},
}
(out_path.parent / "config_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

for row in test_rows:
    qid = row.get("qid", "").strip()
    query = row_value(row, "user_input", "question")
    tq = normalize_tq(row.get("tq"))
    gold_doc_ids = [x.strip() for x in row.get("gold_doc_ids", "").split("|") if x.strip()]
    if not qid or not query or not gold_doc_ids:
        raise ValueError(f"Invalid frozen row {qid}: qid/query/gold_doc_ids are required")
    for system_name, retriever in retrievers.items():
        if (system_name, qid) in completed:
            continue
        print(f"{system_name}: {qid}")
        result = retriever.retrieve(query, tq)
        context = "\n\n".join(f"[{i+1}] {ev.filename}\n{ev.content}" for i, ev in enumerate(result.evidence))
        started = time.perf_counter()
        answer = answer_text(answer_llm.invoke(build_prompt(query, context)))
        generation_latency = time.perf_counter() - started
        record = {
            "system": system_name,
            "qid": qid,
            "category": row.get("category", "").strip(),
            "user_input": query,
            "tq": tq,
            "gold_doc_ids": gold_doc_ids,
            "reference_answer": row.get("reference_answer", ""),
            "retrieved_doc_ids": result.document_ids,
            "retrieved_tu_ids": [ev.text_unit_id for ev in result.evidence],
            "evidence": [{"filename": ev.filename, "text_unit_id": ev.text_unit_id, "content": ev.content, "score": ev.score} for ev in result.evidence],
            "context": context,
            "answer": answer,
            "latency_s": result.latency_s,
            "generation_latency_s": generation_latency,
            "ctx_tokens": result.ctx_tokens,
            "answer_accuracy": None,
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed.add((system_name, qid))

print(f"Completed {len(completed)} records -> {out_path}")
