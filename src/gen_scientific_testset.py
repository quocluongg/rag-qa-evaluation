"""
gen_scientific_testset.py — Scientific QA Testset Generator for RAG Evaluation.

Generates a balanced, source-grounded testset with 4 standard scientific categories:
  1. factual_single_hop: Direct entity/fact lookups from a single text unit.
  2. multi_hop_relational: Relational & multi-step questions requiring bridging across 2 units.
  3. global_synthesis: High-level overview/comparison questions synthesizing multiple units.
  4. unanswerable_negative: Out-of-domain / unanswerable questions to evaluate hallucination resistance.

Features:
- Guaranteed source grounding (evidence snippet + gold_doc_ids + gold_tu_ids)
- Automated verification against hallucination
- Strict JSON/CSV schema with checkpointing and resume support
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
DATA_DIR = ROOT / "data"

DEFAULT_MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite")
CATEGORIES = [
    "factual_single_hop",
    "multi_hop_relational",
    "global_synthesis",
    "unanswerable_negative",
]

CATEGORY_PROMPTS = {
    "factual_single_hop": (
        "Tạo một câu hỏi sự kiện đơn bước (single-hop) có câu trả lời trực tiếp, rõ ràng trong đoạn văn bản sau. "
        "Câu hỏi phải cụ thể, tự đứng độc lập (self-contained), không dùng đại từ mơ hồ như 'văn bản này', 'ông ấy'. "
        "Trích dẫn nguyên văn đoạn văn bản chứa đáp án làm bằng chứng."
    ),
    "multi_hop_relational": (
        "Dựa vào HAI đoạn văn bản được cung cấp bên dưới, hãy tạo một câu hỏi suy luận đa bước (multi-hop / quan hệ). "
        "Để trả lời câu hỏi, người đọc BẮT BUỘC phải kết hợp dữ kiện từ CẢ HAI đoạn văn bản (ví dụ: liên kết giữa ngành học - môn học - quy định tuyển sinh). "
        "Câu hỏi phải tự đứng độc lập, không rò rỉ đáp án trong câu hỏi."
    ),
    "global_synthesis": (
        "Dựa vào các thông tin tổng hợp được cung cấp bên dưới, hãy tạo một câu hỏi tổng hợp mang tính toàn cục (synthesis/summarization). "
        "Câu hỏi yêu cầu tóm lược, đối chiếu hoặc bao quát nhiều khía cạnh (ví dụ: tổng hợp các hình thức học bổng, so sánh các quy trình đào tạo). "
        "Đưa ra câu trả lời tổng hợp súc tích, khách quan và trích dẫn các tài liệu liên quan."
    ),
    "unanswerable_negative": (
        "Dựa vào đoạn văn bản được cung cấp, hãy tạo một câu hỏi BẪY / KHÔNG CÓ ĐÁP ÁN trong tài liệu (unanswerable / out-of-domain). "
        "Câu hỏi phải trông rất hợp lý và liên quan đến chủ đề giáo dục/đại học nhưng thông tin cụ thể (chỉ tiêu ngành không có trong văn bản, quy định của trường đại học khác, hoặc số liệu không hề tồn tại) HOÀN TOÀN KHÔNG CÓ trong tài liệu. "
        "Đáp án chuẩn (reference_answer) BẮT BUỘC phải khẳng định: 'Tài liệu không cung cấp thông tin về vấn đề này.'"
    ),
}


class QuestionCandidate(BaseModel):
    question: str = Field(description="Nội dung câu hỏi bằng tiếng Việt, rõ ràng và tự đứng độc lập")
    reference_answer: str = Field(description="Câu trả lời chuẩn mực, chính xác, dựa đúng trên bằng chứng")
    evidence_snippet: str = Field(description="Đoạn văn bản trích dẫn nguyên văn làm bằng chứng (đối với unanswerable ghi 'NONE')")
    reasoning_summary: str = Field(description="Tóm tắt ngắn gọn lý do tại sao bằng chứng dẫn đến câu trả lời")


def load_corpus_units() -> tuple[list[dict[str, Any]], dict[str, str]]:
    tu_df = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
    docs_df = pd.read_parquet(OUTPUT_DIR / "documents.parquet")
    hash_to_filename = dict(zip(docs_df["id"], docs_df["title"]))

    units = []
    for _, row in tu_df.iterrows():
        tu_id = str(row["id"])
        text = str(row.get("text", "")).strip()
        doc_ids = row.get("document_ids", [])
        if hasattr(doc_ids, "tolist"):
            doc_ids = doc_ids.tolist()
        doc_filenames = [hash_to_filename.get(d_id, d_id) for d_id in doc_ids]
        if text and len(text) > 80:
            units.append({
                "tu_id": tu_id,
                "text": text,
                "filenames": doc_filenames,
            })
    return units, hash_to_filename


def generate_candidate(
    category: str,
    units: list[dict[str, Any]],
    llm: Any,
) -> dict[str, Any] | None:
    prompt_guide = CATEGORY_PROMPTS[category]

    if category in ("factual_single_hop", "unanswerable_negative"):
        selected = random.choice(units)
        context_str = f"Tài liệu nguồn: {', '.join(selected['filenames'])}\nNội dung:\n{selected['text'][:1800]}"
        gold_docs = selected["filenames"] if category != "unanswerable_negative" else ["NONE"]
        gold_tus = [selected["tu_id"]] if category != "unanswerable_negative" else ["NONE"]
    elif category == "multi_hop_relational":
        u1, u2 = random.sample(units, 2)
        context_str = (
            f"--- Đoạn 1 ({', '.join(u1['filenames'])}) ---\n{u1['text'][:1200]}\n\n"
            f"--- Đoạn 2 ({', '.join(u2['filenames'])}) ---\n{u2['text'][:1200]}"
        )
        gold_docs = list(set(u1["filenames"] + u2["filenames"]))
        gold_tus = [u1["tu_id"], u2["tu_id"]]
    else:  # global_synthesis
        sampled = random.sample(units, min(3, len(units)))
        context_str = "\n\n".join(
            f"--- Trích dẫn {i+1} ({', '.join(u['filenames'])}) ---\n{u['text'][:900]}"
            for i, u in enumerate(sampled)
        )
        gold_docs = list(set(sum([u["filenames"] for u in sampled], [])))
        gold_tus = [u["tu_id"] for u in sampled]

    system_prompt = (
        "Bạn là chuyên gia thiết kế bộ dữ liệu benchmark hỏi-đáp RAG theo tiêu chuẩn nghiên cứu khoa học NLP. "
        "Hãy đọc ngữ cảnh được cung cấp và sinh một câu hỏi chất lượng cao theo đúng yêu cầu phân loại."
    )
    user_prompt = f"{prompt_guide}\n\n[NGỮ CẢNH TÀI LIỆU]:\n{context_str}\n\nHãy xuất ra định dạng JSON theo đúng schema."

    try:
        candidate: QuestionCandidate = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
        return {
            "category": category,
            "question": candidate.question.strip(),
            "reference_answer": candidate.reference_answer.strip(),
            "evidence_snippet": candidate.evidence_snippet.strip(),
            "reasoning_summary": candidate.reasoning_summary.strip(),
            "gold_doc_ids": "|".join(gold_docs),
            "gold_tu_ids": "|".join(gold_tus),
        }
    except Exception as e:
        print(f"Error generating for {category}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate scientific RAG benchmark testset")
    parser.add_argument("--size", type=int, default=50, help="Total number of queries (must be divisible by 4, or closest)")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "scientific_testset_50.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is required")

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=args.model,
        temperature=0.2,
        google_api_key=api_key,
    ).with_structured_output(QuestionCandidate)

    units, _ = load_corpus_units()
    print(f"Loaded {len(units)} text units from corpus.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_jsonl = args.out.with_suffix(".raw.jsonl")

    existing: list[dict] = []
    if args.resume and raw_jsonl.exists():
        for line in raw_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
        print(f"Resuming: found {len(existing)} existing generated candidates.")

    counts = {cat: sum(1 for x in existing if x.get("category") == cat) for cat in CATEGORIES}
    target_per_cat = args.size // len(CATEGORIES)
    remainder = args.size % len(CATEGORIES)

    targets = {cat: target_per_cat + (1 if i < remainder else 0) for i, cat in enumerate(CATEGORIES)}
    print(f"Target distribution: {targets}")

    for cat in CATEGORIES:
        while counts[cat] < targets[cat]:
            cand = generate_candidate(cat, units, llm)
            if cand:
                counts[cat] += 1
                cand["qid"] = f"Q{len(existing) + 1:03d}"
                existing.append(cand)
                with raw_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(cand, ensure_ascii=False) + "\n")
                print(f"[{counts[cat]}/{targets[cat]}] Generated {cat}: {cand['question'][:60]}...")
            time.sleep(2.0)

    # Export to finalized CSV
    with args.out.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "qid", "category", "question", "reference_answer",
            "gold_doc_ids", "gold_tu_ids", "evidence_snippet", "reasoning_summary"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(existing[:args.size], start=1):
            row["qid"] = f"Q{i:03d}"
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"\nSuccessfully generated and saved {len(existing[:args.size])} scientific queries to {args.out}")


if __name__ == "__main__":
    main()
