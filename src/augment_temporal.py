"""Generate temporal and historical candidate queries for human review.

The output is intentionally a candidate pool, not ground truth. Every row must be
reviewed before `verify_testset.py freeze` accepts it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite")

parser = argparse.ArgumentParser(description="Generate temporal/historical query candidates")
parser.add_argument("--size", type=int, default=20, help="Total candidates: 10 temporal + 10 historical by default")
parser.add_argument("--temporal", type=int, default=10)
parser.add_argument("--historical", type=int, default=10)
parser.add_argument("--out", default=str(DATA_DIR / "testset_augmented.jsonl"))
parser.add_argument("--resume", action="store_true")
parser.add_argument("--sleep", type=float, default=1.0, help="Delay between requests in seconds")
args = parser.parse_args()

if args.size != args.temporal + args.historical:
    raise ValueError("--size must equal --temporal + --historical")
if args.size <= 0:
    raise ValueError("--size must be positive")
api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY is required")

from google import genai
from google.genai import types


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%d/%m/%Y")
    except ValueError:
        return None


def find_replacement_pairs(meta: dict) -> list[tuple[str, str]]:
    """Return (old_filename, new_filename) pairs from explicit metadata links."""
    pairs: set[tuple[str, str]] = set()
    names = list(meta)
    for new_name, entry in meta.items():
        replacement = str(entry.get("replaces", ""))
        for old_name in names:
            old_stem = Path(old_name).stem
            if old_name == new_name:
                continue
            if old_stem and (old_name in replacement or old_stem in replacement):
                pairs.add((old_name, new_name))
        replaced_by = str(entry.get("replaced_by", ""))
        for newer_name in names:
            newer_stem = Path(newer_name).stem
            if newer_name == new_name:
                continue
            if newer_name in replaced_by or newer_stem in replaced_by:
                pairs.add((new_name, newer_name))

    # Fallback for simple year-versioned filenames whose explicit metadata omits
    # a link. Keep this deterministic and only pair adjacent years.
    by_series: dict[str, list[tuple[int, str]]] = {}
    for name in names:
        match = re.match(r"^(.*?)[_-](20\d{2})\.md$", name)
        if match:
            by_series.setdefault(match.group(1), []).append((int(match.group(2)), name))
    for versions in by_series.values():
        versions.sort()
        for (_, old_name), (_, new_name) in zip(versions, versions[1:]):
            pairs.add((old_name, new_name))
    return sorted(pairs)


meta = json.loads((OUTPUT_DIR / "metadata.json").read_text(encoding="utf-8"))
docs_df = pd.read_parquet(OUTPUT_DIR / "documents.parquet")
doc_text = dict(zip(docs_df["title"], docs_df["text"]))
pairs = find_replacement_pairs(meta)
if not pairs:
    raise RuntimeError("No replacement pairs found in output/metadata.json")
print(f"Found {len(pairs)} version pairs")

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
records: list[dict] = []
if args.resume and out_path.exists():
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) > args.size:
        raise ValueError("Existing augmented records exceed requested size")
else:
    out_path.write_text("", encoding="utf-8")

client = genai.Client(api_key=api_key)
schema = {
    "type": "OBJECT",
    "properties": {
        "user_input": {"type": "STRING"},
        "reference_answer": {"type": "STRING"},
    },
    "required": ["user_input", "reference_answer"],
}


def generate_candidate(category: str, old_name: str, new_name: str) -> dict:
    old_meta = meta[old_name]
    new_meta = meta[new_name]
    old_text = str(doc_text.get(old_name, ""))[:7000]
    new_text = str(doc_text.get(new_name, ""))[:7000]
    old_date = old_meta.get("valid_from", "")
    new_date = new_meta.get("valid_from", "")
    if category == "temporal":
        tq = new_date or old_date
        instruction = (
            "Tạo một câu hỏi tiếng Việt hỏi quy định/thông tin có hiệu lực tại thời điểm tq. "
            "Câu hỏi phải trả lời được chỉ từ phiên bản đang có hiệu lực tại tq, không hỏi so sánh."
        )
        gold = [new_name] if new_date else [old_name]
    else:
        tq = new_date or old_date
        instruction = (
            "Tạo một câu hỏi tiếng Việt yêu cầu so sánh thay đổi giữa hai phiên bản. "
            "Câu trả lời phải nêu các thay đổi có bằng chứng trong hai văn bản, không bịa thêm."
        )
        gold = [old_name, new_name]
    prompt = f"""Bạn là người xây dựng bộ kiểm thử RAG. Chỉ trả về JSON theo schema.
Loại câu hỏi: {category}
Thời điểm tham chiếu tq: {tq}
Phiên bản cũ: {old_name} (hiệu lực từ {old_date})
Phiên bản mới: {new_name} (hiệu lực từ {new_date})
{instruction}

Nội dung phiên bản cũ:
{old_text}

Nội dung phiên bản mới:
{new_text}
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    payload = json.loads(response.text)
    return {
        "category": category,
        "tq": tq,
        "gold_doc_ids": gold,
        "gold_tu_ids": [],
        "user_input": payload["user_input"].strip(),
        "reference_answer": payload["reference_answer"].strip(),
        "reference_contexts": [old_text, new_text],
        "synthesizer_name": MODEL,
        "reviewer_status": "pending",
        "reviewer_note": "LLM candidate; verify category, tq, answer and gold_doc_ids manually.",
        "source_docs": [old_name, new_name],
    }

# Alternate pair selection across the available chains so one chain does not
# dominate the candidate pool. Duplicate semantic candidates still require review.
requested = [("temporal", i) for i in range(args.temporal)] + [("historical", i) for i in range(args.historical)]
for index in range(len(records), len(requested)):
    category, _ = requested[index]
    old_name, new_name = pairs[index % len(pairs)]
    qid = f"aug_{'t' if category == 'temporal' else 'h'}_{index:03d}"
    print(f"Generating {qid}: {category} ({old_name} -> {new_name})")
    try:
        record = generate_candidate(category, old_name, new_name)
        record["qid"] = qid
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        print(f"Checkpoint saved {len(records)}/{len(requested)}")
    except Exception:
        print(f"Generation interrupted; checkpoint preserved at {out_path}")
        raise
    if index + 1 < len(requested):
        time.sleep(max(0.0, args.sleep))

print(f"Generated {len(records)} candidates -> {out_path}")
