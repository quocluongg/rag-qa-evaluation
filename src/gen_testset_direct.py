"""Source-grounded direct-prompt testset generator.

Generates a 90-row candidate pool: 18 factual, 18 temporal, 18 relational,
18 multi-hop, and 18 historical. Each candidate normally uses one API call;
transient failures may retry up to five total attempts per question. The hard
budget is size * max_attempts (450 for the default 90 x 5 configuration).

The model only writes the question, answer, and temporal query. Source IDs are
assigned by this program from the context units supplied to the model; they are
never invented by the model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
DATA_DIR = ROOT / "data"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_SIZE = 90
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_MAX_TOTAL_REQUESTS = 450
CATEGORIES = ("factual", "temporal", "relational", "multi-hop", "historical")
CATEGORY_DESCRIPTIONS = {
    "factual": "Ask for one directly stated fact from one source unit.",
    "temporal": "Ask about validity, timing, effective date, or a dated version. Only use dates explicitly present in the context.",
    "relational": "Ask about an explicit relationship between two entities, courses, outcomes, offices, or concepts.",
    "multi-hop": "Require combining two supplied context units; the answer must use both and not add outside facts.",
    "historical": "Ask how a dated policy, program, decision, or version is described in the supplied historical context. Do not invent chronology.",
}
SCHEMA = {
    "type": "object",
    "properties": {
        "user_input": {"type": "string"},
        "reference": {"type": "string"},
        "tq": {"type": ["string", "null"]},
    },
    "required": ["user_input", "reference", "tq"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate source-grounded direct-prompt JSONL candidates")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--max-total-requests", type=int, default=DEFAULT_MAX_TOTAL_REQUESTS)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "testset_raw.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-units", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.environ.get("GENERATOR_MODEL", DEFAULT_MODEL))
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def load_units(n_units: int | None) -> list[dict[str, Any]]:
    tu_path = OUTPUT_DIR / "text_units.parquet"
    docs_path = OUTPUT_DIR / "documents.parquet"
    meta_path = OUTPUT_DIR / "metadata.json"
    tu_df = pd.read_parquet(tu_path)
    docs_df = pd.read_parquet(docs_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hash_to_filename = dict(zip(docs_df["id"], docs_df["title"]))
    if n_units:
        tu_df = tu_df.iloc[:n_units]

    units: list[dict[str, Any]] = []
    for _, row in tu_df.iterrows():
        doc_hash = row["document_id"]
        filename = hash_to_filename.get(doc_hash)
        if not filename:
            continue
        entry = meta.get(filename, {})
        units.append({
            "text_unit_id": str(row["id"]),
            "document_id": str(doc_hash),
            "filename": str(filename),
            "doc_id": str(entry.get("doc_id", str(filename).replace(".md", ""))),
            "valid_from": str(entry.get("valid_from", "") or ""),
            "valid_to": str(entry.get("valid_to", "") or ""),
            "text": str(row["text"]),
        })
    if not units:
        raise RuntimeError("No usable text units found in output/*.parquet")
    return units


def build_context(category: str, index: int, units: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    dated = [u for u in units if u["valid_from"] or u["valid_to"]]
    if category in {"temporal", "historical"} and dated:
        pool = dated
    else:
        pool = units

    first = pool[(index * 7 + rng.randrange(len(pool))) % len(pool)]
    selected = [first]
    if category in {"relational", "multi-hop"}:
        candidates = [u for u in pool if u["text_unit_id"] != first["text_unit_id"]]
        if category == "multi-hop":
            candidates = [u for u in candidates if u["filename"] != first["filename"]] or candidates
        if candidates:
            selected.append(candidates[(index * 11 + rng.randrange(len(candidates))) % len(candidates)])
    return selected


def prompt_for(category: str, contexts: list[dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        f"[SOURCE {i}] filename={u['filename']} text_unit_id={u['text_unit_id']}\n{u['text'][:7000]}"
        for i, u in enumerate(contexts, start=1)
    )
    return f"""You are constructing one Vietnamese RAG benchmark item.
Category: {category}
Category rule: {CATEGORY_DESCRIPTIONS[category]}

Use ONLY the supplied source context. Do not use outside knowledge. Write a natural Vietnamese user question and a concise Vietnamese reference answer that is fully supported by the context. Do not mention that you are an AI. For temporal or historical questions, set tq to a date or validity phrase only when it is explicitly supported; otherwise set tq to null. Do not invent document IDs: source IDs are assigned by the program.

Return only JSON matching this schema:
{{
  "user_input": "Vietnamese question",
  "reference": "Vietnamese answer",
  "tq": "explicit date/time query or null"
}}

SUPPLIED SOURCE CONTEXT:
{context_text}
"""


def call_model(client: genai.Client, model: str, prompt: str) -> dict[str, Any]:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=SCHEMA,
            max_output_tokens=700,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Model returned empty text")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model JSON is not an object")
    for key in ("user_input", "reference", "tq"):
        if key not in value:
            raise ValueError(f"Model JSON missing {key}")
    if not str(value["user_input"]).strip() or not str(value["reference"]).strip():
        raise ValueError("Question or reference answer is empty")
    if value["tq"] is not None:
        value["tq"] = str(value["tq"])
    return value


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.batch_size <= 0 or args.max_attempts <= 0:
        raise ValueError("size, batch-size, and max-attempts must be positive")
    if args.batch_size != 10:
        raise ValueError("This benchmark protocol requires --batch-size 10")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set")

    max_requests = args.max_total_requests or args.size * args.max_attempts
    if max_requests > args.size * args.max_attempts:
        raise ValueError("max-total-requests cannot exceed size * max-attempts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = read_records(args.out) if args.resume else []
    if not args.resume:
        args.out.write_text("", encoding="utf-8")
    if len(records) > args.size:
        raise ValueError("Existing checkpoint has more records than requested size")

    usage_path = args.out.with_suffix(".usage.json")
    usage = {"requests": 0, "model": args.model, "max_requests": max_requests}
    if args.resume and usage_path.exists():
        usage.update(json.loads(usage_path.read_text(encoding="utf-8")))
    requests_used = int(usage.get("requests", 0))
    if requests_used > max_requests:
        raise RuntimeError("Existing request usage already exceeds hard budget")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Missing google-genai; run pip install -r requirements.txt") from exc
    from google.genai import types  # noqa: PLC0415

    units = load_units(args.n_units)
    rng = random.Random(args.seed)
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    category_plan = [category for category in CATEGORIES for _ in range(args.size // len(CATEGORIES))]
    category_plan += list(CATEGORIES[: args.size % len(CATEGORIES)])
    print(f"Direct generation: {args.size} records, model={args.model}, hard budget={max_requests} requests")
    print(f"Category plan: {dict((c, category_plan.count(c)) for c in CATEGORIES)}")

    while len(records) < args.size:
        batch_end = min(len(records) + args.batch_size, args.size)
        print(f"Batch {len(records) // args.batch_size + 1}: records {len(records)}..{batch_end - 1}")
        while len(records) < batch_end:
            index = len(records)
            category = category_plan[index]
            contexts = build_context(category, index, units, rng)
            prompt = prompt_for(category, contexts)
            result: dict[str, Any] | None = None
            last_error = ""
            attempts = 0
            for attempt in range(1, args.max_attempts + 1):
                if requests_used >= max_requests:
                    raise RuntimeError(f"Hard request budget exhausted at {requests_used}/{max_requests}")
                attempts = attempt
                requests_used += 1
                usage["requests"] = requests_used
                usage_path.write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
                try:
                    result = call_model(client, args.model, prompt)
                    break
                except Exception as exc:  # noqa: BLE001 - preserve checkpoint and retry boundedly
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.max_attempts:
                        time.sleep(min(2 ** (attempt - 1), 8))
            if result is None:
                raise RuntimeError(f"Failed q{index:03d} after {attempts} attempts: {last_error}")

            record = {
                "qid": f"q{index:03d}",
                "category": category,
                "tq": result.get("tq"),
                "gold_doc_ids": sorted({u["doc_id"] for u in contexts}),
                "gold_tu_ids": [u["text_unit_id"] for u in contexts],
                "reviewer_status": "pending",
                "reviewer_note": "",
                "user_input": result["user_input"].strip(),
                "reference": result["reference"].strip(),
                "reference_contexts": [u["text"] for u in contexts],
                "generator_model": args.model,
                "generator_attempts": attempts,
                "source_filenames": [u["filename"] for u in contexts],
            }
            append_record(args.out, record)
            records.append(record)
            time.sleep(max(args.sleep, 0.0))
        print(f"Checkpoint saved: {len(records)}/{args.size}; requests={requests_used}/{max_requests}")

    print(f"Complete: {len(records)} records; requests={requests_used}/{max_requests}; output={args.out}")


if __name__ == "__main__":
    main()
