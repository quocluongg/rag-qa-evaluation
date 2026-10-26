"""
gen_testset.py — legacy RAGAS TestsetGenerator with Gemini 3.5 Flash-Lite
Confirmed working config from B0 tests (2026-08-24):
  - ragas==0.4.3 + langchain-google-genai==4.3.5
  - LLM: gemini-3.5-flash-lite via Google AI Studio (legacy path; use gen_testset_direct.py for the 90-row protocol)
  - Embeddings: BAAI/bge-m3 local (ragas.embeddings.HuggingFaceEmbeddings)
  - with_structured_output: works natively
  - RunConfig(max_workers=1): required for free-tier (16k tokens/min quota)

Input: output/text_units.parquet (376 pre-chunked units, ~1200 tokens each)
       joined with output/documents.parquet for title (filename) lookup
       and output/metadata.json for valid_from / valid_to
"""
import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path

# Fix Windows cp1252 console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
DATA_DIR   = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TU_PARQUET   = OUTPUT_DIR / "text_units.parquet"
DOCS_PARQUET = OUTPUT_DIR / "documents.parquet"
META_JSON    = OUTPUT_DIR / "metadata.json"

# ── Config ────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY", "")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL     = "BAAI/bge-m3"
TESTSET_SIZE    = 90    # oversample 50% → human review → 50 frozen (10×5 categories)
SEED            = 42

PERSONAS_DEF = [
    ("Sinh vien PTIT",
     "Can biet quy che hoc tap, dieu kien tot nghiep, chuong trinh dao tao"),
    ("Phu huynh",
     "Quan tam hoc phi, hoc bong, phuong thuc xet tuyen, yeu cau dau vao"),
    ("Can bo CTSV",
     "Tra cuu van ban hanh chinh, quyet dinh, quy dinh cu the cua hoc vien"),
]

# ── Args ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate RAGAS testset from pre-chunked text_units.parquet"
)
parser.add_argument("--pilot", action="store_true",
                    help="Pilot: use first 20 text_units (2 docs), generate 5 samples")
parser.add_argument("--n-units", type=int, default=None,
                    help="Limit number of text_units to load (default: all 376)")
parser.add_argument("--size", type=int, default=TESTSET_SIZE,
                    help=f"Samples to generate (default: {TESTSET_SIZE})")
parser.add_argument("--out", type=str, default=str(DATA_DIR / "testset_raw.jsonl"),
                    help="Output JSONL path")
parser.add_argument("--batch-size", type=int, default=10,
                    help="Generate and persist this many samples per checkpoint batch")
parser.add_argument("--resume", action="store_true",
                    help="Resume from existing JSONL output instead of overwriting it")
args = parser.parse_args()

if args.pilot:
    args.n_units = args.n_units or 20
    args.size    = 5
    args.out     = str(DATA_DIR / "testset_pilot.jsonl")
    print("[PILOT MODE] 20 text_units, 5 samples")

if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not set.\n"
        "  Windows : $env:GOOGLE_API_KEY = '<key>'\n"
        "  Linux   : export GOOGLE_API_KEY='<key>'\n"
        "  Get key : https://aistudio.google.com/apikey"
    )

# ── Load parquet index ────────────────────────────────────────
print("Loading GraphRAG index...")
tu_df   = pd.read_parquet(TU_PARQUET)
docs_df = pd.read_parquet(DOCS_PARQUET)
meta    = json.load(open(META_JSON, encoding="utf-8"))

# Build document_hash → filename mapping from documents.parquet
# documents.parquet: id (hash), title (filename like CTDT_AIoT_2025.md)
hash_to_filename = dict(zip(docs_df["id"], docs_df["title"]))
print(f"  text_units : {len(tu_df)} rows")
print(f"  documents  : {len(docs_df)} rows")
print(f"  metadata   : {len(meta)} entries")

# ── Build LangChain Documents from text_units ─────────────────
print("\nBuilding LangChain Documents from text_units.parquet...")

if args.n_units:
    tu_slice = tu_df.iloc[: args.n_units]
else:
    tu_slice = tu_df

# Deduplicate: RAGAS works on document-level KG; group text_units per source doc
# but keep individual text_units as pages so RAGAS can build relationships between them.
print(f"Loading dependencies (bge-m3 may take ~30s on first run)...")
from langchain_core.documents import Document

docs = []
skipped = 0
for _, row in tu_slice.iterrows():
    doc_hash = row["document_id"]
    filename = hash_to_filename.get(doc_hash)
    if not filename:
        skipped += 1
        continue
    doc_meta_entry = meta.get(filename, {})
    entity_ids = row["entity_ids"]
    rel_ids    = row["relationship_ids"]
    docs.append(Document(
        page_content=row["text"],
        metadata={
            "text_unit_id": row["id"],
            "document_id":  doc_hash,
            "filename":     filename,
            "doc_id":       doc_meta_entry.get("doc_id", filename.replace(".md", "")),
            "source":       filename,
            "valid_from":   doc_meta_entry.get("valid_from", ""),
            "valid_to":     doc_meta_entry.get("valid_to", ""),
            "n_tokens":     int(row["n_tokens"]),
            "entity_ids":   entity_ids.tolist() if hasattr(entity_ids, "tolist") else list(entity_ids or []),
            "relationship_ids": rel_ids.tolist() if hasattr(rel_ids, "tolist") else list(rel_ids or []),
        }
    ))

total_chars = sum(len(d.page_content) for d in docs)
unique_docs = len(set(d.metadata["filename"] for d in docs))
print(f"  Loaded {len(docs)} text_units from {unique_docs} source documents")
print(f"  Total chars: {total_chars:,}  |  Skipped (no filename): {skipped}")

# ── Compatibility shim for legacy vertexai import in ragas 0.4.3 ───────────
import sys
import types

try:
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa
except ImportError:
    vertex_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    vertex_mod.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = vertex_mod

    import langchain_community.llms as community_llms
    if not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = type("VertexAI", (object,), {})

# ── Init LLM & Embeddings ─────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

print(f"\nInitialising LLM: {GENERATOR_MODEL}")
_llm = ChatGoogleGenerativeAI(
    model=GENERATOR_MODEL,
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY,
)
generator_llm = LangchainLLMWrapper(_llm)

print(f"Initialising embeddings: {EMBED_MODEL} (local)")
try:
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmb
    generator_embeddings = RagasHFEmb(model=EMBED_MODEL)
    print("  Using ragas.embeddings.HuggingFaceEmbeddings")
except (ImportError, AttributeError):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    _emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    generator_embeddings = LangchainEmbeddingsWrapper(_emb)
    print("  Fallback: LangchainEmbeddingsWrapper")

# ── Personas ──────────────────────────────────────────────────
from ragas.testset.persona import Persona
personas = [Persona(name=n, role_description=r) for n, r in PERSONAS_DEF]
print(f"Personas: {[p.name for p in personas]}")

# ── Query distribution ────────────────────────────────────────
# RAGAS 0.4.x: pass list of (SynthesizerInstance, weight) tuples
# TEMPORAL and HISTORICAL are NOT synthesized by RAGAS — added by augment_temporal.py
# So RAGAS covers: FACTUAL (SingleHop), MULTI-HOP (MultiHopSpecific), RELATIONAL (MultiHopAbstract)
try:
    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer,
    )
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.40),  # → FACTUAL
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm),  0.30),  # → MULTI-HOP
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm),  0.30),  # → RELATIONAL
    ]
    print("Query distribution: SingleHop=0.40, MultiHopSpecific=0.30, MultiHopAbstract=0.30")
    use_distribution = True
except ImportError as e:
    print(f"[WARN] Synthesizer import failed ({e}); using default distribution")
    query_distribution = None
    use_distribution = False

# ── TestsetGenerator ──────────────────────────────────────────
print(f"\nInitialising TestsetGenerator...")
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    persona_list=personas,          # ← correctly passed
)

# RunConfig: serial execution avoids free-tier bursts; checkpointing limits lost work.
run_cfg = RunConfig(
    max_workers=1,
    timeout=180,
    max_retries=5,
)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
existing_records: list[dict] = []
if args.resume and out_path.exists():
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            existing_records.append(json.loads(line))
    print(f"Resuming from {len(existing_records)} existing records: {out_path}")
else:
    out_path.write_text("", encoding="utf-8")

if args.batch_size <= 0:
    raise ValueError("--batch-size must be positive")
if len(existing_records) > args.size:
    raise ValueError(f"Existing records ({len(existing_records)}) exceed requested size ({args.size})")

# ── Generation and JSONL checkpointing ─────────────────────────
def _to_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return list(value)
    return value


def _record_from_row(i: int, row) -> dict:
    ref_ctx = _to_jsonable(row.get("reference_contexts", []))
    if ref_ctx is None:
        ref_ctx = []
    return {
        "qid": f"q{i:03d}",
        "category": "tbd",
        "tq": None,
        "gold_doc_ids": [],
        "gold_tu_ids": [],
        "reviewer_status": "pending",
        "reviewer_note": "",
        **{k: _to_jsonable(v) for k, v in row.items()},
    }


def _append_records(records: list[dict]) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_generation_kwargs(batch_size: int) -> dict:
    kwargs = dict(
        documents=docs,
        testset_size=batch_size,
        run_config=run_cfg,
    )
    if use_distribution:
        kwargs["query_distribution"] = query_distribution
    return kwargs

records = list(existing_records)
remaining = args.size - len(records)
t0 = time.time()
print(f"Generating {remaining} samples with model {GENERATOR_MODEL}...")
print(f"  batch_size={args.batch_size}, max_workers=1, temperature=0.0")

while remaining > 0:
    batch_target = min(args.batch_size, remaining)
    print(f"\nGenerating checkpoint batch: {batch_target} samples ({len(records)}/{args.size} complete)")
    try:
        testset = generator.generate_with_langchain_docs(**_build_generation_kwargs(batch_target))
        df = testset.to_pandas()
        batch_records = [_record_from_row(len(records) + i, row) for i, (_, row) in enumerate(df.iterrows())]
        if not batch_records:
            raise RuntimeError("RAGAS returned zero records; refusing to checkpoint an empty batch")
        _append_records(batch_records)
        records.extend(batch_records)
        remaining = args.size - len(records)
        print(f"Checkpoint saved: {len(records)}/{args.size} records -> {out_path}")
    except Exception:
        print(f"Generation interrupted. Existing checkpoint is preserved at {out_path}")
        raise

elapsed = time.time() - t0
print(f"\nGeneration complete in {elapsed/60:.1f} min")
print(f"Generated {len(records)} samples. Output: {out_path}")

# ── Preview ───────────────────────────────────────────────────
print("\n--- Sample preview (first 5) ---")
syn_counts: dict = {}
for r in records[:5]:
    q = r.get("user_input", r.get("question", "N/A"))
    syn = r.get("synthesizer_name", "?")
    syn_counts[syn] = syn_counts.get(syn, 0) + 1
    print(f"  [{r['qid']}] [{syn}]")
    print(f"         {q[:100]}")

print("\n--- Synthesizer distribution ---")
all_syns: dict = {}
for r in records:
    name = r.get("synthesizer_name", "unknown")
    all_syns[name] = all_syns.get(name, 0) + 1
for name, count in sorted(all_syns.items()):
    print(f"  {name}: {count}")

if not any("abstract" in str(r.get("synthesizer_name", "")).lower() for r in records):
    print("\n[WARN] multi_hop_abstract synthesizer was skipped by RAGAS.")

print("\nNext steps:")
print("  1. Run: python src/verify_testset.py export")
print("  2. Review category, tq, gold_doc_ids, and reviewer_status")
print("  3. Run: python src/augment_temporal.py --size 20 --out data/testset_augmented.jsonl")
print("  4. Run: python src/verify_testset.py add-augmented")
print("  5. Run: python src/verify_testset.py freeze  # requires exactly 50 approved rows")
