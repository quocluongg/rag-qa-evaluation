"""
B0 Test Script - 3 checks before running full pipeline
  1. Test Gemini API basic connection
  2. Test structured output (with_structured_output) with Gemini
  3. Test RAGAS TestsetGenerator compat with langchain-google-genai
"""
import os, sys, json, traceback
from pathlib import Path

# Fix Windows cp1252 console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not set. Get key at https://aistudio.google.com/apikey\n"
        "  Windows: $env:GOOGLE_API_KEY = '<key>'\n"
        "  Linux/Mac: export GOOGLE_API_KEY='<key>'"
    )
MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = {}

# ─────────────────────────────────────────────────────────────
# TEST 1 — Gemini API basic invoke
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1: Gemini API basic invoke")
print("="*60)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    resp = llm.invoke("1 + 1 = ? Answer in one word.")
    content = resp.content
    print(f"Model   : {MODEL}")
    print(f"Response: {repr(content[:200])}")
    results["test1_model"] = MODEL
    results["test1_status"] = PASS
    print(f"\n{PASS} - Model '{MODEL}' OK.")
except Exception as e:
    msg = str(e)
    print(f"Error with '{MODEL}': {msg[:300]}")
    # try fallback
    try:
        print(f"\nTrying fallback model: {FALLBACK_MODEL}")
        llm2 = ChatGoogleGenerativeAI(
            model=FALLBACK_MODEL,
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )
        resp2 = llm2.invoke("1 + 1 = ? Answer in one word.")
        print(f"Fallback response: {repr(resp2.content[:200])}")
        results["test1_model"] = FALLBACK_MODEL
        results["test1_status"] = WARN + f" (using fallback {FALLBACK_MODEL})"
        print(f"\n{WARN} - '{MODEL}' unavailable, using '{FALLBACK_MODEL}'.")
    except Exception as e2:
        results["test1_model"] = None
        results["test1_status"] = FAIL + f": {str(e2)[:200]}"
        print(f"\n{FAIL} - Both models failed: {str(e2)[:300]}")

active_model = results.get("test1_model") or FALLBACK_MODEL

# ─────────────────────────────────────────────────────────────
# TEST 2 — Structured output (with_structured_output)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: Structured output (with_structured_output / Pydantic)")
print("="*60)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel, Field

    class QueryItem(BaseModel):
        question: str = Field(description="A question about university regulations")
        category: str = Field(description="Question type: factual/temporal/relational")

    llm_struct = ChatGoogleGenerativeAI(
        model=active_model,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    structured_llm = llm_struct.with_structured_output(QueryItem)
    result_struct = structured_llm.invoke(
        "Generate a question about university academic regulations."
    )
    print(f"Parsed output:")
    print(f"  question : {repr(result_struct.question)}")
    print(f"  category : {repr(result_struct.category)}")
    results["test2_method"] = "with_structured_output"
    results["test2_status"] = PASS
    print(f"\n{PASS} - with_structured_output works.")
except Exception as e:
    print(f"with_structured_output error: {str(e)[:400]}")
    # try JsonOutputParser fallback
    try:
        print("\nTrying JsonOutputParser fallback...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        llm3 = ChatGoogleGenerativeAI(
            model=active_model,
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            "Generate a question about university regulations. "
            "Return JSON with two fields: question (string) and category (string). "
            "Return only JSON, no explanation.\n{format_instructions}"
        )
        chain = prompt | llm3 | parser
        result_json = chain.invoke({"format_instructions": parser.get_format_instructions()})
        print(f"JsonOutputParser result: {result_json}")
        results["test2_method"] = "JsonOutputParser"
        results["test2_status"] = WARN + " (need JsonOutputParser instead of with_structured_output)"
        print(f"\n{WARN} - Need to configure RAGAS to use JsonOutputParser.")
    except Exception as e2:
        results["test2_method"] = None
        results["test2_status"] = FAIL + f": {str(e2)[:200]}"
        print(f"\n{FAIL} - Both structured output methods failed: {str(e2)[:300]}")
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# TEST 3 — RAGAS TestsetGenerator compat
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3: RAGAS TestsetGenerator compat (pilot 2 docs)")
print("="*60)

# Read first 2 docs from input/
INPUT_DIR = Path(__file__).resolve().parents[1] / "input"
md_files = sorted(INPUT_DIR.glob("*.md"))[:2]
if not md_files:
    print(f"{FAIL} - No .md files found in {INPUT_DIR}")
    results["test3_status"] = FAIL + ": no .md files found"
else:
    print(f"Using {len(md_files)} docs: {[f.name for f in md_files]}")
    try:
        # Import ragas 0.4.x API
        from ragas.testset import TestsetGenerator
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.documents import Document

        print("Import RAGAS 0.4.x API: OK")

        # LLM: Gemini via Google AI Studio
        _llm = ChatGoogleGenerativeAI(
            model=active_model,
            temperature=0.0,
            google_api_key=GOOGLE_API_KEY,
        )
        generator_llm = LangchainLLMWrapper(_llm)

        # Embeddings: use local BAAI/bge-m3 (already installed, same as retrieval system)
        # This avoids Google Embeddings API v1beta compat issues
        print("Loading local BAAI/bge-m3 embeddings...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _emb_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _emb_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        generator_embeddings = LangchainEmbeddingsWrapper(_emb_model)
        print("  Embeddings: BAAI/bge-m3 (local, no API quota)")


        # Load docs — truncate to 3000 chars to stay within free-tier RPM
        MAX_CHARS = 3000
        docs = []
        for f in md_files:
            text = f.read_text(encoding="utf-8")[:MAX_CHARS]
            docs.append(Document(page_content=text, metadata={"source": f.name}))
        print(f"Loaded {len(docs)} docs (truncated to {MAX_CHARS} chars each)")
        print(f"  Total chars: {sum(len(d.page_content) for d in docs)}")

        # Init generator
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )
        print("TestsetGenerator initialized: OK")

        # RunConfig: max_workers=1 serialises LLM calls → avoids 429 rate limit
        run_cfg = RunConfig(
            max_workers=1,    # serial execution to stay within free-tier quota
            timeout=120,
            max_retries=3,
        )

        # Generate pilot 2 samples
        print("Generating 2 test samples (pilot, max_workers=1)...")
        print("  [Note: serial execution to avoid free-tier rate limit]")
        testset = generator.generate_with_langchain_docs(
            documents=docs,
            testset_size=2,
            run_config=run_cfg,
        )
        df = testset.to_pandas()
        print(f"\nGenerated {len(df)} samples:")
        for i, row in df.iterrows():
            q = str(row.get("user_input", row.get("question", "N/A")))[:120]
            print(f"  [{i}] {repr(q)}")

        results["test3_status"] = PASS
        results["test3_ragas_api"] = "ragas.testset.TestsetGenerator (0.4.x)"
        results["test3_n_samples"] = len(df)
        results["test3_note"] = "max_workers=1 required for free-tier quota"
        print(f"\n{PASS} - RAGAS TestsetGenerator works with Gemini + ragas==0.4.3")
        print(f"  NOTE: use RunConfig(max_workers=1) for free-tier API to avoid 429.")

    except ImportError as ie:
        print(f"Import error: {ie}")
        try:
            from ragas.testset.generator import TestsetGenerator  # older API
            print("WARN: using old API ragas.testset.generator")
            results["test3_status"] = WARN + " (old API: ragas.testset.generator)"
        except Exception as ie2:
            results["test3_status"] = FAIL + f": ImportError - {str(ie)[:200]}"
            print(f"\n{FAIL} - {ie}")
    except Exception as e:
        err = str(e)
        results["test3_status"] = FAIL + f": {err[:300]}"
        # Special diagnosis for rate limit
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            results["test3_status"] = WARN + ": 429 rate limit hit (free tier 16k tokens/min)"
            print(f"\n{WARN} - Rate limit (429). RAGAS API is compatible but free-tier quota too low.")
            print("  Solution: use RunConfig(max_workers=1) + truncate docs for pilot.")
            print("  For production: paid API key or Ollama local.")
        else:
            print(f"\n{FAIL} - {err[:400]}")
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY B0 TEST")
print("="*60)
print(f"Active model    : {active_model}")
print(f"Test 1 (API)    : {results.get('test1_status','?')}")
print(f"Test 2 (struct) : {results.get('test2_status','?')}")
print(f"Test 3 (RAGAS)  : {results.get('test3_status','?')}")
print()

# Save results
out_path = Path(__file__).resolve().parents[1] / "results" / "b0_test_results.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Results saved to: {out_path}")
