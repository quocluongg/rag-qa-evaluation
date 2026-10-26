# KẾ HOẠCH THỰC NGHIỆM (v3 — RAGAS + Gemini 3.1 Flash-Lite)
## Temporal Vector RAG vs. Temporal GraphRAG (theo `docs/main.tex`)

> **Mục tiêu:** Chạy đủ thí nghiệm để điền toàn bộ chỗ trống `các nội dung cần hoàn thiện` trong `main.tex`.
>
> **Thay đổi v3 (so với v2):**
> - **Generator LLM:** `gemini-3.5-flash-lite` thay `gpt-4o-mini` — không cần `OPENAI_API_KEY`.
> - **Judge LLM** cho RAGAS evaluate: `gemini-3.5-flash-lite` (temperature=0).
> - Truy cập qua **Google AI Studio API** (`langchain-google-genai`) hoặc **Ollama** local.
> - **Embeddings RAGAS KG:** `BAAI/bge-m3` local — dùng nhất quán để tránh dependency vào embedding API khác.
> - Hệ thống retrieval vẫn dùng **`BAAI/bge-m3`** như cũ.

---

## 1. Mục tiêu & Giả thuyết

| # | Giả thuyết | Cách kiểm định |
|---|---|---|
| H1 | Temporal filtering tăng **Temporal Accuracy** ở cả vector lẫn graph | V-RAG ↔ TV-RAG, G-RAG ↔ TG-RAG |
| H2 | Graph chỉ cho lợi ích đáng kể ở **Relational / Multi-hop / Historical** | Table VI, eq.(7): Δ_G(c)=A_TG(c)−A_TV(c) |
| H3 | TG-RAG đắt hơn về latency/context tokens | Accuracy–Efficiency trade-off |

---

## 2. Dữ liệu (không đổi)

29 tài liệu PTIT (`input/*.md`), metadata hiệu lực trong `output/metadata.json`.

**Số liệu thật cho Table II:**
- Admission Policies: 13 (Dh×3, TS×3, ThS×3, TB_THPT×4)
- Curriculum Documents: 10 · Academic Regulations: 2 · Administrative Decisions: 4
- Total Documents: 29 · Document Versions: 19 · Time Span: 2022–2026
- Graph Nodes: 2303 · Graph Relationships: 2804
- Phụ trợ: 376 text units (~1200 tokens/chunk), 362 community reports (level 0–4),
  6 chuỗi phiên bản (Dh, TS, ThS, TB_THPT, TB_TET, QĐ-học-bổng).

---

## 3. Hệ thống 2×2 (không đổi)

| System | Temporal | Graph |
|---|---|---|
| Vector RAG | ✗ | ✗ |
| Temporal Vector RAG | ✓ | ✗ |
| GraphRAG | ✗ | ✓ |
| Temporal GraphRAG | ✓ | ✓ |

Cùng corpus, metadata, embedding (`BAAI/bge-m3` 1024-dim, seq512, cosine),
LLM generation, prompt template, top-k=5, temperature=0.

**Quy tắc temporal filtering:**
- **Vector:** chunk hợp lệ ⟺ doc cha có `t_start ≤ t_q ≤ t_end`; open-ended đến khi bị replaced.
- **Graph pruning:** entity/relationship hợp lệ ⟺ ∃ ≥1 text_unit nguồn hợp lệ
  (parquet `entities`/`relationships` có `text_unit_ids`); community hợp lệ ⟺ ≥1 entity member hợp lệ.
  Prune = tạo view G_tq, không xóa vĩnh viễn.

---

## 4. Indexing (không đổi)

Graph index ✅ có sẵn. Vector index 🔄 đang re-embed qua Colab (`colab_run.ipynb`) →
kéo `lancedb.zip` về đè vào `output/lancedb`.
Lưu ý: graph content tiếng Anh, corpus tiếng Việt → `bge-m3` đa ngôn ngữ chịu được;
phải pilot xác nhận recall 5 câu trước khi full run.

---

## 5. Bộ dữ liệu — direct JSON generator với `gemini-3.5-flash-lite`

### 5.1. Cấu hình model (`src/gen_testset.py`)

```python
# ── CONFIRMED WORKING (tested 2026-08-24, ragas==0.4.3 + langchain-google-genai==4.3.5)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
import os

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  # https://aistudio.google.com/apikey

# LLM: gemini-3.5-flash-lite — with_structured_output() works natively
generator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
    )
)

# Embeddings: BAAI/bge-m3 local — GoogleGenerativeAIEmbeddings KHÔNG dùng được
# (text-embedding-004 lỗi 404 với langchain-google-genai==4.3.5 / API v1beta)
# Dùng ragas native HuggingFaceEmbeddings (không deprecated)
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmb
generator_embeddings = RagasHFEmb(model="BAAI/bge-m3")

# RunConfig: max_workers=1 BẮT BUỘC cho free-tier (quota 16k tokens/min)
run_cfg = RunConfig(max_workers=1, timeout=180, max_retries=5)

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
)
testset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=90,
    run_config=run_cfg,
)
```

> **Confirmed:** `gemini-3.5-flash-lite` ✅ · `with_structured_output` ✅ · `ragas.testset.TestsetGenerator` ✅
> **KHÔNG dùng** `GoogleGenerativeAIEmbeddings` — lỗi 404 v1beta với `langchain-google-genai==4.3.5`.
> **KHÔNG dùng** `LangchainEmbeddingsWrapper` — deprecated trong ragas 0.4.x.
> **Dùng** `ragas.embeddings.HuggingFaceEmbeddings` (native, không deprecated).

### 5.2. Pipeline sinh testset

```
input/*.md ──(metadata.json: doc_id, valid_from, valid_to)──► LangChain Documents
      │
      ▼
RAGAS KnowledgeGraph (Node DOCUMENT)
      │── default_transforms: HeadlineExtractor, KeyphraseExtractor,
      │                       SummaryExtractor, EmbeddingSimilarityRelationshipBuilder
      ▼
TestsetGenerator(
    llm        = gemini-3.5-flash-lite  (LangChain wrapper),
    embeddings = BAAI/bge-m3 (local),
    persona_list = [
        Persona("Sinh viên PTIT", "Cần biết quy chế học tập, điều kiện tốt nghiệp"),
        Persona("Phụ huynh",      "Quan tâm học phí, học bổng, xét tuyển"),
        Persona("Cán bộ CTSV",   "Tra cứu văn bản hành chính, quyết định"),
    ]
)
      │  query_distribution:
      │    SingleHopSpecific  0.40  → nhóm FACTUAL
      │    MultiHopSpecific   0.25  → nhóm MULTI-HOP
      │    MultiHopAbstract   0.25  → nhóm RELATIONAL
      │    (Temporal/Historical: augment thêm ở bước sau)
      ▼
Candidate pool ~90 samples (oversample 50%)
      │
      ├─► gán t_q: từ reference_contexts → doc nguồn → chọn ngày trong [valid_from, valid_to]
      │            doc bị replaced_by và t_q sau hết hạn → tag HISTORICAL
      ├─► direct generator: 18 candidates/category, including temporal/historical prompts
      │    → sinh câu TEMPORAL ("tại thời điểm X quy định nào có hiệu lực?")
      │    → sinh câu HISTORICAL ("version Y thay đổi gì so với Z?") + t_q cụ thể
      ▼
Human verification (CSV review):
      câu hỏi tự nhiên tiếng Việt? answerable từ corpus?
      reference không thêm fact ngoài corpus? gold_doc_ids đúng?
      │
      ▼
data/testset_v1.csv — 50 queries frozen:
      Factual 10 · Temporal 10 · Relational 10 · Multi-hop 10 · Historical 10
```

**Schema output:**
```
qid, category, user_input, tq, gold_doc_ids, reference_answer,
reference_contexts, synthesizer_name, source_docs, reviewer_status
```

### 5.3. Quyết định đã chốt

| Vấn đề | Chốt |
|---|---|
| Generator LLM | `gemini-3.5-flash-lite` · `langchain-google-genai==4.3.5` ✅ confirmed |
| Embeddings RAGAS KG | `BAAI/bge-m3` local · `ragas.embeddings.HuggingFaceEmbeddings` ✅ |
| Embeddings hệ thống | `BAAI/bge-m3` — nhất quán |
| Generation API | `google-genai` structured JSON output · direct prompt generator ✅ |
| Structured output | `with_structured_output()` native ✅ (không cần JsonOutputParser) |
| Rate limit | `RunConfig(max_workers=1)` bắt buộc cho free-tier (16k tokens/min) |
| Ngôn ngữ query | Tiếng Việt; persona ép tiếng Việt; lọc ở review nếu ra tiếng Anh |
| Temperature | 0 (generator, augmentation, judge) |
| Vai trò human review | Bắt buộc — giữ claim "**manually verified**" trong paper |

> ✅ **B0 đã xác nhận (2026-08-24):** `with_structured_output()` hoạt động native với `gemini-3.5-flash-lite`.
> `TestsetGenerator` API `ragas==0.4.3` import từ `ragas.testset` — đã verify.

> ⚠️ **Rate limit free-tier:** `gemma-4-31b` giới hạn 16,000 input tokens/phút.
> Bắt buộc dùng `RunConfig(max_workers=1)`. Full run 29 docs sẽ mất ~3–5 giờ.
> Giải pháp lâu dài: nâng lên paid tier hoặc chạy Ollama local.

> ⚠️ **Multi-hop abstract bị skip** khi graph sparse (ít doc).
> Với 2 doc pilot → bị skip. Full 29 docs → nên có đủ relationships. Kiểm tra sau B7.

---

## 6. Benchmark — metrics + metrics của bài

Sau khi 4 hệ trả về per-query: `retrieved_contexts, answer, latency, context_tokens`,
feed vào RAGAS evaluate với judge LLM = `gemini-3.5-flash-lite` (temperature=0):

```python
# src/eval/evaluate_ragas.py
from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall,
    answer_correctness, faithfulness, answer_relevancy,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)
judge_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
)

result = evaluate(
    dataset=ragas_dataset,
    metrics=[context_precision, context_recall,
             answer_correctness, faithfulness, answer_relevancy],
    llm=judge_llm,
    embeddings=judge_embeddings,
)
```

### Bảng metric đầy đủ

| Metric bài báo | Nguồn tính | Ghi chú |
|---|---|---|
| Precision@5, Recall@5, MRR | **Tự tính deterministic** trên `gold_doc_ids` | Metric chính Table V |
| Context Precision / Context Recall | **RAGAS** (LLM-judged bởi `gemini-3.5-flash-lite`) | Cross-check gold labels |
| **Answer Accuracy** | **RAGAS `answer_correctness`** (`gemini-3.5-flash-lite` judge) | Điểm chính Table VI; rubric 0–1 |
| Faithfulness, Answer Relevancy | **RAGAS** | Phân tích thêm (grounding) |
| **Temporal Accuracy** | **Custom** — tỷ lệ evidence vừa relevant vừa `V(d,tq)=1` | Viết `metrics_custom.py` |
| Latency, Context tokens | Tự đo (`time.perf_counter`) | Retrieval-only latency |

**Thống kê:** mean±std; bootstrap 95% CI / Wilcoxon signed-rank cho các cặp;
McNemar nếu binarize answer accuracy. Δ_G(c) theo eq.(7).

---

## 7. Cấu trúc code

```
src/
  config.py                       # GOOGLE_API_KEY, model names, paths, top-k, seed
  gen_testset.py                  # direct generator (gemini-3.5-flash-lite; max 5 attempts/question) → data/testset_raw.jsonl
  augment_temporal.py             # chuỗi phiên bản → câu temporal/historical + gán t_q
  verify_testset.py               # export CSV review → data/testset_v1.csv (frozen)
  indexing/
    reembed_bge_m3.py             # ✅ đã có
  retrieval/
    base.py                       # retrieve(q, t_q, k) → List[Evidence]
    vector_rag.py                 # Vector RAG
    temporal_vector_rag.py        # Temporal Vector RAG
    graph_rag.py                  # GraphRAG
    temporal_graph_rag.py         # Temporal GraphRAG (pruning G_tq)
  generation/
    answer.py                     # prompt template chung 4 hệ, gọi gemini-3.5-flash-lite
  eval/
    run_experiment.py             # 4 hệ × 50 queries → results/<run_id>/raw.jsonl
    evaluate_ragas.py             # ragas.evaluate() với gemma judge
    metrics_custom.py             # P@5/R@5/MRR, TemporalAcc, Δ_G, latex export
data/
  testset_raw.jsonl               # pool ~90 (chưa review)
  testset_v1.csv                  # frozen 50 câu (sau human review)
results/runs/<run_id>/
  config_manifest.json
  raw.jsonl
  summary.csv
```

**Manifest mỗi lần chạy:**
```json
{
  "ragas_version": "x.x.x",
  "generator_llm": "gemini-3.5-flash-lite",
  "judge_llm": "gemini-3.5-flash-lite",
  "embedding_ragas": "text-embedding-004",
  "embedding_system": "BAAI/bge-m3",
  "generation_llm": "gemini-3.5-flash-lite",
  "top_k": 5,
  "temperature_generate": 0,
  "temperature_judge": 0,
  "testset_sha256": "<hash của testset_v1.csv>",
  "seed": 42
}
```

---

## 8. Checklist thực hiện

- [x] **B0.** ✅ Setup Gemma API + kiểm tra compatibility — **DONE 2026-08-24**
  ```
  Test 1 (API gemini-3.5-flash-lite)  : PASS
  Test 2 (with_structured_output): PASS  
  Test 3 (direct JSON generator): PASS
  
  Key findings:
  - gemini-3.5-flash-lite hoat dong voi GOOGLE_API_KEY
  - with_structured_output() native, khong can JsonOutputParser
  - ragas.testset.TestsetGenerator (0.4.x) compat OK
  - GoogleGenerativeAIEmbeddings KHONG dung duoc (404 v1beta) -> dung BAAI/bge-m3 local
  - RunConfig(max_workers=1) bat buoc cho free-tier (16k tokens/min)
  - Multi-hop abstract bi skip khi graph sparse (<5 docs)
  - Script: src/test_b0.py | Results: results/b0_test_results.json
  ```
- [ ] **B1.** Re-embed xong (`colab_run.ipynb`), kéo `lancedb.zip` về `output/lancedb`
- [ ] **B2.** Viết `config.py` + 4 module retrieval + `generation/answer.py`; smoke-test 5 query × 4 hệ
- [ ] **B3.** `gen_testset_direct.py` sinh pool 90 bằng direct prompt `gemini-3.5-flash-lite`; xác nhận JSON parse không lỗi
- [ ] **B4.** `augment_temporal.py` → sinh câu Temporal/Historical + gán `t_q`
- [ ] **B5.** `verify_testset.py` export CSV → **human review** → freeze `data/testset_v1.csv` (60 câu)
- [ ] **B6.** Pilot 10 câu × 4 hệ: kiểm tra recall bge-m3 với graph-EN/query-VN, sửa pruning/mapping
- [ ] **B7.** Full run 50 × 4 → `results/<run_id>/raw.jsonl`
- [ ] **B8.** `evaluate_ragas.py` (`gemini-3.5-flash-lite` judge) + `metrics_custom.py` → Table V, VI (csv → latex)
- [ ] **B9.** Phân tích Δ_G, trade-off, statistical tests (Wilcoxon / McNemar)
- [ ] **B10.** Điền số vào `main.tex` + viết narrative Results + subsection "Test set construction"

---

## 9. Map bảng các nội dung cần hoàn thiện trong main.tex

| Vị trí | Nguồn điền |
|---|---|
| Table II — Dataset Statistics | Mục 2 (đã có số thật) |
| Table IV — Experimental Settings | Embedding=`BAAI/bge-m3`(1024d/cosine/seq512) · Generation=`gemini-3.5-flash-lite`(temp=0) · Chunk=1200 · Top-k=5 · Queries=50 (direct-prompt-generated, source-grounded, human-verified) · Judge=`gemini-3.5-flash-lite`(temp=0) |
| Table V — Overall Performance | B8: P@5, R@5, MRR, TemporalAcc, AnswerAcc (RAGAS correctness), Latency |
| Table VI — Graph Gain | B8: Δ_G(c) theo eq.(7) p10 per category |

---

## 10. Rủi ro & Phương án (v3)

| Rủi ro | Phương án |
|---|---|
| `gemini-3.5-flash-lite` chưa khả dụng trên Google AI Studio | Fallback `gemini-3.5-flash-lite`; dừng run nếu cả hai model không khả dụng |
| Gemini không support structured output → RAGAS lỗi parse | Kiểm tra B0; nếu lỗi thì dừng run và không ghi kết quả không hợp lệ |
| RAGAS API version không tương thích `langchain-google-genai` | Test compat trước khi pin; tạo virtualenv riêng cho experiment |
| Model sinh query tiếng Anh dù corpus tiếng Việt | Persona/prompt ép tiếng Việt; lọc ở review; ghi tỉ lệ trong limitations |
| Reference answer hallucinate | Bắt buộc human verification từng dòng trước khi freeze |
| Synthetic queries quá dễ | Oversample 50%, ưu tiên giữ câu multi-hop khi lọc xuống 50 |
| Google API rate limit khi full run | `time.sleep(1)` giữa requests; hoặc chạy Ollama local |
| Δ_G ≈ 0 mọi category | Vẫn là kết quả hợp lệ — trả lời research question; viết analysis kỹ |

---

## 11. Timeline (4 tuần)

| Tuần | Việc |
|---|---|
| 1 | B0 setup Gemma API + test compat · B1 re-embed · B2 viết retrievers |
| 2 | B3–B5 testset (gen → augment → review → freeze) · B6 pilot |
| 3 | B7 full run · B8 evaluate (RAGAS + custom metrics) |
| 4 | B9 analysis · B10 viết Results vào `main.tex` |

---

## 12. Dependencies

```
# requirements_experiment.txt
ragas>=0.2.0                   # kiểm tra version trước khi pin chính xác
langchain>=0.2.0
langchain-google-genai>=1.0.0  # ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
google-generativeai>=0.5.0
langchain-community>=0.2.0
langchain-ollama>=0.1.0        # fallback Ollama
lancedb>=0.5.0
sentence-transformers>=2.7.0   # bge-m3
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.13.0                  # Wilcoxon signed-rank test
tqdm>=4.66.0
python-dotenv>=1.0.0
```
