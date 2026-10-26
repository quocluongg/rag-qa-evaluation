# Đánh giá hiệu quả và độ tin cậy của hệ thống Retrieval-Augmented Generation (RAG) trong bài toán hỏi–đáp

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quocluongg/rag-qa-evaluation/blob/main/RAG_QA_Benchmark_Colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repository nghiên cứu đối chứng có kiểm soát giữa 3 kiến trúc xử lý ngôn ngữ tự nhiên trong bài toán Hỏi–Đáp (QA) trên tập dữ liệu học thuật & quy chế hành chính của Học viện Công nghệ Bưu chính Viễn thông (PTIT):
1. **Hệ thống 1 — LLM thuần (Vanilla LLM):** Không truy hồi, trả lời hoàn toàn từ tri thức tham số nội tại của mô hình (`gemini-3.5-flash-lite`, temperature=0.0).
2. **Hệ thống 2 — Vector RAG:** Truy hồi ngữ nghĩa dày đặc (Dense Semantic Retrieval) bằng mô hình embedding `BAAI/bge-m3` và cơ sở dữ liệu vector LanceDB (Top-5 chunks).
3. **Hệ thống 3 — GraphRAG:** Truy hồi dựa trên cấu trúc đồ thị tri thức Microsoft GraphRAG (Entity lookup + Subgraph BFS 1-hop expansion + Semantic re-ranking Top-5).

---

## 🔬 1. Phương pháp luận sinh Bộ câu hỏi Benchmark chuẩn Khoa học

Bộ câu hỏi được thiết kế theo các tiêu chuẩn benchmark quốc tế (như RGB, CRUD-RAG, Microsoft GraphRAG) nhằm đánh giá khách quan điểm mạnh và điểm yếu của từng mô hình:
- **`factual_single_hop` (Sự kiện đơn bước):** Hỏi về dữ kiện cụ thể xuất hiện trực tiếp trong 1 đoạn văn bản duy nhất.
- **`multi_hop_relational` (Suy luận đa bước):** Câu hỏi đòi hỏi xâu chuỗi dữ kiện giữa 2 thực thể hoặc kết hợp 2 đoạn văn bản liên quan.
- **`global_synthesis` (Tổng hợp toàn cục):** Câu hỏi khái quát, tóm tắt và đối chiếu trên phạm vi nhiều văn bản hoặc cộng đồng tri thức.
- **`unanswerable_negative` (Câu hỏi bẫy / Không có đáp án):** Đánh giá trực diện độ tin cậy và khả năng chống ảo giác (hallucination resistance) thông qua khả năng từ chối trả lời chính xác.

Mọi câu hỏi đều được gán bằng chứng trích dẫn nguyên văn (`evidence_snippet`), mã định danh tài liệu chuẩn (`gold_doc_ids`), và mã khối văn bản chuẩn (`gold_tu_ids`).

---

## 📊 2. Hệ thống Chỉ số Đánh giá (Evaluation Metrics)

| Nhóm đánh giá | Chỉ số | Định nghĩa / Phương pháp đo |
|---|---|---|
| **Chất lượng truy hồi** | **Precision@5** | Tỷ lệ tài liệu truy hồi trong Top-5 thực sự liên quan (*LLM thuần: N/A*) |
| | **Recall@5** | Tỷ lệ tài liệu chuẩn được hệ thống bao phủ thành công (*LLM thuần: N/A*) |
| | **MRR** | Nghịch đảo vị trí xuất hiện của tài liệu đúng đầu tiên (*LLM thuần: N/A*) |
| **Chất lượng câu trả lời** | **Answer Accuracy** | Độ chính xác ngữ nghĩa so với đáp án chuẩn (LLM-as-a-Judge, thang [0.0 - 1.0]) |
| **Độ tin cậy** | **Faithfulness** | Tỷ lệ không bịa đặt sự thật, bám sát ngữ cảnh (Chống ảo giác) |
| | **Refusal Accuracy** | Khả năng từ chối chính xác đối với nhóm câu hỏi bẫy (`unanswerable`) |
| **Hiệu năng & Tài nguyên** | **Retrieval Latency** | Thời gian tìm kiếm vector / duyệt đồ thị (giây) |
| | **Generation Latency**| Thời gian sinh câu trả lời của mô hình ngôn ngữ lớn (giây) |
| | **Context Tokens** | Số lượng token ngữ cảnh nạp vào prompt (đo chi phí API) |
| **Kiểm định Thống kê** | **Wilcoxon & Paired t-test** | Kiểm định cặp có ý nghĩa thống kê ($p < 0.05$) giữa các hệ thống |

---

## 🚀 3. Hướng dẫn Chạy Thực nghiệm trên Google Colab

Chỉ cần bấm vào huy hiệu Colab phía trên hoặc mở file [`RAG_QA_Benchmark_Colab.ipynb`](RAG_QA_Benchmark_Colab.ipynb) trên Google Colab:
1. **Thiết lập Secrets:** Thêm API Key của bạn vào mục **Secrets** (biểu tượng chìa khóa ở thanh bên trái của Colab):
   - Tên Secret: `GOOGLE_API_KEY`
   - Giá trị: Lấy miễn phí tại [Google AI Studio](https://aistudio.google.com/apikey).
2. **Chạy tuần tự:** Bấm `Runtime` -> `Run all` (hoặc chạy từng cell theo số thứ tự từ 1 đến 9).
3. **Cơ chế Checkpoint & Resume:** Kết quả và bộ câu hỏi được tự động đồng bộ sang Google Drive tại `/content/drive/MyDrive/rag_qa_benchmark/`. Nếu phiên Colab bị ngắt, việc chạy lại cell sẽ tiếp tục từ vị trí dừng mà không tốn lại quota API.

---

## 💻 4. Chạy cục bộ bằng Command Line

```bash
# 1. Cài đặt môi trường
pip install -r requirements.txt

# 2. Thiết lập API Key
export GOOGLE_API_KEY="your-api-key"      # Linux / macOS
$env:GOOGLE_API_KEY="your-api-key"       # Windows PowerShell

# 3. Sinh bộ câu hỏi benchmark khoa học (50 câu)
python src/gen_scientific_testset.py --size 50 --out data/scientific_testset_50.csv --resume

# 4. Thực thi benchmark 3 hệ thống
python src/run_3systems.py --testset data/scientific_testset_50.csv --output results/benchmark_run.jsonl --k 5 --resume

# 5. Đánh giá chỉ số và xuất bảng LaTeX / Biểu đồ
python src/eval_3systems.py --input results/benchmark_run.jsonl --output-dir results --judge --resume
```

---

## 📁 5. Cấu trúc Thư mục

```
rag-qa-evaluation/
├── RAG_QA_Benchmark_Colab.ipynb    # Notebook tổng thể 1-click chạy trên Google Colab
├── README.md                       # Tài liệu hướng dẫn chi tiết
├── requirements.txt                # Thư viện phụ thuộc được ghim phiên bản
├── input/                          # 29 văn bản hành chính & đào tạo PTIT 2022-2026
├── output/                         # Chỉ mục GraphRAG & LanceDB có sẵn
│   ├── lancedb/                    # Bảng vector LanceDB (BAAI/bge-m3 embeddings)
│   ├── entities.parquet            # Đồ thị thực thể (2.303 nodes)
│   ├── relationships.parquet       # Đồ thị quan hệ (2.804 edges)
│   ├── text_units.parquet          # 376 text chunks (~1.200 tokens/chunk)
│   ├── communities.parquet         # Cụm cộng đồng phân cấp
│   └── metadata.json               # Metadata hiệu lực thời gian của tài liệu
├── src/
│   ├── retrieval/
│   │   ├── vanilla_llm.py          # Hệ thống 1: LLM thuần (No retrieval)
│   │   ├── vector_rag.py           # Hệ thống 2: Vector RAG (LanceDB ANN)
│   │   └── graph_rag.py            # Hệ thống 3: GraphRAG (BFS Subgraph + Rerank)
│   ├── gen_scientific_testset.py   # Bộ sinh câu hỏi 4 nhóm chuẩn khoa học
│   ├── run_3systems.py             # Bộ điều phối thực nghiệm (Streaming JSONL runner)
│   └── eval_3systems.py            # Chấm điểm RAGAS, kiểm định thống kê & xuất LaTeX
└── results/                        # Thư mục chứa kết quả, bảng tổng hợp và biểu đồ
```

---

## 📜 Trích dẫn & Bản quyền
Mã nguồn phục vụ đề tài nghiên cứu khoa học: *"Đánh giá hiệu quả và độ tin cậy của hệ thống Retrieval-Augmented Generation (RAG) trong bài toán hỏi–đáp"*.
