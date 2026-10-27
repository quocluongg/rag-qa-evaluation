"""
build_master_notebook.py — Generates the complete, self-contained Google Colab master notebook.
"""
import json
from pathlib import Path

nb = {
    "cells": [],
    "metadata": {
        "colab": {
            "name": "RAG_QA_Benchmark_Colab.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True)
    })

# --- TITLE & INTRO ---
add_md("""# CHUYÊN ĐỀ NGHIÊN CỨU KHOA HỌC: ĐÁNH GIÁ HIỆU QUẢ VÀ ĐỘ TIN CẬY CỦA HỆ THỐNG RAG TRONG BÀI TOÁN HỎI–ĐÁP

**Thực nghiệm so sánh đối chứng 3 hệ thống:**
1. **Hệ thống 1 — LLM thuần (Vanilla LLM):** Không truy hồi, trả lời hoàn toàn dựa vào tri thức tham số nội tại.
2. **Hệ thống 2 — Vector RAG:** Truy hồi ngữ nghĩa sử dụng mô hình embedding đa ngôn ngữ `BAAI/bge-m3` và cơ sở dữ liệu vector LanceDB.
3. **Hệ thống 3 — GraphRAG:** Truy hồi cấu trúc dựa trên đồ thị tri thức trích xuất thực thể, quan hệ và đồ thị con (Entity-Relation BFS + Semantic Re-ranking).

---
### 📌 Hướng dẫn thực thi trên Google Colab:
- Notebook được thiết kế **100% tương thích với Google Colab runtime miễn phí (CPU hoặc T4 GPU)**.
- Kết quả trung gian và bộ câu hỏi được **tự động lưu liên tục vào Google Drive**, hỗ trợ resume mượt mà khi bị timeout hoặc ngắt kết nối giữa chừng.
- API Key được lấy bảo mật qua **Colab Secrets** (`GOOGLE_API_KEY`).
""")

# --- CELL 1: SETUP ---
add_md("""## 1. Cài đặt môi trường & Kiểm tra phần cứng
Chạy cell này để kiểm tra cấu hình phần cứng Colab và cài đặt các thư viện phụ thuộc đã được ghim phiên bản (pinned versions).
""")

add_code("""# Cell 1: Cài đặt thư viện & kiểm tra phần cứng
import os, sys, subprocess

print("=== CÀI ĐẶT THƯ VIỆN ĐÃ GHIM PHIÊN BẢN ===")
!pip install -q \\
    ragas==0.4.3 \\
    langchain-google-genai==4.3.5 \\
    google-genai>=1.65.0 \\
    sentence-transformers>=3.0.0 \\
    lancedb>=0.15.0 \\
    pyarrow>=14.0 \\
    rapidfuzz>=3.0.0 \\
    pydantic>=2.0 \\
    scipy>=1.10 \\
    tabulate>=0.9.0 \\
    seaborn matplotlib pandas

print("\\n=== KIỂM TRA PHẦN CỨNG ===")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
    else:
        print("Đang chạy trên CPU (Hoàn toàn đáp ứng tốt toàn bộ pipeline)")
    print(f"Torch version: {torch.__version__}")
except Exception as e:
    print(f"[LỖI] Không import được torch sau khi cài đặt: {e}")
    print("→ Chọn Runtime → Restart session, sau đó chạy lại Cell 1.")
    raise

print("\\n[OK] Cài đặt hoàn tất!")
""")

# --- CELL 2: MOUNT DRIVE & CLONE REPO ---
add_md("""## 2. Mount Google Drive & Clone Repository
Thư mục làm việc sẽ được đồng bộ với Google Drive tại `/content/drive/MyDrive/rag_qa_benchmark/` để lưu trữ checkpoints kết quả an toàn.
""")

add_code("""# Cell 2: Mount Google Drive & Clone Repository
from pathlib import Path
import shutil

# 1. Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = Path('/content/drive/MyDrive/rag_qa_benchmark')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Đã mount Drive. Thư mục lưu trữ: {DRIVE_DIR}")
except Exception as e:
    DRIVE_DIR = Path.cwd() / "drive_backup"
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[CẢNH BÁO] Không mount được Drive ({e}). Lưu cục bộ tại {DRIVE_DIR}")

# 2. Clone repository mới nhất
REPO_URL = "https://github.com/quocluongg/rag-qa-evaluation.git"
WORK_DIR = Path("/content/rag-qa-evaluation")

if not WORK_DIR.exists():
    print(f"Đang clone repository từ {REPO_URL}...")
    !git clone $REPO_URL $WORK_DIR
else:
    print(f"Repository đã tồn tại tại {WORK_DIR}. Đang cập nhật git pull...")
    !cd $WORK_DIR && git pull origin main

%cd $WORK_DIR
for p in (WORK_DIR, WORK_DIR / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

print(f"Thư mục làm việc hiện tại: {Path.cwd()}")
""")

# --- CELL 3: SECRETS ---
add_md("""## 3. Cấu hình Google Gemini API Key
Khóa API được đọc an toàn từ **Google Colab Secrets** (biểu tượng chiếc chìa khóa ở thanh công cụ bên trái).  
*Đặt tên Secret là: `GOOGLE_API_KEY`*.
""")

add_code("""# Cell 3: Cấu hình API Key
import os
from getpass import getpass

GOOGLE_API_KEY = None
try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
except Exception:
    pass

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = getpass("Nhập GOOGLE_API_KEY của bạn (lấy tại https://aistudio.google.com/apikey): ")

if not GOOGLE_API_KEY or not GOOGLE_API_KEY.strip():
    raise ValueError("GOOGLE_API_KEY không được để trống!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY.strip()
os.environ["GENERATOR_MODEL"] = "gemini-3.5-flash-lite"
print("[OK] Đã nạp thành công GOOGLE_API_KEY!")
""")

# --- CELL 4: DATA & INDEX VALIDATION ---
add_md("""## 4. Xác thực Dữ liệu Văn bản & Chỉ mục (LanceDB + Parquet)
Kiểm tra sự toàn vẹn của:
- 29 tài liệu PTIT gốc (`input/*.md`).
- Bảng cơ sở dữ liệu vector LanceDB (`output/lancedb/`).
- Bảng đồ thị tri thức GraphRAG (`output/*.parquet`).
""")

add_code("""# Cell 4: Kiểm tra tính toàn vẹn dữ liệu
import subprocess
import sys
import lancedb
import pandas as pd
from pathlib import Path

docs_dir = Path("input")
docs_count = len(list(docs_dir.glob("*.md")))
print(f"1. Số lượng văn bản gốc PTIT (input/*.md): {docs_count} files")
assert docs_count >= 25, "Thiếu văn bản đầu vào trong input/"

print("\\n2. Kiểm tra các bảng LanceDB:")
db = lancedb.connect("output/lancedb")
tables = db.table_names()
print(f"   Các bảng hiện có: {tables}")
for tname in ["text_unit_text", "entity_description", "community_full_content"]:
    assert tname in tables, f"Thiếu bảng {tname} trong LanceDB!"
    count = db.open_table(tname).count_rows()
    print(f"   -> Bảng '{tname}': {count} dòng.")

try:
    db.open_table("text_unit_text").to_pandas()
    print("   -> text_unit_text: đọc dữ liệu vector thành công.")
except Exception as e:
    print(f"   -> [CẢNH BÁO] text_unit_text chưa đọc được dữ liệu: {e}")
    print("      Đang tái tạo chỉ mục vector từ output/text_units.parquet...")
    subprocess.run([sys.executable, "src/build_lancedb_index.py"], check=True)
    db = lancedb.connect("output/lancedb")
    db.open_table("text_unit_text").to_pandas()
    print("   -> [OK] Đã tái tạo và xác thực text_unit_text.")

print("\\n3. Kiểm tra các file Parquet đồ thị tri thức GraphRAG:")
pq_files = ["entities.parquet", "relationships.parquet", "communities.parquet", "community_reports.parquet", "text_units.parquet"]
for pq in pq_files:
    pq_path = Path("output") / pq
    assert pq_path.exists(), f"Thiếu file {pq}!"
    df = pd.read_parquet(pq_path)
    print(f"   -> File '{pq}': {len(df)} dòng, dung lượng {round(pq_path.stat().st_size/1024, 1)} KB")

print("\\n[XÁC THỰC THÀNH CÔNG] Dữ liệu và các chỉ mục đã sẵn sàng để benchmark!")
""")

# --- CELL 5: SCIENTIFIC TESTSET GENERATION ---
add_md("""## 5. Tạo bộ câu hỏi Thực nghiệm chuẩn Khoa học (Scientific Benchmark Testset)

Theo phương pháp luận chuẩn trong các nghiên cứu quốc tế về RAG (như RGB, CRUD-RAG, Microsoft GraphRAG), bộ câu hỏi được phân bổ cân bằng qua **4 nhóm câu hỏi**:
1. **`factual_single_hop`** (Sự kiện đơn bước): Hỏi về dữ kiện cụ thể xuất hiện trực tiếp trong 1 đoạn văn bản.
2. **`multi_hop_relational`** (Suy luận đa bước): Đòi hỏi liên kết thông tin giữa 2 thực thể hoặc 2 đoạn văn bản khác nhau.
3. **`global_synthesis`** (Tổng hợp toàn cục): Đòi hỏi đối chiếu, tóm tắt trên phạm vi nhiều văn bản/cộng đồng tri thức.
4. **`unanswerable_negative`** (Câu hỏi bẫy / Không có đáp án): Kiểm tra khả năng từ chối trả lời và mức độ triệt tiêu ảo giác (hallucination resistance).

*Mọi câu hỏi đều được gán bằng chứng trích dẫn nguyên văn (`evidence_snippet`) và tài liệu chuẩn (`gold_doc_ids`).*

**Bản v2 dùng cho báo cáo:** 67 câu, trong đó **30 câu multi-hop** (17 câu mới sinh có kiểm chứng từ bảng học phần), đủ mẫu để chạy kiểm định thống kê theo nhóm.
""")

add_code("""# Cell 5: Nạp bộ câu hỏi thực nghiệm v2 (67 câu, 30 multi-hop)
import subprocess
from pathlib import Path

DRIVE_TESTSET = DRIVE_DIR / "scientific_testset_v2.csv"
LOCAL_TESTSET = Path("data/scientific_testset_v2.csv")
LOCAL_TESTSET.parent.mkdir(parents=True, exist_ok=True)

if DRIVE_TESTSET.exists() and not LOCAL_TESTSET.exists():
    print(f"Khôi phục testset v2 từ Google Drive: {DRIVE_TESTSET}")
    shutil.copy(DRIVE_TESTSET, LOCAL_TESTSET)

if not LOCAL_TESTSET.exists():
    print("Chưa có testset v2, sinh lại từ bộ 50 câu gốc...")
    !python src/expand_multihop_testset.py --source data/scientific_testset_50.csv --out data/scientific_testset_v2.csv --raw-out data/scientific_testset_v2.raw.jsonl --target-multihop 30

import pandas as pd
from retrieval.source_map import get_source_map
from testset_utils import repair_gold_doc_ids

df_test = pd.read_csv(LOCAL_TESTSET)
testset_rows = df_test.to_dict("records")
repaired = repair_gold_doc_ids(testset_rows, get_source_map())
if repaired:
    df_test = pd.DataFrame(testset_rows)
    df_test.to_csv(LOCAL_TESTSET, index=False, encoding="utf-8")
    print(f"[OK] Đã bổ sung gold_doc_ids cho {repaired} câu hỏi từ gold_tu_ids.")

shutil.copy(LOCAL_TESTSET, DRIVE_TESTSET)
counts = df_test["category"].value_counts()
assert len(df_test) == 67, f"Testset v2 phải có 67 câu, hiện có {len(df_test)}"
assert counts.get("multi_hop_relational", 0) >= 30, "Testset v2 phải có ít nhất 30 câu multi-hop"

print(f"\\nPhân bổ câu hỏi theo nhóm:")
print(counts)
print(f"\\nCác câu multi-hop mới (Q051–Q067):")
display(df_test[df_test["qid"] >= "Q051"][["qid", "question", "reference_answer"]].head(5))
""")

# --- CELL 6: 3 SYSTEMS IMPLEMENTATION ---
add_md("""## 6. Cài đặt chi tiết 3 Hệ thống Thí nghiệm (Architecture & Pipelines)

Cả 3 hệ thống đều được điều khiển chặt chẽ theo các điều kiện đối chứng:
- **Cùng mô hình sinh:** Google `gemini-3.5-flash-lite` với `temperature = 0.0`.
- **Cùng số lượng ngữ cảnh:** Top-$K = 8$ đối với Vector RAG và GraphRAG.
- **Cùng prompt template quy định rõ ràng hành vi trích dẫn và chống bịa đặt.**
""")

add_code("""# Cell 6: Khởi tạo 3 hệ thống thực nghiệm
from retrieval.vanilla_llm import VanillaLLMRetriever
from retrieval.vector_rag import VectorRAG
from retrieval.graph_rag import GraphRAG

print("Đang khởi tạo 3 hệ thống...")
sys1_vanilla = VanillaLLMRetriever(k=0)
sys2_vector  = VectorRAG(k=8, embed_model="BAAI/bge-m3")
sys3_graph   = GraphRAG(k=8, embed_model="BAAI/bge-m3")

print("Đang nạp mô hình embedding và kiểm tra chỉ mục LanceDB...")
sys2_vector.ensure_ready()

# Thử nghiệm truy hồi với 1 câu hỏi mẫu
sample_query = "Ngành Công nghệ vi mạch bán dẫn đào tạo những khối kiến thức nào?"
print(f"\\nThử nghiệm câu hỏi: '{sample_query}'")

res_v = sys2_vector.retrieve(sample_query)
print(f"-> Vector RAG lấy được {len(res_v.evidence)} đoạn evidence, thời gian truy hồi: {round(res_v.latency_s, 3)}s")

res_g = sys3_graph.retrieve(sample_query)
print(f"-> GraphRAG lấy được {len(res_g.evidence)} đoạn evidence, thời gian truy hồi: {round(res_g.latency_s, 3)}s")

print("\\n[OK] Cả 3 hệ thống đã khởi tạo thành công!")
""")

# --- CELL 7: BENCHMARK EXECUTION ---
add_md("""## 7. Chạy Thực nghiệm Đối chứng (Benchmark Execution)
Hệ thống sẽ chạy qua 67 câu hỏi cho cả 3 hệ thống (tổng cộng 201 lượt sinh), với Top-K = 8.
- Lưu kết quả tức thì vào file JSONL sau mỗi câu hỏi.
- **Hỗ trợ tự động Resume (`--resume`):** Nếu phiên Colab bị ngắt, chỉ cần chạy lại cell này, hệ thống sẽ tiếp tục câu hỏi còn dang dở mà không tốn lại chi phí API. Record cũ khác `prompt_version`/`top_k` sẽ tự động bị loại và backup.
""")

add_code("""# Cell 7: Thực thi benchmark với checkpoint tự động
from pathlib import Path
import gc
import shutil
import torch

DRIVE_RUN = DRIVE_DIR / "benchmark_run.jsonl"
LOCAL_RUN = Path("results/benchmark_run.jsonl")
LOCAL_RUN.parent.mkdir(parents=True, exist_ok=True)

# Khôi phục từ Drive nếu có
if DRIVE_RUN.exists() and not LOCAL_RUN.exists():
    print(f"Khôi phục file kết quả từ Drive: {DRIVE_RUN}")
    shutil.copy(DRIVE_RUN, LOCAL_RUN)

# Giải phóng BGE-M3 khỏi kernel trước khi subprocess nạp model riêng,
# tránh CUDA OOM do hai tiến trình cùng giữ model trên GPU.
for _name in ("sys1_vanilla", "sys2_vector", "sys3_graph"):
    globals().pop(_name, None)
try:
    from retrieval.index_utils import load_embedder
    load_embedder.cache_clear()
except Exception:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[GPU] Đã giải phóng bộ nhớ đệm trước khi chạy benchmark.")

print("Bắt đầu chạy benchmark 3 hệ thống...")
!python src/run_3systems.py --testset data/scientific_testset_v2.csv --output results/benchmark_run.jsonl --k 8 --resume

# Tự động sao lưu sang Google Drive
if LOCAL_RUN.exists():
    shutil.copy(LOCAL_RUN, DRIVE_RUN)
    print(f"Đã sao lưu kết quả benchmark lên Drive: {DRIVE_RUN}")
""")

# --- CELL 8: AUTOMATED EVALUATION ---
add_md("""## 8. Đánh giá Chỉ số Tự động (LLM-as-a-Judge & Structured Metrics)

Thực hiện đánh giá toàn diện:
1. **Chỉ số truy hồi (Retrieval):** $P@8$, $R@8$, $MRR$ (đối với LLM thuần quy ước là `N/A`).
2. **Chất lượng câu trả lời & Độ tin cậy:**
   - **Answer Accuracy / Correctness:** Điểm trung bình của **3 lần judge độc lập**, kèm độ lệch chuẩn và mức đồng thuận.
   - **Faithfulness:** Mức độ trung thực, triệt tiêu ảo giác (0.0 = bịa đặt sự thật; 1.0 = bám sát bằng chứng hoặc từ chối đúng).
   - **Refusal Accuracy / Refusal Rate:** Khả năng từ chối đúng câu hỏi không có đáp án và tỷ lệ từ chối trên toàn bộ câu hỏi.
   - **Factual consistency guard:** Hạ điểm các câu trả lời thiếu mã môn / tín chỉ / ngày tháng so với đáp án chuẩn.
3. **Hiệu năng & Tài nguyên:** Thời gian truy hồi, thời gian sinh câu trả lời, số token ngữ cảnh, p50 và p95 latency.
4. **Kiểm định Thống kê:** McNemar (mức fully correct, score ≥ 0.99) + Wilcoxon signed-rank trên điểm liên tục, hiệu chỉnh Holm; xuất cả overall và theo nhóm câu hỏi.
5. **Phân tích lỗi định tính:** Phân loại multi-hop thành `retrieval_miss`, `lazy_refusal`, `partial_answer`, `wrong_answer` và xuất ca tiêu biểu.
""")

add_code("""# Cell 8: Đánh giá chỉ số RAGAS & Structured Judge
from pathlib import Path
import shutil

DRIVE_JUDGED = DRIVE_DIR / "judged_benchmark.jsonl"
LOCAL_JUDGED = Path("results/judged_benchmark.jsonl")

if DRIVE_JUDGED.exists() and not LOCAL_JUDGED.exists():
    shutil.copy(DRIVE_JUDGED, LOCAL_JUDGED)

print("Bắt đầu chấm điểm và tính toán các chỉ số khoa học...")
!python src/eval_3systems.py --input results/benchmark_run.jsonl --output-dir results --judge --resume --judge-repeats 3 --k 8
!python src/analyze_errors.py --input results/judged_benchmark.jsonl --output-dir results --category multi_hop_relational --examples 3

import pandas as pd
for name in [
    "summary_metrics.csv",
    "category_breakdown.csv",
    "judge_reliability.csv",
    "factual_guard_summary.csv",
    "significance_tests.csv",
    "error_summary.csv",
]:
    path = Path("results") / name
    if path.exists():
        print(f"\\n=== {name} ===")
        display(pd.read_csv(path))

report_path = Path("results/error_analysis.md")
if report_path.exists():
    from IPython.display import Markdown, display as display_md
    display_md(Markdown(report_path.read_text(encoding="utf-8")))

# Sao lưu toàn bộ thư mục results sang Google Drive
for f in Path("results").glob("*"):
    if f.is_file():
        shutil.copy(f, DRIVE_DIR / f.name)
print(f"Đã lưu toàn bộ kết quả tổng hợp sang Google Drive: {DRIVE_DIR}")
""")

# --- CELL 9: VISUALIZATIONS & REPORT ---
add_md("""## 9. Trực quan hóa Biểu đồ & Xuất Báo cáo Học thuật (Charts & LaTeX Tables)
Vẽ các biểu đồ trực quan phục vụ bài báo và báo cáo chuyên đề:
- **Biểu đồ Cột:** So sánh độ chính xác theo từng nhóm câu hỏi, kèm **95% bootstrap CI**.
- **Biểu đồ Cột tổng thể:** Accuracy trung bình của 3 hệ thống kèm CI.
- **Biểu đồ Cột chồng:** Phân loại lỗi multi-hop (`retrieval_miss`, `lazy_refusal`, `partial_answer`, `wrong_answer`).
- **Biểu đồ Scatter:** Đánh đổi giữa Độ chính xác (Accuracy) và Độ trễ (Latency).
- **Bảng LaTeX, bảng kiểm định thống kê và báo cáo lỗi** sẵn sàng copy vào báo cáo chuyên đề.
""")

add_code("""# Cell 9: Vẽ biểu đồ và xuất bảng báo cáo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

out_dir = Path("results")
summary_df = pd.read_csv(out_dir / "summary_metrics.csv")
cat_df = pd.read_csv(out_dir / "category_breakdown.csv")

def parse_ci(series):
    lows, highs = [], []
    for value in series.astype(str):
        if "[" in value and "]" in value:
            left, right = value.split("[", 1)[1].split("]", 1)[0].split(",", 1)
            lows.append(float(left))
            highs.append(float(right))
        else:
            lows.append(np.nan)
            highs.append(np.nan)
    return pd.Series(lows, index=series.index), pd.Series(highs, index=series.index)

print("=== BẢNG TỔNG HỢP HIỆU NĂNG 3 HỆ THỐNG ===")
display(summary_df)

print("\\n=== PHÂN TÍCH CHI TIẾT THEO NHÓM CÂU HỎI ===")
display(cat_df)

# 1. Answer Accuracy theo Category, kèm 95% bootstrap CI
categories = list(dict.fromkeys(cat_df["Category"]))
systems = list(dict.fromkeys(cat_df["System"]))
x = np.arange(len(categories))
width = 0.8 / max(len(systems), 1)
fig, ax = plt.subplots(figsize=(13, 6))
for i, system in enumerate(systems):
    sub = cat_df[cat_df["System"] == system].set_index("Category").reindex(categories)
    values = pd.to_numeric(sub["Answer Accuracy"], errors="coerce")
    lo, hi = parse_ci(sub["Accuracy 95% CI"])
    lower = (values - lo).fillna(0).clip(lower=0)
    upper = (hi - values).fillna(0).clip(lower=0)
    ax.bar(
        x + i * width, values, width,
        yerr=np.vstack([lower, upper]), capsize=3, label=system
    )
ax.set_xticks(x + width * (len(systems) - 1) / 2)
ax.set_xticklabels(categories, rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Answer Accuracy")
ax.set_title("Độ chính xác theo nhóm câu hỏi (kèm 95% bootstrap CI)", fontsize=13, fontweight="bold")
ax.legend(title="System")
plt.tight_layout()
plt.savefig(out_dir / "accuracy_ci_by_category.png", dpi=300)
plt.show()

# 2. Faithfulness theo Category
plt.figure(figsize=(11, 5))
sns.barplot(data=cat_df, x="Category", y="Faithfulness", hue="System", palette="mako")
plt.title("Độ trung thực / Chống ảo giác (Faithfulness)", fontsize=12, fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(out_dir / "faithfulness_by_category.png", dpi=300)
plt.show()

# 3. Accuracy tổng thể, kèm 95% CI
summary_df["Accuracy_Num"] = pd.to_numeric(summary_df["Answer Accuracy"], errors="coerce")
lo, hi = parse_ci(summary_df["Accuracy 95% CI"])
lower = (summary_df["Accuracy_Num"] - lo).fillna(0).clip(lower=0)
upper = (hi - summary_df["Accuracy_Num"]).fillna(0).clip(lower=0)
colors = {"vanilla_llm": "#e74c3c", "vector_rag": "#3498db", "graph_rag": "#2ecc71"}
plt.figure(figsize=(8, 5))
plt.bar(
    summary_df["System"], summary_df["Accuracy_Num"],
    yerr=np.vstack([lower, upper]), capsize=4,
    color=[colors.get(s, "gray") for s in summary_df["System"]]
)
plt.ylim(0, 1.05)
plt.ylabel("Answer Accuracy")
plt.title("Độ chính xác tổng thể (kèm 95% bootstrap CI)", fontsize=13, fontweight="bold")
plt.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(out_dir / "overall_accuracy_ci.png", dpi=300)
plt.show()

# 4. Scatter: Accuracy vs Total Latency Trade-off
plt.figure(figsize=(8, 6))
summary_df["Latency_Num"] = pd.to_numeric(summary_df["Latency Median (s)"], errors="coerce")
for _, row in summary_df.iterrows():
    sys_name = row["System"]
    plt.scatter(
        row["Latency_Num"], row["Accuracy_Num"],
        s=300, color=colors.get(sys_name, "gray"),
        label=sys_name, edgecolors="black", linewidth=1.5
    )
    plt.annotate(
        sys_name, (row["Latency_Num"], row["Accuracy_Num"]),
        textcoords="offset points", xytext=(0, 15), ha="center", fontweight="bold"
    )

plt.xlabel("Độ trễ trung vị (giây)", fontsize=11)
plt.ylabel("Độ chính xác câu trả lời (Answer Accuracy)", fontsize=11)
plt.title("Đánh đổi giữa Độ chính xác và Độ trễ (Accuracy vs. Latency Trade-off)", fontsize=13, fontweight="bold")
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(out_dir / "tradeoff_accuracy_latency.png", dpi=300)
plt.show()

# 5. Phân loại lỗi multi-hop
error_path = out_dir / "error_summary.csv"
if error_path.exists():
    err = pd.read_csv(error_path).set_index("system").drop(columns=["Total"], errors="ignore")
    err = err.reindex(
        columns=["correct", "lazy_refusal", "retrieval_miss", "partial_answer", "wrong_answer", "unjudged"],
        fill_value=0,
    )
    err.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="Set2")
    plt.title("Phân loại lỗi câu hỏi multi-hop theo hệ thống", fontsize=13, fontweight="bold")
    plt.ylabel("Số câu")
    plt.xticks(rotation=0)
    plt.legend(title="Error type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "error_taxonomy_multihop.png", dpi=300, bbox_inches="tight")
    plt.show()

# 6. Bảng kiểm định thống kê, độ tin cậy judge và factual guard
for name in [
    "judge_reliability.csv",
    "factual_guard_summary.csv",
    "significance_tests.csv",
    "significance_by_category.csv",
    "error_cases.csv",
]:
    path = out_dir / name
    if path.exists():
        print(f"\\n=== {name} ===")
        table = pd.read_csv(path)
        display(table.head(10) if name == "error_cases.csv" else table)

# 7. In bảng LaTeX hoàn chỉnh
print("\\n=== MÃ LATEX BẢNG KẾT QUẢ ĐỂ DÁN VÀO BÁO CÁO ===")
latex_content = (out_dir / "summary_table.tex").read_text(encoding="utf-8")
print(latex_content)
""")

# --- EXPORT NOTEBOOK ---
output_path = Path("RAG_QA_Benchmark_Colab.ipynb")
output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Master notebook generated successfully: {output_path}")
