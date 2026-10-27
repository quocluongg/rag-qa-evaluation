import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from run_3systems import DEFAULT_K, PROMPT_VERSION, load_completed_records, prompt_rag
from text_utils import answer_contains_token, extract_codes_and_numbers, looks_like_refusal


def test_default_top_k_is_eight():
    assert DEFAULT_K == 8


def test_rag_prompt_forbids_lazy_refusal_and_allows_grounded_reasoning():
    prompt = prompt_rag("Câu hỏi hai vế?", "[1] Tài liệu: A\nNội dung A")
    assert "ĐẦY ĐỦ" in prompt
    assert "từng vế" in prompt
    assert "suy luận logic" in prompt
    assert "cộng/so sánh" in prompt
    assert "Tuyệt đối không suy diễn" not in prompt
    assert "KHÔNG chứa thông tin cho TẤT CẢ các vế" in prompt


def test_refusal_heuristic_distinguishes_answer_from_refusal():
    assert looks_like_refusal("Tài liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này.")
    assert looks_like_refusal("Tôi không biết.")
    assert not looks_like_refusal("Môn học có 3 tín chỉ và được dạy ở học kỳ 5.")


def test_factual_token_matching_normalises_spaces_and_accents():
    tokens = extract_codes_and_numbers("Môn BAS1203 có 3 tín chỉ, tỷ lệ 26,59.")
    assert "BAS1203" in tokens
    assert answer_contains_token("Mã bas1203, ba tín chỉ", "BAS1203")
    assert answer_contains_token("Tỷ lệ 26.59", "26,59")
    assert answer_contains_token("Môn có **3** tín chỉ", "3 tín chỉ")
    assert answer_contains_token("Mã 8480101", "8.48.01.01")
    assert answer_contains_token("Điểm chuẩn là **24.24/30**.", "24.24/30")
    assert not answer_contains_token("Mã ELE1426", "ELE16246")


def test_token_extraction_keeps_dotted_codes_and_cutoff_fractions_whole():
    assert extract_codes_and_numbers("Mã số 8.48.01.01.") == ["8.48.01.01"]
    assert extract_codes_and_numbers("Điểm chuẩn 24.24/30.") == ["24.24/30"]
    assert extract_codes_and_numbers("Ngày 01/01/2025, tỷ lệ 26,59.") == ["01/01/2025", "26,59"]


def test_resume_discards_stale_prompt_rows_and_dedupes(tmp_path):
    path = tmp_path / "run.jsonl"
    rows = [
        {"system": "vector_rag", "qid": "Q001", "prompt_version": "old", "top_k": 8},
        {"system": "vector_rag", "qid": "Q001", "prompt_version": PROMPT_VERSION, "top_k": 8},
        {"system": "vector_rag", "qid": "Q001", "prompt_version": PROMPT_VERSION, "top_k": 8},
        {"system": "vector_rag", "qid": "Q002", "prompt_version": PROMPT_VERSION, "top_k": 5},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    completed = load_completed_records(path, PROMPT_VERSION, 8)

    assert completed == {("vector_rag", "Q001")}
    remaining = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(remaining) == 1
    assert remaining[0]["qid"] == "Q001"
    assert path.with_suffix(".jsonl.bak").exists()
