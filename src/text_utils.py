"""Small text helpers shared by the runner, judge and error analysis."""
from __future__ import annotations

import re
import unicodedata

REFUSAL_PATTERNS = (
    "không có đủ thông tin",
    "không cung cấp thông tin",
    "không có thông tin",
    "không tìm thấy thông tin",
    "không thể trả lời",
    "không biết",
    "không xác định được",
    "tài liệu được cung cấp không",
    "ngữ cảnh được cung cấp không",
    "i don't know",
    "insufficient information",
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", text).strip().lower()


def looks_like_refusal(answer: str) -> bool:
    """Heuristic: does the answer mainly refuse to answer?"""
    text = normalize_text(answer)
    if not text:
        return True
    if any(normalize_text(p) in text for p in REFUSAL_PATTERNS):
        return True
    return len(text) < 20 and ("khong" in text or "no information" in text)


def extract_codes_and_numbers(text: str) -> list[str]:
    """Return normalised codes / numeric tokens that must match a gold answer."""
    raw = str(text)
    tokens: list[str] = []

    codes = re.findall(r"\b[A-Z]{2,}\s?\d{3,}\b", raw.upper())
    tokens.extend(code.replace(" ", "") for code in codes)

    dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", raw)
    tokens.extend(dates)
    raw = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", " ", raw)

    dotted = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){2,}\b", raw)
    tokens.extend(dotted)
    raw = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){2,}\b", " ", raw)

    fractions = re.findall(r"\b\d{1,3}[.,]\d{1,3}/\d{1,3}\b", raw)
    tokens.extend(fractions)
    raw = re.sub(r"\b\d{1,3}[.,]\d{1,3}/\d{1,3}\b", " ", raw)

    tokens.extend(re.findall(r"\b\d{1,3}[.,]\d{1,3}\b", raw))
    tokens.extend(re.findall(r"\b\d+\s*tín\s*chỉ\b", text, flags=re.I))
    return list(dict.fromkeys(tokens))


def answer_contains_token(answer: str, token: str) -> bool:
    answer_norm = re.sub(r"[*_`~]", "", normalize_text(answer))
    token_norm = re.sub(r"[*_`~]", "", normalize_text(token))
    answer_norm = answer_norm.replace(" ", "").replace(",", ".").strip(".;:!?")
    token_norm = token_norm.replace(" ", "").replace(",", ".").strip(".;:!?")

    def variants(value: str) -> set[str]:
        out = {value}
        if re.search(r"\d+(?:\.\d+){2,}", value):
            out.add(value.replace(".", ""))
        return out

    return any(token in answer for token in variants(token_norm) for answer in variants(answer_norm))
