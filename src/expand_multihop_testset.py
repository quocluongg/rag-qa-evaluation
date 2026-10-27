"""
expand_multihop_testset.py — Deterministically expand the multi-hop testset.

The original 50-question set contains only 13 multi-hop questions, which is too
small for stable per-category significance testing.  This script mines course
table rows directly from the indexed text units and builds additional
cross-program questions whose answers require combining two source documents.

Every generated row is verifiable: both gold text units and both gold documents
are taken from the exact rows that contain the course code, name and credits.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "scientific_testset_50.csv"
DEFAULT_SOURCE_RAW = ROOT / "data" / "scientific_testset_50.raw.jsonl"
DEFAULT_OUT = ROOT / "data" / "scientific_testset_v2.csv"
DEFAULT_OUT_RAW = ROOT / "data" / "scientific_testset_v2.raw.jsonl"
TARGET_MULTIHOP = 30
N_NEW = 17

PROGRAM_NAMES = {
    "CTDT_AIoT_2025.md": "Trí tuệ nhân tạo vạn vật",
    "CTDT_ATTT_2025.md": "An toàn thông tin",
    "CTDT_CNBD_2025.md": "Công nghệ vi mạch bán dẫn",
    "CTDT_CNTT_2026.md": "Công nghệ thông tin",
    "CTDT_HKVT_2026.md": "Kỹ thuật Truyền thông hàng không vũ trụ",
    "CTDT_KTDT_2025.md": "Công nghệ Kỹ thuật Điện - Điện tử",
    "CTDT_KTVT_2025.md": "Kỹ thuật Điện tử viễn thông",
    "CTDT_TTDPT_2025.md": "Truyền thông đa phương tiện",
    "CTDT_UAV_2026.md": "UAV và Robot di động tự hành",
}

CODE_RE = re.compile(r"[A-Z]{2,}\s*\d{4,}")
GENERIC_NAME_PARTS = (
    "thực tập",
    "đồ án",
    "khóa luận",
    "tốt nghiệp",
    "giáo dục thể chất",
    "tự chọn",
    "chuyên ngành",
    "học phần thay thế",
    "hàng hải",
)

FIELDS = [
    "qid",
    "category",
    "question",
    "reference_answer",
    "gold_doc_ids",
    "gold_tu_ids",
    "evidence_snippet",
    "reasoning_summary",
]

# Questions that were ambiguous in the original 50-question set because they did
# not name the programme whose table/document contains the gold answer.
V2_OVERRIDES = {
    "Q002": {
        "question": (
            "Trong chương trình Công nghệ vi mạch bán dẫn, học phần nào có số thứ tự 14 "
            "trong bảng kế hoạch học tập?"
        ),
    },
    "Q003": {
        "question": (
            "Trong chương trình An toàn thông tin, học phần Giải tích 1 có mã học phần là gì "
            "và gồm bao nhiêu tín chỉ?"
        ),
    },
    "Q005": {
        "question": (
            "Trong chương trình Công nghệ Kỹ thuật Điện - Điện tử, tổng số tín chỉ của khối "
            "kiến thức giáo dục đại cương là bao nhiêu?"
        ),
    },
    "Q007": {
        "question": (
            "Trong chương trình Công nghệ thông tin, mã môn học của môn An toàn bảo mật "
            "hệ thống thông tin là gì?"
        ),
    },
    "Q009": {
        "question": (
            "Trong thông báo tuyển sinh thạc sĩ năm 2024, mã số chuyên ngành Khoa học máy tính "
            "là gì?"
        ),
    },
    "Q011": {
        "question": (
            "Trong chương trình Công nghệ vi mạch bán dẫn, mã môn học của môn Thiết kế logic số "
            "là gì?"
        ),
    },
    "Q013": {
        "question": (
            "Trong chương trình Truyền thông đa phương tiện, học phần Tư tưởng Hồ Chí Minh "
            "thuộc khối kiến thức nào?"
        ),
    },
}


def clean_name(name: str) -> str:
    name = re.sub(r"<br\s*/?>.*", "", name, flags=re.I)
    name = re.sub(r"\*\*", "", name)
    name = re.sub(r"^\s*\d+\.\s*", "", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", name, flags=re.I)
    name = re.sub(r"\s+(từ|và|của|cho|trong|với|về)$", "", name, flags=re.I)
    name = name.replace("hàng hải không vũ trụ", "hàng không vũ trụ")
    return name.strip(" *_|")


def _names_too_similar(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"\b[1-4]\b", "", s.lower()).strip()
    na, nb = norm(a), norm(b)
    return na == nb or na.startswith(nb) or nb.startswith(na)


def parse_course_rows(tu_df: pd.DataFrame, tu_to_file: dict[str, str]) -> list[dict[str, Any]]:
    """Extract one row per (document, course code) from table-formatted text units."""
    rows: list[dict[str, Any]] = []
    for _, row in tu_df.iterrows():
        filename = tu_to_file.get(str(row["id"]))
        if filename not in PROGRAM_NAMES:
            continue
        text = str(row.get("text", ""))
        for line in text.splitlines():
            if line.count("|") < 5:
                continue
            cells = [clean_name(c) for c in line.strip().strip("|").split("|")]
            for i, cell in enumerate(cells):
                code_cell = cell.replace("*", "").strip()
                if not CODE_RE.fullmatch(code_cell):
                    continue
                if i == 0 or i + 1 >= len(cells):
                    continue
                name = cells[i - 1]
                credits = None
                for j in range(i + 1, min(i + 4, len(cells))):
                    if re.fullmatch(r"\d{1,2}", cells[j]):
                        credits = int(cells[j])
                        break
                if credits is None or not (1 <= credits <= 6):
                    continue
                if len(name) < 4 or len(name) > 80:
                    continue
                if any(part in name.lower() for part in GENERIC_NAME_PARTS):
                    continue
                if re.search(r"[<>${}]", name):
                    continue
                rows.append({
                    "file": filename,
                    "program": PROGRAM_NAMES[filename],
                    "code": re.sub(r"\s+", "", code_cell).upper(),
                    "name": name,
                    "credits": credits,
                    "tu_id": str(row["id"]),
                    "text": line.strip(),
                })
                break

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        dedup.setdefault((r["file"], r["code"]), r)
    return list(dedup.values())


def _existing_tokens(rows: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for row in rows:
        text = f"{row.get('question', '')} {row.get('reference_answer', '')}".upper()
        tokens.update(re.findall(r"[A-Z]{2,}\d{3,}", text.replace(" ", "")))
    return tokens


def select_pairs(
    courses: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    n: int = N_NEW,
    seed: int = 42,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pick cross-program course pairs without reusing a course."""
    excluded = _existing_tokens(existing_rows)
    candidates = [
        c for c in courses
        if c["code"] not in excluded
        and c["name"]
        and c["credits"] in (2, 3, 4)
        and not re.match(r"^(BAS|BSA|SKD)", c["code"])
    ]
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_file[c["file"]].append(c)

    all_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    files = sorted(by_file)
    for i, fa in enumerate(files):
        for fb in files[i + 1:]:
            for a in by_file[fa]:
                for b in by_file[fb]:
                    if a["code"] == b["code"]:
                        continue
                    if _names_too_similar(a["name"], b["name"]):
                        continue
                    if not (4 <= a["credits"] + b["credits"] <= 7):
                        continue
                    all_pairs.append((a, b))

    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_codes: set[str] = set()
    file_uses: Counter[str] = Counter()
    docpair_uses: Counter[tuple[str, str]] = Counter()
    max_file_uses = 4

    def try_add(pair: tuple[dict[str, Any], dict[str, Any]], strict: bool) -> bool:
        a, b = pair
        if a["code"] in used_codes or b["code"] in used_codes:
            return False
        if not (4 <= a["credits"] + b["credits"] <= 7):
            return False
        if strict:
            if file_uses[a["file"]] >= max_file_uses or file_uses[b["file"]] >= max_file_uses:
                return False
            if docpair_uses[tuple(sorted((a["file"], b["file"])))] >= 2:
                return False
        selected.append(pair)
        used_codes.update((a["code"], b["code"]))
        file_uses[a["file"]] += 1
        file_uses[b["file"]] += 1
        docpair_uses[tuple(sorted((a["file"], b["file"])))] += 1
        return True

    # First pass: round-robin across programmes so every programme contributes
    # at least two course facts (9 pairs × 2 slots = 18 slots).
    for i, fa in enumerate(files):
        if len(selected) >= n:
            break
        for offset in range(1, len(files)):
            fb = files[(i + offset) % len(files)]
            added = False
            for a in by_file[fa]:
                if a["code"] in used_codes:
                    continue
                for b in by_file[fb]:
                    if b["code"] in used_codes or a["code"] == b["code"]:
                        continue
                    if _names_too_similar(a["name"], b["name"]):
                        continue
                    if try_add((a, b), strict=True):
                        added = True
                        break
                if added:
                    break
            if added:
                break

    for pair in all_pairs:
        if len(selected) >= n:
            break
        try_add(pair, strict=True)

    for pair in all_pairs:
        if len(selected) >= n:
            break
        try_add(pair, strict=False)

    if len(selected) < n:
        raise RuntimeError(f"Only found {len(selected)} valid cross-program pairs, need {n}")
    return selected[:n]


def build_row(index: int, pair: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    a, b = pair
    template = index % 3
    ca, cb = a["code"], b["code"]
    na, nb = a["name"], b["name"]
    pa, pb = a["program"], b["program"]
    cra, crb = a["credits"], b["credits"]
    total = cra + crb

    if template == 0:
        question = (
            f"Môn học {na} trong chương trình {pa} và môn học {nb} "
            f"trong chương trình {pb} có tổng số tín chỉ là bao nhiêu?"
        )
        reference = (
            f"Tổng số tín chỉ của hai môn là {total} tín chỉ "
            f"({na}: {cra} tín chỉ; {nb}: {crb} tín chỉ)."
        )
    elif template == 1:
        if cra == crb:
            question = (
                f"So sánh số tín chỉ của môn {na} trong chương trình {pa} "
                f"và môn {nb} trong chương trình {pb}."
            )
            reference = (
                f"Hai môn có số tín chỉ bằng nhau: {na} và {nb} đều có {cra} tín chỉ."
            )
        else:
            more, less = (a, b) if cra > crb else (b, a)
            more_cr, less_cr = max(cra, crb), min(cra, crb)
            question = (
                f"So sánh số tín chỉ của môn {na} trong chương trình {pa} "
                f"và môn {nb} trong chương trình {pb}: môn nào nhiều tín chỉ hơn "
                f"và nhiều hơn bao nhiêu?"
            )
            reference = (
                f"Môn {more['name']} nhiều hơn môn {less['name']} "
                f"{more_cr - less_cr} tín chỉ ({more_cr} so với {less_cr} tín chỉ)."
            )
    else:
        question = (
            f"Trong chương trình {pa}, môn {na} có số tín chỉ là bao nhiêu? "
            f"Trong chương trình {pb}, môn {nb} có số tín chỉ là bao nhiêu? "
            f"Tổng số tín chỉ của hai môn là bao nhiêu?"
        )
        reference = (
            f"{na} có {cra} tín chỉ; {nb} có {crb} tín chỉ; "
            f"tổng cộng là {total} tín chỉ."
        )

    return {
        "qid": f"Q{index + 51:03d}",
        "category": "multi_hop_relational",
        "question": question,
        "reference_answer": reference,
        "gold_doc_ids": f"{a['file']}|{b['file']}",
        "gold_tu_ids": f"{a['tu_id']}|{b['tu_id']}",
        "evidence_snippet": f"{a['text']}\n{b['text']}",
        "reasoning_summary": (
            f"Tra cứu số tín chỉ của {ca} trong {a['file']} và {cb} trong {b['file']}, "
            f"sau đó cộng/so sánh hai giá trị."
        ),
    }


def load_base_rows(source: Path) -> list[dict[str, str]]:
    with source.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_expanded_rows(source: Path, target_multihop: int = TARGET_MULTIHOP) -> list[dict[str, str]]:
    from retrieval.source_map import get_source_map

    base = load_base_rows(source)
    for row in base:
        if row["qid"] in V2_OVERRIDES:
            row.update(V2_OVERRIDES[row["qid"]])
    current = sum(1 for r in base if r.get("category") == "multi_hop_relational")
    needed = max(0, target_multihop - current)
    if needed == 0:
        return base

    smap = get_source_map()
    courses = parse_course_rows(smap.tu_df, {r["id"]: smap.tu_to_filename(r["id"]) or "" for _, r in smap.tu_df.iterrows()})
    pairs = select_pairs(courses, base, n=needed)
    new_rows = [build_row(i, pair) for i, pair in enumerate(pairs)]
    return base + new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand multi-hop testset deterministically")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-out", type=Path, default=DEFAULT_OUT_RAW)
    parser.add_argument("--target-multihop", type=int, default=TARGET_MULTIHOP)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    rows = build_expanded_rows(args.source, args.target_multihop)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    with args.raw_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k, "") for k in FIELDS}, ensure_ascii=False) + "\n")

    counts = Counter(r["category"] for r in rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print("Category counts:", dict(counts))
    print("\nNew multi-hop questions:")
    for row in rows:
        if row["qid"] >= "Q051":
            print(f"- {row['qid']}: {row['question']}")
            print(f"  -> {row['reference_answer']}")


if __name__ == "__main__":
    main()
