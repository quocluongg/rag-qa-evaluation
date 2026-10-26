"""
testset_utils.py — Shared helpers for parsing and repairing benchmark testset rows.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and value != value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def gold_docs_for_row(row: Mapping[str, Any], source_map: Any) -> list[str]:
    docs = split_ids(row.get("gold_doc_ids"))
    if docs:
        return docs

    text_unit_ids = split_ids(row.get("gold_tu_ids"))
    if not text_unit_ids or text_unit_ids == ["NONE"]:
        return []
    return source_map.tu_ids_to_filenames(text_unit_ids)


def repair_gold_doc_ids(rows: Iterable[Mapping[str, Any]], source_map: Any) -> int:
    repaired = 0
    for row in rows:
        if split_ids(row.get("gold_doc_ids")):
            continue
        mapped = gold_docs_for_row(row, source_map)
        if mapped:
            row["gold_doc_ids"] = "|".join(mapped)  # type: ignore[index]
            repaired += 1
    return repaired
