"""
build_lancedb_index.py — Rebuild the text_unit_text LanceDB table from parquet.

Usage:
    python src/build_lancedb_index.py
    python src/build_lancedb_index.py --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval.index_utils import ensure_text_unit_index, load_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/repair the LanceDB text_unit_text index")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the table is readable")
    args = parser.parse_args()

    embedder = load_embedder(args.model)
    ensure_text_unit_index(embedder, force=args.force)


if __name__ == "__main__":
    main()
