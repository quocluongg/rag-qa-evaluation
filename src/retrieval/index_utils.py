"""
index_utils.py — Build and repair the LanceDB vector index from local parquet files.
"""
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
LANCEDB_DIR = ROOT / "output" / "lancedb"
TEXT_UNITS_PARQUET = ROOT / "output" / "text_units.parquet"
TEXT_UNIT_TABLE = "text_unit_text"
DEFAULT_MAX_SEQ_LENGTH = 512


@lru_cache(maxsize=4)
def load_embedder(model_name: str, device: Optional[str] = None) -> Any:
    from sentence_transformers import SentenceTransformer

    resolved_device = device or os.environ.get("EMBED_DEVICE") or None
    if resolved_device in {"", "auto"}:
        resolved_device = None
    model = SentenceTransformer(model_name, device=resolved_device)
    max_seq_length = int(os.environ.get("EMBED_MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH))
    model.max_seq_length = max_seq_length
    print(
        f"[embed] {model_name} | device={model.device} | "
        f"max_seq_length={model.max_seq_length}"
    )
    return model


def encode_texts(embedder: Any, texts: Any, **kwargs: Any) -> Any:
    """Encode texts, retrying on CPU when the GPU runs out of memory."""
    try:
        return embedder.encode(texts, **kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message and "oom" not in message:
            raise
        print("[embed] CUDA out of memory; retrying embedding on CPU.")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        to_device = getattr(embedder, "to", None)
        if callable(to_device):
            try:
                to_device("cpu")
            except Exception:
                pass
        kwargs.pop("device", None)
        kwargs["batch_size"] = min(int(kwargs.get("batch_size", 16) or 16), 16)
        return embedder.encode(texts, **kwargs)


def table_is_readable(
    db: Any,
    table_name: str,
    expected_dim: Optional[int] = None,
    require_text: bool = True,
) -> bool:
    try:
        table = db.open_table(table_name)
        table.to_pandas()
        fields = {field.name: field.type for field in table.schema}
        if require_text and "text" not in fields:
            return False
        if expected_dim is not None:
            vector_type = fields.get("vector")
            if getattr(vector_type, "list_size", None) != expected_dim:
                return False
        return True
    except Exception as exc:
        print(f"[index] LanceDB table '{table_name}' is not readable: {exc}")
        return False


def _embedder_dim(embedder: Any) -> Optional[int]:
    get_dim = getattr(embedder, "get_embedding_dimension", None)
    if not callable(get_dim):
        get_dim = getattr(embedder, "get_sentence_embedding_dimension", None)
    if callable(get_dim):
        try:
            return int(get_dim())
        except Exception:
            return None
    return getattr(embedder, "dim", None)


def build_text_unit_table(
    embedder: Any,
    db_dir: Optional[Path] = None,
    table_name: str = TEXT_UNIT_TABLE,
    parquet_path: Optional[Path] = None,
    text_col: str = "text",
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> Any:
    import lancedb

    db_path = Path(db_dir or LANCEDB_DIR)
    parquet_file = Path(parquet_path or TEXT_UNITS_PARQUET)
    db_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_file)
    ids = df["id"].astype(str).tolist()
    texts = df[text_col].fillna("").astype(str).tolist()

    print(f"[index] Embedding {len(texts)} rows for '{table_name}' from {parquet_file.name}...")
    vectors = encode_texts(
        embedder,
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")

    dim = int(vectors.shape[1])
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("text", pa.string()),
        ]
    )
    arrow_table = pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "vector": pa.array(
                [vector.tolist() for vector in vectors],
                type=pa.list_(pa.float32(), dim),
            ),
            "text": pa.array(texts, type=pa.string()),
        },
        schema=schema,
    )

    db = lancedb.connect(str(db_path))
    try:
        db.drop_table(table_name, ignore_missing=True)
    except Exception:
        shutil.rmtree(db_path / f"{table_name}.lance", ignore_errors=True)
        db = lancedb.connect(str(db_path))

    db.create_table(table_name, arrow_table)
    table = db.open_table(table_name)
    table.to_pandas()
    print(f"[index] Rebuilt '{table_name}': {table.count_rows()} rows, {dim} dimensions.")
    return table


def ensure_text_unit_index(
    embedder: Any,
    db_dir: Optional[Path] = None,
    table_name: str = TEXT_UNIT_TABLE,
    parquet_path: Optional[Path] = None,
    force: bool = False,
) -> Any:
    import lancedb

    db_path = Path(db_dir or LANCEDB_DIR)
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))
    expected_dim = _embedder_dim(embedder)

    if not force and table_name in db.table_names():
        if table_is_readable(db, table_name, expected_dim=expected_dim):
            print(f"[index] LanceDB table '{table_name}' is ready.")
            return db.open_table(table_name)

    print(f"[index] LanceDB table '{table_name}' is missing or corrupt; rebuilding from parquet.")
    return build_text_unit_table(
        embedder,
        db_dir=db_path,
        table_name=table_name,
        parquet_path=parquet_path,
    )
