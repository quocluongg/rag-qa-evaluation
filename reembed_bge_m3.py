import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import lancedb
import torch
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

torch.set_num_threads(4)

BASE_DIR = os.environ.get("NCKH_DIR", os.path.dirname(os.path.abspath(__file__)))
LANCEDB_DIR = os.path.join(BASE_DIR, "output", "lancedb")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
MAX_SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (lance_table, parquet_file, text_column, batch_size)
TARGETS = [
    ("community_full_content", "community_reports.parquet", "full_content", 32),
    ("entity_description", "entities.parquet", "description", 128),
    ("text_unit_text", "text_units.parquet", "text", 64),
]

DATE_INT_COLS = [
    "create_date_year", "create_date_month", "create_date_month_name",
    "create_date_day", "create_date_day_of_week", "create_date_hour", "create_date_quarter",
    "update_date_year", "update_date_month", "update_date_month_name",
    "update_date_day", "update_date_day_of_week", "update_date_hour", "update_date_quarter",
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_schema():
    fields = [
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("create_date", pa.string()),
        pa.field("update_date", pa.string()),
    ]
    for c in DATE_INT_COLS:
        fields.append(pa.field(c, pa.int64()))
    return pa.schema(fields)


def embed_texts(model, texts, batch_size):
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        precision="float32",
    )
    return np.asarray(vecs, dtype=np.float32)


def main():
    print(f"Loading embedding model {MODEL_NAME} on {DEVICE} (download ~2.3GB if not cached)...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.max_seq_length = MAX_SEQ_LEN
    print(f"  device={DEVICE}, max_seq_length={MAX_SEQ_LEN}")

    db = lancedb.connect(LANCEDB_DIR)
    schema = build_schema()

    for table_name, parquet_file, text_col, batch_size in TARGETS:
        print(f"\n=== Processing table: {table_name} (source: {parquet_file}.{text_col}) ===")
        pq = pd.read_parquet(os.path.join(OUTPUT_DIR, parquet_file))
        ids = pq["id"].astype(str).tolist()
        texts = pq[text_col].astype(str).tolist()
        print(f"  {len(ids)} rows from parquet")

        vecs = embed_texts(model, texts, batch_size)
        assert vecs.shape[1] == EMBED_DIM, f"unexpected dim {vecs.shape[1]}"

        # copy existing date metadata if the table already exists (local re-embed)
        meta = None
        try:
            old = db.open_table(table_name).to_pandas()
            meta = {row["id"]: row for _, row in old.iterrows()}
            print(f"  reused date metadata from existing table ({len(meta)} ids)")
        except Exception:
            meta = None

        create_list, update_list = [], []
        int_lists = {c: [] for c in DATE_INT_COLS}
        for pid in ids:
            if meta is not None and pid in meta:
                r = meta[pid]
                cd = r.get("create_date")
                create_list.append(str(cd) if pd.notna(cd) else now_iso())
                ud = r.get("update_date")
                update_list.append(str(ud) if pd.notna(ud) else None)
                for c in DATE_INT_COLS:
                    v = r.get(c)
                    int_lists[c].append(int(v) if pd.notna(v) else 0)
            else:
                create_list.append(now_iso())
                update_list.append(None)
                for c in DATE_INT_COLS:
                    int_lists[c].append(0)

        vec_arr = pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32(), EMBED_DIM))
        arrays = [
            pa.array(ids),
            vec_arr,
            pa.array(create_list),
            pa.array(update_list),
        ]
        for c in DATE_INT_COLS:
            arrays.append(pa.array(int_lists[c]))

        pa_tbl = pa.table(arrays, schema=schema)
        db.drop_table(table_name, ignore_missing=True)
        db.create_table(table_name, pa_tbl)
        print(f"  written {db.open_table(table_name).count_rows()} rows (dim {EMBED_DIM})")

    print("\nAll tables re-embedded with", MODEL_NAME, f"({EMBED_DIM}-dim).")


if __name__ == "__main__":
    main()
