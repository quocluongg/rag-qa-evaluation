"""
verify_testset.py — Export raw testset CSV for human review, then import results.

Usage:
  # Step 1: export CSV for review
  python src/verify_testset.py export --input data/testset_raw.jsonl

  # Step 2: reviewer fills in:
  #   - reviewer_assign_category (factual/temporal/relational/multi_hop/historical)
  #   - tq  (dd/mm/yyyy, required for temporal/historical)
  #   - gold_doc_ids  (pipe-separated filenames, e.g. CTDT_AIoT_2025.md|CTDT_ATTT_2025.md)
  #   - reviewer_status  (approved / rejected)
  #   - reviewer_note  (optional)

  # Step 3: import reviewed CSV and freeze 50 queries (10 per category)
  python src/verify_testset.py freeze --reviewed data/testset_review.csv
"""
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import csv
import json
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CATEGORIES = ["factual", "temporal", "relational", "multi_hop", "historical"]
TARGET_PER_CAT = 10          # 10 × 5 = 50 frozen queries
FROZEN_OUTPUT  = DATA_DIR / "testset_v1.csv"

# ── Args ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd")

exp_p = sub.add_parser("export", help="Export raw JSONL to CSV for human review")
exp_p.add_argument("--input",  default=str(DATA_DIR / "testset_raw.jsonl"))
exp_p.add_argument("--output", default=str(DATA_DIR / "testset_review.csv"))

frz_p = sub.add_parser("freeze", help="Import reviewed CSV and freeze 50 queries")
frz_p.add_argument("--reviewed", default=str(DATA_DIR / "testset_review.csv"))
frz_p.add_argument("--output",   default=str(FROZEN_OUTPUT))
frz_p.add_argument("--per-cat",  type=int, default=TARGET_PER_CAT)

aug_p = sub.add_parser("add-augmented",
    help="Merge augment_temporal.py output into review CSV before freezing")
aug_p.add_argument("--augmented", default=str(DATA_DIR / "testset_augmented.jsonl"))
aug_p.add_argument("--review-csv", default=str(DATA_DIR / "testset_review.csv"))
aug_p.add_argument("--output",     default=str(DATA_DIR / "testset_review.csv"))

args = parser.parse_args()

# ── EXPORT ────────────────────────────────────────────────────
if args.cmd == "export":
    in_path  = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    records = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # CSV columns for reviewer
    REVIEW_COLS = [
        "qid",
        "ragas_synthesizer",       # from RAGAS (read-only context)
        "user_input",              # from RAGAS (read-only context)
        "reference_answer",        # from RAGAS (read-only context)
        "reference_contexts_0",    # first context snippet (read-only)
        # ── reviewer fills below ──
        "reviewer_assign_category",  # factual|temporal|relational|multi_hop|historical
        "tq",                        # dd/mm/yyyy — REQUIRED for temporal/historical
        "gold_doc_ids",              # pipe-separated filenames
        "reviewer_status",           # approved | rejected
        "reviewer_note",             # free text
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            ctx = r.get("reference_contexts", [])
            ctx0 = ctx[0][:200] if ctx else ""
            writer.writerow({
                "qid":                    r.get("qid", ""),
                "ragas_synthesizer":      r.get("synthesizer_name", ""),
                "user_input":             r.get("user_input", r.get("question", "")),
                "reference_answer":       r.get("reference_answer", ""),
                "reference_contexts_0":   ctx0,
                "reviewer_assign_category": "",
                "tq":                     "",
                "gold_doc_ids":           "",
                "reviewer_status":        "pending",
                "reviewer_note":          "",
            })

    print(f"Exported {len(records)} rows → {out_path}")
    print("Fill in: reviewer_assign_category, tq, gold_doc_ids, reviewer_status")

# ── ADD AUGMENTED ──────────────────────────────────────────────
elif args.cmd == "add-augmented":
    aug_path = Path(args.augmented)
    csv_path = Path(args.review_csv)
    out_path = Path(args.output)

    if not aug_path.exists():
        raise FileNotFoundError(f"Augmented JSONL not found: {aug_path}")

    aug_records = [
        json.loads(l) for l in aug_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    # Read existing CSV
    existing = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            fieldnames = reader.fieldnames
    else:
        fieldnames = None

    # Upsert augmented rows by qid so rerunning the command is idempotent.
    by_qid = {row.get("qid", ""): row for row in existing if row.get("qid")}
    added = 0
    for aug in aug_records:
        if not fieldnames:
            fieldnames = list(aug.keys())
        row = {
            "qid":                    aug.get("qid", ""),
            "ragas_synthesizer":      aug.get("synthesizer_name", "augment_temporal"),
            "user_input":             aug.get("user_input", aug.get("question", "")),
            "reference_answer":       aug.get("reference_answer", ""),
            "reference_contexts_0":   (aug.get("reference_contexts", [""])[0])[:200],
            "reviewer_assign_category": aug.get("category", ""),
            "tq":                     aug.get("tq", ""),
            "gold_doc_ids":           "|".join(aug.get("gold_doc_ids", [])),
            "reviewer_status":        "pending",
            "reviewer_note":          aug.get("note", "auto-generated"),
        }
        if row["qid"] not in by_qid:
            added += 1
        by_qid[row["qid"]] = row
    existing = list(by_qid.values())

    REVIEW_COLS = [
        "qid", "ragas_synthesizer", "user_input", "reference_answer",
        "reference_contexts_0", "reviewer_assign_category", "tq",
        "gold_doc_ids", "reviewer_status", "reviewer_note",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)

    print(f"Upserted {len(aug_records)} augmented rows ({added} new) → {out_path} (total {len(existing)})")

# ── FREEZE ────────────────────────────────────────────────────
elif args.cmd == "freeze":
    csv_path = Path(args.reviewed)
    out_path = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(f"Reviewed CSV not found: {csv_path}")

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    approved = [r for r in rows if r.get("reviewer_status", "").strip().lower() == "approved"]
    print(f"Total approved: {len(approved)} / {len(rows)}")

    # Validate required fields
    issues = []
    for r in approved:
        cat = r.get("reviewer_assign_category", "").strip()
        if cat not in CATEGORIES:
            issues.append(f"{r['qid']}: invalid category '{cat}'")
        if cat in ("temporal", "historical") and not r.get("tq", "").strip():
            issues.append(f"{r['qid']}: tq required for {cat}")
        if not r.get("gold_doc_ids", "").strip():
            issues.append(f"{r['qid']}: gold_doc_ids is empty")
    if issues:
        print("\n[VALIDATION ERRORS — fix before freeze:]")
        for iss in issues:
            print(f"  {iss}")
        raise ValueError(f"{len(issues)} validation error(s). Fix CSV and re-run.")

    # Select 10 per category (first-come, first-served from approved)
    selected: list[dict] = []
    cat_counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    for r in approved:
        cat = r["reviewer_assign_category"].strip()
        if cat_counts.get(cat, 0) < args.per_cat:
            selected.append(r)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("\nSelected per category:")
    for cat in CATEGORIES:
        n = cat_counts.get(cat, 0)
        status = "OK" if n == args.per_cat else f"WARN: only {n}/{args.per_cat}"
        print(f"  {cat:<12}: {n}  [{status}]")
    print(f"Total frozen: {len(selected)}")
    expected = args.per_cat * len(CATEGORIES)
    if len(selected) != expected or any(cat_counts[c] != args.per_cat for c in CATEGORIES):
        raise ValueError(f"Cannot freeze: need exactly {args.per_cat} approved rows in each category ({expected} total).")

    # Write frozen testset — canonical schema
    FROZEN_COLS = [
        "qid", "category", "user_input", "tq",
        "gold_doc_ids", "reference_answer", "reference_contexts",
        "synthesizer_name", "reviewer_status",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FROZEN_COLS, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(selected):
            writer.writerow({
                "qid":               f"q{i:03d}",
                "category":          r["reviewer_assign_category"].strip(),
                "user_input":        r["user_input"],
                "tq":                r.get("tq", "").strip(),
                "gold_doc_ids":      r.get("gold_doc_ids", "").strip(),
                "reference_answer":  r.get("reference_answer", ""),
                "reference_contexts": r.get("reference_contexts_0", ""),
                "synthesizer_name":  r.get("ragas_synthesizer", ""),
                "reviewer_status":   "approved",
            })

    print(f"\nFrozen testset saved → {out_path}")
    print("IMPORTANT: do not modify testset_v1.csv after this point.")
    print("Compute SHA-256 and record in config_manifest.json.")

    # Print SHA-256 of frozen file
    import hashlib
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"SHA-256: {sha}")

else:
    parser.print_help()
