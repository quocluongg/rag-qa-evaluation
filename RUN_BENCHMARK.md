# Running the 90-candidate / 50-query benchmark

The testset pipeline uses the stable Google API model `gemini-3.5-flash-lite`. It uses a source-grounded direct prompt rather than the request-heavy RAGAS testset synthesizer. The direct generator creates 90 candidates, exactly 18 per category: factual, temporal, relational, multi-hop, and historical. After review, freeze exactly 50 rows, 10 per category.

The generator runs in 9 batches of 10. Each question has at most 5 total API attempts, so the hard request budget is 90 x 5 = 450. Checkpoints and a usage file are written after each question; `--resume` continues without regenerating completed rows. The model never invents source IDs: `gold_doc_ids` and `gold_tu_ids` are assigned from the supplied context units by the program.

## Colab setup

Open [`gen_testset_colab.ipynb`](https://colab.research.google.com/github/quocluongg/nckh-graphrag/blob/main/gen_testset_colab.ipynb), select a runtime with enough RAM, and run the install/clone/index validation cells. Store the key in Colab Secrets as `GOOGLE_API_KEY`; do not paste it into a notebook cell. The notebook is pinned to the latest repository commit.

## Pilot

Run the 10-record pilot first:

```bash
python src/gen_testset_direct.py --size 10 --n-units 20 --batch-size 10 --max-attempts 5 --max-total-requests 50 --out data/testset_pilot.jsonl
```

Confirm that the JSONL parses, the five categories are present, and the source IDs map to the index. The pilot may consume up to 50 requests in the worst case.

## Full candidate pool

Run the full 90-row pool in 9 batches:

```bash
python src/gen_testset_direct.py --size 90 --batch-size 10 --max-attempts 5 --max-total-requests 450 --resume
```

If Colab disconnects, rerun the same command with `--resume`. Do not delete `data/testset_raw.jsonl` or `data/testset_raw.usage.json` if you want to continue from the checkpoint. Stop if the usage file reaches 450 requests.

Export the review sheet:

```bash
python src/verify_testset.py export
```

Review all 90 candidates. Select exactly 10 approved rows in each category, verify the answer against `reference_contexts`, and verify `tq`, `gold_doc_ids`, and `gold_tu_ids`. Then freeze the canonical set:

```bash
python src/verify_testset.py freeze
```

The freeze command must reject any set that is not exactly 50 approved rows with the required 10/10/10/10/10 distribution.

## Benchmark and evaluation

After `data/testset_v1.csv` is frozen and the LanceDB index is available:

```bash
python src/run_experiment.py --testset data/testset_v1.csv --output results/run.jsonl --model gemini-3.5-flash-lite --k 5 --resume
python src/evaluate_ragas.py --input results/run.jsonl --output-dir results --model gemini-3.5-flash-lite --resume
```

Keep generation and judging model settings fixed across systems. Do not change the frozen testset after observing benchmark results.
