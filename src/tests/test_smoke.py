import sys
from pathlib import Path

# Make the src package importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_source_map_and_metrics():
    from retrieval.source_map import SourceMap
    sm = SourceMap()
    assert len(sm.tu_df) == 376
    assert len(sm.docs_df) == 29

    fname = sm.tu_to_filename(sm.tu_df['id'].iloc[0])
    assert fname and fname.endswith('.md'), fname
    valid = sm.valid_filenames_at('01/09/2025')
    assert len(valid) > 0

    from eval.metrics_custom import (
        precision_at_k, recall_at_k, mrr, temporal_accuracy, compute_all_metrics,
    )
    retrieved = [
        'CTDT_AIoT_2025.md', 'CTDT_ATTT_2025.md', 'CTDT_CNTT_2026.md',
        'CTDT_HKVT_2026.md', 'CTDT_KTDT_2025.md',
    ]
    gold = ['CTDT_AIoT_2025.md', 'CTDT_CNTT_2026.md']
    assert abs(precision_at_k(retrieved, gold, 5) - 0.4) < 0.001
    assert abs(recall_at_k(retrieved, gold, 5) - 1.0) < 0.001
    assert abs(mrr(retrieved, gold) - 1.0) < 0.001
    ta = temporal_accuracy(retrieved, gold, '01/09/2025', 5)
    assert 0.0 <= ta <= 1.0
    assert temporal_accuracy(retrieved, gold, None, 5) is None

    agg = compute_all_metrics([
        {
            'retrieved_doc_ids': retrieved, 'gold_doc_ids': gold,
            'tq': '01/09/2025', 'category': 'factual',
            'latency_s': 0.5, 'ctx_tokens': 300, 'answer_accuracy': 0.8,
        },
        {
            'retrieved_doc_ids': ['CTDT_ATTT_2025.md'],
            'gold_doc_ids': ['CTDT_ATTT_2025.md'], 'tq': None,
            'category': 'relational', 'latency_s': 0.6,
            'ctx_tokens': 150, 'answer_accuracy': 0.6,
        },
    ], k=5)
    assert set(agg['per_category'].keys()) == {'factual', 'relational'}
    assert abs(agg['answer_accuracy'] - 0.7) < 0.001
    assert agg['per_category']['relational']['temporal_accuracy'] is None
    row = __import__('eval.metrics_custom', fromlist=['results_to_latex_row']).results_to_latex_row('Vector RAG', agg)
    assert '0.700' in row
    assert '--' not in row
