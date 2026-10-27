import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stats_utils import bootstrap_ci, holm_correction, mcnemar_test, paired_wilcoxon


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    values = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    mean, lo, hi = bootstrap_ci(values, n_boot=500, seed=7)
    assert mean == pytest.approx(0.8)
    assert lo <= mean <= hi
    assert (mean, lo, hi) == bootstrap_ci(values, n_boot=500, seed=7)


def test_bootstrap_ci_handles_empty_and_single_values():
    assert np.isnan(bootstrap_ci([])[0])
    assert bootstrap_ci([0.5]) == (0.5, 0.5, 0.5)


def test_mcnemar_no_discordant_pairs_is_not_significant():
    result = mcnemar_test([True, False, True], [True, False, True])
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_detects_asymmetric_discordance():
    a = [True] * 10 + [False] * 10
    b = [False] * 10 + [False] * 10
    result = mcnemar_test(a, b)
    assert result["b"] == 10
    assert result["c"] == 0
    assert result["p_value"] < 0.05


def test_holm_correction_is_monotone_and_controls_familywise_error():
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_correction_preserves_nan():
    adjusted = holm_correction([0.01, float("nan"), 0.04])
    assert adjusted[0] == pytest.approx(0.02)
    assert np.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.04)


def test_wilcoxon_returns_one_for_identical_scores():
    assert paired_wilcoxon([0.5, 1.0, 0.0], [0.5, 1.0, 0.0]) == 1.0
