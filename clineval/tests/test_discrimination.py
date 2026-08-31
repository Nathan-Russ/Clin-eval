import numpy as np
import pytest

from clineval import discrimination as disc


def test_auroc_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert disc.auroc(y, p) == 1.0


def test_auroc_random_predictions_near_half():
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.binomial(1, 0.5, n)
    p = rng.uniform(0, 1, n)  # uncorrelated with y
    assert disc.auroc(y, p) == pytest.approx(0.5, abs=0.02)


def test_youden_threshold_on_perfect_separator():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = disc.youden_optimal_threshold(y, p)
    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(1.0)


def test_classification_metrics_at_threshold_confusion_counts():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    result = disc.classification_metrics_at_threshold(y, p, threshold=0.5)
    # predicted positive: index 0 (0.9), index 2 (0.6)
    # tp=1 (idx0), fn=1 (idx1), fp=1 (idx2), tn=1 (idx3)
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 1
    assert result["sensitivity"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.5)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(5)
    n = 500
    true_p = rng.uniform(0, 1, n)
    y = rng.binomial(1, true_p)

    result = disc.bootstrap_ci(y, true_p, disc.auroc, n_boot=200, seed=42)
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_validate_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        disc.auroc([0, 1, 1], [0.2, 0.8])
