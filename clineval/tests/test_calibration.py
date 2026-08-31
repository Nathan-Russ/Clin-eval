import numpy as np
import pytest

from clineval import calibration as cal


def test_perfect_calibration_gives_slope_1_intercept_0():
    rng = np.random.default_rng(0)
    n = 20000
    true_p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, true_p)

    result = cal.calibration_slope_intercept(y, true_p)
    assert result["slope"] == pytest.approx(1.0, abs=0.1)
    assert result["intercept"] == pytest.approx(0.0, abs=0.1)


def test_overconfident_model_has_slope_below_1():
    rng = np.random.default_rng(1)
    n = 20000
    true_p = rng.uniform(0.1, 0.9, n)
    y = rng.binomial(1, true_p)

    # push predictions toward the extremes -> overconfident
    overconfident_p = np.clip(0.5 + (true_p - 0.5) * 2.5, 0.001, 0.999)

    result = cal.calibration_slope_intercept(y, overconfident_p)
    assert result["slope"] < 1.0


def test_brier_score_is_zero_for_perfect_predictions():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert cal.brier_score(y, p) == 0.0


def test_brier_score_matches_manual_calculation():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    expected = np.mean((p - y) ** 2)
    assert cal.brier_score(y, p) == pytest.approx(expected)


def test_calibration_curve_data_bin_counts_sum_to_n():
    rng = np.random.default_rng(2)
    n = 500
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)

    data = cal.calibration_curve_data(y, p, n_bins=10)
    assert data["bin_counts"].sum() == n


def test_expected_calibration_error_zero_for_perfect_calibration():
    rng = np.random.default_rng(3)
    n = 50000
    p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p)

    ece = cal.expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.02  # should be close to 0 with enough samples


def test_validate_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        cal.brier_score([0, 1, 2], [0.1, 0.5, 0.9])


def test_validate_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError):
        cal.brier_score([0, 1], [0.1, 1.5])
