"""
Calibration metrics and curve computation for binary clinical prediction models.

Calibration asks: "when the model says 20% risk, do 20% of those patients actually
have the event?" This is distinct from — and just as important as — discrimination
(AUROC), yet it's frequently omitted from clinical ML evaluations.
"""

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression


def _validate(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true and y_prob must be the same shape, got {y_true.shape} and {y_prob.shape}")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must be binary (0/1)")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob must contain probabilities in [0, 1]")
    return y_true, y_prob


def calibration_curve_data(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile"):
    """
    Bin predictions and compute observed event rate per bin.

    strategy: 'quantile' (equal-count bins) or 'uniform' (equal-width bins over [0, 1]).

    Returns a dict with per-bin: mean predicted probability, observed fraction of
    positives, bin count, and a Wilson-score 95% CI on the observed fraction.
    """
    y_true, y_prob = _validate(y_true, y_prob)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:  # degenerate case: too many tied values, fall back to uniform
            edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = np.linspace(0, 1, n_bins + 1)

    bin_idx = np.digitize(y_prob, edges[1:-1], right=True)

    mean_pred, frac_pos, counts, ci_low, ci_high = [], [], [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = mask.sum()
        if n == 0:
            continue
        p_mean = y_prob[mask].mean()
        n_pos = y_true[mask].sum()
        frac = n_pos / n
        lo, hi = _wilson_ci(n_pos, n)

        mean_pred.append(p_mean)
        frac_pos.append(frac)
        counts.append(int(n))
        ci_low.append(lo)
        ci_high.append(hi)

    return {
        "mean_predicted": np.array(mean_pred),
        "observed_fraction": np.array(frac_pos),
        "bin_counts": np.array(counts),
        "ci_low": np.array(ci_low),
        "ci_high": np.array(ci_high),
        "bin_edges": edges,
    }


def _wilson_ci(n_pos: float, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion — better behaved than normal approx at extremes."""
    if n == 0:
        return (np.nan, np.nan)
    p = n_pos / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def brier_score(y_true, y_prob) -> float:
    """Mean squared error between predicted probability and outcome. Lower is better; 0 is perfect."""
    y_true, y_prob = _validate(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_slope_intercept(y_true, y_prob):
    """
    Fit outcome ~ logit(p_model) via logistic regression.

    A perfectly calibrated model gives slope=1, intercept=0.
    Slope < 1 indicates predictions are too extreme (overconfident);
    slope > 1 indicates predictions are too conservative (underconfident).
    Intercept != 0 indicates systematic over/under-estimation of overall risk.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)

    model = LogisticRegression(C=np.inf)
    with warnings.catch_warnings():
        # sklearn currently warns during its penalty-param deprecation transition
        # even though C=inf correctly yields an unpenalized fit; safe to ignore here.
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(logit_p, y_true)

    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return {"slope": slope, "intercept": intercept}


def expected_calibration_error(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> float:
    """
    ECE: the weighted average absolute gap between predicted and observed probability
    across bins, weighted by bin size. Lower is better; 0 is perfect calibration.
    """
    data = calibration_curve_data(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    total_n = data["bin_counts"].sum()
    if total_n == 0:
        return float("nan")
    gaps = np.abs(data["observed_fraction"] - data["mean_predicted"])
    weights = data["bin_counts"] / total_n
    return float(np.sum(gaps * weights))


def calibration_summary(y_true, y_prob, n_bins: int = 10) -> dict:
    """Convenience wrapper bundling the headline calibration numbers for one model."""
    slope_int = calibration_slope_intercept(y_true, y_prob)
    return {
        "brier_score": brier_score(y_true, y_prob),
        "calibration_slope": slope_int["slope"],
        "calibration_intercept": slope_int["intercept"],
        "expected_calibration_error": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }
