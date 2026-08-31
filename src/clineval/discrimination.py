"""
Discrimination metrics for binary clinical prediction models: how well the model
separates people who will have the event from those who won't, independent of
whether its absolute probability estimates are well calibrated.
"""

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def _validate(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true and y_prob must be the same shape, got {y_true.shape} and {y_prob.shape}")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must be binary (0/1)")
    return y_true, y_prob


def roc_curve_data(y_true, y_prob) -> dict:
    y_true, y_prob = _validate(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def pr_curve_data(y_true, y_prob) -> dict:
    y_true, y_prob = _validate(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def auroc(y_true, y_prob) -> float:
    y_true, y_prob = _validate(y_true, y_prob)
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true, y_prob) -> float:
    y_true, y_prob = _validate(y_true, y_prob)
    return float(average_precision_score(y_true, y_prob))


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot: int = 1000, ci: float = 0.95, seed: int = None) -> dict:
    """
    Generic percentile bootstrap CI for any metric_fn(y_true, y_prob) -> float.
    Resamples patients (rows) with replacement, preserving the true prevalence's
    natural variation across resamples.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    estimates = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            estimates[i] = np.nan  # degenerate resample, e.g. all-positive or all-negative
            continue
        estimates[i] = metric_fn(yt, yp)

    estimates = estimates[~np.isnan(estimates)]
    alpha = (1 - ci) / 2
    lo = float(np.quantile(estimates, alpha))
    hi = float(np.quantile(estimates, 1 - alpha))
    point = float(metric_fn(y_true, y_prob))
    return {"estimate": point, "ci_low": lo, "ci_high": hi, "n_boot_used": len(estimates)}


def youden_optimal_threshold(y_true, y_prob) -> dict:
    """Threshold maximizing sensitivity + specificity - 1 (Youden's J statistic)."""
    roc_data = roc_curve_data(y_true, y_prob)
    j_scores = roc_data["tpr"] - roc_data["fpr"]
    best_idx = int(np.argmax(j_scores))
    return {
        "threshold": float(roc_data["thresholds"][best_idx]),
        "sensitivity": float(roc_data["tpr"][best_idx]),
        "specificity": float(1 - roc_data["fpr"][best_idx]),
        "youden_j": float(j_scores[best_idx]),
    }


def classification_metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """Sensitivity, specificity, PPV, NPV, and accuracy at a chosen probability threshold."""
    y_true, y_prob = _validate(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    accuracy = (tp + tn) / len(y_true)

    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def discrimination_summary(y_true, y_prob, bootstrap: bool = False, n_boot: int = 1000, seed: int = None) -> dict:
    """Convenience wrapper bundling headline discrimination numbers for one model."""
    if bootstrap:
        auc_result = bootstrap_ci(y_true, y_prob, auroc, n_boot=n_boot, seed=seed)
        ap_result = bootstrap_ci(y_true, y_prob, auprc, n_boot=n_boot, seed=seed)
        return {
            "auroc": auc_result["estimate"], "auroc_ci": (auc_result["ci_low"], auc_result["ci_high"]),
            "auprc": ap_result["estimate"], "auprc_ci": (ap_result["ci_low"], ap_result["ci_high"]),
        }
    return {"auroc": auroc(y_true, y_prob), "auprc": auprc(y_true, y_prob)}
