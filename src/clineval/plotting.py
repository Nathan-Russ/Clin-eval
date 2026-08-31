"""
Matplotlib plotting helpers for clineval.

These are intentionally simple static plots for notebooks/reports/papers.
The Streamlit demo app (demo_app/) uses Plotly separately for interactivity —
that dependency is NOT required to use the core library or these plots.
"""

import numpy as np

from . import calibration as _cal
from . import discrimination as _disc
from . import decision_curve as _dca


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install clineval[plots]"
        ) from e


def plot_calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile", ax=None, label=None):
    plt = _require_matplotlib()
    data = _cal.calibration_curve_data(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    yerr = np.vstack([
        data["observed_fraction"] - data["ci_low"],
        data["ci_high"] - data["observed_fraction"],
    ])
    ax.errorbar(data["mean_predicted"], data["observed_fraction"], yerr=yerr, marker="o", capsize=3, label=label or "Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return ax


def plot_roc_curve(y_true, y_prob, ax=None, label=None):
    plt = _require_matplotlib()
    roc_data = _disc.roc_curve_data(y_true, y_prob)
    auc = _disc.auroc(y_true, y_prob)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.plot(roc_data["fpr"], roc_data["tpr"], label=f"{label or 'Model'} (AUROC={auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_decision_curve(y_true, y_prob, thresholds=None, ax=None, label=None):
    plt = _require_matplotlib()
    df = _dca.net_benefit(y_true, y_prob, thresholds=thresholds)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    ax.plot(df["threshold"], df["net_benefit_model"], label=label or "Model")
    ax.plot(df["threshold"], df["net_benefit_treat_all"], linestyle="--", color="gray", label="Treat all")
    ax.plot(df["threshold"], df["net_benefit_treat_none"], linestyle=":", color="black", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend()
    return ax
