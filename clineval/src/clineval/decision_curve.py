"""
Decision curve analysis (DCA) for binary clinical prediction models.

DCA answers a question AUROC and calibration can't: "at the probability threshold
a clinician would actually act on, does using this model lead to better decisions
than treating everyone or treating no one?"

Reference: Vickers AJ, Elkin EB. Decision curve analysis: a novel method for
evaluating prediction models. Med Decis Making. 2006.

Paired / longitudinal mode
---------------------------
The functions above give point estimates. When you need uncertainty — e.g. "is
model B's net benefit meaningfully different from model A's, on the same patients?"
— naive independent bootstrapping is wrong in two common situations:

1. Comparing two models on the *same* cohort: their prediction errors are correlated
   (both models see the same hard/easy patients), so resampling each model's
   predictions independently overstates the uncertainty in their difference.
   `compare_paired_net_benefit` fixes this by resampling patient indices once per
   bootstrap draw and applying that same resample to both models.

2. Repeated measurements per patient (longitudinal data — e.g. risk re-assessed at
   multiple visits): rows from the same patient aren't independent, so resampling
   individual rows understates the true uncertainty. `bootstrap_net_benefit_ci` and
   `compare_paired_net_benefit` both accept a `cluster_id` array (e.g. patient ID)
   to resample whole clusters together instead of individual rows.
"""

import numpy as np
import pandas as pd


def _validate(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true and y_prob must be the same shape, got {y_true.shape} and {y_prob.shape}")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must be binary (0/1)")
    return y_true, y_prob


def _clean_thresholds(thresholds):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    thresholds = np.asarray(thresholds, dtype=float)
    return thresholds[(thresholds > 0) & (thresholds < 1)]


def _net_benefit_values(y_true, y_prob, thresholds) -> np.ndarray:
    """
    Vectorized net benefit of the model across thresholds, no DataFrame overhead.
    Used directly by bootstrap loops, where this gets called hundreds/thousands of times.
    """
    n = len(y_true)
    pred_matrix = y_prob[:, None] >= thresholds[None, :]  # (n_samples, n_thresholds)
    is_pos = (y_true == 1)[:, None]
    tp = np.sum(pred_matrix & is_pos, axis=0)
    fp = np.sum(pred_matrix & ~is_pos, axis=0)
    odds = thresholds / (1 - thresholds)
    return (tp / n) - (fp / n) * odds


def net_benefit(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """
    Compute net benefit of the model, "treat all", and "treat none" strategies
    across a range of threshold probabilities.

    Net benefit at threshold pt:
        NB_model    = (TP/n) - (FP/n) * (pt / (1 - pt))
        NB_treat_all = (prevalence) - (1 - prevalence) * (pt / (1 - pt))
        NB_treat_none = 0

    thresholds: array of probabilities in (0, 1), exclusive. Defaults to 0.01..0.99 in 1% steps.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    thresholds = _clean_thresholds(thresholds)
    prevalence = y_true.mean()

    nb_model = _net_benefit_values(y_true, y_prob, thresholds)
    odds = thresholds / (1 - thresholds)
    nb_all = prevalence - (1 - prevalence) * odds

    return pd.DataFrame({
        "threshold": thresholds,
        "net_benefit_model": nb_model,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": np.zeros_like(thresholds),
    })


def decision_curve_analysis(y_true, y_prob, thresholds=None, model_name: str = "model") -> pd.DataFrame:
    """Same as net_benefit but tags the model column with a name — convenient when comparing several models."""
    df = net_benefit(y_true, y_prob, thresholds=thresholds)
    df = df.rename(columns={"net_benefit_model": f"net_benefit_{model_name}"})
    return df


def compare_models_net_benefit(y_true, prob_dict: dict, thresholds=None) -> pd.DataFrame:
    """
    Net benefit for multiple models against the same outcome, plus the shared
    treat-all / treat-none reference curves, all in one long-format DataFrame
    with columns: threshold, strategy, net_benefit.
    """
    y_true = np.asarray(y_true).astype(float)
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    rows = []
    reference_added = False
    for name, y_prob in prob_dict.items():
        nb = net_benefit(y_true, y_prob, thresholds=thresholds)
        for _, r in nb.iterrows():
            rows.append({"threshold": r["threshold"], "strategy": name, "net_benefit": r["net_benefit_model"]})
        if not reference_added:
            for _, r in nb.iterrows():
                rows.append({"threshold": r["threshold"], "strategy": "Treat all", "net_benefit": r["net_benefit_treat_all"]})
                rows.append({"threshold": r["threshold"], "strategy": "Treat none", "net_benefit": r["net_benefit_treat_none"]})
            reference_added = True

    return pd.DataFrame(rows)


def net_benefit_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """Net benefit for a single threshold of interest, alongside treat-all/treat-none for reference."""
    df = net_benefit(y_true, y_prob, thresholds=[threshold])
    row = df.iloc[0]
    return {
        "threshold": float(row["threshold"]),
        "net_benefit_model": float(row["net_benefit_model"]),
        "net_benefit_treat_all": float(row["net_benefit_treat_all"]),
        "net_benefit_treat_none": float(row["net_benefit_treat_none"]),
    }


# ---------------------------------------------------------------------------
# Paired / longitudinal mode
# ---------------------------------------------------------------------------

def _resample_indices(n_rows: int, cluster_id, rng) -> np.ndarray:
    """
    Draw one bootstrap resample of row indices.

    Without cluster_id: standard row-level bootstrap (n draws with replacement).
    With cluster_id: cluster bootstrap — resample cluster labels with replacement,
    then include every row belonging to each sampled cluster. This is the standard
    fix for repeated-measures / longitudinal data, where treating each row as an
    independent draw would understate the true uncertainty.
    """
    if cluster_id is None:
        return rng.integers(0, n_rows, n_rows)

    unique_clusters, inverse = np.unique(cluster_id, return_inverse=True)
    cluster_to_rows = [np.where(inverse == c)[0] for c in range(len(unique_clusters))]

    sampled_cluster_positions = rng.integers(0, len(unique_clusters), len(unique_clusters))
    idx = np.concatenate([cluster_to_rows[c] for c in sampled_cluster_positions])
    return idx


def bootstrap_net_benefit_ci(y_true, y_prob, thresholds=None, n_boot: int = 1000,
                              ci: float = 0.95, seed: int = None, cluster_id=None) -> pd.DataFrame:
    """
    Net benefit curve with a bootstrap confidence band.

    Pass `cluster_id` (e.g. patient ID) when rows are repeated measurements on the
    same patients (longitudinal data) — this resamples whole patients rather than
    individual rows, which is required for a valid CI in that setting.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    thresholds = _clean_thresholds(thresholds)
    n = len(y_true)

    if cluster_id is not None:
        cluster_id = np.asarray(cluster_id)
        if len(cluster_id) != n:
            raise ValueError("cluster_id must be the same length as y_true")

    rng = np.random.default_rng(seed)
    boot_nb = np.empty((n_boot, len(thresholds)))
    for b in range(n_boot):
        idx = _resample_indices(n, cluster_id, rng)
        boot_nb[b] = _net_benefit_values(y_true[idx], y_prob[idx], thresholds)

    alpha = (1 - ci) / 2
    ci_low = np.quantile(boot_nb, alpha, axis=0)
    ci_high = np.quantile(boot_nb, 1 - alpha, axis=0)

    point_df = net_benefit(y_true, y_prob, thresholds=thresholds)
    point_df["ci_low"] = ci_low
    point_df["ci_high"] = ci_high
    return point_df


def compare_paired_net_benefit(y_true, y_prob_a, y_prob_b, thresholds=None, n_boot: int = 1000,
                                ci: float = 0.95, seed: int = None, cluster_id=None,
                                name_a: str = "model_a", name_b: str = "model_b") -> pd.DataFrame:
    """
    Compare net benefit between two models evaluated on the SAME patients (paired design),
    e.g. an existing clinical model vs. a new ML model, or the same model at two timepoints.

    Because both models see the same patients, their errors are correlated — a naive
    independent bootstrap on each model separately would overstate the uncertainty in
    their difference. This function instead draws one bootstrap resample of patient
    indices per iteration and applies it to both models simultaneously, which correctly
    accounts for that correlation and typically gives a tighter, more honest CI on the
    difference than treating the two models as independent samples would.

    Pass `cluster_id` for repeated-measures/longitudinal data (see module docstring).

    Returns a DataFrame with both models' net benefit, their difference (B - A), a
    bootstrap CI on the difference, and a `significant` flag (CI excludes zero).
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob_a = np.asarray(y_prob_a).astype(float)
    y_prob_b = np.asarray(y_prob_b).astype(float)
    if not (y_true.shape == y_prob_a.shape == y_prob_b.shape):
        raise ValueError("y_true, y_prob_a, and y_prob_b must all be the same shape (same patients)")

    thresholds = _clean_thresholds(thresholds)
    n = len(y_true)

    if cluster_id is not None:
        cluster_id = np.asarray(cluster_id)
        if len(cluster_id) != n:
            raise ValueError("cluster_id must be the same length as y_true")

    point_a = _net_benefit_values(y_true, y_prob_a, thresholds)
    point_b = _net_benefit_values(y_true, y_prob_b, thresholds)
    point_diff = point_b - point_a

    rng = np.random.default_rng(seed)
    boot_diff = np.empty((n_boot, len(thresholds)))
    for i in range(n_boot):
        idx = _resample_indices(n, cluster_id, rng)  # SAME resample applied to both models — this is the "paired" part
        yt = y_true[idx]
        nb_a = _net_benefit_values(yt, y_prob_a[idx], thresholds)
        nb_b = _net_benefit_values(yt, y_prob_b[idx], thresholds)
        boot_diff[i] = nb_b - nb_a

    alpha = (1 - ci) / 2
    diff_ci_low = np.quantile(boot_diff, alpha, axis=0)
    diff_ci_high = np.quantile(boot_diff, 1 - alpha, axis=0)
    significant = (diff_ci_low > 0) | (diff_ci_high < 0)

    return pd.DataFrame({
        "threshold": thresholds,
        f"net_benefit_{name_a}": point_a,
        f"net_benefit_{name_b}": point_b,
        "net_benefit_difference": point_diff,
        "diff_ci_low": diff_ci_low,
        "diff_ci_high": diff_ci_high,
        "significant": significant,
    })
