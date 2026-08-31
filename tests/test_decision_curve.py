import numpy as np
import pytest

from clineval import decision_curve as dca


def test_net_benefit_treat_none_is_always_zero():
    rng = np.random.default_rng(0)
    n = 1000
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)

    df = dca.net_benefit(y, p)
    assert np.allclose(df["net_benefit_treat_none"], 0.0)


def test_net_benefit_treat_all_matches_manual_formula():
    y = np.array([1, 1, 0, 0, 0])  # prevalence = 0.4
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    pt = 0.2
    df = dca.net_benefit(y, p, thresholds=[pt])

    prevalence = 0.4
    expected_treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
    assert df.iloc[0]["net_benefit_treat_all"] == pytest.approx(expected_treat_all)


def test_net_benefit_perfect_model_beats_treat_all_at_high_threshold():
    rng = np.random.default_rng(1)
    n = 2000
    true_p = rng.uniform(0, 1, n)
    y = rng.binomial(1, true_p)

    # perfect model: predicted prob == true probability of the event
    df = dca.net_benefit(y, true_p, thresholds=[0.5])
    row = df.iloc[0]
    assert row["net_benefit_model"] >= row["net_benefit_treat_all"]
    assert row["net_benefit_model"] >= row["net_benefit_treat_none"]


def test_net_benefit_at_threshold_matches_full_curve():
    rng = np.random.default_rng(2)
    n = 500
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)

    single = dca.net_benefit_at_threshold(y, p, 0.3)
    full = dca.net_benefit(y, p, thresholds=[0.3]).iloc[0]
    assert single["net_benefit_model"] == pytest.approx(full["net_benefit_model"])


def test_compare_models_net_benefit_includes_all_strategies():
    rng = np.random.default_rng(3)
    n = 300
    y = rng.binomial(1, 0.3, n)
    probs = {"model_a": rng.uniform(0, 1, n), "model_b": rng.uniform(0, 1, n)}

    result = dca.compare_models_net_benefit(y, probs, thresholds=[0.1, 0.5])
    strategies = set(result["strategy"].unique())
    assert strategies == {"model_a", "model_b", "Treat all", "Treat none"}


def test_validate_rejects_non_binary_outcome():
    with pytest.raises(ValueError):
        dca.net_benefit([0, 1, 2], [0.1, 0.5, 0.9])


# ---------------------------------------------------------------------------
# Paired / longitudinal mode
# ---------------------------------------------------------------------------

def test_bootstrap_net_benefit_ci_contains_point_estimate():
    rng = np.random.default_rng(10)
    n = 800
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)

    result = dca.bootstrap_net_benefit_ci(y, p, thresholds=[0.1, 0.3, 0.5], n_boot=300, seed=1)
    assert (result["ci_low"] <= result["net_benefit_model"]).all()
    assert (result["net_benefit_model"] <= result["ci_high"]).all()


def test_compare_paired_net_benefit_identical_models_has_zero_difference():
    rng = np.random.default_rng(11)
    n = 500
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)

    # Same predictions for both "models" -> point difference must be exactly zero
    result = dca.compare_paired_net_benefit(y, p, p, thresholds=[0.1, 0.3, 0.5], n_boot=200, seed=2)
    assert np.allclose(result["net_benefit_difference"], 0.0)
    assert not result["significant"].any()


def test_compare_paired_net_benefit_detects_clearly_better_model():
    rng = np.random.default_rng(12)
    n = 2000
    true_p = rng.uniform(0, 1, n)
    y = rng.binomial(1, true_p)

    # model_a: pure noise; model_b: the true generating probability (much better)
    model_a = rng.uniform(0, 1, n)
    model_b = true_p

    result = dca.compare_paired_net_benefit(y, model_a, model_b, thresholds=[0.2, 0.3, 0.4], n_boot=300, seed=3)
    assert (result["net_benefit_difference"] > 0).all()
    assert result["significant"].any()


def test_compare_paired_net_benefit_requires_matching_shapes():
    with pytest.raises(ValueError):
        dca.compare_paired_net_benefit([0, 1, 1], [0.2, 0.5], [0.3, 0.6, 0.9])


def test_cluster_bootstrap_uses_all_rows_of_sampled_clusters():
    # 4 patients, 3 rows each (longitudinal) -> cluster bootstrap resample sizes should
    # always be a multiple of the cluster size (3), since whole clusters are drawn together
    rng = np.random.default_rng(13)
    cluster_id = np.repeat([0, 1, 2, 3], 3)
    n = len(cluster_id)
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)

    for _ in range(20):
        idx = dca._resample_indices(n, cluster_id, rng)
        assert len(idx) % 3 == 0


def test_cluster_aware_ci_runs_without_error_on_longitudinal_shape():
    rng = np.random.default_rng(14)
    cluster_id = np.repeat(np.arange(50), 3)  # 50 patients, 3 visits each
    n = len(cluster_id)
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)

    result = dca.bootstrap_net_benefit_ci(y, p, thresholds=[0.2, 0.5], n_boot=100, seed=4, cluster_id=cluster_id)
    assert len(result) == 2
    assert (result["ci_low"] <= result["ci_high"]).all()
