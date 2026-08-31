"""
Generates a synthetic clinical prediction demo dataset: a binary outcome
(e.g. "30-day readmission") plus predictions from two synthetic models —
one well-calibrated, one discriminates similarly well but is overconfident
(pushes probabilities toward 0/1). This pairing makes the difference between
discrimination and calibration visually obvious in the demo app.
"""

import numpy as np
import pandas as pd


def make_demo_predictions(n=2000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Simulate a linear predictor from a few "clinical" features, then squash to a true probability
    age = rng.normal(60, 15, n)
    comorbidity_score = rng.poisson(2, n)
    prior_admissions = rng.poisson(1, n)

    linear_pred = (
        -4.0
        + 0.03 * (age - 60)
        + 0.5 * comorbidity_score
        + 0.6 * prior_admissions
        + rng.normal(0, 0.5, n)  # unexplained noise
    )
    true_p = 1 / (1 + np.exp(-linear_pred))
    y = rng.binomial(1, true_p)

    # "Well-calibrated model": the true generating probability, with a little estimation noise
    p_calibrated = np.clip(true_p + rng.normal(0, 0.03, n), 0.001, 0.999)

    # "Overconfident model": same rank-ordering (similar AUROC) but pushed toward the extremes
    logit_true = np.log(true_p / (1 - true_p))
    logit_over = logit_true * 2.2  # exaggerate the logit -> same ranking, worse calibration
    p_overconfident = 1 / (1 + np.exp(-logit_over))
    p_overconfident = np.clip(p_overconfident + rng.normal(0, 0.02, n), 0.001, 0.999)

    df = pd.DataFrame({
        "patient_id": [f"P{i+1:05d}" for i in range(n)],
        "age": age.round(1),
        "comorbidity_score": comorbidity_score,
        "prior_admissions": prior_admissions,
        "readmitted_30d": y,
        "model_calibrated_prob": p_calibrated.round(4),
        "model_overconfident_prob": p_overconfident.round(4),
    })
    return df


def make_longitudinal_demo(n_patients=150, n_visits=4, seed=7) -> pd.DataFrame:
    """
    Synthetic repeated-measures dataset: each patient is assessed at multiple visits,
    with a patient-level random effect inducing correlation between a patient's own
    visits. Also includes an existing "standard-of-care" risk score alongside a new
    ML model's prediction, evaluated at the same visits — for demonstrating paired
    decision curve comparison plus cluster-aware bootstrap CIs.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for pid in range(n_patients):
        patient_effect = rng.normal(0, 0.8)  # shared frailty across this patient's visits
        base_age = rng.normal(60, 12)

        for visit in range(n_visits):
            comorbidity = rng.poisson(2)
            linear_pred = -3.5 + 0.02 * (base_age - 60) + 0.5 * comorbidity + patient_effect + rng.normal(0, 0.4)
            true_p = 1 / (1 + np.exp(-linear_pred))
            outcome = rng.binomial(1, true_p)

            # standard-of-care score: a cruder, less informative predictor (ignores comorbidity trend over time)
            soc_linear = -3.5 + 0.015 * (base_age - 60) + 0.3 * comorbidity + rng.normal(0, 0.7)
            soc_prob = 1 / (1 + np.exp(-soc_linear))

            # new ML model: closer to the true generating probability
            ml_prob = np.clip(true_p + rng.normal(0, 0.05), 0.001, 0.999)

            rows.append({
                "patient_id": f"LP{pid+1:04d}",
                "visit_number": visit + 1,
                "age_at_visit": round(base_age + visit * 0.5, 1),
                "comorbidity_score": comorbidity,
                "outcome_90d": outcome,
                "standard_of_care_prob": round(float(soc_prob), 4),
                "ml_model_prob": round(float(ml_prob), 4),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_demo_predictions()
    df.to_csv("sample_data/demo_predictions.csv", index=False)
    print(df.head())
    print(f"\nSaved {len(df)} rows to sample_data/demo_predictions.csv")
    print(f"Prevalence: {df['readmitted_30d'].mean():.1%}")

    long_df = make_longitudinal_demo()
    long_df.to_csv("sample_data/demo_longitudinal.csv", index=False)
    print(f"\nSaved {len(long_df)} rows ({long_df['patient_id'].nunique()} patients) to sample_data/demo_longitudinal.csv")
    print(f"Outcome rate: {long_df['outcome_90d'].mean():.1%}")
