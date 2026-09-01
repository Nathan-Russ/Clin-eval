# 🩺 clineval

Calibration, discrimination, and decision curve analysis for binary clinical prediction models — as a small, tested, pip-installable Python library, plus an interactive Streamlit app to explore results without writing any code.

Most clinical ML evaluations report AUROC and stop there. AUROC alone can't tell you whether a model's probability estimates are trustworthy (**calibration**), or whether using it actually changes clinical decisions for the better at the threshold clinicians would use in practice (**decision curve analysis**). `clineval` makes both of those easy to compute and visualise properly.


## What's included

- **Calibration**: calibration curves with confidence intervals, Brier score, calibration
  slope/intercept, expected calibration error (ECE)
- **Discrimination**: ROC/PR curves, AUROC/AUPRC with bootstrap confidence intervals,
  Youden's J optimal threshold, sensitivity/specificity/PPV/NPV at any threshold
- **Decision curve analysis**: net benefit of the model vs. treat-all vs. treat-none
  strategies across a range of thresholds ([Vickers & Elkin, 2006](https://doi.org/10.1177/0272989X06295361))
- **Paired / longitudinal mode**: compare two models on the *same* patients using
  correlated bootstrap resampling (correct — naive independent CIs overstate uncertainty
  when both models see the same cohort), with optional cluster-aware resampling for
  repeated-measures/longitudinal data (multiple visits per patient)
- **Literature context**: live search against PubMed and ClinicalTrials.gov from within
  the app, with a best-effort scan for AUROC/c-statistic mentions in abstracts, so you can
  see how your model compares to similar published work
- **Streamlit demo app**: upload your own predictions or use the built-in synthetic
  datasets (cross-sectional and longitudinal), compare multiple models side by side, export results

## Install the library

```bash
pip install -e ".[plots]"
```

(Not yet published to PyPI):
`pip install git+https://github.com/Nathan-Russ/Clin-eval.git`)

## Quickstart

```python
import clineval as ce

# y_true: array of 0/1 outcomes. y_prob: array of predicted probabilities.
ce.calibration.calibration_summary(y_true, y_prob)
# {'brier_score': 0.09, 'calibration_slope': 0.89, 'calibration_intercept': -0.11, 'expected_calibration_error': 0.01}

ce.discrimination.discrimination_summary(y_true, y_prob, bootstrap=True)
# {'auroc': 0.78, 'auroc_ci': (0.76, 0.81), 'auprc': 0.42, 'auprc_ci': (0.38, 0.46)}

ce.decision_curve.net_benefit(y_true, y_prob)
# DataFrame of net benefit for the model, treat-all, and treat-none across thresholds

# Paired comparison: two models on the SAME patients, correlation-aware bootstrap
ce.decision_curve.compare_paired_net_benefit(y_true, y_prob_a, y_prob_b, name_a="existing", name_b="new_model")
# DataFrame with both models' net benefit, their difference, a bootstrap CI on the
# difference, and a `significant` flag (CI excludes zero) — at each threshold

# Longitudinal / repeated-measures: pass a patient ID array to resample whole
# patients (not individual rows) in the bootstrap
ce.decision_curve.bootstrap_net_benefit_ci(y_true, y_prob, cluster_id=patient_ids)
ce.decision_curve.compare_paired_net_benefit(y_true, y_prob_a, y_prob_b, cluster_id=patient_ids)

# Optional matplotlib plots (requires the [plots] extra)
ce.plotting.plot_calibration_curve(y_true, y_prob)
ce.plotting.plot_roc_curve(y_true, y_prob)
ce.plotting.plot_decision_curve(y_true, y_prob)
```

## Run the interactive demo app locally

```bash
pip install -r requirements.txt
streamlit run demo_app/app.py
```

The demo ships with two synthetic datasets:
- A cross-sectional 30-day-readmission dataset comparing two models with similar AUROC
  but very different calibration — a case where discrimination alone would hide a real problem.
- A longitudinal dataset (multiple visits per patient) comparing a standard-of-care score
  against a new ML model, for exploring paired comparison and cluster-aware confidence intervals.

The **Literature Context** tab makes live calls to PubMed and ClinicalTrials.gov, so it
needs internet access to work 

## To Deploy your own copy of the demo app (free, ~2 minutes)

1. Clone this repo to your own GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick your repo/branch, and set the main file path to `demo_app/app.py`.
4. Click **Deploy**.



## Project structure

```
src/clineval/
  calibration.py       # calibration curves, Brier score, slope/intercept, ECE
  discrimination.py    # ROC/PR, AUROC/AUPRC, bootstrap CIs, threshold metrics
  decision_curve.py    # net benefit / DCA, paired comparison, cluster-aware bootstrap
  plotting.py           # optional matplotlib plots (static, for notebooks/papers)
tests/                  # pytest unit tests for every metric, including paired mode
demo_app/
  app.py                # Streamlit app (interactive Plotly plots)
  literature.py          # PubMed + ClinicalTrials.gov connectors
  make_demo_data.py      # generates both synthetic demo datasets
  sample_data/            # the generated demo CSVs
```

## A note on interpretation

- **Calibration slope < 1** means the model's predictions are too extreme
  (overconfident); **> 1** means too conservative (underconfident).
- **AUROC and calibration are independent.** A model can rank patients perfectly
  (AUROC = 1) while being badly calibrated, and vice versa. Always check both.
- **Decision curve analysis depends on the threshold a clinician would actually use.**
  A model can win on AUROC but lose on net benefit at the threshold that matters
  clinically — that's the entire point of computing it separately.
- **Paired comparison needs correlated resampling.** If two models are evaluated on
  the same patients, their errors are correlated — resampling each model's predictions
  independently overstates the uncertainty in their difference. Always use
  `compare_paired_net_benefit` (not two separate `bootstrap_net_benefit_ci` calls
  compared by eye) when the same cohort underlies both models.
- **Repeated measurements need cluster-aware resampling.** If your data has multiple
  rows per patient (e.g. visits over time), pass `cluster_id` so the bootstrap resamples
  whole patients — otherwise the confidence intervals will be too narrow.
- **Literature-extracted performance numbers are a starting point, not ground truth.**
  The app's regex scan for "AUC/AUROC/c-statistic" mentions in PubMed abstracts is a
  heuristic search, not validated data extraction — always check the original abstract.
- This tool is meant for exploration and teaching, not as a replacement for a
  validated evaluation pipeline when preparing results for publication or clinical
  deployment.

## License

MIT — see [LICENSE](LICENSE).
