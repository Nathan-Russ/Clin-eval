"""
clineval demo app — interactive calibration, discrimination, and decision curve
analysis for binary clinical prediction models.

Run locally:  streamlit run demo_app/app.py
Deploy free:  push to GitHub, then deploy on share.streamlit.io
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow running the demo straight from the repo without installing clineval first
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clineval as ce  # noqa: E402
import literature  # noqa: E402

st.set_page_config(page_title="clineval — Clinical Model Evaluation", page_icon="🩺", layout="wide")

DEMO_PATH = Path(__file__).resolve().parent / "sample_data" / "demo_predictions.csv"
LONGITUDINAL_DEMO_PATH = Path(__file__).resolve().parent / "sample_data" / "demo_longitudinal.csv"

# ---------------------------------------------------------------------------
# Sidebar — data loading
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 clineval")
st.sidebar.caption("Calibration, discrimination & decision curve analysis for clinical prediction models.")

st.sidebar.header("1. Load predictions")
source = st.sidebar.radio(
    "Data source",
    ["Try the demo dataset", "Try the longitudinal demo (paired/clustered)", "Upload my own"],
    index=0,
)

df = None
if source == "Try the demo dataset":
    df = pd.read_csv(DEMO_PATH)
    st.sidebar.success(f"Loaded demo: {len(df):,} patients")
    st.sidebar.caption(
        "Synthetic 30-day readmission outcome with two models: one well-calibrated, "
        "one overconfident but similarly discriminative — a case where AUROC alone "
        "would hide an important difference."
    )
elif source == "Try the longitudinal demo (paired/clustered)":
    df = pd.read_csv(LONGITUDINAL_DEMO_PATH)
    st.sidebar.success(f"Loaded: {len(df):,} visits from {df['patient_id'].nunique():,} patients")
    st.sidebar.caption(
        "Synthetic repeated-measures data — each patient has multiple visits. Compares "
        "an existing standard-of-care score against a new ML model, evaluated on the "
        "same patients. Try setting the cluster/patient ID column below to see how the "
        "confidence intervals widen once repeated visits are properly accounted for."
    )
else:
    up = st.sidebar.file_uploader("CSV with outcome + prediction columns", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.sidebar.success(f"Loaded: {len(df):,} rows")

if df is None:
    st.title("🩺 clineval — Clinical Model Evaluation")
    st.markdown("""
    An interactive tool for evaluating binary clinical prediction models properly —
    not just AUROC, but **calibration** (are predicted probabilities trustworthy?),
    **decision curve analysis** (does using the model actually lead to better clinical
    decisions than treating everyone or no one?), and **paired/longitudinal comparison**
    (is model B meaningfully better than model A on the same patients, once you account
    for repeated measurements?).

    Powered by [`clineval`](..), a small open-source Python library — this app is a
    thin interactive layer over it.

    **Get started:** load a demo dataset from the sidebar, or upload your own predictions.

    #### Expected CSV format
    - One row per patient (or per patient-visit, for longitudinal data)
    - One binary outcome column (0/1)
    - One or more predicted-probability columns (values in [0, 1]) — upload multiple
      model columns to compare them side by side
    - Optionally, a patient/cluster ID column if rows include repeated measurements
    """)
    st.info("👈 Load the demo dataset from the sidebar to see it in action.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — column selection
# ---------------------------------------------------------------------------
st.sidebar.header("2. Select columns")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
binary_cols = [c for c in numeric_cols if set(df[c].dropna().unique()).issubset({0, 1})]

if not binary_cols:
    st.error("No binary (0/1) column found to use as the outcome. Check your uploaded file.")
    st.stop()

default_outcome = "readmitted_30d" if "readmitted_30d" in binary_cols else binary_cols[0]
outcome_col = st.sidebar.selectbox("Outcome column (0/1)", binary_cols, index=binary_cols.index(default_outcome))

prob_candidates = [c for c in numeric_cols if c != outcome_col and df[c].between(0, 1).all()]
default_models = [c for c in prob_candidates if "prob" in c.lower()] or prob_candidates[:1]
model_cols = st.sidebar.multiselect("Prediction column(s) to evaluate", prob_candidates, default=default_models)

if not model_cols:
    st.warning("Select at least one prediction column in the sidebar.")
    st.stop()

st.sidebar.header("3. Paired / longitudinal (optional)")
default_cluster = "patient_id" if "patient_id" in df.columns else "(none)"
cluster_options = ["(none)"] + [c for c in df.columns if c != outcome_col]
cluster_col = st.sidebar.selectbox(
    "Patient/cluster ID column",
    cluster_options,
    index=cluster_options.index(default_cluster) if default_cluster in cluster_options else 0,
    help="Set this if rows include repeated measurements per patient (e.g. multiple visits). "
         "Enables cluster-aware bootstrap confidence intervals in the Decision Curve Analysis tab.",
)
cluster_id = df[cluster_col].values if cluster_col != "(none)" else None
if cluster_id is not None:
    n_unique = df[cluster_col].nunique()
    if n_unique == len(df):
        st.sidebar.caption(f"{n_unique:,} unique values — one row per cluster, so this behaves like standard i.i.d. resampling.")
    else:
        st.sidebar.caption(f"{n_unique:,} unique patients across {len(df):,} rows — cluster bootstrap will resample whole patients.")

y_true = df[outcome_col].values
prevalence = y_true.mean()

# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------
st.title("🩺 clineval — Clinical Model Evaluation")
c1, c2, c3 = st.columns(3)
c1.metric("Patients", f"{len(df):,}")
c2.metric("Outcome prevalence", f"{prevalence:.1%}")
c3.metric("Models compared", len(model_cols))

tab_calib, tab_disc, tab_dca, tab_lit, tab_export = st.tabs(
    ["📐 Calibration", "🎯 Discrimination", "⚖️ Decision Curve Analysis", "📚 Literature Context", "💾 Export"]
)

palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# --- Calibration tab ---
with tab_calib:
    st.subheader("Calibration curves")
    n_bins = st.slider("Number of bins", 5, 20, 10)
    strategy = st.radio("Binning strategy", ["quantile", "uniform"], horizontal=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                              name="Perfect calibration"))

    summary_rows = []
    for i, col in enumerate(model_cols):
        p = df[col].values
        data = ce.calibration.calibration_curve_data(y_true, p, n_bins=n_bins, strategy=strategy)
        fig.add_trace(go.Scatter(
            x=data["mean_predicted"], y=data["observed_fraction"],
            mode="lines+markers", name=col,
            line=dict(color=palette[i % len(palette)]),
            error_y=dict(
                type="data", symmetric=False,
                array=data["ci_high"] - data["observed_fraction"],
                arrayminus=data["observed_fraction"] - data["ci_low"],
            ),
        ))
        summary_rows.append({"model": col, **ce.calibration.calibration_summary(y_true, p, n_bins=n_bins)})

    fig.update_layout(
        xaxis_title="Mean predicted probability", yaxis_title="Observed fraction of positives",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        template="plotly_white", height=550, title="Calibration Curve",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Calibration summary")
    summary_df = pd.DataFrame(summary_rows).set_index("model")
    summary_df.columns = ["Brier score", "Calibration slope", "Calibration intercept", "ECE"]
    st.dataframe(summary_df.style.format("{:.4f}"), width="stretch")
    st.caption(
        "Slope=1 & intercept=0 is perfect calibration. Slope<1 means the model is "
        "overconfident (predictions too extreme); intercept≠0 means systematic over/under-estimation "
        "of overall risk. Brier score and ECE: lower is better."
    )

# --- Discrimination tab ---
with tab_disc:
    st.subheader("ROC curves")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Chance"))

    disc_rows = []
    for i, col in enumerate(model_cols):
        p = df[col].values
        roc = ce.discrimination.roc_curve_data(y_true, p)
        auc_result = ce.discrimination.bootstrap_ci(y_true, p, ce.discrimination.auroc, n_boot=300, seed=0)
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines",
            name=f"{col} (AUROC={auc_result['estimate']:.3f})",
            line=dict(color=palette[i % len(palette)]),
        ))
        disc_rows.append({
            "model": col,
            "AUROC": auc_result["estimate"],
            "AUROC 95% CI": f"({auc_result['ci_low']:.3f}, {auc_result['ci_high']:.3f})",
            "AUPRC": ce.discrimination.auprc(y_true, p),
        })

    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                       template="plotly_white", height=550, title="ROC Curve")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Discrimination summary")
    st.dataframe(pd.DataFrame(disc_rows).set_index("model"), width="stretch")

    st.subheader("Classification metrics at a chosen threshold")
    sel_model = st.selectbox("Model", model_cols, key="thresh_model")
    threshold = st.slider("Probability threshold", 0.01, 0.99, 0.5, 0.01)
    metrics = ce.discrimination.classification_metrics_at_threshold(y_true, df[sel_model].values, threshold)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sensitivity", f"{metrics['sensitivity']:.2%}")
    m2.metric("Specificity", f"{metrics['specificity']:.2%}")
    m3.metric("PPV", f"{metrics['ppv']:.2%}")
    m4.metric("NPV", f"{metrics['npv']:.2%}")
    m5.metric("Accuracy", f"{metrics['accuracy']:.2%}")

# --- Decision Curve Analysis tab ---
with tab_dca:
    st.subheader("Decision curve analysis")
    st.caption(
        "At the probability threshold a clinician would actually act on, does using "
        "the model lead to better decisions than treating everyone or treating no one?"
    )

    prob_dict = {col: df[col].values for col in model_cols}
    long_df = ce.decision_curve.compare_models_net_benefit(y_true, prob_dict)

    fig = go.Figure()
    color_map = {}
    for i, name in enumerate(model_cols):
        color_map[name] = palette[i % len(palette)]
    color_map["Treat all"] = "#888888"
    color_map["Treat none"] = "#000000"

    for strategy_name, group in long_df.groupby("strategy"):
        dash = "solid" if strategy_name in model_cols else ("dash" if strategy_name == "Treat all" else "dot")
        fig.add_trace(go.Scatter(
            x=group["threshold"], y=group["net_benefit"], mode="lines",
            name=strategy_name, line=dict(color=color_map.get(strategy_name), dash=dash),
        ))

    fig.update_layout(
        xaxis_title="Threshold probability", yaxis_title="Net benefit",
        template="plotly_white", height=550, title="Decision Curve Analysis",
    )
    # Net benefit can dip negative but rarely needs to be shown much below 0
    fig.update_yaxes(range=[max(-0.05, long_df["net_benefit"].min() - 0.02), long_df["net_benefit"].max() + 0.02])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Net benefit at a specific threshold")
    dca_threshold = st.slider("Threshold probability", 0.01, 0.99, 0.2, 0.01, key="dca_threshold")
    rows = []
    for col in model_cols:
        nb = ce.decision_curve.net_benefit_at_threshold(y_true, df[col].values, dca_threshold)
        rows.append({"model": col, "Net benefit": nb["net_benefit_model"]})
    rows.append({"model": "Treat all", "Net benefit": nb["net_benefit_treat_all"]})
    rows.append({"model": "Treat none", "Net benefit": nb["net_benefit_treat_none"]})
    st.dataframe(pd.DataFrame(rows).set_index("model"), width="stretch")

    st.divider()
    st.subheader("Paired comparison: is one model meaningfully better?")
    st.caption(
        "Compares two models on the SAME patients using correlated bootstrap resampling — "
        "the correct approach when both models see the same cohort, since their errors are "
        "correlated (both do well or badly on the same easy/hard patients). This gives a "
        "tighter, more honest confidence interval on the difference than comparing the two "
        "models' individual CIs by eye."
        + (" Cluster resampling is active, since a patient/cluster ID column is set — "
           "whole patients (all their visits) are resampled together." if cluster_id is not None else "")
    )

    if len(model_cols) < 2:
        st.info("Select at least 2 prediction columns in the sidebar to run a paired comparison.")
    else:
        pc1, pc2 = st.columns(2)
        model_a_name = pc1.selectbox("Model A (baseline)", model_cols, index=0, key="paired_a")
        remaining = [m for m in model_cols if m != model_a_name]
        model_b_name = pc2.selectbox("Model B (comparison)", remaining, index=0, key="paired_b")

        n_boot_paired = st.slider("Bootstrap iterations", 200, 2000, 500, 100, key="paired_nboot")

        with st.spinner("Running paired bootstrap..."):
            paired_result = ce.decision_curve.compare_paired_net_benefit(
                y_true, df[model_a_name].values, df[model_b_name].values,
                n_boot=n_boot_paired, seed=0, cluster_id=cluster_id,
                name_a=model_a_name, name_b=model_b_name,
            )

        fig_paired = go.Figure()
        fig_paired.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_paired.add_trace(go.Scatter(
            x=list(paired_result["threshold"]) + list(paired_result["threshold"][::-1]),
            y=list(paired_result["diff_ci_high"]) + list(paired_result["diff_ci_low"][::-1]),
            fill="toself", fillcolor="rgba(0,114,178,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True,
        ))
        fig_paired.add_trace(go.Scatter(
            x=paired_result["threshold"], y=paired_result["net_benefit_difference"],
            mode="lines", line=dict(color="#0072B2"), name=f"Net benefit: {model_b_name} − {model_a_name}",
        ))
        fig_paired.update_layout(
            xaxis_title="Threshold probability", yaxis_title=f"Net benefit difference ({model_b_name} − {model_a_name})",
            template="plotly_white", height=450, title="Paired Net Benefit Difference",
        )
        st.plotly_chart(fig_paired, width="stretch")

        n_sig = paired_result["significant"].sum()
        n_total = len(paired_result)
        if n_sig > 0:
            sig_range = paired_result.loc[paired_result["significant"], "threshold"]
            st.success(
                f"**{model_b_name}** shows a statistically meaningful net benefit difference vs. "
                f"**{model_a_name}** at {n_sig}/{n_total} thresholds tested "
                f"(roughly {sig_range.min():.2f}–{sig_range.max():.2f})."
            )
        else:
            st.info(f"No threshold showed a statistically meaningful difference between the two models (CI includes zero throughout).")

        st.dataframe(
            paired_result.style.format({
                f"net_benefit_{model_a_name}": "{:.4f}", f"net_benefit_{model_b_name}": "{:.4f}",
                "net_benefit_difference": "{:.4f}", "diff_ci_low": "{:.4f}", "diff_ci_high": "{:.4f}",
            }),
            width="stretch",
        )

# --- Literature Context tab ---
with tab_lit:
    st.subheader("Find comparable published models")
    st.caption(
        "Live search against PubMed and ClinicalTrials.gov, so you can see how your model's "
        "AUROC/calibration stacks up against similar published work. AUROC values extracted from "
        "abstracts below are a best-effort text search, not verified data — always check the "
        "original abstract before citing a number."
    )

    lit_query = st.text_input(
        "Search term (condition, outcome, or model type)",
        value="30-day hospital readmission risk prediction model",
    )
    n_results = st.slider("Number of results", 3, 10, 5, key="lit_n_results")

    if st.button("Search literature", type="primary"):
        pubmed_col, ctgov_col = st.columns(2)

        with pubmed_col:
            st.markdown("#### 📄 PubMed")
            try:
                with st.spinner("Searching PubMed..."):
                    articles = literature.search_pubmed(lit_query, max_results=n_results)
                if not articles:
                    st.info("No PubMed results for this search term.")
                for a in articles:
                    st.markdown(f"**[{a['title']}]({a['url']})**")
                    st.caption(f"{a['authors']} — {a['journal']}, {a['year']}")
                    try:
                        abstract = literature.fetch_pubmed_abstract(a["pmid"])
                        snippets = literature.extract_performance_snippets(abstract)
                        for s in snippets[:2]:
                            st.caption(f"📊 Possible reported {s['metric']}: **{s['value']}** — _{s['snippet']}_")
                    except literature.LiteratureLookupError:
                        pass  # abstract fetch failing shouldn't block showing the article itself
                    st.write("")
            except literature.LiteratureLookupError as e:
                st.error(f"PubMed search failed: {e}")

        with ctgov_col:
            st.markdown("#### 🧪 ClinicalTrials.gov")
            try:
                with st.spinner("Searching ClinicalTrials.gov..."):
                    trials = literature.search_clinical_trials(lit_query, max_results=n_results)
                if not trials:
                    st.info("No ClinicalTrials.gov results for this search term.")
                for t in trials:
                    st.markdown(f"**[{t['title']}]({t['url']})**")
                    st.caption(f"{t['nct_id']} — {t['status']}, {t['phase']}")
                    st.write("")
            except literature.LiteratureLookupError as e:
                st.error(f"ClinicalTrials.gov search failed: {e}")
    else:
        st.info("Enter a search term and click **Search literature** to pull in context from PubMed and ClinicalTrials.gov.")

# --- Export tab ---
with tab_export:
    st.subheader("Download results")
    all_summaries = []
    for col in model_cols:
        p = df[col].values
        row = {"model": col}
        row.update(ce.calibration.calibration_summary(y_true, p))
        row.update(ce.discrimination.discrimination_summary(y_true, p))
        all_summaries.append(row)
    summary_export = pd.DataFrame(all_summaries)
    st.dataframe(summary_export, width="stretch")
    st.download_button(
        "Download summary metrics (CSV)",
        summary_export.to_csv(index=False).encode("utf-8"),
        file_name="clineval_summary.csv",
        mime="text/csv",
    )

st.sidebar.divider()
st.sidebar.caption("Built with [clineval](..) + [Streamlit](https://streamlit.io) · [View source on GitHub](#)")
