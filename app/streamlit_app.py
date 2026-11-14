"""HunarLens — 5-View Streamlit Dashboard for Youth Unemployment Vulnerability & Skills Analysis in Pakistan.

Implements PROJECT_PLAN.md §11, §12.1, §12.2:
1. Overview — Scope, 2024–25 LFS + 2023 Census provenance, cohort eligibility, and methodological limits.
2. Profile Explorer — Validated form (ages 15–24), auto Census 2023 lookup, frozen probability & validation
   threshold (0.37), local counterfactual contribution chart, and global permutation importance.
3. Population Explorer — Survey-weighted & unweighted subgroup summaries with visible denominators and
   `<30` small-cell disclosure control.
4. Skills and Education — Descriptive education, enrolment, and vocational training coverage & unemployment
   associations with explicit non-causal and leakage-prevention notes.
5. Model Evidence — Frozen test comparison table, 1,000-resample PSU bootstrap 95% CIs, ablations, subgroup
   diagnostics, PR/ROC curves, confusion matrix, calibration plot, and leakage protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ModuleNotFoundError:  # Allows importing presets/loaders in headless pytest environments
    _HAS_STREAMLIT = False

    class _StreamlitStub:
        def set_page_config(self, **kwargs: Any) -> None:
            pass

        def cache_resource(self, func: Any) -> Any:
            return func

        def cache_data(self, func: Any) -> Any:
            return func

    st = _StreamlitStub()  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hunarlens.features import DEFAULT_CENSUS_LOOKUP, SUPPORTED_LOCALITIES, SUPPORTED_PROVINCES  # noqa: E402
from hunarlens.inference import HunarLensPredictor  # noqa: E402


st.set_page_config(
    page_title="HunarLens — Youth Unemployment & Skills Analysis (Pakistan)",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_cached_predictor() -> HunarLensPredictor:
    """Load the trusted frozen `models/final_pipeline.joblib` and metadata once per session (§12.2)."""
    return HunarLensPredictor()


@st.cache_data
def load_dashboard_artifacts() -> Dict[str, Any]:
    """Load precomputed JSON metrics, manifests, and safe aggregated population/skills tables."""
    metrics_dir = REPO_ROOT / "reports/metrics"
    agg_path = metrics_dir / "population_and_skills_aggregates.json"

    data: Dict[str, Any] = {
        "split_manifest": json.loads((REPO_ROOT / "data/splits/split_manifest.json").read_text(encoding="utf-8")),
        "eda_summary": json.loads((metrics_dir / "eda_summary.json").read_text(encoding="utf-8")),
        "train_val": json.loads((metrics_dir / "training_validation_results.json").read_text(encoding="utf-8")),
        "test_eval": json.loads((metrics_dir / "final_test_evaluation.json").read_text(encoding="utf-8")),
        "global_imp": json.loads((metrics_dir / "global_feature_importance.json").read_text(encoding="utf-8")),
        "census_df": pd.read_csv(REPO_ROOT / "data/reference/census_2023_regional.csv"),
        "benchmarks_df": pd.read_csv(REPO_ROOT / "data/reference/pbs_lfs_2024_25_empirical_benchmarks.csv"),
    }
    if agg_path.exists():
        data["aggregates"] = json.loads(agg_path.read_text(encoding="utf-8"))
    else:
        data["aggregates"] = {}
    return data


SYNTHETIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "Preset 1 — Urban Female University Graduate (Khyber Pakhtunkhwa)": {
        "age": 22,
        "sex": "Female",
        "marital_status": "Never Married",
        "relationship_to_head": "Son/Daughter",
        "can_read": "Yes",
        "can_write": "Yes",
        "highest_education": "Degree & Postgraduate",
        "current_enrolment": "Not Enrolled",
        "dependents_count": 3,
        "working_age_count": 3,
        "province_territory": "Khyber Pakhtunkhwa",
        "locality": "Urban",
    },
    "Preset 2 — Rural Male Head of Household with Primary Schooling (Sindh)": {
        "age": 23,
        "sex": "Male",
        "marital_status": "Currently Married",
        "relationship_to_head": "Head",
        "can_read": "Yes",
        "can_write": "Yes",
        "highest_education": "Primary",
        "current_enrolment": "Not Enrolled",
        "dependents_count": 2,
        "working_age_count": 3,
        "province_territory": "Sindh",
        "locality": "Rural",
    },
    "Preset 3 — Urban Teen Student Actively Seeking Part-Time Work (Islamabad)": {
        "age": 18,
        "sex": "Male",
        "marital_status": "Never Married",
        "relationship_to_head": "Son/Daughter",
        "can_read": "Yes",
        "can_write": "Yes",
        "highest_education": "Intermediate",
        "current_enrolment": "Currently Enrolled",
        "dependents_count": 2,
        "working_age_count": 4,
        "province_territory": "Islamabad",
        "locality": "Urban",
    },
}


def render_overview_view(predictor: HunarLensPredictor, artifacts: Dict[str, Any]) -> None:
    st.title("📊 HunarLens — Youth Unemployment Vulnerability & Skills Analysis")
    st.markdown(
        "**Course:** CT-378 Machine Learning · **Group:** AlphaDevs "
        "(Danish Meraj `CT-24268`, Sarim Khan `CT-24266`, Muhammad Saraib `CT-24255`)  \n"
        "**Survey Coverage:** Pakistan Bureau of Statistics (PBS) **Labour Force Survey 2024–25** "
        "integrated with **7th Population & Housing Census 2023 (Table 1)** regional indicators."
    )

    st.info(
        "**Scope & Non-Causal Guardrail (§1, §12.1):** HunarLens estimates a model score associated with "
        "**current survey-defined unemployment (19th ICLS) conditional on labour-force membership** among "
        "Pakistani youth aged 15–24. It reflects 2024–25 survey conditions—not live 2026 forecasts—and "
        "does **not** establish causal effects of education/training or automate hiring/benefit decisions."
    )

    sm = artifacts["split_manifest"]
    te = artifacts["test_eval"]
    w_metrics = te["models"][te["winner_model_key"]]["test_metrics_at_frozen_threshold"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Eligible Youth Cohort (15–24)", f"{sm['total_eligible_rows']:,}", f"{sm['total_psus']:,} Independent PSUs")
    col2.metric("Approved Core Predictors", "15", "12 LFS Survey + 3 Census 2023")
    col3.metric(
        f"Frozen Winner ({predictor.metadata['selected_model_display_name']})",
        f"Test F1 = {w_metrics['f1_pos']:.4f}",
        f"Validation Threshold = {predictor.decision_threshold:.2f}",
    )
    col4.metric(
        "Test Average Precision (AP)",
        f"{w_metrics['average_precision']:.4f}",
        f"ROC-AUC = {w_metrics['roc_auc']:.4f} | Brier = {w_metrics['brier_score']:.4f}",
    )

    st.subheader("1. Two-Source Public Data Architecture (§4)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### Source A: PBS Labour Force Survey (LFS) 2024–25
            * **Cohort:** Pakistani youth aged **15–24** in the active labour force (`employed = 0`, `unemployed = 1` under the **19th ICLS** framework).
            * **Grouping Unit:** 8-digit Primary Sampling Unit (`PCode`), keeping all households `(PCode, HHNo)` strictly together across `GroupShuffleSplit` (`70/15/15`) and 5-fold `StratifiedGroupKFold`.
            * **Geography Fix:** First digit of `PCode` distinguishes **Islamabad (`6`)** from **Punjab (`2`)**, resolving the raw `Province` column conflation.
            """
        )
    with c2:
        st.markdown("#### Source B: Pakistan Census 2023 Regional Context (`validate='m:1'`)")
        st.dataframe(
            artifacts["census_df"][
                [
                    "pcode_first_digit",
                    "province_territory",
                    "regional_pop_density",
                    "regional_urban_share",
                    "regional_avg_household_size",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("2. Official PBS LFS 2024–25 Empirical Benchmarks")
    st.dataframe(artifacts["benchmarks_df"], use_container_width=True, hide_index=True)


def render_profile_explorer_view(predictor: HunarLensPredictor, artifacts: Dict[str, Any]) -> None:
    st.title("🧭 Youth Profile Explorer & Local Model Explanation")
    st.markdown(
        "Submit an illustrative youth demographic & household profile (aged **15–24**) to inspect the "
        "frozen model's **estimated probability of current unemployment within the eligible labour-force population**, "
        "the validation-selected decision threshold, and the local feature contribution chart (§12.1)."
    )

    preset_choice = st.selectbox(
        "Load a Synthetic Illustrative Profile Preset:",
        options=list(SYNTHETIC_PRESETS.keys()),
    )
    preset = SYNTHETIC_PRESETS[preset_choice]

    with st.form("profile_explorer_form"):
        st.subheader("1. Eligibility & Demographic Inputs (12 LFS Survey Features)")
        elig_ack = st.checkbox(
            "I acknowledge this illustrative profile represents a youth aged 15–24 in the active labour force "
            "(either working or actively seeking and available for work under 19th ICLS).",
            value=True,
        )

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        age = r1c1.number_input("Age in Completed Years (15–24)", min_value=15, max_value=24, value=int(preset["age"]), step=1)
        sex = r1c2.selectbox("Recorded Sex / Gender", options=["Male", "Female"], index=0 if preset["sex"] == "Male" else 1)
        marital_status = r1c3.selectbox(
            "Marital Status",
            options=["Never Married", "Currently Married", "Widowed/Divorced"],
            index=["Never Married", "Currently Married", "Widowed/Divorced"].index(preset["marital_status"]),
        )
        relationship_to_head = r1c4.selectbox(
            "Relationship to Household Head",
            options=["Son/Daughter", "Head", "Spouse", "Other Relative", "Non-Relative"],
            index=["Son/Daughter", "Head", "Spouse", "Other Relative", "Non-Relative"].index(preset["relationship_to_head"]),
        )

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        can_read = r2c1.selectbox("Can Read with Understanding", options=["Yes", "No", "Unknown"], index=0 if preset["can_read"] == "Yes" else 1)
        can_write = r2c2.selectbox("Can Write with Understanding", options=["Yes", "No", "Unknown"], index=0 if preset["can_write"] == "Yes" else 1)
        edu_opts = [
            "No Formal Education",
            "Primary",
            "Middle",
            "Matric",
            "Intermediate",
            "Degree & Postgraduate",
        ]
        highest_education = r2c3.selectbox(
            "Highest Formal Education",
            options=edu_opts,
            index=edu_opts.index(preset["highest_education"]),
        )
        current_enrolment = r2c4.selectbox(
            "Current Educational Enrolment",
            options=["Not Enrolled", "Currently Enrolled"],
            index=["Not Enrolled", "Currently Enrolled"].index(preset["current_enrolment"]),
        )

        st.subheader("2. Household Composition & Geographic Context (Auto Census 2023 Lookup)")
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        working_age_count = r3c1.number_input(
            "Working-Age Household Members (Aged 15–64, >=1)",
            min_value=1,
            max_value=20,
            value=int(preset["working_age_count"]),
            step=1,
        )
        dependents_count = r3c2.number_input(
            "Dependent Household Members (Aged <15 or >=65, >=0)",
            min_value=0,
            max_value=25,
            value=int(preset["dependents_count"]),
            step=1,
        )
        province_territory = r3c3.selectbox(
            "Province / Federal Territory",
            options=list(SUPPORTED_PROVINCES),
            index=list(SUPPORTED_PROVINCES).index(preset["province_territory"]),
        )
        locality = r3c4.selectbox(
            "Urban / Rural Locality",
            options=list(SUPPORTED_LOCALITIES),
            index=list(SUPPORTED_LOCALITIES).index(preset["locality"]),
        )

        submitted = st.form_submit_button("Evaluate Youth Profile", type="primary")

    if not elig_ack:
        st.warning(
            "Please confirm labour-force eligibility above. The classifier is conditional on active labour-force "
            "membership and does not score out-of-scope or inactive profiles (§12.1)."
        )
        return

    if submitted or True:
        profile_payload = {
            "age": int(age),
            "sex": sex,
            "marital_status": marital_status,
            "relationship_to_head": relationship_to_head,
            "can_read": can_read,
            "can_write": can_write,
            "highest_education": highest_education,
            "current_enrolment": current_enrolment,
            "dependents_count": int(dependents_count),
            "working_age_count": int(working_age_count),
            "province_territory": province_territory,
            "locality": locality,
        }

        try:
            pred = predictor.predict_profile(profile_payload, include_explanation=True)
        except Exception as exc:
            st.error(f"Input Validation Error: {exc}")
            return

        norm = pred["normalized_inputs"]
        st.markdown("---")
        st.subheader("3. Model Estimate & Validation-Selected Decision Threshold")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Estimated Probability of Current Unemployment",
            f"{pred['estimated_probability']:.1%}",
            f"Raw probability = {pred['estimated_probability']:.4f}",
        )
        m2.metric(
            "Saved Validation Threshold",
            f"{pred['decision_threshold']:.2f}",
            f"Model: {pred['model_display_name']}",
        )
        m3.metric(
            "Derived Household Size & Dependency Ratio",
            f"{norm['household_size']} members",
            f"Dependency Ratio = {norm['household_dependency_ratio']:.2f}",
        )
        m4.metric(
            f"Auto-Looked Up Census 2023 ({province_territory})",
            f"{norm['regional_pop_density']:.1f} / km²",
            f"Urban Share = {norm['regional_urban_share']:.1%} | Avg HH = {norm['regional_avg_household_size']:.2f}",
        )

        if pred["predicted_class"] == 1:
            st.warning(
                f"**Model Decision (`prob >= {pred['decision_threshold']:.2f}`):** {pred['decision_summary']}.  \n"
                f"*{pred['cautious_note']}*"
            )
        else:
            st.success(
                f"**Model Decision (`prob < {pred['decision_threshold']:.2f}`):** {pred['decision_summary']}.  \n"
                f"*{pred['cautious_note']}*"
            )

        st.subheader("4. Model Explanations: Local Profile Breakdown & Global Feature Importance (§11)")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("##### A. Local Feature Contributions for This Profile vs. Training Reference")
            local_list = pred["local_explanation"]["top_contributions"]
            ldf = pd.DataFrame(local_list).iloc[::-1]
            fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
            colors = ["#d95f02" if v >= 0 else "#2b5c8f" for v in ldf["probability_contribution"]]
            ax.barh(ldf["display_label"], ldf["probability_contribution"] * 100.0, color=colors, edgecolor="black")
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Marginal Shift in Estimated Unemployment Probability (Percentage Points)")
            ax.set_title("Local Counterfactual Feature Shift vs. Training Baseline")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with ec2:
            st.markdown("##### B. Global Validation Permutation Importance (All 15 Features)")
            imp_fig_path = REPO_ROOT / "reports/figures/08_global_permutation_importance.png"
            if imp_fig_path.exists():
                st.image(str(imp_fig_path), use_container_width=True)


def render_population_explorer_view(artifacts: Dict[str, Any]) -> None:
    st.title("🗺️ Population Pattern Explorer (PBS LFS 2024–25 + Census 2023)")
    st.markdown(
        "Explore **survey-weighted** and **unweighted sample** youth unemployment rates across provinces, "
        "urban/rural localities, and gender. Cells with fewer than **30 respondents** are suppressed per our "
        "disclosure/stability policy (§7)."
    )

    agg = artifacts.get("aggregates", {})
    geo_records = agg.get("province_locality_sex", [])
    if geo_records:
        geo_df = pd.DataFrame(geo_records)
        f1, f2, f3 = st.columns(3)
        sel_provs = f1.multiselect(
            "Filter Province / Territory:",
            options=list(SUPPORTED_PROVINCES),
            default=list(SUPPORTED_PROVINCES),
        )
        sel_locs = f2.multiselect(
            "Filter Locality:",
            options=list(SUPPORTED_LOCALITIES),
            default=list(SUPPORTED_LOCALITIES),
        )
        sel_sex = f3.multiselect(
            "Filter Sex / Gender:",
            options=["Male", "Female"],
            default=["Male", "Female"],
        )

        sub = geo_df[
            geo_df["province_territory"].isin(sel_provs)
            & geo_df["locality"].isin(sel_locs)
            & geo_df["sex"].isin(sel_sex)
        ].copy()
        st.subheader("1. Subgroup Summary Table (With Visible Sample Denominators `n`)")
        st.dataframe(sub, use_container_width=True, hide_index=True)

    st.subheader("2. Training-Partition Development EDA Figures (§7)")
    fig_dir = REPO_ROOT / "reports/figures"
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(fig_dir / "01_class_balance.png"), use_container_width=True)
        st.image(str(fig_dir / "03_numeric_distributions.png"), use_container_width=True)
        st.image(str(fig_dir / "05_numeric_correlation_heatmap.png"), use_container_width=True)
    with c2:
        st.image(str(fig_dir / "02_feature_missingness.png"), use_container_width=True)
        st.image(str(fig_dir / "04_numeric_boxplots_outliers.png"), use_container_width=True)
        st.image(str(fig_dir / "07_province_locality_sex_training_summary.png"), use_container_width=True)


def render_skills_and_education_view(artifacts: Dict[str, Any]) -> None:
    st.title("🎓 Descriptive Skills, Vocational Training & Education Analysis")
    st.warning(
        "**Methodological & Leakage Note (§5.3, §11):** In the PBS Labour Force Survey 2024–25, formal education "
        "(`highest_education`) and `current_enrolment` are pre-outcome predictors included in our 15-feature model. "
        "By contrast, raw vocational training status (`On-the-Job Training` vs. `Prior Training`) is **excluded from "
        "the predictive classifier** to prevent job-timed target leakage, and is analysed here strictly as a "
        "retrospective descriptive summary. Employer vacancy demand is not observed in LFS/Census data."
    )

    fig_dir = REPO_ROOT / "reports/figures"
    st.image(str(fig_dir / "06_education_enrolment_vs_unemployment.png"), use_container_width=True)

    agg = artifacts.get("aggregates", {})
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Unemployment by Formal Education & Enrolment")
        if "education_enrolment" in agg:
            st.dataframe(pd.DataFrame(agg["education_enrolment"]), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("2. Descriptive Vocational Training Coverage & Outcomes")
        if "vocational_training" in agg:
            st.dataframe(pd.DataFrame(agg["vocational_training"]), use_container_width=True, hide_index=True)

    if "training_field_summary" in agg:
        st.subheader("3. Technical / Vocational Training Fields (Sample Counts & Employment Associations)")
        st.dataframe(pd.DataFrame(agg["training_field_summary"]), use_container_width=True, hide_index=True)


def render_model_evidence_view(artifacts: Dict[str, Any]) -> None:
    st.title("🔬 Frozen Model Comparison, Bootstrap CIs & Leakage Protocol")
    te = artifacts["test_eval"]
    tv = artifacts["train_val"]
    boot = te["bootstrap_95ci"]

    st.subheader(
        f"1. One-Time Frozen Test Evaluation Table (Test n = {te['test_row_count']:,}, "
        f"{te['test_psu_count']} Unseen PSUs, Prevalence = {te['test_prevalence']:.2%})"
    )

    rows = []
    for m_key in (
        "dummy_prior",
        "dummy_all_positive",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ):
        m = te["models"][m_key]
        tm = m["test_metrics_at_frozen_threshold"]
        f1_ci = boot["model_intervals"][m_key]["f1_pos"]
        ap_ci = boot["model_intervals"][m_key]["average_precision"]
        rows.append(
            {
                "Model": m["display_name"] + (" ⭐ [Winner]" if m_key == te["winner_model_key"] else ""),
                "Train-CV AP (Mean ± SD)": f"{m['train_cv_ap_mean']:.4f} ± {m['train_cv_ap_std']:.4f}",
                "Frozen Val Threshold": f"{m['frozen_validation_threshold']:.2f}",
                "Test F1 (Class 1)": f"{tm['f1_pos']:.4f} [{f1_ci['ci95_low']:.4f}, {f1_ci['ci95_high']:.4f}]",
                "Test Precision / Recall": f"{tm['precision_pos']:.4f} / {tm['recall_pos']:.4f}",
                "Test AP (95% CI)": f"{tm['average_precision']:.4f} [{ap_ci['ci95_low']:.4f}, {ap_ci['ci95_high']:.4f}]",
                "Test ROC-AUC": f"{tm['roc_auc']:.4f}",
                "Test Brier": f"{tm['brier_score']:.4f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    fig_dir = REPO_ROOT / "reports/figures"
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(fig_dir / "09_test_pr_and_roc_curves.png"), use_container_width=True)
    with c2:
        st.image(str(fig_dir / "10_test_confusion_and_calibration.png"), use_container_width=True)

    st.subheader("2. Preplanned Training-CV Ablations & Subgroup Test Audit")
    ac1, ac2 = st.columns([1.0, 1.3])
    with ac1:
        st.markdown("##### Preplanned Training-CV Ablations (§9.2)")
        st.dataframe(pd.DataFrame(tv["ablations"]), use_container_width=True, hide_index=True)
        st.markdown("##### Survey-Weighted Test Sensitivity (§10.4)")
        st.json(te["subgroup_and_weighted_audit"]["survey_weighted_sensitivity"])
    with ac2:
        st.image(str(fig_dir / "11_test_subgroup_f1_ap.png"), use_container_width=True)

    st.subheader("3. Strict Leakage Prevention & Grouped Split Protocol (§6)")
    st.markdown(
        """
        - **PSU-Disjoint Partitioning (`PCode`):** All households `(PCode, HHNo)` and individuals `(PCode, HHNo, SNo)` within the same 8-digit Primary Sampling Unit stay strictly together across `train` (`1,260` PSUs), `val` (`270` PSUs), and `test` (`270` PSUs), as well as across all 5 `StratifiedGroupKFold` CV splits.
        - **Target & Post-Outcome Denylist:** All target-defining survey questions (`worked_last_week`, `temp_absent_from_work`, `seeking_work`, `available_for_work`), post-outcome fields (`hours_worked`, `monthly_earnings`, `unemployment_duration`), constant `Fstatus`, survey weights, and job-timed training codes are blocked by `HunarLensSchemaAndCensusTransformer`.
        - **Fold-Local Preprocessing:** Numeric median imputation, `StandardScaler`, and `OneHotEncoder(handle_unknown="ignore")` are fitted strictly inside each training fold.
        """
    )


def main() -> None:
    predictor = load_cached_predictor()
    artifacts = load_dashboard_artifacts()

    st.sidebar.title("📊 HunarLens Navigation")
    st.sidebar.caption("CT-378 Machine Learning · Group AlphaDevs")
    view = st.sidebar.radio(
        "Select Dashboard View:",
        options=[
            "1. Overview",
            "2. Profile Explorer",
            "3. Population Explorer",
            "4. Skills and Education",
            "5. Model Evidence",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Active Artifact:** `{predictor.metadata['selected_model_display_name']}`  \n"
        f"**Saved Threshold:** `{predictor.decision_threshold:.2f}`  \n"
        f"**Survey Period:** `PBS LFS 2024–25 + Census 2023`"
    )

    if view.startswith("1."):
        render_overview_view(predictor, artifacts)
    elif view.startswith("2."):
        render_profile_explorer_view(predictor, artifacts)
    elif view.startswith("3."):
        render_population_explorer_view(artifacts)
    elif view.startswith("4."):
        render_skills_and_education_view(artifacts)
    else:
        render_model_evidence_view(artifacts)


if __name__ == "__main__":
    main()
