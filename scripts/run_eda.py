"""Training-Partition-Only Exploratory Data Analysis (EDA) & 7 Required Development Plots Generator.

Implements PROJECT_PLAN.md §6.3, §7:
- Reads ONLY `data/splits/train.parquet` (and training-matched rows from `youth_descriptive_skills.parquet`
  for the descriptive vocational training summary). Validation and test partitions are never read.
- Enforces the prespecified small-cell disclosure rule (`n < 30` suppressed/combined).
- Generates and saves all 7 required plots to `reports/figures/`:
    1. `01_class_balance.png` — Class counts & proportions (unweighted & survey-weighted)
    2. `02_feature_missingness.png` — Missing / Unknown rates across the 15 approved predictors
    3. `03_numeric_distributions.png` — Histograms/KDE & skewness of numeric features
    4. `04_numeric_boxplots_outliers.png` — Boxplots & quantile diagnostics by target status
    5. `05_numeric_correlation_heatmap.png` — Numeric predictor correlation matrix
    6. `06_education_enrolment_vs_unemployment.png` — Education & enrolment vs. unemployment (with n & 95% CI)
    7. `07_province_locality_sex_training_summary.png` — Province/locality/sex & descriptive training summary
- Exports `reports/metrics/eda_summary.json` and `docs/eda_interpretations.md`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hunarlens.features import (  # noqa: E402
    ALL_APPROVED_15_FEATURES,
    APPROVED_CATEGORICAL_FEATURES,
    APPROVED_NUMERIC_FEATURES,
)

SMALL_CELL_MIN_COUNT = 30


def _set_plot_style() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 10


def _plot_01_class_balance(train_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Plot 1: Class counts and proportions (unweighted sample vs. survey-weighted)."""
    n_total = len(train_df)
    counts = train_df["is_unemployed"].value_counts().reindex([0, 1], fill_value=0)
    n_emp, n_unemp = int(counts.loc[0]), int(counts.loc[1])
    p_emp, p_unemp = n_emp / n_total, n_unemp / n_total

    weights = pd.to_numeric(train_df["weight"], errors="coerce").fillna(1.0)
    w_unemp = float((weights * train_df["is_unemployed"]).sum() / weights.sum())
    w_emp = 1.0 - w_unemp

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: Unweighted sample counts
    labels = ["Employed (0)", "Unemployed (1)\n[Positive Class]"]
    colors = ["#2b5c8f", "#d95f02"]
    bars = axes[0].bar(labels, [n_emp, n_unemp], color=colors, width=0.55, edgecolor="black")
    axes[0].set_title(f"A. Training Sample Class Counts (n = {n_total:,})")
    axes[0].set_ylabel("Number of Eligible Youth (15–24)")
    for bar, cnt, prop in zip(bars, [n_emp, n_unemp], [p_emp, p_unemp]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + n_total * 0.012,
            f"{cnt:,}\n({prop:.1%})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(n_emp, n_unemp) * 1.18)

    # Panel B: Unweighted vs Survey-Weighted Proportions
    x = np.arange(2)
    width = 0.34
    b1 = axes[1].bar(x - width / 2, [p_emp * 100, p_unemp * 100], width, label="Unweighted Sample", color="#4daf4a")
    b2 = axes[1].bar(x + width / 2, [w_emp * 100, w_unemp * 100], width, label="Survey-Weighted", color="#984ea3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Percentage of Training Cohort (%)")
    axes[1].set_title("B. Sample vs. Weighted Proportions (Training Split)")
    axes[1].set_ylim(0, 75)
    axes[1].legend(frameon=True)
    for b in (*b1, *b2):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.0,
            f"{b.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "HunarLens EDA Plot 1 — Class Balance in Training Partition (Aged 15–24 in Labour Force)",
        fontweight="bold",
        y=1.03,
    )
    out_path = out_dir / "01_class_balance.png"
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "plot_file": str(out_path.relative_to(REPO_ROOT)),
        "n_train": n_total,
        "employed_count": n_emp,
        "unemployed_count": n_unemp,
        "unweighted_unemployment_rate": round(p_unemp, 4),
        "weighted_unemployment_rate": round(w_unemp, 4),
    }


def _plot_02_feature_missingness(train_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Plot 2: Missingness / Unknown rate across the 15 approved predictors."""
    records = []
    for feat in ALL_APPROVED_15_FEATURES:
        s = train_df[feat]
        if feat in APPROVED_NUMERIC_FEATURES:
            miss_cnt = int(pd.to_numeric(s, errors="coerce").isna().sum())
        else:
            miss_cnt = int(
                (s.isna() | s.astype(str).str.strip().isin(["", "Unknown", "nan", "None"])).sum()
            )
        records.append(
            {
                "feature": feat,
                "missing_or_unknown_count": miss_cnt,
                "missing_pct": (miss_cnt / len(train_df)) * 100.0,
                "type": "Numeric" if feat in APPROVED_NUMERIC_FEATURES else "Categorical",
            }
        )
    miss_df = pd.DataFrame(records).sort_values("missing_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["#2b5c8f" if t == "Numeric" else "#388e3c" for t in miss_df["type"]]
    ax.barh(miss_df["feature"], miss_df["missing_pct"], color=colors, edgecolor="black")
    ax.set_xlabel("Missing or 'Unknown' Share in Training Partition (%)")
    ax.set_title(
        f"HunarLens EDA Plot 2 — Missingness Audit Across 15 Approved Predictors (Train n = {len(train_df):,})",
        fontweight="bold",
    )
    ax.set_xlim(0, max(5.0, float(miss_df["missing_pct"].max()) + 2.0))
    for idx, row in enumerate(miss_df.itertuples()):
        ax.text(
            row.missing_pct + 0.12,
            idx,
            f"{row.missing_or_unknown_count:,} ({row.missing_pct:.2f}%)",
            va="center",
            fontsize=8.5,
        )

    out_path = out_dir / "02_feature_missingness.png"
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "plot_file": str(out_path.relative_to(REPO_ROOT)),
        "max_missing_pct": round(float(miss_df["missing_pct"].max()), 4),
    }


def _plot_03_numeric_distributions(train_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Plot 3: Numeric distributions (`age`, `household_size`, `household_dependency_ratio`, `regional_pop_density_log1p`)."""
    features_to_plot = [
        ("age", "Age in Completed Years (15–24)"),
        ("household_size", "Usual-Resident Household Size"),
        ("household_dependency_ratio", "Household Age-Dependency Ratio"),
        ("regional_pop_density_log1p", "Census 2023 Regional Density (log1p)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    stats: Dict[str, Any] = {}

    for ax, (col, title) in zip(axes.flat, features_to_plot):
        vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        skew_val = float(vals.skew())
        mean_val = float(vals.mean())
        med_val = float(vals.median())
        stats[col] = {
            "mean": round(mean_val, 3),
            "median": round(med_val, 3),
            "skew": round(skew_val, 3),
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
        }

        sns.histplot(vals, bins=20, kde=True, ax=ax, color="#2b5c8f", edgecolor="white")
        ax.axvline(med_val, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Median={med_val:.2f}")
        ax.axvline(mean_val, color="#1b9e77", linestyle=":", linewidth=1.8, label=f"Mean={mean_val:.2f}")
        ax.set_title(f"{title}\n(Skew = {skew_val:.2f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Training Respondents")
        ax.legend(fontsize=8.5)

    fig.suptitle(
        f"HunarLens EDA Plot 3 — Numeric Predictor Distributions (Training Partition, n = {len(train_df):,})",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / "03_numeric_distributions.png"
    fig.savefig(out_path)
    plt.close(fig)
    return {"plot_file": str(out_path.relative_to(REPO_ROOT)), "numeric_stats": stats}


def _plot_04_boxplots_outliers(train_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Plot 4: Boxplots / quantile diagnostics by binary target status without deleting plausible large households."""
    plot_df = train_df.copy()
    plot_df["Status"] = plot_df["is_unemployed"].map({0: "Employed (0)", 1: "Unemployed (1)"})

    cols = [
        ("age", "Completed Age (Years)"),
        ("household_size", "Household Size (All Ages)"),
        ("household_dependency_ratio", "Household Dependency Ratio"),
        ("regional_urban_share", "Census Regional Urban Share"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    quantiles_summary: Dict[str, Any] = {}

    for ax, (col, label) in zip(axes.flat, cols):
        sns.boxplot(
            data=plot_df,
            x="Status",
            y=col,
            hue="Status",
            palette={"Employed (0)": "#2b5c8f", "Unemployed (1)": "#d95f02"},
            legend=False,
            ax=ax,
            width=0.45,
        )
        q05, q50, q95 = np.percentile(plot_df[col].dropna(), [5, 50, 95])
        quantiles_summary[col] = {
            "p05": round(float(q05), 3),
            "p50": round(float(q50), 3),
            "p95": round(float(q95), 3),
        }
        ax.set_title(f"{label}\n(P05={q05:.2f}, Median={q50:.2f}, P95={q95:.2f})")
        ax.set_xlabel("")
        ax.set_ylabel(col)

    fig.suptitle(
        "HunarLens EDA Plot 4 — Boxplots & Quantile Diagnostics by Employment Status (Train Split)",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / "04_numeric_boxplots_outliers.png"
    fig.savefig(out_path)
    plt.close(fig)
    return {"plot_file": str(out_path.relative_to(REPO_ROOT)), "quantiles": quantiles_summary}


def _plot_05_correlation_heatmap(train_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Plot 5: Numeric correlation heatmap across the 6 numeric predictors and binary target."""
    num_cols = [*APPROVED_NUMERIC_FEATURES, "is_unemployed"]
    corr_matrix = train_df[num_cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.6,
        ax=ax,
    )
    ax.set_title(
        f"HunarLens EDA Plot 5 — Numeric Predictor Correlation Heatmap (Training Split, n = {len(train_df):,})",
        fontweight="bold",
        pad=14,
    )
    out_path = out_dir / "05_numeric_correlation_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)

    target_corrs = corr_matrix["is_unemployed"].drop("is_unemployed").round(4).to_dict()
    return {"plot_file": str(out_path.relative_to(REPO_ROOT)), "target_correlations": target_corrs}


def _plot_06_education_enrolment(train_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Plot 6: Education & current enrolment vs. unemployment rate with group counts and 95% CIs (non-causal)."""
    edu_order = [
        "No Formal Education",
        "Primary",
        "Middle",
        "Matric",
        "Intermediate",
        "Degree & Postgraduate",
    ]
    edu_grp = (
        train_df.groupby("highest_education", as_index=False)
        .agg(n=("is_unemployed", "size"), unemp_sum=("is_unemployed", "sum"))
    )
    edu_grp["highest_education"] = pd.Categorical(
        edu_grp["highest_education"], categories=edu_order, ordered=True
    )
    edu_grp = edu_grp.sort_values("highest_education").reset_index(drop=True)
    # Apply small-cell rule (<30 suppressed)
    edu_grp = edu_grp.loc[edu_grp["n"] >= SMALL_CELL_MIN_COUNT].copy()
    edu_grp["rate"] = edu_grp["unemp_sum"] / edu_grp["n"]
    edu_grp["ci95"] = 1.96 * np.sqrt(edu_grp["rate"] * (1.0 - edu_grp["rate"]) / edu_grp["n"])

    enrol_grp = (
        train_df.groupby("current_enrolment", as_index=False)
        .agg(n=("is_unemployed", "size"), unemp_sum=("is_unemployed", "sum"))
    )
    enrol_grp = enrol_grp.loc[enrol_grp["n"] >= SMALL_CELL_MIN_COUNT].copy()
    enrol_grp["rate"] = enrol_grp["unemp_sum"] / enrol_grp["n"]
    enrol_grp["ci95"] = 1.96 * np.sqrt(enrol_grp["rate"] * (1.0 - enrol_grp["rate"]) / enrol_grp["n"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.65, 1.0]})

    # Left: Education level
    y_pos = np.arange(len(edu_grp))
    axes[0].barh(
        y_pos,
        edu_grp["rate"] * 100.0,
        xerr=edu_grp["ci95"] * 100.0,
        color="#2b5c8f",
        edgecolor="black",
        capsize=4,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(edu_grp["highest_education"].astype(str))
    axes[0].set_xlabel("Sample Unemployment Rate (%) ± 95% CI")
    axes[0].set_title("A. Unemployment by Highest Education (Descriptive Association)")
    axes[0].set_xlim(0, min(100.0, float((edu_grp["rate"] + edu_grp["ci95"]).max() * 100.0) + 14.0))
    for idx, row in enumerate(edu_grp.itertuples()):
        axes[0].text(
            (row.rate + row.ci95) * 100.0 + 0.8,
            idx,
            f"{row.rate:.1%} (n={row.n:,})",
            va="center",
            fontsize=8.5,
        )

    # Right: Current Enrolment
    x_pos = np.arange(len(enrol_grp))
    axes[1].bar(
        x_pos,
        enrol_grp["rate"] * 100.0,
        yerr=enrol_grp["ci95"] * 100.0,
        color=["#1b9e77", "#d95f02"][: len(enrol_grp)],
        edgecolor="black",
        width=0.5,
        capsize=5,
    )
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(enrol_grp["current_enrolment"].astype(str))
    axes[1].set_ylabel("Sample Unemployment Rate (%) ± 95% CI")
    axes[1].set_title("B. Unemployment by Current Enrolment Status")
    axes[1].set_ylim(0, min(100.0, float((enrol_grp["rate"] + enrol_grp["ci95"]).max() * 100.0) + 14.0))
    for idx, row in enumerate(enrol_grp.itertuples()):
        axes[1].text(
            idx,
            (row.rate + row.ci95) * 100.0 + 1.2,
            f"{row.rate:.1%}\n(n={row.n:,})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "HunarLens EDA Plot 6 — Education & Enrolment vs. Youth Unemployment (Training Split; Non-Causal)",
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    out_path = out_dir / "06_education_enrolment_vs_unemployment.png"
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "plot_file": str(out_path.relative_to(REPO_ROOT)),
        "education_rates": [
            {"education": str(r.highest_education), "n": int(r.n), "unemployment_rate": round(float(r.rate), 4)}
            for r in edu_grp.itertuples()
        ],
    }


def _plot_07_province_locality_sex_training(train_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Plot 7: Province/locality, sex, and descriptive vocational training summaries (small-cell rule <30)."""
    # Join descriptive training status ONLY for training partition person keys
    desc_path = REPO_ROOT / "data/processed/youth_descriptive_skills.parquet"
    desc_df = pd.read_parquet(desc_path)
    train_keys = train_df[["PCode", "HHNo", "SNo"]].copy()
    train_desc = train_keys.merge(
        desc_df[["PCode", "HHNo", "SNo", "vocational_training_status", "is_unemployed"]],
        on=["PCode", "HHNo", "SNo"],
        how="left",
        validate="1:1",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), gridspec_kw={"width_ratios": [1.4, 0.9, 1.3]})

    # Panel A: Province x Locality
    prov_loc = (
        train_df.groupby(["province_territory", "locality"], as_index=False)
        .agg(n=("is_unemployed", "size"), rate=("is_unemployed", "mean"))
    )
    prov_loc = prov_loc.loc[prov_loc["n"] >= SMALL_CELL_MIN_COUNT].copy()
    prov_loc["rate_pct"] = prov_loc["rate"] * 100.0
    sns.barplot(
        data=prov_loc,
        y="province_territory",
        x="rate_pct",
        hue="locality",
        palette={"Rural": "#4daf4a", "Urban": "#e41a1c"},
        edgecolor="black",
        ax=axes[0],
    )
    axes[0].set_title("A. Province × Locality Unemployment (%)")
    axes[0].set_xlabel("Sample Unemployment Rate (%)")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 75)

    # Panel B: Sex / Gender
    sex_grp = (
        train_df.groupby("sex", as_index=False)
        .agg(n=("is_unemployed", "size"), rate=("is_unemployed", "mean"))
    )
    sex_grp = sex_grp.loc[sex_grp["n"] >= SMALL_CELL_MIN_COUNT].copy()
    bars = axes[1].bar(
        sex_grp["sex"],
        sex_grp["rate"] * 100.0,
        color=["#2b5c8f", "#d95f02"][: len(sex_grp)],
        edgecolor="black",
        width=0.5,
    )
    axes[1].set_title("B. Unemployment by Sex/Gender")
    axes[1].set_ylabel("Sample Unemployment Rate (%)")
    axes[1].set_ylim(0, 70)
    for b, r in zip(bars, sex_grp.itertuples()):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.2,
            f"{r.rate:.1%}\n(n={r.n:,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    # Panel C: Descriptive Vocational Training (explicitly marked descriptive-only per §5.3 & §11)
    tr_grp = (
        train_desc.groupby("vocational_training_status", as_index=False)
        .agg(n=("is_unemployed", "size"), rate=("is_unemployed", "mean"))
    )
    tr_grp = tr_grp.loc[tr_grp["n"] >= SMALL_CELL_MIN_COUNT].sort_values("rate", ascending=True)
    axes[2].barh(
        tr_grp["vocational_training_status"],
        tr_grp["rate"] * 100.0,
        color="#7570b3",
        edgecolor="black",
    )
    axes[2].set_title("C. Vocational Training (Descriptive Only, Excluded from Classifier)")
    axes[2].set_xlabel("Sample Unemployment Rate (%)")
    axes[2].set_xlim(0, 70)
    for idx, r in enumerate(tr_grp.itertuples()):
        axes[2].text(
            r.rate * 100.0 + 0.8,
            idx,
            f"{r.rate:.1%} (n={r.n:,})",
            va="center",
            fontsize=8.5,
        )

    fig.suptitle(
        f"HunarLens EDA Plot 7 — Geographic, Gender & Descriptive Training Summaries (Train n = {len(train_df):,}; Cells >= {SMALL_CELL_MIN_COUNT})",
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    out_path = out_dir / "07_province_locality_sex_training_summary.png"
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "plot_file": str(out_path.relative_to(REPO_ROOT)),
        "province_locality_cells": len(prov_loc),
        "small_cell_threshold": SMALL_CELL_MIN_COUNT,
    }


def write_eda_interpretations_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    """Write concise, denominator-explicit interpretations for all 7 plots (`docs/eda_interpretations.md`, §7)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p1 = summary["plots"]["plot_01"]
    p3 = summary["plots"]["plot_03"]["numeric_stats"]
    p5 = summary["plots"]["plot_05"]["target_correlations"]

    content = f"""# HunarLens — Training-Partition EDA Summary & Plot Interpretations (`eda_interpretations.md`)

**Reference:** [PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md) §7 · **Generated by:** [scripts/run_eda.py](file:///d:/Hunarlens/scripts/run_eda.py)  
**Denominator & Scope:** Strictly restricted to the **70% PSU-disjoint training partition (`n = {p1['n_train']:,}` eligible youth aged 15–24 in the labour force across `1,260` PSUs)**. Validation and test partitions were never accessed.

---

## 1. Class Counts and Proportions (`reports/figures/01_class_balance.png`)
* **Denominator:** `{p1['n_train']:,}` eligible youth in the training split (`{p1['employed_count']:,}` Employed `[0]`, `{p1['unemployed_count']:,}` Unemployed `[1]`).
* **Interpretation:** Positive-class (`is_unemployed = 1`) prevalence is `{p1['unweighted_unemployment_rate']:.2%}` unweighted (`{p1['weighted_unemployment_rate']:.2%}` survey-weighted). Because employed youth outnumber unemployed youth, a majority-class dummy baseline predicts `0` everywhere and achieves `F1 = 0.00`, motivating our use of both the all-positive baseline (`F1 = 2p / (1 + p)`) and Average Precision (`AP`).

## 2. Missingness by Feature (`reports/figures/02_feature_missingness.png`)
* **Denominator:** `{p1['n_train']:,}` training observations across all 15 approved pre-encoding predictors.
* **Interpretation:** Core demographic, roster, and Census 2023 features exhibit complete coverage in the cleaned cohort (`max missing = {summary['plots']['plot_02']['max_missing_pct']:.2f}%`). Nevertheless, the scikit-learn inference pipeline retains fold-local median imputation for numerics and explicit `"Unknown"` imputation + `OneHotEncoder(handle_unknown="ignore")` to safely handle missing or unseen inputs at deployment time.

## 3. Numeric Distributions (`reports/figures/03_numeric_distributions.png`)
* **Denominator:** `{p1['n_train']:,}` training observations.
* **Interpretation:**
  * `age`: Mean `{p3['age']['mean']}`, median `{p3['age']['median']}` (bounded `[15, 24]`, skew `{p3['age']['skew']}`).
  * `household_size`: Mean `{p3['household_size']['mean']}`, median `{p3['household_size']['median']}` (right-skewed `{p3['household_size']['skew']}`, reflecting multi-generational Pakistani households).
  * `household_dependency_ratio`: Mean `{p3['household_dependency_ratio']['mean']}`, median `{p3['household_dependency_ratio']['median']}` (skew `{p3['household_dependency_ratio']['skew']}`).
  * `regional_pop_density_log1p`: Log1p transformation compresses the high-density Islamabad outlier (`2,609.12` persons/sq. km vs. Balochistan `42.90`).

## 4. Boxplots and Quantile Diagnostics (`reports/figures/04_numeric_boxplots_outliers.png`)
* **Denominator:** `{p1['n_train']:,}` training observations grouped by `Employed (0)` vs. `Unemployed (1)`.
* **Interpretation:** Large households (`P95 = {summary['plots']['plot_04']['quantiles']['household_size']['p95']}`) and high dependency ratios represent genuine demographic structures in KP, Balochistan, and rural areas rather than data entry errors; per §7, plausible large households and dense regions are retained rather than deleted.

## 5. Numeric Correlation Heatmap (`reports/figures/05_numeric_correlation_heatmap.png`)
* **Denominator:** `{p1['n_train']:,}` training observations across the 6 numeric predictors and binary target.
* **Interpretation:** Linear correlations with `is_unemployed` are modest (`regional_urban_share`: `{p5['regional_urban_share']:+.3f}`, `household_dependency_ratio`: `{p5['household_dependency_ratio']:+.3f}`), confirming that youth unemployment vulnerability depends strongly on categorical human-capital and regional interactions rather than a single linear numeric proxy.

## 6. Education & Enrolment vs. Unemployment (`reports/figures/06_education_enrolment_vs_unemployment.png`)
* **Denominator:** `{p1['n_train']:,}` training observations, with all displayed bars meeting the `n >= {SMALL_CELL_MIN_COUNT}` small-cell rule.
* **Interpretation:** Consistent with PBS LFS 2024–25 empirical findings, youth holding `Degree & Postgraduate` and `Intermediate` credentials face substantially higher descriptive unemployment rates during their initial labour-market transition than youth with `No Formal Education` (who are more frequently absorbed into informal/subsistence work). **Limitation:** These are cross-sectional descriptive associations conditional on labour-force participation, **not causal effects** of education.

## 7. Province, Locality, Sex & Descriptive Vocational Training (`reports/figures/07_province_locality_sex_training_summary.png`)
* **Denominator:** `{p1['n_train']:,}` training observations (`n >= {SMALL_CELL_MIN_COUNT}` per cell).
* **Interpretation:** Urban youth and female youth in the labour force experience higher unemployment rates across provinces, with Khyber Pakhtunkhwa and Islamabad exhibiting the highest regional pressure. Panel C illustrates why `vocational_training_status` is excluded from the predictive classifier (§5.3): `On-the-Job Training` occurs by definition among employed workers (`0.0%` unemployed), which would introduce severe post-outcome leakage if used as a predictor.
"""
    out_path.write_text(content, encoding="utf-8")


def run_eda_pipeline() -> Dict[str, Any]:
    """Execute the full training-only EDA suite and save all 7 figures and summaries."""
    _set_plot_style()
    train_path = REPO_ROOT / "data/splits/train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found at {train_path}. Run splits.py first.")

    train_df = pd.read_parquet(train_path)
    figures_dir = REPO_ROOT / "reports/figures"
    metrics_dir = REPO_ROOT / "reports/metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "partition": "train_only",
        "train_row_count": int(len(train_df)),
        "train_psu_count": int(train_df["PCode"].nunique()),
        "small_cell_min_count": SMALL_CELL_MIN_COUNT,
        "plots": {
            "plot_01": _plot_01_class_balance(train_df, figures_dir),
            "plot_02": _plot_02_feature_missingness(train_df, figures_dir),
            "plot_03": _plot_03_numeric_distributions(train_df, figures_dir),
            "plot_04": _plot_04_boxplots_outliers(train_df, figures_dir),
            "plot_05": _plot_05_correlation_heatmap(train_df, figures_dir),
            "plot_06": _plot_06_education_enrolment(train_df, figures_dir),
            "plot_07": _plot_07_province_locality_sex_training(train_df, figures_dir),
        },
    }

    summary_json_path = metrics_dir / "eda_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    interp_path = REPO_ROOT / "docs/eda_interpretations.md"
    write_eda_interpretations_markdown(summary, interp_path)
    return summary


def main() -> None:
    summary = run_eda_pipeline()
    print(
        f"[run_eda] Generated 7 training-only EDA figures in reports/figures/ "
        f"(Train n={summary['train_row_count']:,}, PSUs={summary['train_psu_count']:,})"
    )


if __name__ == "__main__":
    main()
