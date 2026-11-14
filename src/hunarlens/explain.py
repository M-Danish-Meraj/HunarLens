"""Global Permutation Importance and Local Profile Explanation Module.

Implements PROJECT_PLAN.md §11:
- Global explanations: Calculates validation permutation importance across the entire fitted scikit-learn
  `Pipeline` using `average_precision` (`n_repeats=10`, `random_state=45`), perturbing the 15 raw conceptual
  features BEFORE one-hot encoding so multi-category variables are evaluated as unified concepts.
- Local explanations: Decomposes a single profile's prediction relative to the training reference baseline
  across the 15 conceptual predictors, supporting both linear log-odds contributions (`LogisticRegression`)
  and exact marginal counterfactual feature contributions relative to the training reference distribution
  for tree ensembles (`GradientBoostingClassifier`, `RandomForestClassifier`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from hunarlens.features import ALL_APPROVED_15_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURE_HUMAN_LABELS: Dict[str, str] = {
    "age": "Age (Completed Years, 15–24)",
    "sex": "Sex / Gender",
    "marital_status": "Marital Status",
    "relationship_to_head": "Relationship to Household Head",
    "can_read": "Literacy: Can Read",
    "can_write": "Literacy: Can Write",
    "highest_education": "Highest Formal Education",
    "current_enrolment": "Current Educational Enrolment",
    "household_size": "Household Size (All Ages)",
    "household_dependency_ratio": "Household Age-Dependency Ratio",
    "province_territory": "Province / Federal Territory",
    "locality": "Urban / Rural Locality",
    "regional_pop_density_log1p": "Census 2023 Regional Pop. Density (log1p)",
    "regional_urban_share": "Census 2023 Regional Urban Share",
    "regional_avg_household_size": "Census 2023 Regional Avg. Household Size",
}


def compute_global_permutation_importance(
    pipeline: Any,
    val_df: pd.DataFrame,
    target_col: str = "is_unemployed",
    n_repeats: int = 10,
    random_state: int = 45,
) -> pd.DataFrame:
    """Compute validation permutation importance on the 15 raw conceptual features using Average Precision (§11)."""
    X_val = val_df[list(ALL_APPROVED_15_FEATURES)].copy()
    y_val = val_df[target_col].to_numpy(dtype=int)

    perm = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    records = []
    for idx, feat in enumerate(ALL_APPROVED_15_FEATURES):
        records.append(
            {
                "feature": feat,
                "display_label": FEATURE_HUMAN_LABELS.get(feat, feat),
                "importance_mean": float(perm.importances_mean[idx]),
                "importance_std": float(perm.importances_std[idx]),
                "importance_min": float(np.min(perm.importances[idx])),
                "importance_max": float(np.max(perm.importances[idx])),
            }
        )

    imp_df = (
        pd.DataFrame(records)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return imp_df


def save_global_importance_artifacts(
    imp_df: pd.DataFrame,
    fig_path: Path | None = None,
    json_path: Path | None = None,
) -> Dict[str, Any]:
    """Save the global permutation importance bar chart (PNG) and JSON summary."""
    out_fig = fig_path or (REPO_ROOT / "reports/figures/08_global_permutation_importance.png")
    out_json = json_path or (REPO_ROOT / "reports/metrics/global_feature_importance.json")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    plot_df = imp_df.sort_values("importance_mean", ascending=True).reset_index(drop=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10.2, 6.2), dpi=140)
    y_pos = np.arange(len(plot_df))
    ax.barh(
        y_pos,
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        color="#2b5c8f",
        edgecolor="black",
        capsize=3.5,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["display_label"])
    ax.set_xlabel("Mean Decrease in Validation Average Precision (AP) ± SD across 10 Permutations")
    ax.set_title(
        "HunarLens Global Model Explanation — Validation Permutation Importance (15 Conceptual Features)",
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "method": "validation_permutation_importance",
        "scoring": "average_precision",
        "figure_path": str(out_fig.relative_to(REPO_ROOT)),
        "features": imp_df.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compute_local_profile_explanation(
    pipeline: Any,
    profile_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Compute a transparent local feature contribution breakdown for a single profile (§11).

    For each of the 15 conceptual predictors, evaluates how replacing that single predictor in the
    user's profile with values from the training reference sample (`reference_df`) shifts the model's
    predicted unemployment probability:
        `contribution_i = P(y=1 | profile) - E_ref[ P(y=1 | profile with feature_i := ref_i) ]`
    Positive values indicate that the profile's value for `feature_i` increases the model's estimated
    unemployment probability relative to the training population reference; negative values indicate
    it decreases the estimated probability.
    """
    schema_step = pipeline.named_steps["schema_census"]
    clean_profile = schema_step.transform(profile_df.iloc[[0]])
    clean_ref = schema_step.transform(reference_df)

    base_prob = float(pipeline.predict_proba(clean_profile)[0, 1])
    ref_mean_prob = float(pipeline.predict_proba(clean_ref)[:, 1].mean())

    contributions: List[Dict[str, Any]] = []
    n_ref = len(clean_ref)

    for feat in ALL_APPROVED_15_FEATURES:
        # Replicate profile n_ref times and substitute only `feat` from the training reference slice
        tiled = pd.concat([clean_profile] * n_ref, ignore_index=True)
        tiled[feat] = clean_ref[feat].to_numpy()
        counterfactual_prob = float(pipeline.predict_proba(tiled)[:, 1].mean())
        delta_prob = base_prob - counterfactual_prob
        raw_val = clean_profile[feat].iloc[0]
        if isinstance(raw_val, (float, np.floating)):
            val_str = f"{float(raw_val):.2f}"
        else:
            val_str = str(raw_val)

        contributions.append(
            {
                "feature": feat,
                "display_label": f"{FEATURE_HUMAN_LABELS.get(feat, feat)} = {val_str}",
                "feature_value": val_str,
                "probability_contribution": round(float(delta_prob), 6),
                "direction": "Increases Score" if delta_prob >= 0 else "Decreases Score",
            }
        )

    contributions.sort(key=lambda r: abs(r["probability_contribution"]), reverse=True)
    return {
        "profile_probability": round(base_prob, 6),
        "reference_mean_probability": round(ref_mean_prob, 6),
        "scale_unit": "probability_shift_vs_training_reference",
        "top_contributions": contributions[:top_k],
        "all_contributions": contributions,
    }

