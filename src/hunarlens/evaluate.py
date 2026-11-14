"""Single-Batch Frozen Test Evaluation, 1,000-Resample PSU Bootstrap CIs, Subgroup Audit, and Diagnostic Plots.

Implements PROJECT_PLAN.md §10.1, §10.3, §10.4, §10.5, §11:
- Verifies `models/freeze_manifest.json` SHA-256 hashes before reading `data/splits/test.parquet`.
- Evaluates all 5 frozen comparison models (`dummy_prior`, `dummy_all_positive`, `logistic_regression`,
  `random_forest`, `gradient_boosting`) in one batch at their frozen validation-selected thresholds.
- Computes 1,000 paired bootstrap resamples of independent test PSUs (`PCode`, `seed=46`) for 95% confidence
  intervals on `F1 (class 1)`, `Average Precision (AP)`, and paired differences vs. baselines.
- Evaluates predeclared subgroups (`sex`, `age 15-19 vs 20-24`, `locality`, `province_territory`) with `<30`
  small-cell suppression, plus a separately labelled survey-weighted test sensitivity summary.
- Generates diagnostic plots (`08_global_permutation_importance.png`, `09_test_pr_and_roc_curves.png`,
  `10_test_confusion_and_calibration.png`, `11_test_subgroup_f1_ap.png`) and `docs/final_evaluation_report.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import hunarlens  # noqa: F401
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from hunarlens.data import load_yaml_config
from hunarlens.explain import (
    compute_global_permutation_importance,
    save_global_importance_artifacts,
)
from hunarlens.train import compute_binary_metrics_at_threshold, compute_file_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_CELL_MIN_COUNT = 30


def verify_freeze_manifest(freeze_path: Path, data_cfg: Dict[str, Any], exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Assert that the freeze manifest exists and split/artifact SHA-256 hashes match before touching test set."""
    if not freeze_path.exists():
        raise FileNotFoundError(
            f"Freeze manifest not found at {freeze_path}. Run `python scripts/train_models.py` first."
        )
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_TEST_EVALUATION":
        raise ValueError(f"Invalid freeze manifest status: {manifest.get('status')}")

    splits_dir = REPO_ROOT / data_cfg["paths"]["splits_dir"]
    for key, fname in (
        ("train_parquet", "train.parquet"),
        ("val_parquet", "val.parquet"),
        ("test_parquet", "test.parquet"),
    ):
        actual_hash = compute_file_sha256(splits_dir / fname)
        expected_hash = manifest["split_files_sha256"][key]
        if actual_hash != expected_hash:
            raise AssertionError(
                f"Split hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}."
            )

    final_pipe_path = REPO_ROOT / exp_cfg["evaluation"]["artifacts"]["pipeline_path"]
    if compute_file_sha256(final_pipe_path) != manifest["final_pipeline_sha256"]:
        raise AssertionError("Frozen final_pipeline.joblib hash does not match freeze_manifest.json.")

    return manifest


def run_paired_psu_bootstrap(
    test_df: pd.DataFrame,
    probs_by_model: Dict[str, np.ndarray],
    thresholds_by_model: Dict[str, float],
    winner_key: str,
    group_col: str = "PCode",
    target_col: str = "is_unemployed",
    n_resamples: int = 1000,
    seed: int = 46,
) -> Dict[str, Any]:
    """Run 1,000 paired bootstrap resamples over independent test PSUs (`PCode`, §10.4)."""
    rng = np.random.default_rng(seed)
    y_true = test_df[target_col].to_numpy(dtype=int)
    psus = test_df[group_col].astype(str).to_numpy()
    unique_psus = np.unique(psus)
    n_psus = len(unique_psus)

    # Pre-index row positions for each PSU for fast clustered resampling
    psu_to_indices: Dict[str, np.ndarray] = {
        p: np.where(psus == p)[0] for p in unique_psus
    }
    psu_index_list = [psu_to_indices[p] for p in unique_psus]

    model_keys = list(probs_by_model.keys())
    preds_by_model = {
        m: (probs_by_model[m] >= thresholds_by_model[m]).astype(int)
        for m in model_keys
    }

    f1_samples: Dict[str, List[float]] = {m: [] for m in model_keys}
    ap_samples: Dict[str, List[float]] = {m: [] for m in model_keys}
    diff_f1_vs_prior: List[float] = []
    diff_f1_vs_allpos: List[float] = []
    diff_ap_vs_prior: List[float] = []
    invalid_single_class_resamples = 0

    for _ in range(n_resamples):
        sampled_cluster_ids = rng.integers(0, n_psus, size=n_psus)
        idx = np.concatenate([psu_index_list[i] for i in sampled_cluster_ids])
        y_b = y_true[idx]

        if len(np.unique(y_b)) < 2:
            invalid_single_class_resamples += 1
            continue

        step_f1: Dict[str, float] = {}
        step_ap: Dict[str, float] = {}
        for m in model_keys:
            p_b = probs_by_model[m][idx]
            pred_b = preds_by_model[m][idx]
            f1_val = float(f1_score(y_b, pred_b, pos_label=1, zero_division=0))
            ap_val = float(average_precision_score(y_b, p_b))
            f1_samples[m].append(f1_val)
            ap_samples[m].append(ap_val)
            step_f1[m] = f1_val
            step_ap[m] = ap_val

        diff_f1_vs_prior.append(step_f1[winner_key] - step_f1["dummy_prior"])
        diff_f1_vs_allpos.append(step_f1[winner_key] - step_f1["dummy_all_positive"])
        diff_ap_vs_prior.append(step_ap[winner_key] - step_ap["dummy_prior"])

    def _ci95(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "ci95_low": round(float(np.percentile(arr, 2.5)), 6),
            "ci95_high": round(float(np.percentile(arr, 97.5)), 6),
        }

    return {
        "n_resamples_requested": n_resamples,
        "n_valid_resamples": n_resamples - invalid_single_class_resamples,
        "invalid_single_class_resamples": invalid_single_class_resamples,
        "independent_test_psus": n_psus,
        "model_intervals": {
            m: {
                "f1_pos": _ci95(f1_samples[m]),
                "average_precision": _ci95(ap_samples[m]),
            }
            for m in model_keys
        },
        "winner_paired_differences": {
            "winner_minus_dummy_prior_f1": _ci95(diff_f1_vs_prior),
            "winner_minus_dummy_all_positive_f1": _ci95(diff_f1_vs_allpos),
            "winner_minus_dummy_prior_ap": _ci95(diff_ap_vs_prior),
        },
    }


def compute_subgroup_and_weighted_audit(
    test_df: pd.DataFrame,
    winner_probs: np.ndarray,
    winner_threshold: float,
) -> Dict[str, Any]:
    """Compute predeclared subgroup metrics with `<30` small-cell suppression and weighted sensitivity (§10.4)."""
    df = test_df.copy()
    df["age_band"] = np.where(df["age"] <= 19, "15-19 (Teen Youth)", "20-24 (Young Adult)")

    subgroup_dimensions = [
        ("sex", "Sex / Gender"),
        ("age_band", "Age Group (15–19 vs. 20–24)"),
        ("locality", "Urban / Rural Locality"),
        ("province_territory", "Province / Territory"),
    ]

    subgroup_records: List[Dict[str, Any]] = []
    for col, dim_label in subgroup_dimensions:
        for val, grp in df.groupby(col, sort=False):
            n_cell = len(grp)
            pos_support = int(grp["is_unemployed"].sum())
            if n_cell < SMALL_CELL_MIN_COUNT or pos_support == 0 or pos_support == n_cell:
                subgroup_records.append(
                    {
                        "dimension": dim_label,
                        "column": col,
                        "subgroup": str(val),
                        "n": int(n_cell),
                        "positive_support": pos_support,
                        "suppressed": True,
                        "reason": f"Cell count < {SMALL_CELL_MIN_COUNT} or single-class support",
                    }
                )
            else:
                idx = grp.index.to_numpy()
                m = compute_binary_metrics_at_threshold(
                    grp["is_unemployed"].to_numpy(dtype=int),
                    winner_probs[idx],
                    threshold=winner_threshold,
                )
                subgroup_records.append(
                    {
                        "dimension": dim_label,
                        "column": col,
                        "subgroup": str(val),
                        "n": int(n_cell),
                        "positive_support": pos_support,
                        "suppressed": False,
                        "prevalence": m["prevalence"],
                        "f1_pos": m["f1_pos"],
                        "precision_pos": m["precision_pos"],
                        "recall_pos": m["recall_pos"],
                        "average_precision": m["average_precision"],
                        "roc_auc": m["roc_auc"],
                        "brier_score": m["brier_score"],
                    }
                )

    # Separately labelled survey-weighted sensitivity table (§10.4)
    y_true = df["is_unemployed"].to_numpy(dtype=int)
    y_pred = (winner_probs >= winner_threshold).astype(int)
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)

    weighted_summary = {
        "note": "Separately labelled survey-weighted test sensitivity summary (same frozen predictions & threshold).",
        "weighted_prevalence": round(float(np.average(y_true, weights=w)), 6),
        "weighted_f1_pos": round(float(f1_score(y_true, y_pred, pos_label=1, sample_weight=w, zero_division=0)), 6),
        "weighted_precision_pos": round(
            float(precision_score(y_true, y_pred, pos_label=1, sample_weight=w, zero_division=0)), 6
        ),
        "weighted_recall_pos": round(
            float(recall_score(y_true, y_pred, pos_label=1, sample_weight=w, zero_division=0)), 6
        ),
        "weighted_average_precision": round(float(average_precision_score(y_true, winner_probs, sample_weight=w)), 6),
        "weighted_roc_auc": round(float(roc_auc_score(y_true, winner_probs, sample_weight=w)), 6),
        "weighted_brier_score": round(float(brier_score_loss(y_true, winner_probs, sample_weight=w)), 6),
    }

    return {
        "small_cell_min_count": SMALL_CELL_MIN_COUNT,
        "subgroups": subgroup_records,
        "survey_weighted_sensitivity": weighted_summary,
    }


def _plot_test_diagnostics(
    y_test: np.ndarray,
    probs_by_model: Dict[str, np.ndarray],
    display_names: Dict[str, str],
    winner_key: str,
    winner_metrics: Dict[str, Any],
    subgroup_records: List[Dict[str, Any]],
    figures_dir: Path,
) -> Dict[str, str]:
    """Generate PR/ROC curves, Confusion Matrix + Reliability plot, and Subgroup F1/AP chart."""
    sns.set_theme(style="whitegrid")
    prevalence = float(y_test.mean())

    # Plot 9: PR & ROC Curves across models
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=140)
    palette = {
        "gradient_boosting": "#d95f02",
        "random_forest": "#1b9e77",
        "logistic_regression": "#2b5c8f",
        "dummy_prior": "#7f7f7f",
        "dummy_all_positive": "#b2df8a",
    }
    for m_key in ("gradient_boosting", "random_forest", "logistic_regression", "dummy_prior"):
        p = probs_by_model[m_key]
        prec, rec, _ = precision_recall_curve(y_test, p)
        ap = average_precision_score(y_test, p)
        lw = 2.5 if m_key == winner_key else 1.7
        axes[0].plot(
            rec,
            prec,
            label=f"{display_names[m_key]} (AP={ap:.3f})",
            color=palette.get(m_key, "black"),
            linewidth=lw,
        )

        fpr, tpr, _ = roc_curve(y_test, p)
        auc_val = roc_auc_score(y_test, p)
        axes[1].plot(
            fpr,
            tpr,
            label=f"{display_names[m_key]} (AUC={auc_val:.3f})",
            color=palette.get(m_key, "black"),
            linewidth=lw,
        )

    axes[0].axhline(prevalence, color="black", linestyle="--", linewidth=1.2, label=f"No-Skill Prevalence ({prevalence:.3f})")
    axes[0].set_xlabel("Recall (Unemployed Class = 1)")
    axes[0].set_ylabel("Precision (Unemployed Class = 1)")
    axes[0].set_title(f"A. Test Precision-Recall Curve (n = {len(y_test):,})")
    axes[0].legend(fontsize=8.5, loc="lower left")

    axes[1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Random Chance (0.500)")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title(f"B. Test ROC Curve (n = {len(y_test):,})")
    axes[1].legend(fontsize=8.5, loc="lower right")

    fig.suptitle(
        "HunarLens Frozen Test Evaluation — Precision-Recall & ROC Curves",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    pr_roc_path = figures_dir / "09_test_pr_and_roc_curves.png"
    fig.savefig(pr_roc_path, bbox_inches="tight")
    plt.close(fig)

    # Plot 10: Confusion Matrix & Calibration Curve for Frozen Winner
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=140)
    cm = np.asarray(winner_metrics["confusion_matrix"]["counts"], dtype=int)
    cm_norm = np.asarray(winner_metrics["confusion_matrix"]["row_normalized"], dtype=float)
    annot = np.array(
        [
            [f"TN = {cm[0,0]:,}\n({cm_norm[0,0]:.1%})", f"FP = {cm[0,1]:,}\n({cm_norm[0,1]:.1%})"],
            [f"FN = {cm[1,0]:,}\n({cm_norm[1,0]:.1%})", f"TP = {cm[1,1]:,}\n({cm_norm[1,1]:.1%})"],
        ]
    )
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Pred: Employed (0)", "Pred: Unemployed (1)"],
        yticklabels=["True: Employed (0)", "True: Unemployed (1)"],
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title(
        f"A. Confusion Matrix — {display_names[winner_key]}\n"
        f"(Threshold = {winner_metrics['threshold']:.2f}, Test F1 = {winner_metrics['f1_pos']:.3f})"
    )

    prob_true, prob_pred = calibration_curve(y_test, probs_by_model[winner_key], n_bins=10, strategy="uniform")
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    axes[1].plot(
        prob_pred,
        prob_true,
        "s-",
        color="#d95f02",
        linewidth=2.2,
        label=f"{display_names[winner_key]} (Brier={winner_metrics['brier_score']:.4f})",
    )
    prior_brier = brier_score_loss(y_test, probs_by_model["dummy_prior"])
    axes[1].axhline(
        prevalence,
        color="#7f7f7f",
        linestyle=":",
        label=f"Constant Prevalence Predictor (Brier={prior_brier:.4f})",
    )
    axes[1].set_xlabel("Mean Predicted Probability of Unemployment")
    axes[1].set_ylabel("Observed Fraction of Unemployed Youth")
    axes[1].set_title("B. Probability Reliability / Calibration Curve (Test Split)")
    axes[1].legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "HunarLens Frozen Test Evaluation — Confusion Matrix & Probability Calibration",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    cm_cal_path = figures_dir / "10_test_confusion_and_calibration.png"
    fig.savefig(cm_cal_path, bbox_inches="tight")
    plt.close(fig)

    # Plot 11: Subgroup Performance Chart
    valid_sub = [r for r in subgroup_records if not r.get("suppressed", False)]
    sub_df = pd.DataFrame(valid_sub)
    sub_df["label"] = sub_df["column"] + ": " + sub_df["subgroup"] + " (n=" + sub_df["n"].astype(str) + ")"

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=140)
    y_pos = np.arange(len(sub_df))
    h = 0.36
    ax.barh(y_pos - h / 2, sub_df["f1_pos"], height=h, label="Test F1 (Class 1)", color="#2b5c8f")
    ax.barh(y_pos + h / 2, sub_df["average_precision"], height=h, label="Test Average Precision (AP)", color="#d95f02")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_df["label"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score on Sealed Test Set")
    ax.set_title(
        f"HunarLens Frozen Test Subgroup Performance — {display_names[winner_key]} (Cells >= {SMALL_CELL_MIN_COUNT})",
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    sub_path = figures_dir / "11_test_subgroup_f1_ap.png"
    fig.savefig(sub_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "pr_roc_plot": str(pr_roc_path.relative_to(REPO_ROOT)),
        "confusion_calibration_plot": str(cm_cal_path.relative_to(REPO_ROOT)),
        "subgroup_plot": str(sub_path.relative_to(REPO_ROOT)),
    }


def write_final_evaluation_markdown(
    eval_payload: Dict[str, Any],
    out_path: Path,
) -> None:
    """Write `docs/final_evaluation_report.md` including the completed §10.5 Results Table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = eval_payload["models"]
    boot = eval_payload["bootstrap_95ci"]
    winner_key = eval_payload["winner_model_key"]

    table_rows = []
    for m_key in (
        "dummy_prior",
        "dummy_all_positive",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ):
        m = models[m_key]
        tm = m["test_metrics_at_frozen_threshold"]
        ci_f1 = boot["model_intervals"][m_key]["f1_pos"]
        ci_ap = boot["model_intervals"][m_key]["average_precision"]
        star = " **(Selected Winner)**" if m_key == winner_key else ""
        table_rows.append(
            f"| {m['display_name']}{star} | `{m['train_cv_ap_mean']:.4f} ± {m['train_cv_ap_std']:.4f}` | "
            f"`{m['frozen_validation_threshold']:.2f}` | "
            f"**`{tm['f1_pos']:.4f}`** (`[{ci_f1['ci95_low']:.4f}, {ci_f1['ci95_high']:.4f}]`) | "
            f"`{tm['precision_pos']:.4f} / {tm['recall_pos']:.4f}` | "
            f"`{tm['average_precision']:.4f}` (`[{ci_ap['ci95_low']:.4f}, {ci_ap['ci95_high']:.4f}]`) / `{tm['roc_auc']:.4f}` | "
            f"`{tm['brier_score']:.4f}` |"
        )
    results_table = "\n".join(table_rows)

    sub_rows = []
    for r in eval_payload["subgroup_and_weighted_audit"]["subgroups"]:
        if r.get("suppressed"):
            sub_rows.append(f"| {r['dimension']} | {r['subgroup']} | `{r['n']}` | Suppressed (`<{SMALL_CELL_MIN_COUNT}`) | — | — | — |")
        else:
            sub_rows.append(
                f"| {r['dimension']} | {r['subgroup']} | `{r['n']:,}` | `{r['prevalence']:.2%}` | "
                f"`{r['f1_pos']:.4f}` | `{r['precision_pos']:.4f} / {r['recall_pos']:.4f}` | `{r['average_precision']:.4f}` |"
            )
    subgroup_table = "\n".join(sub_rows)

    diffs = boot["winner_paired_differences"]
    w_sens = eval_payload["subgroup_and_weighted_audit"]["survey_weighted_sensitivity"]

    content = f"""# HunarLens — One-Time Frozen Test Evaluation & Subgroup Audit (`final_evaluation_report.md`)

**Reference:** [PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md) §10.1–§10.5, §11 · **Generated by:** [scripts/evaluate_once.py](file:///d:/Hunarlens/scripts/evaluate_once.py)  
**Test Cohort:** `{eval_payload['test_row_count']:,}` eligible youth aged 15–24 across `{eval_payload['test_psu_count']:,}` unseen PSUs (`PCode`), positive-class support = `{eval_payload['test_positive_support']:,}` (`prevalence = {eval_payload['test_prevalence']:.2%}`).

---

## 1. Frozen Model Comparison Table (§10.5 Completed Template)

| Model | Train-CV AP (`mean ± SD`) | Validation Threshold | Test F1 (`class 1`, 95% PSU Bootstrap CI) | Test Precision / Recall (`class 1`) | Test AP (95% CI) / ROC-AUC | Test Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{results_table}

---

## 2. Paired PSU-Clustered Bootstrap Significance (`1,000` Resamples, `seed=46`)
* **Independent Test Clusters (`PCode`):** `{boot['independent_test_psus']:,}` PSUs (`{boot['n_valid_resamples']:,}` valid resamples, `{boot['invalid_single_class_resamples']}` invalid single-class resamples).
* **Winner vs. Dummy Prior ($\\Delta F_1$):** `+{diffs['winner_minus_dummy_prior_f1']['mean']:.4f}` (`95% CI: [{diffs['winner_minus_dummy_prior_f1']['ci95_low']:+.4f}, {diffs['winner_minus_dummy_prior_f1']['ci95_high']:+.4f}]`).
* **Winner vs. Dummy All-Positive ($\\Delta F_1$):** `+{diffs['winner_minus_dummy_all_positive_f1']['mean']:.4f}` (`95% CI: [{diffs['winner_minus_dummy_all_positive_f1']['ci95_low']:+.4f}, {diffs['winner_minus_dummy_all_positive_f1']['ci95_high']:+.4f}]`).
* **Winner vs. Dummy Prior ($\\Delta \\text{{AP}}$):** `+{diffs['winner_minus_dummy_prior_ap']['mean']:.4f}` (`95% CI: [{diffs['winner_minus_dummy_prior_ap']['ci95_low']:+.4f}, {diffs['winner_minus_dummy_prior_ap']['ci95_high']:+.4f}]`).

---

## 3. Predeclared Subgroup Performance on Sealed Test Set (`Cells >= {SMALL_CELL_MIN_COUNT}`)

| Dimension | Subgroup | Test Count (`n`) | Unemployment Prevalence | Test F1 (`class 1`) | Precision / Recall | Test AP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{subgroup_table}

---

## 4. Survey-Weighted Sensitivity Summary (Separately Labelled per §10.4)
* **Weighted Test Prevalence:** `{w_sens['weighted_prevalence']:.2%}`
* **Weighted Test F1 (`class 1`):** `{w_sens['weighted_f1_pos']:.4f}` (Precision: `{w_sens['weighted_precision_pos']:.4f}`, Recall: `{w_sens['weighted_recall_pos']:.4f}`)
* **Weighted Test AP / ROC-AUC:** `{w_sens['weighted_average_precision']:.4f}` / `{w_sens['weighted_roc_auc']:.4f}`
* **Weighted Test Brier Score:** `{w_sens['weighted_brier_score']:.4f}`
"""
    out_path.write_text(content, encoding="utf-8")


def run_single_batch_test_evaluation(
    data_cfg_path: str | Path = "configs/data.yaml",
    exp_cfg_path: str | Path = "configs/experiments.yaml",
) -> Dict[str, Any]:
    """Verify freeze manifest, run global permutation importance on validation, and evaluate frozen models once on test."""
    data_cfg = load_yaml_config(REPO_ROOT / data_cfg_path)
    exp_cfg = load_yaml_config(REPO_ROOT / exp_cfg_path)

    freeze_path = REPO_ROOT / exp_cfg["evaluation"]["artifacts"]["freeze_manifest_path"]
    manifest = verify_freeze_manifest(freeze_path, data_cfg, exp_cfg)

    splits_dir = REPO_ROOT / data_cfg["paths"]["splits_dir"]
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    y_test = test_df["is_unemployed"].to_numpy(dtype=int)

    train_val_res_path = REPO_ROOT / "reports/metrics/training_validation_results.json"
    train_val_res = json.loads(train_val_res_path.read_text(encoding="utf-8"))
    winner_key = str(train_val_res["winner_model_key"])

    # 1. Global permutation importance on Validation Set for the frozen deployment winner (§11)
    winner_pipe = joblib.load(REPO_ROOT / exp_cfg["evaluation"]["artifacts"]["pipeline_path"])
    imp_df = compute_global_permutation_importance(
        winner_pipe,
        val_df=val_df,
        target_col="is_unemployed",
        n_repeats=int(exp_cfg["evaluation"]["permutation_repeats"]),
        random_state=int(exp_cfg["seeds"]["permutation_importance"]),
    )
    global_imp_payload = save_global_importance_artifacts(imp_df)

    # 2. Evaluate all 5 frozen candidates once on the sealed test partition
    candidates_dir = REPO_ROOT / "models/candidates"
    probs_by_model: Dict[str, np.ndarray] = {}
    thresholds_by_model: Dict[str, float] = {}
    display_names: Dict[str, str] = {}
    evaluated_models: Dict[str, Any] = {}

    for m_key, m_info in train_val_res["models"].items():
        c_path = candidates_dir / f"{m_key}.joblib"
        pipe = joblib.load(c_path) if c_path.exists() else winner_pipe
        probs = np.asarray(pipe.predict_proba(test_df)[:, 1], dtype=float)
        th = float(m_info["validation_threshold_selection"]["selected_threshold"])

        probs_by_model[m_key] = probs
        thresholds_by_model[m_key] = th
        display_names[m_key] = str(m_info["display_name"])

        test_at_th = compute_binary_metrics_at_threshold(y_test, probs, threshold=th)
        test_at_05 = compute_binary_metrics_at_threshold(y_test, probs, threshold=0.50)

        cv_ap = m_info["cv_summary"]["cv_metrics"]["average_precision"]
        evaluated_models[m_key] = {
            "display_name": m_info["display_name"],
            "is_baseline": m_info["is_baseline"],
            "train_cv_ap_mean": cv_ap["cv_mean"],
            "train_cv_ap_std": cv_ap["cv_std"],
            "frozen_validation_threshold": th,
            "validation_f1_at_threshold": m_info["validation_threshold_selection"][
                "metrics_at_selected_threshold"
            ]["f1_pos"],
            "validation_ap": m_info["validation_threshold_selection"][
                "metrics_at_selected_threshold"
            ]["average_precision"],
            "test_metrics_at_frozen_threshold": test_at_th,
            "test_metrics_at_default_05_threshold": test_at_05,
        }

    # 3. 1,000-resample paired PSU bootstrap confidence intervals (§10.4)
    bootstrap_res = run_paired_psu_bootstrap(
        test_df=test_df,
        probs_by_model=probs_by_model,
        thresholds_by_model=thresholds_by_model,
        winner_key=winner_key,
        group_col=exp_cfg["splits"]["group_column"],
        target_col="is_unemployed",
        n_resamples=int(exp_cfg["evaluation"]["bootstrap_resamples"]),
        seed=int(exp_cfg["seeds"]["bootstrap_ci"]),
    )

    # 4. Predeclared subgroup & survey-weighted sensitivity evaluation (§10.4)
    subgroup_res = compute_subgroup_and_weighted_audit(
        test_df=test_df,
        winner_probs=probs_by_model[winner_key],
        winner_threshold=thresholds_by_model[winner_key],
    )

    # 5. Generate diagnostic figures
    figures_dir = REPO_ROOT / "reports/figures"
    plot_paths = _plot_test_diagnostics(
        y_test=y_test,
        probs_by_model=probs_by_model,
        display_names=display_names,
        winner_key=winner_key,
        winner_metrics=evaluated_models[winner_key]["test_metrics_at_frozen_threshold"],
        subgroup_records=subgroup_res["subgroups"],
        figures_dir=figures_dir,
    )

    payload = {
        "evaluation_status": "ONE_TIME_FROZEN_TEST_EVALUATION_COMPLETE",
        "winner_model_key": winner_key,
        "winner_display_name": display_names[winner_key],
        "winner_frozen_threshold": thresholds_by_model[winner_key],
        "test_row_count": int(len(test_df)),
        "test_psu_count": int(test_df["PCode"].nunique()),
        "test_positive_support": int(y_test.sum()),
        "test_prevalence": round(float(y_test.mean()), 6),
        "models": evaluated_models,
        "bootstrap_95ci": bootstrap_res,
        "subgroup_and_weighted_audit": subgroup_res,
        "global_permutation_importance": global_imp_payload,
        "diagnostic_plots": plot_paths,
        "freeze_manifest_verified": manifest["final_pipeline_sha256"],
    }

    metrics_dir = REPO_ROOT / "reports/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "final_test_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    write_final_evaluation_markdown(payload, REPO_ROOT / "docs/final_evaluation_report.md")
    return payload

