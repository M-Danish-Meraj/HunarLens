"""Model training, 5-fold StratifiedGroupKFold search, ablations, validation threshold selection, and artifact freeze.

Implements PROJECT_PLAN.md §9.1, §9.2, §10.1, §10.2, §13:
- Trains Dummy baselines (`prior` and `constant=1`) and tunes Logistic Regression (`GridSearchCV`, 8 configs),
  Random Forest (`RandomizedSearchCV`, 20 configs), and Gradient Boosting (`RandomizedSearchCV`, 20 configs,
  `n_iter_no_change=None`) using identical 5-fold `StratifiedGroupKFold` split indices on the 70% training set.
- Refits each tuned algorithm on the 70% training partition by mean CV Average Precision (`AP`).
- Runs the two preplanned training-CV ablations (`demographics_only_no_census`, `demographics_plus_census_no_sex`).
- Evaluates shortlisted models on the 15% validation set across the 0.01..0.99 + endpoints threshold grid to
  maximize positive-class F1 (`is_unemployed == 1`), selects the deployment winner under the predeclared tie-break
  rule (§9.2), and freezes all candidate artifacts and metadata before the sealed test set is ever touched.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn

import hunarlens  # noqa: F401 — ensures sklearn.ensemble compatibility guard on Windows
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate

from hunarlens.data import load_yaml_config
from hunarlens.features import (
    ALL_APPROVED_15_FEATURES,
    APPROVED_CATEGORICAL_FEATURES,
    APPROVED_NUMERIC_FEATURES,
    build_estimator_pipeline,
)
from hunarlens.splits import generate_training_cv_folds

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    """Return current git HEAD commit hash if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def compute_binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute all required binary classification and probability metrics at a specified operating threshold (§10.1)."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    f1_pos = float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    prec_pos = float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    rec_pos = float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    ap = float(average_precision_score(y_true_arr, y_prob_arr))
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        roc_auc = 0.5
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    prevalence = float(y_true_arr.mean())

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = (cm / row_sums).round(4).tolist()

    return {
        "threshold": round(float(threshold), 4),
        "n_samples": int(len(y_true_arr)),
        "positive_support": int(y_true_arr.sum()),
        "prevalence": round(prevalence, 6),
        "f1_pos": round(f1_pos, 6),
        "precision_pos": round(prec_pos, 6),
        "recall_pos": round(rec_pos, 6),
        "average_precision": round(ap, 6),
        "ap_lift_over_prevalence": round(ap / prevalence, 4) if prevalence > 0 else 1.0,
        "roc_auc": round(roc_auc, 6),
        "brier_score": round(brier, 6),
        "confusion_matrix": {
            "labels": ["Employed (0)", "Unemployed (1)"],
            "counts": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "row_normalized": cm_norm,
        },
    }


def select_optimal_validation_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    grid_min: float = 0.01,
    grid_max: float = 0.99,
    grid_step: float = 0.01,
    include_endpoints: bool = True,
) -> Dict[str, Any]:
    """Select the operating threshold on validation probabilities maximizing positive-class F1 (§10.2).

    Tie-breaking rule:
      1. Highest `f1_pos`
      2. Highest `recall_pos`
      3. Higher `threshold`
    Also records metrics at the default `0.50` threshold for transparency.
    """
    grid = list(np.round(np.arange(grid_min, grid_max + 1e-9, grid_step), 4))
    if include_endpoints:
        grid = [0.0, *grid, 1.0]

    best_entry: Dict[str, Any] | None = None
    curve_records: List[Dict[str, float]] = []

    for th in grid:
        m = compute_binary_metrics_at_threshold(y_val, val_probs, threshold=float(th))
        curve_records.append(
            {
                "threshold": m["threshold"],
                "f1_pos": m["f1_pos"],
                "precision_pos": m["precision_pos"],
                "recall_pos": m["recall_pos"],
            }
        )
        if best_entry is None:
            best_entry = m
        else:
            cand_key = (
                round(m["f1_pos"], 8),
                round(m["recall_pos"], 8),
                round(m["threshold"], 8),
            )
            best_key = (
                round(best_entry["f1_pos"], 8),
                round(best_entry["recall_pos"], 8),
                round(best_entry["threshold"], 8),
            )
            if cand_key > best_key:
                best_entry = m

    default_05_metrics = compute_binary_metrics_at_threshold(y_val, val_probs, threshold=0.50)
    assert best_entry is not None
    return {
        "selected_threshold": best_entry["threshold"],
        "metrics_at_selected_threshold": best_entry,
        "metrics_at_default_05_threshold": default_05_metrics,
        "threshold_grid_evaluated_count": len(grid),
        "threshold_curve": curve_records,
    }


def _get_cv_scoring() -> Dict[str, str]:
    return {
        "average_precision": "average_precision",
        "f1_pos": "f1",
        "precision_pos": "precision",
        "recall_pos": "recall",
        "roc_auc": "roc_auc",
    }


def _extract_best_search_summary(search_obj: GridSearchCV | RandomizedSearchCV, elapsed_sec: float) -> Dict[str, Any]:
    """Extract CV mean/SD metrics and train-vs-CV diagnostic gaps for the best refitted configuration."""
    idx = int(search_obj.best_index_)
    res = search_obj.cv_results_

    summary: Dict[str, Any] = {
        "best_index": idx,
        "best_params": search_obj.best_params_,
        "n_candidates_evaluated": len(res["params"]),
        "search_runtime_seconds": round(elapsed_sec, 3),
        "cv_metrics": {},
    }
    for metric_name in ("average_precision", "f1_pos", "precision_pos", "recall_pos", "roc_auc"):
        val_mean = float(res[f"mean_test_{metric_name}"][idx])
        val_std = float(res[f"std_test_{metric_name}"][idx])
        tr_mean = float(res[f"mean_train_{metric_name}"][idx])
        tr_std = float(res[f"std_train_{metric_name}"][idx])
        summary["cv_metrics"][metric_name] = {
            "cv_mean": round(val_mean, 6),
            "cv_std": round(val_std, 6),
            "train_mean": round(tr_mean, 6),
            "train_std": round(tr_std, 6),
            "train_minus_cv_gap": round(tr_mean - val_mean, 6),
        }
    return summary


def _evaluate_baseline_cv(
    pipeline: Any,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    fold_indices: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Evaluate a Dummy baseline across the identical 5-fold StratifiedGroupKFold splits and fit on train_df."""
    t0 = time.perf_counter()
    cv_res = cross_validate(
        pipeline,
        train_df,
        y_train,
        cv=fold_indices,
        scoring=_get_cv_scoring(),
        return_train_score=True,
        n_jobs=1,
    )
    pipeline.fit(train_df, y_train)
    elapsed = time.perf_counter() - t0

    metrics_dict: Dict[str, Any] = {}
    for m in ("average_precision", "f1_pos", "precision_pos", "recall_pos", "roc_auc"):
        val_arr = cv_res[f"test_{m}"]
        tr_arr = cv_res[f"train_{m}"]
        metrics_dict[m] = {
            "cv_mean": round(float(np.mean(val_arr)), 6),
            "cv_std": round(float(np.std(val_arr)), 6),
            "train_mean": round(float(np.mean(tr_arr)), 6),
            "train_std": round(float(np.std(tr_arr)), 6),
            "train_minus_cv_gap": round(float(np.mean(tr_arr) - np.mean(val_arr)), 6),
        }
    return {
        "best_index": 0,
        "best_params": {},
        "n_candidates_evaluated": 1,
        "search_runtime_seconds": round(elapsed, 3),
        "cv_metrics": metrics_dict,
    }


def run_training_and_selection(
    data_cfg_path: str | Path = "configs/data.yaml",
    features_cfg_path: str | Path = "configs/features.yaml",
    exp_cfg_path: str | Path = "configs/experiments.yaml",
) -> Dict[str, Any]:
    """Execute full model training, 5-fold grouped search, ablations, validation selection, and artifact freeze."""
    data_cfg = load_yaml_config(REPO_ROOT / data_cfg_path)
    _ = load_yaml_config(REPO_ROOT / features_cfg_path)
    exp_cfg = load_yaml_config(REPO_ROOT / exp_cfg_path)

    splits_dir = REPO_ROOT / data_cfg["paths"]["splits_dir"]
    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "val.parquet"
    test_path = splits_dir / "test.parquet"
    census_path = REPO_ROOT / data_cfg["paths"]["census_reference_file"]

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    y_train = train_df["is_unemployed"].to_numpy(dtype=int)
    y_val = val_df["is_unemployed"].to_numpy(dtype=int)

    # Recover identical 5-fold StratifiedGroupKFold indices inside train_df
    _, fold_indices, _ = generate_training_cv_folds(
        train_df,
        group_col=exp_cfg["splits"]["group_column"],
        target_col="is_unemployed",
        n_splits=int(exp_cfg["splits"]["cv_n_splits"]),
        cv_seed=int(exp_cfg["seeds"]["cv_folds"]),
    )

    seed_model = int(exp_cfg["seeds"]["model_training"])
    scoring = _get_cv_scoring()

    fitted_models: Dict[str, Any] = {}
    model_reports: Dict[str, Any] = {}

    # 1. Baseline 1: Dummy Prior / Majority
    dummy_prior_pipe = build_estimator_pipeline(
        DummyClassifier(strategy="prior", random_state=seed_model),
        scale_numeric=False,
        census_path=census_path,
    )
    prior_cv = _evaluate_baseline_cv(dummy_prior_pipe, train_df, y_train, fold_indices)
    prior_val_probs = dummy_prior_pipe.predict_proba(val_df)[:, 1]
    prior_val_metrics = compute_binary_metrics_at_threshold(y_val, prior_val_probs, threshold=0.50)
    fitted_models["dummy_prior"] = dummy_prior_pipe
    model_reports["dummy_prior"] = {
        "display_name": "Dummy Baseline (Prior / Majority)",
        "is_baseline": True,
        "complexity_rank": 0,
        "cv_summary": prior_cv,
        "validation_threshold_selection": {
            "selected_threshold": 0.50,
            "metrics_at_selected_threshold": prior_val_metrics,
            "metrics_at_default_05_threshold": prior_val_metrics,
        },
    }

    # 2. Baseline 2: Dummy All-Positive (constant = 1)
    dummy_pos_pipe = build_estimator_pipeline(
        DummyClassifier(strategy="constant", constant=1, random_state=seed_model),
        scale_numeric=False,
        census_path=census_path,
    )
    pos_cv = _evaluate_baseline_cv(dummy_pos_pipe, train_df, y_train, fold_indices)
    pos_val_probs = dummy_pos_pipe.predict_proba(val_df)[:, 1]
    pos_val_metrics = compute_binary_metrics_at_threshold(y_val, pos_val_probs, threshold=0.50)
    fitted_models["dummy_all_positive"] = dummy_pos_pipe
    model_reports["dummy_all_positive"] = {
        "display_name": "Dummy Baseline (All-Positive / Unemployed)",
        "is_baseline": True,
        "complexity_rank": 0,
        "cv_summary": pos_cv,
        "validation_threshold_selection": {
            "selected_threshold": 0.50,
            "metrics_at_selected_threshold": pos_val_metrics,
            "metrics_at_default_05_threshold": pos_val_metrics,
        },
    }

    # 3. Algorithm 1: Logistic Regression (GridSearchCV, 8 configurations)
    lr_cfg = exp_cfg["models"]["logistic_regression"]
    lr_pipe = build_estimator_pipeline(
        LogisticRegression(
            max_iter=int(lr_cfg["max_iter"]),
            solver=str(lr_cfg["solver"]),
            random_state=seed_model,
        ),
        scale_numeric=True,
        census_path=census_path,
    )
    t0 = time.perf_counter()
    lr_search = GridSearchCV(
        estimator=lr_pipe,
        param_grid=lr_cfg["param_grid"],
        scoring=scoring,
        refit="average_precision",
        cv=fold_indices,
        return_train_score=True,
        n_jobs=1,
    )
    lr_search.fit(train_df, y_train)
    lr_elapsed = time.perf_counter() - t0
    lr_best = lr_search.best_estimator_
    lr_val_probs = lr_best.predict_proba(val_df)[:, 1]
    lr_th_sel = select_optimal_validation_threshold(y_val, lr_val_probs)
    fitted_models["logistic_regression"] = lr_best
    model_reports["logistic_regression"] = {
        "display_name": "Logistic Regression",
        "is_baseline": False,
        "complexity_rank": 1,
        "cv_summary": _extract_best_search_summary(lr_search, lr_elapsed),
        "validation_threshold_selection": {
            k: v for k, v in lr_th_sel.items() if k != "threshold_curve"
        },
        "validation_threshold_curve": lr_th_sel["threshold_curve"],
    }

    # 4. Algorithm 2: Random Forest (RandomizedSearchCV, 20 configurations)
    rf_cfg = exp_cfg["models"]["random_forest"]
    rf_pipe = build_estimator_pipeline(
        RandomForestClassifier(random_state=seed_model, n_jobs=1),
        scale_numeric=False,
        census_path=census_path,
    )
    t0 = time.perf_counter()
    rf_search = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=rf_cfg["param_distributions"],
        n_iter=int(rf_cfg["n_iter"]),
        scoring=scoring,
        refit="average_precision",
        cv=fold_indices,
        random_state=seed_model,
        return_train_score=True,
        n_jobs=1,
    )
    rf_search.fit(train_df, y_train)
    rf_elapsed = time.perf_counter() - t0
    rf_best = rf_search.best_estimator_
    rf_val_probs = rf_best.predict_proba(val_df)[:, 1]
    rf_th_sel = select_optimal_validation_threshold(y_val, rf_val_probs)
    fitted_models["random_forest"] = rf_best
    model_reports["random_forest"] = {
        "display_name": "Random Forest",
        "is_baseline": False,
        "complexity_rank": 2,
        "cv_summary": _extract_best_search_summary(rf_search, rf_elapsed),
        "validation_threshold_selection": {
            k: v for k, v in rf_th_sel.items() if k != "threshold_curve"
        },
        "validation_threshold_curve": rf_th_sel["threshold_curve"],
    }

    # 5. Algorithm 3: Gradient Boosting (RandomizedSearchCV, 20 configurations, n_iter_no_change=None)
    gb_cfg = exp_cfg["models"]["gradient_boosting"]
    gb_pipe = build_estimator_pipeline(
        GradientBoostingClassifier(
            n_iter_no_change=None,
            random_state=seed_model,
        ),
        scale_numeric=False,
        census_path=census_path,
    )
    t0 = time.perf_counter()
    gb_search = RandomizedSearchCV(
        estimator=gb_pipe,
        param_distributions=gb_cfg["param_distributions"],
        n_iter=int(gb_cfg["n_iter"]),
        scoring=scoring,
        refit="average_precision",
        cv=fold_indices,
        random_state=seed_model,
        return_train_score=True,
        n_jobs=1,
    )
    gb_search.fit(train_df, y_train)
    gb_elapsed = time.perf_counter() - t0
    gb_best = gb_search.best_estimator_
    gb_val_probs = gb_best.predict_proba(val_df)[:, 1]
    gb_th_sel = select_optimal_validation_threshold(y_val, gb_val_probs)
    fitted_models["gradient_boosting"] = gb_best
    model_reports["gradient_boosting"] = {
        "display_name": "Gradient Boosting",
        "is_baseline": False,
        "complexity_rank": 3,
        "cv_summary": _extract_best_search_summary(gb_search, gb_elapsed),
        "validation_threshold_selection": {
            k: v for k, v in gb_th_sel.items() if k != "threshold_curve"
        },
        "validation_threshold_curve": gb_th_sel["threshold_curve"],
    }

    # 6. Preplanned Training-CV Ablations (§9.2 item 4) using the best Gradient Boosting configuration
    gb_best_params = {
        k.replace("classifier__", ""): v for k, v in gb_search.best_params_.items()
    }
    ablation_results: List[Dict[str, Any]] = [
        {
            "ablation_name": "full_15_features",
            "feature_count": 15,
            "dropped_features": [],
            "cv_ap_mean": model_reports["gradient_boosting"]["cv_summary"]["cv_metrics"]["average_precision"]["cv_mean"],
            "cv_ap_std": model_reports["gradient_boosting"]["cv_summary"]["cv_metrics"]["average_precision"]["cv_std"],
            "cv_f1_05_mean": model_reports["gradient_boosting"]["cv_summary"]["cv_metrics"]["f1_pos"]["cv_mean"],
            "cv_roc_auc_mean": model_reports["gradient_boosting"]["cv_summary"]["cv_metrics"]["roc_auc"]["cv_mean"],
        }
    ]
    for abl in exp_cfg["ablations"]:
        drop_set = set(abl["drop_features"])
        kept_features = [f for f in ALL_APPROVED_15_FEATURES if f not in drop_set]
        abl_pipe = build_estimator_pipeline(
            GradientBoostingClassifier(
                n_iter_no_change=None,
                random_state=seed_model,
                **gb_best_params,
            ),
            scale_numeric=False,
            selected_features=kept_features,
            census_path=census_path,
        )
        abl_cv = cross_validate(
            abl_pipe,
            train_df,
            y_train,
            cv=fold_indices,
            scoring=scoring,
            n_jobs=1,
        )
        ablation_results.append(
            {
                "ablation_name": abl["name"],
                "feature_count": len(kept_features),
                "dropped_features": list(abl["drop_features"]),
                "cv_ap_mean": round(float(np.mean(abl_cv["test_average_precision"])), 6),
                "cv_ap_std": round(float(np.std(abl_cv["test_average_precision"])), 6),
                "cv_f1_05_mean": round(float(np.mean(abl_cv["test_f1_pos"])), 6),
                "cv_roc_auc_mean": round(float(np.mean(abl_cv["test_roc_auc"])), 6),
            }
        )

    # 7. Select Deployment Winner on Validation per predeclared rule (§9.2 item 5):
    #    Highest validation F1; if within 0.01 of max validation F1, prefer higher validation AP, then simpler model.
    candidate_keys = ["logistic_regression", "random_forest", "gradient_boosting"]
    max_val_f1 = max(
        model_reports[k]["validation_threshold_selection"]["metrics_at_selected_threshold"]["f1_pos"]
        for k in candidate_keys
    )
    f1_tol = float(exp_cfg["threshold_policy"]["winner_f1_tolerance"])

    top_tier = [
        k
        for k in candidate_keys
        if (
            max_val_f1
            - model_reports[k]["validation_threshold_selection"]["metrics_at_selected_threshold"]["f1_pos"]
        )
        <= f1_tol
    ]
    # Sort top_tier by (-validation_AP, complexity_rank)
    top_tier.sort(
        key=lambda k: (
            -round(
                model_reports[k]["validation_threshold_selection"]["metrics_at_selected_threshold"][
                    "average_precision"
                ],
                4,
            ),
            model_reports[k]["complexity_rank"],
        )
    )
    winner_key = top_tier[0]
    winner_pipe = fitted_models[winner_key]
    winner_report = model_reports[winner_key]
    winner_threshold = float(winner_report["validation_threshold_selection"]["selected_threshold"])

    # 8. Serialize all frozen candidates and winner artifacts (`models/`)
    models_dir = REPO_ROOT / "models"
    candidates_dir = models_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    candidate_hashes: Dict[str, str] = {}
    for m_key, m_pipe in fitted_models.items():
        c_path = candidates_dir / f"{m_key}.joblib"
        joblib.dump(m_pipe, c_path)
        candidate_hashes[m_key] = compute_file_sha256(c_path)

    final_pipeline_path = REPO_ROOT / exp_cfg["evaluation"]["artifacts"]["pipeline_path"]
    joblib.dump(winner_pipe, final_pipeline_path)
    final_pipeline_sha256 = compute_file_sha256(final_pipeline_path)

    # Save decision_threshold.json
    threshold_payload = {
        "selected_model_key": winner_key,
        "selected_model_display_name": winner_report["display_name"],
        "decision_threshold": winner_threshold,
        "default_threshold": 0.50,
        "selection_metric": "f1_pos",
        "selection_split": "validation_15pct",
        "validation_metrics_at_selected_threshold": winner_report["validation_threshold_selection"][
            "metrics_at_selected_threshold"
        ],
        "validation_metrics_at_default_05_threshold": winner_report["validation_threshold_selection"][
            "metrics_at_default_05_threshold"
        ],
        "all_models_thresholds": {
            k: model_reports[k]["validation_threshold_selection"]["selected_threshold"]
            for k in fitted_models
        },
    }
    threshold_path = REPO_ROOT / exp_cfg["evaluation"]["artifacts"]["threshold_path"]
    threshold_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

    # Save feature_schema.json
    schema_payload = {
        "approved_15_features": list(ALL_APPROVED_15_FEATURES),
        "numeric_features": list(APPROVED_NUMERIC_FEATURES),
        "categorical_features": list(APPROVED_CATEGORICAL_FEATURES),
        "target_column": "is_unemployed",
        "class_mapping": {"0": "Employed", "1": "Unemployed"},
        "positive_class": 1,
        "allowed_categories": {
            item["name"]: item["allowed_values"]
            for item in load_yaml_config(REPO_ROOT / features_cfg_path)["predictors"]["categorical"]
        },
        "numeric_bounds": {
            "age": {"min": 15, "max": 24},
            "household_size": {"min": 1, "max": 50},
            "household_dependency_ratio": {"min": 0.0, "max": 20.0},
            "dependents_count": {"min": 0, "max": 30},
            "working_age_count": {"min": 1, "max": 20},
        },
    }
    schema_path = REPO_ROOT / exp_cfg["evaluation"]["artifacts"]["schema_path"]
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    # Save model_metadata.json
    metadata_payload = {
        "project": "HunarLens — Youth Unemployment Vulnerability & Skills Gap Analysis in Pakistan",
        "survey_coverage": "PBS Labour Force Survey 2024-25 + Pakistan Census 2023 Regional Context",
        "intended_use": (
            "Educational and exploratory analysis of survey-defined unemployment associations among "
            "eligible Pakistani youth aged 15-24 in the labour force. Not for automated hiring, "
            "benefit eligibility, or causal policy claims."
        ),
        "selected_model_key": winner_key,
        "selected_model_display_name": winner_report["display_name"],
        "selected_hyperparameters": winner_report["cv_summary"]["best_params"],
        "calibrated": False,
        "decision_threshold": winner_threshold,
        "training_row_count": int(len(train_df)),
        "training_psu_count": int(train_df["PCode"].nunique()),
        "training_prevalence": round(float(y_train.mean()), 6),
        "validation_row_count": int(len(val_df)),
        "validation_psu_count": int(val_df["PCode"].nunique()),
        "validation_prevalence": round(float(y_val.mean()), 6),
        "cv_summary": winner_report["cv_summary"],
        "validation_metrics_at_threshold": winner_report["validation_threshold_selection"][
            "metrics_at_selected_threshold"
        ],
        "artifact_sha256": final_pipeline_sha256,
        "git_commit": get_git_commit_hash(),
        "software_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    metadata_path = REPO_ROOT / exp_cfg["evaluation"]["artifacts"]["metadata_path"]
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    # Save freeze_manifest.json (§10.4 gate required before test evaluation)
    freeze_manifest = {
        "status": "FROZEN_BEFORE_TEST_EVALUATION",
        "winner_model_key": winner_key,
        "winner_threshold": winner_threshold,
        "final_pipeline_sha256": final_pipeline_sha256,
        "candidate_model_sha256": candidate_hashes,
        "split_files_sha256": {
            "train_parquet": compute_file_sha256(train_path),
            "val_parquet": compute_file_sha256(val_path),
            "test_parquet": compute_file_sha256(test_path),
        },
        "feature_count": len(ALL_APPROVED_15_FEATURES),
        "seeds": exp_cfg["seeds"],
        "git_commit": get_git_commit_hash(),
    }
    freeze_path = REPO_ROOT / exp_cfg["evaluation"]["artifacts"]["freeze_manifest_path"]
    freeze_path.write_text(json.dumps(freeze_manifest, indent=2), encoding="utf-8")

    # Save full training & validation comparison summary to reports/metrics/training_validation_results.json
    metrics_dir = REPO_ROOT / "reports/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    full_results = {
        "winner_model_key": winner_key,
        "winner_display_name": winner_report["display_name"],
        "winner_threshold": winner_threshold,
        "models": {
            k: {
                "display_name": v["display_name"],
                "is_baseline": v["is_baseline"],
                "cv_summary": v["cv_summary"],
                "validation_threshold_selection": v["validation_threshold_selection"],
            }
            for k, v in model_reports.items()
        },
        "ablations": ablation_results,
        "freeze_manifest": freeze_manifest,
    }
    (metrics_dir / "training_validation_results.json").write_text(
        json.dumps(full_results, indent=2), encoding="utf-8"
    )
    return full_results

