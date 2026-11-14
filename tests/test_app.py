"""End-to-end tests for the HunarLens Streamlit application and inference wrapper (`tests/test_app.py`).

Implements PROJECT_PLAN.md §12.1, §12.2, §13 & Phase 6, Step 6.2 of PROJECT_TRACKER.md:
- Tests clean loading of the frozen artifact (`models/final_pipeline.joblib`), saved decision threshold,
  feature schema, and model metadata.
- Tests all synthetic presets and dashboard artifact loading in `app/streamlit_app.py`.
- Tests input validation (age bounds 15-24, valid geography, contradictory household counts).
- Tests automatic Census 2023 lookup from `province_territory`.
- Tests exact agreement (parity) between batch `pipeline.predict_proba` and single-profile app inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.streamlit_app import SYNTHETIC_PRESETS, load_dashboard_artifacts  # noqa: E402
from hunarlens.inference import HunarLensPredictor  # noqa: E402


def test_predictor_loads_frozen_artifacts_and_scores_presets() -> None:
    """Verify frozen pipeline, threshold, metadata, and all Streamlit presets score cleanly."""
    predictor = HunarLensPredictor()
    assert 0.01 <= predictor.decision_threshold <= 0.99
    assert predictor.metadata["selected_model_key"] in (
        "gradient_boosting",
        "logistic_regression",
        "random_forest",
    )

    for preset_name, preset_profile in SYNTHETIC_PRESETS.items():
        res = predictor.predict_profile(preset_profile, include_explanation=True)
        assert 0.0 < res["estimated_probability"] < 1.0, f"Preset {preset_name} failed probability check"
        assert res["decision_threshold"] == predictor.decision_threshold
        assert res["predicted_class"] in (0, 1)
        assert "local_explanation" in res
        assert len(res["local_explanation"]["top_contributions"]) == 10


def test_batch_vs_single_profile_app_prediction_parity() -> None:
    """Verify exact numerical agreement between batch DataFrame predictions and single-profile app scoring (§12.2)."""
    predictor = HunarLensPredictor()
    val_df = pd.read_parquet(REPO_ROOT / "data/splits/val.parquet").head(15).copy()

    batch_probs = predictor.predict_batch(val_df)

    for idx, (_, row) in enumerate(val_df.iterrows()):
        profile_dict = {
            "age": int(row["age"]),
            "sex": str(row["sex"]),
            "marital_status": str(row["marital_status"]),
            "relationship_to_head": str(row["relationship_to_head"]),
            "can_read": str(row["can_read"]),
            "can_write": str(row["can_write"]),
            "highest_education": str(row["highest_education"]),
            "current_enrolment": str(row["current_enrolment"]),
            "household_size": int(row["household_size"]),
            "household_dependency_ratio": float(row["household_dependency_ratio"]),
            "province_territory": str(row["province_territory"]),
            "locality": str(row["locality"]),
        }
        single_res = predictor.predict_profile(profile_dict, include_explanation=False)
        assert np.isclose(single_res["estimated_probability"], batch_probs[idx], atol=1e-6)
        expected_class = int(batch_probs[idx] >= predictor.decision_threshold)
        assert single_res["predicted_class"] == expected_class


def test_app_input_validation_rejects_out_of_scope_or_contradictory_inputs() -> None:
    """Verify invalid age, unsupported geography, and contradictory household counts raise clear errors."""
    predictor = HunarLensPredictor()
    base = dict(next(iter(SYNTHETIC_PRESETS.values())))

    # Out-of-range age (<15 or >24)
    bad_age = {**base, "age": 29}
    with pytest.raises(ValueError, match="between 15 and 24"):
        predictor.predict_profile(bad_age, include_explanation=False)

    # Unsupported province
    bad_prov = {**base, "province_territory": "Gilgit-Baltistan-Unverified"}
    with pytest.raises(ValueError, match="Unsupported province_territory"):
        predictor.predict_profile(bad_prov, include_explanation=False)

    # Contradictory household working_age_count < 1
    bad_hh = {**base, "working_age_count": 0}
    with pytest.raises(ValueError, match="Contradictory household counts"):
        predictor.predict_profile(bad_hh, include_explanation=False)


def test_dashboard_artifacts_and_figures_exist() -> None:
    """Verify all precomputed JSON artifacts and 11 PNG figures required by the 5 Streamlit views exist."""
    artifacts = load_dashboard_artifacts()
    for required_key in (
        "split_manifest",
        "eda_summary",
        "train_val",
        "test_eval",
        "global_imp",
        "census_df",
        "benchmarks_df",
        "aggregates",
    ):
        assert required_key in artifacts
        assert artifacts[required_key] is not None

    expected_figures = [
        "01_class_balance.png",
        "02_feature_missingness.png",
        "03_numeric_distributions.png",
        "04_numeric_boxplots_outliers.png",
        "05_numeric_correlation_heatmap.png",
        "06_education_enrolment_vs_unemployment.png",
        "07_province_locality_sex_training_summary.png",
        "08_global_permutation_importance.png",
        "09_test_pr_and_roc_curves.png",
        "10_test_confusion_and_calibration.png",
        "11_test_subgroup_f1_ap.png",
    ]
    for fig_name in expected_figures:
        assert (REPO_ROOT / "reports/figures" / fig_name).exists(), f"Missing figure: {fig_name}"
