"""Unit tests for the reproducible scikit-learn preprocessing and inference pipeline.

Implements PROJECT_PLAN.md §6.4, §8 & Step 3.3 of PROJECT_TRACKER.md:
- Verifies that fitted preprocessing parameters (`SimpleImputer` medians, `StandardScaler` means/scales,
  `OneHotEncoder` categories) depend strictly on training observations and are invariant to mutations in
  held-out data (§6.4).
- Verifies automatic Census 2023 lookup from `province_territory` and household dependency ratio derivation
  from `dependents_count` and `working_age_count` for single-profile app inference.
- Verifies safe handling of unseen categorical values (`handle_unknown="ignore"`) and domain validation errors.
- Verifies `joblib` serialization and reload parity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hunarlens  # noqa: E402, F401
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from hunarlens.features import (  # noqa: E402
    APPROVED_NUMERIC_FEATURES,
    build_estimator_pipeline,
    build_preprocessing_pipeline,
)


def test_fitted_parameters_match_training_only_and_ignore_held_out_mutations() -> None:
    """Verify §6.4 requirement: fitted statistics match training-only inputs and held-out changes cannot alter them."""
    train_df = pd.read_parquet(REPO_ROOT / "data/splits/train.parquet").head(400).copy()
    val_df = pd.read_parquet(REPO_ROOT / "data/splits/val.parquet").head(100).copy()

    pre_pipe = build_preprocessing_pipeline(scale_numeric=True)
    pre_pipe.fit(train_df, train_df["is_unemployed"])

    col_tx = pre_pipe.named_steps["preprocessor"]
    num_pipe = col_tx.named_transformers_["numeric"]
    fitted_medians = num_pipe.named_steps["imputer"].statistics_.copy()
    fitted_means = num_pipe.named_steps["scaler"].mean_.copy()

    # Compute exact medians directly on the training DataFrame
    schema_train = pre_pipe.named_steps["schema_census"].transform(train_df)
    expected_medians = schema_train[list(APPROVED_NUMERIC_FEATURES)].median().to_numpy()
    assert np.allclose(fitted_medians, expected_medians)

    # Mutate held-out validation DataFrame within valid domain bounds and transform
    val_mutated = val_df.copy()
    val_mutated["age"] = 24
    val_mutated["household_size"] = 25
    _ = pre_pipe.transform(val_mutated)

    assert np.allclose(num_pipe.named_steps["imputer"].statistics_, fitted_medians)
    assert np.allclose(num_pipe.named_steps["scaler"].mean_, fitted_means)


def test_single_profile_auto_census_lookup_and_household_ratio_derivation() -> None:
    """Verify Streamlit-style single profile without Census columns auto-populates Census & household ratio."""
    train_df = pd.read_parquet(REPO_ROOT / "data/splits/train.parquet").head(300).copy()
    pipe = build_estimator_pipeline(
        LogisticRegression(C=1.0, max_iter=500, random_state=42),
        scale_numeric=True,
    )
    pipe.fit(train_df, train_df["is_unemployed"])

    single_profile = pd.DataFrame(
        [
            {
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
                "province_territory": "Islamabad",
                "locality": "Urban",
            }
        ]
    )

    schema_out = pipe.named_steps["schema_census"].transform(single_profile)
    assert float(schema_out["household_size"].iloc[0]) == 6.0
    assert np.isclose(float(schema_out["household_dependency_ratio"].iloc[0]), 1.0)
    assert np.isclose(float(schema_out["regional_pop_density_log1p"].iloc[0]), np.log1p(2609.12))

    prob = float(pipe.predict_proba(single_profile)[0, 1])
    assert 0.0 < prob < 1.0


def test_unseen_categories_and_domain_validation_rules() -> None:
    """Verify `OneHotEncoder(handle_unknown='ignore')` handles unseen categories and invalid domain inputs raise."""
    train_df = pd.read_parquet(REPO_ROOT / "data/splits/train.parquet").head(300).copy()
    pipe = build_estimator_pipeline(
        RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42),
        scale_numeric=False,
    )
    pipe.fit(train_df, train_df["is_unemployed"])

    # Unseen education or relationship category should not crash prediction
    unseen_profile = train_df.head(2).copy()
    unseen_profile["highest_education"] = "Foreign Diploma Unseen"
    unseen_profile["marital_status"] = np.nan
    probs = pipe.predict_proba(unseen_profile)[:, 1]
    assert len(probs) == 2
    assert not np.isnan(probs).any()

    # Invalid age outside [15, 24] must raise ValueError
    bad_age = train_df.head(1).copy()
    bad_age["age"] = 30
    with pytest.raises(ValueError, match="outside the eligible youth range"):
        pipe.predict_proba(bad_age)

    # Contradictory household working-age count < 1 must raise ValueError
    bad_hh = pd.DataFrame(
        [
            {
                "age": 20,
                "sex": "Male",
                "marital_status": "Never Married",
                "relationship_to_head": "Head",
                "can_read": "Yes",
                "can_write": "Yes",
                "highest_education": "Matric",
                "current_enrolment": "Not Enrolled",
                "dependents_count": 2,
                "working_age_count": 0,
                "province_territory": "Punjab",
                "locality": "Urban",
            }
        ]
    )
    with pytest.raises(ValueError, match="Contradictory household counts"):
        pipe.predict_proba(bad_hh)


def test_joblib_serialization_parity(tmp_path: Path) -> None:
    """Verify a fitted pipeline saved with `joblib.dump` reloads and produces identical probabilities."""
    train_df = pd.read_parquet(REPO_ROOT / "data/splits/train.parquet").head(300).copy()
    val_df = pd.read_parquet(REPO_ROOT / "data/splits/val.parquet").head(50).copy()

    pipe = build_estimator_pipeline(
        LogisticRegression(C=0.5, max_iter=500, random_state=42),
        scale_numeric=True,
    )
    pipe.fit(train_df, train_df["is_unemployed"])
    orig_probs = pipe.predict_proba(val_df)[:, 1]

    artifact_file = tmp_path / "test_pipeline.joblib"
    joblib.dump(pipe, artifact_file)
    reloaded = joblib.load(artifact_file)
    reloaded_probs = reloaded.predict_proba(val_df)[:, 1]

    assert np.allclose(orig_probs, reloaded_probs, atol=1e-12)

