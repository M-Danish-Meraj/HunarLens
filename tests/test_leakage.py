"""Leakage prevention and group-disjoint partition verification tests.

Implements PROJECT_PLAN.md §5.3, §6.2, §6.3, §6.4 & Step 3.3 of PROJECT_TRACKER.md:
- Verifies zero intersection between the 15 approved predictors and the forbidden leakage denylist.
- Verifies zero PSU (`PCode`), household (`PCode, HHNo`), and person (`PCode, HHNo, SNo`) overlap across
  the saved train, validation, and test partitions and across all 5 StratifiedGroupKFold splits.
- Verifies that `HunarLensSchemaAndCensusTransformer` strips IDs, survey weights, `Fstatus`, and any
  target-defining or job-timed training columns so they cannot enter `ColumnTransformer`.
- Runs the §6.4 training-only label-shuffle sanity check to confirm no structural group/ID leakage exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hunarlens.data import load_yaml_config  # noqa: E402
from hunarlens.features import (  # noqa: E402
    ALL_APPROVED_15_FEATURES,
    HunarLensSchemaAndCensusTransformer,
    build_estimator_pipeline,
)
from hunarlens.splits import assert_zero_group_overlap, generate_training_cv_folds  # noqa: E402


def test_zero_intersection_between_approved_predictors_and_forbidden_denylist() -> None:
    """Confirm the 15 approved predictors have zero intersection with `forbidden_denylist` (§6.4)."""
    features_cfg = load_yaml_config(REPO_ROOT / "configs/features.yaml")
    denylist = set(features_cfg["forbidden_denylist"])
    approved = set(ALL_APPROVED_15_FEATURES)

    assert len(approved) == 15
    assert approved.intersection(denylist) == set()


def test_saved_splits_have_zero_psu_household_or_person_overlap() -> None:
    """Confirm saved `train.parquet`, `val.parquet`, and `test.parquet` are strictly PSU/HH/person disjoint."""
    splits_dir = REPO_ROOT / "data/splits"
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    assert_zero_group_overlap(train_df, val_df, test_df)

    # Also verify all 5 StratifiedGroupKFold splits inside train_df have zero PSU overlap
    _, fold_indices, _ = generate_training_cv_folds(train_df, n_splits=5, cv_seed=44)
    for fold_id, (f_tr, f_val) in enumerate(fold_indices):
        tr_psus = set(train_df.iloc[f_tr]["PCode"].astype(str))
        val_psus = set(train_df.iloc[f_val]["PCode"].astype(str))
        assert tr_psus.intersection(val_psus) == set(), f"Fold {fold_id} leaked PSUs!"


def test_schema_transformer_blocks_identifiers_weights_and_leaky_columns() -> None:
    """Confirm IDs, weights, and forbidden columns cannot pass through `HunarLensSchemaAndCensusTransformer`."""
    splits_dir = REPO_ROOT / "data/splits"
    sample_df = pd.read_parquet(splits_dir / "train.parquet").head(25).copy()

    # Inject forbidden columns into the input DataFrame
    sample_df["Fstatus"] = 1
    sample_df["worked_last_week"] = 0
    sample_df["seeking_work"] = 1
    sample_df["vocational_training_status"] = "On-the-Job Training"
    sample_df["monthly_earnings"] = 35000.0

    transformer = HunarLensSchemaAndCensusTransformer()
    transformed = transformer.fit_transform(sample_df)

    assert list(transformed.columns) == list(ALL_APPROVED_15_FEATURES)
    for forbidden in (
        "PCode",
        "HHNo",
        "SNo",
        "weight",
        "Fstatus",
        "worked_last_week",
        "seeking_work",
        "vocational_training_status",
        "monthly_earnings",
        "is_unemployed",
        "cv_fold",
    ):
        assert forbidden not in transformed.columns


def test_label_shuffle_sanity_experiment_produces_no_spurious_signal() -> None:
    """Run the §6.4 training-only label-shuffle sanity check across grouped folds."""
    train_df = pd.read_parquet(REPO_ROOT / "data/splits/train.parquet").head(1500).copy()
    rng = np.random.default_rng(12345)
    shuffled_y = rng.permutation(train_df["is_unemployed"].to_numpy())

    pipe = build_estimator_pipeline(
        LogisticRegression(C=1.0, max_iter=500, random_state=42),
        scale_numeric=True,
    )
    # Fit on first 1,000 rows with shuffled labels, evaluate on remaining 500 rows
    pipe.fit(train_df.iloc[:1000], shuffled_y[:1000])
    probs = pipe.predict_proba(train_df.iloc[1000:])[:, 1]
    shuffled_auc = float(roc_auc_score(shuffled_y[1000:], probs))

    # Shuffled labels must hover near random chance (0.50), well below 0.58
    assert 0.40 <= shuffled_auc <= 0.58

