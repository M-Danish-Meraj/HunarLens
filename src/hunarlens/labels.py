"""19th ICLS Youth Labour-Force Eligibility & Binary Unemployment Target Module.

Implements PROJECT_PLAN.md §1, §4.3, §5.1, §5.3 and docs/label_definition.md:
- Explicitly ignores and forbids the raw constant `Fstatus` column.
- Filters respondents to youth aged 15-24 (`age_out_of_range` logged for all others).
- Evaluates 19th ICLS labour-force status in strict precedence order:
    1. worked_last_week == 1 -> employed (is_unemployed = 0)
    2. worked_last_week == 0 & temp_absent_from_work == 1 -> employed (is_unemployed = 0)
    3. worked_last_week == 0 & temp_absent_from_work == 0 & seeking_work == 1 & available_for_work == 1
       -> unemployed (is_unemployed = 1)
    4. worked_last_week == 0 & temp_absent_from_work == 0 & (seeking_work == 0 | available_for_work == 0)
       (with non-missing responses) -> outside_labour_force (excluded from binary target)
    5. Missing, blank, or contradictory outcome-defining responses -> unresolved_or_missing (excluded)
- Never imputes missing outcome-defining responses to manufacture labels.
- Preserves valid employed/unemployed students in the labour force.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd


OUTCOME_DEFINING_COLUMNS: Tuple[str, ...] = (
    "worked_last_week",
    "temp_absent_from_work",
    "seeking_work",
    "available_for_work",
)


@dataclass(frozen=True)
class LabelAuditSummary:
    """Comprehensive readiness gate & label audit summary (§4.3, §5.1)."""

    total_records_input: int
    excluded_age_out_of_range: int
    youth_15_24_count: int
    employed_count: int
    unemployed_count: int
    excluded_outside_labour_force: int
    excluded_unresolved_or_missing: int
    eligible_labour_force_count: int
    unemployment_rate_unweighted: float
    unemployment_rate_weighted: float | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_binary_indicator(series: pd.Series) -> pd.Series:
    """Coerce a raw survey response column to {0.0, 1.0, np.nan} without imputing missing values."""
    if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
        mapping = {
            "1": 1.0,
            "0": 0.0,
            "1.0": 1.0,
            "0.0": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "true": 1.0,
            "false": 0.0,
        }
        cleaned = series.astype(str).str.strip().str.lower().map(mapping)
        return cleaned.astype(float)

    num = pd.to_numeric(series, errors="coerce").astype(float)
    valid_mask = num.isin([0.0, 1.0])
    return num.where(valid_mask, np.nan)


def derive_19th_icls_status(
    df: pd.DataFrame,
    age_col: str = "age",
    min_age: int = 15,
    max_age: int = 24,
) -> pd.DataFrame:
    """Assign `cohort_status`, `exclusion_reason`, and binary `is_unemployed` to each person record.

    Does not drop rows; callers can inspect full audit columns or use `build_eligible_youth_cohort`
    to filter to the eligible labour-force cohort.
    """
    missing_cols = [c for c in (age_col, *OUTCOME_DEFINING_COLUMNS) if c not in df.columns]
    if missing_cols:
        raise KeyError(f"DataFrame missing columns required for 19th ICLS label derivation: {missing_cols}")

    out = df.copy()
    age_num = pd.to_numeric(out[age_col], errors="coerce")
    in_youth_age = (age_num >= min_age) & (age_num <= max_age)

    worked = _coerce_binary_indicator(out["worked_last_week"])
    absent = _coerce_binary_indicator(out["temp_absent_from_work"])
    seeking = _coerce_binary_indicator(out["seeking_work"])
    available = _coerce_binary_indicator(out["available_for_work"])

    # Precedence 1: Worked >= 1 hour for pay or profit last week
    cond_emp_worked = in_youth_age & (worked == 1.0)

    # Precedence 2: Did not work last week, but temporarily absent from a paid job/business
    cond_emp_absent = in_youth_age & (worked == 0.0) & (absent == 1.0)

    is_employed = cond_emp_worked | cond_emp_absent

    # Precedence 3: Not employed, actively sought work, and currently available to start
    not_employed_verified = in_youth_age & (worked == 0.0) & (absent == 0.0)
    is_unemployed_mask = not_employed_verified & (seeking == 1.0) & (available == 1.0)

    # Precedence 4: Verified not employed and explicitly not seeking or not available (and neither is NaN)
    is_outside_lf = (
        not_employed_verified
        & seeking.notna()
        & available.notna()
        & ((seeking == 0.0) | (available == 0.0))
    )

    # Initialize status and exclusion reason arrays
    cohort_status = np.full(len(out), "unresolved_or_missing", dtype=object)
    exclusion_reason = np.full(len(out), "unresolved_or_missing", dtype=object)
    target_val = np.full(len(out), np.nan, dtype=float)

    # Age out of range
    out_of_age_mask = ~in_youth_age
    cohort_status[out_of_age_mask] = "age_out_of_range"
    exclusion_reason[out_of_age_mask] = "age_out_of_range"

    # Outside labour force
    cohort_status[is_outside_lf] = "outside_labour_force"
    exclusion_reason[is_outside_lf] = "outside_labour_force"

    # Employed (target = 0)
    cohort_status[is_employed] = "employed"
    exclusion_reason[is_employed] = "eligible_included"
    target_val[is_employed] = 0.0

    # Unemployed (target = 1)
    cohort_status[is_unemployed_mask] = "unemployed"
    exclusion_reason[is_unemployed_mask] = "eligible_included"
    target_val[is_unemployed_mask] = 1.0

    out["cohort_status"] = cohort_status
    out["exclusion_reason"] = exclusion_reason
    out["is_unemployed"] = pd.Series(target_val, index=out.index).astype("Int64")

    return out


def build_eligible_youth_cohort(
    df: pd.DataFrame,
    age_col: str = "age",
    min_age: int = 15,
    max_age: int = 24,
    weight_col: str = "weight",
    drop_outcome_inputs: bool = True,
) -> Tuple[pd.DataFrame, LabelAuditSummary]:
    """Filter person records to eligible youth (15-24) in the labour force and compute audit summary.

    When `drop_outcome_inputs=True`, removes raw outcome-defining survey columns (`worked_last_week`,
    `temp_absent_from_work`, `seeking_work`, `available_for_work`, `Fstatus`, `cohort_status`,
    `exclusion_reason`) from the returned modelling DataFrame so they can never leak into features (§5.3).
    """
    labeled_df = derive_19th_icls_status(df, age_col=age_col, min_age=min_age, max_age=max_age)

    total_input = len(labeled_df)
    status_counts = labeled_df["cohort_status"].value_counts()

    age_excluded = int(status_counts.get("age_out_of_range", 0))
    youth_total = total_input - age_excluded
    emp_count = int(status_counts.get("employed", 0))
    unemp_count = int(status_counts.get("unemployed", 0))
    outside_lf_count = int(status_counts.get("outside_labour_force", 0))
    unresolved_count = int(status_counts.get("unresolved_or_missing", 0))
    eligible_count = emp_count + unemp_count

    eligible_mask = labeled_df["cohort_status"].isin(["employed", "unemployed"])
    eligible_df = labeled_df.loc[eligible_mask].copy().reset_index(drop=True)
    eligible_df["is_unemployed"] = eligible_df["is_unemployed"].astype("int64")

    unweighted_rate = float(unemp_count / eligible_count) if eligible_count > 0 else 0.0

    weighted_rate: float | None = None
    if weight_col in eligible_df.columns and eligible_count > 0:
        weights = pd.to_numeric(eligible_df[weight_col], errors="coerce").fillna(0.0)
        total_weight = float(weights.sum())
        if total_weight > 0:
            weighted_rate = float((weights * eligible_df["is_unemployed"]).sum() / total_weight)

    if drop_outcome_inputs:
        cols_to_remove = [
            *OUTCOME_DEFINING_COLUMNS,
            "Fstatus",
            "cohort_status",
            "exclusion_reason",
        ]
        existing_remove = [c for c in cols_to_remove if c in eligible_df.columns]
        eligible_df = eligible_df.drop(columns=existing_remove)

    audit = LabelAuditSummary(
        total_records_input=total_input,
        excluded_age_out_of_range=age_excluded,
        youth_15_24_count=youth_total,
        employed_count=emp_count,
        unemployed_count=unemp_count,
        excluded_outside_labour_force=outside_lf_count,
        excluded_unresolved_or_missing=unresolved_count,
        eligible_labour_force_count=eligible_count,
        unemployment_rate_unweighted=round(unweighted_rate, 6),
        unemployment_rate_weighted=round(weighted_rate, 6) if weighted_rate is not None else None,
    )
    return eligible_df, audit


def assert_readiness_gate(
    df: pd.DataFrame,
    audit: LabelAuditSummary,
    approved_predictors: Sequence[str],
    forbidden_denylist: Sequence[str],
    min_rows: int = 5000,
    min_predictors: int = 15,
) -> None:
    """Enforce the Data Readiness Gate (§4.3, §5.3) before any splitting or modelling."""
    if audit.eligible_labour_force_count < min_rows or len(df) < min_rows:
        raise ValueError(
            f"Data Readiness Gate failed: eligible rows ({len(df)}) < minimum required ({min_rows})."
        )

    if len(approved_predictors) < min_predictors:
        raise ValueError(
            f"Data Readiness Gate failed: approved predictor count ({len(approved_predictors)}) < {min_predictors}."
        )

    missing_predictors = [p for p in approved_predictors if p not in df.columns]
    if missing_predictors:
        raise KeyError(f"Eligible cohort DataFrame missing approved predictors: {missing_predictors}")

    leak_overlap = sorted(set(approved_predictors).intersection(set(forbidden_denylist)))
    if leak_overlap:
        raise ValueError(
            f"Leakage violation: approved predictors intersect forbidden denylist: {leak_overlap}"
        )

    if df["is_unemployed"].isna().any():
        raise ValueError("Target column 'is_unemployed' contains missing values in eligible cohort.")

    unique_targets = set(df["is_unemployed"].unique().tolist())
    if unique_targets != {0, 1}:
        raise ValueError(f"Target column 'is_unemployed' must contain both classes {{0, 1}}, got {unique_targets}.")

