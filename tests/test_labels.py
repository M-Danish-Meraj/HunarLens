"""Unit tests for 19th ICLS target label derivation, household roster math, and Census crosswalk integrity.

Implements PROJECT_PLAN.md §4.2, §4.3, §5.1, §5.2, §5.3 & Step 2.4 of PROJECT_TRACKER.md.
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

from hunarlens.data import (  # noqa: E402
    compute_household_roster_features,
    join_census_context,
    load_census_reference,
    resolve_province_from_pcode,
    validate_and_deduplicate_identifiers,
)
from hunarlens.labels import (  # noqa: E402
    OUTCOME_DEFINING_COLUMNS,
    assert_readiness_gate,
    build_eligible_youth_cohort,
    derive_19th_icls_status,
)


def _sample_worked_examples_df() -> pd.DataFrame:
    """Construct the 6 worked examples from `docs/label_definition.md` plus out-of-age cases."""
    return pd.DataFrame(
        [
            # Ex-1: Age 21, worked_last_week=1 -> employed (0)
            {
                "PCode": "20010001",
                "HHNo": "1",
                "SNo": 1,
                "age": 21,
                "worked_last_week": 1,
                "temp_absent_from_work": 0,
                "seeking_work": 0,
                "available_for_work": 1,
                "current_enrolment": "Not Enrolled",
                "Fstatus": 1,
                "weight": 100.0,
            },
            # Ex-2: Age 19, worked=0, temp_absent=1, enrolled -> employed (0)
            {
                "PCode": "20010001",
                "HHNo": "1",
                "SNo": 2,
                "age": 19,
                "worked_last_week": 0,
                "temp_absent_from_work": 1,
                "seeking_work": 0,
                "available_for_work": 1,
                "current_enrolment": "Currently Enrolled",
                "Fstatus": 1,
                "weight": 100.0,
            },
            # Ex-3: Age 23, worked=0, absent=0, seeking=1, available=1 -> unemployed (1)
            {
                "PCode": "30010002",
                "HHNo": "105",
                "SNo": 1,
                "age": 23,
                "worked_last_week": 0,
                "temp_absent_from_work": 0,
                "seeking_work": 1,
                "available_for_work": 1,
                "current_enrolment": "Not Enrolled",
                "Fstatus": 1,
                "weight": 200.0,
            },
            # Ex-4: Age 18, student actively seeking and available -> unemployed (1)
            {
                "PCode": "60010003",
                "HHNo": "2",
                "SNo": 1,
                "age": 18,
                "worked_last_week": 0,
                "temp_absent_from_work": 0,
                "seeking_work": 1,
                "available_for_work": 1,
                "current_enrolment": "Currently Enrolled",
                "Fstatus": 1,
                "weight": 150.0,
            },
            # Ex-5: Age 17, inactive student not seeking work -> outside_labour_force (excluded)
            {
                "PCode": "60010003",
                "HHNo": "2",
                "SNo": 2,
                "age": 17,
                "worked_last_week": 0,
                "temp_absent_from_work": 0,
                "seeking_work": 0,
                "available_for_work": 0,
                "current_enrolment": "Currently Enrolled",
                "Fstatus": 1,
                "weight": 150.0,
            },
            # Ex-6: Age 22, missing outcome-defining responses -> unresolved_or_missing (excluded, never imputed)
            {
                "PCode": "10010004",
                "HHNo": "3",
                "SNo": 1,
                "age": 22,
                "worked_last_week": np.nan,
                "temp_absent_from_work": np.nan,
                "seeking_work": 1,
                "available_for_work": np.nan,
                "current_enrolment": "Not Enrolled",
                "Fstatus": 1,
                "weight": 120.0,
            },
            # Ex-7: Age 12 (child) -> age_out_of_range (excluded)
            {
                "PCode": "10010004",
                "HHNo": "3",
                "SNo": 2,
                "age": 12,
                "worked_last_week": 0,
                "temp_absent_from_work": 0,
                "seeking_work": 0,
                "available_for_work": 0,
                "current_enrolment": "Currently Enrolled",
                "Fstatus": 1,
                "weight": 120.0,
            },
            # Ex-8: Age 45 (household head) -> age_out_of_range (excluded)
            {
                "PCode": "10010004",
                "HHNo": "3",
                "SNo": 3,
                "age": 45,
                "worked_last_week": 1,
                "temp_absent_from_work": 0,
                "seeking_work": 0,
                "available_for_work": 1,
                "current_enrolment": "Not Enrolled",
                "Fstatus": 1,
                "weight": 120.0,
            },
        ]
    )


def test_19th_icls_status_precedence_and_worked_examples() -> None:
    """Verify all 6 worked examples and age exclusions match `docs/label_definition.md`."""
    df = _sample_worked_examples_df()
    labeled = derive_19th_icls_status(df)

    expected_statuses = [
        "employed",
        "employed",
        "unemployed",
        "unemployed",
        "outside_labour_force",
        "unresolved_or_missing",
        "age_out_of_range",
        "age_out_of_range",
    ]
    assert labeled["cohort_status"].tolist() == expected_statuses

    eligible_df, audit = build_eligible_youth_cohort(df, drop_outcome_inputs=True)
    assert audit.total_records_input == 8
    assert audit.excluded_age_out_of_range == 2
    assert audit.youth_15_24_count == 6
    assert audit.employed_count == 2
    assert audit.unemployed_count == 2
    assert audit.excluded_outside_labour_force == 1
    assert audit.excluded_unresolved_or_missing == 1
    assert audit.eligible_labour_force_count == 4
    assert len(eligible_df) == 4
    assert eligible_df["is_unemployed"].tolist() == [0, 0, 1, 1]

    # Verify outcome-defining columns and constant Fstatus are stripped
    for forbidden_col in (*OUTCOME_DEFINING_COLUMNS, "Fstatus", "cohort_status", "exclusion_reason"):
        assert forbidden_col not in eligible_df.columns


def test_household_roster_features_before_youth_filtering() -> None:
    """Verify `household_size` and `household_dependency_ratio` count all usual residents across all ages."""
    roster = pd.DataFrame(
        [
            # Household A: 2 children (<15), 1 senior (68), 2 working-age (42, 20) -> size=5, dep_ratio = 3 / 2 = 1.5
            {"PCode": "20010001", "HHNo": "105", "SNo": 1, "age": 42},
            {"PCode": "20010001", "HHNo": "105", "SNo": 2, "age": 20},
            {"PCode": "20010001", "HHNo": "105", "SNo": 3, "age": 10},
            {"PCode": "20010001", "HHNo": "105", "SNo": 4, "age": 6},
            {"PCode": "20010001", "HHNo": "105", "SNo": 5, "age": 68},
        ]
    )
    enriched = compute_household_roster_features(roster)
    assert len(enriched) == 5
    assert (enriched["household_size"] == 5).all()
    assert np.allclose(enriched["household_dependency_ratio"], 1.5)


def test_pcode_province_resolution_separates_islamabad() -> None:
    """Verify first digit of PCode ('6') resolves to 'Islamabad' even when raw Province is 'Punjab'."""
    df = pd.DataFrame(
        [
            {"PCode": "60012345", "Province": "Punjab"},
            {"PCode": "20012345", "Province": "Punjab"},
            {"PCode": "10012345", "Province": "Khyber Pakhtunkhwa"},
            {"PCode": "30012345", "Province": "Sindh"},
            {"PCode": "40012345", "Province": "Balochistan"},
        ]
    )
    resolved = resolve_province_from_pcode(df)
    assert resolved["province_territory"].tolist() == [
        "Islamabad",
        "Punjab",
        "Khyber Pakhtunkhwa",
        "Sindh",
        "Balochistan",
    ]


def test_identifier_deduplication_and_non_truncated_hhno() -> None:
    """Verify 3-digit `HHNo` is preserved and duplicate `(PCode, HHNo, SNo)` rows are removed."""
    df = pd.DataFrame(
        [
            {"PCode": "20010001", "HHNo": "108", "SNo": 1, "age": 20},
            {"PCode": "20010001", "HHNo": "108", "SNo": 1, "age": 20},  # duplicate person key
            {"PCode": "20010001", "HHNo": "08", "SNo": 1, "age": 21},   # distinct HHNo ('08' vs '108')
            {"PCode": None, "HHNo": "1", "SNo": 1, "age": 22},          # missing PCode -> dropped
        ]
    )
    clean_df, summary = validate_and_deduplicate_identifiers(df)
    assert summary.raw_row_count == 4
    assert summary.missing_identifier_rows_dropped == 1
    assert summary.duplicate_person_rows_dropped == 1
    assert summary.clean_row_count == 2
    assert summary.unique_household_count == 2
    assert clean_df["HHNo"].tolist() == ["108", "08"]


def test_census_many_to_one_join_invariance() -> None:
    """Verify Census 2023 lookup table loads cleanly and joins with strict m:1 row-count invariance."""
    census_df = load_census_reference(REPO_ROOT / "data/reference/census_2023_regional.csv")
    assert len(census_df) == 5

    person_df = pd.DataFrame(
        [
            {"PCode": "60010001", "HHNo": "1", "SNo": 1, "province_territory": "Islamabad"},
            {"PCode": "20010002", "HHNo": "1", "SNo": 1, "province_territory": "Punjab"},
            {"PCode": "20010002", "HHNo": "1", "SNo": 2, "province_territory": "Punjab"},
        ]
    )
    merged = join_census_context(person_df, census_df)
    assert len(merged) == len(person_df)
    assert "regional_pop_density_log1p" in merged.columns
    assert merged["regional_pop_density_log1p"].isna().sum() == 0

    # Unmatched region must raise ValueError
    bad_df = pd.DataFrame([{"province_territory": "Unverified_Region"}])
    with pytest.raises(ValueError, match="Unmatched regions"):
        join_census_context(bad_df, census_df)


def test_readiness_gate_rejects_leakage_or_small_sample() -> None:
    """Verify `assert_readiness_gate` fails if row count < min_rows or leakage column is in predictors."""
    df = _sample_worked_examples_df()
    eligible_df, audit = build_eligible_youth_cohort(df)

    with pytest.raises(ValueError, match="Data Readiness Gate failed: eligible rows"):
        assert_readiness_gate(
            df=eligible_df,
            audit=audit,
            approved_predictors=["age"] * 15,
            forbidden_denylist=["worked_last_week"],
            min_rows=5000,
        )

