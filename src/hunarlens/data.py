"""Data ingestion, identifier validation, household roster aggregation, and Census crosswalk module.

Implements PROJECT_PLAN.md §4.1, §4.2, and §5.2:
- Recovers and validates 8-digit PSU (`PCode`), household (`PCode`, `HHNo`), and person (`PCode`, `HHNo`, `SNo`) keys.
- Resolves `province_territory` from the first digit of `PCode` (distinguishing Islamabad '6' from Punjab '2').
- Computes household-level demographic features (`household_size`, `household_dependency_ratio`) across the full
  usual-resident household roster BEFORE youth filtering.
- Performs a verified many-to-one (`validate="m:1"`) join with independent Census 2023 regional indicators and
  asserts strict row-count invariance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


DEFAULT_PCODE_FIRST_DIGIT_MAP: Dict[str, str] = {
    "1": "Khyber Pakhtunkhwa",
    "2": "Punjab",
    "3": "Sindh",
    "4": "Balochistan",
    "6": "Islamabad",
}

REQUIRED_CENSUS_COLUMNS: Tuple[str, ...] = (
    "province_territory",
    "regional_pop_density",
    "regional_urban_share",
    "regional_avg_household_size",
)


@dataclass(frozen=True)
class IdentifierAuditSummary:
    """Audit counts for raw person, household, and PSU identifier integrity."""

    raw_row_count: int
    missing_identifier_rows_dropped: int
    duplicate_person_rows_dropped: int
    clean_row_count: int
    unique_psu_count: int
    unique_household_count: int
    unique_person_count: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and parse a declarative YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML config: {path}")
    return data


def load_census_reference(census_path: str | Path) -> pd.DataFrame:
    """Load and validate the external Census 2023 regional reference table (§4.2).

    Asserts:
    - All required Census indicator columns are present.
    - `province_territory` has zero duplicates (one row per geographic key).
    - Density, urban share, and average household size are strictly positive and within valid bounds.
    """
    path = Path(census_path)
    if not path.exists():
        raise FileNotFoundError(f"Census reference table not found: {path}")

    census_df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_CENSUS_COLUMNS if c not in census_df.columns]
    if missing_cols:
        raise ValueError(f"Census table missing required columns: {missing_cols}")

    if census_df["province_territory"].isna().any():
        raise ValueError("Census table contains null province_territory keys.")

    if census_df["province_territory"].duplicated().any():
        dups = census_df.loc[census_df["province_territory"].duplicated(), "province_territory"].tolist()
        raise ValueError(f"Census table must have a unique row per province_territory; duplicates: {dups}")

    if (census_df["regional_pop_density"] <= 0).any():
        raise ValueError("Census regional_pop_density must be strictly positive.")

    if ((census_df["regional_urban_share"] < 0.0) | (census_df["regional_urban_share"] > 1.0)).any():
        raise ValueError("Census regional_urban_share must lie within [0.0, 1.0].")

    if (census_df["regional_avg_household_size"] <= 0).any():
        raise ValueError("Census regional_avg_household_size must be strictly positive.")

    return census_df.copy()


def normalize_identifiers(
    df: pd.DataFrame,
    psu_col: str = "PCode",
    hh_col: str = "HHNo",
    person_col: str = "SNo",
) -> pd.DataFrame:
    """Normalize PSU (`PCode`), household (`HHNo`), and person (`SNo`) identifiers without truncating `HHNo`.

    Per PROJECT_PLAN.md §1 Data-intake update:
    - `PCode` is normalized to an 8-digit zero-padded string.
    - `HHNo` is normalized as a stripped string (never truncated to 2 digits).
    - `SNo` is normalized as a stripped integer/string representation.
    """
    required = [psu_col, hh_col, person_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame missing required identifier columns: {missing}")

    out = df.copy()
    for col in required:
        # Preserve NA so missing identifiers can be audited and dropped cleanly
        not_na = out[col].notna()
        out[col] = out[col].astype(object)
        cleaned = (
            out.loc[not_na, col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        if col == psu_col:
            cleaned = cleaned.str.zfill(8)
        out.loc[not_na, col] = cleaned

    return out


def validate_and_deduplicate_identifiers(
    df: pd.DataFrame,
    psu_col: str = "PCode",
    hh_cols: Sequence[str] = ("PCode", "HHNo"),
    person_cols: Sequence[str] = ("PCode", "HHNo", "SNo"),
) -> Tuple[pd.DataFrame, IdentifierAuditSummary]:
    """Validate PSU/household/person keys, drop missing-ID rows, and deduplicate on `(PCode, HHNo, SNo)`."""
    raw_count = len(df)
    norm_df = normalize_identifiers(df, psu_col=psu_col, hh_col=hh_cols[1], person_col=person_cols[2])

    # Identify rows with missing or empty identifiers
    id_missing_mask = pd.Series(False, index=norm_df.index)
    for col in person_cols:
        id_missing_mask |= norm_df[col].isna() | (norm_df[col].astype(str).str.strip() == "")

    missing_id_count = int(id_missing_mask.sum())
    valid_id_df = norm_df.loc[~id_missing_mask].copy()

    # Deduplicate person keys keeping first occurrence and recording audit count
    dup_mask = valid_id_df.duplicated(subset=list(person_cols), keep="first")
    dup_count = int(dup_mask.sum())
    clean_df = valid_id_df.loc[~dup_mask].reset_index(drop=True)

    unique_psus = int(clean_df[psu_col].nunique())
    unique_hhs = int(clean_df.groupby(list(hh_cols), dropna=False).ngroups) if len(clean_df) > 0 else 0
    unique_persons = len(clean_df)

    summary = IdentifierAuditSummary(
        raw_row_count=raw_count,
        missing_identifier_rows_dropped=missing_id_count,
        duplicate_person_rows_dropped=dup_count,
        clean_row_count=len(clean_df),
        unique_psu_count=unique_psus,
        unique_household_count=unique_hhs,
        unique_person_count=unique_persons,
    )
    return clean_df, summary


def resolve_province_from_pcode(
    df: pd.DataFrame,
    pcode_col: str = "PCode",
    pcode_map: Dict[str, str] | None = None,
    output_col: str = "province_territory",
) -> pd.DataFrame:
    """Derive verified `province_territory` from the first digit of the 8-digit `PCode` (§1 Data-intake update).

    Distinguishes Islamabad ('6') from Punjab ('2'), which the raw `Province` column conflates.
    """
    if pcode_col not in df.columns:
        raise KeyError(f"Missing PSU column '{pcode_col}' required for province resolution.")

    mapping = pcode_map or DEFAULT_PCODE_FIRST_DIGIT_MAP
    out = df.copy()
    first_digit = out[pcode_col].astype(str).str.strip().str.zfill(8).str[0]
    out[output_col] = first_digit.map(mapping)
    return out


def compute_household_roster_features(
    df: pd.DataFrame,
    hh_cols: Sequence[str] = ("PCode", "HHNo"),
    age_col: str = "age",
    working_age_min: int = 15,
    working_age_max: int = 64,
) -> pd.DataFrame:
    """Compute `household_size` and `household_dependency_ratio` across the full household roster (§5.2).

    MUST be called on the complete usual-resident household table BEFORE filtering to youth aged 15-24.
    Formula:
      - `household_size`: count of all usual residents in `(PCode, HHNo)`
      - `household_dependency_ratio`: count(age < 15 or age >= 65) / count(15 <= age <= 64)
        (For households with 0 working-age members—which cannot contain an eligible youth aged 15-24—
         denominator is safely floored at 1 to avoid division by zero before youth filtering.)
    """
    for col in [*hh_cols, age_col]:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' required for household roster aggregation.")

    out = df.copy()
    numeric_age = pd.to_numeric(out[age_col], errors="coerce")

    out["_is_dependent"] = ((numeric_age < working_age_min) | (numeric_age > working_age_max)).astype(int)
    out["_is_working_age"] = (
        (numeric_age >= working_age_min) & (numeric_age <= working_age_max)
    ).astype(int)

    roster_agg = (
        out.groupby(list(hh_cols), as_index=False, sort=False)
        .agg(
            household_size=(age_col, "size"),
            _dependent_count=("_is_dependent", "sum"),
            _working_age_count=("_is_working_age", "sum"),
        )
    )

    safe_working_age = roster_agg["_working_age_count"].clip(lower=1)
    roster_agg["household_dependency_ratio"] = (
        roster_agg["_dependent_count"] / safe_working_age
    ).astype(float)

    roster_agg = roster_agg.drop(columns=["_dependent_count", "_working_age_count"])

    # Drop pre-existing columns if recomputing
    drop_existing = [c for c in ("household_size", "household_dependency_ratio") if c in out.columns]
    out = out.drop(columns=drop_existing + ["_is_dependent", "_is_working_age"])

    pre_len = len(out)
    merged = out.merge(roster_agg, on=list(hh_cols), how="left", validate="m:1")
    if len(merged) != pre_len:
        raise AssertionError(
            f"Household roster merge altered person row count from {pre_len} to {len(merged)}."
        )

    return merged


def join_census_context(
    df: pd.DataFrame,
    census_df: pd.DataFrame,
    join_key: str = "province_territory",
) -> pd.DataFrame:
    """Join external Census 2023 regional indicators via a strict `many-to-one` check (§4.2).

    Asserts:
    - Row count of `df` is strictly unchanged after the join.
    - Every row with a valid `province_territory` matches a Census reference record.
    - Adds `regional_pop_density_log1p = log1p(regional_pop_density)` alongside raw Census columns.
    """
    if join_key not in df.columns:
        raise KeyError(f"Person DataFrame is missing join key '{join_key}'.")
    if join_key not in census_df.columns:
        raise KeyError(f"Census DataFrame is missing join key '{join_key}'.")

    census_cols = [
        join_key,
        "regional_pop_density",
        "regional_urban_share",
        "regional_avg_household_size",
    ]
    census_sub = census_df[census_cols].copy()

    # Drop any pre-existing Census columns before joining
    existing_census_cols = [
        c for c in ("regional_pop_density", "regional_pop_density_log1p", "regional_urban_share", "regional_avg_household_size")
        if c in df.columns
    ]
    base_df = df.drop(columns=existing_census_cols) if existing_census_cols else df.copy()

    pre_row_count = len(base_df)
    merged = base_df.merge(census_sub, on=join_key, how="left", validate="m:1")

    if len(merged) != pre_row_count:
        raise AssertionError(
            f"Census join violated row-count invariance: expected {pre_row_count}, got {len(merged)}."
        )

    unmatched_mask = merged[join_key].notna() & merged["regional_pop_density"].isna()
    if unmatched_mask.any():
        unmatched_regions = merged.loc[unmatched_mask, join_key].unique().tolist()
        raise ValueError(f"Unmatched regions in Census crosswalk join: {unmatched_regions}")

    merged["regional_pop_density_log1p"] = np.log1p(
        merged["regional_pop_density"].astype(float)
    )
    return merged

