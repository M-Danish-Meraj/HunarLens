"""Reproducible Scikit-Learn Feature Engineering, Census Lookup, and ColumnTransformer Module.

Implements PROJECT_PLAN.md §5.2, §5.3, §6.4, §8:
- Provides importable custom scikit-learn transformers (`HunarLensSchemaAndCensusTransformer`) so saved
  `joblib` pipelines load cleanly in evaluation scripts and Streamlit Community Cloud.
- Automatically looks up fixed external Census 2023 regional covariates from `province_territory` at
  prediction time (so app users never type regional density or urban share).
- Computes `household_dependency_ratio` from household age-band counts if raw counts are supplied, and
  computes `regional_pop_density_log1p = log1p(regional_pop_density)`.
- Strictly filters columns to the 15 approved predictors (or a preplanned ablation subset), guaranteeing
  that IDs (`PCode`, `HHNo`, `SNo`), `weight`, and forbidden leakage columns can NEVER pass into a classifier.
- Wraps fold-local numeric median imputation (+ `StandardScaler` for linear models) and categorical explicit
  `"Unknown"` imputation + `OneHotEncoder(handle_unknown="ignore")`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hunarlens.data import load_census_reference


APPROVED_NUMERIC_FEATURES: Tuple[str, ...] = (
    "age",
    "household_size",
    "household_dependency_ratio",
    "regional_pop_density_log1p",
    "regional_urban_share",
    "regional_avg_household_size",
)

APPROVED_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "sex",
    "marital_status",
    "relationship_to_head",
    "can_read",
    "can_write",
    "highest_education",
    "current_enrolment",
    "province_territory",
    "locality",
)

ALL_APPROVED_15_FEATURES: Tuple[str, ...] = (
    *APPROVED_NUMERIC_FEATURES,
    *APPROVED_CATEGORICAL_FEATURES,
)

SUPPORTED_PROVINCES: Tuple[str, ...] = (
    "Khyber Pakhtunkhwa",
    "Punjab",
    "Sindh",
    "Balochistan",
    "Islamabad",
)

SUPPORTED_LOCALITIES: Tuple[str, ...] = ("Rural", "Urban")

DEFAULT_CENSUS_LOOKUP: Dict[str, Dict[str, float]] = {
    "Khyber Pakhtunkhwa": {
        "regional_pop_density": 401.57,
        "regional_urban_share": 0.1501,
        "regional_avg_household_size": 6.95,
    },
    "Punjab": {
        "regional_pop_density": 621.83,
        "regional_urban_share": 0.4070,
        "regional_avg_household_size": 6.06,
    },
    "Sindh": {
        "regional_pop_density": 395.25,
        "regional_urban_share": 0.5373,
        "regional_avg_household_size": 5.64,
    },
    "Balochistan": {
        "regional_pop_density": 42.90,
        "regional_urban_share": 0.3096,
        "regional_avg_household_size": 6.41,
    },
    "Islamabad": {
        "regional_pop_density": 2609.12,
        "regional_urban_share": 0.4690,
        "regional_avg_household_size": 5.59,
    },
}


def build_census_lookup_dict(census_path: str | Path | None = None) -> Dict[str, Dict[str, float]]:
    """Build a serializable dictionary mapping `province_territory` to its fixed Census 2023 indicators."""
    if census_path is None:
        return {k: dict(v) for k, v in DEFAULT_CENSUS_LOOKUP.items()}
    census_df = load_census_reference(census_path)
    lookup: Dict[str, Dict[str, float]] = {}
    for _, row in census_df.iterrows():
        prov = str(row["province_territory"])
        lookup[prov] = {
            "regional_pop_density": float(row["regional_pop_density"]),
            "regional_urban_share": float(row["regional_urban_share"]),
            "regional_avg_household_size": float(row["regional_avg_household_size"]),
        }
    return lookup


class HunarLensSchemaAndCensusTransformer(BaseEstimator, TransformerMixin):
    """Fixed schema validator, household feature engineer, and external Census 2023 lookup transformer (§8).

    Parameters
    ----------
    selected_features : Sequence[str] | None
        Subset of the 15 approved predictors to output. Defaults to all 15 approved predictors.
    census_lookup : Dict[str, Dict[str, float]] | None
        Fixed external Census 2023 lookup table keyed by `province_territory`.
    validate_domain : bool
        If True, enforces age [15, 24], valid `province_territory`, valid `locality`, and consistent
        household counts (`household_size >= 1`, `household_dependency_ratio >= 0`).
    """

    def __init__(
        self,
        selected_features: Sequence[str] | None = None,
        census_lookup: Dict[str, Dict[str, float]] | None = None,
        validate_domain: bool = True,
    ) -> None:
        self.selected_features = selected_features
        self.census_lookup = census_lookup
        self.validate_domain = validate_domain

    def _get_active_features(self) -> List[str]:
        if self.selected_features is None:
            return list(ALL_APPROVED_15_FEATURES)
        unknown = [f for f in self.selected_features if f not in ALL_APPROVED_15_FEATURES]
        if unknown:
            raise ValueError(
                f"Requested features are not in the 15 approved predictors allowlist: {unknown}"
            )
        return list(self.selected_features)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "HunarLensSchemaAndCensusTransformer":
        """Validate configuration; this stage performs fixed codebook/reference transformations only."""
        self.active_features_ = self._get_active_features()
        self.lookup_table_ = (
            self.census_lookup if self.census_lookup is not None else DEFAULT_CENSUS_LOOKUP
        )
        # Validate input schema during fit
        _ = self._prepare_dataframe(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate inputs, derive household/Census covariates if needed, and output strict feature allowlist."""
        return self._prepare_dataframe(X)

    def _prepare_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HunarLensSchemaAndCensusTransformer expects a pandas DataFrame input.")

        df = X.copy()
        lookup = getattr(self, "lookup_table_", self.census_lookup or DEFAULT_CENSUS_LOOKUP)
        active_features = getattr(self, "active_features_", self._get_active_features())

        # 1. Validate required geographic keys when used or needed for Census lookup
        if "province_territory" not in df.columns:
            raise KeyError("Missing required column 'province_territory'.")
        if "locality" in active_features and "locality" not in df.columns:
            raise KeyError("Missing required column 'locality'.")

        if self.validate_domain:
            invalid_prov = set(df["province_territory"].dropna().astype(str).unique()) - set(lookup.keys())
            if invalid_prov or df["province_territory"].isna().any():
                raise ValueError(
                    f"Unsupported or missing province_territory values: {sorted(invalid_prov)}"
                )
            if "locality" in df.columns:
                invalid_loc = set(df["locality"].dropna().astype(str).unique()) - set(SUPPORTED_LOCALITIES)
                if invalid_loc or df["locality"].isna().any():
                    raise ValueError(
                        f"Unsupported or missing locality values: {sorted(invalid_loc)}"
                    )

        # 2. Derive household_size and household_dependency_ratio if household age-band counts are passed
        if (
            ("household_size" not in df.columns or "household_dependency_ratio" not in df.columns)
            and {"dependents_count", "working_age_count"}.issubset(df.columns)
        ):
            dep = pd.to_numeric(df["dependents_count"], errors="coerce")
            work = pd.to_numeric(df["working_age_count"], errors="coerce")
            if self.validate_domain and ((dep < 0).any() or (work < 1).any()):
                raise ValueError(
                    "Contradictory household counts: an eligible youth aged 15-24 requires working_age_count >= 1 "
                    "and dependents_count >= 0."
                )
            df["household_size"] = (dep + work).astype(float)
            df["household_dependency_ratio"] = (dep / work.clip(lower=1)).astype(float)

        # 3. Perform deterministic Census 2023 lookup from `province_territory` whenever Census columns are absent
        for census_col in (
            "regional_pop_density",
            "regional_urban_share",
            "regional_avg_household_size",
        ):
            if census_col not in df.columns or df[census_col].isna().all():
                df[census_col] = df["province_territory"].astype(str).map(
                    lambda p, c=census_col: lookup[p][c] if p in lookup else np.nan
                )

        # 4. Derive `regional_pop_density_log1p` if needed
        if "regional_pop_density_log1p" in active_features:
            if (
                "regional_pop_density_log1p" not in df.columns
                or df["regional_pop_density_log1p"].isna().any()
            ):
                density = pd.to_numeric(df["regional_pop_density"], errors="coerce")
                df["regional_pop_density_log1p"] = np.log1p(density.clip(lower=0.0))

        # 5. Domain validation on core numeric inputs
        if self.validate_domain:
            if "age" in df.columns:
                ages = pd.to_numeric(df["age"], errors="coerce")
                out_of_bounds = ages.notna() & ((ages < 15) | (ages > 24))
                if out_of_bounds.any():
                    raise ValueError(
                        "Input contains age outside the eligible youth range [15, 24]."
                    )
            if "household_size" in df.columns:
                hh_s = pd.to_numeric(df["household_size"], errors="coerce")
                if (hh_s.notna() & (hh_s < 1)).any():
                    raise ValueError("Input contains invalid household_size < 1.")
            if "household_dependency_ratio" in df.columns:
                dep_r = pd.to_numeric(df["household_dependency_ratio"], errors="coerce")
                if (dep_r.notna() & (dep_r < 0.0)).any():
                    raise ValueError("Input contains negative household_dependency_ratio.")

        # 6. Check all active features exist and coerce types cleanly
        missing_feats = [f for f in active_features if f not in df.columns]
        if missing_feats:
            raise KeyError(f"Input DataFrame is missing required predictor columns: {missing_feats}")

        out = pd.DataFrame(index=df.index)
        for feat in active_features:
            if feat in APPROVED_NUMERIC_FEATURES:
                out[feat] = pd.to_numeric(df[feat], errors="coerce").astype(float)
            else:
                s = df[feat].astype(object)
                na_mask = s.isna() | (s.astype(str).str.strip().isin(["", "nan", "None", "<NA>"]))
                s = s.astype(str).str.strip()
                s = s.where(~na_mask, "Unknown")
                out[feat] = s.astype(object)

        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        active = getattr(self, "active_features_", self._get_active_features())
        return np.asarray(active, dtype=object)


def build_preprocessing_pipeline(
    scale_numeric: bool = True,
    selected_features: Sequence[str] | None = None,
    census_path: str | Path | None = None,
    validate_domain: bool = True,
) -> Pipeline:
    """Build the fold-local preprocessing Pipeline (`schema_census -> column_transformer`, §8).

    Parameters
    ----------
    scale_numeric : bool
        True for Logistic Regression (`StandardScaler`); False for tree/dummy models (`"passthrough"`).
    selected_features : Sequence[str] | None
        Optional subset of the 15 approved features (e.g. for preplanned training-CV ablations).
    census_path : str | Path | None
        Optional path to `data/reference/census_2023_regional.csv`.
    validate_domain : bool
        Whether to enforce domain bounds on age, household size, and geography.
    """
    active = list(selected_features) if selected_features is not None else list(ALL_APPROVED_15_FEATURES)
    num_cols = [c for c in active if c in APPROVED_NUMERIC_FEATURES]
    cat_cols = [c for c in active if c in APPROVED_CATEGORICAL_FEATURES]

    census_lookup = build_census_lookup_dict(census_path)

    numeric_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: List[Tuple[str, Any, List[str]]] = []
    if num_cols:
        transformers.append(("numeric", numeric_transformer, num_cols))
    if cat_cols:
        transformers.append(("categorical", categorical_transformer, cat_cols))

    col_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            (
                "schema_census",
                HunarLensSchemaAndCensusTransformer(
                    selected_features=active,
                    census_lookup=census_lookup,
                    validate_domain=validate_domain,
                ),
            ),
            ("preprocessor", col_transformer),
        ]
    )


def build_estimator_pipeline(
    estimator: Any,
    scale_numeric: bool = True,
    selected_features: Sequence[str] | None = None,
    census_path: str | Path | None = None,
    validate_domain: bool = True,
) -> Pipeline:
    """Wrap the full fold-local preprocessing steps and a scikit-learn `estimator` into a single `Pipeline`."""
    pre_pipe = build_preprocessing_pipeline(
        scale_numeric=scale_numeric,
        selected_features=selected_features,
        census_path=census_path,
        validate_domain=validate_domain,
    )
    return Pipeline(
        steps=[
            *pre_pipe.steps,
            ("classifier", estimator),
        ]
    )
