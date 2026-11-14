"""Validated Inference Wrapper for the Frozen HunarLens Scikit-Learn Pipeline.

Implements PROJECT_PLAN.md §8, §11, §12.1, §13:
- Loads the trusted frozen `models/final_pipeline.joblib`, `models/decision_threshold.json`,
  `models/feature_schema.json`, and `models/model_metadata.json`.
- Validates user profile inputs (enforcing age 15-24, supported geography, and consistent household counts).
- Automatically derives `household_size`, `household_dependency_ratio`, and external Census 2023 covariates
  (`regional_pop_density`, `regional_pop_density_log1p`, `regional_urban_share`, `regional_avg_household_size`)
  from the selected `province_territory` so app users never manually enter regional statistics.
- Returns estimated unemployment probability, the frozen validation-selected decision threshold, cautious
  non-causal interpretation text, and local feature explanations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

import hunarlens  # noqa: F401
from hunarlens.explain import compute_local_profile_explanation
from hunarlens.features import DEFAULT_CENSUS_LOOKUP, SUPPORTED_LOCALITIES, SUPPORTED_PROVINCES

REPO_ROOT = Path(__file__).resolve().parents[2]


class HunarLensPredictor:
    """Production/Streamlit inference wrapper around the frozen HunarLens pipeline artifact."""

    def __init__(
        self,
        pipeline_path: str | Path | None = None,
        threshold_path: str | Path | None = None,
        schema_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.pipeline_path = Path(pipeline_path or (REPO_ROOT / "models/final_pipeline.joblib"))
        self.threshold_path = Path(threshold_path or (REPO_ROOT / "models/decision_threshold.json"))
        self.schema_path = Path(schema_path or (REPO_ROOT / "models/feature_schema.json"))
        self.metadata_path = Path(metadata_path or (REPO_ROOT / "models/model_metadata.json"))

        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Frozen pipeline artifact not found: {self.pipeline_path}")

        self.pipeline = joblib.load(self.pipeline_path)
        self.threshold_info: Dict[str, Any] = json.loads(self.threshold_path.read_text(encoding="utf-8"))
        self.schema_info: Dict[str, Any] = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.metadata: Dict[str, Any] = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.decision_threshold: float = float(self.threshold_info["decision_threshold"])

        # Build a lightweight deterministic training-reference background table for fast local explanations
        self._reference_df = self._load_reference_background()

    def _load_reference_background(self, n_samples: int = 80) -> pd.DataFrame:
        train_path = REPO_ROOT / "data/splits/train.parquet"
        if train_path.exists():
            df = pd.read_parquet(train_path)
            return df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)
        # Fallback synthetic reference if private splits are absent on cloud host
        rows = []
        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            prov = str(rng.choice(SUPPORTED_PROVINCES))
            c = DEFAULT_CENSUS_LOOKUP[prov]
            dep = int(rng.integers(1, 5))
            work = int(rng.integers(2, 5))
            rows.append(
                {
                    "age": int(rng.integers(15, 25)),
                    "sex": str(rng.choice(["Male", "Female"])),
                    "marital_status": str(rng.choice(["Never Married", "Currently Married"])),
                    "relationship_to_head": str(rng.choice(["Son/Daughter", "Head", "Spouse", "Other Relative"])),
                    "can_read": str(rng.choice(["Yes", "No"], p=[0.78, 0.22])),
                    "can_write": str(rng.choice(["Yes", "No"], p=[0.76, 0.24])),
                    "highest_education": str(
                        rng.choice(
                            [
                                "No Formal Education",
                                "Primary",
                                "Middle",
                                "Matric",
                                "Intermediate",
                                "Degree & Postgraduate",
                            ]
                        )
                    ),
                    "current_enrolment": str(rng.choice(["Not Enrolled", "Currently Enrolled"], p=[0.72, 0.28])),
                    "household_size": dep + work,
                    "household_dependency_ratio": dep / work,
                    "province_territory": prov,
                    "locality": str(rng.choice(SUPPORTED_LOCALITIES)),
                    "regional_pop_density": c["regional_pop_density"],
                    "regional_pop_density_log1p": float(np.log1p(c["regional_pop_density"])),
                    "regional_urban_share": c["regional_urban_share"],
                    "regional_avg_household_size": c["regional_avg_household_size"],
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def validate_profile_dict(profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single profile dictionary and return normalized fields + derived Census/household values."""
        age = int(profile.get("age", -1))
        if age < 15 or age > 24:
            raise ValueError(f"Age ({age}) must be between 15 and 24 years for the eligible youth cohort.")

        prov = str(profile.get("province_territory", "")).strip()
        if prov not in SUPPORTED_PROVINCES:
            raise ValueError(f"Unsupported province_territory '{prov}'. Allowed: {list(SUPPORTED_PROVINCES)}")

        loc = str(profile.get("locality", "")).strip()
        if loc not in SUPPORTED_LOCALITIES:
            raise ValueError(f"Unsupported locality '{loc}'. Allowed: {list(SUPPORTED_LOCALITIES)}")

        if "dependents_count" in profile and "working_age_count" in profile:
            dep = int(profile["dependents_count"])
            work = int(profile["working_age_count"])
            if dep < 0 or work < 1:
                raise ValueError(
                    "Contradictory household counts: an eligible youth aged 15-24 requires at least 1 working-age "
                    "household member (15-64) and non-negative dependents (<15 or 65+)."
                )
            hh_size = dep + work
            dep_ratio = float(dep / work)
        else:
            hh_size = int(profile.get("household_size", 0))
            dep_ratio = float(profile.get("household_dependency_ratio", -1.0))
            if hh_size < 1 or dep_ratio < 0.0:
                raise ValueError("Invalid household_size (<1) or household_dependency_ratio (<0).")

        census_vals = DEFAULT_CENSUS_LOOKUP[prov]
        normalized = {
            "age": age,
            "sex": str(profile.get("sex", "Unknown")),
            "marital_status": str(profile.get("marital_status", "Unknown")),
            "relationship_to_head": str(profile.get("relationship_to_head", "Unknown")),
            "can_read": str(profile.get("can_read", "Unknown")),
            "can_write": str(profile.get("can_write", "Unknown")),
            "highest_education": str(profile.get("highest_education", "Unknown")),
            "current_enrolment": str(profile.get("current_enrolment", "Unknown")),
            "household_size": hh_size,
            "household_dependency_ratio": round(dep_ratio, 4),
            "province_territory": prov,
            "locality": loc,
            "regional_pop_density": census_vals["regional_pop_density"],
            "regional_pop_density_log1p": round(float(np.log1p(census_vals["regional_pop_density"])), 6),
            "regional_urban_share": census_vals["regional_urban_share"],
            "regional_avg_household_size": census_vals["regional_avg_household_size"],
        }
        return normalized

    def predict_profile(self, profile: Dict[str, Any], include_explanation: bool = True) -> Dict[str, Any]:
        """Score a single youth profile and return probability, threshold decision, and local explanation."""
        norm_profile = self.validate_profile_dict(profile)
        df = pd.DataFrame([norm_profile])

        prob = float(self.pipeline.predict_proba(df)[0, 1])
        pred_class = int(prob >= self.decision_threshold)

        decision_text = (
            "Above Validation Threshold (Flagged as Unemployed Class = 1)"
            if pred_class == 1
            else "Below Validation Threshold (Classified as Employed Class = 0)"
        )

        result: Dict[str, Any] = {
            "estimated_probability": round(prob, 6),
            "decision_threshold": self.decision_threshold,
            "predicted_class": pred_class,
            "decision_summary": decision_text,
            "model_display_name": self.metadata["selected_model_display_name"],
            "normalized_inputs": norm_profile,
            "cautious_note": (
                "Estimated probability of current survey-defined unemployment conditional on labour-force "
                "membership (PBS LFS 2024–25 + Census 2023). This is an observational association, not a "
                "causal effect or forecast."
            ),
        }
        if include_explanation:
            result["local_explanation"] = compute_local_profile_explanation(
                self.pipeline,
                df,
                self._reference_df,
                top_k=10,
            )
        return result

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Return positive-class (`is_unemployed == 1`) probabilities for a batch DataFrame."""
        return np.asarray(self.pipeline.predict_proba(df)[:, 1], dtype=float)

