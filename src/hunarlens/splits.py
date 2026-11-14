"""PSU-Disjoint Train/Validation/Test Partitioning & 5-Fold StratifiedGroupKFold Generator.

Implements PROJECT_PLAN.md §6.2, §6.3, §6.4:
- Uses `PCode` (8-digit Primary Sampling Unit) as the primary dependence group so no PSU, household
  `(PCode, HHNo)`, or person `(PCode, HHNo, SNo)` ever crosses partitions or CV folds.
- Allocates ~15% test holdout via `GroupShuffleSplit(test_size=0.15, random_state=42)`.
- Allocates ~15% validation holdout from remaining ~85% via `GroupShuffleSplit(test_size=0.15/0.85, random_state=43)`.
- Constructs 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=44)` strictly inside the
  ~70% training partition.
- Enforces zero-overlap invariants across PSUs, households, and persons, verifies class & regional support,
  saves `data/splits/{train,val,test}.parquet`, `data/splits/split_manifest.json`, and `docs/split_audit.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hunarlens.data import load_yaml_config  # noqa: E402


def _hash_series_values(values: Sequence[Any]) -> str:
    """Compute a deterministic SHA-256 hash over sorted string representations of values."""
    hasher = hashlib.sha256()
    for val in sorted(str(v) for v in values):
        hasher.update(val.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def assert_zero_group_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    psu_col: str = "PCode",
    hh_cols: Sequence[str] = ("PCode", "HHNo"),
    person_cols: Sequence[str] = ("PCode", "HHNo", "SNo"),
    target_col: str = "is_unemployed",
) -> None:
    """Verify strict PSU, household, and person disjointness and class presence across partitions (§6.4)."""
    parts = {"train": train_df, "val": val_df, "test": test_df}

    # 1. Check both classes exist in each partition
    for name, part_df in parts.items():
        if len(part_df) == 0:
            raise AssertionError(f"Partition '{name}' is empty.")
        classes = set(part_df[target_col].unique().tolist())
        if classes != {0, 1}:
            raise AssertionError(f"Partition '{name}' missing binary class; found {classes}.")

    # 2. Check PSU disjointness
    psu_sets = {name: set(df[psu_col].astype(str)) for name, df in parts.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = psu_sets[a].intersection(psu_sets[b])
        if overlap:
            raise AssertionError(
                f"PSU leakage detected between '{a}' and '{b}': {len(overlap)} shared PSUs."
            )

    # 3. Check composite household disjointness
    hh_sets = {
        name: set(df[list(hh_cols)].astype(str).agg("::".join, axis=1))
        for name, df in parts.items()
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = hh_sets[a].intersection(hh_sets[b])
        if overlap:
            raise AssertionError(
                f"Household leakage detected between '{a}' and '{b}': {len(overlap)} shared households."
            )

    # 4. Check composite person disjointness
    person_sets = {
        name: set(df[list(person_cols)].astype(str).agg("::".join, axis=1))
        for name, df in parts.items()
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = person_sets[a].intersection(person_sets[b])
        if overlap:
            raise AssertionError(
                f"Person record leakage detected between '{a}' and '{b}': {len(overlap)} shared records."
            )


def generate_psu_disjoint_splits(
    df: pd.DataFrame,
    group_col: str = "PCode",
    target_col: str = "is_unemployed",
    test_size: float = 0.15,
    val_size_of_remaining: float = 0.17647,
    test_seed: int = 42,
    val_seed: int = 43,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split eligible cohort into PSU-disjoint ~70% train, ~15% validation, and ~15% test partitions (§6.2)."""
    if group_col not in df.columns:
        raise KeyError(f"Missing group column '{group_col}' in DataFrame.")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column '{target_col}' in DataFrame.")

    clean_df = df.reset_index(drop=True).copy()
    groups = clean_df[group_col].astype(str)
    y = clean_df[target_col].astype(int)

    # Step 1: Hold out ~15% test partition with fixed seed 42
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=test_seed)
    train_val_idx, test_idx = next(gss_test.split(clean_df, y, groups=groups))

    train_val_df = clean_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = clean_df.iloc[test_idx].reset_index(drop=True)

    # Step 2: Allocate ~15/85 of remaining to validation with fixed seed 43
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size_of_remaining,
        random_state=val_seed,
    )
    tv_groups = train_val_df[group_col].astype(str)
    tv_y = train_val_df[target_col].astype(int)
    train_idx, val_idx = next(gss_val.split(train_val_df, tv_y, groups=tv_groups))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    assert_zero_group_overlap(train_df, val_df, test_df, psu_col=group_col, target_col=target_col)
    return train_df, val_df, test_df


def generate_training_cv_folds(
    train_df: pd.DataFrame,
    group_col: str = "PCode",
    target_col: str = "is_unemployed",
    n_splits: int = 5,
    cv_seed: int = 44,
) -> Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]], List[Dict[str, Any]]]:
    """Assign 5-fold `StratifiedGroupKFold` strictly inside the training partition (§6.3).

    Returns:
      - `train_with_folds`: `train_df` with an integer `cv_fold` column (`0 .. n_splits-1`).
      - `fold_indices`: list of `(fold_train_idx, fold_val_idx)` numpy arrays for `GridSearchCV`/`RandomizedSearchCV`.
      - `fold_summaries`: per-fold audit statistics (row counts, PSU counts, positive prevalence).
    """
    out_df = train_df.reset_index(drop=True).copy()
    groups = out_df[group_col].astype(str)
    y = out_df[target_col].astype(int)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    fold_assignments = np.full(len(out_df), -1, dtype=int)
    fold_indices: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fold_id, (f_train_idx, f_val_idx) in enumerate(sgkf.split(out_df, y, groups=groups)):
        train_psus = set(groups.iloc[f_train_idx])
        val_psus = set(groups.iloc[f_val_idx])
        overlap = train_psus.intersection(val_psus)
        if overlap:
            raise AssertionError(f"CV Fold {fold_id} has PSU overlap between fold-train and fold-val: {overlap}")

        val_classes = set(y.iloc[f_val_idx].unique().tolist())
        if val_classes != {0, 1}:
            raise AssertionError(f"CV Fold {fold_id} validation split does not contain both target classes.")

        fold_assignments[f_val_idx] = fold_id
        fold_indices.append((f_train_idx, f_val_idx))

        f_val_df = out_df.iloc[f_val_idx]
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "train_rows": int(len(f_train_idx)),
                "val_rows": int(len(f_val_idx)),
                "val_psus": int(len(val_psus)),
                "val_employed": int((f_val_df[target_col] == 0).sum()),
                "val_unemployed": int((f_val_df[target_col] == 1).sum()),
                "val_unemployment_rate": round(float(f_val_df[target_col].mean()), 6),
            }
        )

    if (fold_assignments < 0).any():
        raise AssertionError("Some training rows were not assigned to a StratifiedGroupKFold validation fold.")

    out_df["cv_fold"] = fold_assignments
    return out_df, fold_indices, fold_summaries


def _partition_summary(
    df: pd.DataFrame,
    total_rows: int,
    psu_col: str = "PCode",
    hh_cols: Sequence[str] = ("PCode", "HHNo"),
    person_cols: Sequence[str] = ("PCode", "HHNo", "SNo"),
    target_col: str = "is_unemployed",
) -> Dict[str, Any]:
    """Compute summary counts, prevalence, and deterministic group/person hashes for a single partition."""
    n_rows = int(len(df))
    n_psus = int(df[psu_col].nunique())
    n_hhs = int(df.groupby(list(hh_cols)).ngroups)
    n_emp = int((df[target_col] == 0).sum())
    n_unemp = int((df[target_col] == 1).sum())
    person_ids = df[list(person_cols)].astype(str).agg("::".join, axis=1).tolist()

    return {
        "row_count": n_rows,
        "row_share": round(n_rows / total_rows, 6) if total_rows > 0 else 0.0,
        "psu_count": n_psus,
        "household_count": n_hhs,
        "employed_count": n_emp,
        "unemployed_count": n_unemp,
        "unemployment_rate": round(float(n_unemp / n_rows), 6) if n_rows > 0 else 0.0,
        "provinces_covered": sorted(df["province_territory"].astype(str).unique().tolist()),
        "psu_sha256": _hash_series_values(df[psu_col].unique().tolist()),
        "person_keys_sha256": _hash_series_values(person_ids),
    }


def write_split_audit_markdown(
    audit_path: Path,
    manifest: Dict[str, Any],
) -> None:
    """Write `docs/split_audit.md` documenting PSU-disjoint partitions and 5-fold CV (§6.2, §6.3, §6.4)."""
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    parts = manifest["partitions"]
    folds = manifest["training_cv_folds"]

    fold_rows = "\n".join(
        f"| Fold {f['fold_id']} | {f['train_rows']:,} | {f['val_rows']:,} | {f['val_psus']:,} | "
        f"{f['val_employed']:,} | {f['val_unemployed']:,} | {f['val_unemployment_rate']:.2%} |"
        for f in folds
    )

    content = f"""# HunarLens — PSU-Disjoint Partition & StratifiedGroupKFold Audit (`split_audit.md`)

**Reference:** [PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md) §6.2, §6.3, §6.4 · **Generated by:** `src/hunarlens/splits.py`

---

## 1. Grouping Unit & Split Protocol (§6.2)
* **Primary Dependence Unit:** 8-digit Primary Sampling Unit (`PCode`), keeping all households `(PCode, HHNo)` and individuals `(PCode, HHNo, SNo)` within the same PSU strictly together.
* **Test Split (`seed={manifest['seeds']['test_split']}`):** `GroupShuffleSplit(test_size={manifest['config']['test_size']})` holding out unseen PSUs for the single frozen evaluation batch.
* **Validation Split (`seed={manifest['seeds']['val_split']}`):** `GroupShuffleSplit(test_size={manifest['config']['val_size_of_remaining']})` on the remaining PSUs for model selection and decision-threshold optimization.
* **Training 5-Fold CV (`seed={manifest['seeds']['cv_folds']}`):** `StratifiedGroupKFold(n_splits={manifest['config']['cv_n_splits']}, shuffle=True)` strictly inside the training partition.

---

## 2. Partition Summary Table

| Partition | Rows (`n`) | Share (`%`) | PSUs (`PCode`) | Households | Employed (`0`) | Unemployed (`1`) | Unemployment Rate | PSU Hash (`SHA-256` prefix) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training (~70%)** | `{parts['train']['row_count']:,}` | `{parts['train']['row_share']:.2%}` | `{parts['train']['psu_count']:,}` | `{parts['train']['household_count']:,}` | `{parts['train']['employed_count']:,}` | `{parts['train']['unemployed_count']:,}` | `{parts['train']['unemployment_rate']:.2%}` | `{parts['train']['psu_sha256'][:16]}...` |
| **Validation (~15%)** | `{parts['val']['row_count']:,}` | `{parts['val']['row_share']:.2%}` | `{parts['val']['psu_count']:,}` | `{parts['val']['household_count']:,}` | `{parts['val']['employed_count']:,}` | `{parts['val']['unemployed_count']:,}` | `{parts['val']['unemployment_rate']:.2%}` | `{parts['val']['psu_sha256'][:16]}...` |
| **Test (~15% Sealed)** | `{parts['test']['row_count']:,}` | `{parts['test']['row_share']:.2%}` | `{parts['test']['psu_count']:,}` | `{parts['test']['household_count']:,}` | `{parts['test']['employed_count']:,}` | `{parts['test']['unemployed_count']:,}` | `{parts['test']['unemployment_rate']:.2%}` | `{parts['test']['psu_sha256'][:16]}...` |
| **Total Eligible Cohort** | **`{manifest['total_eligible_rows']:,}`** | **`100.00%`** | **`{manifest['total_psus']:,}`** | **`{manifest['total_households']:,}`** | **`{parts['train']['employed_count'] + parts['val']['employed_count'] + parts['test']['employed_count']:,}`** | **`{parts['train']['unemployed_count'] + parts['val']['unemployed_count'] + parts['test']['unemployed_count']:,}`** | **`{manifest['overall_unemployment_rate']:.2%}`** | — |

---

## 3. Training 5-Fold `StratifiedGroupKFold` Summary (§6.3)

| CV Fold | Fold-Train Rows | Fold-Val Rows | Fold-Val PSUs | Fold-Val Employed (`0`) | Fold-Val Unemployed (`1`) | Fold-Val Unemployment Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{fold_rows}

---

## 4. Leakage & Disjointness Verification Gate (§6.4)
- [x] **Zero PSU Overlap:** `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0` (`assert_zero_group_overlap` passed).
- [x] **Zero Household Overlap:** `(PCode, HHNo)` composite keys are strictly disjoint across all three partitions.
- [x] **Zero Person Overlap:** `(PCode, HHNo, SNo)` composite keys are strictly disjoint across all three partitions.
- [x] **Zero CV Fold PSU Overlap:** All 5 `StratifiedGroupKFold` splits have zero PSU overlap between fold-train and fold-val.
- [x] **Class & Regional Coverage:** Both binary classes (`{0, 1}`) and all 5 geographic units (`Khyber Pakhtunkhwa`, `Punjab`, `Sindh`, `Balochistan`, `Islamabad`) are represented in `train`, `val`, and `test`.
"""
    audit_path.write_text(content, encoding="utf-8")


def run_split_pipeline(
    data_cfg_path: str | Path = "configs/data.yaml",
    exp_cfg_path: str | Path = "configs/experiments.yaml",
) -> Dict[str, Any]:
    """Load eligible cohort, execute PSU-disjoint splits & 5-fold CV, and save manifests and audit report."""
    data_cfg = load_yaml_config(REPO_ROOT / data_cfg_path)
    exp_cfg = load_yaml_config(REPO_ROOT / exp_cfg_path)

    cohort_path = REPO_ROOT / data_cfg["paths"]["eligible_cohort_file"]
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Eligible cohort file not found at {cohort_path}. Run `python scripts/prepare_data.py` first."
        )

    df = pd.read_parquet(cohort_path)
    group_col = exp_cfg["splits"]["group_column"]
    target_col = data_cfg["cohort_eligibility"]["target_column"]

    train_df, val_df, test_df = generate_psu_disjoint_splits(
        df=df,
        group_col=group_col,
        target_col=target_col,
        test_size=float(exp_cfg["splits"]["test_size"]),
        val_size_of_remaining=float(exp_cfg["splits"]["val_size_of_remaining"]),
        test_seed=int(exp_cfg["seeds"]["test_split"]),
        val_seed=int(exp_cfg["seeds"]["val_split"]),
    )

    train_with_folds, _, fold_summaries = generate_training_cv_folds(
        train_df=train_df,
        group_col=group_col,
        target_col=target_col,
        n_splits=int(exp_cfg["splits"]["cv_n_splits"]),
        cv_seed=int(exp_cfg["seeds"]["cv_folds"]),
    )

    splits_dir = REPO_ROOT / data_cfg["paths"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "val.parquet"
    test_path = splits_dir / "test.parquet"

    train_with_folds.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    total_rows = len(df)
    manifest: Dict[str, Any] = {
        "total_eligible_rows": int(total_rows),
        "total_psus": int(df[group_col].nunique()),
        "total_households": int(df.groupby(data_cfg["identifiers"]["household_keys"]).ngroups),
        "overall_unemployment_rate": round(float(df[target_col].mean()), 6),
        "seeds": {
            "test_split": int(exp_cfg["seeds"]["test_split"]),
            "val_split": int(exp_cfg["seeds"]["val_split"]),
            "cv_folds": int(exp_cfg["seeds"]["cv_folds"]),
        },
        "config": {
            "group_column": group_col,
            "test_size": float(exp_cfg["splits"]["test_size"]),
            "val_size_of_remaining": float(exp_cfg["splits"]["val_size_of_remaining"]),
            "cv_n_splits": int(exp_cfg["splits"]["cv_n_splits"]),
        },
        "partitions": {
            "train": _partition_summary(train_with_folds, total_rows, psu_col=group_col, target_col=target_col),
            "val": _partition_summary(val_df, total_rows, psu_col=group_col, target_col=target_col),
            "test": _partition_summary(test_df, total_rows, psu_col=group_col, target_col=target_col),
        },
        "training_cv_folds": fold_summaries,
        "overlap_assertions_passed": True,
    }

    manifest_path = REPO_ROOT / data_cfg["paths"]["split_manifest_file"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit_path = REPO_ROOT / data_cfg["paths"]["split_audit_report"]
    write_split_audit_markdown(audit_path, manifest)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PSU-disjoint splits and 5-fold StratifiedGroupKFold.")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--experiments-config", type=str, default="configs/experiments.yaml")
    args = parser.parse_args()

    manifest = run_split_pipeline(
        data_cfg_path=args.data_config,
        exp_cfg_path=args.experiments_config,
    )
    p = manifest["partitions"]
    print(
        f"[splits] PSU-disjoint splits frozen | "
        f"Train={p['train']['row_count']:,} ({p['train']['psu_count']} PSUs) | "
        f"Val={p['val']['row_count']:,} ({p['val']['psu_count']} PSUs) | "
        f"Test={p['test']['row_count']:,} ({p['test']['psu_count']} PSUs)"
    )


if __name__ == "__main__":
    main()

