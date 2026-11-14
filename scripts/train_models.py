"""CLI script to execute 5-fold grouped hyperparameter search, ablations, validation threshold selection, and freeze.

Implements PROJECT_PLAN.md §9, §10.2, §13 & Phase 5, Step 5.1 of PROJECT_TRACKER.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hunarlens.train import run_training_and_selection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baselines and 3 ML algorithms with 5-fold StratifiedGroupKFold, run ablations, and freeze winner."
    )
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--features-config", type=str, default="configs/features.yaml")
    parser.add_argument("--experiments-config", type=str, default="configs/experiments.yaml")
    args = parser.parse_args()

    results = run_training_and_selection(
        data_cfg_path=args.data_config,
        features_cfg_path=args.features_config,
        exp_cfg_path=args.experiments_config,
    )

    print("\n=== HunarLens Training-CV & Validation Model Comparison ===")
    for m_key, m_info in results["models"].items():
        cv_ap = m_info["cv_summary"]["cv_metrics"]["average_precision"]
        th_info = m_info["validation_threshold_selection"]
        val_m = th_info["metrics_at_selected_threshold"]
        print(
            f"  - {m_info['display_name']:<42} | "
            f"Train-CV AP: {cv_ap['cv_mean']:.4f} ± {cv_ap['cv_std']:.4f} | "
            f"Val Threshold: {th_info['selected_threshold']:.2f} | "
            f"Val F1(pos): {val_m['f1_pos']:.4f} | Val AP: {val_m['average_precision']:.4f}"
        )

    print("\n=== Preplanned Training-CV Ablations (Gradient Boosting) ===")
    for abl in results["ablations"]:
        print(
            f"  - {abl['ablation_name']:<34} ({abl['feature_count']:2d} features) | "
            f"CV AP: {abl['cv_ap_mean']:.4f} ± {abl['cv_ap_std']:.4f} | "
            f"CV ROC-AUC: {abl['cv_roc_auc_mean']:.4f}"
        )

    print(
        f"\n[train_models] FROZEN DEPLOYMENT WINNER: {results['winner_display_name']} "
        f"(key='{results['winner_model_key']}', validation threshold={results['winner_threshold']:.2f})"
    )


if __name__ == "__main__":
    main()

