"""CLI script to execute the one-time frozen test evaluation batch after model freeze.

Implements PROJECT_PLAN.md §10.4, §10.5, §11 & Phase 5, Step 5.3 of PROJECT_TRACKER.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hunarlens.evaluate import run_single_batch_test_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-batch frozen test evaluation, 1,000-resample PSU bootstrap CIs, and subgroup audit."
    )
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--experiments-config", type=str, default="configs/experiments.yaml")
    args = parser.parse_args()

    res = run_single_batch_test_evaluation(
        data_cfg_path=args.data_config,
        exp_cfg_path=args.experiments_config,
    )

    print(
        f"\n=== HunarLens One-Time Frozen Test Evaluation (n={res['test_row_count']:,}, "
        f"PSUs={res['test_psu_count']:,}, Prevalence={res['test_prevalence']:.2%}) ==="
    )
    boot = res["bootstrap_95ci"]["model_intervals"]
    for m_key, m_info in res["models"].items():
        tm = m_info["test_metrics_at_frozen_threshold"]
        f1_ci = boot[m_key]["f1_pos"]
        ap_ci = boot[m_key]["average_precision"]
        print(
            f"  - {m_info['display_name']:<42} | "
            f"Th={m_info['frozen_validation_threshold']:.2f} | "
            f"Test F1={tm['f1_pos']:.4f} [{f1_ci['ci95_low']:.4f}, {f1_ci['ci95_high']:.4f}] | "
            f"Test AP={tm['average_precision']:.4f} [{ap_ci['ci95_low']:.4f}, {ap_ci['ci95_high']:.4f}] | "
            f"ROC-AUC={tm['roc_auc']:.4f} | Brier={tm['brier_score']:.4f}"
        )

    diffs = res["bootstrap_95ci"]["winner_paired_differences"]
    print(
        f"\n[evaluate_once] Winner ({res['winner_display_name']}) vs. Dummy All-Positive Delta F1: "
        f"+{diffs['winner_minus_dummy_all_positive_f1']['mean']:.4f} "
        f"(95% CI: [{diffs['winner_minus_dummy_all_positive_f1']['ci95_low']:+.4f}, "
        f"{diffs['winner_minus_dummy_all_positive_f1']['ci95_high']:+.4f}])"
    )
    print("[evaluate_once] Saved reports/metrics/final_test_evaluation.json & docs/final_evaluation_report.md")


if __name__ == "__main__":
    main()

