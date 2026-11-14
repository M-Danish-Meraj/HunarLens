"""HunarLens — Explainable ML System for Youth Unemployment Vulnerability & Skills Gap Analysis in Pakistan.

Course: CT-378 Machine Learning
Group: AlphaDevs (Sarim Khan CT-24266, Muhammad Saraib CT-24255, Danish Meraj CT-24268)
"""

from __future__ import annotations

import sys
import types

__version__ = "0.1.0"
__all__ = ["__version__", "ensure_sklearn_ensemble_compat"]


def ensure_sklearn_ensemble_compat() -> None:
    """Guard against Windows Smart App Control blocking the unused `_hist_gradient_boosting` Cython extension
    when importing `RandomForestClassifier` and `GradientBoostingClassifier` from `sklearn.ensemble`.
    """
    target_mod = "sklearn.ensemble._hist_gradient_boosting.gradient_boosting"
    if target_mod in sys.modules:
        return
    try:
        __import__(target_mod)
    except ImportError:
        stub = types.ModuleType(target_mod)
        stub.HistGradientBoostingClassifier = object  # type: ignore[attr-defined]
        stub.HistGradientBoostingRegressor = object  # type: ignore[attr-defined]
        sys.modules[target_mod] = stub


ensure_sklearn_ensemble_compat()
