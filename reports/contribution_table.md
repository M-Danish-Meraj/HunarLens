# HunarLens — Individual Team Contributions & Viva Readiness (`contribution_table.md`)

**Course:** CT-378 Machine Learning · **Group:** AlphaDevs  
**Team:** Danish Meraj (`CT-24268`), Sarim Khan (`CT-24266`), Muhammad Saraib (`CT-24255`)

---

## 1. Individual Contribution Matrix (§15)

| Team Member | Roll Number | Primary Modules & Code Ownership | Cross-Review & Demonstration Role |
| :--- | :---: | :--- | :--- |
| **Danish Meraj** | `CT-24268` | **System Architecture, Feature Pipeline, Model Training, Explainability, Inference Engine & Streamlit Application:**<br>- `src/hunarlens/features.py` (`HunarLensSchemaAndCensusTransformer`, fold-local `ColumnTransformer` & `Pipeline`)<br>- `src/hunarlens/train.py`, `scripts/train_models.py` (5-fold `StratifiedGroupKFold` search for Logistic Regression, Random Forest, Gradient Boosting, ablations, validation threshold optimization & artifact freeze)<br>- `src/hunarlens/explain.py`, `src/hunarlens/inference.py` (Global validation permutation importance, local counterfactual explanations & `HunarLensPredictor` API)<br>- `app/streamlit_app.py` (All 5 interactive dashboard views & deployment integration)<br>- `configs/experiments.yaml`, `models/final_pipeline.joblib`, `models/decision_threshold.json`, `models/feature_schema.json`, `models/model_metadata.json`<br>- `tests/test_pipeline.py`, `tests/test_app.py`, `notebooks/03_model_comparison.ipynb`, `notebooks/04_explanations.ipynb`<br>- `README.md`, `reports/model_card.md`, `docs/deployment_exception.md` | Leads **3:00–7:00** live Streamlit demonstration and core architectural walk-through; reviews end-to-end inference parity, model selection, threshold optimization, and explainability lifecycle. |
| **Sarim Khan** | `CT-24266` | **Data Provenance, Census 2023 Crosswalk & 19th ICLS Target Construction:**<br>- `configs/data.yaml`, `configs/features.yaml`<br>- `src/hunarlens/data.py`, `src/hunarlens/labels.py`<br>- `scripts/download_data.py`, `scripts/prepare_data.py`<br>- `docs/label_definition.md`, `docs/feature_review.csv`, `docs/data_audit.md`<br>- `tests/test_labels.py`, `notebooks/01_data_audit.ipynb` | Audited deployment feature schema (`feature_schema.json`); leads **0:00–3:00** of the 10-minute demo covering problem motivation, two-source PBS/Census data integration, 19th ICLS target definition, and leakage denylist. |
| **Muhammad Saraib** | `CT-24255` | **PSU-Disjoint Splitting, Training-Only EDA & Frozen Test Evaluation:**<br>- `src/hunarlens/splits.py`, `src/hunarlens/evaluate.py`<br>- `scripts/run_eda.py`, `scripts/evaluate_once.py`<br>- `docs/split_audit.md`, `docs/eda_interpretations.md`, `docs/final_evaluation_report.md`<br>- `tests/test_leakage.py`, `notebooks/02_training_eda.ipynb` | Verified target label unit tests; leads **7:00–10:00** of the demo covering PSU-disjoint partitioning, EDA highlights, 1,000-resample PSU bootstrap confidence intervals, and subgroup audits. |

---

## 2. Core Viva Questions & Shared Team Readiness (§15)
All three team members have cross-reviewed the full codebase and are prepared to answer:
1. **Why did the test set stay sealed until `evaluate_once.py`?** To prevent iterative test-set tuning from inflating generalization estimates; `freeze_manifest.json` locks SHA-256 hashes of splits, pipelines, and thresholds before test evaluation.
2. **Why were `worked_last_week`, `seeking_work`, `Fstatus`, and `vocational_training_status` excluded from the 15 predictors?** The first two directly define the 19th ICLS target (`is_unemployed`), `Fstatus` is an unlabelled constant in the raw extract, and `vocational_training_status` distinguishes *on-the-job* training (which occurs only among employed workers).
3. **Why group by 8-digit `PCode` (PSU) instead of random row splitting?** Respondents in the same household `(PCode, HHNo)` and neighbourhood PSU share household size, dependency ratio, and local labour-market shocks; random row splitting leaks household/PSU context across train and test.
4. **How does Positive-Class $F_1$ differ from Average Precision ($\text{AP}$), and why is the threshold `0.37` instead of `0.50`?** $\text{AP}$ summarizes ranking quality across all probability cutoffs (`0.6511` vs. `0.4448` prevalence), while $F_1$ measures precision/recall balance at a single operating threshold. On validation probabilities, `0.37` maximized class-1 $F_1$ (`0.6358` validation, `0.6495` frozen test).
