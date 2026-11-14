# HunarLens — Individual Team Contributions & Viva Readiness (`contribution_table.md`)

**Course:** CT-378 Machine Learning · **Group:** AlphaDevs

---

## 1. Individual Contribution Matrix (§15)

| Team Member | Roll Number | Primary Modules & Code Ownership | Cross-Review & Demonstration Role |
| :--- | :---: | :--- | :--- |
| **Sarim Khan** | `CT-24266` | **Data Ingestion, Provenance, Census Crosswalk & Target Construction:**<br>- `configs/data.yaml`, `configs/features.yaml`<br>- `src/hunarlens/data.py`, `src/hunarlens/labels.py`<br>- `scripts/download_data.py`, `scripts/prepare_data.py`<br>- `docs/label_definition.md`, `docs/feature_review.csv`, `docs/data_audit.md`<br>- `tests/test_labels.py`, `notebooks/01_data_audit.ipynb` | Audited deployment feature schema (`feature_schema.json`); leads **0:00–3:00** of the 10-minute demo covering problem motivation, two-source PBS/Census data integration, 19th ICLS target definition, and leakage denylist. |
| **Muhammad Saraib** | `CT-24255` | **PSU-Disjoint Splitting, Pipelines, 5-Fold Search & Frozen Evaluation:**<br>- `configs/experiments.yaml`<br>- `src/hunarlens/splits.py`, `src/hunarlens/features.py`<br>- `src/hunarlens/train.py`, `src/hunarlens/evaluate.py`<br>- `scripts/run_eda.py`, `scripts/train_models.py`, `scripts/evaluate_once.py`<br>- `tests/test_leakage.py`, `tests/test_pipeline.py`<br>- `notebooks/02_training_eda.ipynb`, `notebooks/03_model_comparison.ipynb` | Verified target label unit tests; leads **7:00–10:00** of the demo covering model comparison, positive-class $F_1$ vs. $\text{AP}$, validation threshold selection (`0.37`), 1,000-resample PSU bootstrap CIs, and subgroup audits. |
| **Danish Meraj** | `CT-24268` | **Explainability, Validated Inference API, Streamlit Dashboard & App Testing:**<br>- `src/hunarlens/explain.py`, `src/hunarlens/inference.py`<br>- `app/streamlit_app.py` (all 5 interactive views)<br>- `tests/test_app.py`, `notebooks/04_explanations.ipynb`<br>- `README.md`, `reports/model_card.md`, `docs/deployment_exception.md` | Verified batch-vs-app prediction parity; leads **3:00–7:00** live Streamlit demonstration covering validated profile scoring, auto Census lookup, local/global explanations, and descriptive skills explorer. |

---

## 2. Core Viva Questions & Shared Team Readiness (§15)
All three team members have cross-reviewed the full codebase and are prepared to answer:
1. **Why did the test set stay sealed until `evaluate_once.py`?** To prevent iterative test-set tuning from inflating generalization estimates; `freeze_manifest.json` locks SHA-256 hashes of splits, pipelines, and thresholds before test evaluation.
2. **Why were `worked_last_week`, `seeking_work`, `Fstatus`, and `vocational_training_status` excluded from the 15 predictors?** The first two directly define the 19th ICLS target (`is_unemployed`), `Fstatus` is an unlabelled constant in the raw extract, and `vocational_training_status` distinguishes *on-the-job* training (which occurs only among employed workers).
3. **Why group by 8-digit `PCode` (PSU) instead of random row splitting?** Respondents in the same household `(PCode, HHNo)` and neighbourhood PSU share household size, dependency ratio, and local labour-market shocks; random row splitting leaks household/PSU context across train and test.
4. **How does Positive-Class $F_1$ differ from Average Precision ($\text{AP}$), and why is the threshold `0.37` instead of `0.50`?** $\text{AP}$ summarizes ranking quality across all probability cutoffs (`0.6511` vs. `0.4448` prevalence), while $F_1$ measures precision/recall balance at a single operating threshold. On validation probabilities, `0.37` maximized class-1 $F_1$ (`0.6358` validation, `0.6495` frozen test).
