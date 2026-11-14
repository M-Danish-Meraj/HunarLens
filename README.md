# HunarLens — An Explainable Machine Learning System for Youth Unemployment Vulnerability and Skills Gap Analysis in Pakistan

**Course:** CT-378 Machine Learning · **Group:** AlphaDevs  
**Team Members:** Sarim Khan (`CT-24266`), Muhammad Saraib (`CT-24255`), Danish Meraj (`CT-24268`)  
**Detailed Implementation Plan:** [PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md) · **Live Progress Tracker:** [PROJECT_TRACKER.md](file:///d:/Hunarlens/PROJECT_TRACKER.md)

---

## 1. Project Overview & Intended Use
**HunarLens** is an explainable machine learning and descriptive analytics system built to investigate **survey-defined youth unemployment vulnerability (`is_unemployed`: `1 = unemployed`, `0 = employed`) among Pakistani youth aged 15–24 in the active labour force**.

* **Primary Users:** Student researchers, youth-employment analysts, and vocational program planners exploring regional/demographic patterns and model behaviour.
* **Methodological Guardrails (§1):** The classifier estimates a probability score associated with **current survey-defined unemployment conditional on labour-force membership (2024–25)**. It does **not** predict future job loss, determine official benefit/hiring eligibility, or establish causal effects of education or training.
* **Deployment Scope Exception ([docs/deployment_exception.md](file:///d:/Hunarlens/docs/deployment_exception.md)):** Per confirmed instructor approval, a single multi-view **Streamlit Community Cloud** application (`app/streamlit_app.py`) replaces the three-platform deployment listing in the general course handout.

---

## 2. Two-Source Public Data Provenance & 15-Feature Contract
HunarLens integrates two distinct official publications from the **Pakistan Bureau of Statistics (PBS)** via a verified **many-to-one (`validate="m:1"`)** geographic crosswalk ([data/README.md](file:///d:/Hunarlens/data/README.md), [data/source_manifest.csv](file:///d:/Hunarlens/data/source_manifest.csv)):

1. **PBS Labour Force Survey (LFS) 2024–25 Microdata (`12` Predictors):**
   * Individual demographics & literacy: `age` (`15–24`), `sex`, `marital_status`, `relationship_to_head`, `can_read`, `can_write`.
   * Human capital & enrolment: `highest_education` (non-ordinal categorical), `current_enrolment`.
   * Household roster composition (computed across all household ages before youth filtering): `household_size`, `household_dependency_ratio`.
   * Geography: `province_territory` (derived from the first digit of the 8-digit Primary Sampling Unit code `PCode` so Islamabad `6` is separated from Punjab `2`), `locality` (`Urban` / `Rural`).
2. **PBS 7th Population & Housing Census 2023 — Table 1 (`3` Regional Predictors):**
   * `regional_pop_density_log1p` ($\log(1 + \text{population density})$), `regional_urban_share`, and `regional_avg_household_size` ([data/reference/census_2023_regional.csv](file:///d:/Hunarlens/data/reference/census_2023_regional.csv)).

### Cohort Eligibility & Leakage Prevention ([docs/label_definition.md](file:///d:/Hunarlens/docs/label_definition.md), [docs/feature_review.csv](file:///d:/Hunarlens/docs/feature_review.csv))
* **19th ICLS Target (`is_unemployed`):** Youth aged `15–24` are classified in strict precedence order: Employed via work last week (`0`) $\rightarrow$ Employed via temporary absence (`0`) $\rightarrow$ Unemployed via active job search + current availability (`1`). Individuals outside the labour force or with unresolved responses are excluded without imputation.
* **Strict Leakage Denylist:** All target-defining questions (`worked_last_week`, `temp_absent_from_work`, `seeking_work`, `available_for_work`), post-outcome fields (`hours_worked`, `monthly_earnings`, `unemployment_duration`), administrative IDs (`PCode`, `HHNo`, `SNo`), survey design weights (`weight`), constant `Fstatus`, and job-timed vocational training codes (`vocational_training_status`) are blocked by `HunarLensSchemaAndCensusTransformer`.
* **PSU-Disjoint Splitting ([docs/split_audit.md](file:///d:/Hunarlens/docs/split_audit.md)):** Using `PCode` as the grouping unit (`25,584` eligible youth across `1,800` PSUs):
  * **Training (`70.02%`):** `17,913` rows (`1,260` PSUs) + 5-fold `StratifiedGroupKFold(seed=44)`
  * **Validation (`14.90%`):** `3,811` rows (`270` PSUs, `seed=43`)
  * **Sealed Test (`15.09%`):** `3,860` rows (`270` PSUs, `seed=42`)

---

## 3. Measured Experimental Results ([docs/final_evaluation_report.md](file:///d:/Hunarlens/docs/final_evaluation_report.md))

| Model | Train-CV AP (`mean ± SD`) | Frozen Val Threshold | Test F1 (`class 1`, 95% PSU Bootstrap CI) | Test Precision / Recall (`class 1`) | Test AP (95% CI) / ROC-AUC | Test Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Prior / Majority)** | `0.4324 ± 0.0005` | `0.50` | `0.0000` (`[0.0000, 0.0000]`) | `0.0000 / 0.0000` | `0.4448` (`[0.4263, 0.4640]`) / `0.5000` | `0.2471` |
| **Dummy Baseline (All-Positive)** | `0.4324 ± 0.0005` | `0.50` | `0.6157` (`[0.5978, 0.6339]`) | `0.4448 / 1.0000` | `0.4448` (`[0.4263, 0.4640]`) / `0.5000` | `0.5552` |
| **Logistic Regression** | `0.6332 ± 0.0107` | `0.34` | `0.6522` (`[0.6327, 0.6719]`) | `0.5435 / 0.8154` | `0.6466` (`[0.6219, 0.6748]`) / `0.7046` | `0.2162` |
| **Random Forest** | `0.6275 ± 0.0133` | `0.37` | `0.6475` (`[0.6280, 0.6674]`) | `0.5535 / 0.7800` | `0.6428` (`[0.6170, 0.6711]`) / `0.6985` | `0.2200` |
| **Gradient Boosting (Selected Winner)** | **`0.6338 ± 0.0128`** | **`0.37`** | **`0.6495`** (`[0.6307, 0.6701]`) | **`0.5582 / 0.7764`** | **`0.6511`** (`[0.6267, 0.6782]`) / **`0.7065`** | **`0.2165`** |

* **Paired PSU-Clustered Bootstrap Significance (`1,000` resamples over `270` test PSUs):**
  * **Winner vs. Dummy Prior ($\Delta F_1$):** `+0.6495` (`95% CI: [+0.6307, +0.6701]`)
  * **Winner vs. Dummy All-Positive ($\Delta F_1$):** `+0.0338` (`95% CI: [+0.0200, +0.0481]`)
  * **Winner vs. Dummy Prior ($\Delta \text{AP}$):** `+0.2063` (`95% CI: [+0.1843, +0.2289]`)

---

## 4. Reproducible Setup & Execution Commands

```bash
# 1. Install dependencies
pip install -r requirements-dev.txt

# 2. Verify or reconstruct raw PBS 2024-25 youth dataset & run readiness gate audit
python scripts/prepare_data.py

# 3. Generate PSU-disjoint 70/15/15 splits and 5-fold StratifiedGroupKFold assignments
python src/hunarlens/splits.py

# 4. Generate all 7 required training-only EDA figures -> reports/figures/
python scripts/run_eda.py

# 5. Train baselines + 3 algorithms with 5-fold grouped CV, run ablations, and freeze winner -> models/
python scripts/train_models.py

# 6. Run the one-time frozen test evaluation, 1,000-resample PSU bootstrap CIs, and subgroup audit
python scripts/evaluate_once.py

# 7. Run the complete automated test suite (labels, leakage, pipeline invariance, and Streamlit app parity)
python -m pytest tests/ -v

# 8. Launch the 5-view Streamlit dashboard locally
streamlit run app/streamlit_app.py
```

---

## 5. Repository Structure
* `configs/` — Declarative YAML files (`data.yaml`, `features.yaml`, `experiments.yaml`).
* `data/` — Provenance manifest (`source_manifest.csv`), Census 2023 lookup (`reference/`), and split manifest (`splits/split_manifest.json`).
* `src/hunarlens/` — Modular Python package (`data.py`, `labels.py`, `splits.py`, `features.py`, `train.py`, `explain.py`, `inference.py`, `evaluate.py`).
* `scripts/` — CLI runners (`download_data.py`, `prepare_data.py`, `run_eda.py`, `train_models.py`, `evaluate_once.py`).
* `app/streamlit_app.py` — 5-view Streamlit Community Cloud application entrypoint.
* `models/` — Frozen inference artifact (`final_pipeline.joblib`), `decision_threshold.json`, `feature_schema.json`, `model_metadata.json`, and `freeze_manifest.json`.
* `reports/` — 11 publication-grade PNG plots (`figures/`), JSON metrics (`metrics/`), [model_card.md](file:///d:/Hunarlens/reports/model_card.md), and [contribution_table.md](file:///d:/Hunarlens/reports/contribution_table.md).
* `tests/` — Automated pytest suite (`test_labels.py`, `test_leakage.py`, `test_pipeline.py`, `test_app.py`).
