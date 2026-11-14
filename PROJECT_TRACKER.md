# HunarLens — Project Execution Tracker (`PROJECT_TRACKER.md`)

**Project:** An Explainable Machine Learning System for Youth Unemployment Vulnerability and Skills Gap Analysis in Pakistan  
**Course:** CT-378 Machine Learning · **Group:** AlphaDevs  
**Team:** Sarim Khan (CT-24266), Muhammad Saraib (CT-24255), Danish Meraj (CT-24268)  
**Source Plan:** [PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md)

---

## Collaboration & Workflow Rules
1. **Isolated Modules:** Each sub-task touches decoupled files (`configs/`, `src/hunarlens/`, `scripts/`, `app/`, `tests/`, `docs/`) to prevent merge conflicts.
2. **Real-Time Tracking:** Immediately upon completing a sub-task, check off (`- [x]`) the item below and append a concise completion note.
3. **Atomic Commits:** Commit each completed sub-task together with its `PROJECT_TRACKER.md` update so any teammate pulling the branch gets a clean, functional state.

---

## Phase 1: Repository Scaffolding, Configuration & Data Governance
*Objective: Establish the modular directory hierarchy, dependency pins, YAML configurations, and data provenance documentation (§4, §5, §13).*

- [x] **Step 1.1 — Repository Layout, `.gitignore`, & Base Dependencies** *(Completed: Initialized Git repository, created full §13 modular directory tree with `.gitkeep` anchors, added privacy-enforcing `.gitignore`, pinned `requirements.txt` & `requirements-dev.txt`, and initialized `src/hunarlens/__init__.py`.)*
  - Create standard directory tree (`configs/`, `data/{raw,reference,processed,splits}`, `notebooks/`, `src/hunarlens/`, `scripts/`, `app/`, `models/`, `reports/{figures,metrics}`, `tests/`, `docs/`).
  - Add `.gitignore` protecting private microdata (`data/raw/`, `data/processed/`, `data/splits/*.parquet`) while tracking manifests and reference tables.
  - Create `requirements.txt`, `requirements-dev.txt`, and `src/hunarlens/__init__.py`.
- [x] **Step 1.2 — Declarative YAML Configurations (`configs/`)** *(Completed: Created `configs/data.yaml`, `configs/features.yaml`, and `configs/experiments.yaml` encoding PSU/household/person keys, the 15-feature contract, explicit leakage denylist, 70/15/15 split seeds 42/43/44, 5-fold StratifiedGroupKFold search spaces, ablations, and validation threshold policy.)*
  - Create `configs/data.yaml` (LFS 2024–25 & Census 2023 paths, PSU/household/person keys, age 15–24 filter).
  - Create `configs/features.yaml` (15 approved predictors, explicit forbidden/leakage denylist, data types, transformations).
  - Create `configs/experiments.yaml` (seeds 42/43/44, 70/15/15 split ratios, 5-fold `StratifiedGroupKFold`, hyperparameter grids for Dummy, Logistic Regression, Random Forest, and Gradient Boosting, threshold policy).
- [x] **Step 1.3 — Data Provenance, Manifest & Governance Docs (`data/`, `docs/`)** *(Completed: Created `data/README.md`, `data/source_manifest.csv`, `docs/label_definition.md` with 19th ICLS decision rules & worked examples, `docs/feature_review.csv` auditing all 15 accepted predictors and 18 rejected leakage/identifier fields, and `docs/deployment_exception.md`.)*
  - Create `data/README.md` and `data/source_manifest.csv` documenting PBS LFS 2024–25 and Census 2023 sources, access terms, and checksums.
  - Create `docs/label_definition.md` (19th ICLS binary unemployment rules for youth 15–24, exclusion rules for outside-labour-force).
  - Create `docs/feature_review.csv` (accepted 15-feature contract and rejected leakage/proxy register) and `docs/deployment_exception.md`.

---

## Phase 2: Data Ingestion, Census Crosswalk & Target Construction
*Objective: Implement modular source loading, household roster derivation, Census regional join, and strict binary label construction (§4, §5).*

- [x] **Step 2.1 — Census Reference Table & Geographic Crosswalk (`data/reference/`, `src/hunarlens/data.py`)** *(Completed: Created `data/reference/census_2023_regional.csv` with PBS Census 2023 Table-1 indicators for all 5 regions and implemented `src/hunarlens/data.py` with identifier normalization/deduplication, `PCode` digit-1 province resolution, pre-filter household roster aggregation, and verified `many-to-one` Census join assertions.)*
  - Create verified province/territory & regional Census 2023 lookup table (`data/reference/census_2023_regional.csv`).
  - Implement `src/hunarlens/data.py` with schema validation, identifier recovery (`PCode`, `HHNo`, `SNo`), household roster aggregation (size and age-dependency ratio before youth filtering), and many-to-one Census join assertions.
- [x] **Step 2.2 — Target Label Construction & Eligibility Module (`src/hunarlens/labels.py`)** *(Completed: Implemented `derive_19th_icls_status`, `build_eligible_youth_cohort`, `assert_readiness_gate`, and `LabelAuditSummary` in `src/hunarlens/labels.py` enforcing 19th ICLS precedence rules, student retention, `Fstatus` & outcome-input stripping, and the >=5,000 rows / 15 predictors readiness gate.)*
  - Implement `src/hunarlens/labels.py` encoding the 19th ICLS youth (15–24) labour-force eligibility filter and `is_unemployed` binary target (`1 = unemployed`, `0 = employed`), with explicit exclusion reason logging.
- [x] **Step 2.3 — Data Acquisition & Preparation CLI Scripts (`scripts/download_data.py`, `scripts/prepare_data.py`)** *(Completed: Implemented `scripts/download_data.py` with SHA-256 verification and deterministic 600-PSU multi-household LFS 2024–25 reconstruction (`seed=202425`), and `scripts/prepare_data.py` orchestrating identifier deduplication, `PCode` digit-1 province mapping, pre-filter household roster math, verified `m:1` Census 2023 join, 19th ICLS target derivation, descriptive skills separation, and `docs/data_audit.md` generation.)*
  - Implement `scripts/download_data.py` (verifies raw PBS files or generates a deterministic schema-faithful synthetic/sample dataset for CI & offline reproducibility when raw microdata is absent) and `scripts/prepare_data.py` producing `docs/data_audit.md`.
- [x] **Step 2.4 — Unit Tests for Labels & Data Integrity (`tests/test_labels.py`)** *(Completed: Calibrated against official PBS LFS 2024–25 empirical benchmarks (`data/reference/pbs_lfs_2024_25_empirical_benchmarks.csv`) and executed `scripts/prepare_data.py` producing 25,584 eligible youth records across 1,800 PSUs with zero irrelevant non-youth rows; ran `tests/test_labels.py` — 6/6 unit tests passing.)*
  - Verify cohort filtering, household roster math, many-to-one Census merge row-count invariance, and label precedence rules.

---

## Phase 3: Leakage-Free Grouped Splitting & Preprocessing Pipeline
*Objective: Enforce PSU-disjoint partitioning and fold-local scikit-learn transformations (§6, §8).*

- [x] **Step 3.1 — PSU-Disjoint Split & Fold Generator (`src/hunarlens/splits.py`)** *(Completed: Implemented and executed `src/hunarlens/splits.py`, freezing PSU-disjoint partitions on the 25,584 eligible youth cohort: Train=17,913 rows [1,260 PSUs, 70.02%], Val=3,811 rows [270 PSUs, 14.90%], Test=3,860 rows [270 PSUs, 15.09%] plus 5-fold StratifiedGroupKFold inside training; verified zero PSU/household/person overlap and saved `data/splits/split_manifest.json` and `docs/split_audit.md`.)*
  - Implement `GroupShuffleSplit` (70/15/15 train/val/test with seeds 42 and 43) and 5-fold `StratifiedGroupKFold` (seed 44) inside training.
  - Enforce zero PSU/household overlap assertions and export `docs/split_audit.md`.
- [x] **Step 3.2 — Custom Transformers & Scikit-Learn Pipeline (`src/hunarlens/features.py`)** *(Completed: Implemented `HunarLensSchemaAndCensusTransformer`, `build_preprocessing_pipeline`, and `build_estimator_pipeline` in `src/hunarlens/features.py` with automatic Census 2023 lookup from `province_territory`, household age-band ratio derivation, domain bounds validation, strict 15-feature allowlist filtering, fold-local median imputation + `StandardScaler`, and `"Unknown"` + `OneHotEncoder(handle_unknown="ignore")`.)*
  - Implement importable scikit-learn transformers: schema/codebook validator, Census lookup transformer, household dependency ratio & `log1p` density engineer, and `ColumnTransformer` (fold-local median imputation + `StandardScaler` for linear models; explicit missing category + `OneHotEncoder(handle_unknown="ignore")` for categoricals).
- [x] **Step 3.3 — Leakage & Pipeline Verification Tests (`tests/test_leakage.py`, `tests/test_pipeline.py`)** *(Completed: Implemented and executed `tests/test_leakage.py` and `tests/test_pipeline.py` verifying zero forbidden feature intersection, zero PSU/household/person overlap across partitions & 5 CV folds, label-shuffle sanity check, held-out mutation invariance of fitted parameters, single-profile Census/household derivation, unseen category handling, and `joblib` parity — 14/14 total tests passing.)*
  - Verify zero overlap with forbidden predictors, group-disjoint partition invariants, and invariance of fitted training parameters to held-out data mutations.

---

## Phase 4: Exploratory Data Analysis (Training Partition Only)
*Objective: Generate the 7 required development plots and descriptive summaries strictly on the 70% training split (§7).*

- [ ] **Step 4.1 — Training-Only EDA Generator & Figures (`scripts/run_eda.py`, `notebooks/01_data_audit.ipynb`, `notebooks/02_training_eda.ipynb`)**
  - Generate and save all 7 required plots to `reports/figures/` (class balance, feature missingness, numeric distributions, boxplots/quantiles, correlation heatmap, education/enrolment vs. unemployment, province/locality/sex & training summaries with small-cell `<30` suppression).

---

## Phase 5: Model Training, 5-Fold Search, Threshold Selection & One-Time Test Evaluation
*Objective: Train baselines and 3 algorithms with grouped CV, select optimal decision thresholds on validation, freeze artifacts, and run single-batch test evaluation (§9, §10, §11).*

- [ ] **Step 5.1 — Model Training, Grouped CV & Ablation Module (`src/hunarlens/train.py`, `scripts/train_models.py`)**
  - Implement training and hyperparameter search for `DummyClassifier` (`prior` and `constant=1`), `LogisticRegression` (GridSearchCV), `RandomForestClassifier` (RandomizedSearchCV), and `GradientBoostingClassifier` (RandomizedSearchCV, `n_iter_no_change=None`).
  - Run preplanned training-CV ablations (demographics-only vs. +Census; with vs. without sex/gender) and validation threshold selection (grid 0.01–0.99 maximizing class-1 F1).
  - Serialize frozen winner and metadata to `models/` (`final_pipeline.joblib`, `model_metadata.json`, `decision_threshold.json`, `feature_schema.json`).
- [ ] **Step 5.2 — Explainability & Inference Wrapper (`src/hunarlens/explain.py`, `src/hunarlens/inference.py`)**
  - Implement grouped validation permutation importance (on raw conceptual features), local feature contribution breakdown, and validated single-profile inference API.
- [ ] **Step 5.3 — Frozen Single-Batch Test Evaluation (`src/hunarlens/evaluate.py`, `scripts/evaluate_once.py`, `notebooks/03_model_comparison.ipynb`, `notebooks/04_explanations.ipynb`)**
  - Verify freeze manifest, evaluate all frozen models once on the 15% test split, compute 1,000-resample PSU-clustered bootstrap 95% CIs for class-1 F1 and AP, subgroup metrics with `<30` suppression, reliability/PR/confusion plots, and export `reports/metrics/`.

---

## Phase 6: Streamlit Application, Testing & Final Deliverables
*Objective: Build the 5-view Streamlit app, verify end-to-end tests, and compile final documentation and report artifacts (§12, §16).*

- [ ] **Step 6.1 — Streamlit Multi-View Dashboard (`app/streamlit_app.py`)**
  - Implement all 5 required views: (1) Overview, (2) Profile Explorer (validated form, probability, saved threshold decision, local/global explanation chart), (3) Population Explorer, (4) Skills & Education Analysis, and (5) Model Evidence & Leakage Protocol.
- [ ] **Step 6.2 — Application & End-to-End Tests (`tests/test_app.py`)**
  - Test artifact loading, input validation rules, unseen category handling, Census auto-lookup, and batch-vs-app prediction parity.
- [ ] **Step 6.3 — Documentation, Model Card & Final Report Artifacts (`README.md`, `reports/model_card.md`, `reports/contribution_table.md`)**
  - Finalize `README.md`, `reports/model_card.md`, and `reports/contribution_table.md` with complete reproducibility instructions and measured results.
