# HunarLens — Model Card (`model_card.md`)

## 1. Model Details
* **Project:** HunarLens — Youth Unemployment Vulnerability & Skills Gap Analysis in Pakistan
* **Course & Group:** CT-378 Machine Learning · Group AlphaDevs (Danish Meraj `CT-24268`, Sarim Khan `CT-24266`, Muhammad Saraib `CT-24255`)
* **Selected & Frozen Estimator:** `GradientBoostingClassifier` wrapped in an end-to-end scikit-learn `Pipeline` (`models/final_pipeline.joblib`) with `HunarLensSchemaAndCensusTransformer` and fold-local `ColumnTransformer`.
* **Selected Hyperparameters:** `n_estimators=100`, `learning_rate=0.05`, `max_depth=3`, `min_samples_leaf=20`, `subsample=0.8`, `n_iter_no_change=None`.
* **Validation-Selected Decision Threshold:** `0.37` (selected on the 15% validation split to maximize positive-class $F_1$).

## 2. Intended Use & Out-of-Scope Uses
* **Intended Use:** Educational, research, and policy-planning exploration of demographic, educational, household, and regional factors associated with survey-defined unemployment (`19th ICLS`) among Pakistani youth aged 15–24 in the labour force.
* **Out-of-Scope / Forbidden Uses:**
  * Automated candidate screening, hiring, loan/credit scoring, or social-benefit eligibility decisions.
  * Causal claims that changing a demographic or educational attribute will guarantee employment.
  * Forecasting live 2026+ macroeconomic unemployment rates from 2024–25 cross-sectional data.

## 3. Training, Validation & Sealed Test Data
* **Primary Sources:** PBS Labour Force Survey 2024–25 + PBS 7th Population & Housing Census 2023 (Table 1).
* **Total Eligible Youth Cohort:** `25,584` youth aged 15–24 in the active labour force across `1,800` Primary Sampling Units (`PCode`).
  * **Training Split (`70.02%`):** `17,913` rows (`1,260` PSUs; 5-fold `StratifiedGroupKFold`)
  * **Validation Split (`14.90%`):** `3,811` rows (`270` PSUs)
  * **Sealed Test Split (`15.09%`):** `3,860` rows (`270` PSUs, evaluated once after `freeze_manifest.json`)

## 4. Measured Performance on Sealed Test Set (`n = 3,860`)
* **Headline Metric — Positive-Class (`Unemployed = 1`) $F_1$:** **`0.6495`** (`95% PSU Bootstrap CI: [0.6307, 0.6701]`)
* **Precision / Recall (`class 1`):** `0.5582` / `0.7764`
* **Average Precision (`AP`):** **`0.6511`** (`95% CI: [0.6267, 0.6782]`, `1.46x` lift over `0.4448` test prevalence)
* **ROC-AUC:** `0.7065`
* **Brier Score:** `0.2165` (vs. `0.2471` for constant prevalence predictor)
* **Survey-Weighted Test Sensitivity:** Weighted $F_1 = 0.6479$, Weighted $\text{AP} = 0.6524$, Weighted $\text{ROC-AUC} = 0.7065$.

## 5. Limitations & Caveats
1. **Cross-Sectional Observational Data:** Reflects associations during the 2024–25 survey period; cannot establish causal treatment effects of vocational training or formal degrees.
2. **Regional Granularity:** Because district boundaries are not uniquely verified in public LFS extracts, Census 2023 linkage is performed at the verified 5-unit `province_territory` level (`PCode` digit 1).
3. **Skills Demand Side:** Employer job vacancies and competency demands are not observed in household surveys; the Skills & Education module reports descriptive participation and employment associations only.

