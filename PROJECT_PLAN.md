# HunarLens — Complete Project Plan

**An Explainable Machine Learning System for Youth Unemployment Vulnerability and Skills Gap Analysis in Pakistan**

**Course:** CT-378 Machine Learning · **Group:** AlphaDevs · **Prepared:** 13 November 2026

**Team:** Danish Meraj (CT-24268), Sarim Khan (CT-24266), Muhammad Saraib (CT-24255)

**Planning basis:** All three pages of `AlphaDevs.pdf`, both pages of `CT-378 ML Project Guidelines (1).pdf`, and the user's confirmed instructor approval for Streamlit-only deployment.

**Status:** Implementation plan, not an experimental-results report. The prepared dataset and trained models described in the proposal were not supplied. Original PBS source files were downloaded and audited after this plan was drafted; see `reports/audit/data_intake_report.md`. The survey contains 59,491 people aged 15–24 before labour-force eligibility is applied. The final eligible row count and 15 defensible predictors remain unverified. All performance targets below are planning goals, not measured or guaranteed results.

**Data-intake update:** The source's unlabelled `Fstatus` is constant and cannot be the target. The eight-digit `PCode` identifies a PSU, while `(PCode, HHNo)` is the observed household candidate key; do not assume `PCode` alone identifies a household or that `HHNo` fits two digits. A person key `(PCode, HHNo, SNo)` has no duplicates in the downloaded file. The source `Province` field folds Islamabad into Punjab, whereas the first digit of `PCode` distinguishes it. No unambiguous current-district field has yet been verified, so the initial Census reference uses five province/territory units. Publication timing of the exact Census inputs still requires verification before making any historical prediction-time claim. The label rules and grouped split remain unapproved until the data audit's open issues are resolved.

## 1. Project decisions

| Decision | Adopted approach |
| --- | --- |
| Task | Binary classification of current unemployment among eligible Pakistani youth aged 15–24 in the labour force. |
| Target | `is_unemployed`: 1 = unemployed; 0 = employed. Exclude people outside the labour force and records whose status cannot be resolved. |
| Primary user | A student researcher, career-support analyst or youth-program planner exploring population patterns and model behaviour. |
| Data | PBS Labour Force Survey 2024–25 microdata combined with independent Census 2023 regional context. |
| Main metric | Positive-class F1, with unemployment as the positive class, at a threshold chosen using validation data. |
| Supporting metric | Average Precision (AP) for ranking under class imbalance; also precision, recall, ROC-AUC, confusion matrix and probability diagnostics. |
| Models | Dummy baselines, Logistic Regression, Random Forest and Gradient Boosting. |
| Validation | Approximately 70/15/15 train/validation/test; five-fold grouped CV inside training only. Prefer PSU-disjoint partitions, which also keep households together. |
| Deployment | One public Streamlit Community Cloud application. |
| Explanation | Global feature importance plus accurate model-specific local explanations where feasible. |
| Skills component | Descriptive education/training coverage and employment associations. Actual employer-demand mismatch cannot be established from these two sources alone. |

The app estimates a model score associated with **current survey-defined unemployment conditional on labour-force membership**. It does not predict future job loss, determine official employment status, or establish that education or training causes employment. A cross-sectional survey cannot validate those stronger claims.

The guideline's overview, README requirement and marking table mention three deployments, while its deployment heading says to choose one. Apply the instructor-approved exception consistently: Streamlit replaces all three-platform requirements. Record the approval already obtained in the README and report; no new approval is assumed necessary. Do not invent a revised marking allocation.

## 2. Scope and deliverables

The finished project will provide a reproducible data-preparation workflow, a reviewed target definition, at least 5,000 valid eligible observations and 15 legitimate predictors, training-only EDA, a model comparison, a frozen test evaluation, explanations, and a working Streamlit dashboard.

The dashboard will support an illustrative profile form, model probability and decision, a chart explaining the model, population-pattern exploration, descriptive skills/training analysis, and a methodology/results page. It will identify the 2024–25 survey period and actual geographic coverage. It will not present historical data as live 2026 conditions.

The submission consists of a GitHub repository, final `joblib` inference artifact, pinned dependencies, a public Streamlit URL, a 3–4 page PDF report, presentation/demo material and a truthful individual contribution table. This longer planning document supports implementation and is separate from the required short final report.

Exclude job recommendations, automated hiring or benefit eligibility, causal training recommendations, forecasts, a separate API, Hugging Face deployment and a Flask/FastAPI application from the core scope. These do not advance the approved semester deliverables.

## 3. Complete guideline coverage

| Guideline requirement | Planned implementation and submission evidence |
| --- | --- |
| Real problem and clear user | Youth unemployment analysis for researchers and youth-program analysts; scope in Sections 1–2. |
| Unique approved topic | Keep the approved HunarLens proposal and topic record; do not claim duplicate-topic clearance beyond the recorded approval. |
| Group of 2–3; all contribute code | Three named members, module ownership, commits and cross-review; Section 15. |
| Proposal in Week 12 | Preserve the supplied proposal; retain a half-page summary covering problem, data, target and metric if required for the original milestone. |
| Final demo/report/viva in Week 15 | Work backwards from the course's actual Week 15 deadline; Section 14. No calendar deadline has been supplied. |
| No prohibited or previously used dataset | Use PBS sources; confirm course-specific prior-use restrictions. Do not substitute Iris, Titanic, MNIST or Boston Housing. |
| Collect data or combine two public sources | Combine LFS microdata with Census regional tables through a verified geographic crosswalk. |
| Minimum 5,000 rows and 15 features | Check after eligibility filtering, label audit and leakage removal, before encoding; Section 4. |
| Sources, collection, licence and date | Source manifest, download/reconstruction instructions, checksums and access/coverage dates. |
| Remove personal identifiers | Remove names, contact details, addresses and exact coordinates; retain grouping keys privately, outside predictors. |
| EDA: distributions, missingness, outliers, correlations, class balance; at least five plots | Seven planned development plots with concise interpretations; Section 7. |
| Imputation, encoding, scaling and feature engineering in training-fitted Pipeline | Complete scikit-learn inference pipeline, including custom transformations; Section 8. |
| Train/validation/test | Saved grouped partition manifest and overlap assertions; Section 6. |
| Baseline plus at least three course algorithms | Dummy, Logistic Regression, Random Forest, Gradient Boosting; Section 9. |
| Five-fold GridSearchCV or RandomizedSearchCV | Identical saved group-aware folds for all algorithms; search configuration and `cv_results_` saved. |
| Task-appropriate metrics and confusion matrix | F1 for class 1, AP, precision, recall, ROC-AUC and confusion matrix; Section 10. |
| Test evaluation once; fitting/error discussion | Freeze every candidate and threshold before a single test evaluation batch; examine errors without further tuning. |
| Save pipeline with joblib; pin versions | Versioned inference artifact, metadata and tested `requirements.txt`; Section 13. |
| Streamlit form, probability, explanatory chart, public URL | Validated form, estimated probability, saved decision threshold and explanation chart; Section 12. |
| Hugging Face and local backend options | Omitted under the user's instructor-approved Streamlit-only deployment scope. |
| Repo folders and README | `data/` or download script, `notebooks/`, `app/`, `models/`, dependencies and reproducible setup; Section 13. |
| README deployment links | Public Streamlit link and a short note recording the approved exception. |
| Report PDF, 3–4 pages, all required sections | Four-page layout in Section 16; comparison table, plots, screenshots, references and contributions included. |
| Ten-minute demo, followed by viva | Three minutes problem/data, four minutes Streamlit demo, three minutes results; Section 16. |
| Marking: demo 4, report 3, individual viva 3 | Prioritize lifecycle correctness, functioning deployment, readable evidence and every member's understanding. Original demo split: lifecycle 2, deployments 2; treatment of the exception follows the instructor's approval. |
| Live failures, citation and lateness rules | Rehearse the public app, cite data/code/report sources, submit early. The original rules state zero for a failed platform, zero for uncited copying, and one mark lost per late day. |

## 4. Data acquisition, integration and feasibility

### 4.1 Source records

The [PBS Labour Force Statistics page](https://www.pbs.gov.pk/labour-force-statistics/) provides LFS 2024–25 microdata and supporting documentation. Obtain the matching data, questionnaire, coding scheme, published report and variable labels. Preserve original downloads unchanged. The prepared file mentioned in the proposal must be traced back to these sources before it is trusted.

Use Census 2023 district-level Table 1 for population density, urban proportion and average household size where the LFS geography supports a reliable match. These indicators are listed in the [official Census Excel results](https://www.pbs.gov.pk/result-excel/). Both datasets can come from PBS and still be distinct public sources.

For each downloaded file, record source URL, publisher, dataset/table/version, geographic and temporal coverage, actual access date, collection/download method, file checksum, format, unit of observation, licence or reuse terms, transformations and required citation. Public availability does not establish an open redistribution licence. If permission to redistribute microdata is unclear, publish scripts and instructions instead of the microdata.

### 4.2 Integration rules

1. Recover person, household, PSU and geographic identifiers using the matching release documentation. Verify uniqueness and identifier reuse across quarters; never guess substrings from an older LFS release.
2. Assemble the household roster and person-level tables using validated one-to-one or one-to-many relationships as appropriate. Audit accidental row multiplication.
3. Build a documented Census crosswalk using stable geographic codes and verified boundaries. District names alone require normalization and manual review of ambiguous matches.
4. Keep one Census row per chosen geographic key, then join to LFS people with a **many-to-one** check. Assert that the join does not change the number of person records.
5. Report match coverage, unmatched regions, boundary changes and duplicate keys. Census values stay fixed external context; do not calculate regional unemployment rates from the full LFS and feed them back as features.
6. Prefer complete verified matches. If district mapping is impossible, use a consistent province-level linkage decided before modelling, document the loss of detail and reassess predictive value. Never silently mix geographic levels or impute an unverified district identity.

Repeated Census values among residents of the same region are expected. They are not themselves target leakage because the Census covariates are independently sourced and contain no LFS outcome. They do limit claims about performance in entirely unseen regions.

### 4.3 Data readiness gate

Before fitting, publish an audit summary: raw people; youth aged 15–24; resolved employed/unemployed; excluded outside-labour-force cases; unresolved labels; duplicates; Census match counts; final eligible rows; independent households/PSUs; class counts; and final predictor count.

At least 5,000 genuine eligible rows and 15 meaningful predictors must remain. Do not count the target, IDs, weights, one-hot-expanded columns, duplicate fields or synthetic oversampling as satisfying these requirements. If the proposal's 26,000 records include inactive youth, the usable total may shrink. Resolve a shortfall using legitimate additional raw observations or documented predictors before modelling; do not manufacture rows or keep leaky fields to meet the minimum.

## 5. Target definition and predictor contract

### 5.1 Finalize the label before model development

Use the documented LFS 2024–25 framework consistently, preferring its 19th ICLS classification where the release supports it. Do not mix 13th- and 19th-ICLS variables. The [LFS annual report's unemployment section](https://www.pbs.gov.pk/wp-content/uploads/2020/07/LFS-2024-25-Annual-Report.pdf) describes unemployment through lack of work, recent active job search and availability; the exact implementation must also respect official employment rules and exceptions.

Prefer a documented official derived labour-force-status variable when available and verify its codes against source responses. Otherwise implement a reviewed decision table from the matching questionnaire/manual. Cover temporary absence from a job, own-use production, future starters, search/availability exceptions and ambiguous responses according to the selected framework. Do not reduce the label to a single "worked last week" answer.

Keep only people aged 15–24 classified as employed or unemployed. Students may be in either class when the official criteria apply; student status alone is neither an exclusion nor an unemployment label. Keep outside-labour-force, unknown and inconsistent records out of the binary target with explicit exclusion reasons. Do not impute missing outcome-defining answers to manufacture labels.

Create `label_definition.md` with codebook references, raw-column mappings, precedence rules and worked examples. At the user's direction, Codex reviewed a stratified sample of source records and documented the resulting rule changes; this is self-review, not independent validation. Compare weighted rates with the same published age group, geography and framework as a plausibility check; never change labels simply to match a published aggregate.

### 5.2 Proposed 15-feature core

The following are **candidate model features**, not verified columns in the unavailable prepared dataset. Resolve every source variable and availability condition during the audit. Basic demographic and education questions appear in the [official LFS questionnaire](https://www.pbs.gov.pk/wp-content/uploads/2020/07/Questionnaire-of-LFS-2024-25-Final.pdf).

| # | Candidate feature | Source and treatment |
| --- | --- | --- |
| 1 | Age in completed years | LFS; numerical; enforce 15–24. |
| 2 | Recorded sex/gender category | LFS; categorical; retain documented categories and audit subgroup performance. |
| 3 | Marital status | LFS; categorical. |
| 4 | Relationship to household head | LFS; categorical. |
| 5 | Can read with understanding | LFS; categorical/binary with unknown preserved. |
| 6 | Can write with understanding | LFS; categorical/binary with unknown preserved. |
| 7 | Highest education category | LFS; categorical; do not assume codes are equally spaced years. |
| 8 | Current enrolment category | LFS; categorical; retain valid employed/unemployed students. |
| 9 | Household size | Complete usual-resident roster; count all ages, not just eligible youth. |
| 10 | Household age-dependency ratio | `(members aged <15 or >=65) / members aged 15–64`; demographics only. |
| 11 | Province/territory category | LFS; only verified coverage. |
| 12 | Urban/rural locality | LFS; category from the documented coding. |
| 13 | Regional population density | Census; numerical; candidate `log1p` transformation. |
| 14 | Regional urban population share | Census; numerical; consistent fraction or percentage units. |
| 15 | Regional average household size | Census; numerical; distinguish from the person's household size. |

Compute roster-based inputs before youth filtering, using only household composition. Put the actual feature derivations and external Census lookup into the inference pipeline. For the app, collect household age-band counts rather than other household members' identities. A youth aged 15–24 implies at least one working-age roster member; reject contradictory input.

If a candidate is unavailable or inappropriate, a replacement must have a documented source, a clear meaning, availability at prediction time and a leakage review. The final dictionary records input type, allowed values, missingness semantics, source question, transformation and inference-time input for every feature. These demographic predictors may have modest signal; that is an empirical result, not a reason to relax the leakage standard.

### 5.3 Explicit forbidden predictors

Exclude employment-status fields, all answers used to construct the target, current work activity, job search, availability, unemployment duration, reasons for not working, current hours, earnings, occupation, industry, job tenure, employment contract and employer details. Exclude administrative IDs, raw person/household/PSU codes, row number, survey weight and any prediction or label-derived aggregate.

Missingness can leak too: a job-only question being blank may reveal unemployment. Do not include a missing-indicator version of an excluded field. Migration reasons such as finding or searching for a job also require exclusion from the core predictor set.

The training question distinguishes training received during a job, before a job, currently received, and no training. Treat this as a post-outcome/proxy risk. **Exclude raw job-timed training codes and related unverified course variables from the primary classifier.** Preserve vocational training in the descriptive skills module. Add a predictive training feature only if its definition and timing can be justified independently of current employment; never assume collapsing categories removes the timing problem.

This conservative choice preserves the proposal's education/training analysis while protecting the prediction claim. Keep an accepted/rejected feature register explaining every decision.

## 6. Strict leakage prevention and split protocol

### 6.1 Separate audit work from model development

Allowed before splitting: fixed codebook-based validation, label construction, identifier checks, duplicate reconciliation, a fixed external Census join and collection of group/class counts needed to construct valid partitions. These steps must not use predictive performance or dataset-wide fitted statistics.

Only after partitioning: outcome-aware EDA, feature selection, imputation, scaling, learned outlier thresholds, category-frequency grouping, model fitting, calibration and threshold selection. Every learned transformation belongs inside the relevant training fold. This follows the separation described in [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Identical predictor profiles can be legitimate different people. Detect repeated source records using person identifiers and provenance; do not delete different respondents merely because their 15 features match.

### 6.2 Group unit and split construction

Use the highest reliably identified dependence unit: **PSU by default**, with all its households together. If a person/household recurs across extracts or quarters, connect those records into the same group even if another identifier changes. The [2024–25 coding scheme](https://www.pbs.gov.pk/wp-content/uploads/2020/07/CODING-SCHEME-LFS-2024-25.pdf) must guide identifier reconstruction. IDs remain audit metadata and never enter the feature transformer.

Create an approximately 15% test holdout with `GroupShuffleSplit` and a predeclared seed (42). From the remaining approximately 85%, allocate approximately 15/85 to validation using a second fixed seed (43). This gives roughly 70/15/15; group sizes mean row counts need not be exact. Verify both classes, enough positive groups and reasonable province/locality coverage in each partition.

If a deterministic split is infeasible, use a documented group-stratified allocation based only on group sizes, class counts and geography. Never search for a split that gives better model scores. Save assignments and checksums before EDA. Do not reshuffle after weak results.

If PSU identifiers truly cannot be recovered, use household-disjoint partitions as the documented fallback, explain residual within-PSU dependence, and narrow generalization claims. Missing household identifiers are a blocking data-quality issue; ordinary random row splitting is not an acceptable fallback.

The main test estimates performance on unseen groups from the covered survey population. It is not an independent new survey, a future-year test or proof of transfer to unseen districts. Optionally run a district-held-out development analysis using training data only, if there are sufficient districts and positives.

### 6.3 Exact allocation of work

| Data portion | Permitted use | Forbidden use |
| --- | --- | --- |
| Training, about 70% | Development EDA, feature decisions, five-fold searches, fitting, training-internal calibration and ablations. | Fitting anything using validation/test information. |
| Validation, about 15% | One planned comparison of shortlisted models, decision-threshold selection, final deployment choice and descriptive error review. | Training imputation/scaling, unrestricted iterative feature redesign, reporting tuned results as unbiased. |
| Test, about 15% | One frozen evaluation batch, predeclared subgroup metrics, plots and uncertainty from saved predictions. | Tuning, choosing a winner, changing thresholds/features, calibrating or selecting a lucky seed. |

Inside the training partition use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=44)` with the same group definition. Supply explicit split indices to each search, or correctly pass groups to `fit`. Audit all five folds for disjoint groups and both classes. [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) aims for class balance while keeping groups intact; exact balance is not guaranteed.

### 6.4 Checks that must pass

- No person, household, PSU or linked repeat-record group crosses partitions or CV folds under the chosen protocol.
- Source-record duplicates are resolved; the Census merge does not multiply records.
- The selected feature list has zero intersection with the target/proxy denylist; IDs and weights cannot pass through a catch-all transformer.
- Preprocessing, feature selection, balancing and any calibration use only the appropriate training observations.
- No resampling before splitting. The core plan uses no synthetic oversampling; validation/test class distributions remain natural.
- Learned age/education/category mappings are not inferred from test values. New input categories follow a predefined safe handling policy.
- Test files are absent from development notebooks and the app. A dedicated evaluation script reads them only after the freeze manifest is written.
- Pipeline tests verify fitted statistics against training-only inputs and confirm that changing a held-out value cannot change fitted parameters.
- A training-only label-shuffle sanity experiment, respecting groups, must not produce persistent unexplained ranking signal. Investigate suspiciously near-perfect real-label scores before celebrating them.

## 7. Exploratory analysis and descriptive statistics

Create at least these seven plots using the training partition. Save each figure and a short interpretation stating its denominator and limitations.

| Plot | Question and required interpretation |
| --- | --- |
| Class counts and proportions | How imbalanced are employed and unemployed youth in this sample? Display positive support. |
| Missingness by feature | What is unknown, structurally inapplicable or potentially employment-dependent? |
| Numeric distributions | Age, household size, dependency ratio and Census density; report skew and implausible values. |
| Boxplots or quantile plots | Identify outliers without deleting plausible large households or dense regions merely for being extreme. |
| Numeric correlation heatmap | Detect redundant numeric context; do not interpret arbitrary category codes as quantities. |
| Education/enrolment versus unemployment | Compare rates with group counts, uncertainty where feasible, and explicit noncausal wording. |
| Province/locality/sex and training summaries | Inspect representation and descriptive education/training differences; suppress very small cells. |

Use a prespecified small-cell display rule, initially fewer than 30 respondents; this is a project disclosure/stability rule, not an assertion of an official PBS threshold. Plot combined or suppressed categories consistently rather than hiding inconvenient results.

Keep two statistical purposes separate. Unweighted classification metrics describe performance on the held-out sample. Population unemployment summaries require survey weights and the correct labour-force denominator. A training-partition chart is a development-subsample chart, even when weighted. After model freeze and final evaluation, build population dashboards from the full eligible survey with proper weights; these tables must not feed model features or further tuning.

For geographic unemployment rates, verify whether the LFS design supports estimates at that level. Default to adequately supported province/locality summaries; a district-level Census covariate does not make a district LFS unemployment estimate representative. If weights or design metadata are unavailable, label charts as sample proportions and avoid national population claims.

## 8. Reproducible preprocessing and inference

Use one importable scikit-learn `Pipeline` per estimator. It should accept the same schema in notebooks, evaluation scripts and Streamlit:

```text
Validated demographic inputs + household counts + geography
  -> fixed schema/codebook transformer and Census lookup
  -> feature engineering (household ratio, optional log density)
  -> ColumnTransformer
       numeric: training-fold median imputation -> LR scaling
       categorical: explicit missing category -> one-hot encoding
  -> classifier
  -> optional group-fitted probability calibration
  -> probability + separately saved decision threshold
```

The fixed Census lookup uses a versioned external table, never statistics estimated from held-out LFS people. Raw source assembly and label construction stay in reproducible preparation scripts; all prediction-time transformations are included in the saved inference object.

Use `OneHotEncoder(handle_unknown="ignore")` with category handling learned inside the fold; validate genuinely unsupported app inputs before prediction. Preserve education as categories unless a documented ordinal mapping is justified. Use `StandardScaler` for Logistic Regression; scaling is unnecessary for tree models. Choose sparse/dense output compatible with the selected estimator and measure memory use.

Handle impossible values with predefined domain rules. If quantile clipping or rare-category grouping is tested, learn cutoffs only inside each training fold. Keep retained missingness indicators limited to approved predictors. Do not fit imputers, encoders, scalers or feature selectors on the full dataset before CV.

Implement custom transformers in source modules rather than notebook-local functions, so saved artifacts load in a fresh environment. Pin the tested Python and package versions during implementation; this plan does not invent a dependency lock before compatibility is tested.

## 9. Models and tuning budget

### 9.1 Baselines and required algorithms

Use `DummyClassifier(strategy="prior")` for a constant training-prevalence probability and its default class prediction. Also report a fixed all-unemployed baseline (`strategy="constant", constant=1`). If employed is the majority, a majority classifier has unemployment F1 of zero; beating only that is weak evidence. The all-positive baseline has F1 `2p / (1 + p)` at evaluation prevalence `p`, so it supplies a more useful comparison.

| Model | Purpose | Initial bounded search |
| --- | --- | --- |
| Logistic Regression | Interpretable linear reference | `C`: 0.01, 0.1, 1, 10; class weight: none/balanced; L2 penalty and compatible solver; raise iterations until convergence. Eight configurations. |
| Random Forest | Nonlinear interactions | 200/400 trees; depth 5/10/20/unlimited; leaf size 2/5/10/20; max features sqrt/0.7; class weight none/balanced_subsample. Randomly sample 20 configurations. |
| GradientBoostingClassifier | Boosted nonlinear reference from the proposal | 100/200/300 trees; learning rate 0.03/0.05/0.1; depth 1/2/3; leaf size 5/10/20; subsample 0.8/1.0. Randomly sample 20 configurations. |

Use five-fold `GridSearchCV` for the small logistic grid and `RandomizedSearchCV` for forests/boosting. The initial budget is about 240 fitted search folds plus refits and baselines. Measure one trial first; reduce search size, not the required five folds, if the available machine cannot complete it. Limit parallel jobs to available memory.

Disable Gradient Boosting's automatic random validation split (`n_iter_no_change=None`) so internal early stopping does not silently violate grouping. Keep all estimator and search seeds in configuration. For the core boosting run, use ordinary training weights; do not pass an unsupported `class_weight` argument. Class balancing and survey design weights are different concepts.

### 9.2 Selection sequence

1. Fix the core feature set, split manifest, metrics, search spaces and seed policy before searches.
2. Within training CV, score AP, positive-class F1 at 0.5, precision, recall and ROC-AUC. Refit the best configuration in each algorithm by mean CV AP. AP is the threshold-independent search criterion; final project success remains unemployment F1.
3. Compare training and CV scores, fold variability, runtime, convergence and feature behaviour. Training-CV gaps are diagnostic, not proof of leakage by themselves.
4. Conduct only preplanned training-CV ablations: demographics alone versus demographics plus Census; and all approved demographic attributes versus removal of sex/gender. These are evidence about incremental value/sensitivity, not additional required model families.
5. Fit the three shortlisted configurations on the 70% training partition. Choose a validation threshold for each using the fixed procedure in Section 10. Select the deployment model by validation F1; if within 0.01, prefer higher validation AP, then the simpler/faster model. Record this rule before viewing validation scores.
6. Freeze the selected model, all comparison models, thresholds and evaluation configuration. Do not refit on train+validation in this semester protocol: that would change the probabilities after threshold selection. Deploy the exact artifact evaluated on the test set.

For probability calibration, start by reporting raw probability reliability. If calibration is included, decide before the validation comparison and use grouped splits entirely inside the 70% training partition. `CalibratedClassifierCV` must wrap the entire preprocessing pipeline and receive explicit group-disjoint split indices. Comparing calibrated and uncalibrated variants requires an outer training-only assessment with calibration inside each outer training fold. Do not accept the calibrator's default ungrouped CV or fit calibration on the threshold-selection/test set. Calibration can improve interpretability of probabilities; it does not guarantee improvement. See [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).

## 10. Evaluation metrics, thresholds and success criteria

### 10.1 Metric choice

**Use F1 for the unemployed class as the headline metric.** It preserves the proposal's commitment and balances missed unemployed youth against false unemployment flags when neither cost has been formally prioritized. It is not weighted-F1, micro-F1 or accuracy. State `pos_label=1`, the threshold and positive support everywhere. If a real deployment later assigns higher cost to missed cases, a recall-constrained or F2 policy would need separate justification and validation.

| Metric | Role and interpretation |
| --- | --- |
| F1 for class 1 | Primary operating-point quality: `2TP / (2TP + FP + FN)`. It ignores true negatives, so accompany it with the confusion matrix. |
| Precision for class 1 | Among flagged observations, how many are unemployed? `TP / (TP + FP)`. |
| Recall for class 1 | Among unemployed observations, how many are identified? `TP / (TP + FN)`. |
| Average Precision | Main ranking metric; computed from positive-class probabilities with `average_precision_score`, not predicted labels. |
| ROC-AUC | Secondary ranking discrimination, never the only headline in an imbalanced task. |
| Confusion matrix | Counts in a fixed employed/unemployed order, plus row-normalized rates. |
| Brier score and reliability curve | Check whether displayed probabilities are credible; compare with the constant training-prevalence predictor. Lower Brier is better, but it also reflects discrimination. |
| Accuracy / macro-F1 | Optional context only; do not use these to hide weak minority-class performance. |

The [F1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) defines the positive-class calculation. Use the label **Average Precision (AP)** consistently: scikit-learn AP is not identical to trapezoidal PR-AUC, as its [AP documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) explains. Plot a precision-recall curve with the evaluation prevalence as the no-skill reference. For constant scores, AP equals prevalence; AP lift is `AP / prevalence`.

### 10.2 Threshold policy

Using each shortlisted model's validation probabilities, evaluate a fixed grid from 0.01 through 0.99 in steps of 0.01, plus the all-positive and all-negative endpoints. Choose maximum validation F1; break exact ties by higher recall, then the higher threshold. Record the threshold and resulting precision/recall. Also retain the 0.5-threshold result for transparency.

Threshold selection is a model-development decision; the resulting validation F1 is optimistic and is not the final evidence. Never select a threshold from the test precision-recall curve. Separate fitting and threshold data as described in [scikit-learn's threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html). Freeze threshold values before testing and do not expose a production slider that changes the reported model's operating point.

### 10.3 What counts as a good result

| Level | Predeclared interpretation |
| --- | --- |
| Mandatory methodological quality | Valid labels; 5,000+ eligible rows; 15+ defensible predictors; passed leakage checks; three models plus baselines; complete honest metrics; working app and deliverables. |
| Useful predictive evidence | Frozen model beats both the majority/prior and all-positive baselines on unemployment F1, and AP exceeds test prevalence. Report paired uncertainty rather than relying on a tiny point difference. |
| Practical development ambition | At least +0.05 absolute F1 over the stronger trivial baseline, AP lift at least 1.5, and validation precision/recall that make the chosen threshold intelligible. These are proposed ambitions, not instructor requirements. |
| Stretch aspiration | Unemployment F1 of 0.60 or better with stable subgroup behaviour. Feasibility is unknown until the cleaned data are evaluated; no 90–95% claim is justified in advance. |

Absolute score goals are not guarantees. Honest moderate performance with strong methodology is more defensible than inflated scores from target proxies. If the frozen test result misses an ambition, report it and its limitations; do not reopen the test set for optimization.

### 10.4 One final test evaluation

Run one evaluation command after the freeze manifest records feature schema, source/split checksums, model hashes, thresholds, metrics, subgroup rules and software versions. It may evaluate **all already-frozen comparison models in one batch**, satisfying both the comparison-table requirement and the rule against repeated test-guided development. The deployment winner has already been selected on validation and is not changed based on this table.

Save test probabilities and predictions once. Produce all plots, subgroup summaries and uncertainty intervals from that frozen output without refitting. Use 1,000 paired bootstrap resamples of independent test groups (PSUs under the preferred split) for approximate 95% intervals for F1, AP and their differences from baselines. Reuse the same sampled groups for comparisons, report invalid single-class resamples and avoid treating CV fold standard deviation as a confidence interval. These intervals describe sampling variability for the frozen model; they do not include all model-selection uncertainty or replace full complex-survey variance estimation.

Report metrics by sex/gender, age 15–19 versus 20–24, urban/rural locality and province where sample and positive-class support suffice. Use the prespecified small-cell policy and mark undefined rates as unavailable. Discuss false negatives and false positives, sample coverage and calibration; do not adjust group-specific thresholds after seeing test results.

Primary model comparisons are unweighted and explicitly sample-level. If valid survey weights exist, include a separately labelled weighted sensitivity table using the same frozen predictions and thresholds. Keep weights out of predictors. Do not silently mix weighted and unweighted metrics in one ranking or claim population representativeness from an unweighted classifier.

### 10.5 Results table template

| Model | Train-CV AP, mean ± SD | Validation threshold | Test F1 (class 1) | Test precision / recall | Test AP / ROC-AUC | Test Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy prior/majority | To measure | Fixed baseline | To measure | To measure | To measure | To measure |
| Dummy all-positive | To measure | Fixed rule | To measure | To measure | To measure | To measure |
| Logistic Regression | To measure | Validation only | To measure | To measure | To measure | To measure |
| Random Forest | To measure | Validation only | To measure | To measure | To measure | To measure |
| Gradient Boosting | To measure | Validation only | To measure | To measure | To measure | To measure |

Add row counts, prevalence, confidence intervals and the preselected deployment model in the final presentation. Never fill this table with expected, illustrative or invented results.

## 11. Explainability and skills analysis

For global explanations, calculate validation permutation importance for the entire pipeline using AP and repeat permutations to show variability. Perturb raw conceptual features together when they expand into multiple encoded columns. Recognize that correlated regional/demographic features can share importance and that arbitrary permutations can create implausible combinations; interpret the chart as a model diagnostic.

For local explanations, use a compatible method for the chosen model, with a training-only reference/background sample. Logistic Regression contributions should be described in log-odds; tree SHAP contributions must use and label the supported output scale. If the saved predictor includes calibration or an ensemble, explain that final object correctly or restrict the app to a clearly labelled global importance chart. Do not display a base-model explanation as if it exactly decomposed a calibrated probability.

The skills page will show education and training coverage, training types where valid, and descriptive unemployment differences with denominators and sample counts. Use employment-timed training categories only as explicitly labelled retrospective descriptions. Treat training gaps as gaps in reported participation or credentials, not measured employer-demand shortages. Employer vacancies, demanded competencies and longitudinal outcomes are absent, so a causal course recommender or true skills-demand matching system is future work.

Avoid "changing this attribute will reduce unemployment" language. Demographic associations and explanations of a model are not intervention effects. A synthetic profile demonstration is acceptable; real person-level training data must not be displayed or logged.

## 12. Streamlit application and deployment

### 12.1 App views

| View | Required behaviour |
| --- | --- |
| Overview | Explain scope, survey year, eligible population, covered geography, sample size and limitations. |
| Profile explorer | Validated form; synthetic examples; probability, class-1 threshold and a cautious model decision; explanatory chart. |
| Population explorer | Weighted descriptive summaries when supported, filters and visible denominators; small-cell handling. |
| Skills and education | Descriptive training/education participation and employment associations with explicit limits. |
| Model evidence | Frozen metrics, comparison table, PR curve, confusion matrix, calibration plot and leakage protocol. |

Limit profiles to ages 15–24 and supported geography. Present the score as conditional on labour-force eligibility. Use an informational eligibility acknowledgement rather than asking all target-defining employment questions and then pretending the model discovered the answer. Do not score out-of-scope or unknown-eligibility cases as if validated.

Collect only approved demographic inputs and household counts. Derive Census values automatically from the selected region; do not ask users to type regional statistics. Reject inconsistent household counts, invalid categories and unsupported geographic combinations with clear messages. Use codebook-valid missing options where the training pipeline supports them.

Show "Estimated probability of current unemployment within the model's eligible population" and the saved threshold. Do not describe a probability as a guarantee or use unsupported low/medium/high risk bands. If probabilities are poorly calibrated, visibly describe them as model estimates and report the calibration evidence. The form is educational and exploratory, not a decision system for jobs, benefits or admission.

### 12.2 Hosting and reliability

Implement the app locally, then deploy the repository branch and `app/streamlit_app.py` entrypoint to Streamlit Community Cloud, following its [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app). Pin compatible packages and Python, commit only permitted assets, load the trusted frozen artifact with caching, and keep training out of app startup. Use only the Streamlit URL in the final deployment list.

Test a clean installation, artifact loading, normal/missing/invalid inputs, unseen categories, Census lookup, one explanatory chart and agreement between batch and app predictions. Add Streamlit `AppTest` coverage where feasible and manually test the public URL in a fresh browser session and on a second device. Measure cold-start time and warm inference time; target a warm single-profile response below two seconds on the actual host, without promising it before measurement.

Commit no secrets or personal microdata. Do not retain form inputs or use them for automatic retraining. Precompute safe aggregated charts. Prepare screenshots and a short recording, plus a local run of the same Streamlit app as a demo contingency; these do not replace the required public deployment or constitute another platform.

## 13. Repository, artifacts and reproducibility

Proposed implementation layout:

```text
HunarLens/
  README.md
  PROJECT_PLAN.md
  requirements.txt
  requirements-dev.txt
  .gitignore
  configs/
    data.yaml
    features.yaml
    experiments.yaml
  data/
    README.md
    source_manifest.csv
    raw/                    # local/ignored unless redistribution is permitted
    reference/              # approved Census table and geographic crosswalk
    processed/              # local/ignored person-level data
    splits/                 # private assignments; public audit summary
  notebooks/
    01_data_audit.ipynb
    02_training_eda.ipynb
    03_model_comparison.ipynb
    04_explanations.ipynb
  src/hunarlens/
    data.py
    labels.py
    features.py
    splits.py
    train.py
    evaluate.py
    explain.py
    inference.py
  scripts/
    download_data.py
    prepare_data.py
    train_models.py
    evaluate_once.py
  app/
    streamlit_app.py
  models/
    final_pipeline.joblib
    model_metadata.json
    decision_threshold.json
    feature_schema.json
  reports/
    figures/
    metrics/
    final_report.pdf
    model_card.md
    contribution_table.md
  tests/
    test_labels.py
    test_leakage.py
    test_pipeline.py
    test_app.py
  docs/
    label_definition.md
    feature_review.csv
    data_audit.md
    split_audit.md
    deployment_exception.md
```

Save complete fitted inference objects, not only bare estimators. Save feature order/types, class mapping, training prevalence, calibrated/uncalibrated status, decision threshold, selected parameters, seeds, source and split hashes, git commit, artifact hash, package versions, survey coverage and intended use. Keep raw test predictions private when they identify individuals; publish aggregate metrics and nonidentifying audit evidence.

The README must explain purpose, data acquisition and licence, label/eligibility definition, preprocessing and grouping, environment setup, preparation/training/evaluation commands, app startup, public URL, results, limitations, citations and team contributions. Mark code paths described here as planned until implemented; do not present absent scripts as runnable now.

Verify reproducibility from a clean environment on a second team member's machine. Check that the frozen artifact gives the same probabilities within a documented numerical tolerance and identical threshold decisions. Only load project-controlled `joblib` files; never accept model uploads from arbitrary users.

## 14. Schedule and milestone gates

The guideline fixes proposal submission in Week 12 and final demonstration in Week 15. The current course week and exact deadline are unknown. Use the following relative schedule and map it to the actual course calendar; do not assume 13 November is Week 12.

| Course period | Work and evidence | Completion gate |
| --- | --- | --- |
| Week 12 / first available workdays | Audit proposal's prepared data; acquire originals; record sources/terms; finalize cohort and labels; inspect Census join and IDs. | Label review complete; minimum rows/features feasible; no unresolved source/identity problem. |
| Week 13, first half | Freeze splits; complete seven EDA plots; implement schema/feature pipeline and leakage checks; prepare Streamlit shell using synthetic inputs. | Zero cross-group overlap; all preprocessing demonstrably fold-local. |
| Week 13, second half | Run baselines and three algorithms; complete five-fold searches; inspect train/CV gaps; measure compute cost. | Saved reproducible search evidence for every required algorithm. |
| Week 14, first half | Complete bounded training-only ablations and any calibration; make the one planned validation comparison; choose thresholds and winner. | Feature/model/threshold/metric freeze; no further tuning planned. |
| Week 14, second half | Evaluate test once; create comparison/error/subgroup results; explanations and safe descriptive aggregates; finish deployed app. | Frozen results and exact matching deployed artifact; public app passes checks. |
| Week 15, early | Write and cross-review 3–4 page report; clean README/repo; rehearse ten-minute demo and viva; test cold start. | All deliverables ready at least 48 hours before the actual deadline. |
| Week 15, deadline | Submit URLs/files; warm and verify live app; present and answer viva. | Successful submission with actual contribution evidence. |

Critical dependency: source/label audit → split freeze → EDA/pipeline → CV → validation choice → model freeze → one test evaluation → final report/demo. App layout and synthetic demonstrations can proceed while modelling is underway. No external calendar date is invented.

If time is short, remove optional calibration comparisons, optional local SHAP and optional district-transfer analysis first. Keep valid labels, leakage controls, five-fold tuning, the required model count, one test evaluation and all submission deliverables.

## 15. Team responsibilities

These are proposed responsibilities to make the plan executable; update the final contribution table to reflect actual work.

| Member | Primary coding responsibility | Cross-review and presentation responsibility |
| --- | --- | --- |
| Danish Meraj, CT-24268 | System architecture, feature pipelines, model training/search, threshold optimization, explainability, inference engine, and Streamlit application/deployment. | Review inference parity; lead live demonstration; explain the full model lifecycle. |
| Sarim Khan, CT-24266 | Source preparation, Census merge, target construction, feature dictionary and data audit. | Review deployment schema; present problem/data; explain label and leakage logic. |
| Muhammad Saraib, CT-24255 | Grouped splitting, training-only EDA, frozen test evaluation outputs and subgroup audits. | Review target tests; present results; explain metrics, overfitting and baseline comparisons. |
| All three | Documentation, substantive code contributions, tests, peer review and reproducibility run. | Each rehearses every module, not only their assigned component. |

Keep a concise weekly contribution log with commit or issue references. Every member should be able to explain why the test set stayed sealed, why target questions were excluded, how groups are formed, how F1 differs from AP, why a threshold differs from 0.5, and what the model cannot infer.

## 16. Final report and demonstration

### 16.1 Four-page report layout

| Page | Required content |
| --- | --- |
| 1 | Title/team, short abstract, problem/motivation, intended user, two-source collection/provenance, target/eligibility and final row/feature counts. |
| 2 | EDA highlights and compact figures; pipeline, split diagram, leakage safeguards, algorithms and five-fold tuning method. |
| 3 | Comparison table and selected test metrics; PR/confusion/calibration visuals as space permits; baseline comparison, uncertainty, over/under-fitting and error discussion. |
| 4 | Streamlit screenshot and public URL; deployment exception; limitations/future work; references and individual contribution table. |

Ensure every required heading is visibly represented even when space is shared. Put the complete five-plus EDA plots, detailed grids and full audit records in the repository; include legible representative figures in the 3–4 page report. Cite borrowed code and external material and identify adaptations.

### 16.2 Ten-minute demonstration

- **0:00–3:00:** Problem, intended user, two sources, eligible population, label, leakage-prone fields and grouped split.
- **3:00–7:00:** Open the public Streamlit URL; submit a synthetic in-scope profile; explain probability/threshold/chart; show input validation, descriptive skills page and methodology page.
- **7:00–10:00:** Explain model comparison, primary F1, AP versus prevalence, precision/recall, confusion matrix, limitations and contribution evidence.
- **After 10:00:** Individual viva. Each member explains their code and can answer questions about the complete system.

## 17. Risks and response plan

| Risk | Prevention and response |
| --- | --- |
| Target misclassification | Verify the selected LFS framework and official derivation; review edge cases; do not train until resolved. |
| Leakage through target questions or job-timed training | Feature allowlist plus denylist, codebook review and timing checks; keep risky training fields descriptive. |
| Fewer valid rows/features than proposed | Audit before modelling; recover legitimate source observations/predictors; never pad counts or retain leaks. |
| Missing group identifiers | Recover from matching source documentation; keep data work open until household separation is possible. |
| Invalid regional merge | Unique-key tests and row-count checks; documented crosswalk or uniform coarser geography. |
| Weak predictive signal | Improve label quality, safe feature representation, regularization and validation thresholds within the fixed budget. Report weak final evidence honestly. |
| Unusually excellent scores | Audit forbidden predictors, missingness, IDs, duplicate records and split overlap immediately. |
| Imbalance and small subgroups | AP and positive-class metrics, class-weight comparisons, uncertainty and small-cell rules. |
| Unsupported skills-gap claims | Report observed training/education patterns; defer employer-demand matching to a future dataset. |
| Calibration or explanation mismatch | Evaluate probabilities, explain the exact saved predictor or clearly use a global chart. |
| Survey/population overclaim | Distinguish sample metrics from weighted population summaries; state coverage/design limits. |
| Cloud memory/dependency failure | Bounded model size, pinned versions, cached loading, precomputed charts and early public deployment tests. |
| Deadline or viva failure | Freeze scope, retain a 48-hour submission buffer and cross-train all members. |

## 18. Definition of done

- Both source PDFs are reflected in the compliance matrix and the approved Streamlit exception is documented.
- Source manifest, reuse terms, raw-data provenance, label decision table and eligibility/exclusion counts are complete.
- The final dataset has at least 5,000 eligible genuine records and 15 reviewed predictors before one-hot expansion.
- All group, duplicate, join and train-only preprocessing checks pass; no target/proxy field enters the classifier.
- At least five EDA plots cover every required EDA category; interpretations and denominators are recorded.
- Dummy baselines plus all three algorithms have reproducible five-fold tuning and saved comparison evidence.
- A primary F1 metric, AP, threshold policy, subgroup policy and model choice were fixed before testing.
- Test evaluation occurred once for the frozen candidates; all reported results are measured, with limitations and uncertainty.
- The deployed pipeline, feature schema, probability handling and saved threshold match the evaluated artifact.
- Public Streamlit form, prediction/probability, explanatory chart, validation and results page work in a fresh session.
- GitHub structure, pinned setup, source instructions, model artifact, README, report, screenshots and URL are complete.
- The final report is 3–4 pages and includes references and truthful individual contributions.
- All three members contributed code and rehearsed the full ten-minute demo and individual viva.

## 19. Reference and verification notes

The project-specific requirements come from the supplied proposal, supplied course guidelines and the user's confirmed deployment exception. Technical design choices and targets in this plan are recommendations for HunarLens, not statements that the instructor mandated those exact methods or scores.

External references linked at their points of use were checked on 13 November 2026: official PBS LFS/data documentation and Census results; scikit-learn documentation for leakage, grouped CV, metrics, thresholds and calibration; and Streamlit Community Cloud deployment documentation. Exact microdata variable names, redistribution terms, geographic coverage, final cohort size and achievable metrics still require the actual dataset audit. No dataset inspection or model training is claimed by this plan.
