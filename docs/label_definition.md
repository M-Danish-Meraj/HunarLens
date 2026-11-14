# HunarLens — Target Label & Cohort Eligibility Definition (`label_definition.md`)

**Reference:** [PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md) §1, §5.1 · **Framework:** 19th International Conference of Labour Statisticians (19th ICLS), PBS Labour Force Survey (LFS) 2024–25.

---

## 1. Objective & Target Variable Contract
* **Target Column:** `is_unemployed`
* **Task Type:** Binary classification conditional on youth labour-force membership.
* **Class Encoding:**
  * `1` = **Unemployed** (Positive class for headline `F1` and `Average Precision`).
  * `0` = **Employed** (Negative class).
* **Out-of-Scope / Excluded Records:** Individuals aged `< 15` or `> 24`, individuals **outside the labour force** (neither employed nor unemployed under 19th ICLS), and records with unresolved/contradictory status answers are **excluded from the modelling cohort** with explicit audit reason codes. Missing outcome-defining responses are never imputed to manufacture labels.

---

## 2. Critical Data-Intake Finding on `Fstatus`
As documented in the project's data-intake audit ([PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md#L13)):
* The raw unlabelled `Fstatus` column in the downloaded PBS source file is constant and **must never be used as the target or as a predictor**.
* Instead, labour-force status and the binary target `is_unemployed` are derived strictly from the official 19th ICLS employment, temporary absence, job-search, and availability criteria below.

---

## 3. Step-by-Step Cohort & Label Decision Rules

### Step A: Household Roster Pre-Aggregation (Before Youth Filtering)
Before filtering to youth aged 15–24, compute household-level demographic predictors across the **entire usual-resident household roster** `(PCode, HHNo)`:
1. `household_size`: Total count of usual residents of all ages in `(PCode, HHNo)`.
2. `household_dependency_ratio`:
   $$\text{Dependency Ratio} = \frac{\text{Count}(\text{age} < 15 \text{ or } \text{age} \ge 65)}{\text{Count}(15 \le \text{age} \le 64)}$$
   *(Note: Any household containing an eligible youth aged 15–24 guarantees at least one working-age member in the denominator $\ge 1$.)*

### Step B: Youth Age Filter
* Retain respondents with completed years of age $15 \le \text{age} \le 24$.
* Respondents with $\text{age} < 15$ or $\text{age} > 24$ are excluded with reason `age_out_of_range`.

### Step C: 19th ICLS Labour-Force Status Classification
Evaluated in strict precedence order using the LFS activity, job-search, and availability fields:

| Precedence | Condition | Derived `labour_force_status` | Binary Target (`is_unemployed`) |
| :---: | :--- | :--- | :---: |
| **1** | Worked for pay or profit for $\ge 1$ hour during the reference week (`worked_last_week == 1`) | `employed` | `0` |
| **2** | Did not work last week, but had a paid job or enterprise from which they were **temporarily absent** (`temp_absent_from_work == 1`) | `employed` | `0` |
| **3** | Not employed (neither 1 nor 2), **actively sought work** during the past 4 weeks (`seeking_work == 1`) **AND** **currently available** to start work (`available_for_work == 1`), OR verified future starter waiting to begin an arranged job within the official window | `unemployed` | `1` |
| **4** | Not employed, and explicitly reported not seeking work (`seeking_work == 0`) or not available (`available_for_work == 0`) without qualifying exception (e.g., full-time inactive students, homemakers, discouraged/inactive persons outside labour force) | `outside_labour_force` | *Excluded* (`outside_labour_force`) |
| **5** | Missing, blank, or contradictory responses on outcome-defining items where neither employment nor active search + availability can be established | `unresolved_or_missing` | *Excluded* (`unresolved_or_missing`) |

---

## 4. Treatment of Students & Vocational Trainees
* **Current Enrolment (`current_enrolment`):** Per 19th ICLS rules (§5.1), students aged 15–24 who work for pay/profit (`employed`) or actively seek and are available for work (`unemployed`) are legitimate members of the labour force and remain in the cohort. Student status alone neither excludes a respondent nor defines them as unemployed.
* **Vocational Training (`vocational_training_status`):** Because the LFS training question distinguishes *on-the-job* vs. *prior* training, raw training variables carry post-outcome leakage risk and are **excluded from the 15-feature predictive contract** (§5.3), while preserved in `youth_descriptive_skills.parquet` for the descriptive Skills & Education dashboard tab.

---

## 5. Worked Examples

| Example ID | `age` | `worked_last_week` | `temp_absent_from_work` | `seeking_work` | `available_for_work` | `current_enrolment` | Resulting Status | `is_unemployed` |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **Ex-1** | 21 | `1` (Yes) | `0` (No) | `0` | `1` | `Not Enrolled` | `employed` | `0` |
| **Ex-2** | 19 | `0` (No) | `1` (Yes) | `0` | `1` | `Currently Enrolled` | `employed` | `0` |
| **Ex-3** | 23 | `0` (No) | `0` (No) | `1` (Yes) | `1` (Yes) | `Not Enrolled` | `unemployed` | `1` |
| **Ex-4** | 18 | `0` (No) | `0` (No) | `1` (Yes) | `1` (Yes) | `Currently Enrolled` | `unemployed` | `1` |
| **Ex-5** | 17 | `0` (No) | `0` (No) | `0` (No) | `0` (No) | `Currently Enrolled` | `outside_labour_force` | *Excluded* |
| **Ex-6** | 22 | `NaN` | `NaN` | `1` (Yes) | `NaN` | `Not Enrolled` | `unresolved_or_missing` | *Excluded* |

