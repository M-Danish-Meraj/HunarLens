# HunarLens — Data Directory & Provenance Guide

This directory stores source manifests, approved external reference tables, and local (git-ignored) microdata and split partitions for the **HunarLens** project (CT-378 Machine Learning, Group AlphaDevs).

---

## 1. Directory Structure & Privacy Policy

| Path | Git Tracked? | Contents & Purpose |
| :--- | :---: | :--- |
| `data/README.md` | Yes | Provenance, licensing, and reconstruction instructions (this file). |
| `data/source_manifest.csv` | Yes | Official source URLs, publishers, coverage periods, access dates, units, and checksum/format specifications (§4.1). |
| `data/reference/` | Yes | Approved external **Pakistan Census 2023** regional lookup table (`census_2023_regional.csv`) and geographic crosswalk (§4.2). |
| `data/raw/` | **No** (`.gitignore`) | Original unmodified **PBS Labour Force Survey (LFS) 2024–25** microdata extracts (or deterministic schema-faithful synthetic microdata generated for offline/CI reproducibility). |
| `data/processed/` | **No** (`.gitignore`) | Cleaned, deduplicated, and cohort-filtered youth (aged 15–24) labour-force records and descriptive skills tables. |
| `data/splits/` | Partial | `split_manifest.json` (audit hashes, group/row counts) is tracked; individual-level `train.parquet`, `val.parquet`, and `test.parquet` are git-ignored (§6.4, §13). |

---

## 2. Two-Source Public Data Provenance (§4.1)

### Source 1: Pakistan Bureau of Statistics (PBS) — Labour Force Survey (LFS) 2024–25
* **Publisher:** Pakistan Bureau of Statistics (PBS), Ministry of Planning, Development & Special Initiatives, Government of Pakistan.
* **Official Portal:** https://www.pbs.gov.pk/labour-force-statistics/
* **Supporting Documentation:**
  * Annual Report: https://www.pbs.gov.pk/wp-content/uploads/2020/07/LFS-2024-25-Annual-Report.pdf
  * Questionnaire: https://www.pbs.gov.pk/wp-content/uploads/2020/07/Questionnaire-of-LFS-2024-25-Final.pdf
  * Coding Scheme: https://www.pbs.gov.pk/wp-content/uploads/2020/07/CODING-SCHEME-LFS-2024-25.pdf
* **Temporal & Geographic Coverage:** 2024–25 survey period covering Khyber Pakhtunkhwa, Punjab, Sindh, Balochistan, and Islamabad Capital Territory (distinguished via the first digit of the 8-digit Primary Sampling Unit code `PCode`).
* **Identifier Hierarchy:**
  * **PSU Key:** `PCode` (8-digit Primary Sampling Unit identifier; default grouping unit for `GroupShuffleSplit` and `StratifiedGroupKFold`).
  * **Household Key:** `(PCode, HHNo)` composite key.
  * **Person Key:** `(PCode, HHNo, SNo)` unique respondent key.
* **Redistribution & Privacy Policy:** Because public availability on the PBS website does not automatically grant an open microdata redistribution licence, raw respondent-level files in `data/raw/` and `data/processed/` are excluded from Git version control. Teammates and evaluators can either place the official PBS LFS 2024–25 extract at `data/raw/lfs_2024_25_microdata.csv` or run `python scripts/download_data.py` to verify/reconstruct a deterministic, schema-faithful dataset for reproducible pipeline execution.

### Source 2: Pakistan Bureau of Statistics (PBS) — 7th Population & Housing Census 2023 (Table 1)
* **Publisher:** Pakistan Bureau of Statistics (PBS), Government of Pakistan.
* **Official Portal:** https://www.pbs.gov.pk/result-excel/
* **Indicators Used:**
  1. `regional_pop_density`: Population density (persons per sq. km), transformed via `log1p` inside the feature pipeline.
  2. `regional_urban_share`: Proportion of population residing in urban areas (`0.0` to `1.0`).
  3. `regional_avg_household_size`: Regional average household size (distinct from the individual respondent's household size).
* **Crosswalk & Join Rule:** Stored in `data/reference/census_2023_regional.csv` keyed by verified `province_territory` (5 units: Khyber Pakhtunkhwa, Punjab, Sindh, Balochistan, Islamabad). Joined to LFS person records via a verified **many-to-one** relationship with strict row-count invariance assertions (§4.2).
