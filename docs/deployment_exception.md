# HunarLens — Instructor-Approved Deployment Scope Exception (`deployment_exception.md`)

**Course:** CT-378 Machine Learning  
**Group:** AlphaDevs — Sarim Khan (CT-24266), Muhammad Saraib (CT-24255), Danish Meraj (CT-24268)  
**Reference:** [PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md) §1, §2, §3, §12

---

## 1. Context in Course Guidelines
The CT-378 Machine Learning Project Guidelines contain an internal discrepancy regarding deployment platforms:
* The overview, README checklist, and marking rubric table mention three deployment targets (Streamlit, Hugging Face Spaces, and a local Flask/FastAPI backend).
* Conversely, the **"Deployment"** section heading explicitly instructs teams to **choose one** platform.

## 2. Recorded Instructor Approval
Prior to implementation planning (recorded on 13 November 2026 in [PROJECT_PLAN.md](file:///d:/Hunarlens/PROJECT_PLAN.md#L9)), Group AlphaDevs obtained explicit instructor confirmation that:
1. **A single public Streamlit Community Cloud application (`app/streamlit_app.py`) fully satisfies the project deployment deliverable.**
2. Separate Hugging Face Spaces and Flask/FastAPI backend deployments are omitted under this approved exception.
3. All required end-to-end ML lifecycle features—validated profile input form, frozen `joblib` pipeline loading, estimated unemployment probability, validation-selected decision threshold, model explanation chart, descriptive population & skills exploration, and frozen test evaluation evidence—are consolidated into the multi-view Streamlit application.

## 3. Compliance Checklist for the Streamlit Deployment
- [x] Single unified scikit-learn `Pipeline` artifact (`models/final_pipeline.joblib`) loaded with caching (`@st.cache_resource`).
- [x] Five dedicated views (`Overview`, `Profile Explorer`, `Population Explorer`, `Skills and Education`, `Model Evidence`).
- [x] Automatic regional Census 2023 covariate lookup by selected `province_territory` (users never manually type regional density or urban share).
- [x] Explicit display of the frozen validation-selected decision threshold alongside the predicted positive-class probability.
- [x] Automated test suite (`tests/test_app.py`) verifying artifact loading, input validation, unseen category safety, and batch-vs-app prediction parity.
