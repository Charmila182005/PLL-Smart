# PLL-Smart 2.1 — Semiconductor Lab Theme

## UI update
- Deep navy/teal semiconductor laboratory theme with cyan instrumentation accents, thin technical borders and restrained amber safety emphasis.
- Compact PLL-Smart 2.1 header with a wafer/chip illustration and four dataset/model KPI cards.
- Sidebar workspace selector for all 13 engineering sections, with upload controls retained in place.
- Executive Overview uses a two-row grid for held-out model metrics, existing model comparisons, supplied actual-vs-predicted data, and engineering insights.
- Desktop spacing is tuned for 1920×1080 and 1600×900 screens; the overview remains scrollable at 1366×768.
- Sidebar labels use high-contrast inactive/active states, and the optional Presentation layout hides secondary status details while retaining navigation and uploads.
- Shared styling is applied to charts, metrics, tables, inputs, upload controls and report downloads across all sections.
- Existing prediction, validation, comparison, reporting and data-management workflows are preserved.

## Safety and research framing retained
- Synthetic-data-only / non-production disclaimer remains visible.
- The global safety inhibit remains active: all recommendations remain `RUN FULL JITTER TEST`.
- No model retraining, model replacement, calibration changes, or test-skip authorization was introduced.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
