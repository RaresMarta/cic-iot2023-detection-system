# Analyzer: binary classification + manual single-flow entry

**Date:** 2026-06-28
**Status:** approved (design)

## Problem

The analyzer (`/api/classify` + frontend `ClassifyPage`) has two gaps:

1. **Binary not exposed.** The backend already accepts `mode` (`'2'` works), but the
   frontend hardcodes `form.append('mode', '8')` ([ClassifyPage.tsx:40]), so users
   can only run 8-class. Binary (Benign vs Attack) — the alerting use case — is
   unreachable from the UI.
2. **No manual single-flow classification.** Every path requires a file upload (CSV or
   pcap). There is no way to type/inspect a single flow's 25 features and get a
   prediction. This is wanted for the demo (load a real flow, perturb one feature,
   watch the verdict change — e.g. the `Number` leakage made visible live).

## Scope

In: a JSON endpoint for a single typed flow; 8 baked real-row presets; a frontend
mode toggle (binary/8-class) applied to both upload and manual paths; a manual
25-field form with preset loaders. Out: changing the model, the file-upload path's
behavior, or the live monitor.

## Design

### 1. Backend — `POST /api/classify-flow`

New endpoint in `ids/apps/analyzer/app.py`, alongside the untouched file-upload
`/api/classify`.

Request (JSON):
```json
{ "features": { "Rate": 12.5, "IAT": 0.08, ... all 25 X_COLUMNS_SELECTED ... },
  "model_type": "mlp", "mode": "2", "split": "random" }
```

Behavior:
- Resolve `predictor_key = f'{model_type}/{split}/{mode}'`; 404-style error dict if
  not in `PREDICTORS` (same contract as `/api/classify`).
- **Validate all 25 features present.** Missing keys → error dict listing them. No
  median-fill — the form always supplies all 25 (via presets), so a missing key is a
  real client bug, not a convenience case.
- Build a 1-row Polars DataFrame in `X_COLUMNS_SELECTED` order.
- Run the **same `predictor.predict(df, timer=timer)`** path as file upload, then
  `_aggregate` (flow_count=1). Identical preprocessing/inference → manual and upload
  agree by construction.
- Response: same shape as `/api/classify` (`top_label`, `confidence`, `probabilities`,
  `class_names`, `timing`, etc.), plus `input_type: 'manual'`.
- `mode` flows through, so binary works here too.

Pydantic model for the body keeps validation/typing clean.

### 2. Presets — `flow_presets.json`

A script (`scripts/extract_flow_presets.py`) reads `data/cic_iot_2023.parquet`, and
for each of the 8 families (remap_labels mode '8') picks the **median-closest real
row** (the actual row with minimum L1 distance to the per-feature median — real, not
synthetic, but representative rather than a noisy outlier). Writes the raw 25-feature
values per family to `ids-frontend/src/app/data/flow_presets.json`:
```json
{ "Benign": { "Rate": 186.29, "Number": 10.0, ... }, "DDoS": { "Number": 100.0, ... }, ... }
```
8 families × 25 features. (Verified: Benign/Recon Number=10, DDoS Number=100 — the
presets visibly carry the window-size leak, which is a demo feature.)

### 3. Frontend — mode toggle

In `ClassifyPage.tsx`: a Binary / 8-class selector (state `mode`, default `'8'` to
preserve current behavior). Stop hardcoding `mode='8'`; send the selected mode for
**both** the file-upload and manual paths.

### 4. Frontend — manual flow form

A toggle on the analysis/classify page: **"Upload file" | "Enter flow manually"**
(not a new route — same page, switched view). Manual view:
- 25 labeled numeric inputs, grouped (rates/timing, sizes, flags, protocol one-hots,
  counts) for scanability.
- 8 preset loader buttons (one per family) + Clear. Loading a preset fills all 25 from
  `flow_presets.json`.
- Classify button → `POST /api/classify-flow` with `{features, model_type, mode, split}`.
- Result reuses the existing `ResultsPage` (navigate with the result, as upload does).

### 5. Testing

- Backend (`tests/test_classify_api.py` or a new `test_classify_flow.py`):
  valid 25-feature payload → prediction with `flow_count==1`; missing feature →
  error dict naming it; `mode='2'` and `mode='8'` both return the right class_names.
- Presets: a check that `flow_presets.json` has 8 families each with all 25
  `X_COLUMNS_SELECTED` keys.

## Non-goals / decisions

- Require all 25 (no partial/median-fill) — the form guarantees completeness.
- Manual entry is a toggle on the existing page, not a separate route.
- Presets are median-closest **real** rows, baked at build time (static).
- File-upload path and live monitor unchanged.
