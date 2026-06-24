# Benign-Augmentation Experiment + Preprocessor/Harness Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, batch with checkpoints) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. This is an autonomous overnight run: **every gate that says STOP must halt and write a report — never fabricate or guess a number past a failed gate.**

**Goal:** Fix the stale-preprocessor bug (which silently breaks serving and cross-dataset eval), harden the preprocessing/predict path with regression tests, then retrain the 2-class and 8-class models with augmented benign data and measure whether it reduces the ~99% false-positive rate on out-of-distribution IoT-23 traffic — validated on a second IoT-23 capture.

**Architecture:** Inference/serving load a saved `RobustScaler`-based `Preprocessor`. The saved `preprocessor_random.joblib` drifted out of sync with the deployed models, so every input is mis-scaled → model outputs all-attack. The correct preprocessor is reproducible by re-fitting on the train split. We (0) fix + test it, (1) bump benign volume, (2) retrain, (3) A/B the cross-dataset FPR, (4) record + commit.

**Tech Stack:** Python 3 (`.venv/bin/python`), Polars, PyTorch, scikit-learn (RandomForest, RobustScaler), dpkt, joblib. Project is a git repo.

## Global Constraints

- **Project root:** `/Users/wolfpack/uni/thesis-ids/cic-iot2023-detection-system` — all paths below are relative to it. It IS a git repo (`origin` = `github.com/RaresMarta/cic-iot2023-detection-system`). Top-level `/Users/wolfpack/uni/thesis-ids` is NOT a repo.
- **Run Python as:** `cd <root> && PYTHONPATH=<root> .venv/bin/python <script>` OR `.venv/bin/python - <<'PY'` heredoc from root (running a script *file* puts the script dir on `sys.path`, NOT the project — set `PYTHONPATH` or use heredoc).
- **Behavioral (the user is early-career and explicit about this):** NO favorable reframing. Verify every claim against source/output before asserting. Report negative results plainly. A failed gate = STOP + report, never a fabricated number. (See memory `no-favorable-reframing`.)
- **Commit messages:** no "Co-Authored-By", no AI attribution (user global rule).
- **Do NOT commit:** `docs/literature/` (user instruction) or `data/` (large: 151 MB parquet + pcaps — keep gitignored).
- **Hold ONE variable:** retrain at `max_depth=25` (unchanged). Do NOT also uncap depth — that confounds the benign experiment.
- **Training devices (Apple M1, 8 cores, torch 2.11, MPS available):** the **MLP trains on the M1 GPU (MPS) automatically** — `ids/core/models.py` sets `device = cuda → mps → cpu` and moves model+batches `.to(device)`; no change needed. **RandomForest is CPU-only** (`n_jobs=-1` across the 8 cores — scikit-learn has no GPU backend). The **RF retrains are the wall-clock bottleneck** (augmented ~5M rows, depth 25, 300–450 trees → >1 GB model, tens of min each, ×2 for 2-class+8-class); the M1 GPU cannot accelerate them. No AMP on MPS (CUDA-only in this code) — MLP runs full-precision, which is fine.

---

## CRITICAL CONTEXT (verified this session — a fresh worker has none of this)

### The bug
`models/preprocessor_random.joblib` is **stale / mismatched with the saved models**. Its `RobustScaler.center_` is ~2× off from a correct fit. Feeding the saved MLP through the LOADED preprocessor → predicts **all-attack** (FPR 1.00 on the CIC test set — impossible for a real in-dist model). Feeding it through a **fresh** `fit_preprocess` on the train split → FPR 0.021, matching the saved ground-truth artifact (`models/run_artifacts_random_2class.joblib`: `y_true`/`y_pred`, MLP test, TPR 0.911 / FPR 0.0177). **`ids/runtime/predictor.py` (`_BasePredictor.preprocess`) loads this stale file and is the SHARED train/serving path → the live Gradio demo currently mis-scales and would predict all-attack.**

### The correct-preprocessor recipe (verified)
```python
import numpy as np, polars as pl
from ids.data.preprocessing import split_random, fit_preprocess
from ids.core.config import SEED, X_COLUMNS_SELECTED, LOG_COLUMNS_SELECTED
df = pl.read_parquet('data/cic_iot_2023.parquet')
y34 = df['Label'].to_numpy()
Xall = df.select(list(X_COLUMNS_SELECTED)).to_numpy().astype(np.float32)
tr, va, te = split_random(y34, np.zeros(len(y34)), SEED)
Xtr, Xva, Xte, prep = fit_preprocess(Xall, tr, va, te,
    x_columns=list(X_COLUMNS_SELECTED), log_columns=list(LOG_COLUMNS_SELECTED))
# `prep` is the correct preprocessor. prep.transform(rawX) before model.predict.
```
`feature_columns.joblib == X_COLUMNS_SELECTED` (verified True). Preprocessing pipeline (`ids/core/preprocessor.py`): inf→null, log1p on non-flag cols, median impute, RobustScaler — fit on train only.

### Locked baselines (depth-25 models, CORRECT prep, attack = positive) — the A/B reference
| Test set | RF FPR | RF TPR | MLP FPR | MLP TPR |
|---|--:|--:|--:|--:|
| CIC in-dist test | 0.351 | 0.971 | 0.021 | 0.902 |
| IoT-23 honeypot-4-1 (benign-only, 1699 win) | 0.999 | — | 0.994 | — |
| IoT-23 34-1 mixed (2614 ben / 20248 atk) | 0.997 | 1.000 | 0.981 | 0.659 |

The **experiment succeeds if the augmented model's IoT-23 benign FPR drops materially below these** (and in-dist TPR does not collapse).

### Key files
- `ids/core/config.py` — `MAX_ROWS_PER_CLASS=200_000`, `SEED`, `X_COLUMNS_SELECTED`, `LOG_COLUMNS_SELECTED`, `MODELS_DIR`, `HPARAMS_PATH`, `load_hparams(model,mode)`.
- `ids/core/labels.py::remap_labels(y_34, mode)` — 34→8 / 34→2 mapping ('8' / '2').
- `ids/data/preprocessing.py` — `split_random(y34, source_csv, seed)`, `fit_preprocess(...)`.
- `ids/core/preprocessor.py::Preprocessor` — `.fit()/.transform()`.
- `ids/runtime/predictor.py` — `RFClassifier`, `MLPClassifier` (load `preprocessor_{split}.joblib`).
- `ids/runtime/extractor.py` — `extract_features`, `_packet_record`, `_window_features`, `_iter_packets`.
- `ids/eval/cross_dataset_eval.py` — `predict()` **(BUGGED: loads stale prep)**, `evaluate()`, `model_feature_columns()`.
- `ids/eval/iot23_align.py` — `extract_windows`, `parse_conn_log_labeled`, `label_windows` (windowing + Zeek-label join).
- `ids/training/` — training entry (`run_training`), `trainers.py` (reads `hparams.json`), `tune.py`. `hparams.json`: rf/2,rf/8,mlp/2,mlp/8 tuned params (rf depth=25).
- `models/` — `ids_rf_random_{2,8}class.joblib`, `ids_dnn_random_{2,8}class.pth`, `label_encoder_random_{2,8}class.joblib`, `feature_columns.joblib`, `preprocessor_random.joblib` (STALE), `run_artifacts_random_{2,8}class.joblib` (saved y_true/y_pred).

### Data
- `data/cic_iot_2023.parquet` — 4,448,253 rows; 34 classes capped 200k each; columns = 25 features + `Label`. Benign = 200,000.
- `data/iot-23/2018-12-21-15-50-14-192.168.1.195.pcap` + `data/iot-23/conn.log.labeled` — capture 34-1 (mixed).
- `data/iot-23/honeypot-4-1/honeypot-4-1.pcap` + `data/iot-23/honeypot-4-1/conn.log.labeled` — benign-only.
- **Benign CSVs (user downloaded this session):** location UNKNOWN to this plan — Task 4 locates them (search `~/Downloads`).

### Reference: the corrected cross-dataset script
A working corrected eval (fresh-prep) was written to the session scratchpad as `corrected_eval.py`. Its logic: build `prep` via the recipe above; for each capture `extract_windows(pcap)` → build `X` in `X_COLUMNS_SELECTED` order from the feature dicts → `prep.transform(X)` → `rf.predict_proba` / mlp forward → compare to joined labels. Fold this into the harness in Task 2.

---

## Phase 0 — Fix the preprocessor bug + lock it with tests (TDD)

> **REQUIRED SUB-SKILL for this phase:** invoke the **tdd** skill (`superpowers:test-driven-development`, or the top-level `tdd` skill) BEFORE writing Tasks 1–2. Follow its red→green→refactor discipline exactly: write the failing assertion first, run it and SEE it fail for the right reason (stale-prep → FPR≈1.0), make the minimal change, see it pass. Do not write the fix before the failing test exists. These parity tests are the regression guard that would have caught this bug — they are the point of this phase, not scaffolding.

### Task 1: Regression test proving serving prep reproduces the in-dist baseline

**Files:**
- Create: `tests/test_preprocessor_parity.py`
- Read: `models/run_artifacts_random_2class.joblib`, `data/cic_iot_2023.parquet`

**Interfaces:**
- Consumes: `ids.runtime.predictor.MLPClassifier`, `ids.data.preprocessing.split_random`, `ids.core.config.{SEED,X_COLUMNS_SELECTED,MODELS_DIR}`, `ids.core.labels.remap_labels`.
- Produces: `tests/test_preprocessor_parity.py::test_mlp_indist_fpr_matches_artifact`.

- [ ] **Step 1: Write the failing test** (asserts MLPClassifier — which uses the saved preprocessor — reproduces in-dist FPR ≈ 0.02; with the stale prep it produces ~1.0 and FAILS)

```python
# tests/test_preprocessor_parity.py
import numpy as np, polars as pl
from ids.runtime.predictor import MLPClassifier
from ids.data.preprocessing import split_random
from ids.core.config import SEED, X_COLUMNS_SELECTED, MODELS_DIR
from ids.core.labels import remap_labels

def test_mlp_indist_fpr_matches_artifact():
    df = pl.read_parquet('data/cic_iot_2023.parquet')
    y34 = df['Label'].to_numpy()
    _, _, te = split_random(y34, np.zeros(len(y34)), SEED)
    test_df = df[te].select(list(X_COLUMNS_SELECTED))
    y_true = np.array([s.lower() for s in remap_labels(y34[te], '2')])
    out = MLPClassifier(MODELS_DIR, split='random', mode='2').predict(test_df)
    pred = np.array([str(l).lower() for l in out['labels']])
    fp = ((y_true == 'benign') & (pred == 'attack')).sum()
    tn = ((y_true == 'benign') & (pred == 'benign')).sum()
    fpr = fp / (fp + tn)
    assert fpr < 0.05, f"MLP in-dist FPR={fpr:.3f} (expected ~0.02). Stale preprocessor?"
```

- [ ] **Step 2: Run it — expect FAIL** (`fpr` ≈ 1.0, stale prep)

Run: `cd <root> && PYTHONPATH=<root> .venv/bin/python -m pytest tests/test_preprocessor_parity.py -v`
Expected: FAIL, `MLP in-dist FPR=1.000`.

### Task 2: Regenerate the correct preprocessor + fold the fix into the harness

**Files:**
- Create: `scripts/regenerate_preprocessor.py`
- Modify: `models/preprocessor_random.joblib` (back up first)
- Modify: `ids/eval/cross_dataset_eval.py` (make `predict()` robust — see step 3)

- [ ] **Step 1: Back up the stale artifact**

Run: `cp models/preprocessor_random.joblib models/preprocessor_random.joblib.stale-bak`

- [ ] **Step 2: Write + run the regenerator** (fit on train split, save over `preprocessor_random.joblib`)

```python
# scripts/regenerate_preprocessor.py
import numpy as np, polars as pl, joblib
from ids.data.preprocessing import split_random, fit_preprocess
from ids.core.config import SEED, X_COLUMNS_SELECTED, LOG_COLUMNS_SELECTED, MODELS_DIR
df = pl.read_parquet('data/cic_iot_2023.parquet')
y34 = df['Label'].to_numpy()
Xall = df.select(list(X_COLUMNS_SELECTED)).to_numpy().astype(np.float32)
tr, va, te = split_random(y34, np.zeros(len(y34)), SEED)
_, _, _, prep = fit_preprocess(Xall, tr, va, te,
    x_columns=list(X_COLUMNS_SELECTED), log_columns=list(LOG_COLUMNS_SELECTED))
joblib.dump(prep, MODELS_DIR / 'preprocessor_random.joblib')
print('regenerated preprocessor_random.joblib')
```
Run: `cd <root> && PYTHONPATH=<root> .venv/bin/python scripts/regenerate_preprocessor.py`

- [ ] **Step 3: Re-run the Task 1 test — expect PASS** (FPR ≈ 0.02)

Run: `PYTHONPATH=<root> .venv/bin/python -m pytest tests/test_preprocessor_parity.py -v`
Expected: PASS.
**GATE 0a:** If still failing, STOP — the saved models may need retraining-with-matched-prep; report and halt.

- [ ] **Step 4: Verify `ids/eval/cross_dataset_eval.predict()` now works** (it loads the now-fixed joblib via the classifiers). Add an assertion test:

```python
# append to tests/test_preprocessor_parity.py
def test_cross_dataset_predict_indist_sane():
    import polars as pl, numpy as np
    from ids.eval.cross_dataset_eval import predict, model_feature_columns
    from ids.data.preprocessing import split_random
    from ids.core.config import SEED
    from ids.core.labels import remap_labels
    df = pl.read_parquet('data/cic_iot_2023.parquet'); y34 = df['Label'].to_numpy()
    _, _, te = split_random(y34, np.zeros(len(y34)), SEED)
    feats = df[te].select(model_feature_columns())
    lab, _ = predict(feats, model='mlp')
    yt = np.array([s.lower() for s in remap_labels(y34[te], '2')])
    fpr = ((yt=='benign')&(lab=='attack')).sum() / (yt=='benign').sum()
    assert fpr < 0.05, f"harness predict() in-dist FPR={fpr:.3f}"
```
Run pytest; expect PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_preprocessor_parity.py scripts/regenerate_preprocessor.py models/preprocessor_random.joblib
git commit -m "fix: regenerate matched preprocessor; add in-dist FPR parity tests"
```

### Task 3: Verify serving (live demo) no longer predicts all-attack

- [ ] **Step 1:** Identify the demo entrypoint (search: `grep -rni "gradio\|demo\|app.py\|launch(" --include=*.py | head`). Run it headless or call the same `MLPClassifier.predict` on a handful of benign parquet rows.
- [ ] **Step 2:** Confirm benign rows classify mostly benign (not all-attack). **GATE 0b:** if still all-attack, STOP + report (the deployed model itself may be mismatched, needs Phase 2 retrain first).
- [ ] **Step 3:** Update `cross-dataset-iot23-fails` / `preprocessor-serving-bug` memory notes: bug fixed for the existing models, demo verified.

---

## Phase 1 — Benign augmentation data

### Task 4: Locate + load + dedup the downloaded benign CSVs

**Files:**
- Create: `scripts/build_augmented_parquet.py`
- Read: benign CSV(s) (location TBD), `data/cic_iot_2023.parquet`
- Create: `data/cic_iot_2023_benignaug.parquet`

- [ ] **Step 1: Find the benign CSVs**

Run: `ls -lh ~/Downloads/*.csv 2>/dev/null; find ~/Downloads -iname '*benign*' 2>/dev/null; find ~/Downloads -iname '*.csv' -size +10M 2>/dev/null | head`
Record the path(s). **GATE 1a:** if no benign CSV is found, STOP + report ("benign CSVs not located — cannot augment").

- [ ] **Step 2: Inspect columns + map to the 25 model features**

Read the CSV header. The model needs exactly `X_COLUMNS_SELECTED` (25 cols) + a `Label`. Build a rename map from CSV column names → `X_COLUMNS_SELECTED` (names should match the dataset's CIC feature names; reconcile any whitespace/case differences against the parquet's columns as ground truth). Verify all 25 are present and numeric.
**GATE 1b:** if any of the 25 features cannot be sourced from the CSV, STOP + report the missing columns.

- [ ] **Step 3: Dedup + count unique benign**

```python
# core of scripts/build_augmented_parquet.py
import polars as pl
from ids.core.config import X_COLUMNS_SELECTED
COLS = list(X_COLUMNS_SELECTED)
benign = pl.read_csv("<BENIGN_CSV_PATH>")        # add rename map as needed
benign = benign.select(COLS).unique()             # dedup on the 25 features
print("unique benign rows:", benign.height)
```
**GATE 1c:** if `unique benign` ≲ 250,000 (i.e. barely above the current 200k), STOP + report — the augmentation buys too little to justify a retrain. Otherwise continue.

- [ ] **Step 4: Build the augmented parquet** (unchanged attack rows + all unique benign)

```python
full = pl.read_parquet('data/cic_iot_2023.parquet')
attack = full.filter(pl.col('Label') != 'BenignTraffic')   # confirm the exact benign label string first!
benign = benign.with_columns(pl.lit('BenignTraffic').alias('Label')).select(attack.columns)
aug = pl.concat([attack, benign], how='vertical')
aug.write_parquet('data/cic_iot_2023_benignaug.parquet')
print("attack:", attack.height, "benign:", benign.height, "total:", aug.height)
```
NOTE: confirm the exact benign label string in the parquet first (`full['Label'].unique()` — it is `Benign` or `BenignTraffic` etc.). **GATE 1d:** benign fraction should now be ~15–35%; print it.

- [ ] **Step 5: Commit the scripts (not the data)**

```bash
git add scripts/build_augmented_parquet.py
git commit -m "feat: build benign-augmented parquet (attack unchanged + all unique benign)"
```

---

## Phase 2 — Retrain (2-class AND 8-class) on augmented data

### Task 5: Point training at the augmented parquet and retrain

**Files:**
- Read/Modify: `ids/training/` entry (inspect `run_training` first), `ids/core/config.py` (data path)
- Create: `models/` augmented artifacts (write to a SUFFIXED path to preserve baselines — see step 1)

- [ ] **Step 1: Inspect the training entrypoint** — `grep -rni "def run_training\|read_parquet\|cic_iot_2023" ids/training ids/core` — learn how it loads data, splits (must use `split_random`, `SEED`), fits the preprocessor, and where it saves models + preprocessor. Confirm it **saves the preprocessor alongside the models in the same run** (this is what drifted before — they must be saved together).

- [ ] **Step 2: Retrain 2-class** on `data/cic_iot_2023_benignaug.parquet`, `max_depth=25` (from `hparams.json`), class-weighted CE, single random split. Save augmented RF + MLP + preprocessor + label encoder + `run_artifacts` under an `_aug` suffix or a separate dir so the depth-25 baselines are preserved.
**GATE 2a:** training completes without error; print final val macro-F1.

- [ ] **Step 3: Retrain 8-class** identically (mode `'8'`). **GATE 2b:** completes; print val macro-F1.

- [ ] **Step 4: In-dist sanity** — compute augmented-model CIC test FPR/TPR (2-class) using the SAME method as the baselines (correct prep is now the one saved in this run). **GATE 2c:** MLP in-dist FPR should be sane (< ~0.10) and TPR should not collapse (> ~0.85). If TPR collapsed, the benign shift over-corrected — record it, continue to Phase 3 (it's still a valid datapoint), but flag prominently.

- [ ] **Step 5: Commit** (models only if not too large; otherwise note them as gitignored and commit the run-config/metrics).

```bash
git add hparams.json docs/experiments/  # + any small metric files
git commit -m "feat: retrain 2-class + 8-class on benign-augmented data (depth 25)"
```

---

## Phase 3 — A/B cross-dataset eval + second-capture validation

### Task 6: Re-run IoT-23 with the augmented model (primary A/B)

**Files:**
- Use: `ids/eval/iot23_align.py` (with the augmented model + its preprocessor)

- [ ] **Step 1:** Score `data/iot-23/honeypot-4-1/honeypot-4-1.pcap` (benign-only) and `data/iot-23/.../34-1` (mixed) with the augmented RF + MLP, window=10, using the augmented run's preprocessor. Record FPR (and TPR on mixed).
- [ ] **Step 2:** Build the A/B table vs the locked baselines (honeypot-4-1: baseline RF 0.999 / MLP 0.994). **The headline question: did benign FPR drop?**

### Task 7: Second cross-dataset capture (validation)

**Files:**
- Create: `data/iot-23/honeypot-5-1/` (download)

- [ ] **Step 1: Download a SECOND benign honeypot + a different malware capture** (validates the pattern isn't 4-1/34-1-specific). Individual files (no 21 GB tarball) from:
  `https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-Honeypot-Capture-5-1/`
  — fetch `*.pcap` (or `*.pcap.xz`, decompress with `python -c "import lzma,shutil;shutil.copyfileobj(lzma.open('f.xz'),open('f','wb'))"`) and `bro/conn.log.labeled`. (List the folder first to get exact filenames; sizes are small.)
- [ ] **Step 2:** Score it with BOTH the baseline (depth-25) and augmented models → confirm the direction of the FPR change replicates on an unseen capture. **GATE 3a:** if the second capture contradicts the first, report the discrepancy plainly (do not pick the favorable one).

---

## Phase 4 — Record + commit

### Task 8: Write the corrected/augmented experiment record

- [ ] **Step 1:** Create `docs/experiments/2026-06-25-benign-augmentation-ab.md` with: setup, the locked baselines, augmented in-dist FPR/TPR, the IoT-23 A/B table (4-1 + 34-1 + the 5-1 validation), and an honest conclusion — *did more benign reduce OOD false positives, and at what cost to attack recall?* If it didn't move, say so (that means the failure is broad distribution shift, not benign starvation — still a real finding).
- [ ] **Step 2:** Create `docs/experiments/results/benign_augmentation_ab.json` (machine-readable, mirror the prior cross-dataset JSON schema).
- [ ] **Step 3:** Update memory: revise `cross-dataset-iot23-fails` with the A/B outcome; mark `preprocessor-serving-bug` fixed (or note residual). Keep numbers exact, no spin.

### Task 9: Final commit (everything except literature + data)

- [ ] **Step 1:** Stage everything except `docs/literature/` and `data/`:

```bash
git add -A
git reset docs/literature/ data/
git status   # verify literature/ and data/ are NOT staged
```
Include the pre-existing modified files (`.gitignore`, `docs/report/figures/*.png`, `ids_pipeline.ipynb`, `slides_ml_data.md`) per "commit everything except literature".

- [ ] **Step 2: Commit + push**

```bash
git commit -m "experiment: benign-augmentation A/B + preprocessor fix + cross-dataset validation"
git push origin <branch>
```
**GATE 4a:** print final `git log --oneline -5` and the morning summary: baselines vs augmented FPR/TPR for both captures, plus any gate that halted.

---

## Morning report (what the user wakes up to)

A single summary at the top of `docs/experiments/2026-06-25-benign-augmentation-ab.md`:
- ✅/⛔ each phase (or where it STOPPED + why).
- Unique-benign count found.
- A/B table: baseline vs augmented IoT-23 FPR (honeypot-4-1, validation 5-1) + in-dist TPR (regression check).
- One honest sentence: did augmentation fix the OOD false positives, partially, or not at all.

## Self-review notes (done)
- Every gate halts rather than fabricating (spec'd in header + each GATE).
- Preprocessor fix precedes any new numbers (Phase 0 before 2/3).
- One variable held (depth=25); benign is the only change.
- Baselines are frozen constants in this doc; second capture guards against single-capture flukes.
- Open unknowns flagged as discovery gates (benign CSV location, exact benign Label string, training entrypoint, demo entrypoint) — the worker investigates and STOPS if blocked.
