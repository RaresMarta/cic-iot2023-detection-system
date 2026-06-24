# Experiment — Cross-dataset transfer of the CIC-IoT-2023 model to IoT-23

- **Date:** 2026-06-24
- **Status:** complete (v2 — corrected after a preprocessor bug; see "Correction")
- **Type:** out-of-distribution generalization probe (inference only, no retraining)
- **Result in one line:** the model is **healthy in-distribution** (MLP FPR ~2%) but flags **~99% of genuinely-benign IoT-23 traffic as attack** → poor cross-dataset transfer / testbed overfitting.

## ⚠ Correction (read first)

The **first version of this experiment was invalid.** It scored via the harness `predict()` path, which loads `models/preprocessor_random.joblib` — a **stale preprocessor that does not match the saved models**. That mis-scaled every input, making the model output *all-attack on everything, including CIC in-distribution data* (FPR 100%) — impossible for a real model, which is what exposed the bug. All "100% / all-attack / not-threshold-recoverable" claims from v1 are **retracted**. This v2 re-runs with the **training-matched preprocessor** (a fresh `fit_preprocess` on the train split, which reproduces the saved-artifact in-dist FPR of 1.8%). The stale preprocessor is a separate serving bug — see end.

## Question

Does the 2-class (benign-vs-attack) CIC-IoT-2023 model generalize to a *different* IoT network (IoT-23), or has it overfit to the CIC testbed?

## Setup

- **Model (frozen, inference only):** RF `ids_rf_random_2class.joblib`, MLP `ids_dnn_random_2class.pth`; encoder `['Attack','Benign']`. **Preprocessor: fresh `fit_preprocess` on the CIC train split** (NOT the stale joblib).
- **External data (IoT-23, unseen):** `CTU-IoT-Malware-Capture-34-1` (mixed: DDoS+C&C+scan+benign); `CTU-Honeypot-Capture-4-1` (benign-only).
- **Harness:** `ids/eval/iot23_align.py` — reproduces the extractor's per-host-pair tumbling windows + joins to Zeek `conn.log.labeled` (window=attack if any overlapping connection is Malicious). `window=10`. Feature-parity gate: PASS (25/25). Alignment: 0 dropped.

## Results (corrected, attack = positive)

### CIC test set — in-distribution sanity

| Model | TPR (recall) | FPR | Benign correct |
|---|--:|--:|--:|
| MLP | 0.902 | **0.021** | 29,383 / 30,000 |
| RF | 0.971 | 0.351 | 19,471 / 30,000 |

MLP FPR 0.021 matches the saved artifact (0.018) → preprocessor correct, model healthy. RF is notably false-positive-prone even in-distribution.

### IoT-23 honeypot-4-1 — benign-only (1,699 windows, 0 attack)

| Model | FPR | Benign correct |
|---|--:|--:|
| RF | **0.9994** | 1 / 1,699 |
| MLP | **0.9941** | 10 / 1,699 |

### IoT-23 34-1 — mixed (2,614 benign / 20,248 attack)

| Model | FPR | TPR | Benign correct |
|---|--:|--:|--:|
| RF | 0.9973 | 0.9996 | 7 / 2,614 |
| MLP | 0.9805 | 0.6589 | 51 / 2,614 |

## Interpretation

The model is fine in-distribution (MLP FPR ~2%) yet flags ~99% of foreign benign traffic as attack. The **2% → 99% gap** is valid evidence of overfitting to the CIC testbed: the learned "benign" region does not cover a different network's normal traffic. MLP is the clean exhibit (2%→99%); RF is a weaker basis (35% in-dist FPR). On the mixed capture the MLP is not literally all-attack (attack recall drops to 0.66) — it just gets the benign side almost entirely wrong.

## Scope / caveats

- One external dataset (IoT-23); supports "fails to transfer to IoT-23", not "never transfers".
- `window=10` pins `Number`≈10 (known leak); failure is benign-side and consistent across RF + MLP.
- v1's all-attack numbers were preprocessor-bug artifacts — do not cite them.

## Reproduction

Corrected eval (training-matched preprocessor): `scripts`-equivalent in session scratchpad `corrected_eval.py`; key step is `fit_preprocess` on the CIC train split, then `prep.transform` on extractor output before `model.predict`. The harness `ids.eval.cross_dataset_eval.predict()` is currently BUGGED (loads the stale preprocessor) — do not use until the preprocessor is regenerated.

## Separate bug discovered

`models/preprocessor_random.joblib` is stale/mismatched with the saved models (RobustScaler centers ~2× off). `ids/runtime/predictor.py` (shared train/serving path) loads it → **live demo would mis-scale and predict all-attack.** Fix: regenerate the preprocessor to match the deployed models, verify serving reproduces ~2% in-dist FPR.

## Follow-ups

- Re-run this identical IoT-23 baseline after the benign-augmentation retrain → does ~99% FPR drop?
- Fix the preprocessor serving bug (independent priority).
