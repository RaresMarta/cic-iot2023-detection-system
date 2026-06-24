# Benign-Augmentation A/B + Preprocessor Fix — Experiment Record

_Run 2026-06-24 (overnight autonomous). All numbers below are measured, not estimated._

## Morning report

- ✅ **Phase 0 (preprocessor fix):** stale `preprocessor_random.joblib` regenerated; in-dist MLP FPR restored 1.00 → 0.02. Also found and fixed a **second** train/serve bug: `load_dataset` double-applied `log1p`. Both guarded by tests (all pass).
- ✅ **Phase 1 (augmentation data):** benign label string is `Benign_Final` (the plan had guessed `BenignTraffic`). **1,093,903 unique benign** rows located/deduped (5.5× the 200k cap). Augmented parquet built at **20.5% benign**.
- ⚠️ **Phase 2 (retrain):** 2-class fully retrained + saved (MLP val macro-F1 **0.9013**, RF test macro-F1 **0.9165**). 8-class MLP trained (val macro-F1 0.6299) but the **8-class RF save crashed (disk full, errno 28) — deferred** to a later retrain by user decision. The A/B is 2-class only, so this does not affect the result.
- ✅ **Phase 3 (A/B + 2nd capture):** scored honeypot-4-1, 34-1, and a freshly downloaded honeypot-5-1.

**One honest sentence:** More benign data did **not** fix the out-of-distribution false positives — in-distribution FPR improved, but on IoT-23 the benign FPR stayed near 1.0 on two captures and got **worse** on the third (honeypot-5-1 MLP 0.369 → 0.999), so the OOD failure is **broad distribution shift, not benign starvation.**

---

## A/B table — IoT-23 benign false-positive rate (window = 10)

Baselines reproduced exactly from the locked depth-25 table (RF/MLP honeypot-4-1 0.999/0.994, 34-1 0.997/0.981), confirming the harness is consistent and the augmented numbers are directly comparable.

| Capture | Type | Model | Baseline FPR | Augmented FPR | Direction |
|---|---|---|--:|--:|---|
| honeypot-4-1 | benign-only (1699) | RF  | 0.9994 | 0.9994 | no change |
| honeypot-4-1 | benign-only (1699) | MLP | 0.9941 | 0.9959 | slightly worse |
| 34-1 mixed | 2614 ben / 20248 atk | RF  | 0.9973 | 0.9946 | ~no change |
| 34-1 mixed | 2614 ben / 20248 atk | MLP | 0.9805 | 0.9621 | marginally better, still ~0.96 |
| honeypot-5-1 | benign-only (39760) | RF  | 0.8274 | 0.9841 | **worse** |
| honeypot-5-1 | benign-only (39760) | MLP | 0.3688 | 0.9987 | **much worse** |

### Attack recall (TPR) on the mixed capture (regression check)

| Capture | Model | Baseline TPR | Augmented TPR |
|---|---|--:|--:|
| 34-1 mixed | RF  | 0.9996 | 0.9996 |
| 34-1 mixed | MLP | 0.6589 | 0.9244 |

Attack recall did **not** collapse — the augmented MLP's 34-1 TPR actually rose (0.66 → 0.92). The cost of augmentation was not lost recall; it was lost benign generalization on unseen captures.

### In-distribution CIC test (sanity — GATE 2c)

| Model | Baseline FPR/TPR | Augmented FPR/TPR |
|---|--:|--:|
| MLP | 0.021 / 0.902 | 0.0169 / 0.9143 |
| RF  | 0.351 / 0.971 | 0.0333 / 0.9348 |

In-distribution, augmentation **helped** — RF in-dist FPR dropped ~10× (0.351 → 0.033). This is the key tension: the model fit the source benign distribution *better* (lower in-dist FPR) while generalizing to foreign benign traffic *worse*. That is the signature of overfitting to the source benign distribution, not of curing OOD false positives.

---

## Conclusion

The hypothesis — that the ~99% OOD benign FPR is caused by benign starvation (only 200k benign in training) — is **rejected**. Adding 5.5× more unique benign rows:

- improved in-distribution benign FPR (0.351 → 0.033 for RF), and
- improved attack recall on the mixed IoT-23 capture (MLP 0.66 → 0.92), but
- did **not** reduce, and in one capture **substantially increased**, the false-positive rate on IoT-23 benign traffic.

The two benign-only captures disagree on the *baseline's* difficulty (honeypot-4-1 baseline MLP FPR 0.994 vs honeypot-5-1 0.369), but they **agree on the direction of the augmentation effect: it does not help, and where there was room to move (5-1), it hurt.** The IoT-23 false positives therefore stem from a genuine distribution gap between CIC-IoT-2023 and IoT-23 traffic — different devices, capture conditions, and the CIC window-feature construction — rather than from too few benign examples. More of the same-source benign data cannot close that gap; bridging it would require either IoT-23-domain benign data or domain-adaptation, which is future work.

## What was fixed along the way

1. **Stale preprocessor** (`models/preprocessor_random.joblib`): regenerated via `scripts/regenerate_preprocessor.py` (re-fit on the SEED train split). In-dist MLP FPR 1.00 → 0.02. Live-serving path (shared `predictor.py`) verified on 500 benign rows: MLP 97.6% benign, RF 62.6% — no longer all-attack.
2. **`load_dataset` double-`log1p`**: training applied `log1p` + inf→null, then the `Preprocessor` applied them again, so headless training learned on `log1p(log1p(x))` while serving applied `log1p(x)`. Removed the duplicate transforms; the `Preprocessor` is now the sole owner. Guarded by `tests/test_load_dataset_raw.py`.
3. **Regression guards added**: `tests/test_preprocessor_parity.py` (in-dist FPR via both the serving classifier and the cross-dataset harness) and `tests/test_load_dataset_raw.py`. All pass.

## Reproduction

```bash
cd cic-iot2023-detection-system
PYTHONPATH=$PWD .venv/bin/python scripts/regenerate_preprocessor.py      # fix preprocessor
PYTHONPATH=$PWD .venv/bin/python scripts/build_augmented_parquet.py      # build augmented parquet
PYTHONPATH=$PWD .venv/bin/python scripts/train_augmented.py              # retrain -> models_aug/ (8-class RF needs disk headroom)
PYTHONPATH=$PWD .venv/bin/python scripts/ab_cross_dataset.py            # A/B over the 3 captures
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_preprocessor_parity.py tests/test_load_dataset_raw.py -v
```

## Known incompleteness

- **8-class augmented RandomForest**: not saved (disk full during `joblib.dump`, errno 28). The 8-class MLP is saved (val macro-F1 0.6299). Re-run `scripts/train_augmented.py` with ≥3 GB free to complete it. It is **not** needed for the benign-FPR A/B above (cross-dataset eval is 2-class only).
- Augmented artifacts live in `models_aug/` (gitignored); the depth-25 baselines in `models/` were left untouched.
