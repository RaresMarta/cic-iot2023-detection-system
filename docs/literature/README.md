# CIC-IoT-2023 literature (read & audited)

Papers experimenting on CIC-IoT-2023, archived here for traceability. Audited 2026-06 for: (a) headline metric, (b) feature selection, (c) treatment of the `Number` feature, (d) whether they flag the per-class packet-window leakage. **None of the 8 flag the window leakage.** See `../../.claude/CLAUDE.md` "Known data caveat" and the memory notes for the distilled findings.

## SOTA on the *honest* metric (macro-F1)
- **Binary:** ~0.99 (saturated, partly trivial / leakage-aided).
- **8-class:** ~0.77–0.89 — honest frontier.
- **34-class:** ~0.74–0.89.
- Many "99%+" headlines are inflated by **SMOTE-before-split** (synthetic minority rows leak into test) or are accuracy/weighted-F1 (not macro-F1) on imbalanced data. Only train-only-fit pipelines (Almahaqeri, our project) give trustworthy macro-F1.
- Web/Recon classes are the universal weak point (F1 ~0.6–0.72), misclassified as Benign — transport-layer flow stats can't see application-layer attacks. Confirmed across papers + our own results.

## Papers

| File | Citation | Task | Headline | Feature selection | `Number` | Flags window? |
|------|----------|------|----------|-------------------|----------|---------------|
| `almahaqeri-2026-xgboost-scireports.pdf` | Almahaqeri et al., *Sci. Reports* 2026, 16:16909. doi 10.1038/s41598-026-47399-5 | bin/8/34 | macro-F1 **0.995 / 0.890 / 0.888** (XGBoost) | gain-based 46→23 | **kept**; top in binary, drops in multiclass | no (only preprocessing leakage) |
| `tseng-2024-transformer.pdf` | Tseng, Wang & Wang, *Future Internet* 2024, 16:284. doi 10.3390/fi16080284 | bin/8 | bin ~99.5%, 8-cls ~92.5% acc | none ("all features") | kept (all 46, #40) | no |
| `jony-arnob-2024-lstm.pdf` | Jony & Arnob, *J. Edge Computing* 2024, 3(1):28–42. doi 10.55056/jec.648 | 34 | 98.75% acc / 0.986 F1 | none | kept (all) | no |
| `modi-2024-xgboost.pdf` | Modi, arXiv:2408.10267, 2024 | binary | 97.64% (CICIoT2023) | hybrid filter 46→18 | **kept; ranks #4** | no |
| `hajjouz-2024-catboost.pdf` | Hajjouz & Avksentieva, *Data & Metadata* 2024, 3:577. doi 10.56294/dm2024577 | 20 subtypes | 99.96% | Spearman+hierarchical →23 | kept; **near bottom** | no |
| `aelstmcnn-2025-sensors.pdf` | (AE-LSTM-CNN hybrid), *Sensors* 2025, 25(2):580. doi 10.3390/s25020580 | 8 | F1 99.19% | none (all 46) | kept | no — ⚠️ **SMOTE before split** (optimistic) |
| `gnn-neuralode-2025-ieeeaccess.pdf` | Hybrid GNN + Neural-ODE, *IEEE Access* 2025 (accepted/unedited) | 5 (custom) | F1 98.0% | TOA optimizer | unclear | no — custom classes + SMOTE; not directly comparable |
| `khanday-2024-etcnn-ijmems.pdf` | Khanday, Fatima & Rakesh, *IJMEMS* 2024, 9(1):188–204. doi 10.33889/IJMEMS.2024.9.1.010 | 34 | 99.87% (1D-CNN) | ExtraTrees → top 20 | **dropped** (low Gini, 34-cls) | no — ⚠️ SMOTE before split; macro-F1 1.00 implausible |

## Not in this folder
- **Official dataset paper:** Neto et al., *CICIoT2023*, Sensors 2023, 23(13):5941 (PMC10346235) — documents the 10/100 per-class windowing and defines `Number` = "number of packets in the flow". Read via web, not archived. Baseline: RF binary 99.68% acc, 8-cls 99.43%, 34-cls 99.16% (accuracy; macro-F1 lower).
- Two XAI papers we examined are NOT CIC-IoT-2023 and remain in `~/Downloads`: Patil et al. (Electronics 2022, CIC-IDS-2017) and Sharma et al. (ESWA 2024, NSL-KDD+UNSW-NB15 — the paper actually cited in the thesis as `Almohri2024XAI`, a mislabeled bib entry).
