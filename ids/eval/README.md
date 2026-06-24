# `ids/eval` — cross-dataset evaluation harness

Score **foreign-dataset pcaps** with the **frozen CIC-IoT-2023 2-class
(benign-vs-attack) model**. Inference only — no retraining. This measures how the
model trained on CIC-IoT-2023 *transfers* to another dataset's traffic.

```
pcap(s) → dpkt window extractor → 25 CIC features → feature-parity gate
        → frozen RobustScaler → frozen RF / MLP → attack score + verdict
        → label_maps.to_binary(foreign labels) → recall / precision / F1 / F2 / CM / PR
```

## Quick start (once you have provisioned pcaps + an aligned label CSV)

```bash
python -m ids.eval.cross_dataset_eval \
    --pcaps  /path/to/bot-iot/pcaps_or_single.pcap \
    --labels /path/to/labels.csv \
    --dataset bot-iot \
    --model  rf            # or: mlp
    # --window 10          # packets per window (dataset default = 10)
```

`--pcaps` accepts a single pcap file **or** a directory; in directory mode every
`*.pcap` / `*.pcapng` / `*.cap` is extracted and concatenated, sorted by filename
for deterministic row order.

### `--labels` CSV schema

One row per extracted feature **window**, in the **same order** the extractor
emits windows, with at least a `label` column carrying the dataset's *raw*
ground-truth label:

```csv
label
Normal
DDoS
DoS
...
```

The harness aligns row *i* of the CSV to window *i* of the feature matrix,
normalises each raw label via `label_maps.to_binary(dataset, raw)`, and compares.
**If the row counts differ it errors loudly** rather than truncating — a mismatch
means the label CSV was not produced from the same windowing as the features.

> **You must produce this aligned CSV yourself.** The extractor windows packets
> by host-pair; foreign datasets label per-flow or per-packet. Lining labels up
> 1:1 with this extractor's windows is a dataset-specific step (label packets,
> then aggregate by the *same* windowing). The harness only consumes the result.

## Which `--dataset` names are wired

Binary path is ready for these (benign vocab known, in `label_maps._BENIGN_ALIASES`):

| `--dataset`   | benign token(s)        |
|---------------|------------------------|
| `ciciot2023`  | `benign`, `benign_final` |
| `bot-iot`     | `normal`               |
| `ton-iot`     | `normal`, `0`          |
| `iot-23`      | `benign`, `-`          |

Any other `--dataset` name still works via the global aliases
(`benign`, `normal`, `background`, `legitimate`, `-`, …) plus the default rule:
**any non-benign label → `attack`**. To add a dataset, add its benign token(s)
to `_BENIGN_ALIASES` — the default rule covers every attack label.

## What is still STUBBED (left for the human)

The **8-class family maps are intentionally empty** in `label_maps.py`:
`BOT_IOT_FAMILY_MAP`, `TON_IOT_FAMILY_MAP`, `IOT_23_FAMILY_MAP` (all `{}`,
marked `TODO(human)`). Mapping a foreign dataset's attack taxonomy onto the
CIC-IoT-2023 8-family scheme (`DDoS / DoS / Mirai / Recon / Spoofing / Web /
BruteForce / Benign`) is a research judgement, not mechanical, and the 8-class
cross-dataset path is out of scope here. The harness currently runs the
**2-class model only**. When the maps are filled, add a `to_family()` resolver
mirroring `to_binary()` and an 8-class predict/evaluate path.

## Public API

- `extract_features(pcap_path_or_dir, window=10) -> polars.DataFrame` — wraps the
  custom dpkt extractor; returns the 25-feature matrix in model column order.
- `check_feature_parity(df) -> dict` — asserts columns == `feature_columns.joblib`
  (names + order). Raises loudly on mismatch. Run automatically inside `predict`.
- `predict(features_df, model='rf'|'mlp') -> (labels, scores)` — `labels` are
  `'benign'`/`'attack'`; `scores` are attack-class probability in `[0,1]`.
- `evaluate(pred_labels, true_labels, scores) -> dict` — recall, precision, F1,
  **F2**, 2×2 confusion matrix, and PR-curve points (precision/recall/thresholds).
- `label_maps.to_binary(dataset_name, raw_label) -> 'benign'|'attack'`.

## Self-test

`tests/test_cross_dataset_eval.py` proves the wiring with **no external data**:
it synthesises pcaps (SYN flood + benign HTTPS), runs the real extractor →
parity gate → `predict` for both RF and MLP, plus an extractor-bypass path with
hand-crafted 25-column rows through `predict` + `evaluate`.

```bash
python tests/test_cross_dataset_eval.py        # needs models/ present
```
