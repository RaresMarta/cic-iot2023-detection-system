# Project context — RT-IDS for CIC-IoT-2023

Bachelor thesis project: build, evaluate, and demo a real-time intrusion detection system for IoT networks using the CIC-IoT-2023 dataset.

## Goal

Two deliverables:

1. **Pipeline** — defensible PyTorch MLP baseline with reproducible training, evaluation, and benchmarking on CIC-IoT-2023.
2. **Live demo** — end-to-end app where the user triggers attacks, traffic is captured, features are extracted, and the model classifies flows in near-real-time.

## Stack

- **Data**: Polars (lazy + parquet, Zstd-compressed)
- **Model**: PyTorch MLP (39→128→64→n) with BatchNorm + Dropout
- **Preprocessing**: sklearn `RobustScaler`, `LabelEncoder`, `compute_class_weight`
- **Demo flow extraction**: CICFlowMeter (Java tool from the CIC team — do NOT reimplement in Python)
- **Demo UI**: Gradio (lowest-friction option for thesis demo)
- **Hosting**: self-host on user's PC during defense; Cloudflare Tunnel if a public URL is needed

## Dataset

- **Release**: 39-feature CSV version from the official UNB CIC website. The 46-feature version most published papers use was **not available** from the official source — the 39-feature scope is a deliberate decision, not an oversight. Acknowledge in thesis methodology.
- **Size**: ~46M raw flow rows across 34 attack folders + Benign.
- **Known issue**: ~24M exact duplicate rows in the raw data (CIC-IoT-2023 literature notes this consistently). Pipeline must `unique()` before sampling.
- **Severe imbalance**: largest classes ~7M+ rows, smallest (Uploading_Attack) ~1.2k rows. Handled via per-class cap + class-weighted CE.
- **Sampling**: random per-class sample with seed (`groupby().sample(n=200_000, seed=SEED)`), NOT `lf.head()`. Earlier rows of each attack folder come from a single capture session; head-slicing biases the model toward that session's quirks.

## Classification granularities (invariant)

All three are required — most CIC-IoT-2023 papers report on this taxonomy and the comparison is part of the thesis contribution:

- **2-class** — Benign vs Attack (inline real-time gate)
- **8-class** — Attack families (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce, Benign)
- **34-class** — Specific attack variants

Same train/val/test rows used across all three (labels remapped per mode).

## Split methodology

Three split variants reported, with **temporal as the headline result**:

1. **Temporal (headline)** — sort source CSVs by filename per folder, train = earliest 70%, val = 15%, test = latest 15%. Mirrors deployment (train on past, test on future). Catches concept drift.
2. **Per-CSV hold-out** — `GroupShuffleSplit` with source CSV as group ID, stratified by 34-class label. Prevents within-session row leakage but ignores temporal order.
3. **Random row** — original `train_test_split`, kept for parity with published numbers.

Expected gap: temporal F1 will be the lowest of the three by 2–10 points. That gap is the honest generalization story — frame it as a finding, not a failure.

## "Real-time" framing

Narrower than the literature default. The demo is **interactive single-request inference**, not line-rate gateway protection.

- **Claim**: per-flow inference latency suitable for interactive demo use (target <100ms end-to-end including feature extraction, scaling, forward pass).
- **Out of scope**: line-rate (10 Gbps / >1M flows/sec) deployment. Do not claim this.
- **Benchmark must report**: latency at batch sizes [1, 32, 256, 1024], p50/p95/p99 over 10k runs, throughput (flows/sec), end-to-end timing including `scaler.transform`. NOT mean-only single-batch.

## Demo architecture

Pipeline (each step visible in the UI, not hidden):

```
attack script → tcpdump capture → PCAP → CICFlowMeter → CSV (39 features)
              → RobustScaler → MLP → verdict per flow → Gradio dashboard
```

- Attack scripts use existing tools: `hping3` (SYN/ICMP floods), `nmap` (port scan), `slowhttptest` (Slowloris), curl scripts (web attacks). One script per representative attack.
- Web-family attacks (SQL injection, XSS, command injection) will detect poorly from flow stats alone — this matches the dataset paper's findings and is a feature, not a bug. Discuss in thesis.

## Conventions and decisions already made

- **Per-class cap**: 200k random rows (high end of literature range; defensible).
- **Imbalance handling**: class-weighted CE only. No SMOTE (memory + meaningless synthetic flows). No `WeightedRandomSampler` (redundant with weighted CE).
- **Evaluation metrics**: accuracy, macro-F1, weighted-F1, macro-precision/recall, per-class report, normalized confusion matrix. No ROC-AUC.
- **Tree baselines required**: RandomForest + XGBoost on same splits. Committee will ask if a tree beats the MLP — answer it preemptively.
- **No transformers, no graph nets, no RL.** Clean MLP baseline first.

## Files and structure

- `ids_pipeline.ipynb` — main pipeline (ingestion → training → evaluation → benchmark)
- `data/CSV/CSV/` — raw CIC-IoT-2023 CSVs, one folder per attack class
- `data/cic_iot_2023.parquet` — preprocessed parquet (regenerated when ingest changes)
- `models/` — saved scaler, label encoders, model state dicts
- `sampling_research.md` — survey of how other CIC-IoT-2023 projects handle per-class sampling
