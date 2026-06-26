# `ids.runtime` — inference layer

Runtime components shared by the analyzer REST API (`ids/apps/analyzer`) and the live
monitor (`ids/apps/monitor`). Turns raw input (pcap or CIC-format CSV) into model verdicts,
using the **same preprocessing the model was trained with**. This is a library, not an app.

## Modules

- `extractor.py` — faithful pcap → CIC-IoT-2023 feature extractor. Reproduces the dataset
  authors' **dpkt** packet-window method (tumbling windows over unordered host pairs — 10
  packets for most classes, 100 for floods), **not CICFlowMeter**: CICFlowMeter's
  flow-timeout / Fwd-Bwd feature set mismatches the 25 features the model uses. Entry point:
  `extract_features(pcap_path, window=10)`.
- `predictor.py` — inference backends sharing one preprocessing path and output contract, so
  callers can swap by key:
  - `MLPClassifier` — the PyTorch MLP, with temperature-scaled (calibrated) confidence.
  - `RFClassifier` — the RandomForest baseline (`predict_proba`).
- `explain.py` — `FlowExplainer`: per-flow SHAP attributions (`GradientExplainer`) for the
  live demo, built once over a small stratified background sample.
- `validate_extractor.py` — extractor validation. No args → mechanical self-test (synthetic
  SYN flood + benign HTTPS through extractor+model); with a pcap path → distributional check
  against real traffic.

Models load from `../models/` (`MODELS_DIR`). Run the training notebook end-to-end first so
the scaler, encoders, and checkpoints exist.

## How it's served

- **`ids/apps/analyzer/app.py`** — FastAPI REST backend (`POST /api/classify`,
  `GET /api/health`) consumed by the React frontend. Run: `python -m ids.apps.analyzer.app`
  (port 7860). This is what deploys to HuggingFace Spaces.
- **`ids/apps/monitor/`** — the live beside-path NIDS: capture → window → detect → SSE feed.

## Live attack scripts (demo loop)

For the end-to-end demo (trigger attack → captured → classified), attack scripts target a
controlled local victim — the `website` (mock site) container pinned at `172.30.0.10` on the
`idsnet` Docker bridge (see `deploy/docker-compose.yml`); the detector sniffs that bridge
(`ids-br0`). Suggested tools per family:

- **DDoS / DoS / Mirai floods** — `hping3`, `t50`, `iperf3` UDP storms
- **Recon** — `nmap` (SYN / OS scan), `fping` sweeps
- **Web** — `curl` payloads, `sqlmap` (SQL injection), `slowhttptest` (Slowloris)
- **Spoofing** — `arpspoof`, `dnschef`
- **BruteForce** — `hydra` against a local dummy SSH

These are environment-specific (interface, target, privilege level) — defense-day
scaffolding, not something a generic script produces.
