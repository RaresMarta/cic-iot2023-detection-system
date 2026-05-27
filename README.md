# RT-IDS for CIC-IoT-2023

Bachelor thesis project: a real-time intrusion detection system for IoT networks, trained on the CIC-IoT-2023 dataset and exposed through a cyberpunk-themed web interface.

## Architecture

```
Frontend (React + Vite)          → Vercel
Backend  (FastAPI + Gradio)      → HuggingFace Spaces
Database (Auth + history)        → Supabase (PostgreSQL)
Experiment tracking              → Weights & Biases
```

**Live URLs:**
- Frontend: https://ids-frontend-five.vercel.app
- Backend API: https://baresman-ids-backend.hf.space/api/health
- Frontend repo: https://github.com/RaresMarta/ids-frontend

## What's in here

| Path | Purpose |
| --- | --- |
| `ids_pipeline.ipynb` | Full ML pipeline — ingestion, EDA, feature selection, splits, Optuna tuning, MLP + tree baselines, wandb logging, latency benchmark |
| `config.py` | Feature columns (39 raw → 25 after selection), training hyperparameters, split/mode config |
| `models.py` | Flexible IDSModel (variable depth, activation, optimizer), AMP training, Optuna pruning support |
| `preprocessing.py` | Three split strategies (temporal / per-CSV / random), median imputation, RobustScaler |
| `labels.py` | 34→8→2 class label mappings |
| `demo/` | FastAPI + Gradio inference server. `POST /api/classify` serves the React frontend |
| `models/` | Saved scaler, label encoders, MLP state dicts per (split, granularity) |
| `docs/report/` | Thesis report (LaTeX) |

## Feature Selection

Started with 39 CIC-IoT-2023 features. Reduced to **25** via two EDA steps documented in the notebook:

1. **Variance check** — dropped 12 near-constant binary flags (Var < 0.01)
2. **Pearson correlation** — dropped 2 mathematical duplicates (`Variance` = Std², `Tot size` ∝ `AVG`)

## Classification Granularities

| Mode | Classes | Use case |
|------|---------|----------|
| 2-class | Benign / Attack | Binary gate, real-time blocking |
| 8-class | Attack families (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce, Benign) | Operational routing, alert triage |

34-class (specific attack variants) is out of scope — see Methodological Decisions in the notebook.

## Split Strategies

| Strategy | Description | Role |
|----------|-------------|------|
| **Temporal** *(headline)* | Per-folder: earliest 70% CSVs → train, next 15% → val, latest 15% → test | Mirrors deployment: train on past, test on future |
| Per-CSV | GroupShuffleSplit on source CSV | Removes within-session leakage, ignores temporal order |
| Random | Stratified row split | Parity with published CIC-IoT-2023 numbers |

## Running the pipeline

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open notebook and run top-to-bottom
jupyter lab ids_pipeline.ipynb
```

Key config knobs in `config.py`:
- `MODES_TO_RUN` — `['2', '8']` for both granularities
- `SPLITS_TO_RUN` — `['temporal']` for headline results
- `MAX_ROWS_PER_CLASS` — 200,000 (subsampling cap per attack class)

## Running the demo locally

```
python -m demo.app
```

Binds at `http://localhost:7860`. Gradio UI at `/`, REST API at `/api/classify`.

The React frontend (`G:/uni/ids-frontend`) points at this via `VITE_API_URL=http://localhost:7860` in `.env`.

## Deploying updates

**Models** (after retraining):
```bash
hf upload baresman/ids-backend models/ids_dnn_temporal_2class.pth models/ids_dnn_temporal_2class.pth --repo-type space
hf upload baresman/ids-backend models/scaler_temporal.joblib models/scaler_temporal.joblib --repo-type space
# repeat for each model file
```

**Code changes:**
```bash
hf upload baresman/ids-backend demo/app.py demo/app.py --repo-type space
hf upload baresman/ids-backend models.py models.py --repo-type space
```

HF Space redeploys automatically on every upload.

## Thesis report

```
cd docs/report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
