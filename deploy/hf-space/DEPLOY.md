# Deploying the analyzer to Hugging Face Spaces

This folder is a self-contained **Docker Space**: `Dockerfile` + `requirements.txt` +
`README.md` (HF frontmatter, `app_port: 7860`) + `ids/` + a trimmed `models/` + the
`data/cic_iot_2023.parquet` flow pool the SHAP explainer samples.

Verified locally: `docker build` succeeds and `/api/classify` returns correct results for
`mlp/random/2`, `mlp/random/8`, and `rf/random/2`, identical to the native dev server.

## What ships (and what doesn't)

- Included: MLP 2-class + 8-class, RF 2-class (165 MB), parquet (145 MB) for SHAP. ~311 MB.
- Excluded: `ids_rf_random_8class.joblib` (1.3 GB) — too large for the free CPU tier and
  an OOM risk. Requesting `rf` + `8` returns a clean "model not found". To add it back,
  copy it into `models/` and re-upload.

## Push it (needs YOUR Hugging Face account)

The `hf` CLI lives in pyenv's 3.11.9. Run these from the repo root. Replace `USER`.

```sh
HF=~/.pyenv/versions/3.11.9/bin/hf

# 1. log in with a WRITE token from https://huggingface.co/settings/tokens
$HF auth login

# 2. create the Docker Space (SDK is required for spaces)
$HF repo create rt-ids-analyzer --repo-type space --space-sdk docker

# 3. upload this folder as the Space root (HTTP upload — no git-lfs needed)
$HF upload USER/rt-ids-analyzer ./cic-iot2023-detection-system/deploy/hf-space . \
    --repo-type space \
    --commit-message "Deploy RT-IDS analyzer backend"
```

HF then builds the image (a few minutes). Watch the build log on the Space page.

## Wire up the frontend

Once live, the Space serves at `https://USER-rt-ids-analyzer.hf.space`. Point the frontend
at it (Vercel env var, or `ids-frontend/.env`):

```
VITE_API_URL=https://USER-rt-ids-analyzer.hf.space
```

Sanity check after build: `curl https://USER-rt-ids-analyzer.hf.space/api/health`.
