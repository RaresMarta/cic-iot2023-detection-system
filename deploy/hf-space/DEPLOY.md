# Deploying the analyzer to Hugging Face Spaces

This folder holds **only the HF-specific files**: `Dockerfile` + `requirements.txt` +
`README.md` (HF frontmatter, `app_port: 7860`) + `pyproject.toml` + `build.sh`. The code
(`ids/`), models, and SHAP parquet are NOT stored here — `build.sh` stages them from the
canonical repo at deploy time, so the Space can never drift behind `ids/`.

Verified locally: `docker build` succeeds and `/api/classify` returns correct results for
`mlp/random/2`, `mlp/random/8`, `rf/random/2`, and `rf/random/8`, identical to the native
dev server.

## What ships

`build.sh` assembles the bundle by combining this folder's HF-only files with:

- `ids/` — the canonical package, copied verbatim (single source of truth).
- A curated `models/` set: MLP 2/8-class, RF 2-class (165 MB), **RF 8-class (1.3 GB)**.
  The 8-class forest loads in ~2 GB RAM, well within the free tier's 16 GB. To change
  what ships, edit the `MODELS=(...)` list in `build.sh` — the analyzer auto-discovers
  whatever model files are present, so the bundle's `models/` dir is the curation.
- `data/cic_iot_2023.parquet` (145 MB) — the SHAP background pool.

Total ~1.8 GB. Note: the cold start loads all models eagerly (~11 s for the 8-class RF).

## Push it (needs YOUR Hugging Face account)

The `hf` CLI lives in pyenv's 3.11.9. Run from the repo root. Replace `USER`.

```sh
export HF=~/.pyenv/versions/3.11.9/bin/hf

# 1. log in with a WRITE token from https://huggingface.co/settings/tokens
$HF auth login

# 2. create the Docker Space once (SDK is required for spaces)
$HF repo create rt-ids-analyzer --repo-type space --space-sdk docker

# 3. stage the bundle from canonical source AND upload in one step:
deploy/hf-space/build.sh USER/rt-ids-analyzer

# (run build.sh with no args to stage only and inspect the bundle before uploading.)
```

HF then builds the image (a few minutes). Watch the build log on the Space page.

## Wire up the frontend

Once live, the Space serves at `https://USER-rt-ids-analyzer.hf.space`. Point the frontend
at it (Vercel env var, or `ids-frontend/.env`):

```
VITE_API_URL=https://USER-rt-ids-analyzer.hf.space
```

Sanity check after build: `curl https://USER-rt-ids-analyzer.hf.space/api/health`.
