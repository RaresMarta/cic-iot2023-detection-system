#!/usr/bin/env bash
# Stage the HF Space bundle from the canonical source tree, then (optionally) upload.
#
# There is ONE copy of the code: the repo's ids/ package. This script assembles a
# throwaway bundle dir by combining the HF-only files that live here (Dockerfile,
# requirements.txt, README.md, pyproject.toml) with the canonical ids/ + a curated
# subset of models/ + the SHAP parquet. Nothing is copied into version control, so
# the Space can never drift behind ids/ again.
#
# Usage:
#   deploy/hf-space/build.sh                 # stage only -> prints the bundle path
#   deploy/hf-space/build.sh USER/rt-ids-analyzer   # stage + hf upload to that Space
#
# The hf CLI lives in pyenv 3.11.9 (see DEPLOY.md). Log in first: $HF auth login.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # deploy/hf-space
ROOT="$(cd "$HERE/../.." && pwd)"                       # repo root (cic-iot2023-detection-system)
SPACE="${1:-}"

BUNDLE="$(mktemp -d)/hf-space"
mkdir -p "$BUNDLE"

echo "[build] repo root:   $ROOT"
echo "[build] bundle dir:  $BUNDLE"

# 1. HF-only files (the only things that physically live in deploy/hf-space/).
cp "$HERE/Dockerfile" "$HERE/requirements.txt" "$HERE/README.md" "$HERE/pyproject.toml" "$BUNDLE/"

# 2. Canonical code — single source of truth.
rsync -a --exclude='__pycache__' --exclude='*.pyc' "$ROOT/ids/" "$BUNDLE/ids/"

# 3. Curated models — only what the analyzer serves. Edit this list to change what ships.
#    The analyzer auto-discovers whatever model files are present, so the bundle's
#    models/ dir IS the curation.
MODELS=(
  feature_columns.joblib
  preprocessor_random.joblib
  temperature_scaling.joblib
  label_encoder_random_2class.joblib
  label_encoder_random_8class.joblib
  ids_dnn_random_2class.pth
  ids_dnn_random_8class.pth
  ids_rf_random_2class.joblib
  ids_rf_random_8class.joblib      # 1.3G — loads in ~2GB RAM, fits the 16GB free tier
)
mkdir -p "$BUNDLE/models"
for m in "${MODELS[@]}"; do
  cp "$ROOT/models/$m" "$BUNDLE/models/$m"
done

# 4. SHAP background pool.
mkdir -p "$BUNDLE/data"
cp "$ROOT/data/cic_iot_2023.parquet" "$BUNDLE/data/cic_iot_2023.parquet"

echo "[build] staged bundle:"
du -sh "$BUNDLE"
du -sh "$BUNDLE"/* | sort -h

if [[ -z "$SPACE" ]]; then
  echo "[build] no Space arg given — staged only. To upload:"
  echo "    \$HF upload $BUNDLE  # see DEPLOY.md, or re-run: $0 USER/rt-ids-analyzer"
  echo "BUNDLE=$BUNDLE"
  exit 0
fi

HF="${HF:-$HOME/.pyenv/versions/3.11.9/bin/hf}"
echo "[build] uploading to Space: $SPACE  (hf=$HF)"
"$HF" upload "$SPACE" "$BUNDLE" . \
  --repo-type space \
  --commit-message "Deploy RT-IDS analyzer backend (build.sh)"
echo "[build] done. Watch the build log on the Space page."
