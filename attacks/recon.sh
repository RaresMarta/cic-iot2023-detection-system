#!/usr/bin/env bash
# Port scan -> Recon family. A SYN sweep across many destination ports.
# Expected: methodical probing across ports -> Recon verdict (lower confidence than
# floods; may take more windows or stay at "alert" depending on the ban threshold).
set -euo pipefail
TARGET="${TARGET:-172.30.0.10}"

echo "[recon] TCP SYN port scan -> ${TARGET}"
# -sS SYN scan, -T4 fast, -p- all ports; -Pn skip host discovery (lab host is up)
nmap -sS -T4 -Pn -p 1-1024 "${TARGET}" || true
echo "[recon] done"
