#!/usr/bin/env bash
# Entry point for each attacker swarm container. With `--scale attacker=N`, N copies
# run this, each with its own distinct source IP on idsnet -> a distributed attack
# (the Mirai-style "fan-in": many sources converging on one target). Each source is
# real and routable within the bridge, so each is attributed and alerted individually,
# and the dashboard shows many attacker nodes lighting up one by one.
#
# Pick the attack via ATTACK env (synflood|udpflood|recon|spoofed_flood). DURATION=0
# runs until the container is stopped.
set -euo pipefail
ATTACK="${ATTACK:-synflood}"
export TARGET="${TARGET:-172.30.0.10}"
export DURATION="${DURATION:-0}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[swarm] $(hostname -i) running '${ATTACK}' -> ${TARGET}"

case "${ATTACK}" in
  synflood|udpflood|recon|spoofed_flood) exec "${DIR}/${ATTACK}.sh" ;;
  *) echo "[swarm] unknown ATTACK='${ATTACK}'"; exit 1 ;;
esac
