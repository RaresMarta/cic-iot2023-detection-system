# Attack scripts (lab-only)

Standalone scripts that generate the demo's attack traffic against the mock website.
They are **not** integrated into the dashboard — a real attacker is a separate party
from the defender. Run them from a terminal (or via the on-demand `attacker` compose
service) inside the isolated `idsnet` lab network.

**Scope / safety:** target defaults to the mock site (`172.30.0.10`) and is meant only
for this isolated lab. Only the families the model reliably detects are produced:
DDoS/DoS (floods), Recon (scan), and Mirai-style distributed flood (the swarm). These
require `hping3` and `nmap` (preinstalled in the attacker image).

| Script | Family | Expected detector reaction |
|--------|--------|----------------------------|
| `synflood.sh`     | DoS / DDoS | high `Rate`, all-SYN → ban within ~2 windows |
| `udpflood.sh`     | DoS / DDoS | UDP flood → ban |
| `recon.sh`        | Recon      | port sweep → alert/ban (lower confidence) |
| `spoofed_flood.sh`| DDoS       | `--rand-source`: bans can't keep up — the deliberate "where IP-banning fails" demo |
| swarm (compose)   | Mirai/DDoS | many distinct source IPs each banned (fan-in) |

Single-source demo:
```sh
TARGET=172.30.0.10 ./synflood.sh
```

Distributed (multi-source) demo — scale the swarm on the same host:
```sh
docker compose -f ../deploy/docker-compose.yml up --scale attacker=30 attacker
```
