# Deployment — live IDS demo

Three components on one Linux host, plus an on-demand attacker swarm. The detector is
a passive, co-located sensor that bans malicious sources out-of-band at the host
firewall (a flow-based NIDS with automated response — not an inline IPS).

## Topology

```
            idsnet bridge (172.30.0.0/16, host iface "ids-br0")
  attacker × N ──┐
  (172.30.0.x)   ├──► website 172.30.0.10  (mock customer site, :8080 on host)
  browser ───────┘                ▲
                                  │ traffic bridged through ids-br0
        detector (HOST netns) ────┘  sniffs ids-br0, bans via nftables on the host
        binds host :7870 (SSE /api/stream, /api/blocklist, /api/stats)
```

The detector must be host-networked so it (a) sees container traffic on `ids-br0` with
real per-source IPs and (b) can write the host's nftables. The website and attackers are
on the bridge so each attacker replica has its own routable `172.30.0.x` (enables the
distributed / Mirai "fan-in" and real per-IP bans).

## Run

```sh
# from the project root (cic-iot2023-detection-system/)
export DETECTOR_URL=http://<HOST_IP>:7870     # browser-reachable detector for the SSE feed
docker compose -f deploy/docker-compose.yml up --build website detector
#   browse the protected site:  http://<HOST_IP>:8080
#   detector stream:            http://<HOST_IP>:7870/api/stream

# add the dashboard once the frontend image exists (frontend workstream):
docker compose -f deploy/docker-compose.yml --profile full up dashboard
```

## Drive attacks (separate party — not wired to the dashboard)

```sh
# single source:
docker compose -f deploy/docker-compose.yml --profile attack run --rm \
  -e ATTACK=synflood -e DURATION=20 attacker

# distributed (many sources, the Mirai fan-in):
ATTACK=synflood docker compose -f deploy/docker-compose.yml --profile attack \
  up --scale attacker=30 attacker

# the "IP-banning fails" contrast:
ATTACK=spoofed_flood docker compose -f deploy/docker-compose.yml --profile attack up attacker
```
ATTACK ∈ {synflood, udpflood, recon, spoofed_flood}.

## Verify (on the Linux host)

1. `curl http://<HOST_IP>:7870/api/health` → `{mode: live, enforcer: nft}`.
2. Start a `synflood`. On the detector you should see `flow` events classified
   DoS/DDoS, then an `alert`, then a `ban` for the attacker's IP.
3. `sudo nft list set inet ids banned` → the attacker IP is present (with a timeout).
4. From the attacker container, `curl http://172.30.0.10` now times out (dropped).
5. After `IDS_BAN_TTL_S` (120s) the element expires → traffic flows again.
6. Scale the swarm to 30 → `curl http://<HOST_IP>:7870/api/blocklist` shows ~30 IPs,
   each visibly quarantined on the dashboard's fan-in view.
7. Confirm rule ordering: `sudo nft list ruleset` — the `inet ids` `drop_banned`
   chain (priority -150) runs before Docker's own forward rules.

## Full fidelity locally on macOS — Colima (NOT Docker Desktop)

Docker Desktop runs containers in a hidden LinuxKit VM that blocks the detector's
host-net raw capture + nftables. Colima gives a real Linux kernel you control:

```sh
brew install colima docker docker-compose
colima start --cpu 2 --memory 4 --disk 30      # real Linux VM; switches docker context
docker compose -f deploy/docker-compose.yml up -d --build website detector
# drive a continuous attack (managed bg service, distinct source IP on idsnet):
DURATION=0 ATTACK=synflood docker compose -f deploy/docker-compose.yml --profile attack \
  up -d --scale attacker=1 attacker
curl -s localhost:7870/api/blocklist        # attacker IP appears once banned
# prove cut-off from the banned attacker (100% loss) vs site still up:
docker compose -f deploy/docker-compose.yml exec attacker hping3 -S -c5 -p80 172.30.0.10
curl -s -o/dev/null -w '%{http_code}\n' localhost:8080/health
```
Verified end-to-end this way: real capture → 2-class gate (Attack) → nft ban → 100%
packet loss for the attacker, site unaffected for everyone else, TTL auto-expiry.

## Two firewall drop points (why)

Same-bridge container↔container traffic is L2-bridged and skips the IP `forward` hook
unless `br_netfilter` is loaded — so the `inet/forward` rule alone won't drop a
same-host attacker. The detector therefore also installs a **`bridge` family** drop
rule (`ether type ip ip saddr @banned drop`) which sees bridged frames. The `inet`
rule still covers routed/external attackers on a real VPS. No host sysctl needed.

## Offline (macOS, no Docker / no root)

The detector pipeline (capture → window → classify → decide → publish) is fully
exercisable offline, which is how it was developed and validated:

```sh
.venv/bin/python -m ids.apps.monitor simulate          # real sampled CIC-IoT-2023 flows
#   curl -XPOST localhost:7870/api/inject -d '{"family":"DDoS","count":20}'
.venv/bin/python -m ids.apps.monitor replay path/to.pcap --realtime
.venv/bin/python -m ids.apps.monitor.synth ./pcaps     # generate synthetic test pcaps
```
In offline/dev the enforcer auto-falls back to a logging no-op (no nftables needed).
