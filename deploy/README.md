# Deployment — live IDS demo

The system runs as a **passive NIDS — it detects and alerts, it does not block.** There is
no firewall/nftables interaction and no IP banning in the current build. (Out-of-band IP
enforcement is deferred to a possible future IPS iteration.)

Three components on one Linux host, plus an on-demand attacker swarm. The detector is a
passive, co-located sensor that flags malicious sources from flow statistics.

## Topology

```
            idsnet bridge (172.30.0.0/16, host iface "ids-br0")
  attacker × N ──┐
  (172.30.0.x)   ├──► website 172.30.0.10  (mock customer site, :8080 on host)
  browser ───────┘                ▲
                                  │ traffic bridged through ids-br0
        detector (HOST netns) ────┘  sniffs ids-br0 (passive observe only)
        binds host :7870 (SSE /api/stream, /api/stats, /api/health)
```

The detector is host-networked so it sees container traffic on `ids-br0` with real
per-source IPs. The website and attackers are on the bridge so each attacker replica has
its own routable `172.30.0.x` (enables the distributed / Mirai "fan-in" view). The detector
only observes — it never sits in the forwarding path and never modifies host networking.

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

# random-source flood — still detected, but no stable source IP to attribute
# (illustrates why per-source response is hard; the detector classifies it regardless):
ATTACK=spoofed_flood docker compose -f deploy/docker-compose.yml --profile attack up attacker
```
ATTACK ∈ {synflood, udpflood, recon, spoofed_flood}.

## Verify (on the Linux host)

1. `curl http://<HOST_IP>:7870/api/health` → `{status: ok, mode: live, model: ...}`.
2. Start a `synflood`. On `/api/stream` you should see `flow` events classified DoS/DDoS,
   then an `alert` for the attacker's IP, then a `recovered` once it goes quiet.
3. `curl http://<HOST_IP>:7870/api/stats` → `malicious` and the `by_family` histogram climb
   during the attack.
4. Scale the swarm to 30 → many distinct source IPs each raise an `alert`, shown as the
   fan-in view on the dashboard. The protected site stays reachable throughout (the detector
   never touches traffic): `curl -s -o/dev/null -w '%{http_code}\n' http://<HOST_IP>:8080/health`.

## Full fidelity locally on macOS — Colima (NOT Docker Desktop)

Docker Desktop runs containers in a hidden LinuxKit VM that blocks the detector's host-net
raw capture. Colima gives a real Linux kernel for live sniffing:

```sh
brew install colima docker docker-compose
colima start --cpu 2 --memory 4 --disk 30      # real Linux VM; switches docker context
docker compose -f deploy/docker-compose.yml up -d --build website detector
# drive a continuous attack (managed bg service, distinct source IP on idsnet):
DURATION=0 ATTACK=synflood docker compose -f deploy/docker-compose.yml --profile attack \
  up -d --scale attacker=1 attacker
curl -s localhost:7870/api/stats             # malicious / by_family climb during the flood
# the site stays up for everyone (the detector is out of the path):
curl -s -o/dev/null -w '%{http_code}\n' localhost:8080/health
```
Verified end-to-end this way: real capture → 2-class gate (Attack) → alert on the SSE feed,
with the protected site unaffected.

## Offline (macOS, no Docker / no root)

The detector pipeline (capture → window → classify → decide → publish) is fully exercisable
offline, which is how it was developed and validated:

```sh
.venv/bin/python -m ids.apps.monitor simulate          # real sampled CIC-IoT-2023 flows
#   curl -XPOST localhost:7870/api/inject -d '{"family":"DDoS","count":20}'
.venv/bin/python -m ids.apps.monitor replay path/to.pcap --realtime
.venv/bin/python -m ids.apps.monitor.synth ./pcaps     # generate synthetic test pcaps
```
