# RT-IDS Demo

Live demo wrapper around the trained CIC-IoT-2023 MLP. Two input modes:

1. **CSV upload** — already in CIC 39-feature format; classified directly.
2. **PCAP upload** — CICFlowMeter extracts flows first, then the model classifies.

## Setup

```powershell
pip install -r requirements.txt   # from project root, includes gradio + xgboost
```

The demo loads whatever models exist in `../models/`. Run the notebook end-to-end first so the models are saved.

## CICFlowMeter (required for PCAP mode only)

The Java CICFlowMeter is the canonical implementation. Two options:

### Option A — Java CICFlowMeter (recommended, matches dataset exactly)

```powershell
git clone https://github.com/ahlashkari/CICFlowMeter.git
cd CICFlowMeter
.\gradlew.bat installDist
$env:CICFLOWMETER_HOME = "$PWD\build\install\CICFlowMeter"
```

The runner auto-detects `$env:CICFLOWMETER_HOME` and invokes the bundled `CICFlowMeter.bat` script.

### Option B — Python `cicflowmeter` (fallback)

```powershell
pip install cicflowmeter
```

Less complete than the Java version but no JVM required.

## Run

```powershell
python -m demo.app
```

Opens at <http://localhost:7860>.

## Public URL for thesis defense

```powershell
cloudflared tunnel --url http://localhost:7860
```

Cloudflare prints a temporary public URL valid for the session — paste it in the defense slides.

## Live attack scripts (for the demo loop)

To make the demo end-to-end (user clicks "run attack" → captured → classified), add per-attack scripts under `demo/attacks/` that:

1. Spin up an attack against a controlled local target (a docker container or a VM).
2. Capture the resulting traffic to a PCAP via `tcpdump` (Linux/WSL) or `tshark`/`dumpcap` (Windows + Npcap).
3. Hand the PCAP to `cicflowmeter_runner.run_cicflowmeter` and surface results in the UI.

Suggested tools per attack family:

- **DDoS / DoS / Mirai-like floods** — `hping3`, `t50`, `iperf3` UDP storms
- **Recon** — `nmap` (TCP SYN scan, OS scan), `fping` for ping sweeps
- **Web** — `curl` scripts with payloads, `sqlmap` (SQL injection), `slowhttptest` (Slowloris)
- **Spoofing** — `arpspoof` (in WSL with proper interface bridging), `dnschef`
- **BruteForce** — `hydra` against a local dummy SSH

These are environment-specific (which interface, which target, what privilege level). They are scaffolding work for thesis defense day, not something a generic script can produce.

## Files

- `inference.py` — model + scaler + encoder loader; `IDSPredictor` class
- `cicflowmeter_runner.py` — subprocess wrapper, auto-detects Java or Python backend
- `app.py` — Gradio UI
