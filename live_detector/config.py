"""Configuration + ban policy for the live beside-path NIDS.

All values are overridable via environment variables so the same image runs in
offline-dev (Mac, pcap replay, logging enforcer) and on the Linux VPS (live
capture, nftables) without code changes.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_set(name: str, default: set[str]) -> set[str]:
    raw = os.environ.get(name)
    if not raw:
        return set(default)
    return {p.strip() for p in raw.split(',') if p.strip()}


# ── Model ────────────────────────────────────────────────────────────────────
# Two heads on the same 25 features (thesis taxonomy):
#   gate  (2-class Benign/Attack) — the inline decision that drives the ban. Robust;
#          correctly + confidently flags real attack-tool traffic (e.g. hping3 floods)
#          that the 8-class head mislabels due to train/deploy domain shift.
#   family (8-class) — the attack-family label for triage + the dashboard's per-type
#          "signature" visuals. Informational, not the ban trigger.
MODEL_SPLIT = _env_str('IDS_SPLIT', 'temporal')
MODEL_MODE_GATE = _env_str('IDS_MODE_GATE', '2')
MODEL_MODE_FAMILY = _env_str('IDS_MODE_FAMILY', '8')

# ── Capture / windowing ──────────────────────────────────────────────────────
# Deterministic bridge name pinned in docker-compose (com.docker.network.bridge.name).
IFACE = _env_str('IDS_IFACE', 'ids-br0')
# A "flow" is an unordered host pair; tumbling window of WINDOW packets.
# Fixed at 10 for low-latency live detection (a flood fills 10 packets in ms).
# The paper used 10/100 per class, but class is unknown at capture time.
WINDOW = _env_int('IDS_WINDOW', 10)
# Emit a partial window for a host pair idle longer than this (seconds), so slow
# flows (and the tail of a burst) still get classified.
IDLE_FLUSH_S = _env_float('IDS_IDLE_FLUSH_S', 1.5)
MIN_PARTIAL = _env_int('IDS_MIN_PARTIAL', 2)
# Bounded handoff queue (capture thread -> async consumer). Drop-oldest on full.
QUEUE_MAXSIZE = _env_int('IDS_QUEUE_MAXSIZE', 2000)

# ── Enforcement / policy ─────────────────────────────────────────────────────
# The protected party (mock website) — never banned, and used to identify which
# endpoint of a flow is the external attacker.
PROTECTED_IPS = _env_set('IDS_PROTECTED_IPS', {'172.30.0.10'})
# Hard allowlist: never ban these (site, bridge gateway, anything critical).
NEVER_BAN = _env_set('IDS_NEVER_BAN', {'172.30.0.10', '172.30.0.1'})

# Only ban above this calibrated confidence AND after this many consecutive
# malicious windows from the same source (prevents a single misclassification
# from nuking an IP / flapping). Default 0.6: the temperature-calibrated per-family
# confidence varies widely (Mirai ~1.0, DDoS ~0.86, DoS ~0.72, Recon ~0.62), so a
# 0.9 gate would never ban DoS/Recon; 0.6 + the consecutive-window requirement is
# the robustness compromise.
BAN_THRESHOLD = _env_float('IDS_BAN_THRESHOLD', 0.60)
CONSECUTIVE_FOR_BAN = _env_int('IDS_CONSECUTIVE_FOR_BAN', 2)
# Ban lifetime (nftables set element timeout). After this the IP is unbanned.
BAN_TTL_S = _env_int('IDS_BAN_TTL_S', 120)
# A source is considered "recovered" if no malicious window is seen for this long.
RECOVER_AFTER_S = _env_float('IDS_RECOVER_AFTER_S', 5.0)

# Informational family taxonomy for the dashboard (which families the demo treats as
# reliably-detectable "green"; see demo/sampler.py). NOT the ban trigger — the ban is
# driven by the 2-class gate (see enforcement.Policy). Kept for display/labelling.
POLICY: dict[str, str] = {
    'Benign': 'allow',
    'DDoS': 'ban',
    'DoS': 'ban',
    'Mirai': 'ban',
    'Recon': 'ban',
    'Web': 'alert',
    'Spoofing': 'alert',
    'BruteForce': 'alert',
}

# ── Capture source selection ─────────────────────────────────────────────────
# IDS_SOURCE = 'live' (LiveCapture on IFACE) or 'replay' (PcapReplay on IDS_PCAP).
SOURCE = _env_str('IDS_SOURCE', 'replay')
PCAP_PATH = _env_str('IDS_PCAP', '')
PCAP_REALTIME = os.environ.get('IDS_PCAP_REALTIME', '0') == '1'

# Enforcer: 'nft' (Linux host) or 'log' (dev/Mac no-op). Auto-detects if unset.
ENFORCER = os.environ.get('IDS_ENFORCER', '').strip().lower()
