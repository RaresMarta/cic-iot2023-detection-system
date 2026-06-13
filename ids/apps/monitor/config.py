"""Configuration + ban policy for the live beside-path NIDS.

All values are overridable via environment variables so the same image runs in
offline-dev (Mac, pcap replay, logging enforcer) and on the Linux VPS (live
capture, nftables) without code changes.
"""
from __future__ import annotations

import os

from ids.core.config import MODELS_DIR


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


# Model: gate (2-class, ban trigger) + family (8-class, triage label) on 25 features
MODEL_SPLIT = _env_str('IDS_SPLIT', 'temporal')
MODEL_MODE_GATE = _env_str('IDS_MODE_GATE', '2')
MODEL_MODE_FAMILY = _env_str('IDS_MODE_FAMILY', '8')

# Capture: bridge interface name from docker-compose
IFACE = _env_str('IDS_IFACE', 'ids-br0')
# Windowing: tumbling window of 10 packets per unordered host pair (low-latency)
WINDOW = _env_int('IDS_WINDOW', 10)
# Flush partial windows after idle timeout (seconds)
IDLE_FLUSH_S = _env_float('IDS_IDLE_FLUSH_S', 1.5)
MIN_PARTIAL = _env_int('IDS_MIN_PARTIAL', 2)
# Queue: capture thread → async consumer (drop-oldest on overflow)
QUEUE_MAXSIZE = _env_int('IDS_QUEUE_MAXSIZE', 2000)

# Protected IPs: never banned (mock website + bridge gateway)
PROTECTED_IPS = _env_set('IDS_PROTECTED_IPS', {'172.30.0.10'})
NEVER_BAN = _env_set('IDS_NEVER_BAN', {'172.30.0.10', '172.30.0.1'})
# Ban: threshold 0.6 + consecutive windows (robustness vs domain shift)
BAN_THRESHOLD = _env_float('IDS_BAN_THRESHOLD', 0.60)
CONSECUTIVE_FOR_BAN = _env_int('IDS_CONSECUTIVE_FOR_BAN', 2)
# Ban lifetime and recovery window
BAN_TTL_S = _env_int('IDS_BAN_TTL_S', 120)
RECOVER_AFTER_S = _env_float('IDS_RECOVER_AFTER_S', 5.0)
# Family taxonomy for dashboard (informational; ban driven by gate model)
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

# Capture source: 'live' (interface) or 'replay' (pcap file)
SOURCE = _env_str('IDS_SOURCE', 'replay')
PCAP_PATH = _env_str('IDS_PCAP', '')
PCAP_REALTIME = os.environ.get('IDS_PCAP_REALTIME', '0') == '1'
# Enforcer: 'nft' (Linux) or 'log' (dev/Mac no-op, auto-detects)
ENFORCER = os.environ.get('IDS_ENFORCER', '').strip().lower()
