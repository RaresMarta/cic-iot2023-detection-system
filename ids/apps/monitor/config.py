"""Configuration for the live beside-path NIDS.

All values are overridable via environment variables so the same image runs in
offline-dev (Mac, pcap replay) and on the Linux VPS (live capture) without code
changes.
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


MODEL_SPLIT = _env_str('IDS_SPLIT', 'random')
MODEL_MODE_GATE = _env_str('IDS_MODE_GATE', '2')
MODEL_MODE_FAMILY = _env_str('IDS_MODE_FAMILY', '8')
GATE_MODEL = _env_str('IDS_GATE_MODEL', 'mlp').lower()
FAMILY_MODEL = _env_str('IDS_FAMILY_MODEL', 'mlp').lower()
DECISION_MODE = _env_str('IDS_DECISION_MODE', 'gate').lower()

IFACE = _env_str('IDS_IFACE', 'ids-br0')
WINDOW = _env_int('IDS_WINDOW', 10)
IDLE_FLUSH_S = _env_float('IDS_IDLE_FLUSH_S', 1.5)
MIN_PARTIAL = _env_int('IDS_MIN_PARTIAL', 2)
QUEUE_MAXSIZE = _env_int('IDS_QUEUE_MAXSIZE', 2000)

PROTECTED_IPS = _env_set('IDS_PROTECTED_IPS', {'172.30.0.10'})
RECOVER_AFTER_S = _env_float('IDS_RECOVER_AFTER_S', 5.0)
SAFE_IPS = _env_set('IDS_SAFE_IPS', set())

SOURCE = _env_str('IDS_SOURCE', 'replay')
PCAP_PATH = _env_str('IDS_PCAP', '')
PCAP_REALTIME = os.environ.get('IDS_PCAP_REALTIME', '0') == '1'

SUPABASE_ENABLED = _env_str('IDS_SUPABASE_ENABLED', 'false').lower() == 'true'
SUPABASE_URL = _env_str('IDS_SUPABASE_URL', '').rstrip('/')
SUPABASE_KEY = _env_str('IDS_SUPABASE_KEY', '')
MONITOR_ID = _env_str('IDS_MONITOR_ID', '')
MONITOR_NAME = _env_str('IDS_MONITOR_NAME', '')
MONITOR_PUBLIC_IP = _env_str('IDS_MONITOR_PUBLIC_IP', '')
MONITOR_OWNER = _env_str('IDS_MONITOR_OWNER', '')
SUPABASE_FLOW_RATE = _env_float('IDS_SUPABASE_FLOW_RATE', 25.0)
SUPABASE_SNAPSHOT_S = _env_float('IDS_SUPABASE_SNAPSHOT_S', 15.0)
