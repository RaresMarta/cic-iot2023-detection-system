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


# Model: gate (2-class, alert trigger) + family (8-class, triage label) on 25 features
MODEL_SPLIT = _env_str('IDS_SPLIT', 'random')
MODEL_MODE_GATE = _env_str('IDS_MODE_GATE', '2')
MODEL_MODE_FAMILY = _env_str('IDS_MODE_FAMILY', '8')
# Model type per head, independently swappable: 'mlp' (tiny PyTorch net, ~1 MB) or
# 'rf' (RandomForest joblib — the 8-class forest is ~1.3 GB, so mind RAM/disk).
# Both share the same predict() contract, so the detector is agnostic to the choice.
GATE_MODEL = _env_str('IDS_GATE_MODEL', 'mlp').lower()
FAMILY_MODEL = _env_str('IDS_FAMILY_MODEL', 'mlp').lower()
# Decision architecture: 'gate' (default, thesis design — a 2-class gate triggers
# alerts, the 8-class head labels the family) or 'single' (one 8-class model does
# both: argmax != Benign triggers, and is itself the family label). 'single' drops
# the gate stage and so diverges from the thesis; keep it for comparison only.
DECISION_MODE = _env_str('IDS_DECISION_MODE', 'gate').lower()

# Capture: bridge interface name from docker-compose
IFACE = _env_str('IDS_IFACE', 'ids-br0')
# Windowing: tumbling window of 10 packets per unordered host pair (low-latency)
WINDOW = _env_int('IDS_WINDOW', 10)
# Flush partial windows after idle timeout (seconds)
IDLE_FLUSH_S = _env_float('IDS_IDLE_FLUSH_S', 1.5)
MIN_PARTIAL = _env_int('IDS_MIN_PARTIAL', 2)
# Queue: capture thread → async consumer (drop-oldest on overflow)
QUEUE_MAXSIZE = _env_int('IDS_QUEUE_MAXSIZE', 2000)

# Protected IPs: identifies the monitored target so the attacker side of a flow
# can be attributed (mock website + bridge gateway).
PROTECTED_IPS = _env_set('IDS_PROTECTED_IPS', {'172.30.0.10'})
# An active attacker is considered recovered after this many idle seconds.
RECOVER_AFTER_S = _env_float('IDS_RECOVER_AFTER_S', 5.0)

# Capture source: 'live' (interface) or 'replay' (pcap file)
SOURCE = _env_str('IDS_SOURCE', 'replay')
PCAP_PATH = _env_str('IDS_PCAP', '')
PCAP_REALTIME = os.environ.get('IDS_PCAP_REALTIME', '0') == '1'

# Supabase backplane: a broker consumer that registers this worker as a "monitor",
# broadcasts live flows over Supabase Realtime (ephemeral, never stored), and persists
# incidents + periodic snapshots to Postgres. Opt-in and default-off; never on the
# detection path (see supabase_sink.py). Multi-tenant = one worker per environment, each
# with its own stable MONITOR_ID, all writing to one shared Supabase project. The KEY is a
# service_role key kept server-side on the worker (it bypasses row-level security).
SUPABASE_ENABLED = _env_str('IDS_SUPABASE_ENABLED', 'false').lower() == 'true'
SUPABASE_URL = _env_str('IDS_SUPABASE_URL', '').rstrip('/')
SUPABASE_KEY = _env_str('IDS_SUPABASE_KEY', '')
# Stable per-worker identity. MONITOR_ID is the upsert key in the monitors table; NAME is
# the human label shown in the dashboard picker; PUBLIC_IP is informational.
MONITOR_ID = _env_str('IDS_MONITOR_ID', '')
MONITOR_NAME = _env_str('IDS_MONITOR_NAME', '')
MONITOR_PUBLIC_IP = _env_str('IDS_MONITOR_PUBLIC_IP', '')
# Dashboard user (auth.users uuid) this monitor belongs to. The worker sets monitors.owner_id
# so row-level security shows the monitor only to that user. Empty -> owner_id null (the
# monitor exists but no user sees it under RLS until claimed).
MONITOR_OWNER = _env_str('IDS_MONITOR_OWNER', '')
# Max flows/sec broadcast to the dashboard feed. The feed is a sample for the eye; the
# aggregate counters remain truthful. Excess flows are dropped from the display only.
SUPABASE_FLOW_RATE = _env_float('IDS_SUPABASE_FLOW_RATE', 25.0)
# How often to persist a stats snapshot (seconds).
SUPABASE_SNAPSHOT_S = _env_float('IDS_SUPABASE_SNAPSHOT_S', 15.0)
