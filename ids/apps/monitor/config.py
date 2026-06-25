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

# Event store: persist incidents + periodic stats snapshots to SQLite. Opt-in and
# default-off so the detector is unchanged unless explicitly enabled. The DB lives
# under the gitignored data/ dir by default. (MODELS_DIR.parent == PROJECT_ROOT.)
DB_ENABLED = _env_str('IDS_DB_ENABLED', 'false').lower() == 'true'
DB_PATH = _env_str('IDS_DB_PATH', str(MODELS_DIR.parent / 'data' / 'events.db'))
DB_SNAPSHOT_S = _env_float('IDS_DB_SNAPSHOT_S', 15.0)

# ntfy push notifications: alert the admin's phone on each attack episode. Opt-in and
# default-off; never on the detection path (see notifier.py). Set IDS_NTFY_URL to your
# topic, e.g. https://ntfy.sh/<your-random-topic>.
NTFY_ENABLED = _env_str('IDS_NTFY_ENABLED', 'false').lower() == 'true'
NTFY_URL = _env_str('IDS_NTFY_URL', '')
NTFY_ON_RECOVER = _env_str('IDS_NTFY_ON_RECOVER', 'true').lower() == 'true'
