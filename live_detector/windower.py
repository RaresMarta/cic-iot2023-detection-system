"""Streaming host-pair windower.

Same windowing semantics as demo/dpkt_extractor.extract_features, but driven one
packet at a time so it can run on a live capture instead of a finished pcap:
a "flow" is an unordered host pair (src,dst); packets accumulate into a tumbling
window of WINDOW packets; a completed window is turned into the 25 CIC-IoT-2023
features via the same _window_features() used for the file path (so live and
offline results match exactly). Partial windows are flushed after an idle timeout.

All mutation happens in a single thread (the capture thread), so no locking.
"""
from __future__ import annotations

import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the validated parser + feature math — do NOT reimplement.
from demo.dpkt_extractor import _packet_record, _window_features  # noqa: E402

from . import config  # noqa: E402


def _ip_str(raw: bytes) -> str:
    """Decode a raw IP address (4 bytes v4, 16 bytes v6) to a string."""
    try:
        if len(raw) == 4:
            return socket.inet_ntop(socket.AF_INET, raw)
        if len(raw) == 16:
            return socket.inet_ntop(socket.AF_INET6, raw)
    except (OSError, ValueError):
        pass
    return raw.hex()


@dataclass
class WindowResult:
    features: dict          # the 25 CIC-IoT-2023 features (model input)
    ip_a: str               # one endpoint of the host pair
    ip_b: str               # the other endpoint
    n_packets: int
    ts_start: float
    ts_end: float


@dataclass
class _Bucket:
    pkts: list = field(default_factory=list)
    last_seen: float = 0.0


class StreamWindower:
    def __init__(self, window: int | None = None, idle_flush_s: float | None = None,
                 min_partial: int | None = None):
        self.window = window or config.WINDOW
        self.idle_flush_s = idle_flush_s if idle_flush_s is not None else config.IDLE_FLUSH_S
        self.min_partial = min_partial if min_partial is not None else config.MIN_PARTIAL
        self._buckets: dict[frozenset, _Bucket] = {}

    def add(self, ts: float, buf: bytes) -> WindowResult | None:
        """Feed one raw ethernet frame. Returns a WindowResult when a window fills."""
        rec = _packet_record(ts, buf)
        if rec is None:
            return None
        key = frozenset((rec['src'], rec['dst']))
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = _Bucket()
        bucket.pkts.append(rec)
        bucket.last_seen = ts
        if len(bucket.pkts) >= self.window:
            del self._buckets[key]
            return self._result(bucket.pkts)
        return None

    def flush_idle(self, now: float) -> list[WindowResult]:
        """Emit partial windows for host pairs idle longer than idle_flush_s."""
        out: list[WindowResult] = []
        stale = [k for k, b in self._buckets.items()
                 if now - b.last_seen >= self.idle_flush_s]
        for key in stale:
            bucket = self._buckets.pop(key)
            if len(bucket.pkts) >= self.min_partial:
                out.append(self._result(bucket.pkts))
        return out

    @staticmethod
    def _result(pkts: list) -> WindowResult:
        feats = _window_features(pkts)
        ip_a = _ip_str(pkts[0]['src'])
        ip_b = _ip_str(pkts[0]['dst'])
        ts = [p['ts'] for p in pkts]
        return WindowResult(
            features=feats, ip_a=ip_a, ip_b=ip_b,
            n_packets=len(pkts), ts_start=min(ts), ts_end=max(ts),
        )
