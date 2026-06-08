"""Faithful pcap -> CIC-IoT-2023 feature extractor (reproduces the paper's DPKT method).

The CIC-IoT-2023 model was NOT trained on CICFlowMeter features. The authors used a custom
dpkt-based extractor that groups packets into fixed-size windows (10 packets for most classes,
100 for the flooding classes) "between two hosts" and mean-aggregates per-packet statistics.
This module reproduces that for the 25 features the model actually uses, so pcap/live
classification matches the training distribution rather than the skewed CICFlowMeter mapping.

Design choices where the paper is silent (documented for the thesis):
  * windows are NON-overlapping (tumbling); overlap/stride is unspecified in the paper.
  * a "flow" is an UNORDERED host pair (srcIP,dstIP) — i.e. bidirectional, port-agnostic —
    matching "a sequence of packets carrying information between two hosts".
  * packet length = full captured frame length (matches the dataset's Min≈60 for tiny frames).
  * *_flag_number / protocol one-hots are the per-window FRACTION of packets (the dataset
    mean-aggregates the per-packet 0/1 indicators); *_count are per-window sums.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import dpkt
import joblib
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

_TCP_FLAGS = {
    'fin': dpkt.tcp.TH_FIN, 'syn': dpkt.tcp.TH_SYN, 'rst': dpkt.tcp.TH_RST,
    'psh': dpkt.tcp.TH_PUSH, 'ack': dpkt.tcp.TH_ACK,
}


def _iter_packets(pcap_path: Path):
    """Yield (ts, buf) for each frame, handling both pcap and pcapng."""
    with open(pcap_path, 'rb') as f:
        magic = f.read(4); f.seek(0)
        reader = dpkt.pcapng.Reader(f) if magic == b'\x0a\x0d\x0d\x0a' else dpkt.pcap.Reader(f)
        for ts, buf in reader:
            yield ts, buf


def _packet_record(ts: float, buf: bytes):
    """Per-packet fields needed for the features; None if not IP/IPv6."""
    try:
        eth = dpkt.ethernet.Ethernet(buf)
    except Exception:
        return None
    ip = eth.data
    is_v6 = isinstance(ip, dpkt.ip6.IP6)
    if not isinstance(ip, dpkt.ip.IP) and not is_v6:
        return None
    proto = ip.nxt if is_v6 else ip.p
    ttl = ip.hlim if is_v6 else ip.ttl
    ip_hdr = 40 if is_v6 else ip.hl * 4
    rec = {
        'ts': ts, 'len': len(buf), 'ttl': ttl, 'proto': proto,
        'src': bytes(ip.src), 'dst': bytes(ip.dst),
        'tcp': 0, 'udp': 0, 'sport': 0, 'dport': 0,
        'flags': {k: 0 for k in _TCP_FLAGS}, 'hdr': ip_hdr,
    }
    l4 = ip.data
    if isinstance(l4, dpkt.tcp.TCP):
        rec['tcp'] = 1; rec['sport'] = l4.sport; rec['dport'] = l4.dport
        rec['hdr'] = ip_hdr + l4.off * 4
        for name, bit in _TCP_FLAGS.items():
            rec['flags'][name] = 1 if (l4.flags & bit) else 0
    elif isinstance(l4, dpkt.udp.UDP):
        rec['udp'] = 1; rec['sport'] = l4.sport; rec['dport'] = l4.dport
        rec['hdr'] = ip_hdr + 8
    return rec


def _window_features(pkts: list) -> dict:
    n = len(pkts)
    lengths = np.array([p['len'] for p in pkts], dtype=float)
    ts = np.sort(np.array([p['ts'] for p in pkts], dtype=float))
    duration = float(ts[-1] - ts[0])
    iats = np.diff(ts)
    has = lambda port: sum(1 for p in pkts if p['sport'] == port or p['dport'] == port) / n
    frac = lambda key: sum(p['flags'][key] for p in pkts) / n
    count = lambda key: float(sum(p['flags'][key] for p in pkts))
    return {
        'Header_Length': float(np.mean([p['hdr'] for p in pkts])),
        'Protocol Type': float(np.mean([p['proto'] for p in pkts])),
        'Time_To_Live': float(np.mean([p['ttl'] for p in pkts])),
        'Rate': float(n / duration) if duration > 0 else 0.0,
        'fin_flag_number': frac('fin'), 'syn_flag_number': frac('syn'),
        'rst_flag_number': frac('rst'), 'psh_flag_number': frac('psh'),
        'ack_flag_number': frac('ack'),
        'ack_count': count('ack'), 'syn_count': count('syn'),
        'fin_count': count('fin'), 'rst_count': count('rst'),
        'HTTP': has(80), 'HTTPS': has(443), 'DNS': has(53),
        'TCP': sum(p['tcp'] for p in pkts) / n, 'UDP': sum(p['udp'] for p in pkts) / n,
        'Tot sum': float(lengths.sum()), 'Min': float(lengths.min()), 'Max': float(lengths.max()),
        'AVG': float(lengths.mean()), 'Std': float(lengths.std()),
        'IAT': float(iats.mean()) if len(iats) else 0.0,
        'Number': float(n),
    }


def extract_features(pcap_path: str | Path, window: int = 10, include_partial: bool = True) -> pl.DataFrame:
    """Parse a pcap into windowed CIC-IoT-2023 features (25 columns, in model order).

    window: packets per record (paper used 10 / 100). include_partial: emit a trailing
    short window per host pair (useful for small captures / live timeouts).
    """
    feature_columns = list(joblib.load(MODELS_DIR / 'feature_columns.joblib'))
    buckets: dict = defaultdict(list)
    rows: list[dict] = []
    for ts, buf in _iter_packets(Path(pcap_path)):
        rec = _packet_record(ts, buf)
        if rec is None:
            continue
        key = frozenset((rec['src'], rec['dst']))
        buckets[key].append(rec)
        if len(buckets[key]) >= window:
            rows.append(_window_features(buckets[key]))
            buckets[key] = []
    if include_partial:
        for pkts in buckets.values():
            if len(pkts) >= 2:
                rows.append(_window_features(pkts))
    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 for c in feature_columns})
    return pl.DataFrame(rows).select(feature_columns)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python -m demo.dpkt_extractor <file.pcap> [window]')
        raise SystemExit(1)
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    df = extract_features(sys.argv[1], window=win)
    print(f'{df.height} windows x {df.width} features (window={win})')
    print(df.head())
