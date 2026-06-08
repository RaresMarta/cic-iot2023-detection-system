"""Validate the DPKT extractor end-to-end.

Two modes:
  * `python -m demo.validate_extractor <file.pcap> [window]` — extract real pcap, classify,
    print the per-window verdicts. Use this for DISTRIBUTIONAL validation once real
    CIC-IoT-2023 pcaps are available (compare predicted labels to ground truth).
  * `python -m demo.validate_extractor` (no args) — MECHANICAL self-test: synthesize a SYN
    flood and a benign HTTPS exchange, run them through extractor + model, and confirm the
    pipeline produces model-compatible features and sane verdicts.

NOTE: full distributional fidelity (extractor output ≈ training distribution) requires the
original CIC-IoT-2023 raw pcaps, which are not present locally (only the extracted CSVs are).
"""
from __future__ import annotations

import socket
import sys
import tempfile
from pathlib import Path

import dpkt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo.dpkt_extractor import extract_features  # noqa: E402
from demo.inference import IDSPredictor  # noqa: E402


def _frame(src, dst, sport, dport, proto='tcp', flags=0, ttl=64, payload=b'') -> bytes:
    if proto == 'tcp':
        l4 = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, data=payload)
        ipp = dpkt.ip.IP_PROTO_TCP
    else:
        l4 = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
        l4.ulen = len(l4)
        ipp = dpkt.ip.IP_PROTO_UDP
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=ipp, ttl=ttl, data=l4)
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(src=b'\x02\x00\x00\x00\x00\x01', dst=b'\x02\x00\x00\x00\x00\x02', data=ip)
    return bytes(eth)


def _write_pcap(path: Path, frames: list[tuple[float, bytes]]):
    with open(path, 'wb') as f:
        w = dpkt.pcap.Writer(f)
        for ts, buf in frames:
            w.writepkt(buf, ts=ts)


def _synth_syn_flood(n=100, t0=1000.0) -> list[tuple[float, bytes]]:
    # one attacker -> one victim:80, rapid SYNs (fills a single 100-packet window)
    return [(t0 + i * 1e-4,
             _frame('185.10.0.9', '10.0.0.5', 40000 + i, 80, flags=dpkt.tcp.TH_SYN, ttl=64))
            for i in range(n)]


def _synth_benign(n=20, t0=2000.0) -> list[tuple[float, bytes]]:
    # client <-> server:443 exchange, ACK+PSH, realistic sizes/timing
    frames = []
    for i in range(n):
        if i % 2 == 0:
            buf = _frame('192.168.1.20', '10.0.0.5', 51000, 443,
                         flags=dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, ttl=128, payload=b'x' * 400)
        else:
            buf = _frame('10.0.0.5', '192.168.1.20', 443, 51000,
                         flags=dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, ttl=64, payload=b'y' * 800)
        frames.append((t0 + i * 0.02, buf))
    return frames


def _classify(predictor, df, scenario):
    if df.height == 0:
        print(f'  [{scenario}] no windows extracted'); return
    pred = predictor.predict(df)
    for i in range(df.height):
        probs = pred['probabilities'][i]
        print(f'  [{scenario}] window {i}: {pred["labels"][i]:8s} ({pred["confidences"][i]:.2f})'
              f'  Number={df["Number"][i]:.0f} Rate={df["Rate"][i]:.0f} '
              f'syn_cnt={df["syn_count"][i]:.0f} ack_cnt={df["ack_count"][i]:.0f}')


def main():
    predictor = IDSPredictor(PROJECT_ROOT / 'models', split='temporal', mode='8')

    if len(sys.argv) > 1:
        path = sys.argv[1]
        win = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        df = extract_features(path, window=win)
        print(f'Real pcap {path}: {df.height} windows (window={win})')
        _classify(predictor, df, 'pcap')
        return

    print('MECHANICAL self-test (synthetic pcaps — not distributional validation)\n')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        flood_p, benign_p = tmp / 'flood.pcap', tmp / 'benign.pcap'
        _write_pcap(flood_p, _synth_syn_flood(100))
        _write_pcap(benign_p, _synth_benign(20))

        flood_df = extract_features(flood_p, window=100)
        benign_df = extract_features(benign_p, window=10)

        print(f'Extracted: flood={flood_df.height} window(s), benign={benign_df.height} window(s)')
        print(f'Schema matches model: {list(flood_df.columns) == list(predictor.x_columns)}\n')
        _classify(predictor, flood_df, 'SYN flood ')
        _classify(predictor, benign_df, 'benign    ')


if __name__ == '__main__':
    main()
