"""Synthetic pcap generator for offline dev/demo (no real captures are committed).

Produces small pcaps with recognisable traffic shapes so the live pipeline can be
exercised on macOS via PcapReplay:
  * benign  — bidirectional HTTPS-ish TCP between a client and the protected host
  * synflood — DDoS SYN flood: many tiny all-SYN packets from attacker(s) -> host:80
  * udpflood — UDP flood -> host
  * scan     — recon: SYNs sweeping across many destination ports

These are illustrative traffic *shapes* for the demo/tests, not captured attacks.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import dpkt

HOST = '172.30.0.10'
CLIENT = '172.30.0.50'
ATTACKER = '172.30.0.66'
MAC_A = b'\x02\x00\x00\x00\x00\x01'
MAC_B = b'\x02\x00\x00\x00\x00\x02'


def _eth(src_ip: str, dst_ip: str, l4, smac=MAC_A, dmac=MAC_B) -> bytes:
    ip = dpkt.ip.IP(src=socket.inet_aton(src_ip), dst=socket.inet_aton(dst_ip), p=l4.__class__ is dpkt.tcp.TCP and dpkt.ip.IP_PROTO_TCP or dpkt.ip.IP_PROTO_UDP)
    ip.data = l4
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(src=smac, dst=dmac, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def _tcp(sport: int, dport: int, flags: int, payload: bytes = b'') -> dpkt.tcp.TCP:
    return dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, data=payload)


def _udp(sport: int, dport: int, payload: bytes = b'') -> dpkt.udp.UDP:
    u = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    u.ulen = len(u)  # type: ignore[attr-defined]
    return u


def gen_benign(n: int = 40) -> list[tuple[float, bytes]]:
    out, t = [], 1_000_000.0
    for i in range(n):
        if i % 2 == 0:
            l4 = _tcp(51000 + i, 443, dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK, b'x' * 400)
            out.append((t, _eth(CLIENT, HOST, l4, MAC_A, MAC_B)))
        else:
            l4 = _tcp(443, 51000 + i - 1, dpkt.tcp.TH_ACK, b'y' * 800)
            out.append((t, _eth(HOST, CLIENT, l4, MAC_B, MAC_A)))
        t += 0.02
    return out


def gen_synflood(n: int = 200, attacker: str = ATTACKER) -> list[tuple[float, bytes]]:
    out, t = [], 2_000_000.0
    for i in range(n):
        l4 = _tcp(40000 + (i % 1000), 80, dpkt.tcp.TH_SYN)
        out.append((t, _eth(attacker, HOST, l4, MAC_A, MAC_B)))
        t += 0.00005
    return out


def gen_udpflood(n: int = 200, attacker: str = ATTACKER) -> list[tuple[float, bytes]]:
    out, t = [], 3_000_000.0
    for i in range(n):
        l4 = _udp(50000 + (i % 1000), 53, b'\x00' * 10)
        out.append((t, _eth(attacker, HOST, l4, MAC_A, MAC_B)))
        t += 0.00008
    return out


def gen_scan(n: int = 120, attacker: str = ATTACKER) -> list[tuple[float, bytes]]:
    out, t = [], 4_000_000.0
    for i in range(n):
        l4 = _tcp(45000, 1 + i, dpkt.tcp.TH_SYN)
        out.append((t, _eth(attacker, HOST, l4, MAC_A, MAC_B)))
        t += 0.002
    return out


GENERATORS = {
    'benign': gen_benign,
    'synflood': gen_synflood,
    'udpflood': gen_udpflood,
    'scan': gen_scan,
}


def write_pcap(path: str | Path, packets: list[tuple[float, bytes]]) -> Path:
    path = Path(path)
    with open(path, 'wb') as f:
        w = dpkt.pcap.Writer(f)
        for ts, buf in packets:
            w.writepkt(buf, ts=ts)
    return path


def build_scenario(name: str) -> list[tuple[float, bytes]]:
    """A mixed scenario: benign baseline then the named attack (sorted by ts)."""
    pkts = gen_benign(40)
    if name in GENERATORS and name != 'benign':
        pkts = pkts + GENERATORS[name]()
    return sorted(pkts, key=lambda p: p[0])


if __name__ == '__main__':
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, gen in GENERATORS.items():
        pkts = sorted(gen_benign(40) + (gen() if name != 'benign' else []), key=lambda p: p[0])
        p = write_pcap(out_dir / f'{name}.pcap', pkts)
        print(f'{p}  ({len(pkts)} packets)')
