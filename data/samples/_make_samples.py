"""Generate three plausible CIC-format sample CSVs for the preview-mode demo.

These are NOT real captures — they're shape-correct fixtures so the upload path
in the Gradio UI has something realistic-looking to chew on. Run once:

    .venv/bin/python data/samples/_make_samples.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

X_COLUMNS = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance',
]


def benign_row(rng: random.Random) -> list:
    avg = rng.uniform(400, 1100)
    return [
        rng.uniform(40, 320),       # Header_Length
        6,                           # Protocol Type (TCP)
        rng.randint(48, 64),         # TTL
        rng.uniform(2.0, 35.0),      # Rate
        rng.choice([0, 0, 0, 1]),    # fin
        rng.choice([0, 0, 1]),       # syn
        0,                            # rst
        rng.choice([0, 1]),          # psh
        rng.choice([0, 1, 1, 1]),    # ack
        0, 0,                         # ece, cwr
        rng.randint(1, 12),          # ack_count
        rng.randint(0, 2),           # syn_count
        rng.randint(0, 1),           # fin_count
        0,                            # rst_count
        rng.choice([0, 1]),          # HTTP
        rng.choice([0, 1, 1]),       # HTTPS
        rng.choice([0, 1]),          # DNS
        0, 0, 0, 0,                  # Telnet, SMTP, SSH, IRC
        1,                            # TCP
        rng.choice([0, 1]),          # UDP
        0, 0, 0, 0,                  # DHCP, ARP, ICMP, IGMP
        4,                            # IPv
        1,                            # LLC
        rng.uniform(2_000, 22_000),  # Tot sum
        rng.uniform(40, 200),        # Min
        rng.uniform(800, 1500),      # Max
        avg,                          # AVG
        rng.uniform(80, 320),        # Std
        avg,                          # Tot size
        rng.uniform(0.001, 0.4),     # IAT
        rng.randint(2, 30),          # Number
        rng.uniform(2_000, 60_000),  # Variance
    ]


def syn_flood_row(rng: random.Random) -> list:
    return [
        rng.uniform(20, 60),
        6,
        rng.randint(28, 64),
        rng.uniform(1500, 28_000),
        0,
        1,
        0,
        0,
        0,
        0, 0,
        0,
        rng.randint(40, 400),
        0,
        0,
        0, 0, 0, 0, 0, 0, 0,
        1,
        0,
        0, 0, 0, 0,
        4,
        1,
        rng.uniform(2_000, 6_000),
        rng.uniform(40, 60),
        rng.uniform(60, 80),
        rng.uniform(50, 70),
        rng.uniform(0.1, 4.0),
        rng.uniform(50, 70),
        rng.uniform(0.00002, 0.0005),
        rng.randint(60, 500),
        rng.uniform(0.5, 12.0),
    ]


def port_scan_row(rng: random.Random) -> list:
    return [
        rng.uniform(20, 80),
        rng.choice([6, 6, 17]),
        rng.randint(40, 64),
        rng.uniform(50, 900),
        0,
        rng.choice([0, 1, 1]),
        rng.choice([0, 1]),
        0,
        0,
        0, 0,
        rng.randint(0, 2),
        rng.randint(0, 3),
        0,
        rng.randint(0, 1),
        0, 0, 0, 0, 0, 0, 0,
        rng.choice([0, 1]),
        rng.choice([0, 1]),
        0, 0, 0, 0,
        4,
        1,
        rng.uniform(60, 600),
        rng.uniform(40, 60),
        rng.uniform(60, 200),
        rng.uniform(50, 120),
        rng.uniform(2.0, 30.0),
        rng.uniform(50, 120),
        rng.uniform(0.0005, 0.05),
        rng.randint(1, 6),
        rng.uniform(2.0, 400.0),
    ]


def write_csv(path: Path, n: int, gen, seed: int) -> None:
    rng = random.Random(seed)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(X_COLUMNS)
        for _ in range(n):
            w.writerow(gen(rng))


def main() -> None:
    out = Path(__file__).parent
    write_csv(out / 'sample_benign_browsing.csv', 80, benign_row, seed=1)
    write_csv(out / 'sample_syn_flood.csv', 250, syn_flood_row, seed=2)
    write_csv(out / 'sample_port_scan.csv', 60, port_scan_row, seed=3)
    print(f'Wrote 3 sample CSVs to {out}')


if __name__ == '__main__':
    main()
