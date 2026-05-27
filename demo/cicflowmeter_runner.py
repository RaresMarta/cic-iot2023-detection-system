"""Subprocess wrapper around CICFlowMeter — turns a PCAP into the 39-feature CSV the model expects.

Two backends are supported:

1. **Java CICFlowMeter** (the original, from CIC). Set CICFLOWMETER_HOME to the cloned/built
   repo. Invoked via the bundled `CICFlowMeter` script.
2. **Python `cicflowmeter` package** (community port). Installed via `pip install cicflowmeter`.
   Used as fallback. Output columns are mapped to CIC-IoT-2023 format via map_to_cic_format().

Either backend writes the output CSV next to the input PCAP.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


class CICFlowMeterError(RuntimeError):
    pass


# ── Column mapping: Python cicflowmeter → CIC-IoT-2023 ────────────────────

# Direct renames
_RENAME = {
    'protocol':       'Protocol Type',
    'flow_byts_s':    'Rate',
    'pkt_len_min':    'Min',
    'pkt_len_max':    'Max',
    'pkt_len_mean':   'AVG',
    'pkt_len_std':    'Std',
    'pkt_len_var':    'Variance',
    'flow_iat_mean':  'IAT',
    'fin_flag_cnt':   'fin_flag_number',
    'syn_flag_cnt':   'syn_flag_number',
    'rst_flag_cnt':   'rst_flag_number',
    'psh_flag_cnt':   'psh_flag_number',
    'ack_flag_cnt':   'ack_flag_number',
    'ece_flag_cnt':   'ece_flag_number',
    'cwr_flag_count': 'cwr_flag_number',
}

# Port → protocol flag (well-known port numbers)
_PORT_FLAGS = {
    'HTTP':   {80},
    'HTTPS':  {443},
    'DNS':    {53},
    'SSH':    {22},
    'Telnet': {23},
    'SMTP':   {25, 465, 587},
    'IRC':    {6667, 6668, 6669},
    'DHCP':   {67, 68},
}


def map_to_cic_format(df: pd.DataFrame) -> pd.DataFrame:
    """Map Python cicflowmeter output columns to CIC-IoT-2023 column names.

    Handles:
    - Direct column renames
    - Computed columns (Header_Length, Number, Tot sum/size, flag counts)
    - Port-based binary protocol flags (HTTP, HTTPS, DNS, ...)
    - IP-protocol-based flags (TCP, UDP, ICMP, IGMP, ARP)
    - Unavailable features filled with 0 (Time_To_Live, IPv, LLC)

    Args:
        df: Raw DataFrame from Python cicflowmeter

    Returns:
        DataFrame with CIC-IoT-2023 column names
    """
    out = df.rename(columns=_RENAME).copy()

    # Computed columns
    out['Header_Length'] = df.get('fwd_header_len', 0) + df.get('bwd_header_len', 0)
    out['Number']        = df.get('tot_fwd_pkts', 0)   + df.get('tot_bwd_pkts', 0)
    out['Tot sum']       = df.get('totlen_fwd_pkts', 0) + df.get('totlen_bwd_pkts', 0)
    out['Tot size']      = out['Tot sum']  # same underlying quantity

    # Flag counts (ACK/SYN/FIN/RST packet counts mirror flag numbers in CIC-IoT-2023)
    out['ack_count'] = df.get('ack_flag_cnt', 0)
    out['syn_count'] = df.get('syn_flag_cnt', 0)
    out['fin_count'] = df.get('fin_flag_cnt', 0)
    out['rst_count'] = df.get('rst_flag_cnt', 0)

    # IP-protocol-based binary flags
    proto = df.get('protocol', pd.Series(0, index=df.index))
    out['TCP']  = (proto == 6).astype(int)
    out['UDP']  = (proto == 17).astype(int)
    out['ICMP'] = (proto == 1).astype(int)
    out['IGMP'] = (proto == 2).astype(int)
    out['ARP']  = (proto == 0).astype(int)  # ARP is L2; rarely seen at IP level

    # Port-based binary protocol flags
    src = df.get('src_port', pd.Series(0, index=df.index))
    dst = df.get('dst_port', pd.Series(0, index=df.index))
    for flag, ports in _PORT_FLAGS.items():
        out[flag] = (src.isin(ports) | dst.isin(ports)).astype(int)

    # Features not available from Python cicflowmeter — filled with 0
    # Time_To_Live: not captured by Python pkg; 0 introduces some bias but affects 1/25 features
    # IPv, LLC: layer-2 framing info not available at flow level
    for col in ('Time_To_Live', 'IPv', 'LLC'):
        if col not in out.columns:
            out[col] = 0

    return out


def _is_python_cicflow_format(df: pd.DataFrame) -> bool:
    """Detect whether a DataFrame came from the Python cicflowmeter package."""
    return 'flow_byts_s' in df.columns or 'fin_flag_cnt' in df.columns


def _find_java_cli() -> Path | None:
    home = os.environ.get('CICFLOWMETER_HOME')
    if not home:
        return None
    home_path = Path(home)
    for candidate in ('bin/CICFlowMeter', 'bin/CICFlowMeter.bat'):
        p = home_path / candidate
        if p.exists():
            return p
    return None


def _find_python_cli() -> str | None:
    return shutil.which('cicflowmeter')


def run_cicflowmeter(pcap_path: Path, output_dir: Path) -> Path:
    """Process a PCAP and return the path to the resulting CIC-IoT-2023-format CSV."""
    pcap_path  = Path(pcap_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = output_dir / f'{pcap_path.stem}.csv'

    java_cli = _find_java_cli()
    if java_cli is not None:
        cmd  = [str(java_cli), str(pcap_path), str(output_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise CICFlowMeterError(
                f'Java CICFlowMeter failed (exit {proc.returncode}):\n{proc.stderr}'
            )
        produced = output_dir / f'{pcap_path.name}_Flow.csv'
        if produced.exists():
            produced.replace(expected)
        elif not expected.exists():
            raise CICFlowMeterError(f'Expected output not found: {produced}')
        return expected  # Java output already in CIC-IoT-2023 format

    python_cli = _find_python_cli()
    if python_cli is not None:
        cmd  = [python_cli, '-f', str(pcap_path), '-c', str(expected)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise CICFlowMeterError(
                f'Python cicflowmeter failed (exit {proc.returncode}):\n{proc.stderr}'
            )
        # Map Python cicflowmeter columns → CIC-IoT-2023 format
        raw = pd.read_csv(expected)
        if _is_python_cicflow_format(raw):
            mapped = map_to_cic_format(raw)
            mapped.to_csv(expected, index=False)
        return expected

    raise CICFlowMeterError(
        'No CICFlowMeter backend found. Either:\n'
        '  - set CICFLOWMETER_HOME to a built CICFlowMeter Java repo, OR\n'
        '  - pip install cicflowmeter'
    )
