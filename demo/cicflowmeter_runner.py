"""Subprocess wrapper around CICFlowMeter — turns a PCAP into the 39-feature CSV the model expects.

Two backends are supported:

1. **Java CICFlowMeter** (the original, from CIC). Set CICFLOWMETER_HOME to the cloned/built
   repo. Invoked via the bundled `CICFlowMeter` script.
2. **Python `cicflowmeter` package** (community port). Installed via `pip install cicflowmeter`.
   Less complete but cross-platform. Used as a fallback.

Either backend writes the output CSV next to the input PCAP.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class CICFlowMeterError(RuntimeError):
    pass


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
    """Process a PCAP and return the path to the resulting CSV.

    Output filename convention matches both backends: <pcap_stem>.pcap_Flow.csv (Java)
    or <pcap_stem>.csv (Python). We rename to a stable name.
    """
    pcap_path = Path(pcap_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = output_dir / f'{pcap_path.stem}.csv'

    java_cli = _find_java_cli()
    if java_cli is not None:
        cmd = [str(java_cli), str(pcap_path), str(output_dir)]
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
        return expected

    python_cli = _find_python_cli()
    if python_cli is not None:
        cmd = [python_cli, '-f', str(pcap_path), '-c', str(expected)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise CICFlowMeterError(
                f'Python cicflowmeter failed (exit {proc.returncode}):\n{proc.stderr}'
            )
        return expected

    raise CICFlowMeterError(
        'No CICFlowMeter backend found. Either:\n'
        '  - set CICFLOWMETER_HOME to a built CICFlowMeter Java repo, OR\n'
        '  - pip install cicflowmeter'
    )
