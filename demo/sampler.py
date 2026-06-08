"""Per-class flow sampler for the live demo.

Pulls real CIC-IoT-2023 flows from the training parquet so the SOC dashboard can replay
genuine, in-distribution traffic through the model (simulate mode). Only the "green" classes
the model detects reliably via flow statistics are exposed:
DDoS, DoS, Mirai, Recon, Benign (Web/Spoofing excluded — see thesis methodology).

A small in-memory pool per family is loaded once at startup; sampling then serves random
rows from memory, fast enough for a real-time stream.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from labels import DICT_8CLASSES  # noqa: E402

PARQUET_PATH = PROJECT_ROOT / 'data' / 'cic_iot_2023.parquet'
MODELS_DIR = PROJECT_ROOT / 'models'

# Families the demo exposes (reliably detectable from flow statistics).
GREEN_FAMILIES: list[str] = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon']

# raw 34-class label -> family, restricted to the green families
_RAW_BY_FAMILY: dict[str, list[str]] = {}
for _raw, _fam in DICT_8CLASSES.items():
    if _fam in GREEN_FAMILIES:
        _RAW_BY_FAMILY.setdefault(_fam, []).append(_raw)


class FlowSampler:
    """Loads a capped per-family pool of real flows and serves random samples from memory."""

    def __init__(self, parquet_path: Path = PARQUET_PATH, pool_size: int = 2000, seed: int = 42):
        self.feature_columns: list[str] = list(joblib.load(MODELS_DIR / 'feature_columns.joblib'))
        self.pool_size = pool_size
        self._pools: dict[str, pl.DataFrame] = {}

        lf = pl.scan_parquet(str(parquet_path))
        for family, raws in _RAW_BY_FAMILY.items():
            df = (
                lf.filter(pl.col('Label').is_in(raws))
                  .select(self.feature_columns)
                  .collect()
            )
            if df.height == 0:
                raise ValueError(f'No rows in parquet for family {family!r} (labels {raws})')
            if df.height > pool_size:
                df = df.sample(n=pool_size, seed=seed)
            self._pools[family] = df

    @property
    def families(self) -> list[str]:
        return [f for f in GREEN_FAMILIES if f in self._pools]

    def sample_flows(self, family: str, n: int = 1, seed: int | None = None) -> pl.DataFrame:
        """Return n random real flows for the given green family (25 feature columns)."""
        if family not in self._pools:
            raise ValueError(f'Unknown/non-green family {family!r}; choose from {self.families}')
        pool = self._pools[family]
        return pool.sample(n=n, with_replacement=n > pool.height, shuffle=True, seed=seed)

    def benign_baseline(self, n: int = 1, seed: int | None = None) -> pl.DataFrame:
        """Convenience: sample normal/benign background traffic for the idle stream."""
        return self.sample_flows('Benign', n=n, seed=seed)


if __name__ == '__main__':
    s = FlowSampler()
    print('Families:', s.families)
    for fam in s.families:
        print(f'  {fam:8s} pool={s._pools[fam].height}')
    print('\nSample DDoS flow:')
    print(s.sample_flows('DDoS', n=2))
