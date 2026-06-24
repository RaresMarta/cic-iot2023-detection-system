"""Build a benign-augmented parquet: all attack rows unchanged + all unique benign rows.

The base parquet caps benign at 200k (per-class cap). To test whether benign
starvation drives the ~99% out-of-distribution false-positive rate on IoT-23,
we replace the capped benign block with every unique benign row from the
freshly downloaded BenignTraffic captures, holding the attack rows fixed.

Benign label string in the parquet is 'Benign_Final' (verified), NOT
'BenignTraffic'. The downloaded CSVs carry no Label column.
"""
import glob
import polars as pl
from ids.core.config import X_COLUMNS_SELECTED

DEDUP_COLS = list(X_COLUMNS_SELECTED)   # dedup on the 25 model features (per plan)
BENIGN_GLOB = 'data/cic-iot-2023-benign/*.pcap.csv'
BASE = 'data/cic_iot_2023.parquet'
OUT = 'data/cic_iot_2023_benignaug.parquet'
BENIGN_LABEL = 'Benign_Final'

# --- attack rows unchanged from the base parquet ---
full = pl.read_parquet(BASE)
attack = full.filter(pl.col('Label') != BENIGN_LABEL)
print('attack rows (base, unchanged):', attack.height)
feat_cols = [c for c in full.columns if c not in ('Label', 'source_csv')]  # 39 raw features

# --- load benign CSVs (all 39 raw features) and dedup on the 25 model features ---
paths = sorted(glob.glob(BENIGN_GLOB))
print('benign CSVs:', paths)
benign = pl.concat([pl.read_csv(p).select(feat_cols) for p in paths], how='vertical')
print('raw benign rows (pre-dedup):', benign.height)
benign = benign.unique(subset=DEDUP_COLS)   # keep one full 39-feature row per unique 25-feature combo
print('unique benign rows:', benign.height)

# align benign to the full schema: add Label + source_csv, order columns
benign = benign.with_columns(
    pl.lit(BENIGN_LABEL).alias('Label'),
    pl.lit('benign_aug').alias('source_csv'),
).select(attack.columns)

aug = pl.concat([attack, benign], how='vertical')
aug.write_parquet(OUT)
frac = benign.height / aug.height
print('attack:', attack.height, 'benign:', benign.height, 'total:', aug.height)
print('benign fraction: %.4f' % frac)
print('wrote', OUT)
