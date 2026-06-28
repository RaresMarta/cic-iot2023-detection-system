"""
Compare feature distributions between:
  A) canonical 39-feature UNB release  -> data/cic_iot_2023.parquet
  B) Kaggle 46-feature release         -> data/kaggle_ciciot23/CICIOT23/{train,validation,test}.csv

Outputs:
  - feature set diff (already known, re-printed for the report)
  - per-feature distribution stats on the 37 SHARED features
  - drift metrics: standardized mean diff, KS-like quantile distance, range overlap
  - label taxonomy diff

Sampling: take an equal random sample from each source (default 300k rows) so the
comparison is not dominated by the larger flood classes. We pool all labels (the
question asked is about *feature* distribution shift between the two releases, not
per-class), but we also stratify-cap per label to avoid flood domination.
"""
import sys
import numpy as np
import polars as pl

CANON = "data/cic_iot_2023.parquet"
KAGGLE_TRAIN = "data/kaggle_ciciot23/CICIOT23/train/train.csv"

SHARED = [
    'Header_Length', 'Protocol Type', 'Rate', 'fin_flag_number',
    'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count', 'HTTP',
    'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
    'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max',
    'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance',
]

PER_LABEL_CAP = 10_000  # equalize across classes before pooling


def load_canonical():
    lf = pl.scan_parquet(CANON)
    cols = SHARED + ['Label']
    lf = lf.select(cols)
    # cap per label
    df = (
        lf.collect()
        .group_by('Label')
        .map_groups(lambda g: g.sample(n=min(len(g), PER_LABEL_CAP), seed=42))
    )
    return df


def load_kaggle():
    # only read train (largest); cap per label
    lf = pl.scan_csv(KAGGLE_TRAIN, infer_schema_length=10000)
    cols = SHARED + ['label']
    lf = lf.select(cols).rename({'label': 'Label'})
    df = (
        lf.collect()
        .group_by('Label')
        .map_groups(lambda g: g.sample(n=min(len(g), PER_LABEL_CAP), seed=42))
    )
    return df


def stats(df, col):
    s = df[col].drop_nulls().drop_nans() if df[col].dtype.is_float() else df[col].drop_nulls()
    a = s.to_numpy().astype(np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    qs = np.quantile(a, [0, 0.25, 0.5, 0.75, 1.0])
    return {
        'n': a.size, 'mean': a.mean(), 'std': a.std(),
        'min': qs[0], 'p25': qs[1], 'p50': qs[2], 'p75': qs[3], 'max': qs[4],
        'arr': a,
    }


def ks_stat(a, b):
    """Two-sample KS statistic via empirical CDF on a shared grid."""
    grid = np.quantile(np.concatenate([a, b]), np.linspace(0, 1, 200))
    grid = np.unique(grid)
    ca = np.searchsorted(np.sort(a), grid, side='right') / a.size
    cb = np.searchsorted(np.sort(b), grid, side='right') / b.size
    return float(np.max(np.abs(ca - cb)))


def main():
    print("Loading canonical (39-feat UNB)...", file=sys.stderr)
    dc = load_canonical()
    print(f"  canonical pooled rows: {len(dc):,}", file=sys.stderr)
    print("Loading Kaggle (46-feat)...", file=sys.stderr)
    dk = load_kaggle()
    print(f"  kaggle pooled rows:    {len(dk):,}", file=sys.stderr)

    print("\n" + "=" * 110)
    print("PER-FEATURE DISTRIBUTION COMPARISON (37 shared features, per-label-capped pool)")
    print("CANON = 39-feature UNB release | KAGGLE = 46-feature release")
    print("=" * 110)
    hdr = f"{'feature':<18}{'src':<7}{'mean':>12}{'std':>12}{'p50':>12}{'p25':>12}{'p75':>12}{'max':>14}{'KS':>8}{'SMD':>8}"
    print(hdr)
    print("-" * 110)

    rows = []
    for col in SHARED:
        sc = stats(dc, col)
        sk = stats(dk, col)
        if sc is None or sk is None:
            print(f"{col:<18}  (missing data)")
            continue
        ks = ks_stat(sc['arr'], sk['arr'])
        pooled_std = np.sqrt((sc['std'] ** 2 + sk['std'] ** 2) / 2) or 1.0
        smd = abs(sc['mean'] - sk['mean']) / (pooled_std if pooled_std else 1.0)
        print(f"{col:<18}{'CANON':<7}{sc['mean']:>12.3f}{sc['std']:>12.3f}{sc['p50']:>12.3f}{sc['p25']:>12.3f}{sc['p75']:>12.3f}{sc['max']:>14.1f}{'':>8}{'':>8}")
        print(f"{'':<18}{'KAGGLE':<7}{sk['mean']:>12.3f}{sk['std']:>12.3f}{sk['p50']:>12.3f}{sk['p25']:>12.3f}{sk['p75']:>12.3f}{sk['max']:>14.1f}{ks:>8.3f}{smd:>8.2f}")
        rows.append((col, ks, smd))
        print()

    print("=" * 110)
    print("DRIFT RANKING (highest KS first) — KS≈0 same dist, KS→1 disjoint")
    print("=" * 110)
    for col, ks, smd in sorted(rows, key=lambda r: -r[1]):
        flag = "  <-- LARGE" if ks > 0.3 else ("  <- moderate" if ks > 0.1 else "")
        print(f"  {col:<20} KS={ks:>6.3f}  SMD={smd:>6.2f}{flag}")


if __name__ == "__main__":
    main()
