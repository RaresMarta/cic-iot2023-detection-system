"""IoT-23 label alignment + cross-dataset scoring.

The CIC-IoT-2023 extractor (`ids.runtime.extractor.extract_features`) returns only
the 25 model features, dropping host/time identity, so a foreign dataset's
per-connection labels cannot be joined to its output. This module reproduces the
SAME windowing (reusing the extractor's `_packet_record` / `_window_features`, so
the feature values are byte-for-byte identical) while ALSO capturing each window's
unordered host pair and time span. It then labels each window from a Zeek
`conn.log.labeled` file and scores the labeled windows with the frozen 2-class
model via `cross_dataset_eval.predict` / `evaluate`.

Alignment is approximate by construction: windows are per-host-pair over a short
packet span; conn.log labels are per 5-tuple connection. Rule: a window is
`attack` if ANY connection on its host pair overlapping its time span is
Malicious; if none overlap in time, fall back to any connection on that host pair;
if the host pair never appears in conn.log, the window is dropped (and counted).

Window size defaults to 10 (the CIC non-flood default). NOTE: a full window then
has Number == 10 — the known CIC-IoT-2023 window-size leak — so the `Number`
column here reflects the chosen window, not the traffic. Reported for visibility.

INFERENCE ONLY. Does not modify the extractor, models, or training code.
"""
from __future__ import annotations

import argparse
import socket
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from ids.eval.cross_dataset_eval import (
    evaluate,
    model_feature_columns,
    predict,
    _print_report,
)
from ids.runtime.extractor import _iter_packets, _packet_record, _window_features


def _ip(b: bytes) -> str:
    return socket.inet_ntop(socket.AF_INET if len(b) == 4 else socket.AF_INET6, b)


# ---------------------------------------------------------------------------
# Extraction with per-window identity (mirrors extractor.extract_features order)
# ---------------------------------------------------------------------------
def extract_windows(pcap_path: str | Path, window: int = 10, include_partial: bool = True):
    """Return (feat_rows, meta_rows) in the exact order extract_features emits
    windows. feat_rows: list of 25-feature dicts. meta_rows: list of
    (host_a, host_b, t_start, t_end) with host_a <= host_b as dotted strings."""
    buckets: dict = defaultdict(list)
    feat_rows: list[dict] = []
    meta_rows: list[tuple] = []

    def flush(pkts: list) -> None:
        feat_rows.append(_window_features(pkts))
        ips = sorted({p['src'] for p in pkts} | {p['dst'] for p in pkts})
        hosts = [_ip(b) for b in ips]
        a = hosts[0]
        b = hosts[1] if len(hosts) > 1 else hosts[0]
        tss = [p['ts'] for p in pkts]
        meta_rows.append((a, b, min(tss), max(tss)))

    for ts, buf in _iter_packets(Path(pcap_path)):
        rec = _packet_record(ts, buf)
        if rec is None:
            continue
        key = frozenset((rec['src'], rec['dst']))
        buckets[key].append(rec)
        if len(buckets[key]) >= window:
            flush(buckets[key])
            buckets[key] = []

    if include_partial:
        for pkts in buckets.values():
            if len(pkts) >= 2:
                flush(pkts)

    return feat_rows, meta_rows


# ---------------------------------------------------------------------------
# Zeek conn.log.labeled parsing
# ---------------------------------------------------------------------------
def parse_conn_log_labeled(path: str | Path) -> dict:
    """Parse a Zeek conn.log.labeled into dict[frozenset{orig_h,resp_h}] ->
    sorted list of (t0, t1, is_malicious). Labels are read robustly: a row is
    malicious iff the token 'Malicious' appears in it (the label only lives in the
    trailing label field)."""
    fields = None
    idx: dict = {}
    by_pair: dict = defaultdict(list)

    with open(path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('#'):
                if line.startswith('#fields'):
                    fields = line.rstrip('\n').split('\t')[1:]
                    idx = {name: i for i, name in enumerate(fields)}
                continue
            if fields is None:
                continue
            parts = line.rstrip('\n').split('\t')
            try:
                oh = parts[idx['id.orig_h']]
                rh = parts[idx['id.resp_h']]
                ts = float(parts[idx['ts']])
            except (KeyError, IndexError, ValueError):
                continue
            di = idx.get('duration')
            d = parts[di] if di is not None and di < len(parts) else '-'
            try:
                dur = 0.0 if d in ('-', '', '(empty)') else float(d)
            except ValueError:
                dur = 0.0
            is_mal = 'malicious' in line.lower()
            by_pair[frozenset((oh, rh))].append((ts, ts + dur, is_mal))

    for k in by_pair:
        by_pair[k].sort(key=lambda c: c[0])
    return by_pair


# ---------------------------------------------------------------------------
# Window labelling
# ---------------------------------------------------------------------------
def label_windows(meta_rows: list, by_pair: dict):
    """Assign each window a binary label. Returns (labels, keep_mask, stats)."""
    labels: list = []
    keep: list = []
    stats = {'overlap': 0, 'pair_fallback': 0, 'dropped': 0}

    for (a, b, t0, t1) in meta_rows:
        conns = by_pair.get(frozenset((a, b)))
        if not conns:
            labels.append(None)
            keep.append(False)
            stats['dropped'] += 1
            continue
        overlap = [c for c in conns if not (c[1] < t0 or c[0] > t1)]
        if overlap:
            stats['overlap'] += 1
            mal = any(c[2] for c in overlap)
        else:
            stats['pair_fallback'] += 1
            mal = any(c[2] for c in conns)
        labels.append('attack' if mal else 'benign')
        keep.append(True)

    return labels, keep, stats


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(pcap: str | Path, labels_path: str | Path, model: str = 'rf', window: int = 10) -> dict:
    cols = model_feature_columns()

    print(f'[1/4] extracting windows from {pcap} (window={window}) ...')
    feat_rows, meta_rows = extract_windows(pcap, window=window)
    print(f'      -> {len(feat_rows)} windows')
    if not feat_rows:
        print('no windows extracted; aborting')
        return {}

    print(f'[2/4] parsing labels from {labels_path} ...')
    by_pair = parse_conn_log_labeled(labels_path)
    print(f'      -> {len(by_pair)} labelled host pairs')

    print('[3/4] aligning labels to windows ...')
    labels, keep, stats = label_windows(meta_rows, by_pair)
    keep_np = np.array(keep, dtype=bool)
    n_kept = int(keep_np.sum())
    feats = pl.DataFrame(feat_rows).select(cols)
    feats_kept = feats.filter(pl.Series(keep_np))
    true_labels = [l for l, k in zip(labels, keep) if k]
    n_mal = sum(1 for l in true_labels if l == 'attack')
    print(f'      labelled {n_kept}/{len(feat_rows)} windows '
          f'(time-overlap {stats["overlap"]}, pair-fallback {stats["pair_fallback"]}); '
          f'dropped {stats["dropped"]} (host pair not in conn.log)')
    print(f'      ground truth: attack={n_mal}, benign={n_kept - n_mal}')
    if n_kept:
        num = feats_kept['Number']
        print(f"      Number (window-leak check): min={num.min()} "
              f"median={num.median()} max={num.max()}")

    if n_kept == 0:
        print('no labelled windows; cannot score')
        return {}

    print(f'[4/4] scoring with {model} ...')
    pred_labels, scores = predict(feats_kept, model=model)
    metrics = evaluate(pred_labels, true_labels, scores)
    _print_report(metrics)
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='IoT-23 cross-dataset scoring of the frozen CIC-IoT-2023 2-class model.')
    ap.add_argument('--pcap', required=True, help='IoT-23 capture pcap')
    ap.add_argument('--labels', required=True, help='matching conn.log.labeled')
    ap.add_argument('--model', default='rf', choices=['rf', 'mlp'])
    ap.add_argument('--window', type=int, default=10)
    args = ap.parse_args(argv)
    run(args.pcap, args.labels, model=args.model, window=args.window)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
