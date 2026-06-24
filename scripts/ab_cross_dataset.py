"""A/B cross-dataset eval: baseline (models/) vs augmented (models_aug/).

For each IoT-23 capture, extract windows ONCE (reusing the iot23_align windowing
+ Zeek-label join), then score the same labelled windows with both model sets and
both backends (RF, MLP). Report benign FPR (the headline) and attack TPR.

Window = 10 (CIC non-flood default), matching the locked baselines.

Run: cd <root> && PYTHONPATH=<root> .venv/bin/python scripts/ab_cross_dataset.py
"""
import json
import os

import numpy as np
import polars as pl

from ids.core.config import MODELS_DIR
from ids.eval.cross_dataset_eval import model_feature_columns, _attack_index
from ids.eval.iot23_align import extract_windows, parse_conn_log_labeled, label_windows
from ids.runtime.predictor import MLPClassifier, RFClassifier

AUG_DIR = MODELS_DIR.parent / 'models_aug'
WINDOW = 10

CAPTURES = [
    ('honeypot-4-1', 'data/iot-23/honeypot-4-1/honeypot-4-1.pcap',
     'data/iot-23/honeypot-4-1/conn.log.labeled'),
    ('34-1-mixed', 'data/iot-23/2018-12-21-15-50-14-192.168.1.195.pcap',
     'data/iot-23/conn.log.labeled'),
    ('honeypot-5-1', 'data/iot-23/honeypot-5-1/honeypot-5-1.pcap',
     'data/iot-23/honeypot-5-1/conn.log.labeled'),
]


def make_clf(models_dir, backend):
    if backend == 'rf':
        return RFClassifier(models_dir, kind='rf', split='random', mode='2')
    return MLPClassifier(models_dir, split='random', mode='2')


def score(clf, feats):
    out = clf.predict(feats)
    ai = _attack_index(out['class_names'])
    # binary label per window from argmax
    lab = np.array([str(l).strip().lower() for l in out['labels']])
    lab = np.where(lab == 'attack', 'attack', 'benign')
    return lab


def rates(pred, truth):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    ben = truth == 'benign'
    atk = truth == 'attack'
    fp = int(((pred == 'attack') & ben).sum())
    tn = int(((pred == 'benign') & ben).sum())
    tp = int(((pred == 'attack') & atk).sum())
    fn = int(((pred == 'benign') & atk).sum())
    fpr = fp / (fp + tn) if (fp + tn) else None
    tpr = tp / (tp + fn) if (tp + fn) else None
    return {'fpr': fpr, 'tpr': tpr, 'n_benign': int(ben.sum()), 'n_attack': int(atk.sum())}


def main():
    cols = model_feature_columns()
    results = {}
    for name, pcap, labels in CAPTURES:
        if not (os.path.exists(pcap) and os.path.exists(labels)):
            print(f'[skip] {name}: missing {pcap} or {labels}')
            results[name] = {'error': 'capture files missing'}
            continue
        print(f'\n=== {name} ===')
        feat_rows, meta_rows = extract_windows(pcap, window=WINDOW)
        by_pair = parse_conn_log_labeled(labels)
        wlabels, keep, stats = label_windows(meta_rows, by_pair)
        keep_np = np.array(keep, dtype=bool)
        feats = pl.DataFrame(feat_rows).select(cols).filter(pl.Series(keep_np))
        truth = np.array([l for l, k in zip(wlabels, keep) if k])
        n_kept = int(keep_np.sum())
        print(f'  windows={len(feat_rows)} kept={n_kept} '
              f'(overlap {stats["overlap"]}, pair-fallback {stats["pair_fallback"]}, '
              f'dropped {stats["dropped"]}); '
              f'truth benign={int((truth=="benign").sum())} attack={int((truth=="attack").sum())}')
        if n_kept == 0:
            results[name] = {'error': 'no labelled windows'}
            continue
        cap = {'n_windows': len(feat_rows), 'n_kept': n_kept,
               'truth_benign': int((truth == 'benign').sum()),
               'truth_attack': int((truth == 'attack').sum())}
        for tag, mdir in (('baseline', MODELS_DIR), ('augmented', AUG_DIR)):
            for backend in ('rf', 'mlp'):
                try:
                    clf = make_clf(mdir, backend)
                    pred = score(clf, feats)
                    r = rates(pred, truth)
                    cap[f'{tag}_{backend}'] = r
                    fps = f'{r["fpr"]:.4f}' if r['fpr'] is not None else 'n/a'
                    tps = f'{r["tpr"]:.4f}' if r['tpr'] is not None else 'n/a'
                    print(f'  {tag:9s} {backend:3s}: FPR={fps} TPR={tps}')
                except Exception as e:
                    cap[f'{tag}_{backend}'] = {'error': str(e)}
                    print(f'  {tag:9s} {backend:3s}: ERROR {e}')
        results[name] = cap
    print('\nJSON ' + json.dumps(results))
    return results


if __name__ == '__main__':
    main()
