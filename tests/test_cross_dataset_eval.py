"""Self-test for the cross-dataset eval harness — proves the wiring without any
external datasets.

Two layers:
  A. End-to-end via the real extractor on a SYNTHESISED pcap (a SYN flood + a
     benign HTTPS exchange written to a temp pcap), exercising
     extract_features -> feature-parity -> predict for BOTH rf and mlp.
  B. Extractor-bypass: hand-crafted rows with the exact 25 feature columns run
     straight through predict() for rf and mlp, then through evaluate(), proving
     scaler->model column alignment and the metrics path independently of pcap
     parsing.

Run directly:  python tests/test_cross_dataset_eval.py
(or via pytest; the model-dependent parts skip if artifacts are absent.)
"""
from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import dpkt
import numpy as np
import polars as pl

from ids.eval import cross_dataset_eval as cde
from ids.eval.cross_dataset_eval import (
    check_feature_parity,
    evaluate,
    extract_features,
    model_feature_columns,
    predict,
)
from ids.eval.label_maps import to_binary


# ---- synthetic pcap helpers (mirrors ids/runtime/validate_extractor.py) -----
def _frame(src, dst, sport, dport, flags=0, ttl=64, payload=b'') -> bytes:
    l4 = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, data=payload)
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst),
                    p=dpkt.ip.IP_PROTO_TCP, ttl=ttl, data=l4)
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(src=b'\x02\x00\x00\x00\x00\x01',
                                 dst=b'\x02\x00\x00\x00\x00\x02', data=ip)
    return bytes(eth)


def _write_pcap(path: Path, frames):
    with open(path, 'wb') as f:
        w = dpkt.pcap.Writer(f)
        for ts, buf in frames:
            w.writepkt(buf, ts=ts)


def _syn_flood(n=100, t0=1000.0):
    return [(t0 + i * 1e-4,
             _frame('185.10.0.9', '10.0.0.5', 40000 + i, 80, flags=dpkt.tcp.TH_SYN))
            for i in range(n)]


def _benign_https(n=20, t0=2000.0):
    out = []
    for i in range(n):
        if i % 2 == 0:
            buf = _frame('192.168.1.20', '10.0.0.5', 51000, 443,
                         flags=dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, ttl=128, payload=b'x' * 400)
        else:
            buf = _frame('10.0.0.5', '192.168.1.20', 443, 51000,
                         flags=dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, ttl=64, payload=b'y' * 800)
        out.append((t0 + i * 0.02, buf))
    return out


# ---------------------------------------------------------------------------
def test_label_maps_default_rule():
    assert to_binary('bot-iot', 'Normal') == 'benign'
    assert to_binary('bot-iot', 'DDoS') == 'attack'
    assert to_binary('ton-iot', 0) == 'benign'
    assert to_binary('iot-23', '-') == 'benign'
    assert to_binary('whatever', 'SomeBrandNewAttack') == 'attack'  # default: non-benign -> attack
    print('[label_maps] default rule OK (non-benign -> attack)')


def test_handcrafted_predict_and_evaluate():
    """Extractor-bypass: hand-built 25-col rows -> predict (rf & mlp) -> evaluate."""
    cols = model_feature_columns()
    assert len(cols) == 25

    # Two crafted rows: one flood-like (high rate/syn), one benign-like.
    flood = {c: 0.0 for c in cols}
    flood.update({'Rate': 9.5e4, 'Number': 100.0, 'syn_count': 100.0,
                  'syn_flag_number': 1.0, 'TCP': 1.0, 'IAT': 1e-4,
                  'Header_Length': 20.0, 'Protocol Type': 6.0, 'Time_To_Live': 64.0,
                  'Tot sum': 6000.0, 'Min': 60.0, 'Max': 60.0, 'AVG': 60.0, 'Std': 0.0,
                  'HTTP': 1.0})
    benign = {c: 0.0 for c in cols}
    benign.update({'Rate': 50.0, 'Number': 10.0, 'ack_count': 10.0,
                   'ack_flag_number': 1.0, 'psh_flag_number': 1.0, 'TCP': 1.0,
                   'IAT': 0.02, 'Header_Length': 32.0, 'Protocol Type': 6.0,
                   'Time_To_Live': 128.0, 'Tot sum': 8000.0, 'Min': 400.0,
                   'Max': 800.0, 'AVG': 600.0, 'Std': 200.0, 'HTTPS': 1.0})
    df = pl.DataFrame([flood, benign]).select(cols)

    # parity gate must pass on a correctly-built frame
    parity = check_feature_parity(df, raise_on_mismatch=False)
    assert parity['ok'], parity

    for model in ('rf', 'mlp'):
        labels, scores = predict(df, model=model)
        assert labels.shape == (2,)
        assert scores.shape == (2,)
        assert set(np.unique(labels)) <= {'benign', 'attack'}
        assert np.all((scores >= 0) & (scores <= 1)), scores
        print(f'[predict/{model}] labels={list(labels)} attack_scores={np.round(scores,4).tolist()}')

        # evaluate() against a crafted ground truth (row0 attack, row1 benign)
        truth = np.array(['attack', 'benign'])
        m = evaluate(labels, truth, scores)
        assert 0.0 <= m['f2'] <= 1.0
        assert m['confusion_matrix'] and m['n'] == 2
        assert m['pr_curve'] is not None and len(m['pr_curve']['precision']) >= 2
        print(f"[evaluate/{model}] recall={m['recall']:.3f} prec={m['precision']:.3f} "
              f"F1={m['f1']:.3f} F2={m['f2']:.3f} cm={m['confusion_matrix']}")


def test_parity_mismatch_is_loud():
    cols = model_feature_columns()
    bad = pl.DataFrame({c: [0.0] for c in cols[:-1]})  # drop one column
    rep = check_feature_parity(bad, raise_on_mismatch=False)
    assert not rep['ok'] and rep['missing'] == [cols[-1]]
    raised = False
    try:
        check_feature_parity(bad, raise_on_mismatch=True)
    except AssertionError:
        raised = True
    assert raised
    print(f"[parity] mismatch detected & raised; missing={rep['missing']}")


def test_end_to_end_synthetic_pcap():
    """Real extractor on a synthesised pcap -> parity -> predict (rf & mlp)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_pcap(tmp / 'flood.pcap', _syn_flood(100))
        _write_pcap(tmp / 'benign.pcap', _benign_https(20))

        # directory mode (concatenates both pcaps, sorted by name)
        feats = extract_features(tmp, window=10)
        print(f'[e2e] extracted {feats.height} windows x {feats.width} feats from 2 pcaps')
        assert feats.height >= 2

        parity = check_feature_parity(feats, raise_on_mismatch=False)
        assert parity['ok'], parity
        print(f"[e2e] feature-parity OK: {parity['n_actual']}/25 columns match, "
              f"order_ok={parity['order_ok']}")

        for model in ('rf', 'mlp'):
            labels, scores = predict(feats, model=model)
            assert labels.shape[0] == feats.height == scores.shape[0]
            assert np.all((scores >= 0) & (scores <= 1))
            print(f'[e2e/{model}] verdicts={list(labels)} '
                  f'attack_scores={np.round(scores, 4).tolist()}')


if __name__ == '__main__':
    print('feature_columns (25):', model_feature_columns())
    print('encoder/attack-class wiring via predictors\n')
    test_label_maps_default_rule()
    test_parity_mismatch_is_loud()
    test_handcrafted_predict_and_evaluate()
    test_end_to_end_synthetic_pcap()
    print('\nALL SELF-TESTS PASSED')
