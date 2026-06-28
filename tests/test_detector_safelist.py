"""Known-safe-infrastructure short-circuit in the detector.

The live demo streams trusted infra (DNS/CDN/cloud) as green Benign WITHOUT
running the model, while every non-safe flow still gets a real verdict — so an
attack from an unlisted source is not blind-spotted. These tests pin both halves:
a safe-listed flow is stamped Benign with the model untouched; a non-safe flow
reaches the model.
"""
from __future__ import annotations

import asyncio

import pytest

from ids.apps.monitor import config
from ids.apps.monitor.detector import Detector
from ids.apps.monitor.events import Broker
from ids.apps.monitor.windower import WindowResult


class _FakeGate:
    """Stand-in gate predictor: records whether predict() was called and always
    returns 'Attack' so we can prove the safe-list path skips it."""
    x_columns = ['Rate']  # only needs to exist for the df.select() in _handle

    def __init__(self):
        self.called = False

    def predict(self, df, timer=None):
        self.called = True
        return {'labels': ['Attack'], 'confidences': [0.99],
                'class_names': ['Attack', 'Benign'], 'probabilities': [[0.99, 0.01]]}


class _FakeFamily(_FakeGate):
    def predict(self, df, timer=None):
        self.called = True
        return {'labels': ['DDoS'], 'confidences': [0.95],
                'class_names': ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai',
                                'Recon', 'Spoofing', 'Web'],
                'probabilities': [[0.0, 0.0, 0.95, 0.05, 0.0, 0.0, 0.0, 0.0]]}


def _wr(ip_a, ip_b):
    feats = {'Rate': 100.0}
    return WindowResult(features=feats, ip_a=ip_a, ip_b=ip_b,
                        n_packets=10, ts_start=0.0, ts_end=1.0)


def _run_handle(detector, wr):
    async def go():
        await detector._handle(wr, dequeued_at=0.0)
    asyncio.run(go())


def _make_detector():
    gate, fam = _FakeGate(), _FakeFamily()
    det = Detector(producer=None, gate_predictor=gate, family_predictor=fam,
                   broker=Broker())
    return det, gate, fam


def test_safelisted_flow_is_benign_without_model(monkeypatch):
    monkeypatch.setattr(config, 'SAFE_IPS', {'8.8.8.8'})
    det, gate, fam = _make_detector()
    _run_handle(det, _wr('8.8.8.8', '10.0.0.5'))
    assert gate.called is False           # model skipped
    assert fam.called is False
    assert det.stats['by_family']['Benign'] == 1
    assert det.stats['malicious'] == 0


def test_non_safe_flow_hits_the_model(monkeypatch):
    monkeypatch.setattr(config, 'SAFE_IPS', {'8.8.8.8'})
    det, gate, fam = _make_detector()
    _run_handle(det, _wr('185.10.0.9', '10.0.0.5'))   # attacker, not safe-listed
    assert gate.called is True            # model consulted
    assert det.stats['malicious'] == 1    # fake gate says Attack


def test_empty_safelist_classifies_everything(monkeypatch):
    monkeypatch.setattr(config, 'SAFE_IPS', set())
    det, gate, fam = _make_detector()
    _run_handle(det, _wr('8.8.8.8', '10.0.0.5'))
    assert gate.called is True            # no safe-list => model always runs
