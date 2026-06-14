"""Shared fixtures + the needs_model auto-skip.

Tests run from the project root; we put it on sys.path so `import ids...`
resolves without an editable install. Synthetic packets are reused from
ids/runtime/validate_extractor.py — the pre-existing extractor self-test that
this suite promotes into real tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / 'models'

from ids.runtime.validate_extractor import _synth_syn_flood, _write_pcap  # noqa: E402


def _have_models() -> bool:
    return (MODELS_DIR / 'feature_columns.joblib').exists() and \
           (MODELS_DIR / 'ids_dnn_random_8class.pth').exists()


def pytest_collection_modifyitems(config, items):
    """Auto-skip needs_model tests on a machine without trained artifacts, so the
    contract-only tests (#1 config, #2 parity, #6 imports) still run anywhere."""
    if _have_models():
        return
    skip = pytest.mark.skip(reason='trained artifacts not present in models/')
    for item in items:
        if 'needs_model' in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def models_dir() -> Path:
    return MODELS_DIR


@pytest.fixture(scope='session')
def predictor():
    """The 8-class random-split predictor — the model the upload UI uses."""
    from ids.runtime.predictor import MLPClassifier
    return MLPClassifier(MODELS_DIR, split='random', mode='8')


@pytest.fixture
def single_pair_frames():
    """10 packets between one host pair → exactly one full window at window=10."""
    return _synth_syn_flood(n=10, t0=1000.0)


@pytest.fixture
def synth_pcap(tmp_path, single_pair_frames) -> Path:
    p = tmp_path / 'capture.pcap'
    _write_pcap(p, single_pair_frames)
    return p
