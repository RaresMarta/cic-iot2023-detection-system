"""POST /api/classify-flow — manual single-flow classification.

A JSON dict of the 25 features goes in; a single-flow result in the same shape as
/api/classify comes out. Missing features are rejected with a clear error; both
binary and 8-class modes return their respective class sets.
"""
import pytest

from ids.core.config import X_COLUMNS_SELECTED


def _full_features() -> dict:
    """A complete 25-feature flow (a high-rate flood-ish shape)."""
    base = {c: 0.0 for c in X_COLUMNS_SELECTED}
    base.update({'Rate': 10000.0, 'IAT': 0.0001, 'Number': 100.0,
                 'syn_flag_number': 1.0, 'TCP': 1.0, 'Min': 54.0, 'Max': 54.0,
                 'AVG': 54.0, 'Tot sum': 5400.0})
    return base


@pytest.mark.needs_model
def test_classify_flow_contract():
    from fastapi.testclient import TestClient
    from ids.apps.analyzer.app import api

    client = TestClient(api)
    resp = client.post('/api/classify-flow', json={
        'features': _full_features(), 'model_type': 'mlp', 'mode': '8', 'split': 'random'})

    assert resp.status_code == 200
    body = resp.json()
    assert body.get('success') is True
    assert body.get('input_type') == 'manual'
    assert body.get('flow_count') == 1
    assert isinstance(body.get('probabilities'), dict) and body['probabilities']


@pytest.mark.needs_model
def test_classify_flow_missing_feature_errors():
    from fastapi.testclient import TestClient
    from ids.apps.analyzer.app import api

    feats = _full_features()
    feats.pop('Rate')  # drop one required feature

    client = TestClient(api)
    resp = client.post('/api/classify-flow', json={
        'features': feats, 'model_type': 'mlp', 'mode': '8', 'split': 'random'})

    body = resp.json()
    assert 'error' in body
    assert 'Rate' in body['error']


@pytest.mark.needs_model
def test_classify_flow_binary_vs_8class():
    from fastapi.testclient import TestClient
    from ids.apps.analyzer.app import api

    client = TestClient(api)
    feats = _full_features()

    b2 = client.post('/api/classify-flow', json={
        'features': feats, 'model_type': 'mlp', 'mode': '2', 'split': 'random'}).json()
    b8 = client.post('/api/classify-flow', json={
        'features': feats, 'model_type': 'mlp', 'mode': '8', 'split': 'random'}).json()

    assert set(b2['class_names']) == {'Attack', 'Benign'}
    assert 'DDoS' in b8['class_names'] and len(b8['class_names']) == 8
