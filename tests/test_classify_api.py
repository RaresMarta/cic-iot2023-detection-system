"""#4 /api/classify end-to-end — the contract with the React frontend.

A CSV goes in; a result with the agreed shape comes out. SHAP is optional: when
the explainer is available `top_features` appears in the agreed shape; when it
isn't, classification still succeeds (graceful degradation).
"""
import pytest

from ids.runtime.extractor import extract_features


@pytest.fixture
def csv_bytes(synth_pcap):
    df = extract_features(synth_pcap, window=10)
    return df.write_csv().encode()


@pytest.mark.needs_model
def test_classify_endpoint_contract(csv_bytes):
    from fastapi.testclient import TestClient
    from ids.apps.analyzer.app import api

    client = TestClient(api)
    resp = client.post(
        '/api/classify',
        files={'file': ('capture.csv', csv_bytes, 'text/csv')},
        data={'model_type': 'mlp', 'mode': '8', 'split': 'random'},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get('success') is True
    assert 'top_label' in body
    assert isinstance(body.get('probabilities'), dict) and body['probabilities']

    # SHAP explanation is optional but, when present, must match the frontend shape.
    if 'top_features' in body:
        assert len(body['top_features']) <= 8
        assert all({'feature', 'contribution'} <= set(f) for f in body['top_features'])
