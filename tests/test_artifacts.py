"""#3 Artifact round-trip — scaler + encoder + model load and predict.

Guards against "moved the files / changed the save format and now nothing loads".
"""
import pytest

from ids.runtime.extractor import extract_features


@pytest.mark.needs_model
def test_predictor_loads_and_predicts(predictor, synth_pcap):
    df = extract_features(synth_pcap, window=10)
    n = df.height
    assert n >= 1

    pred = predictor.predict(df)
    n_classes = len(predictor.encoder.classes_)

    assert pred['probabilities'].shape == (n, n_classes)
    # Each flow's probabilities form a valid distribution.
    assert pred['probabilities'].sum(axis=1) == pytest.approx([1.0] * n, abs=1e-4)
    assert len(pred['labels']) == n
    # Predicted labels are always within the encoder's known classes.
    assert set(pred['labels']).issubset(set(predictor.encoder.classes_))
