"""#2 Offline ↔ live parity — the same packets must yield the same features.

The file path (extract_features, whole pcap) and the live path (StreamWindower,
one packet at a time) share the same windowing + aggregation code. This test
feeds both the identical packets and asserts the 25 features match, so an
"optimisation" to one path can't silently diverge live results from the demo.
"""
import pytest

from ids.runtime.extractor import extract_features
from ids.apps.monitor.windower import StreamWindower


@pytest.mark.needs_model  # extract_features loads feature_columns.joblib from models/
def test_offline_and_live_produce_identical_features(synth_pcap, single_pair_frames):
    # Offline: the whole capture collapses to one full window.
    df = extract_features(synth_pcap, window=10, include_partial=False)
    assert df.height == 1
    offline = df.row(0, named=True)

    # Live: same packets, fed one frame at a time.
    win = StreamWindower(window=10)
    results = [r for ts, buf in single_pair_frames if (r := win.add(ts, buf)) is not None]
    assert len(results) == 1
    live = results[0].features

    for col in df.columns:
        assert live[col] == pytest.approx(offline[col]), f'feature {col!r} diverges'
