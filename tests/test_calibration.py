"""#5 Calibration format — the saved temperature is a valid positive scalar per mode.

If this drifts, temperature scaling silently applies a wrong factor across every
probability the model emits.
"""
import joblib
import pytest


@pytest.mark.needs_model
def test_temperature_scaling_format(models_dir):
    path = models_dir / 'temperature_scaling.joblib'
    if not path.exists():
        pytest.skip('no calibration artifact present')

    cal = joblib.load(path)
    assert isinstance(cal, dict)
    for mode in ('2', '8'):
        assert mode in cal, f'missing calibration for {mode}-class'
        temp = cal[mode]['T']
        assert isinstance(temp, float)
        assert temp > 0
