"""#1 Feature contract — the model consumes exactly 25 columns, in a fixed order.

The dangerous failure here is silent: drift between the config selection and the
saved column list yields garbage predictions with no error. These tests pin both.
"""
import joblib
import pytest

import ids.core.config as config


def test_selected_feature_count():
    """Pure config invariant — needs no artifacts, runs anywhere."""
    assert config.N_FEATURES_SELECTED == 25
    assert len(config.X_COLUMNS_SELECTED) == 25
    # Selection is exactly the 39-feature set minus the documented drops.
    assert set(config.X_COLUMNS_SELECTED) == set(config.X_COLUMNS) - config.DROPPED_FEATURES
    # No dropped feature leaked back in; no duplicates.
    assert not (set(config.X_COLUMNS_SELECTED) & config.DROPPED_FEATURES)
    assert len(set(config.X_COLUMNS_SELECTED)) == len(config.X_COLUMNS_SELECTED)


@pytest.mark.needs_model
def test_config_matches_saved_columns(models_dir):
    """The config selection must equal the columns persisted at training time —
    order included, since the scaler and model consume columns positionally."""
    saved = list(joblib.load(models_dir / 'feature_columns.joblib'))
    assert config.X_COLUMNS_SELECTED == saved
