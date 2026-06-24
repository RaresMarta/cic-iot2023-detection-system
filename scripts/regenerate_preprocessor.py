"""Regenerate models/preprocessor_random.joblib so it matches the deployed models.

The saved preprocessor drifted out of sync with the trained models, mis-scaling
every input and forcing all-attack predictions. The correct preprocessor is the
RobustScaler-based Preprocessor fit on the same train split the models saw.
"""
import numpy as np, polars as pl, joblib
from ids.data.preprocessing import split_random, fit_preprocess
from ids.core.config import SEED, X_COLUMNS_SELECTED, LOG_COLUMNS_SELECTED, MODELS_DIR

df = pl.read_parquet('data/cic_iot_2023.parquet')
y34 = df['Label'].to_numpy()
Xall = df.select(list(X_COLUMNS_SELECTED)).to_numpy().astype(np.float32)
tr, va, te = split_random(y34, np.zeros(len(y34)), SEED)
_, _, _, prep = fit_preprocess(Xall, tr, va, te,
    x_columns=list(X_COLUMNS_SELECTED), log_columns=list(LOG_COLUMNS_SELECTED))
joblib.dump(prep, MODELS_DIR / 'preprocessor_random.joblib')
print('regenerated preprocessor_random.joblib')
