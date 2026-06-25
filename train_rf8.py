"""Train ONLY the RF 8-class model and save it to models/, reusing the same
split, the saved preprocessor, and the saved 8-class encoder so it is consistent
with the models already trained. Does not retrain or overwrite anything else.
"""
import time

import joblib

from ids.core.config import SEED, MODELS_DIR
from ids.core.labels import remap_labels
from ids.data.preprocessing import SPLIT_FUNCS
from ids.training.data import load_dataset
from ids.training.trainers import train_rf
from ids.training import artifacts

SPLIT, MODE = 'random', '8'

print('Loading dataset...')
X_all, y_all_34, source_csv = load_dataset()
tr, va, te = SPLIT_FUNCS[SPLIT](y_all_34, source_csv, SEED)
print(f'  train {len(tr):,} rows')

prep = joblib.load(MODELS_DIR / f'preprocessor_{SPLIT}.joblib')
le = joblib.load(MODELS_DIR / f'label_encoder_{SPLIT}_{MODE}class.joblib')

X_train = prep.transform(X_all[tr])
y_tr = le.transform(remap_labels(y_all_34[tr], MODE))

print(f'Training RF {MODE}-class on {X_train.shape[0]:,} x {X_train.shape[1]} ...')
t0 = time.time()
rf = train_rf(X_train, y_tr, MODE)
print(f'  done in {time.time() - t0:.0f}s')

artifacts.save_tree_model(rf, SPLIT, MODE, 'rf')
out = artifacts.tree_model_path(SPLIT, MODE, 'rf')
print(f'Saved {out} ({out.stat().st_size / 1e6:.0f} MB)')
