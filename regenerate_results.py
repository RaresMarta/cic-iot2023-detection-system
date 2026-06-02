"""Full 25-feature re-run: retrain MLP + RF + XGBoost at 2-class and 8-class on the
CURRENT temporal split, saving consistent artifacts + train/val/test metrics +
MLP logits (for calibration) + permutation importance. CPU-only friendly.
"""
from __future__ import annotations
import os
os.environ.setdefault('JOBLIB_TEMP_FOLDER', r'G:\uni\_jobtmp')  # keep memmap temp off the small C: drive
import json, time
import numpy as np
import polars as pl
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)
import xgboost as xgb

from config import (X_COLUMNS as ORIG_X, FLAG_COLUMNS as ORIG_FLAGS, Y_COLUMN, SEED,
                    PARQUET_PATH, MODELS_DIR, BATCH_SIZE, N_EPOCHS, PATIENCE, LR)
from labels import remap_labels
from preprocessing import split_temporal, fit_preprocess
from models import IDSDataset, IDSModel, train_model, evaluate, device

torch.manual_seed(SEED); np.random.seed(SEED)
t_start = time.time()
print(f'device={device}')

# ── Load + clean (notebook cells 14,16,17,18,24,26) ─────────────────────────
df = pl.read_parquet(str(PARQUET_PATH))
feat_cols = [c for c in df.columns if c not in (Y_COLUMN, 'source_csv')]
flag_cols = [c for c in feat_cols if c in set(ORIG_FLAGS)]
log_cols  = [c for c in feat_cols if c not in set(ORIG_FLAGS)]
df = df.drop_nulls(subset=[Y_COLUMN])
df = df.with_columns([pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
                      for c in feat_cols])
df = df.with_columns([pl.col(c).log1p().alias(c) for c in log_cols])
X_all = df.select(feat_cols).to_numpy().astype(np.float32)
y_all_34 = df[Y_COLUMN].to_numpy()
source_csv = df['source_csv'].to_numpy()
del df
n_features = X_all.shape[1]
print(f'features={n_features} (flags={len(flag_cols)} continuous={len(log_cols)}) rows={len(y_all_34):,}')

# ── Temporal split + train-only preprocessing ───────────────────────────────
tr, va, te = split_temporal(y_all_34, source_csv, SEED)
print(f'split: train={len(tr):,} val={len(va):,} test={len(te):,}')
X_train, X_val, X_test, scaler, _ = fit_preprocess(X_all, tr, va, te)
joblib.dump(scaler, MODELS_DIR / 'scaler_temporal.joblib')
joblib.dump(feat_cols, MODELS_DIR / 'feature_columns.joblib')

def m5(yt, yp):
    return dict(accuracy=accuracy_score(yt, yp),
               macro_f1=f1_score(yt, yp, average='macro', zero_division=0),
               weighted_f1=f1_score(yt, yp, average='weighted', zero_division=0),
               macro_precision=precision_score(yt, yp, average='macro', zero_division=0),
               macro_recall=recall_score(yt, yp, average='macro', zero_division=0))

RESULTS = {}   # (mode, model, split) -> metrics ; plus extras under string keys

for mode in ['2', '8']:
    print(f'\n========== MODE {mode}-class ==========')
    le = LabelEncoder().fit(remap_labels(y_all_34, mode))
    class_names = list(le.classes_); K = len(class_names)
    y_tr = le.transform(remap_labels(y_all_34[tr], mode))
    y_va = le.transform(remap_labels(y_all_34[va], mode))
    y_te = le.transform(remap_labels(y_all_34[te], mode))
    joblib.dump(le, MODELS_DIR / f'label_encoder_temporal_{mode}class.joblib')
    RESULTS[f'class_names_{mode}'] = class_names

    # balanced class weights (size K)
    present = np.unique(y_tr)
    w = compute_class_weight('balanced', classes=present, y=y_tr)
    cw = np.ones(K, dtype=np.float32)
    for c, wt in zip(present, w): cw[c] = wt

    # ---- MLP ----
    print('  [MLP] training...')
    tr_loader = DataLoader(IDSDataset(torch.tensor(X_train), torch.tensor(y_tr)),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(IDSDataset(torch.tensor(X_val), torch.tensor(y_va)),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_loader = DataLoader(IDSDataset(torch.tensor(X_test), torch.tensor(y_te)),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = IDSModel(n_features, K, hidden_sizes=[128, 64], dropout=0.3, activation='relu').to(device)
    ckpt = MODELS_DIR / f'ids_dnn_temporal_{mode}class.pth'
    model, history = train_model(model, tr_loader, va_loader,
                                 torch.tensor(cw), N_EPOCHS, PATIENCE, LR, device,
                                 checkpoint_path=ckpt, optimizer_name='adam')

    def mlp_eval(loader):
        return evaluate(model, loader, class_names, device)
    mlp_test = mlp_eval(te_loader); mlp_val = mlp_eval(va_loader); mlp_train = mlp_eval(tr_loader)
    RESULTS[('mode'+mode, 'mlp', 'test')]  = m5(mlp_test['y_true'], mlp_test['y_pred'])
    RESULTS[('mode'+mode, 'mlp', 'val')]   = m5(mlp_val['y_true'], mlp_val['y_pred'])
    RESULTS[('mode'+mode, 'mlp', 'train')] = m5(mlp_train['y_true'], mlp_train['y_pred'])
    RESULTS[f'mlp_report_{mode}'] = mlp_test['report']
    RESULTS[f'mlp_cm_{mode}'] = mlp_test['confusion_matrix']
    joblib.dump({'history': history, 'y_true': mlp_test['y_true'], 'y_pred': mlp_test['y_pred']},
                MODELS_DIR / f'run_artifacts_temporal_{mode}class.joblib')

    # logits for calibration (val + test)
    model.eval()
    def logits_of(Xs):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xs), BATCH_SIZE):
                xb = torch.tensor(Xs[i:i+BATCH_SIZE]).to(device)
                out.append(model(xb).cpu().numpy())
        return np.concatenate(out)
    np.savez(MODELS_DIR / f'mlp_logits_temporal_{mode}class.npz',
             val_logits=logits_of(X_val), val_y=y_va,
             test_logits=logits_of(X_test), test_y=y_te)
    print(f'    MLP test: acc={RESULTS[("mode"+mode,"mlp","test")]["accuracy"]:.4f} '
          f'wF1={RESULTS[("mode"+mode,"mlp","test")]["weighted_f1"]:.4f}')

    # ---- Random Forest ----
    print('  [RF] training...')
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced',
                                n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_tr)
    for split, Xs, ys in (('test', X_test, y_te), ('val', X_val, y_va), ('train', X_train, y_tr)):
        RESULTS[('mode'+mode, 'rf', split)] = m5(ys, rf.predict(Xs))
    RESULTS[f'rf_report_{mode}'] = classification_report(y_te, rf.predict(X_test),
                                       target_names=class_names, zero_division=0, digits=4)

    # ---- XGBoost ----
    print('  [XGB] training...')
    wsamp = np.ones(len(y_tr), dtype=np.float32)
    for c, wt in zip(present, w): wsamp[y_tr == c] = wt
    xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                tree_method='hist', n_jobs=-1, random_state=SEED,
                                objective='binary:logistic' if K == 2 else 'multi:softprob',
                                eval_metric='logloss' if K == 2 else 'mlogloss')
    xgb_clf.fit(X_train, y_tr, sample_weight=wsamp)
    for split, Xs, ys in (('test', X_test, y_te), ('val', X_val, y_va), ('train', X_train, y_tr)):
        RESULTS[('mode'+mode, 'xgb', split)] = m5(ys, xgb_clf.predict(Xs))
    RESULTS[f'xgb_report_{mode}'] = classification_report(y_te, xgb_clf.predict(X_test),
                                        target_names=class_names, zero_division=0, digits=4)

    # stash 8-class trees for post-loop permutation importance
    if mode == '8':
        rf8, xgb8, Xte8, yte8 = rf, xgb_clf, X_test, y_te

# ── Save metrics FIRST (so a perm-importance failure cannot lose them) ──────
def dump_results():
    out = {('|'.join(k) if isinstance(k, tuple) else k):
           (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in RESULTS.items()}
    joblib.dump(RESULTS, '_results_full.joblib')
    with open('_results_summary.json', 'w') as f:
        json.dump({k: v for k, v in out.items()
                   if not k.startswith('mlp_cm') and not k.startswith('perm')}, f, indent=2, default=str)
dump_results()
print(f'\nmetrics saved at {(time.time()-t_start)/60:.1f} min — now permutation importance')

# ── Permutation importance (8-class, RF+XGB avg) — single-process, small temp ──
try:
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Xte8), size=min(10000, len(Xte8)), replace=False)
    pi_rf = permutation_importance(rf8, Xte8[sub], yte8[sub], n_repeats=3,
                                   random_state=SEED, scoring='f1_macro', n_jobs=1)
    pi_xgb = permutation_importance(xgb8, Xte8[sub], yte8[sub], n_repeats=3,
                                    random_state=SEED, scoring='f1_macro', n_jobs=1)
    imp = (pi_rf.importances_mean + pi_xgb.importances_mean) / 2.0
    RESULTS['perm_importance_8'] = {'features': feat_cols, 'importance': imp.tolist()}
    joblib.dump({'features': feat_cols, 'importance': imp}, MODELS_DIR / 'perm_imp_global_test.joblib')
    dump_results()
    print('permutation importance done')
except Exception as e:
    print(f'WARNING: permutation importance failed ({e!r}); falling back to built-in importances')
    imp = (rf8.feature_importances_ + xgb8.feature_importances_) / 2.0
    RESULTS['perm_importance_8'] = {'features': feat_cols, 'importance': imp.tolist(), 'fallback': 'builtin'}
    joblib.dump({'features': feat_cols, 'importance': imp}, MODELS_DIR / 'perm_imp_global_test.joblib')
    dump_results()

print(f'\nDONE in {(time.time()-t_start)/60:.1f} min — wrote _results_full.joblib + _results_summary.json')
