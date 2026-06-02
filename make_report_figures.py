"""Regenerate all result figures from the consistent 25-feature re-run, and perform
temperature-scaling calibration (ECE before/after + reliability diagram).
Run after regenerate_results.py.
"""
from __future__ import annotations
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

FIG = 'docs/report/figures'
R = joblib.load('_results_full.joblib')
print('loaded _results_full.joblib')

def metr(mode, model, split, key):
    return R[(f'mode{mode}', model, split)][key]

MODELS = ['mlp', 'rf', 'xgb']
LABELS = {'mlp': 'MLP', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
COLORS = {'mlp': '#5f8dd3', 'rf': '#3aa17e', 'xgb': '#e07a5f'}
SPLITS = ['train', 'val', 'test']

# ── 1. Confusion matrices (row-normalised) ──────────────────────────────────
for mode in ['2', '8']:
    cm = np.array(R[f'mlp_cm_{mode}'], dtype=np.float64)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    cls = R[f'class_names_{mode}']
    d = max(5, len(cls) * 0.7)
    plt.figure(figsize=(d, d * 0.85))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=cls, yticklabels=cls, cbar=True, vmin=0, vmax=1)
    plt.title(f'MLP temporal {mode}-class confusion matrix (row-normalised = recall)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(f'{FIG}/confusion_{mode}class.png', dpi=140); plt.close()
    print(f'wrote confusion_{mode}class.png')

# ── 2. Training curves (2-class | 8-class) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, mode, title in ((axes[0], '2', 'Binary (2-class)'), (axes[1], '8', '8-class')):
    h = joblib.load(f'models/run_artifacts_temporal_{mode}class.joblib')['history']
    ep = range(1, len(h['train_loss']) + 1)
    ax.plot(ep, h['train_loss'], '-o', ms=3, label='train', color='#e07a5f')
    ax.plot(ep, h['val_loss'], '-o', ms=3, label='val', color='#3aa17e')
    ax.set_title(f'{title} — class-weighted CE'); ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{FIG}/training_curves.png', dpi=140); plt.close()
print('wrote training_curves.png')

# ── 3. Splits figures (8-class): weighted_f1, macro_f1, accuracy, generalization
def splits_bar(key, fname, title, ylabel):
    x = np.arange(len(SPLITS)); w = 0.25
    plt.figure(figsize=(8, 5))
    for i, m in enumerate(MODELS):
        vals = [metr('8', m, s, key) for s in SPLITS]
        plt.bar(x + (i - 1) * w, vals, w, label=LABELS[m], color=COLORS[m], edgecolor='black', linewidth=0.4)
    plt.xticks(x, [s.capitalize() for s in SPLITS]); plt.ylabel(ylabel); plt.title(title)
    plt.ylim(0, 1); plt.legend(); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{FIG}/{fname}', dpi=140); plt.close()
    print(f'wrote {fname}')

splits_bar('weighted_f1', 'weighted_f1_splits.png', 'Weighted F1 across splits (8-class, temporal)', 'Weighted F1')
splits_bar('macro_f1', 'macro_f1_splits.png', 'Macro F1 across splits (8-class, temporal)', 'Macro F1')
splits_bar('accuracy', 'model_comparison.png', 'Accuracy across splits (8-class, temporal)', 'Accuracy')
splits_bar('accuracy', 'generalization.png', 'Train/Val/Test accuracy by model (8-class)', 'Accuracy')

# ── 4. Feature importance (perm, RF+XGB avg, top 20) ────────────────────────
pi = R['perm_importance_8']
feats = np.array(pi['features']); imp = np.array(pi['importance'])
order = np.argsort(imp)[::-1][:20]
plt.figure(figsize=(8, 6))
plt.barh([feats[i] for i in order][::-1], [imp[i] for i in order][::-1],
         color='#5f8dd3', edgecolor='black', linewidth=0.4)
plt.xlabel('Permutation importance (mean macro-F1 drop, RF+XGB avg)')
plt.title('Top 20 features by permutation importance (8-class, temporal)')
plt.tight_layout(); plt.savefig(f'{FIG}/feature_importance.png', dpi=140); plt.close()
print('wrote feature_importance.png')

# ── 5. Temperature-scaling calibration (8-class) ────────────────────────────
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

def nll(T, logits, y):
    p = softmax(logits / T)
    return -np.mean(np.log(p[np.arange(len(y)), y].clip(1e-12)))

def ece(probs, y, n_bins=15):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return e

calib = {}
for mode in ['8', '2']:
    d = np.load(f'models/mlp_logits_temporal_{mode}class.npz')
    vlog, vy, tlog, ty = d['val_logits'], d['val_y'], d['test_logits'], d['test_y']
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded', args=(vlog, vy))
    T = float(res.x)
    p_before = softmax(tlog); p_after = softmax(tlog / T)
    e_before, e_after = ece(p_before, ty), ece(p_after, ty)
    calib[mode] = {'T': T, 'ece_before': e_before, 'ece_after': e_after}
    print(f'mode {mode}: T*={T:.3f}  ECE {e_before:.4f} -> {e_after:.4f}')
    if mode == '8':
        # reliability diagram (8-class test)
        nb = 15; bins = np.linspace(0, 1, nb + 1); mids = (bins[:-1] + bins[1:]) / 2
        def curve(p):
            conf = p.max(1); pred = p.argmax(1); acc = (pred == ty).astype(float)
            xs, ys = [], []
            for i in range(nb):
                m = (conf > bins[i]) & (conf <= bins[i + 1])
                if m.sum() > 30:
                    xs.append(conf[m].mean()); ys.append(acc[m].mean())
            return xs, ys
        plt.figure(figsize=(5.5, 5.5))
        plt.plot([0, 1], [0, 1], '--', color='gray', label='perfect calibration')
        xb, yb = curve(p_before); xa, ya = curve(p_after)
        plt.plot(xb, yb, '-o', ms=4, color='#e07a5f', label=f'uncalibrated (ECE={e_before:.3f})')
        plt.plot(xa, ya, '-o', ms=4, color='#3aa17e', label=f'T-scaled, T={T:.2f} (ECE={e_after:.3f})')
        plt.xlabel('Confidence (max softmax)'); plt.ylabel('Accuracy')
        plt.title('Reliability diagram — MLP, 8-class temporal test')
        plt.legend(); plt.grid(alpha=0.3); plt.xlim(0, 1); plt.ylim(0, 1)
        plt.tight_layout(); plt.savefig(f'{FIG}/reliability_8class.png', dpi=140); plt.close()
        print('wrote reliability_8class.png')

joblib.dump(calib, 'models/temperature_scaling.joblib')

# ── Print paper-ready number summary ────────────────────────────────────────
print('\n================ PAPER NUMBERS ================')
for mode in ['2', '8']:
    print(f'\n--- {mode}-class temporal TEST ---')
    print(f'{"model":<14}{"acc":>8}{"wF1":>8}{"macroF1":>9}{"valW-F1":>9}{"valMacro":>9}')
    for m in MODELS:
        print(f'{LABELS[m]:<14}{metr(mode,m,"test","accuracy"):>8.4f}{metr(mode,m,"test","weighted_f1"):>8.4f}'
              f'{metr(mode,m,"test","macro_f1"):>9.4f}{metr(mode,m,"val","weighted_f1"):>9.4f}'
              f'{metr(mode,m,"val","macro_f1"):>9.4f}')
print('\n--- 8-class MLP per-class report ---')
print(R['mlp_report_8'])
print('\n--- calibration ---')
print(json.dumps(calib, indent=2))
print('\n--- 8-class confusion (MLP), row-normalised recall + top off-diagonal ---')
cm = np.array(R['mlp_cm_8'], dtype=float); cls = R['class_names_8']
cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
for i, c in enumerate(cls):
    row = cmn[i].copy(); row[i] = -1
    j = row.argmax()
    print(f'  {c:<11} recall={cmn[i,i]:.2f}  main->{cls[j]} ({cmn[i,j]*100:.0f}%)')

with open('_paper_numbers.json', 'w') as f:
    json.dump({
        'test': {mode: {m: R[(f'mode{mode}', m, 'test')] for m in MODELS} for mode in ['2', '8']},
        'val':  {mode: {m: R[(f'mode{mode}', m, 'val')] for m in MODELS} for mode in ['2', '8']},
        'calibration': calib,
    }, f, indent=2)
print('\nwrote _paper_numbers.json')
