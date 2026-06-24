import numpy as np, polars as pl
from ids.runtime.predictor import MLPClassifier
from ids.data.preprocessing import split_random
from ids.core.config import SEED, X_COLUMNS_SELECTED, MODELS_DIR
from ids.core.labels import remap_labels


def test_mlp_indist_fpr_matches_artifact():
    df = pl.read_parquet('data/cic_iot_2023.parquet')
    y34 = df['Label'].to_numpy()
    _, _, te = split_random(y34, np.zeros(len(y34)), SEED)
    test_df = df[te].select(list(X_COLUMNS_SELECTED))
    y_true = np.array([s.lower() for s in remap_labels(y34[te], '2')])
    out = MLPClassifier(MODELS_DIR, split='random', mode='2').predict(test_df)
    pred = np.array([str(l).lower() for l in out['labels']])
    fp = ((y_true == 'benign') & (pred == 'attack')).sum()
    tn = ((y_true == 'benign') & (pred == 'benign')).sum()
    fpr = fp / (fp + tn)
    assert fpr < 0.05, f"MLP in-dist FPR={fpr:.3f} (expected ~0.02). Stale preprocessor?"


def test_cross_dataset_predict_indist_sane():
    import polars as pl, numpy as np
    from ids.eval.cross_dataset_eval import predict, model_feature_columns
    from ids.data.preprocessing import split_random
    from ids.core.config import SEED
    from ids.core.labels import remap_labels
    df = pl.read_parquet('data/cic_iot_2023.parquet'); y34 = df['Label'].to_numpy()
    _, _, te = split_random(y34, np.zeros(len(y34)), SEED)
    feats = df[te].select(model_feature_columns())
    lab, _ = predict(feats, model='mlp')
    yt = np.array([s.lower() for s in remap_labels(y34[te], '2')])
    fpr = ((yt == 'benign') & (lab == 'attack')).sum() / (yt == 'benign').sum()
    assert fpr < 0.05, f"harness predict() in-dist FPR={fpr:.3f}"
