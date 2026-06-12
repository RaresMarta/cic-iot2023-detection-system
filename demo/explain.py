"""SHAP-based per-flow explainability for the live demo.

Uses shap.GradientExplainer rather than DeepExplainer — DeepExplainer has additivity-check
failures on torch 2.x (the project runs torch 2.6). The explainer is built once at startup
over a small stratified background sample; per-flow explanations are then cheap enough to
compute inline in the stream.

A positive SHAP value pushes the prediction *toward* the predicted class. We translate that
into an attack/benign direction for the UI based on whether the predicted class is Benign.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import shap
import torch


class FlowExplainer:
    def __init__(self, predictor, sampler, background_size: int = 100, seed: int = 42):
        self.predictor = predictor
        self.feature_names: list[str] = list(predictor.x_columns)
        self.class_names: list[str] = list(predictor.encoder.classes_)

        # Stratified background across the green families the demo uses.
        per = max(1, background_size // len(sampler.families))
        frames = [sampler.sample_flows(f, n=per, seed=seed) for f in sampler.families]
        bg_scaled = predictor.preprocess(pl.concat(frames)).astype(np.float32)
        bg_tensor = torch.from_numpy(bg_scaled).to(predictor.device).cpu().numpy()

        predictor.model.eval()
        self.explainer = shap.GradientExplainer(predictor.model, bg_tensor)

    def _class_contrib(self, shap_values, class_idx: int) -> np.ndarray:
        """Normalize shap output (list or ndarray — varies by version) to a 1-D contribution
        vector for the given class."""
        n = len(self.feature_names)
        sv = shap_values
        if isinstance(sv, list):                 # list[n_classes] of [1, n_features]
            return np.asarray(sv[class_idx]).reshape(-1)[:n]
        sv = np.asarray(sv)
        if sv.ndim == 3:                         # [1, n_features, n_classes]
            return sv[0, :, class_idx]
        if sv.ndim == 2:                         # [1, n_features]
            return sv[0]

        return sv.reshape(-1)[:n]

    def explain(self, x_scaled: np.ndarray, raw_values: dict, pred_class_idx: int,
                top_k: int = 8) -> list[dict]:
        """Return the top_k features driving this prediction, each with its raw value, signed
        SHAP contribution, and attack/benign direction for the UI."""
        xt = torch.from_numpy(x_scaled.reshape(1, -1).astype(np.float32)).to(self.predictor.device)
        contrib = self._class_contrib(self.explainer.shap_values(xt), pred_class_idx)

        pred_is_attack = self.class_names[pred_class_idx] != 'Benign'
        out: list[dict] = []
        for i in np.argsort(-np.abs(contrib))[:top_k]:
            supports_pred = contrib[i] > 0
            if pred_is_attack:
                direction = 'attack' if supports_pred else 'benign'
            else:
                direction = 'benign' if supports_pred else 'attack'
            name = self.feature_names[i]
            out.append({
                'feature': name,
                'value': raw_values.get(name),
                'shap': float(contrib[i]),
                'direction': direction,
            })

        return out


if __name__ == '__main__':
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from demo.inference import IDSPredictor
    from demo.sampler import FlowSampler

    predictor = IDSPredictor(PROJECT_ROOT / 'models', split='temporal', mode='8')
    sampler = FlowSampler()
    explainer = FlowExplainer(predictor, sampler)

    for fam in ('Benign', 'DDoS', 'Recon'):
        df = sampler.sample_flows(fam, n=1, seed=1)
        scaled = predictor.preprocess(df)
        pred = predictor.predict(df)
        idx = int(np.argmax(pred['probabilities'][0]))
        reasons = explainer.explain(scaled[0], df.row(0, named=True), idx, top_k=4)
        print(f'\n[{fam}] -> {pred["labels"][0]} ({pred["confidences"][0]:.2f})')
        for r in reasons:
            print(f'   {r["feature"]:16s} val={r["value"]!s:>10}  shap={r["shap"]:+.4f}  -> {r["direction"]}')
