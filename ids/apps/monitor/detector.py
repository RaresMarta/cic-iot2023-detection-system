"""Detector orchestrator: capture -> window -> classify -> decide -> publish.

Threading model (see plan):
  * One capture thread does all packet-rate work (recv + parse + windowing) and pushes
    only completed WindowResults onto a bounded queue, dropping oldest on overflow.
  * One asyncio consumer task pulls WindowResults, runs the (sub-ms, batch-1) MLP inline,
    raises an alert on sustained malicious traffic, and publishes events to the SSE broker.
  * One asyncio lifecycle task emits "recovered" when an attacker goes quiet.
All windower state stays in the capture thread, so no locks are needed.
"""
from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import polars as pl
import torch


from . import config
from .events import Broker
from .producers import WindowProducer
from .windower import WindowResult

torch.set_num_threads(1)   # the model is tiny; threading hurts more than helps


def attacker_ip(ip_a: str, ip_b: str, protected: set[str]) -> str | None:
    """The endpoint of a flow that is NOT the protected host. None if ambiguous."""
    a_prot, b_prot = ip_a in protected, ip_b in protected
    if a_prot and not b_prot:
        return ip_b
    if b_prot and not a_prot:
        return ip_a
    return None  # neither or both protected -> can't attribute the source


class Detector:
    def __init__(self, producer: WindowProducer, gate_predictor, family_predictor,
                 broker: Broker, explainer=None):
        self.producer = producer
        self.gate_predictor = gate_predictor       # 2-class Benign/Attack (alert trigger)
        self.family_predictor = family_predictor   # 8-class family (label/visual)
        self.broker = broker
        self.explainer = explainer                 # optional gate SHAP; proxy fallback
        self._q: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._tasks: list[asyncio.Task] = []
        self._flow_id = 0
        self._started_at = time.time()
        # active attackers: ip -> {last_malicious, family}
        self._active: dict[str, dict] = {}
        self.stats = {
            'flows_total': 0,
            'by_family': defaultdict(int),
            'dropped': 0,
            'malicious': 0,
        }

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._capture_thread = threading.Thread(target=self._produce_loop, daemon=True)
        self._capture_thread.start()
        self._tasks = [
            asyncio.create_task(self._consume()),
            asyncio.create_task(self._lifecycle_loop()),
        ]

    async def stop(self) -> None:
        self._stop.set()
        self.producer.close()
        self._q.put(None)            # unblock the consumer
        for t in self._tasks:
            t.cancel()

    # ── capture thread: producer emits WindowResults onto the bounded queue ────
    def _produce_loop(self) -> None:
        try:
            self.producer.run(self._enqueue, self._stop)
        finally:
            self._q.put(None)                               # signal end-of-stream

    def _enqueue(self, wr: WindowResult) -> None:
        try:
            self._q.put_nowait(wr)
        except queue.Full:
            try:
                self._q.get_nowait()                        # drop oldest
                self.stats['dropped'] += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(wr)
            except queue.Full:
                pass

    # ── async consumer (classify + decide + publish) ───────────────────────────
    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            wr = await loop.run_in_executor(None, self._q.get)
            if wr is None:                                  # end-of-stream sentinel
                break
            try:
                await self._handle(wr, loop)
            except Exception as e:                          # never die on one window
                print(f'[detector] handle error: {e}', flush=True)

    async def _handle(self, wr: WindowResult, loop) -> None:
        df = pl.DataFrame([wr.features]).select(self.family_predictor.x_columns)
        # 2-class gate drives the decision; 8-class gives the family label/probabilities.
        gate = self.gate_predictor.predict(df)
        gate_label = str(gate['labels'][0])          # 'Benign' | 'Attack'
        gate_conf = float(gate['confidences'][0])
        malicious = gate_label != 'Benign'
        
        if malicious:
            fam = self.family_predictor.predict(df)
            family = str(fam['labels'][0])
            fam_conf = float(fam['confidences'][0])
            probs = {n: float(p) for n, p in zip(fam['class_names'], fam['probabilities'][0])}
        else:
            family = 'Benign'
            fam_conf = gate_conf
            probs = {'Benign': 1.0, 'DDoS': 0.0, 'DoS': 0.0, 'Mirai': 0.0,
                     'Recon': 0.0, 'Spoofing': 0.0, 'Web': 0.0, 'BruteForce': 0.0}

        atk = attacker_ip(wr.ip_a, wr.ip_b, config.PROTECTED_IPS)
        protected = wr.ip_b if atk == wr.ip_a else wr.ip_a

        self.stats['flows_total'] += 1
        self.stats['by_family'][family] += 1
        if malicious:
            self.stats['malicious'] += 1

        self._flow_id += 1
        now = time.time()
        self.broker.publish({
            'type': 'flow',
            'flow_id': self._flow_id,
            'ts': now,
            'src': atk or wr.ip_a,
            'dst': protected,
            'family': family,
            'gate': 'block' if malicious else 'allow',
            'gate_confidence': gate_conf,
            'confidence': fam_conf,
            'probabilities': probs,
            'top_features': self._explain(wr.features, probs),
            'n_packets': wr.n_packets,
        })

        if malicious and atk is not None:
            state = self._active.get(atk)
            if state is None:                               # first malicious window
                self._active[atk] = {'last_malicious': now, 'family': family}
                # One real SHAP attribution per attack episode (here, not per window);
                # runs off the event loop and falls back to the per-window proxy.
                reasons = await loop.run_in_executor(None, self._shap_gate, df, wr.features)
                if not reasons:
                    reasons = self._explain(wr.features, probs)
                self.broker.publish({'type': 'alert', 'ts': now, 'attacker_ip': atk,
                                     'family': family, 'confidence': gate_conf,
                                     'top_features': reasons})
            else:
                state['last_malicious'] = now
                state['family'] = family

    # ── lifecycle: emit "recovered" when an attacker goes quiet ────────────────
    async def _lifecycle_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
                now = time.time()
                for atk in [a for a, s in self._active.items()
                            if now - s['last_malicious'] >= config.RECOVER_AFTER_S]:
                    self._active.pop(atk, None)
                    self.broker.publish({'type': 'recovered', 'ts': now, 'attacker_ip': atk})
        except asyncio.CancelledError:
            pass

    # ── per-alert SHAP attribution for the gate's Attack verdict ───────────────
    def _shap_gate(self, df: pl.DataFrame, features: dict, top_k: int = 6) -> list[dict] | None:
        """Real SHAP for why the 2-class gate flagged this source. Runs once per
        attack episode (in the consumer's executor); None on any failure."""
        if self.explainer is None:
            return None
        try:
            classes = list(self.gate_predictor.encoder.classes_)
            attack_idx = classes.index('Attack') if 'Attack' in classes else 0
            x_scaled = self.gate_predictor.preprocess(df)[0]
            reasons = self.explainer.explain(x_scaled, features, attack_idx, top_k=top_k)
            return [{'feature': r['feature'], 'contribution': r['shap'],
                     'direction': r['direction']} for r in reasons]
        except Exception as e:
            print(f'[detector] gate SHAP skipped: {e}', flush=True)
            return None

    # ── cheap per-window saliency proxy (every flow); SHAP runs per alert ──────
    def _explain(self, features: dict, probs: dict, top_k: int = 5) -> list[dict]:
        # |scaled value| as a quick saliency proxy; ranks which features stood out.
        try:
            scaled = self.family_predictor.preprocess(
                pl.DataFrame([features]).select(self.family_predictor.x_columns))[0]
            order = sorted(range(len(scaled)), key=lambda i: -abs(float(scaled[i])))[:top_k]
            return [{'feature': self.family_predictor.x_columns[i],
                     'contribution': round(float(scaled[i]), 4)} for i in order]
        except Exception:
            return []

    # ── read-only views for the API ───────────────────────────────────────────
    def snapshot_stats(self) -> dict:
        return {
            'flows_total': self.stats['flows_total'],
            'malicious': self.stats['malicious'],
            'by_family': dict(self.stats['by_family']),
            'dropped': self.stats['dropped'],
            'uptime_s': round(time.time() - self._started_at, 1),
        }
