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


from ids.core.timing import Timer
from . import config
from .events import Broker
from .producers import WindowProducer
from .windower import WindowResult

torch.set_num_threads(1)


def attacker_ip(ip_a: str, ip_b: str, protected: set[str]) -> str | None:
    """The endpoint of a flow that is NOT the protected host. None if ambiguous."""
    a_prot, b_prot = ip_a in protected, ip_b in protected
    if a_prot and not b_prot:
        return ip_b
    if b_prot and not a_prot:
        return ip_a
    return None


class Detector:
    def __init__(self, producer: WindowProducer, gate_predictor, family_predictor,
                 broker: Broker):
        self.producer = producer
        self.gate_predictor = gate_predictor
        self.family_predictor = family_predictor
        self.broker = broker
        self._q: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._tasks: list[asyncio.Task] = []
        self._flow_id = 0
        self._started_at = time.time()
        self._active: dict[str, dict] = {}
        self.stats = {
            'flows_total': 0,
            'by_family': defaultdict(int),
            'dropped': 0,
            'malicious': 0,
        }

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
        self._q.put(None)
        for t in self._tasks:
            t.cancel()

    def _produce_loop(self) -> None:
        try:
            self.producer.run(self._enqueue, self._stop)
        finally:
            self._q.put(None)

    def _enqueue(self, wr: WindowResult) -> None:
        wr._enqueued_at = time.perf_counter()   # for queue-wait latency in _handle
        try:
            self._q.put_nowait(wr)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.stats['dropped'] += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(wr)
            except queue.Full:
                pass

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            wr = await loop.run_in_executor(None, self._q.get)
            if wr is None:
                break
            dequeued_at = time.perf_counter()
            try:
                await self._handle(wr, dequeued_at)
            except Exception as e:
                print(f'[detector] handle error: {e}', flush=True)

    async def _handle(self, wr: WindowResult, dequeued_at: float) -> None:
        timer = self._start_timer(wr, dequeued_at)
        verdict = self._classify(wr, timer)
        self._record(verdict)
        self._publish_flow(wr, verdict, timer, dequeued_at)
        self._track_episode(wr, verdict)

    def _start_timer(self, wr: WindowResult, dequeued_at: float) -> Timer:
        timer = Timer()
        enq = getattr(wr, '_enqueued_at', None)
        if enq is not None:
            timer.record('queue_wait_ms', (dequeued_at - enq) * 1000.0)
        return timer

    def _is_safelisted(self, wr: WindowResult) -> bool:
        return bool(config.SAFE_IPS) and (
            wr.ip_a in config.SAFE_IPS or wr.ip_b in config.SAFE_IPS)

    def _classify(self, wr: WindowResult, timer: Timer) -> dict:
        df = pl.DataFrame([wr.features]).select(self.family_predictor.x_columns)
        safelisted = self._is_safelisted(wr)
        if safelisted:
            gate_label, gate_conf = 'Benign', 1.0
        else:
            gate = self.gate_predictor.predict(df, timer=timer)
            gate_label, gate_conf = str(gate['labels'][0]), float(gate['confidences'][0])
        malicious = gate_label != 'Benign'

        if malicious:
            fam = self.family_predictor.predict(df, timer=timer)
            family, fam_conf = str(fam['labels'][0]), float(fam['confidences'][0])
            probs = {n: float(p) for n, p in zip(fam['class_names'], fam['probabilities'][0])}
        else:
            family, fam_conf = 'Benign', gate_conf
            probs = self._benign_probs()

        attacker = attacker_ip(wr.ip_a, wr.ip_b, config.PROTECTED_IPS)
        protected = wr.ip_b if attacker == wr.ip_a else wr.ip_a
        return {'safelisted': safelisted, 'malicious': malicious, 'family': family,
                'gate_conf': gate_conf, 'fam_conf': fam_conf, 'probs': probs,
                'attacker': attacker, 'protected': protected}

    @staticmethod
    def _benign_probs() -> dict:
        return {'Benign': 1.0, 'DDoS': 0.0, 'DoS': 0.0, 'Mirai': 0.0,
                'Recon': 0.0, 'Spoofing': 0.0, 'Web': 0.0, 'BruteForce': 0.0}

    def _record(self, verdict: dict) -> None:
        self.stats['flows_total'] += 1
        self.stats['by_family'][verdict['family']] += 1
        if verdict['malicious']:
            self.stats['malicious'] += 1

    def _publish_flow(self, wr: WindowResult, verdict: dict, timer: Timer,
                      dequeued_at: float) -> None:
        self._flow_id += 1
        spans = timer.as_dict()
        spans['detect_ms'] = round((time.perf_counter() - dequeued_at) * 1000.0, 3)
        self.broker.publish({
            'type': 'flow',
            'flow_id': self._flow_id,
            'ts': time.time(),
            'src': verdict['attacker'] or wr.ip_a,
            'dst': verdict['protected'],
            'family': verdict['family'],
            'gate': 'block' if verdict['malicious'] else 'allow',
            'gate_confidence': verdict['gate_conf'],
            'confidence': verdict['fam_conf'],
            'probabilities': verdict['probs'],
            'top_features': [],
            'n_packets': wr.n_packets,
            'safelisted': verdict['safelisted'],
            'timing': spans,
        })

    def _track_episode(self, wr: WindowResult, verdict: dict) -> None:
        attacker = verdict['attacker']
        if not verdict['malicious'] or attacker is None:
            return
        now = time.time()
        state = self._active.get(attacker)
        if state is None:
            self._active[attacker] = {'last_malicious': now, 'family': verdict['family']}
            self.broker.publish({'type': 'alert', 'ts': now, 'attacker_ip': attacker,
                                 'family': verdict['family'],
                                 'confidence': verdict['gate_conf'], 'top_features': []})
        else:
            state['last_malicious'] = now
            state['family'] = verdict['family']

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

    def snapshot_stats(self) -> dict:
        return {
            'flows_total': self.stats['flows_total'],
            'malicious': self.stats['malicious'],
            'by_family': dict(self.stats['by_family']),
            'dropped': self.stats['dropped'],
            'uptime_s': round(time.time() - self._started_at, 1),
        }
