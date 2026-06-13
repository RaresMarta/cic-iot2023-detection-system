"""Window producers — where the detector's flow windows come from.

The detector classifies WindowResults and doesn't care how they are produced:

  PacketWindowProducer  — real path: a packet source (live NIC or pcap replay) fed
                          through the streaming windower. Runs the packet-rate loop.
  SampledWindowProducer — offline-demo path: real CIC-IoT-2023 flows sampled from the
                          parquet (FlowSampler), wrapped as WindowResults with synthetic
                          endpoints, with an inject queue for on-demand attack bursts.
                          Lets the full classify -> decide -> enforce -> publish path be
                          demoed on macOS with in-distribution traffic (so DDoS/DoS/Mirai/
                          Recon actually classify correctly and trigger bans).

Both run in the detector's capture thread via run(emit, stop).
"""
from __future__ import annotations

import random
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from pathlib import Path


from . import config
from .capture import IDLE_TICK, PacketSource
from .windower import StreamWindower, WindowResult

EmitFn = Callable[[WindowResult], None]


class WindowProducer(ABC):
    mode: str = 'unknown'

    @abstractmethod
    def run(self, emit: EmitFn, stop: threading.Event) -> None:
        """Produce WindowResults via emit() until stop is set or the source ends."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class PacketWindowProducer(WindowProducer):
    def __init__(self, source: PacketSource, windower: StreamWindower | None = None):
        self.source = source
        self.windower = windower or StreamWindower()
        self.mode = source.mode

    def run(self, emit: EmitFn, stop: threading.Event) -> None:
        last_flush = 0.0
        try:
            for item in self.source.packets():
                if stop.is_set():
                    break
                ts, buf = item
                if (ts, buf) == IDLE_TICK:                  # live recv timeout
                    for wr in self.windower.flush_idle(time.time()):
                        emit(wr)
                    continue
                wr = self.windower.add(ts, buf)
                if wr is not None:
                    emit(wr)
                if ts - last_flush >= self.windower.idle_flush_s:
                    for wr in self.windower.flush_idle(ts):
                        emit(wr)
                    last_flush = ts
        finally:
            for wr in self.windower.flush_idle(float('inf')):
                emit(wr)

    def close(self) -> None:
        self.source.close()


class SampledWindowProducer(WindowProducer):
    mode = 'simulate'

    def __init__(self, sampler, inject_queue: deque | None = None,
                 idle_interval: float = 0.6, burst_interval: float = 0.15):
        self.sampler = sampler
        self.inject_queue = inject_queue if inject_queue is not None else deque()
        self.idle_interval = idle_interval
        self.burst_interval = burst_interval
        self._rng = random.Random(1234)
        # Stable source IP per attack family, so an injected burst comes from one
        # attacker and consecutive malicious windows accumulate into a ban
        # (mirrors a real single-source attack). Assigned lazily on first use.
        self._attacker_ip: dict[str, str] = {}

    def _endpoints(self, family: str) -> tuple[str, str]:
        protected = next(iter(config.PROTECTED_IPS))
        if family == 'Benign':
            return f'172.30.0.{self._rng.randint(20, 254)}', protected
        ip = self._attacker_ip.get(family)
        if ip is None:
            ip = self._attacker_ip[family] = (
                f'185.{self._rng.randint(1, 254)}.{self._rng.randint(1, 254)}.{self._rng.randint(1, 254)}')
        return ip, protected

    def run(self, emit: EmitFn, stop: threading.Event) -> None:
        while not stop.is_set():
            family = self.inject_queue.popleft() if self.inject_queue else 'Benign'
            try:
                df = self.sampler.sample_flows(family, n=1)
                feats = df.row(0, named=True)
                src, dst = self._endpoints(family)
                now = time.time()
                emit(WindowResult(features=feats, ip_a=src, ip_b=dst,
                                  n_packets=config.WINDOW, ts_start=now, ts_end=now))
            except Exception as e:
                print(f'[producer:simulate] {e}', flush=True)
            time.sleep(self.burst_interval if self.inject_queue else self.idle_interval)


def from_config():
    """Build the window producer + (optional) inject queue selected by config."""
    if config.SOURCE == 'simulate':
        from ids.data.sampler import FlowSampler
        q: deque = deque()
        return SampledWindowProducer(FlowSampler(), inject_queue=q), q
    from . import capture
    source = capture.from_config()
    return PacketWindowProducer(source), None
