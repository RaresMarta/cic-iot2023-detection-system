"""Lightweight span timing for the serving paths.

A Timer accumulates named spans measured with a monotonic clock and exposes them
as a flat dict of milliseconds. Used to decompose end-to-end latency on the
analyzer and live-monitor paths into stages (extract / preprocess / inference /
explain / serialize), so the dashboard can show where the time actually went and
the pure-inference cost can be reported in isolation.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


class Timer:
    """Accumulates named spans in milliseconds. Not thread-safe; one per request/window."""

    def __init__(self) -> None:
        self._spans: dict[str, float] = {}

    @contextmanager
    def span(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._spans[name] = self._spans.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def record(self, name: str, ms: float) -> None:
        self._spans[name] = self._spans.get(name, 0.0) + ms

    def as_dict(self, round_to: int = 3) -> dict[str, float]:
        return {k: round(v, round_to) for k, v in self._spans.items()}


class _NullTimer:
    """No-op timer so callers can always pass one without a None-check."""

    @contextmanager
    def span(self, name: str):
        yield

    def record(self, name: str, ms: float) -> None:
        pass

    def as_dict(self, round_to: int = 3) -> dict[str, float]:
        return {}


NULL_TIMER = _NullTimer()
