"""In-process SSE broker: fan out detector events to many dashboard clients.

Each subscriber gets its own bounded asyncio.Queue. A slow client only drops its
own oldest events; it can never apply backpressure to detection or to other clients.
"""
from __future__ import annotations

import asyncio


class Broker:
    def __init__(self, maxsize: int = 200):
        self._subs: set[asyncio.Queue] = set()
        self._maxsize = maxsize

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, event: dict) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()          # drop this client's oldest
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)
