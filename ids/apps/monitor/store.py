"""SQLite event store — a broker subscriber that persists what the detector finds.

The detector publishes ``alert``/``recovered``/``flow`` events to the in-process
``Broker`` (see ``events.py``); the SSE endpoint is one consumer. This sink is a second,
independent consumer that writes the durable record the live stream otherwise forgets:

  * incidents     — one row per attack episode (alert -> recovered), correlated by source IP
  * stats_snapshots — periodic counters, the macro/window-level view of the stream

It is opt-in (``IDS_DB_ENABLED``) and never touches the detection path. SQLite writes run in
the event loop's default executor so the blocking I/O cannot stall the loop, mirroring how
the detector offloads its own blocking work (``detector.py``). Any DB error is logged and
swallowed — persistence must never crash detection.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path

from .events import Broker

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    attacker_ip  TEXT    NOT NULL,
    family       TEXT,
    confidence   REAL,
    started_ts   REAL    NOT NULL,
    ended_ts     REAL,
    duration_s   REAL,
    top_features TEXT,
    status       TEXT    NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS ix_incidents_active ON incidents (attacker_ip, status);

CREATE TABLE IF NOT EXISTS stats_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    flows_total INTEGER,
    malicious   INTEGER,
    dropped     INTEGER,
    uptime_s    REAL,
    by_family   TEXT
);
"""


class SqliteSink:
    """Persist detector events to SQLite. Sync handlers are unit-testable on their own;
    ``run()`` wires them to the broker as a long-lived background consumer."""

    def __init__(self, db_path: str | Path, broker: Broker | None = None,
                 snapshot_s: float = 15.0):
        self.db_path = str(db_path)
        self.broker = broker
        self.snapshot_s = snapshot_s
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: writes are funnelled one-at-a-time through the loop's
        # executor (and tests call the handlers directly), never concurrently.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ── sync handlers (directly unit-testable) ────────────────────────────────
    def handle_event(self, evt: dict) -> None:
        try:
            t = evt.get('type')
            if t == 'alert':
                self._on_alert(evt)
            elif t == 'recovered':
                self._on_recovered(evt)
            # 'flow' events are intentionally not persisted (a flood is thousands of them);
            # incidents + periodic snapshots are the durable record, not every benign flow.
        except Exception as e:                       # never let one event kill the sink
            print(f'[store] handle_event error: {e}', flush=True)

    def _on_alert(self, evt: dict) -> None:
        self.conn.execute(
            'INSERT INTO incidents (attacker_ip, family, confidence, started_ts, '
            'top_features, status) VALUES (?, ?, ?, ?, ?, ?)',
            (evt.get('attacker_ip'), evt.get('family'), evt.get('confidence'),
             evt.get('ts', time.time()), json.dumps(evt.get('top_features')), 'active'))
        self.conn.commit()

    def _on_recovered(self, evt: dict) -> None:
        ended = evt.get('ts', time.time())
        # close the most recent still-active incident for this source
        self.conn.execute(
            "UPDATE incidents SET ended_ts = ?, duration_s = ? - started_ts, "
            "status = 'recovered' WHERE id = ("
            "  SELECT id FROM incidents WHERE attacker_ip = ? AND status = 'active' "
            "  ORDER BY id DESC LIMIT 1)",
            (ended, ended, evt.get('attacker_ip')))
        self.conn.commit()

    def snapshot(self, stats: dict) -> None:
        try:
            self.conn.execute(
                'INSERT INTO stats_snapshots (ts, flows_total, malicious, dropped, '
                'uptime_s, by_family) VALUES (?, ?, ?, ?, ?, ?)',
                (time.time(), stats.get('flows_total'), stats.get('malicious'),
                 stats.get('dropped'), stats.get('uptime_s'),
                 json.dumps(stats.get('by_family'))))
            self.conn.commit()
        except Exception as e:
            print(f'[store] snapshot error: {e}', flush=True)

    # ── read view (for the /api/incidents endpoint) ───────────────────────────
    def recent_incidents(self, limit: int = 50) -> list[dict]:
        cur = self.conn.execute(
            'SELECT attacker_ip, family, confidence, started_ts, ended_ts, duration_s, '
            'status, top_features FROM incidents ORDER BY id DESC LIMIT ?', (int(limit),))
        cols = [c[0] for c in cur.description]
        rows = []
        for r in cur.fetchall():
            d = dict(zip(cols, r))
            if d.get('top_features'):
                try:
                    d['top_features'] = json.loads(d['top_features'])
                except (TypeError, ValueError):
                    pass
            rows.append(d)
        return rows

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ── long-lived broker consumer ────────────────────────────────────────────
    async def run(self, detector) -> None:
        """Drain broker events into the store; snapshot detector stats every snapshot_s.
        Cancelled cleanly on shutdown."""
        loop = asyncio.get_running_loop()
        q = self.broker.subscribe()
        last_snap = 0.0
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=self.snapshot_s)
                    await loop.run_in_executor(None, self.handle_event, evt)
                except asyncio.TimeoutError:
                    pass
                now = time.time()
                if now - last_snap >= self.snapshot_s:
                    last_snap = now
                    await loop.run_in_executor(None, self.snapshot, detector.snapshot_stats())
        except asyncio.CancelledError:
            pass
        finally:
            self.broker.unsubscribe(q)
            self.close()
