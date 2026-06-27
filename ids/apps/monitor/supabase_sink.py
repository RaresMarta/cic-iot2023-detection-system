"""Supabase backplane sink — a broker subscriber that pushes the detector's output to
Supabase so a remote (Vercel) dashboard can consume it without reaching this worker.

A broker consumer that never touches the detection path. Three jobs:

  * register — on startup, upsert this worker's row in the ``monitors`` table so the
    dashboard picker discovers it (multi-tenant = one worker per environment, each with a
    stable ``monitor_key``; all workers share one Supabase project).
  * broadcast flows — each ``flow`` event is sent over Supabase Realtime *Broadcast*
    (ephemeral, never written to a table) on a per-monitor channel, throttled to a display
    rate so a flood cannot drown the feed or the browser. The feed is a sample for the eye;
    the aggregate counters stay truthful.
  * persist incidents + snapshots — ``alert``/``recovered`` become rows in ``incidents``
    and periodic ``stats_snapshots`` rows, the durable record the live stream forgets.

Opt-in (``IDS_SUPABASE_ENABLED``). Uses stdlib ``urllib.request`` (zero new deps, works in
the slim detector image) via an injectable ``transport`` so the handlers are unit-testable
without a network. Every failure is logged and swallowed — the backplane must never crash
detection. Writes use a service_role key (kept server-side here) so they bypass RLS.
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from .events import Broker


class SupabaseSink:
    """Push detector events to Supabase (Realtime Broadcast + Postgres REST).

    Sync handlers (``handle_event``/``snapshot``/``_broadcast_flows``) are directly
    unit-testable; ``run()`` wires them to the broker as a long-lived background consumer.
    """

    _FLUSH_S = 0.25
    _BATCH_CAP = 500

    def __init__(self, url: str, key: str, monitor_key: str, monitor_name: str,
                 broker: Broker | None = None, *, owner_id: str = '', public_ip: str = '',
                 protected_ips=(), flow_rate: float = 25.0, snapshot_s: float = 15.0,
                 timeout: float = 5.0, transport=None, clock=time.time):
        self.url = url.rstrip('/')
        self.key = key
        self.monitor_key = monitor_key
        self.monitor_name = monitor_name or monitor_key
        self.broker = broker
        self.owner_id = owner_id
        self.public_ip = public_ip
        self.protected_ips = list(protected_ips)
        self.flow_rate = float(flow_rate)
        self.snapshot_s = snapshot_s
        self.timeout = timeout
        self._transport = transport or self._default_transport
        self._clock = clock
        self._headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
        }
        self.monitor_uuid: str | None = None
        self._tokens = self.flow_rate
        self._last_refill = self._clock()
        self._flow_batch: list[dict] = []
        self._display_dropped = 0

    def _default_transport(self, method: str, url: str, headers: dict, body):
        data = body.encode('utf-8') if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.status, resp.read().decode('utf-8')

    def _send(self, method: str, url: str, body_obj=None, extra_headers: dict | None = None):
        headers = dict(self._headers)
        if extra_headers:
            headers.update(extra_headers)
        body = json.dumps(body_obj) if body_obj is not None else None
        try:
            status, text = self._transport(method, url, headers, body)
            if status is not None and status >= 300:
                print(f'[supabase] {method} {url} -> {status}: {str(text)[:200]}', flush=True)
                return None
            if text:
                try:
                    return json.loads(text)
                except (TypeError, ValueError):
                    return None
            return None
        except Exception as e:
            print(f'[supabase] {method} {url} failed: {e}', flush=True)
            return None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def register_monitor(self) -> str | None:
        """Upsert this worker's monitors row (by monitor_key) and capture its uuid."""
        url = f'{self.url}/rest/v1/monitors?on_conflict=monitor_key'
        row = {
            'monitor_key': self.monitor_key,
            'name': self.monitor_name,
            'public_ip': self.public_ip or None,
            'protected_ips': self.protected_ips,
            'status': 'online',
            'last_seen_at': self._now_iso(),
        }
        if self.owner_id:
            row['owner_id'] = self.owner_id
        body = [row]
        res = self._send('POST', url, body,
                         {'Prefer': 'resolution=merge-duplicates,return=representation'})
        if isinstance(res, list) and res and res[0].get('id'):
            self.monitor_uuid = res[0]['id']
            print(f'[supabase] monitor registered: {self.monitor_key} -> {self.monitor_uuid}',
                  flush=True)
        return self.monitor_uuid

    def handle_event(self, evt: dict) -> None:
        try:
            t = evt.get('type')
            if t == 'alert':
                self._on_alert(evt)
            elif t == 'recovered':
                self._on_recovered(evt)
        except Exception as e:
            print(f'[supabase] handle_event error: {e}', flush=True)

    def _on_alert(self, evt: dict) -> None:
        if not self.monitor_uuid:
            return
        body = [{
            'monitor_id': self.monitor_uuid,
            'attacker_ip': evt.get('attacker_ip'),
            'family': evt.get('family'),
            'confidence': evt.get('confidence'),
            'started_ts': evt.get('ts', self._clock()),
            'top_features': evt.get('top_features'),
            'status': 'active',
        }]
        self._send('POST', f'{self.url}/rest/v1/incidents', body)

    def _on_recovered(self, evt: dict) -> None:
        ip = evt.get('attacker_ip')
        if not self.monitor_uuid or not ip:
            return
        ended = evt.get('ts', self._clock())
        q = (f'{self.url}/rest/v1/incidents'
             f'?monitor_id=eq.{self.monitor_uuid}'
             f'&attacker_ip=eq.{urllib.parse.quote(str(ip))}'
             f'&status=eq.active')
        self._send('PATCH', q, {'ended_ts': ended, 'status': 'recovered'})

    def snapshot(self, stats: dict) -> None:
        if not self.monitor_uuid:
            return
        body = [{
            'monitor_id': self.monitor_uuid,
            'ts': self._clock(),
            'flows_total': stats.get('flows_total'),
            'malicious': stats.get('malicious'),
            'dropped': stats.get('dropped'),
            'uptime_s': stats.get('uptime_s'),
            'by_family': stats.get('by_family'),
        }]
        self._send('POST', f'{self.url}/rest/v1/stats_snapshots', body)
        self._send('PATCH', f'{self.url}/rest/v1/monitors?id=eq.{self.monitor_uuid}',
                   {'last_seen_at': self._now_iso(), 'status': 'online'})

    def _take_token(self) -> bool:
        """Return True if a flow may be broadcast now (rate-limited to flow_rate/sec)."""
        if self.flow_rate <= 0:
            return True
        now = self._clock()
        self._tokens = min(self.flow_rate,
                           self._tokens + (now - self._last_refill) * self.flow_rate)
        self._last_refill = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _broadcast_flows(self, batch: list[dict]) -> None:
        if not batch:
            return
        topic = f'flows:{self.monitor_uuid or self.monitor_key}'
        msgs = [{'topic': topic, 'event': 'flow', 'payload': f} for f in batch]
        self._send('POST', f'{self.url}/realtime/v1/api/broadcast', {'messages': msgs})

    async def run(self, detector) -> None:
        """Register, then drain broker events: throttle+broadcast flows on a fast cadence,
        persist incidents immediately, snapshot stats every snapshot_s. Cancelled cleanly."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.register_monitor)
        q = self.broker.subscribe()
        last_snap = 0.0
        last_flush = 0.0
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=self._FLUSH_S)
                    t = evt.get('type')
                    if t == 'flow':
                        if self._take_token() and len(self._flow_batch) < self._BATCH_CAP:
                            self._flow_batch.append(evt)
                        else:
                            self._display_dropped += 1
                    elif t in ('alert', 'recovered'):
                        await loop.run_in_executor(None, self.handle_event, evt)
                except asyncio.TimeoutError:
                    pass

                now = self._clock()
                if self._flow_batch and now - last_flush >= self._FLUSH_S:
                    last_flush = now
                    batch, self._flow_batch = self._flow_batch, []
                    await loop.run_in_executor(None, self._broadcast_flows, batch)
                if now - last_snap >= self.snapshot_s:
                    last_snap = now
                    if not self.monitor_uuid:
                        await loop.run_in_executor(None, self.register_monitor)
                    await loop.run_in_executor(None, self.snapshot, detector.snapshot_stats())
        except asyncio.CancelledError:
            pass
        finally:
            self.broker.unsubscribe(q)
