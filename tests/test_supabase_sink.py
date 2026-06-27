"""Supabase backplane sink (SupabaseSink) — request contract, no network or model needed.

Exercises the sync handlers directly with an injected transport that records the HTTP
calls the sink would make. The async run() loop just wires these to the broker, so testing
the handlers is fast and deterministic.
"""
from __future__ import annotations

import json

from ids.apps.monitor.supabase_sink import SupabaseSink


class FakeTransport:
    """Records (method, url, headers, parsed-body); returns canned responses."""

    def __init__(self, raise_on=None):
        self.calls = []
        self.raise_on = raise_on            # substring -> raise to simulate a network error

    def __call__(self, method, url, headers, body):
        if self.raise_on and self.raise_on in url:
            raise OSError('simulated network failure')
        self.calls.append({'method': method, 'url': url, 'headers': headers,
                           'body': json.loads(body) if body else None})
        if '/rest/v1/monitors' in url and method == 'POST':
            return 200, json.dumps([{'id': 'uuid-123', 'monitor_key': 'm1'}])
        return 201, ''

    def find(self, method, needle):
        return [c for c in self.calls if c['method'] == method and needle in c['url']]


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _sink(transport, clock=None, **kw):
    clock = clock or Clock()
    return SupabaseSink('https://proj.supabase.co/', 'service-key', 'm1', 'Droplet One',
                        broker=None, protected_ips=['10.0.0.5'], clock=clock, transport=transport, **kw)


def test_register_monitor_upserts_and_captures_uuid():
    tr = FakeTransport()
    s = _sink(tr)
    uuid = s.register_monitor()
    assert uuid == 'uuid-123'
    assert s.monitor_uuid == 'uuid-123'
    call = tr.find('POST', '/rest/v1/monitors')[0]
    assert 'on_conflict=monitor_key' in call['url']
    assert call['headers']['Prefer'] == 'resolution=merge-duplicates,return=representation'
    assert call['headers']['apikey'] == 'service-key'
    row = call['body'][0]
    assert row['monitor_key'] == 'm1' and row['name'] == 'Droplet One'
    assert row['protected_ips'] == ['10.0.0.5']


def test_alert_inserts_incident_tagged_with_monitor():
    tr = FakeTransport()
    s = _sink(tr)
    s.register_monitor()
    s.handle_event({'type': 'alert', 'ts': 100.0, 'attacker_ip': '185.1.2.3',
                    'family': 'DDoS', 'confidence': 0.97,
                    'top_features': [{'feature': 'Rate', 'contribution': 1.2}]})
    call = tr.find('POST', '/rest/v1/incidents')[0]
    row = call['body'][0]
    assert row['monitor_id'] == 'uuid-123'
    assert row['attacker_ip'] == '185.1.2.3' and row['family'] == 'DDoS'
    assert row['started_ts'] == 100.0 and row['status'] == 'active'
    assert row['top_features'][0]['feature'] == 'Rate'


def test_alert_skipped_before_registration():
    tr = FakeTransport()
    s = _sink(tr)                            # not registered -> monitor_uuid is None
    s.handle_event({'type': 'alert', 'ts': 1.0, 'attacker_ip': 'A', 'family': 'DoS'})
    assert tr.find('POST', '/rest/v1/incidents') == []


def test_recovered_patches_active_incident_for_source():
    tr = FakeTransport()
    s = _sink(tr)
    s.register_monitor()
    s.handle_event({'type': 'recovered', 'ts': 112.5, 'attacker_ip': '185.1.2.3'})
    call = tr.find('PATCH', '/rest/v1/incidents')[0]
    assert 'monitor_id=eq.uuid-123' in call['url']
    assert 'attacker_ip=eq.185.1.2.3' in call['url']
    assert 'status=eq.active' in call['url']
    assert call['body'] == {'ended_ts': 112.5, 'status': 'recovered'}


def test_recovered_without_ip_is_a_noop():
    tr = FakeTransport()
    s = _sink(tr)
    s.register_monitor()
    s.handle_event({'type': 'recovered', 'ts': 5.0})        # no attacker_ip
    assert tr.find('PATCH', '/rest/v1/incidents') == []


def test_snapshot_posts_counters_and_heartbeat():
    tr = FakeTransport()
    s = _sink(tr)
    s.register_monitor()
    s.snapshot({'flows_total': 50, 'malicious': 12, 'dropped': 3, 'uptime_s': 7.5,
                'by_family': {'DDoS': 12, 'Benign': 38}})
    snap = tr.find('POST', '/rest/v1/stats_snapshots')[0]['body'][0]
    assert snap['monitor_id'] == 'uuid-123'
    assert (snap['flows_total'], snap['malicious'], snap['dropped']) == (50, 12, 3)
    assert snap['by_family'] == {'DDoS': 12, 'Benign': 38}
    # heartbeat bumps last_seen on the monitors row
    hb = tr.find('PATCH', '/rest/v1/monitors')[0]
    assert 'id=eq.uuid-123' in hb['url'] and hb['body']['status'] == 'online'


def test_flow_events_are_not_persisted_via_handle_event():
    tr = FakeTransport()
    s = _sink(tr)
    s.register_monitor()
    s.handle_event({'type': 'flow', 'family': 'Benign', 'src': 'x', 'dst': 'y'})
    assert tr.find('POST', '/rest/v1/incidents') == []
    assert tr.find('POST', '/realtime/v1/api/broadcast') == []   # broadcast is on the run path


def test_broadcast_flows_targets_realtime_channel():
    tr = FakeTransport()
    s = _sink(tr)
    s.register_monitor()
    s._broadcast_flows([{'type': 'flow', 'flow_id': 7, 'family': 'DoS'}])
    call = tr.find('POST', '/realtime/v1/api/broadcast')[0]
    msg = call['body']['messages'][0]
    assert msg['topic'] == 'flows:uuid-123'
    assert msg['event'] == 'flow'
    assert msg['payload']['flow_id'] == 7


def test_token_bucket_throttles_to_flow_rate():
    tr = FakeTransport()
    clk = Clock(1000.0)
    s = _sink(tr, clock=clk, flow_rate=5.0)
    # bucket starts full at capacity 5 -> first 5 pass, 6th is dropped (no time elapsed)
    assert sum(s._take_token() for _ in range(5)) == 5
    assert s._take_token() is False
    # advance 1 second -> refills 5 tokens (capped) -> 5 more pass
    clk.t += 1.0
    assert sum(s._take_token() for _ in range(5)) == 5
    assert s._take_token() is False


def test_flow_rate_zero_means_unlimited():
    s = _sink(FakeTransport(), flow_rate=0.0)
    assert all(s._take_token() for _ in range(100))


def test_handle_event_swallows_transport_errors():
    tr = FakeTransport(raise_on='/rest/v1/incidents')
    s = _sink(tr)
    s.register_monitor()
    # transport raises inside _send; handler must not propagate
    s.handle_event({'type': 'alert', 'ts': 1.0, 'attacker_ip': 'A', 'family': 'DoS'})
