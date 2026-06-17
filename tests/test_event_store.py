"""Event store (SqliteSink) — persistence contract, no model or network needed.

Exercises the sync handlers directly (the async run() loop just wires these to the broker),
so the test is fast and deterministic.
"""
from __future__ import annotations

from ids.apps.monitor.store import SqliteSink


def _sink(tmp_path):
    return SqliteSink(tmp_path / 'events.db', broker=None, snapshot_s=1.0)


def test_alert_then_recovered_closes_one_incident(tmp_path):
    s = _sink(tmp_path)
    s.handle_event({'type': 'alert', 'ts': 100.0, 'attacker_ip': '185.1.2.3',
                    'family': 'DDoS', 'confidence': 0.97,
                    'top_features': [{'feature': 'Rate', 'contribution': 1.2}]})
    rows = s.recent_incidents()
    assert len(rows) == 1
    assert rows[0]['attacker_ip'] == '185.1.2.3'
    assert rows[0]['family'] == 'DDoS'
    assert rows[0]['status'] == 'active'
    assert rows[0]['ended_ts'] is None
    # top_features round-trips through JSON back to a list
    assert rows[0]['top_features'][0]['feature'] == 'Rate'

    s.handle_event({'type': 'recovered', 'ts': 112.5, 'attacker_ip': '185.1.2.3'})
    row = s.recent_incidents()[0]
    assert row['status'] == 'recovered'
    assert row['ended_ts'] == 112.5
    assert row['duration_s'] == 12.5


def test_recovered_closes_only_the_matching_active_incident(tmp_path):
    s = _sink(tmp_path)
    s.handle_event({'type': 'alert', 'ts': 1.0, 'attacker_ip': 'A', 'family': 'DoS'})
    s.handle_event({'type': 'alert', 'ts': 2.0, 'attacker_ip': 'B', 'family': 'Mirai'})
    s.handle_event({'type': 'recovered', 'ts': 9.0, 'attacker_ip': 'A'})

    by_ip = {r['attacker_ip']: r for r in s.recent_incidents()}
    assert by_ip['A']['status'] == 'recovered' and by_ip['A']['duration_s'] == 8.0
    assert by_ip['B']['status'] == 'active' and by_ip['B']['ended_ts'] is None


def test_flow_events_are_not_persisted(tmp_path):
    s = _sink(tmp_path)
    s.handle_event({'type': 'flow', 'family': 'Benign', 'src': 'x', 'dst': 'y'})
    assert s.recent_incidents() == []


def test_snapshot_persists_counters(tmp_path):
    s = _sink(tmp_path)
    s.snapshot({'flows_total': 50, 'malicious': 12, 'dropped': 3, 'uptime_s': 7.5,
                'by_family': {'DDoS': 12, 'Benign': 38}})
    cur = s.conn.execute('SELECT flows_total, malicious, dropped, by_family FROM stats_snapshots')
    flows, mal, dropped, by_family = cur.fetchone()
    assert (flows, mal, dropped) == (50, 12, 3)
    assert '"DDoS"' in by_family            # by_family stored as JSON


def test_schema_creation_is_idempotent(tmp_path):
    SqliteSink(tmp_path / 'events.db', broker=None).close()
    # re-opening the same file must not raise (CREATE TABLE IF NOT EXISTS)
    s = SqliteSink(tmp_path / 'events.db', broker=None)
    assert s.recent_incidents() == []
    s.close()


def test_malformed_event_is_swallowed(tmp_path):
    s = _sink(tmp_path)
    s.handle_event({'type': 'alert'})       # missing fields -> NOT NULL attacker_ip fails
    # handler must not raise; nothing persisted for the bad row
    assert s.recent_incidents() == []
