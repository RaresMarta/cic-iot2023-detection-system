"""ntfy notifier — event→push contract, no model and no network.

Replaces the actual HTTP call with a recorder, so the routing/formatting logic is tested
deterministically; one test exercises the real _post error path with a raising urlopen.
"""
from __future__ import annotations

import ids.apps.monitor.notifier as notifier_mod
from ids.apps.monitor.notifier import NtfyNotifier


def _recording_notifier(on_recover=True):
    n = NtfyNotifier('https://ntfy.test/topic', broker=None, on_recover=on_recover)
    calls = []
    n._post = lambda title, body, priority, tags: calls.append(
        {'title': title, 'body': body, 'priority': priority, 'tags': tags})
    return n, calls


def test_alert_pushes_high_priority_with_source():
    n, calls = _recording_notifier()
    n.handle_event({'type': 'alert', 'attacker_ip': '185.1.2.3',
                    'family': 'DDoS', 'confidence': 0.97})
    assert len(calls) == 1
    c = calls[0]
    assert c['priority'] == 'high'
    assert 'DDoS' in c['title']
    assert '185.1.2.3' in c['body']
    assert '97%' in c['body']            # confidence rendered as a percentage


def test_flow_events_do_not_push():
    n, calls = _recording_notifier()
    n.handle_event({'type': 'flow', 'family': 'Benign', 'src': 'x', 'dst': 'y'})
    assert calls == []


def test_recovered_pushes_when_enabled_and_not_when_disabled():
    n, calls = _recording_notifier(on_recover=True)
    n.handle_event({'type': 'recovered', 'attacker_ip': '185.1.2.3'})
    assert len(calls) == 1 and calls[0]['priority'] == 'low'

    n2, calls2 = _recording_notifier(on_recover=False)
    n2.handle_event({'type': 'recovered', 'attacker_ip': '185.1.2.3'})
    assert calls2 == []


def test_alert_without_confidence_still_pushes():
    n, calls = _recording_notifier()
    n.handle_event({'type': 'alert', 'attacker_ip': 'A', 'family': 'Mirai'})
    assert len(calls) == 1
    assert 'Mirai' in calls[0]['body']   # no percentage, but still a valid message


def test_post_swallows_network_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError('connection refused')
    monkeypatch.setattr(notifier_mod.urllib.request, 'urlopen', boom)
    n = NtfyNotifier('https://ntfy.test/topic')
    # must not raise even though the HTTP call fails
    n._post(title='t', body='b', priority='high', tags='rotating_light')
