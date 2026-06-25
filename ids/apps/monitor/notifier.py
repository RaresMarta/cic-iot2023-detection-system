"""ntfy push notifier — a broker subscriber that pushes a phone notification when an
attack episode starts (and, optionally, when it clears).

The detector already debounces per source IP: it emits exactly one ``alert`` per episode
and one ``recovered`` when the source goes quiet (see ``detector.py``). So this notifier
needs no debounce of its own — it just reacts to those two event types.

Opt-in (``IDS_NTFY_ENABLED`` + ``IDS_NTFY_URL``) and off the detection path. Uses stdlib
``urllib.request`` (zero new deps, works in the slim detector image). A failed push is
logged and swallowed — notifications must never affect detection.

ntfy contract: POST the message as the body to the topic URL; the ``Title``/``Priority``/
``Tags`` HTTP headers set the notification's title, priority, and emoji tags.
"""
from __future__ import annotations

import asyncio
import urllib.request

from .events import Broker


class NtfyNotifier:
    def __init__(self, url: str, broker: Broker | None = None,
                 on_recover: bool = True, timeout: float = 5.0):
        self.url = url
        self.broker = broker
        self.on_recover = on_recover
        self.timeout = timeout

    # ── sync event -> push (directly unit-testable) ───────────────────────────
    def handle_event(self, evt: dict) -> None:
        t = evt.get('type')
        if t == 'alert':
            fam = evt.get('family', 'attack')
            ip = evt.get('attacker_ip', '?')
            conf = evt.get('confidence')
            conf_s = f' ({conf:.0%})' if isinstance(conf, (int, float)) else ''
            self._post(title=f'IDS alert: {fam}', body=f'{fam} from {ip}{conf_s}',
                       priority='high', tags='rotating_light')
        elif t == 'recovered' and self.on_recover:
            ip = evt.get('attacker_ip', '?')
            self._post(title='IDS cleared', body=f'{ip} went quiet',
                       priority='low', tags='white_check_mark')

    def _post(self, title: str, body: str, priority: str, tags: str) -> None:
        try:
            req = urllib.request.Request(
                self.url, data=body.encode('utf-8'), method='POST',
                headers={'Title': title, 'Priority': priority, 'Tags': tags})
            urllib.request.urlopen(req, timeout=self.timeout).close()
        except Exception as e:                       # network/DNS/timeout — never propagate
            print(f'[notifier] push failed: {e}', flush=True)

    # ── long-lived broker consumer ────────────────────────────────────────────
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        q = self.broker.subscribe()
        try:
            while True:
                evt = await q.get()
                await loop.run_in_executor(None, self.handle_event, evt)
        except asyncio.CancelledError:
            pass
        finally:
            self.broker.unsubscribe(q)
