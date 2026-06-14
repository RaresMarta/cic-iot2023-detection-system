// Customer-site reaction layer. Subscribes to the detector's SSE feed and reflects
// the threat lifecycle on the page. Presentation only — no detection happens here.
(function () {
  const body = document.body;
  const badge = document.getElementById('guardian');
  const stateEl = document.getElementById('guardian-state');
  const detailEl = document.getElementById('guardian-detail');
  const url = (window.DETECTOR_URL || 'http://localhost:7870') + '/api/stream';

  // Minimal status only — no attack-family detail (that lives in the IDS dashboard).
  // The IDS is detect-and-alert: it never blocks, so there is no "blocked" state here.
  function setThreat(level, state, detail) {
    body.dataset.threat = level;
    badge.className = 'guardian guardian--' + level;
    stateEl.textContent = state;
    detailEl.textContent = detail;
  }

  function onEvent(evt) {
    switch (evt.type) {
      case 'alert':
        setThreat('elevated', 'Threat detected', 'analysing traffic');
        break;
      case 'flow':
        // gate === 'block' is the 2-class verdict (malicious), not an enforcement action.
        if (evt.gate === 'block') {
          setThreat('engaged', 'Under attack', 'from ' + evt.src);
        }
        break;
      case 'recovered':
        setThreat('calm', 'Protected', 'recovered');
        break;
    }
  }

  function connect() {
    const es = new EventSource(url);
    es.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)); } catch (_) {}
    };
    es.onerror = () => { detailEl.textContent = 'sensor link reconnecting…'; };
  }
  connect();

  // "Make a request" button — gives visitors a benign interaction to fire.
  const ping = document.getElementById('ping');
  const pingResult = document.getElementById('ping-result');
  if (ping) {
    ping.addEventListener('click', async () => {
      pingResult.textContent = '…';
      try {
        const r = await fetch('/api/products');
        pingResult.textContent = (await r.json()).length + ' items loaded';
      } catch (_) { pingResult.textContent = 'request failed'; }
    });
  }
})();
