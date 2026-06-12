// Customer-site reaction layer. Subscribes to the detector's SSE feed and reflects
// the threat lifecycle on the page. Presentation only — no detection happens here.
(function () {
  const body = document.body;
  const badge = document.getElementById('guardian');
  const stateEl = document.getElementById('guardian-state');
  const detailEl = document.getElementById('guardian-detail');
  const url = (window.DETECTOR_URL || 'http://localhost:7870') + '/api/stream';

  const FAMILY_LABEL = {
    DDoS: 'Distributed flood', DoS: 'Denial-of-service flood',
    Mirai: 'Botnet flood', Recon: 'Reconnaissance scan',
  };

  let recoverTimer = null;

  function setThreat(level, state, detail) {
    body.dataset.threat = level;
    badge.className = 'guardian guardian--' + level;
    stateEl.textContent = state;
    detailEl.textContent = detail;
  }

  function scheduleCalm(delay) {
    clearTimeout(recoverTimer);
    recoverTimer = setTimeout(() => setThreat('calm', 'Protected', 'All traffic nominal'), delay);
  }

  function onEvent(evt) {
    switch (evt.type) {
      case 'alert':
        setThreat('elevated', 'Threat detected',
          (FAMILY_LABEL[evt.family] || evt.family) + ' — analysing');
        break;
      case 'flow':
        if (evt.gate === 'block' && body.dataset.threat !== 'blocked') {
          setThreat('engaged', 'Under attack',
            (FAMILY_LABEL[evt.family] || evt.family) + ' from ' + evt.src);
        }
        break;
      case 'ban':
        setThreat('blocked', 'Threat blocked', evt.attacker_ip + ' banned for ' + evt.ttl_s + 's');
        scheduleCalm(8000);
        break;
      case 'recovered':
        setThreat('calm', 'Protected', 'Recovered — traffic nominal');
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
