"""Packet sources — one interface, two backends so the detector is source-agnostic.

PcapReplay   : reads a pcap/pcapng file. Cross-platform; the offline-dev path used
               to validate the whole pipeline on macOS without root or a real NIC.
LiveCapture  : raw AF_PACKET socket on a Linux interface. The deployment path; needs
               Linux + CAP_NET_RAW. Imported lazily so this module loads on macOS.

Both yield (timestamp, raw_ethernet_bytes) tuples. A short recv timeout on the live
source lets the caller run windower.flush_idle() between packets.
"""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path


from ids.runtime.extractor import _iter_packets  # noqa: E402

# Sentinel yielded by LiveCapture on a recv timeout so the consumer can flush idle
# windows even when no packet arrived. ts is None for this marker.
IDLE_TICK: tuple[None, None] = (None, None)


class PacketSource(ABC):
    mode: str = 'unknown'

    @abstractmethod
    def packets(self) -> Iterator[tuple]:
        """Yield (ts, raw_bytes). LiveCapture may also yield IDLE_TICK."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class PcapReplay(PacketSource):
    mode = 'replay'

    def __init__(self, path: str | Path, realtime: bool = False, loop: bool = False):
        self.path = Path(path)
        self.realtime = realtime
        self.loop = loop
        if not self.path.exists():
            raise FileNotFoundError(f'pcap not found: {self.path}')

    def packets(self) -> Iterator[tuple]:
        while True:
            prev_ts: float | None = None
            wall_start = time.time()
            file_start: float | None = None
            for ts, buf in _iter_packets(self.path):
                if self.realtime:
                    if file_start is None:
                        file_start = ts
                    # pace to the capture's own inter-arrival timing
                    target = wall_start + (ts - file_start)
                    delay = target - time.time()
                    if delay > 0:
                        time.sleep(min(delay, 1.0))
                    prev_ts = ts
                yield ts, buf
            if not self.loop:
                return


class LiveCapture(PacketSource):
    mode = 'live'

    def __init__(self, iface: str, recv_timeout: float = 0.5, snaplen: int = 65535):
        self.iface = iface
        self.recv_timeout = recv_timeout
        self.snaplen = snaplen
        self._sock = None

    def _open(self):
        import socket
        import time
        ETH_P_ALL = 0x0003
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))  # type: ignore[attr-defined]
        # The bridge interface may appear a moment after the container starts
        # (docker creates idsnet/ids-br0 around the same time). Retry the bind.
        deadline = time.time() + 30
        while True:
            try:
                sock.bind((self.iface, 0))
                break
            except OSError as e:
                if time.time() >= deadline:
                    sock.close()
                    raise SystemExit(f'cannot bind to interface {self.iface!r}: {e}')
                print(f'[capture] waiting for interface {self.iface!r}…', flush=True)
                time.sleep(1.0)
        sock.settimeout(self.recv_timeout)
        self._sock = sock

    def packets(self) -> Iterator[tuple]:
        import socket
        if self._sock is None:
            self._open()
        while True:
            try:
                buf = self._sock.recv(self.snaplen)  # type: ignore[union-attr]
            except socket.timeout:
                yield IDLE_TICK         # let the consumer flush idle windows
                continue
            except OSError:
                break

            yield time.time(), buf

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def from_config():
    """Build the packet source selected by environment config."""
    from . import config
    if config.SOURCE == 'live':
        return LiveCapture(config.IFACE)
    if not config.PCAP_PATH:
        raise SystemExit('IDS_SOURCE=replay requires IDS_PCAP=<path to pcap>')
    import os
    loop = os.environ.get('IDS_PCAP_LOOP', '0') == '1'

    return PcapReplay(config.PCAP_PATH, realtime=config.PCAP_REALTIME, loop=loop)
