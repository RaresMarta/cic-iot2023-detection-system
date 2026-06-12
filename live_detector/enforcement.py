"""Ban policy + out-of-band enforcement.

Policy decides allow/alert/ban from (family, confidence, attacker IP), applying the
confidence gate, a consecutive-malicious-window streak, and the NEVER_BAN allowlist.

Enforcers carry out a ban out-of-band (the protected site is never touched):
  NftablesEnforcer — adds the source IP to an nftables set with a timeout, so the
                     host kernel drops its forwarded packets until the TTL expires.
  LoggingEnforcer  — dev/macOS no-op that only tracks the blocklist in memory.
Both keep an in-memory blocklist so the dashboard never has to shell out to nft.
"""
from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import config


def attacker_ip(ip_a: str, ip_b: str, protected: set[str]) -> str | None:
    """The endpoint of a flow that is NOT the protected host. None if ambiguous."""
    a_prot, b_prot = ip_a in protected, ip_b in protected
    if a_prot and not b_prot:
        return ip_b
    if b_prot and not a_prot:
        return ip_a
    return None  # neither or both are protected -> can't attribute -> don't ban


class Policy:
    """Ban decision from the 2-class gate verdict, with confidence + streak gating.

    The gate (Benign/Attack) is the robust inline signal; the 8-class family is only a
    label. Ban requires Attack + calibrated confidence >= threshold + N consecutive
    malicious windows from the same source (so one misclassified window can't ban),
    and never bans the NEVER_BAN allowlist.
    """

    def __init__(self, threshold=None, consecutive=None, never_ban=None):
        self.threshold = threshold if threshold is not None else config.BAN_THRESHOLD
        self.consecutive = consecutive if consecutive is not None else config.CONSECUTIVE_FOR_BAN
        self.never_ban = never_ban if never_ban is not None else config.NEVER_BAN
        self._streak: dict[str, int] = {}

    def evaluate(self, gate_label: str, gate_confidence: float, attacker: str | None) -> str:
        if gate_label == 'Benign':
            if attacker:
                self._streak.pop(attacker, None)   # benign resets the streak
            return 'allow'
        # gate_label == 'Attack'
        if attacker is None or attacker in self.never_ban:
            return 'alert'
        if gate_confidence < self.threshold:
            return 'alert'
        self._streak[attacker] = self._streak.get(attacker, 0) + 1
        return 'ban' if self._streak[attacker] >= self.consecutive else 'alert'

    def reset(self, attacker: str) -> None:
        self._streak.pop(attacker, None)


@dataclass
class BanEntry:
    ip: str
    family: str
    banned_at: float
    expires_at: float
    hit_count: int = 1


class Enforcer(ABC):
    def __init__(self, ttl: int | None = None):
        self.ttl = ttl if ttl is not None else config.BAN_TTL_S
        self._bans: dict[str, BanEntry] = {}

    def setup(self) -> None:
        pass

    def ban(self, ip: str, family: str, now: float | None = None) -> BanEntry:
        now = now if now is not None else time.time()
        entry = self._bans.get(ip)
        if entry is None:
            entry = self._bans[ip] = BanEntry(ip, family, now, now + self.ttl)
        else:
            entry.hit_count += 1
            entry.expires_at = now + self.ttl   # refresh TTL
            entry.family = family
        self._apply(ip)
        return entry

    def is_banned(self, ip: str, now: float | None = None) -> bool:
        self._prune(now)
        return ip in self._bans

    def blocklist(self, now: float | None = None) -> list[dict]:
        self._prune(now)
        return [
            {'ip': e.ip, 'family': e.family, 'expires_at': e.expires_at,
             'banned_at': e.banned_at, 'hit_count': e.hit_count}
            for e in self._bans.values()
        ]

    def _prune(self, now: float | None = None) -> list[str]:
        now = now if now is not None else time.time()
        expired = [ip for ip, e in self._bans.items() if e.expires_at <= now]
        for ip in expired:
            del self._bans[ip]
        return expired

    @abstractmethod
    def _apply(self, ip: str) -> None:
        """Backend-specific enforcement for a single IP."""
        raise NotImplementedError


class LoggingEnforcer(Enforcer):
    """No-op enforcement for macOS/dev: blocklist is tracked, nothing is dropped."""
    backend = 'log'

    def _apply(self, ip: str) -> None:
        print(f'[enforce:log] would ban {ip} (ttl={self.ttl}s)', flush=True)


class NftablesEnforcer(Enforcer):
    """Drops banned sources at the host firewall via an nftables set with timeout."""
    backend = 'nft'

    def setup(self) -> None:
        # Idempotent: keep each table, flush its contents, recreate set/chain/rule.
        # Two drop points cover both topologies:
        #   inet/forward   — routed traffic (external attacker -> container on a VPS).
        #   bridge/forward — L2-bridged traffic between containers on the SAME bridge
        #                    (the demo's attacker<->website), which skips the IP forward
        #                    hook unless br_netfilter is loaded. The bridge family sees it.
        script = (
            'add table inet ids\n'
            'flush table inet ids\n'
            'add set inet ids banned { type ipv4_addr; flags timeout; }\n'
            'add chain inet ids drop_banned { type filter hook forward priority -150; policy accept; }\n'
            'add rule inet ids drop_banned ip saddr @banned drop\n'
            'add table bridge ids\n'
            'flush table bridge ids\n'
            'add set bridge ids banned { type ipv4_addr; flags timeout; }\n'
            'add chain bridge ids drop_banned { type filter hook forward priority -300; policy accept; }\n'
            'add rule bridge ids drop_banned ether type ip ip saddr @banned drop\n'
        )
        self._nft(['-f', '-'], stdin=script)

    def _apply(self, ip: str) -> None:
        # add element refreshes the timeout if it already exists; ban in both tables.
        elem = f'{{ {ip} timeout {self.ttl}s }}'
        self._nft(['add', 'element', 'inet', 'ids', 'banned', elem])
        self._nft(['add', 'element', 'bridge', 'ids', 'banned', elem])

    @staticmethod
    def _nft(args: list[str], stdin: str | None = None) -> None:
        try:
            subprocess.run(['nft', *args], input=stdin, text=True,
                           check=True, capture_output=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            detail = getattr(e, 'stderr', '') or str(e)
            print(f'[enforce:nft] command failed ({args}): {detail}', flush=True)


def from_config() -> Enforcer:
    """Pick the enforcer: explicit IDS_ENFORCER, else nft if available, else log."""
    choice = config.ENFORCER
    if not choice:
        import shutil
        choice = 'nft' if shutil.which('nft') else 'log'
    enforcer: Enforcer = NftablesEnforcer() if choice == 'nft' else LoggingEnforcer()
    enforcer.setup()
    return enforcer
