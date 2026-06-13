"""Live beside-path NIDS: sniff -> window -> features -> model -> verdict -> ban.

A passive, host-co-located sensor (NOT an inline IPS). It observes copies of traffic,
classifies per-host-pair flow windows with the trained MLP, and bans attacking source IPs
out-of-band at the host firewall (nftables). See ids/apps/monitor/config.py for the policy.
"""
