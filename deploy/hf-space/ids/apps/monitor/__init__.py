"""Live beside-path NIDS: sniff -> window -> features -> model -> verdict -> alert.

A passive, host-co-located sensor (NOT an inline IPS). It observes copies of traffic and
classifies per-host-pair flow windows with the trained MLP, raising an alert on sustained
malicious traffic. It only detects and reports — it does not block, ban, or touch the
firewall. (Out-of-band IP enforcement is deferred to a possible future IPS iteration.)
"""
