# Deployment insights — how this system maps onto real-world NIDS

Production network detection comes in two shapes, and this project happens to build both.
The online, streaming shape has a sensor capture traffic, compute flows in memory, and
classify in the detection path with low latency — this is our `live` monitor, an edge
sensor. The store-and-forward shape captures to a file or flow log, ships it to a central
system, and classifies in batches at higher latency — this is our `analyzer` (upload a
pcap, get a verdict) together with `replay`. The two are not redundant: they are the two
halves of the canonical NDR architecture, an edge probe and a central batch engine, and a
real deployment runs both for different purposes (live alerting versus forensics, hunting,
and retraining).

A common misconception is that a central IDS ingests raw packet captures. At scale it
almost never does, because full PCAP is bandwidth- and storage-prohibitive (a saturated
1 Gbps link is roughly 7.5 GB/min) and carries sensitive payloads. Production instead
exports flow records — NetFlow, IPFIX, sFlow, or Zeek logs — which is exactly the
granularity our 39-feature model consumes; full packets are shipped only as triggered
slices after an alert. So our flow-statistics input is not a thesis simplification but the
same abstraction real products are built on. It is also worth being precise about where
latency actually lives: computing a flow is cheap (group packets by 5-tuple, accumulate
counters, emit on window close), and the MLP forward pass is sub-millisecond, so the cost
in the file path is everything *around* the math — JVM cold start for the file-based
extractor, reading and transferring the whole file, and the batch wait. This is why the
real-time path streams flows in memory while the batch path tolerates the round-trip.

Notably, `live` and `replay` run the *identical* flow engine (`StreamWindower`); only the
packet source differs (a live NIC versus a pcap file), with `simulate` substituting sampled
CIC-IoT-2023 flows from the parquet. That makes `replay` best framed not as a real-time
deployment but as the production-grade engine fed by a reproducible offline source — a
testing and forensic harness. Feeding pcap files into a classifier is, in fact, the bedrock
practice of the whole IDS-ML field: the CIC-IoT-2023 dataset itself was built by capturing
PCAPs in a lab, running CICFlowMeter offline, and producing the flow CSVs we train on. That
practice is valid for batch, forensic, and evaluation work; it simply cannot support a
real-time claim. Running the streaming path, by contrast, needs a real Linux kernel, raw
capture access (`CAP_NET_RAW`, plus `CAP_NET_ADMIN` for promiscuous mode on a bridge), and
actual traffic on an interface — which Colima satisfies locally for a demo and a small KVM
VPS satisfies for the closest-to-production build. Throughout, the system stays a passive
NIDS: it detects and alerts, it does not block, which keeps it off the critical path and
avoids the false-positive blast radius of an inline ML blocker.

## Design patterns

The live monitor is worth presenting as a small study in concurrent-systems design, because
it composes three classic patterns across distinct stages. Between capture and
classification sits a **producer–consumer** boundary: a single capture thread produces
completed flow windows onto a bounded queue, and a single asyncio consumer drains it to run
the model. The bounded queue decouples packet rate from model rate, and on overflow it sheds
load by dropping the oldest window (counting the drop) rather than growing without bound or
stalling capture — which is also what keeps latency bounded. Because all windower state
stays in the capture thread, no locks are needed; the threads communicate by passing
messages through the queue rather than sharing mutable state, and the blocking queue is
bridged into the event loop through an executor so it never blocks async serving.

Between classification and the clients sits a second, different boundary: a
**publish–subscribe** broker that fans every event out to every subscriber, each with its
own bounded queue, so a slow dashboard client drops only its own events and can never apply
backpressure to detection or to the other clients. This is what makes the system extensible
at zero cost to the detection path — adding mobile push notifications, for instance, is just
one more subscriber. The distinction is worth stating precisely: the producer–consumer queue
hands each item of *work* to one consumer to decouple rates and shed load, whereas pub/sub
fans each *event* out to all subscribers. Underlying both is a **strategy / dependency-
inversion** arrangement: the detector depends on an abstract packet/window source injected at
construction, so `live`, `replay`, and `simulate` all drive the same detection engine
without it knowing which source it is reading — the same property that lets the whole
pipeline be developed and validated offline.

## A data-streaming system

Framed in the vocabulary of stream processing, this project is a textbook five-layer
stream architecture, and saying so explicitly is worth doing because it connects the work
to a well-understood body of theory. Network packets are the *source* that emits events;
an in-process *broker* and bounded queue form the ingestion layer (the single-host analog
of a Kafka broker with its topics and consumers); the `StreamWindower` is the *processing
engine* that does feature engineering, turning a window of packets into flow statistics
exactly as a Spark/Flink job turns raw clicks into an engagement score; the MLP gate and
family classifiers are the *model / intelligence* layer; and the SSE dashboard, alerts,
stats, and the event store below are the *output* layer. Several stream-mining techniques
appear in the design already: the windower is a count-based tumbling window (ten packets
per host pair), training uses per-class sampling, and overload is handled by dropping the
oldest event under a memory constraint — load-shedding rather than unbounded buffering —
with an end-to-end latency target (<100 ms) that matches the streaming norm. Detection is
the *micro*, event-level view (classify each window as it arrives); the reporting below is
the complementary *macro*, window-level view (analyse the last minute or hour). The honest
gap to name is that the model is statically trained, not an online/incremental learner that
updates per event, so the system does not yet adapt to concept drift on its own — concept
drift being precisely the streaming phenomenon behind why a model trained on past traffic
degrades on future traffic. Online learning and a distributed, fault-tolerant ingestion
tier (real Kafka, multiple partitions) are the natural future-work directions this framing
surfaces.

## Event store, reporting, and measuring effectiveness

The output layer is also where persistence belongs, and adding it is the most valuable next
feature because right now the system's events are ephemeral — the broker fans them out and
forgets, and the stats are live counters. A lightweight event store (a SQLite sink that is
simply one more broker subscriber, costing the detection path nothing) closes that gap and
turns the system into something that can look backward, not only react forward. What it
should persist is not every benign flow — that bloats storage the same way full PCAP does —
but incidents (alert/recovered episodes) and periodic windowed aggregates: per-minute and
per-hour counts, the family histogram, and top-k talkers, with approximate sketches
(count-distinct, top-k) where exactness is not needed. Those stored aggregations are exactly
the "insightful aggregations from the stream" worth surfacing to an operator or client, and
they feed two report flavours — a per-incident report generated when an attacker recovers
(the episode is already a complete object: source, family, duration, peak confidence, the
SHAP reasons), and a scheduled periodic summary over a time range. The higher-value payoff,
though, is measurement: because the live demo has ground truth (the operator launched the
attack and knows which sources are malicious), the logged flows can be labelled after the
fact and compared against what the detector flagged, yielding deployment-time precision,
recall, and false-alarm rate on real traffic rather than only held-out test-set F1. That
comparison loop — offline evaluation versus online effectiveness, and competing models
scored on the same recorded stream — is a genuine thesis contribution that the event store
makes possible.

## Where the sensor sits: layers and placement

It helps to sediment where in the network stack this system actually operates. The
application layer carries the content of a conversation (HTTP between a browser and a web
server); the transport layer packages and reliably delivers that content between hosts
(TCP/UDP, ports); and the network layer handles addressing and routing, the level at which
data becomes IP packets. The detector reads flow statistics derived almost entirely from the
network and transport layers — IP addresses for attribution, ports and protocol in the
5-tuple, packet sizes, flags, and inter-arrival timing — and deliberately does not inspect
application-layer payloads. That single fact explains the system's headline strength and
weakness at once: volumetric and structural attacks that distort traffic shape (DDoS, DoS,
recon) are visible in flow statistics and detect well, whereas application-layer attacks
(SQL injection, XSS) live in payloads the flow view never sees and detect poorly — which
matches the dataset paper's own findings. As for physical placement, the detector is a
passive sensor reading a copy of traffic off the bridge, the lab analog of a switch SPAN
port or a network tap; at larger scale an IDS load balancer would aggregate taps and SPAN
feeds, reassemble sessions split across links, and distribute copies to several sensors.
This is the passive architecture of the two classic NIST sensor diagrams (the alternative
being the inline architecture that can block); the project chose passive on purpose. In the
NIST IDPS taxonomy the result is a network-based system using network-behavior-analysis
style flow detection — not host-based (which would monitor a single machine's own OS, logs,
and processes; being on-premises does not make it host-based), and not wireless. The 2007
guide is dated, but its placement and sensor-type taxonomy remains the standard vocabulary.

## Correlation

Matching event information from multiple sensors or agents — for instance, finding the
events triggered by the same IP address — is known as correlation, and the system already
does a primitive single-sensor version of it: the detector keeps per-attacker state keyed by
source IP, so the many malicious windows of one flood are correlated into a single sustained
alert and a later "recovered" event rather than a storm of duplicates. The richer forms are
a natural extension. Correlating across sources turns the distributed Mirai fan-in (many
attacker IPs against one target) into a single recognised campaign rather than thirty
unrelated alerts; correlating across sensors, aided by the load balancer reassembling split
sessions, is what makes detection coherent at scale; and correlating across time — querying
the event store retrospectively for everything tied to an IP or an episode — is what turns
stored logs into investigations. Naming correlation explicitly, and pointing at the per-IP
state as its seed, is a clean way to show the design anticipates where a real SOC pipeline
goes next.
