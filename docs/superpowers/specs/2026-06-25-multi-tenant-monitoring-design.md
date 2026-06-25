# Central monitoring — design

- **Date:** 2026-06-25
- **Status:** draft (awaiting review)
- **Goal:** a central dashboard that monitors one customer's NIDS detector. Everything is organised around a named **customer** rather than a single hardwired sensor — that shape keeps the door open to more later, but we build for exactly one now.

## What this is

One customer = one detector on the droplet, watching one protected IP. The droplet streams its flows + predictions to a central service, which shows them live and files the important parts into the central database.

## Architecture

```
detector (droplet NIDS, watches 1 protected IP)
        │  flows + predictions (flow / alert / recovered)
        ▼
CENTRAL COLLECTOR  (new service)
   • bound to one customer
   • shows the live stream to the dashboard, labelled by customer
   • files incidents + periodic summaries → Supabase
        │                                  │
   live stream                        incidents + summaries
        ▼                                  ▼
 Dashboard (React)  ◀──── history ──── Supabase (Postgres)
   • customer is selected on the monitor page
   • live: lists classified flows (green benign / red attack)
   • history: incidents + summary charts for that customer
```

## Components

### Collector (new)
- Bound to the one customer; ingests the detector's stream by **subscribing to the detector's existing live feed** (the collector pulls; the detector is unchanged).
- Labels every event with the customer.
- Relays the live stream to the dashboard.
- Files `incidents` (an attack episode, start→end) and periodic `stats_snapshots` into Supabase. The per-flow firehose is shown live but **never stored**.

### Supabase (reuses existing auth)
- `customers` — the customer being tracked.
- `incidents` — one row per attack episode, linked to the customer.
- `stats_snapshots` — one row per time interval, holding **what happened in that interval only** (flows, malicious, dropped, family breakdown) — not running totals. Linked to the customer. "Last X" = aggregate the rows in that range.

### Dashboard — monitor page
- The customer is **selected on the page** (instead of a single hardwired detector address).
- **Live:** a list of classified flows for that customer (green benign / red attack), showing verdict + confidence (no per-flow explanation — that's the analyzer's job).
- **History:** incidents + summary charts for that customer, read from Supabase.

## Data flow

- **Live:** detector → collector → dashboard. Listed as classified flows; not stored.
- **History:** collector → Supabase (incidents + summaries). Dashboard reads history from Supabase.

## Removals / detector simplification

- The per-detector SQLite store is removed. The central database is now the only place history lives.
- **Explainability is removed from the live detector** — no per-flow saliency proxy, no per-episode SHAP, no explainer setup. Live `flow`/`alert` events carry **verdict + family + confidence only** (no `top_features`). This keeps the live path lean and avoids the consumer pause at attack onset.
- **SHAP explainability stays in the analyzer** (on-demand, latency-tolerant) — that is where "why was this flagged" lives. The live dashboard shows verdict + confidence only.

## Non-goals

- Model accuracy work (separate, deferred track).
