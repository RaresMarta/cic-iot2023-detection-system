# Multi-tenant monitoring (N=1, extensible) — design

- **Date:** 2026-06-25
- **Status:** draft (awaiting review)
- **Goal:** a central dashboard that monitors customer NIDS detectors. The demo runs **one** customer, but the system is built so adding customers = **config + DB rows, not code changes**. Model-performance work is explicitly deferred.

## Scope

**In:**
- A `customer` abstraction (first-class everywhere).
- A thin **central collector** service (new).
- **Supabase** persistence of incidents + periodic aggregates, tagged by `customer_id`.
- **Live Monitor page overhaul** — list classified flows for the selected customer.
- **Remove** the per-detector SQLite store.

**Out (future work, mention in presentation):**
- Multiple *live* detectors (demo runs one; system supports N).
- Edge buffering / offline replay (the removed SQLite).
- Firewall traversal + detector→collector push auth.
- Hardened per-tenant data isolation.

## Architecture

```
detector (droplet NIDS, watches 1 protected IP)
        │  flows + predictions (alert / recovered / flow)
        ▼
CENTRAL COLLECTOR  (new FastAPI service)
   • configured with a customer list  [ {customer A → detector A source} ]  (1 entry now)
   • tags every event with customer_id
   • relays live SSE to the dashboard, scoped per customer
   • writes incidents + periodic aggregates → Supabase
        │                                   │
   live SSE                            incidents + stats
        ▼                                   ▼
 Dashboard (React, static)  ◀── history ── Supabase (Postgres)
   • customer selector (from `customers`, 1 row now)
   • live: lists classified flows (green benign / red attack)
   • history: incidents + aggregate charts, filtered by customer_id
```

**Customer model:** one customer = one detector on the droplet, watching one protected IP. The droplet streams its flows + predictions to the collector.

## Components

### 1. Collector (new — `ids/apps/collector/`)
- **Config:** a list of `{customer_id, detector_source}` (one item for the demo).
- **Ingest:** subscribes to the detector's event stream (`flow` / `alert` / `recovered`).
- **Tag:** stamps each event with `customer_id`.
- **Relay:** exposes an SSE endpoint to the dashboard, filterable by customer (`/api/stream?customer=<id>`).
- **Persist:** writes `incidents` (alert→recovered) and periodic `stats_snapshots` to Supabase. **Never** persists the raw `flow` firehose.

> **Ingest direction — DECISION TO CONFIRM.** Earlier we said detectors *push*. For the N=1 demo the lighter, zero-detector-change option is the **collector pulls** the droplet detector's existing `/api/stream` SSE; from the dashboard's view the droplet still "streams to us." Recommendation: **pull for the demo**, document **push** (detector dials out) as the production/firewall path. Confirm on review.

### 2. Supabase schema (new tables; reuses existing auth)
- `customers (id, name, detector_ref, created_at)`
- `incidents (id, customer_id → customers, attacker_ip, family, confidence, started_ts, ended_ts, duration_s, top_features, status)`
- `stats_snapshots (id, customer_id → customers, ts, flows_total, malicious, dropped, …)`

(Mirrors the removed SQLite schema, plus `customer_id`.)

### 3. Dashboard — Live Monitor overhaul (`ids-frontend`)
- **Customer selector** driven by the `customers` table (one entry now). Replaces the hardcoded `VITE_DETECTOR_URL`.
- **Live view:** a list of **classified flows** for the selected customer (green benign / red attack), fed by the collector SSE.
- **History view:** incidents + aggregate trend charts read from Supabase, filtered by `customer_id`.

## Data flow

- **Live (ephemeral):** detector → collector → dashboard. Listed as classified flows; **not** stored.
- **Historical (durable):** collector → Supabase (incidents + aggregates, per `customer_id`). Dashboard reads history from Supabase.

## Extensibility seams (how N=1 becomes N)

1. **Customer is an entity, not a URL** — `customers` table + `customer_id` on every record.
2. **Collector config is a list** — one item now; add an item per customer.
3. **Dashboard is customer-scoped** — selector, not a single hardwired endpoint.

**To add a customer:** deploy a detector on their server → `INSERT INTO customers` → point the collector at it.

**To demonstrate at the defense:** seed a second `customers` row (no live detector) so the selector shows two; switch to it (shows "offline / no data"). Same code path — one just has no sensor yet. Line: *"Scaling to real customers means a detector per customer + hardening the collector's fan-in and per-tenant auth; the data model, dashboard, and persistence are already N-ready — that hardening is future work."*

## Removals

- Delete `ids/apps/monitor/store.py` and its wiring in `service.py` + `config.py` (the `IDS_DB_*` flags), the `deploy/hf-space` mirror, and related tests. Edge buffering becomes a conceptual future-work slide (no code stub).

## Testing

- Collector: unit-test event tagging + the persist-summary-not-firehose rule (raw `flow` events never written to Supabase).
- Collector: incident open/close (alert→recovered) writes one row with correct duration.
- Dashboard: selector lists customers from Supabase; switching customer re-scopes live + history.

## Out of scope / non-goals

- Model accuracy / benign-augmentation (separate, deferred track).
- Real multi-detector deployment, firewall traversal, tenant-isolation hardening (future work).
