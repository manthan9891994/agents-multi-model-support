# Roadmap

This roadmap reflects intent, not commitments. Things shift based on user feedback and what actually matters in production.

## v0.2 — Adoption (Q3 2026)

- LangGraph integration
- Multi-provider failover (Google → Anthropic → OpenAI on outage)
- `dmr cookbook` — 12+ ready-to-paste integration recipes
- Hosted documentation on readthedocs
- Conda-forge recipe

## v0.3 — Production Hardening (Q4 2026)

- Per-tenant API credentials and budget limits
- Prometheus metrics endpoint
- L1 keyword scan via Aho-Corasick (handles 1K+ keywords at <1ms)
- L3 batch inference (`router.classify_batch`)
- Snapshot tests + property-based tests (`hypothesis`)

## v0.4 — Enterprise (2027)

- Multi-region geo-routing
- Immutable audit logging
- HIPAA / SOC 2 compliance documentation
- Multilingual L1 (10+ languages)

## Out of scope (deliberately)

- A REST API service (use the package directly; build your own gateway)
- A managed/hosted offering
- Multi-tenancy beyond per-tenant config (use one Router per tenant)

Want to influence what lands when? Open a discussion.
