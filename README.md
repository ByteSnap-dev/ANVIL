# ANVIL

**Adaptive Notes · Verification · Inference · Liaison**

A lean, local-first agent harness that makes a single 24GB GPU punch above its
class. It climbs an **all-Ollama** model ladder — **local models → Ollama Cloud
models** — only as far as a task actually needs, takes notes constantly, recalls
them under a strict token budget, schedules its own cron jobs, and reaches you on
Discord while you're away from the box.

The full design rationale is in **[ANVIL_BLUEPRINT.md](ANVIL_BLUEPRINT.md)**.

## Why it punches above its class

- **Cheapest-capable-first.** Routine work runs on the local model at $0. The
  router escalates one rung only on a concrete signal — low self-confidence,
  JSON-schema failure, a critic veto, or an explicit floor — never by default.
- **Context is a budget.** Notes are always *available* (a cheap always-loaded
  index) but only *loaded* when they earn their tokens. The transcript
  self-compacts before it ever hits the window.
- **Real cost discipline.** A per-call cost ledger and a daily spend cap that
  throttles escalation and pings you on Discord — on top of the flat-rate Ollama
  Cloud (Max) subscription, so sustained cloud use is already paid for.
- **Zero-dependency core.** Router, memory, scheduler, context budgeting, and
  Discord push all run on the Python standard library — every provider call is
  raw `urllib`, no SDK. Optional extras (`discord.py`, `tiktoken`, `APScheduler`)
  light up richer features and degrade gracefully when absent.

## Layout

```
anvil/
  config.py     load TOML/JSON + env overrides, define the ladder
  providers.py  Ollama local + Ollama Cloud — all via stdlib urllib
  router.py     escalation engine + cost ledger  ← "smart token use"
  context.py    token estimation, budget packing, transcript compaction
  memory.py     Markdown notes, ranked recall, consolidation
  scheduler.py  zero-dep cron parser + job runner  ← autonomous cron
  comms.py      Discord webhook (zero-dep) + optional two-way bot
  pipeline.py   Planner / Worker / Critic / Scribe  ← the "team"
  cli.py        ask | note | recall | status | schedule | run | serve
anvil.toml      the tuned defaults
jobs/           one JSON spec per scheduled job (agent can write these itself)
memory/         your notes (created on first write)
```

## Quick start

```bash
# 1. Point at your models (local Ollama must be running; Ollama Cloud optional)
ollama pull qwen3.6
ollama pull nomic-embed-text

# 2. Configure secrets
cp .env.example .env && $EDITOR .env
set -a; . ./.env; set +a

# 3. Use it (core needs no pip install)
python -m anvil status
python -m anvil ask "Write a bash one-liner to find the 10 largest files under /var"
python -m anvil ask --plan "Design a backup rotation for my Proxmox VMs"
python -m anvil note "anvil-01 needs pve-firewall restart after bridge changes"
python -m anvil recall "proxmox firewall"

# 4. Run the scheduler (and Discord bot if a token is set)
python -m anvil serve
```

## How escalation works

Every step starts at rung 0 (local). The Worker answers; the Critic — running
one rung up, on a different provider so errors don't correlate — checks for
hallucinated commands, bad flags, unsafe operations, and false claims, and can
veto. A veto triggers exactly one bounded re-run one rung higher. The Scribe
then distills any durable facts into `memory/` at $0. Each call is logged to
`ledger.jsonl`; once `daily_cost_cap_usd` is hit, paid rungs stop and you get a
Discord ping. Tune the ladder from the ledger (see the blueprint's roadmap).

## Scheduling

Drop a JSON spec in `jobs/` (or let the agent write one). Standard 5-field
cron. `notify: "discord"` pushes the result to your webhook. Example in
`jobs/homelab-morning-digest.json`. Jobs that run shell commands pass through
the Critic's destructive-command denylist before anything executes.

## Safety rails

- Destructive-command denylist (`rm -rf`, `mkfs`, `dd if=`, fork bombs, …) hard-
  vetoes regardless of what any model says.
- Daily spend cap throttles paid escalation.
- Discord approval flow: high-stakes jobs DM you and wait for `!approve`.

> Status: a runnable scaffold with working core logic. The provider calls
> assume your local Ollama (and, optionally, Ollama Cloud) is reachable; wire
> your real job commands and review the safety denylist before production.

## License

ANVIL is licensed under the [GNU Affero General Public License v3.0](LICENSE)
(AGPL-3.0). You can run, study, modify, and share it freely; if you offer a
modified version to others — including as a network service — you must make
your source available under the same terms.
