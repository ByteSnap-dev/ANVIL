# ANVIL

**Adaptive Notes · Verification · Inference · Liaison**

A lean, self-hosted agent harness for a household AI assistant. Your data —
memory, conversations, schedules, house state — lives on your box and never
syncs to anyone's cloud; the *intelligence* is a pluggable **tiered ladder**
the router climbs only as far as a task actually needs. The reference
configuration is honest about its trade: it runs an all-Claude ladder
(Haiku → Sonnet → Opus → Fable), so prompts go to the Anthropic API over TLS
and an internet outage pauses the assistant. The same harness runs fully
local on Ollama models (its original mode), Ollama Cloud, or any mix — the
ladder is config, not code, and every provider call is raw stdlib `urllib`,
no SDK.

The assistant it ships is **Lara**: a persistent family agent with a PWA chat
surface, evidence-gated long-term memory, an autonomous background mind, and a
git-guarded self-development loop that lets her ship reviewed changes to her
own harness.

The original design rationale is in **[ANVIL_BLUEPRINT.md](ANVIL_BLUEPRINT.md)**.

## What makes it lean

- **Cheapest-capable-first.** Every step starts on the ladder's base rung and
  escalates only on a concrete signal — schema failure, empty output, low
  confidence, a struggling agent loop — never by default. Foreground chat can
  pin a floor (e.g. Sonnet) while background automation stays on the base.
- **Real cost discipline.** A per-call ledger prices every token (prompt-cache
  reads and server-side web searches included). Three caps — daily, background,
  rolling-monthly — throttle spend per *plane*, so the assistant's own overnight
  work can never starve the family's chat budget. Latency-insensitive
  background planes ride the Anthropic Message Batches API at half price.
- **Context is a budget.** Notes are always *available* (an always-loaded
  index + FTS5 ranking) but only *loaded* when they earn their tokens. The
  transcript self-compacts before it ever hits a window; overflow is detected
  and retried, never surfaced to the user.
- **Memory that resists confabulation.** Long-term notes are evidence-gated
  (claims need a tool result behind them), salience-decayed, deduplicated, and
  reconsolidated during sleep. Each family member's memory is private by
  default with explicit sharing.
- **Zero-dependency core.** Router, memory, scheduler, context budgeting, and
  the web server run on the Python standard library. Optional extras
  (`cryptography` for Web Push, `websocket-client` for Home Assistant
  streaming) light up richer features and degrade gracefully when absent.

## What it does

- **Family chat** — a PWA (installable on phones) with profiles and logins,
  per-person conversation isolation, a live transparency trace of every tool
  call, and self-hosted Web Push notifications. No third-party messaging
  dependency.
- **Autonomous mind** — an autopilot thinks and consolidates memory all day,
  dreams distill short-term events into durable notes, and a skill flywheel
  learns reusable procedures from real conversations.
- **Tools with consent** — shell, SSH to other nodes, Home Assistant senses
  and actions, weather (keyless NWS), web search, maps, shared lists,
  reminders, image generation. Destructive actions pause for approval, routed
  to an adult's device; minors can never approve danger.
- **Self-development** — the assistant works an issue queue on its own repo:
  a driver codes the change, a separate reviewer model critiques the diff, a
  ~290-test doctor gate must go green, and failures revert cleanly. Promotion
  to main goes through a pull request.
- **Hive** — parallel ephemeral worker drones for research-shaped asks, with
  structurally narrowed tool access.

## Layout

```
anvil/
  config.py     TOML/JSON + env config; the ladder is data, not code
  providers.py  Anthropic + Ollama adapters (stdlib urllib, SSE + batch)
  router.py     escalation engine, circuit breaker, cost ledger + caps
  pipeline.py   the agent loop: plan / act / critic / scribe
  memory.py     Markdown notes, salience, evidence gate, per-profile privacy
  mind.py       autopilot, dreams, sleep reconsolidation
  server.py     zero-dep web server + PWA (hearth.html)
  forge.py      git-guarded self-dev cycle (doctor gate, consensus revert)
  selfdev.py    the coding driver + reviewer
  scheduler.py  zero-dep cron parser + job runner
  tools.py      the tool registry (danger-gated)
anvil.toml      your config (ladder, caps, persona) — gitignored, see docs
jobs/           one JSON spec per scheduled job (the agent writes these too)
memory/         notes, skills, journal (created on first write; never in git)
```

## Quick start

```bash
# 1. Configure secrets (an Anthropic key lights the Claude ladder; a local
#    Ollama and/or Ollama Cloud key are equally valid rungs)
cp .env.example .env && $EDITOR .env

# 2. Core needs no pip install
python -m anvil status

# 3. Launch the web interface (chat, setup wizard, approvals, memory mirror)
python -m anvil serve-web

# 4. Or just the scheduler loop, headless
python -m anvil serve
```

Open the printed URL, run the setup wizard, and you have a household
assistant. Bind `bind_host` to a Tailscale address to reach it from phones.

## Safety rails

- Destructive-command denylist (`rm -rf`, `mkfs`, `dd if=`, fork bombs, …)
  hard-vetoes regardless of what any model says; the critic scans commands,
  not prose.
- Approval gates on every state-changing tool, with identity-aware routing —
  a child's request pings an adult, and approvals expire.
- Autonomy modes (`ask` / `trusted` / `auto`) plus a taught allowlist decide
  what runs free; read-only commands always do.
- Spend caps per plane; a hit cap degrades gracefully and notifies once.
- Prompt-injection scanning frames untrusted web content before it reaches
  trusted context.

> Status: a live, daily-driven system — but built for one household and shaped
> by its needs. Expect to read the config and docs rather than find a polished
> installer. The doctor suite (`python -m anvil doctor`) is the ground truth.

## License

ANVIL is licensed under the [GNU Affero General Public License v3.0](LICENSE)
(AGPL-3.0). You can run, study, modify, and share it freely; if you offer a
modified version to others — including as a network service — you must make
your source available under the same terms.
