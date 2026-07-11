# Claude tiered ladder — activation

The `AnthropicProvider` + config plumbing is shipped. Activation is two edits +
a restart; nothing goes live until the key is present (rungs skip gracefully).

## 1. The key (never committed — `.env` is gitignored)

```
ANTHROPIC_API_KEY=sk-ant-...
```

## 2. The ladder — replace ALL `[[ladder]]` blocks in `anvil.toml`

No local rung: Haiku is the BASE — every turn, even "hello", runs on Claude
(operator's call: trade the free local tier for consistency). Escalation
climbs Haiku → Sonnet → Opus → Fable on real signals.

```toml
# --- BASE / WORKHORSE: every turn starts here ---
[[ladder]]
name = "claude-haiku"
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
cost_in = 1.0
cost_out = 5.0
cache_read = 0.10
max_context = 200000

# --- REASONING / CODING ---
[[ladder]]
name = "claude-sonnet"
provider = "anthropic"
model = "claude-sonnet-5"
cost_in = 3.0
cost_out = 15.0
cache_read = 0.30
max_context = 200000

# --- HARD calls / final reviews (limited) ---
[[ladder]]
name = "claude-opus"
provider = "anthropic"
model = "claude-opus-4-8"
cost_in = 5.0
cost_out = 25.0
cache_read = 0.50
max_context = 200000

# --- RARE ceiling escalation ---
[[ladder]]
name = "claude-fable"
provider = "anthropic"
model = "claude-fable-5"
cost_in = 10.0     # confirmed 2026-07 list price ($10/$50, cache read $1)
cost_out = 50.0
cache_read = 1.00
max_context = 200000
```

## 3. Role rungs + budget (top of `anvil.toml`)

Every role rung must name a rung that EXISTS in the ladder above (no more
`local-*`). All five below are Claude tiers now.

```toml
chat_rung          = "claude-sonnet"   # FOREGROUND floor: family chat starts on
                                       # Sonnet (first-shot quality); struggle
                                       # escalation climbs Sonnet -> Opus
planner_rung       = "claude-haiku"
critic_rung        = "claude-haiku"    # cheap judge over a strong generator
hive_worker_rung   = "claude-haiku"
vision_rung        = "claude-sonnet"   # Sonnet reads images better than Haiku
selfdev_rung       = "claude-sonnet"   # coding driver — reliability > $
skill_review_rung  = "claude-haiku"    # was local-fast; must point at a real rung

# Background planes (dreams, scribe, autopilot, hive) do NOT inherit
# chat_rung — they enter through the Router at the ladder base (Haiku).

daily_cost_cap_usd        = 10.0   # foreground protected up to here, then notify
background_cost_cap_usd    = 4.0    # dreams/self-dev/hive stop at this soft cap
monthly_cost_cap_usd       = 250.0  # rolling-30-day ceiling (the sub-$300 guard)
```

**No-local caveat:** with the local base gone, a Claude outage OR a hit daily
cap leaves nothing free to fall back to — a turn will error rather than
degrade. That's the accepted trade for consistency; the monthly cap + the
"degrade to free" path simply have no free rung to reach, so they surface the
cap as an error instead. Raise the daily cap or re-add a single local rung at
the bottom if you want a safety net.

## Pricing note

Rates above are per-million-tokens and match the current list — Fable 5 was
confirmed July 2026 at $10 in / $50 out (cache read $1), double Opus but half
the early placeholder guess. The ledger prices every call off these numbers with real token counts,
and `spent_last_days(30)` drives the monthly cap, so getting the rates right
keeps both the cost display and the cap honest. Prompt caching (cache_read is
~10% of input) and the direct-route/local base are what keep the realistic
month at ~$60–150 well under the $250 cap.

## Message Batches: the overnight 50% lever

`batch_background = true` routes every paid call on the latency-insensitive
background planes (`selfdev`, `dreams`, `scribe`) through Anthropic's async
Message Batches API — **half price on every token** (fresh input, cache reads,
output alike; only the $10/1k web-search fee is never discounted). The
mechanism is deliberately boring: a *batch of one*. Each call submits a
single-request batch, polls `processing_status` every `batch_poll_s` (20s)
until it ends, and reads the result — same payload, same parsing, same
Completion. No suspend/resume rewrite of the forge; a self-dev model call just
takes minutes instead of seconds, which nobody watching only the dev queue
ever notices.

Structural guarantees:

- **Chat, approvals, and hive drones can never batch.** The lane is gated on
  the Router's plane tag (`_BATCH_PLANES`), and `hive` is deliberately
  excluded — drones answer live turns.
- A caller-cancel or a `batch_wait_s` (1h) timeout **cancels the server-side
  batch** so an abandoned request can't bill later.
- The ledger records `batched: true` and prices the call at 50% via
  `_estimate`; cap accounting therefore sees the real (discounted) spend, so
  the $6 background cap now buys roughly twice the overnight work.
- The `temperature`-deprecated retry works here too — the rejection arrives as
  an errored *result* rather than an HTTP 400, and is classified the same way.

```toml
batch_background = true   # default false; flip deliberately
batch_wait_s     = 3600
batch_poll_s     = 20
```
