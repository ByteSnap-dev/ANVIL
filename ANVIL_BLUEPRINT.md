# ANVIL — Architecture Blueprint

**Adaptive Notes · Verification · Inference · Liaison**

A self-hosted Python harness that turns a single strong local GPU + Ollama Cloud into one coordinated agent that takes notes constantly, recalls them under tight token budgets, escalates to bigger models only when it pays off, schedules its own jobs, and reaches you on Discord while you're away from the box.

Built for Joe: systems administrator, father, tinkerer, board gamer, Twitch streamer, gamer. The design goal is bluntly stated: make a 24GB local model *punch above its class* by spending cloud tokens like a miser and spending context like it's expensive — because it is.

---

## 1. The core idea

A local model is cheap and private but limited. A frontier model is capable but costs money and round-trips. Most harnesses pick one. Anvil treats them as a **ladder** and climbs only as far as a task actually requires.

Two principles drive every design decision:

1. **Cheapest-capable-first.** Every task starts at the lowest rung that can plausibly do it. Anvil escalates only on concrete signals (low self-confidence, schema-validation failure, tool-loop thrash, explicit difficulty), never by default. This is what makes the local agent "punch above its class": the local model does the volume, and the expensive models are reserved for the 5–10% of steps that genuinely move the needle.
2. **Context is a budget, not a bucket.** Tokens spent recalling notes are tokens not spent thinking. Anvil tracks a per-call token budget, retrieves only what's relevant, and compresses aggressively when it approaches the limit.

---

## 2. The model ladder

The ladder is configurable in the Setup tab; the default tuned for a 24GB GPU + an Ollama Cloud (Max) subscription is **all-Ollama**:

| Rung | Name | Runs on | Default model | Job |
|------|------|---------|---------------|-----|
| 0 | `local-fast` | Ollama **local** (GPU) | `qwen3.6` | Fronts every chat + tool call; drafts, classification, note distillation, planner/critic/vision — the first attempt at everything |
| 1 | `local-reason` | Ollama **local** (GPU) | `gemma4` | Heavier local reasoning step-up |
| 2 | `cloud-open` | Ollama **Cloud** | `glm-5.2` | Substantive-answer synthesis; the default research drone |
| 3 | `cloud-heavy` | Ollama **Cloud** | `kimi-k2.6` | The code specialist |
| 4 | `cloud-logic` | Ollama **Cloud** | `deepseek-v4-pro` | The logic/math specialist; autonomous self-dev coding |

Why this shape: qwen3.6 is the current 24GB sweet spot — fast and RL-trained for tool calling, which the pipeline leans on heavily. Every rung — local *and* cloud — speaks the **same** OpenAI-compatible HTTP surface through the local Ollama daemon: the cloud rungs are `:cloud`-suffixed models the local Ollama transparently proxies to Ollama Cloud. So there is one code path, no third-party SDK, and no per-token billing to manage — the Max subscription is flat-rate, so sustained cloud use is already paid for. Cost discipline is a per-call ledger + a daily cap, not prompt-cache accounting.

### Escalation triggers (concrete, not vibes)

Anvil escalates one rung when **any** fires:

- **Self-confidence** below threshold. The worker model returns a structured `{answer, confidence, reason}`; confidence < 0.6 escalates.
- **Schema failure.** If a step must return JSON matching a schema and the model fails validation twice, escalate (don't burn retries on a model that can't comply — Ollama's constrained structured outputs catch most of this at rung 0).
- **Tool-loop thrash.** Same tool called 3× with no state change → escalate.
- **Explicit difficulty tag.** A task or cron job can be tagged `min_rung: 3` to force a floor.
- **Critic veto.** The Critic agent (below) can demand a re-run one rung up.
- **Budget guard, inverted.** If a cheap rung has already burned more wall-clock/retries than the next rung would cost, escalate — sometimes one call on a top rung is cheaper than five on a cheap one.

De-escalation matters too: once a hard sub-problem is solved by a high rung, its result is cached as a note and subsequent similar steps drop back to rung 0 referencing that note.

---

## 3. The agent pipeline (the "team")

Rather than one model doing everything, Anvil runs a small role pipeline. Each role is a prompt + a default rung, and roles can independently escalate.

```
        ┌─────────┐   plan    ┌─────────┐  result  ┌─────────┐
  task →│ PLANNER │──────────▶│ WORKER  │─────────▶│ CRITIC  │──┐
        └─────────┘  rung 2-3 └─────────┘ rung 0-1 └─────────┘  │
             ▲                                          │ veto  │ pass
             └──────────────── re-plan ─────────────────┘       ▼
                                                          ┌─────────┐
                       notes in/out (every stage) ───────▶│ SCRIBE  │
                                                          └─────────┘
                                                               │
                                                       memory store + Discord
```

- **Planner** (default rung 2–3): decomposes the task into steps, decides which steps are local-able vs. need cloud, sets per-step rung floors. Runs once per task; cheap relative to the work it saves.
- **Worker** (default rung 0): does the actual steps on the local model. The workhorse — most tokens flow here at $0.
- **Critic / Verifier** (default rung 2, escalates to 4 for high-stakes): independent check. Catches the classic local-model failure modes — hallucinated commands, wrong flags, plausible-but-false facts. Can veto and force a re-run. For high-stakes output (anything touching prod, money, or published content) the Critic is a *different provider* than the Worker so errors don't correlate.
- **Scribe** (rung 0): after every stage, distills what was learned into notes — decisions, gotchas, working commands, dead ends — and writes them to the memory store. Runs locally and constantly; this is the "take notes all the time" requirement.
- **Router** (no model — pure logic): the escalation engine that picks rungs and tracks cost.

This maps cleanly onto Joe's worlds: a Planner/Worker/Critic loop is exactly how you'd want help drafting a stream-automation script (Worker writes it, Critic checks the OBS websocket calls), triaging a server alert (Worker proposes, Critic vetoes anything destructive before it ever reaches you), or prepping a board-game night (Worker drafts, Scribe remembers house rules for next time).

---

## 4. Memory & notes — context-budget-aware

The store is **file-based, plain Markdown with YAML frontmatter** — greppable, git-able, survives the process, no database to babysit. One fact per file, plus a loaded-every-session `INDEX.md`. (Deliberately the same shape as an agent memory directory, so it's portable.)

```
memory/
  INDEX.md                     # one line per note — the always-loaded map
  notes/
    homelab-proxmox-quirk.md
    twitch-obs-scene-names.md
    boardgame-house-rules.md
  embeddings.jsonl             # vector per note for semantic recall (optional)
```

Each note:

```markdown
---
name: homelab-proxmox-quirk
type: reference            # user | feedback | project | reference
tags: [proxmox, homelab]
created: 2026-06-28
salience: 0.8              # decays over time unless re-touched
---
Proxmox node `anvil-01` needs `pve-firewall restart` after any bridge change
or VMs lose the gateway. Learned the hard way 2026-06-28.
```

### Retrieval under a budget

Every call gets a **context budget** (e.g. 8K tokens of the model's window reserved for notes). Retrieval:

1. `INDEX.md` is always in — it's the cheap map of what exists.
2. **Recall** ranks notes by: semantic similarity (Ollama embeddings, cosine) + tag overlap + recency + salience. Pure-keyword fallback if embeddings are off, so it works with zero extra models.
3. **Pack** greedily fills the budget with top-ranked notes by token count (estimated chars/4, or `tiktoken` if installed).
4. **Compress** when the working transcript itself nears the window: the Scribe summarizes older turns into a single rolling-summary note and evicts the raw turns. The conversation never blows the window; it just gets denser.

This is the answer to "reference notes constantly with special care around context limits": notes are always *available* (via the index) but only *loaded* when they earn their token cost, and the transcript self-compacts before it ever hits the wall.

### Constant note-taking

The Scribe runs after every pipeline stage and every cron job, extracting only durable facts (not transcript noise) using a tight local prompt. A nightly `consolidate` job merges duplicates, decays salience, and prunes the index — so the store stays small and the always-loaded `INDEX.md` stays cheap.

---

## 5. Autonomous scheduling (cron)

Anvil owns a `jobs/` directory of job specs and a scheduler loop. **The agent can create, edit, and remove jobs itself** — when a conversation implies recurrence ("check the cert expiry weekly"), the Planner emits a job spec and writes it; no human cron editing required.

```yaml
# jobs/cert-watch.yaml
name: cert-watch
cron: "0 8 * * 1"            # Mondays 08:00, standard 5-field cron
pipeline: shell_then_report
inputs:
  command: "echo | openssl s_client -connect anvil.lan:443 2>/dev/null | openssl x509 -noout -enddate"
notify: discord
min_rung: 0
```

The scheduler uses a small built-in cron parser (5-field, standard syntax) so it has **zero dependencies**; it can optionally hand off to APScheduler if installed for finer control. Jobs run pipelines, and any job can `notify: discord` to push its result to Joe. Safety rail: jobs that run shell commands go through the Critic with a destructive-command denylist before execution, and anything flagged waits for a Discord thumbs-up rather than auto-running.

Natural fits: morning homelab health digest, weekly backup verification, Twitch-schedule reminder an hour before a stream, "did the kids' tablet time-limit job actually run," board-game-night prep the afternoon before.

---

## 6. Reaching you on the go — Discord

Discord is the liaison layer (chosen for the streaming/gaming overlap). Two directions:

- **Outbound (zero-dep):** any job or pipeline can POST to a Discord **webhook** with `urllib` alone — alerts, digests, "your render finished," "cert expires in 6 days." No bot, no library needed.
- **Inbound / two-way (optional):** a `discord.py` bot lets Joe issue commands from his phone — `!ask`, `!status`, `!note`, `!schedule`, `!approve` — routed straight into the pipeline. Replies come back in the thread. Long answers get summarized by the Scribe before sending so mobile stays readable.

Approval flow ties it together: when a job needs a human yes (destructive command, a cloud spend over a cap, publishing something), Anvil DMs Joe with the proposal and waits for `!approve <id>`.

---

## 7. Token & cost discipline (the "smart token use" requirement)

Concrete levers Anvil pulls automatically:

- **Flat-rate cloud.** The Ollama Cloud (Max) subscription means sustained cloud use is already paid for — so the cost lever is *routing* (stay local when you can), not per-token accounting.
- **Batch API for non-urgent work.** Nightly digests, bulk note consolidation, and non-interactive cron jobs go through Batch at 50% off, stacked with caching → as low as ~5% of naive cost.
- **Local-first for volume.** Drafting, classification, extraction, and note-writing never touch a paid token.
- **Budgeted context.** Per-call token ceilings for notes + transcript; compression before overflow.
- **Cost ledger.** Every call logs `{rung, provider, model, in_tok, out_tok, est_cost}` to `ledger.jsonl`. A daily cap (configurable) throttles escalation when hit and pings Discord. You can see exactly where money went.
- **Escalation is logged with its trigger**, so the ladder can be tuned from real data ("cheap-rung escalations a higher rung had to redo" → raise that step's floor).

---

## 8. Component map

```
anvil/
  config.py      # load TOML/JSON + env overrides, validate the ladder
  providers.py   # OllamaLocal, OllamaCloud — all via stdlib urllib
  router.py      # escalation engine, rung selection, cost ledger
  context.py     # token estimation + budget packing + compaction
  memory.py      # note store, frontmatter, ranked recall, consolidation
  scheduler.py   # zero-dep cron parser + job runner
  comms.py       # Discord webhook (zero-dep) + optional bot hooks
  pipeline.py    # Planner/Worker/Critic/Scribe orchestration
  cli.py         # `anvil ask|run|schedule|status|note|serve`
anvil.toml       # the tuned defaults
.env.example     # API keys, webhook URL
requirements.txt # all optional — core runs on pure stdlib
```

**Dependency philosophy:** the core (router, memory, scheduler, context, webhook push) runs on the Python standard library alone — no install, nothing to break at 2am. Optional extras (`discord.py`, `tiktoken`, `apscheduler`, `httpx`) light up richer features when present and degrade gracefully when not. That portability is itself part of punching above the class: the harness is as lean as the model it drives.

---

## 9. Tuning roadmap (after it's running)

1. **Calibrate confidence.** Log Worker confidence vs. whether the Critic vetoed; fit the escalation threshold to your real false-pass rate.
2. **Right-size the ladder.** From `ledger.jsonl`, find steps where a rung consistently failed and got redone one up — raise that step's floor. Find rungs that never get vetoed — lower their floor.
3. **Embeddings on/off.** Measure recall hit-rate with keyword vs. embedding ranking on your actual notes; keep embeddings only if they earn the local compute.
4. **Local-vs-cloud ratio.** Watch the fraction of turns answered fully local; push routine work down to rung 0 so the cloud is reserved for what genuinely needs it.
5. **Local model bake-off.** Swap rung 0 between Qwen3-Coder 30B and Qwen3.6 27B on your task mix; keep whichever vetoes less.

---

## Sources

- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs) · [Ollama API Reference (DeepWiki)](https://deepwiki.com/ollama/ollama/3-api-reference) · [Ollama Cloud Authentication](https://docs.ollama.com/api/authentication)
- [Ollama Cloud Pricing 2026](https://ollama.com/pricing) · [Pro/Max plan breakdown](https://pooya.blog/blog/ollama-cloud-pricing-hardware-requirements-2026/)
- [Best Local LLMs for 24GB VRAM 2026](https://localllm.in/blog/best-local-llms-24gb-vram) · [Best Ollama Models — Coding/RAG/Agents (June 2026)](https://www.morphllm.com/best-ollama-models) · [Qwen3-Coder ranked #1](https://localaimaster.com/models/best-local-ai-coding-models)
