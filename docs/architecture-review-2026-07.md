# Architecture review — 2026-07-08

Six parallel deep reviews (memory, routing, chat engine, hive/skills/conversations,
self-dev machinery, serving/sensing) hunting **design-level** weaknesses: state that
resets when it shouldn't, fail-open where fail-closed is required, effort spent on the
wrong path, security seams. Findings are tiered by urgency; each carries its home file.
Tier 0 was fixed immediately (2026-07-08); Tier 1/2 items are tracked as Gitea issues.

## Recurring diseases

1. **In-RAM state that must be durable** — the tick-counter bug's siblings.
2. **Silent failure wearing a 200** — fail-open metering/verdicts/truncation.
3. **Security seams** — unvalidated origins, laundered trust, self-approval.
4. **Effort on the wrong path** — verification/cost spent where risk isn't.

## Tier 0 — active holes (FIXED 2026-07-08)

| # | Finding | Home |
|---|---------|------|
| 0.1 | No Origin/Host/Content-Type validation on POST: in the no-login default, any web page a family device visits can POST `/api/ask` (or any mutating endpoint) — drive-by command of Lara. DNS-rebinding generalizes it. | server.py |
| 0.2 | Reviewer = author: with `selfdev_cloud_first`, coding and review both resolve to cloud-open — glm approves glm's diff ("the author, asked twice") while `auto_promote` merges the result. Also: diffs are truncated to 8k chars then *approved*; the prose fallback parser approves on keyword heuristics. | selfdev.py |
| 0.3 | Promote takes no forge_lock, gates whatever tree is checked out (may be mid-edit), and merges the REMOTE PR based on LOCAL ref state (pushes are best-effort) — the gate's subject and the merge's object can be different SHAs. | promote.py |
| 0.4 | Skills are a prompt-injection persistence vector: web text → background reviewer (weakest rung, holds save_skill, no scan) → permanently trusted prompt context. Provenance is recorded but not used in framing. | skills.py, pipeline.py, tools.py |
| 0.5 | Decision outages become verdicts: council `{}` → approval; rebuttal-gauge `{}` → public rejection of a human's argument; `_is_mine` matches the string "— Lara" so a human quoting her is invisible as a rebuttal. | issuework.py |
| 0.6 | Cost metering fails open: paid call with no usage frame records $0 (the cap goes blind exactly when transport misbehaves); `cached_input_tokens` never populated; native path excludes KV-hit tokens. | providers.py, router.py |
| 0.7 | `dream()` clears STM even when consolidation failed (model down → day's events deleted, zero lessons) and alphabetically starves the 5th+ actor; Mind's MemoryStore has no embedder so dream-written notes are permanently invisible to semantic recall; no backfill. | mind.py, memory.py |

## Tier 1 — structural wins (M each)

| # | Finding | Home |
|---|---------|------|
| 1.1 | Pending approvals / auth / adult-PIN state is RAM-only in a self-restarting system: restart silently drops a parked danger approval (full turn transcript) and logs out every device. Persist via atomic_write under memory_dir; RAM as cache. | server.py |
| 1.2 | Long-conversation compaction runs a blocking model summarize on EVERY turn's hot path and throws the result away; the 400-turn disk cap deletes facts un-summarized. Persist a rolling summary per sid, update incrementally off-path (scribe pattern), fold before capping. | pipeline.py, conversations.py, context.py |
| 1.3 | No rung health memory: a black-holed cloud host costs every call the full retry ladder (~6 min) forever. Three-state circuit breaker per provider:model + cheap connect preflight (~60 lines, mirrors _SPENT_CACHE pattern). | router.py, providers.py |
| 1.4 | Client disconnect never cancels generation: a backgrounded phone burns the single GPU to completion. Tie liveness to /api/progress polls — no poll for 30-60s → cancel_fn returns True. | server.py, pipeline.py |
| 1.5 | Critic guards the wrong path: every plain run() turn pays a critic call; the agent/tool path (highest confabulation risk) ships unverified. Make it signal-driven (tools ran / low confidence / danger / DESTRUCTIVE-first) and apply to the agent path; give the critic the observations. | pipeline.py |
| 1.6 | Driver retries blind: gate emits one bit ("red"), attempt 2 never sees attempt 1's diff or failing test output; multi-edit application is partial (half-applied change passes as whole); DRIVER_RULES tells the model to run doctor (it can't — no tool loop). Thread failure context forward; all-or-nothing edits. | forge.py, selfdev.py |
| 1.7 | Forge edits the LIVE server's working copy: unreviewed model-written code is importable by chat requests during the edit-gate window; kept commits skew disk vs RAM; a hard crash mid-cycle leaves the dirty-tree stall; operator branch checkouts get silently flipped back. Do surgery in a `git worktree`. | forge.py, selfdev.py |
| 1.8 | Two hand-maintained house detectors drift: HA_INTENT primes a live HA snapshot; narrower _HOUSE_RE decides privacy routing — "is the office warm enough" sends sensor states to the cloud. Single source of truth: primed-HA-context ⇒ house turn. | pipeline.py |
| 1.9 | Resume-after-approval runs a degraded engine: no streaming/progress/cancel, min_rung=0, no cloud synthesis — and the danger floor doesn't apply because the executed step is prepended AFTER the loop. Carry TurnState through the approval session. | pipeline.py, server.py |

## Tier 2 — deeper redesigns

| # | Finding | Home |
|---|---------|------|
| 2.1 | Salience lifecycle: decay per-dream not wall-clock; +0.15 for any 1-word overlap with STM (saturates to a dead constant); deletion floor driven by the weakest signal; recall usage never feeds it. ACT-R read-time activation over an access log; sidecar salience.json (stop rewriting every note twice per dream — O(n) churn defeats caches, git, backups). | memory.py, mind.py |
| 2.2 | Retrieval is Θ(n) pure-Python per turn (re-tokenize every note, 768-dim cosine over all); dedup scans all notes under the write lock; INDEX.md read-modify-write per write has NO runtime consumer. SQLite FTS5 as a derived disposable index (stdlib), BM25 → rerank top-k; interim: cache term-sets in _NOTE_CACHE. | memory.py |
| 2.3 | House sensing polls a full /api/states diff every 15-min heartbeat: misses sub-interval transitions (door opened+closed), sees events up to 15 min late. HA websocket `state_changed` subscription; interim: decouple sense cadence (30-60s, model-free) from think cadence. | mind.py, homeassistant.py |
| 2.4 | Hive delegate timeout is illusory: future.result(timeout) doesn't stop the drone and pool __exit__ blocks on it anyway; per-future budgets sum (N×330s); no cancel= passed to _agent_loop (plumbing exists); local-lane semaphore is advisory (timeout → runs anyway); tasks 5-6 silently dropped. One shared deadline threaded down; wait(); shutdown(wait=False, cancel_futures=True); binding semaphore. | hive.py, pipeline.py, tools.py |
| 2.5 | Skill flywheel telemetry counts keyword-match frequency as "use": promiscuous generic skills become immortal and crowd out good ones; inactivity-based curation starves quarterly skills. Split surfaced vs used; score uses/surfaced; dedup at write; hard cap with displacement. | skills.py, pipeline.py |
| 2.6 | Gate economics: full 228-test suite ≤7×/issue (smoke tier + --only= exist unused); one flaky test halts all self-dev AND rewrites the task to "fix the tests"; red promotion batch = permanent silent stall (issues already closed) — needs bisect (git bisect run) + revert-culprit + reopen. Daily cap counts attempts not keeps; a structurally-impossible ticket starves the queue forever (add attempts→stuck). | forge.py, selfdev.py, promote.py |
| 2.7 | Escalation/streaming: retries and rung escalations re-run generation into the SAME live token sink — duplicated partial answers in the chat bubble. Attempt-scoped sink with reset frame. | router.py, server.py |
| 2.8 | Context management: no preflight budget vs rung.max_context (unused field); local Ollama silently truncates (system prompt falls out first, router sees 200); blind 5-msg×4000-char shrink. Preflight estimate + compact; post-flight prompt_eval_count ≪ estimate ⇒ truncated ⇒ retry compacted. | router.py, providers.py |
| 2.9 | Schema enforcement diverges across 3 paths; native /api/chat path silently DROPS schema (tools/images/tool-history force that path) — Ollama native `format` field supports it. Unify; in-band injection for all eff_schema. | providers.py, router.py |
| 2.10 | AGENT_SYS accretes a paragraph per incident (~1100 words to weak local models). Migrate situational paragraphs behind triggers (the _research_hint pattern). | pipeline.py |
| 2.11 | Assorted: shell_timeout (600s) > turn budget (240s) — cap tool runtime at remaining budget; scheduled jobs run the tool-less run() path but the schedule tool promises actions; context assembly is serial HTTP (fan out recall/skills/HA/weather); sid travels in GET query strings and adult-until binds to it (bind to auth token); introspect harness_bug=false decisions are permanent (decay by recurrence count); privilege-bearing state on client-supplied sid. | tools.py, server.py, pipeline.py, introspect.py |

## Verified solid (don't churn)

Push crypto (RFC 8291 + VAPID, prune-on-410) · web_fetch SSRF defense (IP pinning,
redirect re-vetting) · transcript hygiene (tool-call trimming, compaction edge-snapping)
· git seatbelt (finally-revert, test-count anti-cheat, O_EXCL lock) · fork-bomb defense
in depth (4 layers) · structural drone tool narrowing (schema omission, not prompt) ·
scribe/reviewer GPU etiquette (foreground gate, self-cancel) · ledger caching/rotation ·
streaming per-line timeout reset · conversations durability (fsync, atomic fallback,
traversal-safe sids) · issue state machine (assignee-as-turn, marker comments) ·
faithfulness gates · per-actor privacy scoping throughout the pulse/mind views.
