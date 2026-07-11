# Planning & Follow-Through Capability — Design & Scoping

## 1. The problem (observed, not hypothetical)

In a live session Lara was asked to scope the "edit existing jobs" feature. She
planned *well*:

> Established: jobs are JSON in `jobs/`, scheduler in `anvil/scheduler.py`…
> Unknown: haven't read scheduler.py yet…
> Next step: read scheduler.py and grep tools.py for the schedule tool.

…and then **stopped**: "Are you working on the next step?" → "I'm here and ready.
What would you like to work on?"

She had the next action written down and *yielded to the operator instead of doing
it*. This is not a planning failure — it's a **follow-through** failure. The model
emitted a FINAL answer (no tool call) that was really a deferral, and the agent
loop accepted it as "done."

## 2. What ANVIL already has (do NOT reinvent)

`pipeline._agent_loop` already ships most of the machinery:

- **Step + wall-clock budgets** (`ask_time_budget_s`) with a graceful `wrap_up`.
- **Stuck detection** (`_note_repeat`): nudge on the first exact (tool,args,result)
  repeat, break on the second.
- **Planning interval** (smolagents-style): periodic step-back re-plan on long tasks.
- **Context watchdog** + deterministic compaction.
- **Approval yield**: `status:"approve"` is already the clean "truly blocked, need
  the human" exit.

The gap is narrow and specific: **there is no durable, externalized plan the loop
can consult, so it cannot tell "premature deferral" from "genuinely done," and it
cannot resume a multi-step task across turns or sleep cycles.**

## 3. Launchpad (borrow, don't build from scratch)

- **OpenHands `ControlFlag`** (`scratchpad/control_flags.py`): iteration/budget flags
  with headless-aware *limit expansion*. We reuse the pattern to bound auto-continue
  so it can never loop forever, and to allow "headed" sessions to extend the budget.
- **smolagents planning interval**: already in the loop — we connect it to the Plan.
- **todo.md / task-list pattern** (persistent agent task lists, Manus todo.md): the
  durable, externalized, step-status plan artifact the agent maintains and re-reads.

## 4. Design — four components

### 4.1 The Plan artifact (`anvil/plan.py`)

A `Plan` is an ordered list of `Step`s, each with a `status`:
`pending | doing | done | blocked`. Persisted as **actor-scoped** JSON at
`memory_dir/plans/u_<actor>.json` (same privacy model as conversations/memory;
`atomic_write` for crash-safety). One active plan per profile.

```
Step:  { id, text, status, note }
Plan:  { task, steps[], created, updated, actor }
       .open_steps()      -> steps not done/blocked
       .next_step()       -> first open step, or None
       .is_complete()     -> no open steps
       .mark(id, status, note)
```

A `plan` **tool** lets Lara create/update it:
`plan(action="set", steps=[...])`, `plan(action="update", id=, status=, note=)`,
`plan(action="show")`. The tool is SAFE (no approval) and actor-scoped.

### 4.2 The completion gate (the stall fix) — `pipeline._agent_loop`

Before accepting a FINAL/`done` answer, gate it against the Plan:

- If `plan.is_complete()` **or** there is no active plan → accept `done` (today's
  behavior; no regression for one-shot Q&A).
- Else if the final answer is a **legitimate block** — it asks for info only the
  operator has, or the loop already returned `status:"approve"` — → yield (correct).
- Else (open steps remain **and** the answer is a *deferral*: "what would you like…",
  "let me know", "I'm ready") → **do not finish**. Re-inject a user turn:
  *"Plan step N is still open: '<text>'. Continue with it now; don't ask me — act.
  Only stop if you hit an approval or need info I alone have."* and keep looping.

Bounded by an OpenHands-style `ControlFlag`: at most `auto_continue_max` re-injections
per turn (default ~3), expandable in a headed session, so it can never spin forever.
"Deferral" is detected cheaply (short answer + no tool call + interrogative/hand-back
phrasing) — when unsure, prefer to accept `done` (never trap the operator in a loop).

### 4.3 Resume across turns & sleep — `agent_start` / `mind.autopilot`

On task start, load any resumable Plan for the actor. If the incoming task is a
continuation ("keep going", "next step", empty autopilot tick) and an incomplete Plan
exists, thread it into context so Lara resumes mid-plan instead of re-deriving it.
This is what makes "develop edit-jobs during deep sleep" work: a multi-step Plan is
carried across `mind.autopilot` / deep-sleep cycles until `is_complete()`.

### 4.4 Next-action critic — at the planning interval

When the planning interval fires, in addition to re-planning, force a single-line
**next concrete action** ("the next action is: <verb object>") and reconcile it with
the Plan (advance/append steps). Prevents both drift *and* premature stopping.

## 5. Where each change lands

| Component | File | Kind |
|---|---|---|
| Plan artifact + store | `anvil/plan.py` (new) | new module |
| `plan` tool | `anvil/tools.py` | tool def + handler |
| Completion gate + ControlFlag bound | `anvil/pipeline.py` (`_agent_loop`) | core loop |
| AGENT_SYS: "maintain a plan; act on open steps, don't defer" | `anvil/pipeline.py` | prompt |
| Resume load | `anvil/pipeline.py` (`agent_start`), `anvil/mind.py` (`autopilot`) | wiring |
| Next-action critic | `anvil/pipeline.py` (planning interval) | core loop |
| Config: `auto_continue_max`, `plan_enabled` | `anvil/config.py` | config |
| Tests | `anvil/selftest.py` | tests |

## 6. Acceptance criteria

1. A Plan round-trips through the store (create → update step → reload) and is
   actor-scoped (profile A cannot see profile B's plan).
2. `plan` tool creates/updates/shows the active plan.
3. **Stall regression test**: with an open Plan, a model turn that returns a bare
   deferral ("what would you like to work on?") is NOT accepted as done — the loop
   re-injects a continue directive. With NO plan (or a complete one), the same
   deferral IS accepted (no regression for one-shot chat).
4. Auto-continue is bounded: it cannot exceed `auto_continue_max` re-injections;
   an approval yield and a genuine info-request still stop the loop.
5. A multi-step Plan survives a simulated resume (reload + continue).
6. `doctor` stays green; test count does not drop.

## 7. Build order

1. `plan.py` (Plan/Step/store) + `plan` tool + tests. ← **start here**
2. Completion gate + ControlFlag bound + AGENT_SYS + stall regression test.
3. Resume load (agent_start + autopilot) + next-action critic + tests.
4. Config flags + docs; `doctor` green; commit each step.

Each step is independently doctor-green and committed, so the capability lands
incrementally and the loop is never left half-wired.
