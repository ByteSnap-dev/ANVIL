# ANVIL's Mind — autonomy, two-tier memory, sleep & dreaming

ANVIL has a mind layer that lets it think between tasks and learn from
experience — the difference between a tool you drive and something that grows.

## The pieces

- **Short-term memory (STM)** — `memory\short_term.jsonl`, a capped rolling
  buffer of recent events, your questions, and ANVIL's own thoughts.
- **Long-term memory (LTM)** — the durable note store; "lessons" and "about-you"
  facts distilled from experience live here and are recalled into every task.
- **Heartbeat (`think`)** — a cheap periodic tick that produces one spontaneous,
  useful thought from what's recently on its mind. The spark of autonomy.
- **Sleep / dream (`dream`)** — consolidation: it reviews STM, writes durable
  **lessons** and **operator facts** into LTM, raises **open questions** (seeded
  back into STM), and writes concrete **self-improvement proposals** to
  `test-reports\proposals\`. Then it decays old memories, lets its persona
  evolve, and prunes STM — just like sleep clearing the day.
- **Pulse** — the circadian loop: frequent heartbeats, periodic dreams.

## Use it

In the web UI, open the **Mind** tab: watch the thought stream, hit **Think now**
or **Dream now**. Or from the CLI:

```powershell
python -m anvil think     # one heartbeat
python -m anvil dream      # one consolidation pass
python -m anvil pulse      # the loop (8h default)
```

**For overnight autonomy, run `Pulse-Anvil.bat`** (heartbeat every 10 min, dream
every 6 ticks). In the morning, read `memory\journal.md` for its train of
thought and `test-reports\proposals\` for the improvements it dreamed up.

## Self-improvement — the safety stance

The dream's self-improvement ideas are written as **proposals for your review**.
ANVIL does **not** edit its own code automatically. The guarded path we can
enable later: apply a proposed change to a *copy*, run `anvil doctor`, and
promote it only if every test stays green — autonomy with a seatbelt.

Run `Pulse-Anvil.bat` (thinking) and `Soak-Anvil.bat` (self-testing) together to
let ANVIL both reflect and verify itself all night.
