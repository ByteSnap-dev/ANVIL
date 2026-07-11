# Testing ANVIL

ANVIL ships with its own self-test harness so it can verify every part of
itself — including talking to your real Ollama models.

## One-shot check

```powershell
python -m anvil doctor          # fast: unit + regression + UI lint (no models)
python -m anvil doctor --live   # also tests your local Ollama + Ollama Cloud
```

Exit code is 0 when everything passes, 1 on any failure. A report is written to
`test-reports\` and the human-readable summary to `test-reports\latest.txt`.

## Overnight soak (churn while you sleep)

Double-click **`Soak-Anvil.bat`** (or `./Soak-Anvil.ps1`) before bed. It runs
the **full battery, including live model tests**, on a cycle for 8 hours by
default, logging each pass. On any failure it asks a reachable model to triage
the likely cause and writes suggestions to `test-reports\proposals\`.

```powershell
./Soak-Anvil.ps1 -Minutes 480 -Interval 15   # defaults
./Soak-Anvil.ps1 -NoLive                     # skip model tests
```

In the morning, read **`test-reports\soak-summary.md`** — one line per cycle
(pass/fail/skip) and, for any failed cycle, the failing tests plus a pointer to
the triage proposals. **The loop never edits your code** — it tests, diagnoses,
and reports, leaving fixes for you to review.

## What gets tested

- **Unit/integration:** config, persona, tools + sandbox safety, memory recall,
  scheduler/cron, router escalation, the agent tool loop (safe + approval gate),
  and the web server endpoints.
- **Regression (bugs that already bit us, pinned so they can't return):** file
  truncation, cp1252/UTF-8 encoding, the wizard `[hidden]` CSS override, no-store
  cache headers, the wizard close path, and **UI JavaScript syntax** (via Node if
  present).
- **Live (your models):** each reachable model answers; structured outputs are
  honored (with a free-form fallback when they aren't); the agent uses a tool
  end-to-end; embeddings work.

## Adding a test

Open `anvil/selftest.py` and add a function decorated with `@test("name")`
(or `@test("name", live=True)` for one that needs a model). Raise/`assert` on
failure; return a short string on success, or `("SKIP", "reason")` to skip.
