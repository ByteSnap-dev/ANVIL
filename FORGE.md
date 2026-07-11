# The Forge Loop - ANVIL's autonomous self-development

The Forge lets ANVIL improve itself overnight while a hard safety net (git)
guarantees it can never leave you with broken code.

## How a cycle works

1. Commit the current state to a dedicated **`forge-auto`** git branch (it never
   touches your main branch and **never pushes**).
2. Run the full test suite (`anvil doctor`) and gather the mind's dream
   proposals (`test-reports/proposals/`).
3. Hand the failures + a proposal to ANVIL's own coding driver — its **Ollama
   ladder** (local → Ollama Cloud) — with strict rules: make one small change,
   keep every test green, never weaken tests.
4. Re-run the suite.
5. **Commit if green, revert if red.** A change is kept only when the whole
   suite is green AND the test count did not drop (so it can't cheat by deleting
   a test). Everything else is rolled back with `git checkout` / `git clean`.

The worst case for any cycle is a no-op. You wake up to a working tree and a log
of exactly what stuck.

## Requirements

- **git** installed.
- A running **Ollama** (local; plus an Ollama Cloud/Max key for the top rungs).
  The editing is done by ANVIL's own Ollama ladder — no external CLI.

## Run it

```bash
python -m anvil forge --cycles=20      # 20 cycles
python -m anvil forge --minutes=480    # or time-boxed (8h)
python -m anvil forge --dry-run        # test the loop+git WITHOUT calling a model
```
It also runs automatically in deep sleep — see `Pulse-Anvil`.

## Review in the morning

```bash
git log forge-auto --oneline          # what was kept
git diff main..forge-auto             # the full diff
cat dev-reports/forge-log.md          # per-cycle kept/reverted log
```
Like what you see? `git checkout main && git merge forge-auto`. Don't?
`git branch -D forge-auto` and nothing reaches your main branch.

## Safety summary

- Dedicated branch, **never pushes**, never force-anything.
- Test-gate: green-or-revert, every cycle.
- Anti-cheat: reverts if the test count drops.
- Secrets/state (`.env`, `persona.json`, `memory/`, ...) are git-ignored and
  are never committed and survive every revert.

Pair it with `Pulse-Anvil.bat` (which generates the improvement proposals while
it "dreams") and the Forge will have a steady supply of ideas to work through.
