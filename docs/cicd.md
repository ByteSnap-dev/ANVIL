# ANVIL CI/CD — the self-improving promotion pipeline

The goal: a repository that improves itself **continuously and safely**. Changes
never land on `main` until they've passed layered scrutiny; the harness and Lara
drive an endless loop of *file an issue → fix it on `test` → verify through gates →
promote to `main` → monitor → repeat*. `main` is what the family runs; `test` is
where everything is tried first.

## Branch model

- **`test`** — the working branch. Every autonomous self-dev commit, every fix, and
  every dev change lands here first. Fast-moving. Every commit is already
  doctor-green (the forge's git-guard). This is the old `forge-auto`, renamed.
- **`main`** — the protected, stable branch. The family runs it. It advances **only**
  by a reviewed, gated promotion from `test` — never a direct push.

## The gates a change passes before it reaches `main`

1. **Commit-time (exists today).** The forge is git-guarded: every commit on `test`
   is `doctor`-green or it's reverted, and the test count can't drop (no cheating by
   deleting tests). A red change never even lands on `test`.
2. **Review over time (Lara).** On a schedule (deep sleep), a review pass reads the
   accumulated `test..main` diff and judges it — correctness, scope, regressions,
   whether it's promotion-worthy. Small/risky batches wait; solid batches are
   proposed for promotion. This is the "scrutinized over time" layer.
3. **External CI (Gitea Actions).** `.gitea/workflows/ci.yml` runs `python -m anvil
   doctor` (and, over time, more suites Lara writes) on every push to `test`, every
   PR to `main`, and every push to `main` — on a runner, in a clean checkout. This is
   ground truth *independent of the harness's own self-report* — the antidote to the
   harness confabulating "it's green." (Activates when the operator registers a
   Gitea Actions runner.)
4. **Promotion PR (gated).** Promotion is a **pull request `test → main`**, not a
   push. It merges only when: CI is green **and** Lara's review approved **and** (by
   policy) the operator approves. The PR is the audit trail — the "true revision
   history with layers of testing before it hits main."
5. **Post-`main` monitoring.** CI also runs on `main`; the family uses it live; a
   monitor pass watches for regressions that slipped through and **files a new Gitea
   issue** — which re-enters the loop at step 0.

## The loop

```
   ┌─────────────────────────────────────────────────────────────┐
   │  Gitea Issue (filed by Lara, the harness, CI, or you)         │
   │      label: selfdev / bug / idea                              │
   └───────────────┬─────────────────────────────────────────────┘
                   ▼
   forge claims the issue → works the fix on `test`
                   ▼   (Gate 1: doctor-green-or-revert per commit)
   commits accumulate on `test`  ──►  auto-push to Gitea (done)
                   ▼   (Gate 3: CI runs doctor on every push to test)
   Lara review pass reads test..main  (Gate 2: scrutinized over time)
                   ▼   approved?
   open Promotion PR  test → main
                   ▼   (Gate 3: CI on the PR)  (Gate 4: review + your approval)
   merge to `main`  →  issue auto-closed, referenced in the PR
                   ▼   (Gate 5: CI on main + real family use)
   monitor finds a regression?  →  files a new Issue  ──► back to top
```

## Components

| Component | State | File |
|---|---|---|
| Gitea API client (issues + PRs) | **build now** | `anvil/gitea.py` |
| `gitea_issue` tool (Lara files/works issues from chat) | **build now** | `anvil/tools.py` |
| CI workflow (doctor on test/PR/main) | **build now** (runs when runner exists) | `.gitea/workflows/ci.yml` |
| `test` working branch + forge commits there | **build now** | branch + `forge_branch` config |
| Auto-push of `test` to Gitea | **done** | `anvil/forge.py` |
| Issue → forge work-queue bridge | Phase 2 | `anvil/selfdev.py` |
| Lara review pass + auto-promotion PR | Phase 3 | `anvil/mind.py` (deep sleep) |
| `main` branch protection (require CI + review) | Phase 4 (needs runner) | Gitea settings / API |
| Post-`main` regression monitor → files issues | Phase 4 | `anvil/mind.py` |

## Phased rollout

- **Phase 1 — foundation (now):** Gitea API client, the `gitea_issue` tool, the CI
  workflow file, and the `test`/`main` branch model. After this, Lara + you can file
  and work issues, self-dev lands on `test`, and CI is ready the moment a runner is up.
- **Phase 2 — issues as the queue:** the forge pulls open `selfdev`-labelled issues
  as work items, references them in commits, and closes them on promotion.
- **Phase 3 — autonomous promotion:** the deep-sleep review pass judges `test..main`
  and opens a gated promotion PR when it's confident.
- **Phase 4 — full protection + monitoring:** `main` branch protection (CI + review
  required), and a post-`main` monitor that files issues for regressions — closing
  the loop. Needs the Gitea Actions runner.

## Safety invariants

- **`main` is never pushed to directly** — only merged via a gated PR.
- **CI is independent of the harness's self-report** — a clean-checkout `doctor` run
  is the source of truth, so a confident-but-wrong self-assessment can't promote.
- **The operator can always gate** — promotion policy can require your approval; you
  can protect `main` so nothing merges without it.
- **Every promotion is a PR** — permanent, reviewable revision history.
