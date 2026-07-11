"""The Forge Loop -- ANVIL's autonomous, git-guarded self-development pipeline.

Each cycle:
  1. Make sure the working tree is committed (a clean baseline on a dedicated
     branch -- never the user's main branch, and it NEVER pushes).
  2. Run the test suite (`anvil doctor`) and gather the mind's dream proposals.
  3. Hand the failures + a proposal to a coding driver (ANVIL's own Ollama
     ladder, local -> Ollama Cloud) with strict rules: keep every test green,
     never weaken tests.
  4. Re-run the suite.
  5. **Commit if green, revert if red.** A change is kept only when the whole
     suite is green AND the number of tests did not drop (so it can't "win" by
     deleting a test). Anything else is rolled back with `git checkout`/`clean`.

The git revert is the seatbelt: the loop can churn all night and the worst case
is a no-op cycle. Secrets and local state are git-ignored, so nothing sensitive
is ever committed.

``driver``, ``tester`` and ``test_count_fn`` are injectable so the loop logic is
fully unit-testable without invoking a real model.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEV_REPORTS = ROOT / "dev-reports"
PROPOSALS = ROOT / "test-reports" / "proposals"

# Single-writer lock. Only ONE forge/self-dev/issue-work run may touch the working
# tree at a time — a second concurrent run (server deep-sleep + a manual run, two
# overlapping deep-sleeps, etc.) would fight over the branch and cause spurious
# failures/escalations. Gitignored so it never dirties the tree.
_LOCK_PATH = ROOT / ".forge.lock"


import contextlib as _contextlib


@_contextlib.contextmanager
def forge_lock(stale_s: int = 1800, path: Optional[Path] = None):
    """Yield True if this process grabbed the exclusive forge lock, else False.
    Steals a lock older than ``stale_s`` (a crashed run left it behind).

    ``path`` overrides the lock file — tests MUST pass a temp path so they never
    contend with the REAL lock, which the forge legitimately holds while its test
    gate (`anvil doctor`) runs."""
    lock = Path(path) if path else _LOCK_PATH
    acquired = False

    def _grab() -> bool:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False
        except OSError:
            return False

    try:
        if not _grab():
            try:                            # steal a stale lock from a dead run
                if time.time() - os.path.getmtime(str(lock)) > stale_s:
                    os.unlink(str(lock))
                    acquired = _grab()
            except OSError:
                pass
        else:
            acquired = True
        yield acquired
    finally:
        if acquired:
            try:
                os.unlink(str(lock))
            except OSError:
                pass

DRIVER_RULES = (
    "You are improving the ANVIL harness, a local Python project. Work in the "
    "current directory.\n"
    "RULES (non-negotiable):\n"
    "- Make ONE small, focused change this run.\n"
    "- `python -m anvil doctor` MUST stay green (exit 0); the harness runs it "
    "on your change and reverts anything red.\n"
    "- NEVER delete, skip, rename-away, or weaken any existing test. Prefer "
    "adding tests. The harness counts tests and will revert you if the count "
    "drops.\n"
    "- Always pin encoding='utf-8' on file I/O; keep .ps1/.bat launchers "
    "ASCII-only; keep config loading tolerant of corruption.\n"
    "- Do NOT touch .env or secrets, do NOT run git push, do NOT delete user "
    "data (memory/, persona.json, anvil.toml).\n"
)


@dataclass
class CycleResult:
    n: int
    kept: bool
    reason: str
    before_failed: int
    after_failed: int
    after_passed: int
    # Failure evidence for iterate-on-failure (review 1.6): what the reverted
    # attempt changed and which tests it broke — thread into the NEXT attempt's
    # prompt so retries stop being blind re-derivations of the same mistake.
    fail_diff: str = ""
    fail_out: str = ""


# Every autonomous forge/self-dev commit is authored as Lara, so a genuine
# self-improvement cycle is always distinct from the operator's own work (the
# repo's user identity) regardless of git config — and links to her Gitea
# profile (register lara@anvil.local on her account). She owns what she writes.
FORGE_AUTHOR = "Lara <lara@anvil.local>"


class Forge:
    def __init__(self, root: Path = ROOT, branch: str = "test",
                 driver: Optional[Callable[[str], str]] = None,
                 tester: Optional[Callable[[], Tuple[bool, int, int, str]]] = None,
                 test_count_fn: Optional[Callable[[], int]] = None,
                 reviewer: Optional[Callable[[str], Tuple[bool, str]]] = None,
                 push_remote: Optional[str] = None,
                 push_token: Optional[str] = None,
                 logger: Callable[[str], None] = print):
        self.root = Path(root)
        self.branch = branch
        # The coding driver is ANVIL's OWN Ollama ladder (selfdev.build_local_driver).
        # No external driver: if none is supplied, _no_driver raises a clear error.
        self.driver = driver or self._no_driver
        self.tester = tester or self._doctor
        self.test_count_fn = test_count_fn or self._count_tests
        # Optional consensus gate: given the staged diff, returns (approved,
        # reason). When set, a green change is committed ONLY if the reviewer also
        # approves — "the author's change AND an independent reviewer agree",
        # brought in-harness. Default None keeps the classic green-or-revert loop.
        self.reviewer = reviewer
        # Optional explicit work item (e.g. a claimed council-queue ticket). When
        # set, it's used as the cycle's proposal instead of the dream archive, so
        # the internal self-dev and the external council share ONE backlog.
        self.work_item: Optional[str] = None
        self.log = logger
        # Auto-push: when both are set, each kept commit is pushed to this remote
        # with the token (best-effort — never raises, and the token is never logged).
        self.push_remote = push_remote
        self.push_token = push_token
        self.dev_reports = self.root / "dev-reports"
        self.proposals_dir = self.root / "test-reports" / "proposals"
        # Surgery happens in a git WORKTREE, never the live checkout (review 1.7):
        # the running server imports from self.root, so editing it in place put
        # unreviewed model-written code one lazy import away from the family's
        # assistant, skewed disk vs RAM after keeps, and made Lara and a human
        # editing at the same time clobber each other (happened live). The live
        # tree now only ever receives a --ff-only merge of a committed, gated,
        # reviewed change.
        self.work_root = self.root / ".forge-work"
        self.wip_branch = "forge-wip"

    # -- git helpers --------------------------------------------------- #
    def _git(self, *args: str) -> Tuple[int, str]:
        p = subprocess.run(["git", *args], cwd=str(self.root),
                           capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()

    def _wgit(self, *args: str) -> Tuple[int, str]:
        p = subprocess.run(["git", *args], cwd=str(self.work_root),
                           capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()

    def _ensure_worktree(self) -> bool:
        """Create/reset the surgery worktree at `branch`'s tip on its own
        `forge-wip` branch (a branch can be checked out in only ONE worktree,
        and the live checkout holds `branch`). Also copies the gitignored env/
        config in so the test gate runs with real settings."""
        try:
            self._git("worktree", "prune")
            if not (self.work_root / ".git").exists():
                code, out = self._git("worktree", "add", str(self.work_root),
                                      "-B", self.wip_branch, self.branch)
                if code != 0:
                    self.log(f"[forge] worktree add failed: {out[:160]}")
                    return False
            else:
                self._wgit("checkout", "-B", self.wip_branch, self.branch)
                self._wgit("checkout", "--", ".")
                self._wgit("clean", "-fd")
            for name in (".env", "anvil.toml"):      # gitignored -> not checked out
                src = self.root / name
                if src.exists():
                    (self.work_root / name).write_bytes(src.read_bytes())
            return True
        except Exception as exc:
            self.log(f"[forge] worktree setup failed: {exc}")
            return False

    def ensure_repo(self) -> None:
        if not (self.root / ".git").exists():
            self.log("[forge] initializing git repo")
            self._git("init")
            # Only seed a committer identity if the operator hasn't set one — never
            # clobber their git config. Forge's OWN commits stamp --author (below),
            # so they read as the harness regardless of the operator's identity.
            code, out = self._git("config", "user.email")
            if code != 0 or not out.strip():
                self._git("config", "user.email", "lara@anvil.local")
                self._git("config", "user.name", "Lara")
        self._ensure_gitignore()
        # Commit a baseline ONLY when there's something uncommitted to protect
        # from a later revert. Skipping the empty commit keeps forge-auto history
        # clean — no "forge: baseline" noise every cycle (esp. during sleep).
        if self._dirty():
            self._git("add", "-A")
            self._git("commit", "-m", "forge: baseline", "--author", FORGE_AUTHOR)
        # dedicated branch; never touch the user's current branch history
        code, _ = self._git("rev-parse", "--verify", self.branch)
        if code != 0:
            self._git("branch", self.branch)
        self._git("checkout", self.branch)

    def _ensure_gitignore(self) -> None:
        gi = self.root / ".gitignore"
        needed = [".env", "persona.json", "anvil.toml", "memory/", "jobs/",
                  "ledger.jsonl", "test-reports/", "dev-reports/", "workspace/",
                  ".venv/", "__pycache__/", "*.pyc", "*.tmp", ".anvil_write_probe",
                  ".forge.lock", ".forge-work/"]
        have = gi.read_text("utf-8", "replace").splitlines() if gi.exists() else []
        missing = [x for x in needed if x not in have]
        if missing:
            from . import config as cfgmod
            cfgmod.atomic_write(gi, "\n".join(have + missing).strip() + "\n")

    def _dirty(self) -> bool:
        _, out = self._git("status", "--porcelain")
        return bool(out.strip())

    def _commit(self, msg: str) -> bool:
        """Commit the gated change in the WORKTREE, then fast-forward the live
        checkout to it. The live tree is only touched by this atomic ff-merge —
        and only when it is clean and on the forge branch; if the operator
        started editing mid-cycle, their tree is sacred: the change is dropped
        (stateless) and the driver simply redoes it next cycle."""
        self._wgit("add", "-A")
        code, out = self._wgit("commit", "-m", msg, "--author", FORGE_AUTHOR)
        if code != 0:
            self.log(f"[forge] worktree commit failed: {out[:160]}")
            return False
        cur = self._git("rev-parse", "--abbrev-ref", "HEAD")[1]
        if self._dirty() or cur != self.branch:
            self.log("[forge] live tree busy — gated change NOT merged (redo next cycle)")
            self._wgit("reset", "--hard", self.branch)   # drop the wip commit
            return False
        code, out = self._git("merge", "--ff-only", self.wip_branch)
        if code != 0:
            self.log(f"[forge] ff-merge failed: {out[:160]}")
            self._wgit("reset", "--hard", self.branch)
            return False
        self._maybe_push()
        return True

    def _maybe_push(self) -> None:
        """Best-effort off-box backup: push the forge branch to the configured
        remote using the token. NEVER raises (a failed or absent push must not
        break the forge cycle) and NEVER logs the token. Bounded by a timeout so
        an unreachable remote can't hang the cycle."""
        if not (self.push_remote and self.push_token):
            return
        code = 1
        try:
            rc, url = self._git("remote", "get-url", self.push_remote)
            url = url.strip()
            if rc == 0 and "://" in url:
                scheme, rest = url.split("://", 1)
                rest = rest.split("@", 1)[-1]          # drop any embedded creds
                push_url = f"{scheme}://{self.push_token}@{rest}"
                env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # never prompt/hang
                p = subprocess.run(["git", "push", push_url, self.branch],
                                   cwd=str(self.root), capture_output=True,
                                   text=True, timeout=60, env=env)
                code = p.returncode
        except Exception:
            code = 1
        # Log only a generic result — the push URL (with token) is never emitted.
        self.log("[forge] pushed to remote" if code == 0
                 else "[forge] remote push failed — kept locally, retry next cycle")

    def _revert(self) -> None:
        # Reverts touch the WORKTREE only — the live checkout is never reset.
        self._wgit("checkout", "--", ".")
        self._wgit("clean", "-fd")       # drop new untracked files (ignored kept)

    def _wdirty(self) -> bool:
        _, out = self._wgit("status", "--porcelain")
        return bool(out.strip())

    def _diff(self) -> str:
        # Everything the driver changed vs HEAD (tracked edits + new files).
        self._wgit("add", "-A")
        _, out = self._wgit("diff", "--cached", "--stat")
        _, full = self._wgit("diff", "--cached")
        self._wgit("reset", "-q")        # unstage; commit stages again if we keep
        return (out + "\n\n" + full).strip()

    # -- testing ------------------------------------------------------- #
    def _doctor(self) -> Tuple[bool, int, int, str]:
        # The gate runs IN the worktree: it must judge the edited code, and the
        # live checkout must never import a candidate change.
        p = subprocess.run([sys.executable, "-m", "anvil", "doctor"],
                           cwd=str(self.work_root), capture_output=True, text=True,
                           timeout=600)
        out = p.stdout + p.stderr
        m = re.search(r"passed=(\d+)\s+failed=(\d+)", out)
        passed = int(m.group(1)) if m else 0
        failed = int(m.group(2)) if m else (0 if p.returncode == 0 else 1)
        return (p.returncode == 0 and failed == 0, passed, failed, out)

    def _count_tests(self) -> int:
        f = self.work_root / "anvil" / "selftest.py"
        if not f.exists():
            f = self.root / "anvil" / "selftest.py"
        if not f.exists():
            return 0
        return len(re.findall(r'@test\(', f.read_text("utf-8", "replace")))

    # -- gate economics (review 2.6) ------------------------------------ #
    # The full suite ran up to 7x per issue (before+after x 3 attempts + the
    # promotion gate) even when nothing changed between runs. Three levers:
    # a SHA-keyed cache (a clean tree just verified green stays verified for a
    # TTL — the promote gate reuses the forge's own post-commit run), a one-
    # retry flake absorber (one flaky test used to halt ALL self-dev and
    # rewrite the task into "fix the failing tests"), and an import smoke that
    # fails a syntax/import break in seconds instead of a full suite run.
    _GATE_TTL_S = 1800

    def _gate_cache_path(self) -> Path:
        return self.dev_reports / "gate-cache.json"

    def _gate_cache_get(self, sha: str):
        try:
            d = __import__("json").loads(
                self._gate_cache_path().read_text("utf-8"))
            rec = d.get(sha)
            if rec and time.time() - float(rec.get("ts", 0)) < self._GATE_TTL_S:
                return rec
        except Exception:
            pass
        return None

    def _gate_cache_put(self, sha: str, green: bool, passed: int,
                        failed: int, tail: str = "") -> None:
        try:
            import json as _json
            p = self._gate_cache_path()
            try:
                d = _json.loads(p.read_text("utf-8"))
            except Exception:
                d = {}
            d[sha] = {"green": bool(green), "passed": int(passed),
                      "failed": int(failed), "ts": time.time(),
                      "tail": str(tail)[-1500:]}
            if len(d) > 20:                       # keep it tiny
                d = dict(sorted(d.items(), key=lambda kv: kv[1]["ts"])[-20:])
            p.parent.mkdir(parents=True, exist_ok=True)
            from . import config as cfgmod
            cfgmod.atomic_write(p, _json.dumps(d))
        except Exception:
            pass

    def _import_smoke(self) -> str:
        """Try to import every module the driver touched — a syntax/import
        break (the most common red) fails in seconds, not a suite run.
        Returns '' when clean, else the error."""
        try:
            # -uall: plain porcelain collapses an untracked DIRECTORY to 'anvil/',
            # hiding the .py files inside it from the smoke.
            _, out = self._wgit("status", "--porcelain", "-uall")
            files = [ln[3:].strip().strip('"') for ln in out.splitlines() if ln.strip()]
            mods = [f[6:-3].replace("/", ".") for f in files
                    if f.startswith("anvil/") and f.endswith(".py")]
            for m in mods[:6]:
                p = subprocess.run([sys.executable, "-c", f"import anvil.{m.split('.')[-1]}"],
                                   cwd=str(self.work_root), capture_output=True,
                                   text=True, timeout=60)
                if p.returncode != 0:
                    return (p.stderr or p.stdout).strip()[-400:]
        except Exception:
            pass                                   # smoke is advisory only
        return ""

    # -- driver -------------------------------------------------------- #
    def _no_driver(self, prompt: str) -> str:
        raise RuntimeError(
            "Forge has no coding driver. Pass driver=selfdev.build_local_driver(cfg) "
            "— ANVIL's own Ollama (local -> Ollama Cloud) self-dev driver. There is "
            "no external driver.")

    def _proposals(self, n: int = 2) -> str:
        if not self.proposals_dir.exists():
            return ""
        files = sorted(self.proposals_dir.glob("*.md"))[-n:]
        return "\n\n".join(f.read_text("utf-8", "replace")[:1500] for f in files)

    def build_prompt(self, doctor_out: str, proposals: str,
                     assigned: bool = False) -> str:
        tail = "\n".join(doctor_out.strip().splitlines()[-25:])
        # An ASSIGNED work item (issue work) is THE task — it used to ride the
        # 'dream proposals' slot under an 'implement one small reliability
        # improvement, ideally from the proposals' framing, so a feature-shaped
        # issue read as out-of-scope and the planner returned empty file lists
        # (issue #95 refused 12x this way).
        if "failed=0" not in doctor_out:
            task = "There are FAILING tests below -- fix the root cause."
        elif assigned:
            task = ("Implement the ASSIGNED WORK ITEM below — exactly that, "
                    "nothing else. It may be a feature, a fix, or a doc; its "
                    "size is whatever the item needs (still the SMALLEST "
                    "correct change that fully does it).")
        else:
            task = ("All tests pass. Implement ONE small reliability/efficiency "
                    "improvement, ideally from the proposals below.")
        # Iterate-on-failure (review 1.6): show the model exactly what its last
        # reverted attempt changed and which tests broke — otherwise every retry
        # blindly re-derives (usually) the same mistake from the same prompt.
        retry = ""
        if getattr(self, "retry_context", ""):
            retry = ("\n\n--- YOUR PREVIOUS ATTEMPT (it was REVERTED) ---\n"
                     + self.retry_context +
                     "\nDo better this time: fix WHY it failed, or take a "
                     "different approach — do not resubmit the same change.")
        section = "ASSIGNED WORK ITEM" if assigned else "dream proposals"
        return (DRIVER_RULES + "\nTASK THIS CYCLE: " + task +
                "\n\n--- `anvil doctor` output ---\n" + tail +
                (f"\n\n--- {section} ---\n" + proposals if proposals else "") +
                retry +
                "\n\nMake the change now; the harness will verify it is green.")

    # -- one cycle ----------------------------------------------------- #
    def cycle(self, n: int) -> CycleResult:
        # Never run on a dirty tree. self-dev and issue-work share the `test` branch
        # with the operator, and our revert path (git checkout -- . / clean) would
        # DISCARD their uncommitted work. Skip until the tree is clean — the operator's
        # in-progress edits are sacred.
        if self._dirty():
            return CycleResult(n, False, "skipped: uncommitted changes in tree", 0, 0, 0)
        # All surgery happens in the worktree from here on; the live checkout is
        # read-only to this cycle until (at most) the final --ff-only merge.
        if not self._ensure_worktree():
            return CycleResult(n, False, "skipped: worktree unavailable", 0, 0, 0)
        _, tip = self._wgit("rev-parse", "HEAD")
        cached = self._gate_cache_get(tip.strip())
        if cached:
            before_green, bp, bf = cached["green"], cached["passed"], cached["failed"]
            before_out = cached.get("tail", "") or f"failed={bf}"
        else:
            before_green, bp, bf, before_out = self.tester()
            self._gate_cache_put(tip.strip(), before_green, bp, bf,
                                 "\n".join(before_out.strip().splitlines()[-25:]))
        before_count = self.test_count_fn()
        prompt = self.build_prompt(before_out, self.work_item or self._proposals(),
                                   assigned=bool(self.work_item))
        try:
            outcome = self.driver(prompt)
        except Exception as exc:
            self.log(f"[forge] driver error: {exc}")
            self._revert()
            return CycleResult(n, False, f"driver-error: {exc}", bf, bf, bp)

        # Once the driver has edited the worktree, EVERY non-kept exit must revert — a
        # return OR an exception (e.g. a doctor timeout in tester()). The finally is the
        # single guaranteed cleanup (harness-bug #10) — and even a hard crash now leaves
        # debris only in the worktree, which resets itself at the next cycle's setup.
        kept = False
        try:
            if not self._wdirty():                 # nothing changed: no suite run
                # Carry the DRIVER'S OWN reason instead of a bare 'no-change' —
                # a cap-hit, an allowlist refusal, and a missed edit are three
                # different problems, and the issue escalation quotes this
                # string verbatim (Lara burned 3 attempts on 'no-change' that
                # was really 'cap-hit@claude-sonnet:selfdev').
                why = str(outcome or "").strip()[:200]
                return CycleResult(n, False,
                                   f"no-change: {why}" if why else "no-change",
                                   bf, bf, bp)
            smoke = self._import_smoke()
            if smoke:                              # seconds, not a suite run
                return CycleResult(n, False, "import-error", bf, bf, bp,
                                   fail_diff=self._diff()[:3000],
                                   fail_out=f"import failed: {smoke}")
            after_green, ap, af, after_out = self.tester()
            if not after_green:
                # One retry absorbs a flaky test — a single flake used to halt
                # ALL self-dev and rewrite the task into "fix the failing tests".
                self.log("[forge] gate red — one retry (flake check)")
                after_green, ap, af, after_out = self.tester()
            after_count = self.test_count_fn()
            if after_count < before_count:
                return CycleResult(n, False,
                                   f"tests-dropped {before_count}->{after_count}", bf, af, ap,
                                   fail_diff=self._diff()[:3000],
                                   fail_out=f"test count dropped {before_count}->{after_count}")
            if not after_green:
                # Capture the evidence BEFORE the finally-revert destroys it —
                # the next attempt gets the diff + the failing tests' output.
                return CycleResult(n, False, f"red (failed={af})", bf, af, ap,
                                   fail_diff=self._diff()[:3000],
                                   fail_out="\n".join(
                                       after_out.strip().splitlines()[-40:]))
            # Consensus gate: green isn't enough — an independent reviewer must also
            # approve the diff before it's kept (brought in-harness from the council).
            if self.reviewer is not None:
                try:
                    approved, why = self.reviewer(self._diff())
                except Exception as exc:
                    approved, why = False, f"reviewer-error: {exc}"
                if not approved:
                    # Carry the reviewer's FULL complaint (a 120-char cap fed
                    # the issue thread rejections cut off exactly before the
                    # actionable part — #109 stalled three attempts on it).
                    return CycleResult(n, False,
                                       f"reviewer-rejected: {why}"[:700], bf, af, ap,
                                       fail_diff=self._diff()[:3000],
                                       fail_out=f"reviewer rejected: {why}")
            if not self._commit(f"forge: cycle {n} kept (pass {ap}/{ap})"):
                return CycleResult(n, False, "kept-not-merged: live tree busy", bf, af, ap)
            kept = True
            # The suite is green on exactly this new tip — the promotion gate
            # can reuse this run instead of paying its own within the TTL.
            _, new_tip = self._git("rev-parse", self.branch)
            self._gate_cache_put(new_tip.strip(), True, ap, af)
            return CycleResult(n, True, "kept", bf, af, ap)
        except Exception as exc:
            self.log(f"[forge] cycle error after edit: {exc}")
            return CycleResult(n, False, f"cycle-error: {exc}"[:120], bf, bf, bp)
        finally:
            if not kept and self._wdirty():
                self._revert()      # guaranteed: no non-kept cycle leaves the worktree dirty

    # -- the loop ------------------------------------------------------ #
    def run(self, cycles: int = 20, minutes: int = 0) -> List[CycleResult]:
        self.ensure_repo()
        self.dev_reports.mkdir(parents=True, exist_ok=True)
        logf = self.dev_reports / "forge-log.md"
        end = time.time() + minutes * 60 if minutes else None
        results: List[CycleResult] = []
        self.log(f"[forge] starting on branch '{self.branch}' "
                 f"({cycles} cycles{', ' + str(minutes) + ' min cap' if minutes else ''})")
        for i in range(1, cycles + 1):
            r = self.cycle(i)
            results.append(r)
            mark = "KEPT " if r.kept else "rvrt "
            line = (f"- {datetime.now().strftime('%H:%M')} cycle {i}: {mark}"
                    f"({r.reason}) failed {r.before_failed}->{r.after_failed}")
            self.log("[forge] " + line)
            with logf.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if end and time.time() >= end:
                break
        kept = sum(1 for r in results if r.kept)
        self.log(f"[forge] done: {kept}/{len(results)} cycles kept. Review: "
                 f"git log {self.branch}  |  diff: git diff main..{self.branch}")
        return results
