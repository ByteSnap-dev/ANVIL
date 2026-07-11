"""Phase 3 of the CI/CD loop — the reviewed promotion of `test` -> `main`.

Lara's forge lands every fix on `test` (each already passed the forge's independent
reviewer + test gate). `main` is the STABLE branch, reached only through this gate:

  1. Is `test` actually ahead of `main`? (else nothing to do)
  2. Re-run the FULL test suite on `test` — the final CI gate before release.
  3. Open (idempotently) a `test` -> `main` pull request summarising what's promoting,
     assigned to the operator.
  4. Merge it ONLY if approved — `auto_promote` on (fully autonomous release), or an
     explicit ``approve=True`` (e.g. `anvil promote --approve`). Otherwise the PR waits
     for the operator to click merge.

So `main` never advances on an unreviewed, un-green change: the forge gate guards each
commit onto `test`, and this gate guards the batch onto `main`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args) -> str:
    p = subprocess.run(["git", *args], cwd=str(ROOT),
                       capture_output=True, text=True)
    return (p.stdout or "").strip()


def pending(cfg) -> bool:
    """Cheap work check for the autopilot: is the forge branch ahead of main?
    One local git call — no network, no test gate."""
    src = getattr(cfg, "forge_branch", "test")
    dst = getattr(cfg, "main_branch", "main")
    return bool(_git("log", f"{dst}..{src}", "--oneline").strip())


def _sync_local(dst: str, cfg, logger) -> None:
    """After a remote merge, fast-forward the LOCAL ``dst`` ref to the remote —
    otherwise ``dst..src`` stays non-empty forever and every later promote pass
    would re-run the gate and try to re-open a PR. Best-effort, never raises,
    and the token is never logged."""
    try:
        if _git("rev-parse", "--abbrev-ref", "HEAD") == dst:
            return                       # never rewrite the checked-out branch
        url = _git("remote", "get-url", getattr(cfg, "forge_push_remote", "gitea"))
        token = getattr(cfg, "gitea_token", None)
        if "://" in url and token:
            scheme, rest = url.split("://", 1)
            url = f"{scheme}://{token}@{rest.split('@', 1)[-1]}"
        if url:
            p = subprocess.run(["git", "fetch", url, f"{dst}:{dst}"], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
            if p.returncode == 0:
                logger(f"[promote] local {dst} synced to remote")
                return
        logger(f"[promote] could not sync local {dst} — will re-gate next pass")
    except Exception:
        pass


def _remote_tip(cfg, branch: str) -> str:
    """SHA of the branch on the Gitea remote ('' if unreachable). Token never logged."""
    try:
        url = _git("remote", "get-url", getattr(cfg, "forge_push_remote", "gitea"))
        token = getattr(cfg, "gitea_token", None)
        if "://" in url and token:
            scheme, rest = url.split("://", 1)
            url = f"{scheme}://{token}@{rest.split('@', 1)[-1]}"
        p = subprocess.run(["git", "ls-remote", url, f"refs/heads/{branch}"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        return (p.stdout or "").split()[0] if p.returncode == 0 and p.stdout.strip() else ""
    except Exception:
        return ""


def promote(cfg, logger=print, approve: bool = None) -> str:
    """Open (or advance/merge) the reviewed promotion of the forge branch into main."""
    if os.environ.get("ANVIL_IN_DOCTOR"):
        return "skipped (in doctor)"
    src = getattr(cfg, "forge_branch", "test")
    dst = getattr(cfg, "main_branch", "main")

    log = _git("log", f"{dst}..{src}", "--oneline")
    if not log.strip():
        return f"nothing to promote ({src} not ahead of {dst})"

    from .gitea import GiteaClient, GiteaError
    gitea = GiteaClient(cfg)
    if not gitea.ok:
        return "gitea not configured"

    existing = gitea.find_open_pull(src, dst)
    do_merge = getattr(cfg, "auto_promote", False) if approve is None else approve
    # A PR is already waiting and we're not merging -> don't re-run the gate every cycle.
    if existing and not do_merge:
        return f"promotion PR #{existing.get('number')} open, awaiting operator approval"

    # The gate's subject must be exactly what the merge releases. Three checks:
    # (1) single writer — never gate while a forge/issue cycle is mid-edit;
    # (2) the gate runs on src's tip with a clean tree (not a half-edited checkout);
    # (3) local src == remote src (pushes are best-effort and can have failed —
    #     gating 5 local commits and merging a 2-commit remote PR would release
    #     code the gate never saw).
    from .forge import forge_lock
    with forge_lock() as acquired:
        if not acquired:
            return "skipped: forge busy (will retry next pass)"
        if _git("rev-parse", "--abbrev-ref", "HEAD") != src:
            return f"skipped: HEAD is not {src} — refusing to gate a different checkout"
        if _git("status", "--porcelain").strip():
            return "skipped: working tree dirty — refusing to gate uncommitted changes"
        remote = _remote_tip(cfg, src)
        local = _git("rev-parse", src)
        if not remote:
            return "skipped: remote unreachable — cannot verify what would merge"
        if remote != local:
            logger(f"[promote] local {src} != remote {src} — pushing before gating")
            from .forge import Forge
            try:
                f = Forge(branch=src, push_remote=getattr(cfg, "forge_push_remote", "gitea"),
                          push_token=getattr(cfg, "gitea_token", None))
                f._maybe_push()
            except Exception:
                pass
            if _remote_tip(cfg, src) != local:
                return "skipped: could not sync remote — gate subject != merge object"
        return _gated_promote(cfg, logger, gitea, existing, do_merge, src, dst, log)


def _gate_cache_green(sha: str, ttl: int = 1800) -> bool:
    """Fresh green verdict for exactly this SHA from the forge's gate cache
    (review 2.6): the forge just ran the whole suite on this tip after its
    commit — re-running it minutes later on the identical tree bought nothing."""
    try:
        import json as _json
        d = _json.loads((ROOT / "dev-reports" / "gate-cache.json").read_text("utf-8"))
        rec = d.get(sha) or {}
        import time as _time
        return bool(rec.get("green")) and (_time.time() - float(rec.get("ts", 0))) < ttl
    except Exception:
        return False


def _find_culprit(cfg, logger, src, dst, max_runs: int = 4) -> str:
    """Stage-1 culprit hunt: the batch gate went red and there is more than
    one commit in the batch. Walk oldest-first, skip anything the forge's
    gate-cache already knows is fresh-green, and doctor-check the rest in the
    forge's own worktree (never the live checkout) — capped at ``max_runs``
    doctor runs. Returns a specific culprit message, or '' if inconclusive
    (batch has only one commit, or nothing red was found within the cap)."""
    try:
        shas = [s for s in _git("rev-list", "--reverse", f"{dst}..{src}").splitlines() if s.strip()]
    except Exception:
        shas = []
    if len(shas) <= 1:
        return ""

    from .forge import _gate_cache_get
    work_root = ROOT / ".forge-work"
    runs = 0
    culprit = None
    try:
        for sha in shas:
            cached = _gate_cache_get(sha)
            if cached and cached.get("green"):
                continue
            if runs >= max_runs:
                break
            runs += 1
            subj = _git("log", "-1", "--format=%s", sha)
            try:
                co = subprocess.run(["git", "checkout", "--force", sha],
                                    cwd=str(work_root), capture_output=True, text=True, timeout=60)
                if co.returncode != 0:
                    continue
                gate = subprocess.run([sys.executable, "-m", "anvil", "doctor"],
                                      cwd=str(work_root), capture_output=True, text=True, timeout=900)
            except Exception:
                continue
            if gate.returncode != 0:
                culprit = (sha, subj, (gate.stdout + gate.stderr))
                break
    finally:
        try:
            subprocess.run(["git", "checkout", "--force", src], cwd=str(work_root),
                           capture_output=True, text=True, timeout=60)
        except Exception:
            pass

    if not culprit:
        return ""
    sha, subj, evidence = culprit
    logger(f"[promote] batch red -- first bad commit {sha} {subj}")
    try:
        from . import introspect
        introspect.record("promote-culprit", "promote", f"first bad commit {sha}",
                          evidence=evidence[-800:])
    except Exception:
        pass
    return f"test gate red -- culprit {sha} ({subj})"


def _gated_promote(cfg, logger, gitea, existing, do_merge, src, dst, log) -> str:
    """The gate + PR + merge, entered only under the forge lock with a verified tree."""
    from .gitea import GiteaError
    # About to create or merge -> the WHOLE suite must be green on the forge branch.
    sha = _git("rev-parse", src)
    if _gate_cache_green(sha):
        logger(f"[promote] gate cache hit — suite already green on {sha[:9]}")
        class gate:                      # duck-typed green result
            returncode = 0
            stdout = stderr = ""
    else:
        try:
            gate = subprocess.run([sys.executable, "-m", "anvil", "doctor"],
                                  cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=900)
        except Exception as exc:
            logger(f"[promote] test gate could not run: {exc}")
            return "test gate error"
    if gate.returncode != 0:
        logger(f"[promote] test gate RED on {src} — NOT promoting")
        try:
            from . import introspect
            introspect.record("promote-gate-red", "promote", f"{src}->{dst} blocked",
                              evidence=(gate.stdout + gate.stderr)[-800:])
        except Exception:
            pass
        culprit_msg = _find_culprit(cfg, logger, src, dst)
        if culprit_msg:
            return culprit_msg
        return "test gate not green — not promoting"

    if existing:
        pr = existing
    else:
        stat = _git("diff", f"{dst}..{src}", "--stat")
        body = (f"Promote `{src}` -> `{dst}`. Final test gate is **GREEN**. Every commit "
                f"below already passed the forge's independent reviewer + test gate on "
                f"`{src}`.\n\n**Commits ({len(log.splitlines())}):**\n```\n{log[:2000]}\n```\n"
                f"**Files:**\n```\n{stat[:1500]}\n```\n\n"
                "_Opened by ANVIL's promotion gate. Merging releases these to `main`._")
        try:
            pr = gitea.create_pull(src, dst, f"Promote {src} -> {dst}", body)
            gitea.set_assignees(pr["number"], [getattr(cfg, "issue_operator", "bytesnap")])
            logger(f"[promote] opened promotion PR #{pr.get('number')}")
        except GiteaError as exc:
            return f"could not open promotion PR: {exc}"

    if not do_merge:
        return f"promotion PR #{pr.get('number')} open, awaiting operator approval"
    try:
        gitea.merge_pull(pr["number"])
        logger(f"[promote] merged PR #{pr.get('number')} — {dst} released")
        _sync_local(dst, cfg, logger)    # keep the local ref honest post-merge
        return f"promoted: merged PR #{pr.get('number')} ({len(log.splitlines())} commits)"
    except GiteaError as exc:
        return f"PR #{pr.get('number')} open; merge failed: {exc}"
