"""In-harness self-development — ANVIL edits its own code on its own model ladder.

This is the coding engine behind sleep-time self-improvement. The classic Forge
(``forge.py``) supplies the git seatbelt: a dedicated ``forge-auto`` branch, a
test gate (commit-on-green / revert-on-red), the anti-cheat (the test count can't
drop), and it never pushes or touches ``main``. Here we plug into it:

* a **driver** that makes the change using ANVIL's OWN model ladder — local model
  first, escalating to Ollama Cloud — with no external coding CLI, so ANVIL
  genuinely develops itself on its own hardware; and
* a **reviewer** that independently approves the diff (on a higher rung than the
  driver, so the critic isn't the author) before anything is kept.

Together with Forge that's the council's *build + consensus + commit* loop, brought
inside the harness so it can run during sleep.

Safety, in layers: edits are confined to an allowlist of harness files
(``anvil/*.py`` and docs) — secrets and local state can never be touched; a
non-matching edit is a no-op; the reviewer can veto; the test gate reverts red
changes; and the worst case of any cycle is a no-op, guaranteed by git.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# What the self-dev driver is allowed to edit — everything else is refused.
# --------------------------------------------------------------------------- #
_DENY_NAMES = {".env", "persona.json", "anvil.toml", "ledger.jsonl"}
_DENY_DIRS = ("memory/", "jobs/", ".git/", ".venv/", "test-reports/",
              "dev-reports/", "forge/queue/", "forge/builds/", "forge/reviews/",
              "workspace/", "__pycache__/")
# The MACHINERY of self-modification itself — the git seatbelt (forge.py), the coding
# driver (selfdev.py), and the self-awareness layer (introspect.py). Autonomous edits
# here could disable the very safety that reverts bad changes, or blind the harness to
# its own failures, so they are HUMAN-review-only: the driver refuses them, and a
# harness-bug that needs one escalates to a person (see harness-bug #10, fixed by hand).
_SELF_CRITICAL = {"anvil/forge.py", "anvil/selfdev.py", "anvil/introspect.py"}


def _allowed(rel: str) -> bool:
    """True only for harness source/docs; never secrets, state, config, or the
    self-modification machinery itself."""
    rel = rel.replace("\\", "/").lstrip("./")
    if not rel or ".." in rel:
        return False
    if rel in _SELF_CRITICAL:
        return False
    if Path(rel).name in _DENY_NAMES:
        return False
    if rel.startswith(".env") or rel.endswith((".toml", ".env")):
        return False
    if any(rel.startswith(d) for d in _DENY_DIRS):
        return False
    if rel.startswith("anvil/") and rel.endswith(".py"):
        return True
    if rel == "anvil/hearth.html":      # the Hearthlight family UI — hers to shape
        return True
    if "/" not in rel and rel.endswith(".md"):          # top-level docs
        return True
    if rel.startswith(("forge/", "docs/")) and rel.endswith(".md"):
        return True
    return False


PLAN_SCHEMA = {
    "type": "object",
    "properties": {"files": {"type": "array", "items": {"type": "string"}},
                   "intent": {"type": "string"}},
    "required": ["files", "intent"],
}
# A LIST of precise edits, possibly across several files — so one cycle can add a
# function AND its call, or land code in one file AND a test in another. This is the
# capability the single-edit driver lacked (why issue #6 was unreachable for Lara).
EDITS_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"file": {"type": "string"},
                               "find": {"type": "string"},
                               "replace": {"type": "string"}},
                "required": ["file", "find", "replace"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["edits"],
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"approve": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["approve"],
}


def _list_source() -> str:
    lines = []
    for p in sorted((ROOT / "anvil").glob("*.py")):
        doc = ""
        try:
            parts = p.read_text("utf-8", "replace").split('"""')
            if len(parts) > 1 and parts[1].strip():
                doc = parts[1].strip().splitlines()[0]
        except OSError:
            pass
        lines.append(f"anvil/{p.name}: {doc[:80]}")
    # The family UI is editable source too — without this line the planner never
    # knew it existed, so every UI issue refused with 'no editable file chosen'.
    if (ROOT / "anvil" / "hearth.html").exists():
        lines.append("anvil/hearth.html: the Hearthlight family UI (HTML+CSS+JS, "
                     "served at '/'; any change here MUST also bump UI_BUILD in "
                     "anvil/server.py so installed apps self-update)")
    for p in sorted((ROOT / "docs").glob("*.md")):
        lines.append(f"docs/{p.name}: project design/plan document")
    return "\n".join(lines)


def _loads(text: str) -> dict:
    from .mind import _loads as _ml
    return _ml(text)


# --------------------------------------------------------------------------- #
# The driver: ANVIL writes a code change with its own ladder (local -> cloud)
# --------------------------------------------------------------------------- #
_FILE_CAP = 30000            # chars of each chosen file shown to the model


def _file_view(body: str, intent: str, cap: int) -> str:
    """What the editor model SEES of one file. Small files come whole. A huge
    file (server.py ~150K, selftest.py ~200K) used to show only its first
    ``cap`` chars — so edits that had to anchor mid-file (wiring an endpoint
    into the dispatch chain, appending a test) could never quote verbatim
    find-text and every attempt missed (#117 wired imports but no endpoints:
    the dispatch region was simply invisible). Instead: keep the head, then
    add windows around lines matching the INTENT's keywords, with omission
    markers between regions. Verbatim lines stay verbatim inside windows."""
    if len(body) <= cap:
        return body
    import re as _re
    words = {w.lower() for w in _re.findall(r"[A-Za-z_][\w/]{3,}", intent)}
    lines = body.splitlines()
    keep = set(range(min(60, len(lines))))          # always the head
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(w in low for w in words):
            keep.update(range(max(0, i - 30), min(len(lines), i + 31)))
    out: List[str] = []
    total = 0
    prev = -1
    for i in sorted(keep):
        if i != prev + 1:
            out.append("... <middle of file omitted — do NOT anchor edits "
                       "in unseen regions> ...")
        out.append(lines[i])
        total += len(lines[i]) + 1
        prev = i
        if total > cap:
            out.append("... <view truncated at cap> ...")
            break
    if prev < len(lines) - 1:
        out.append("... <rest of file omitted> ...")
    return "\n".join(out)


def build_local_driver(cfg, logger: Callable[[str], None] = print,
                       rung: int = 0, root: Path = None) -> Callable[[str], str]:
    """Driver that writes a code change at a given ladder ``rung``. It plans across a
    FEW files and returns a LIST of precise find/replace edits, so one cycle can make a
    multi-part change (new function + its call, or code + its test) — not just a single
    edit. The forge's test gate + reviewer still guard every result.

    ``root`` is where the edits land — the forge's surgery WORKTREE, never the live
    checkout (review 1.7)."""
    from .router import Router
    router = Router(cfg, plane="selfdev")
    root = Path(root) if root else ROOT

    def _complete_json(messages, system, schema, max_tokens, want_key, tries=3):
        """A schema call that RETRIES on an empty/unparseable completion. Cloud
        rungs under strict-JSON mode intermittently return an empty string
        (measured ~1-in-3 on the plan call) — a single-shot planner then refuses
        'planned: (none)' and burns a whole issue attempt on pure model flake.
        Retry until we get a parse with the required key, or give up cleanly."""
        last = {}
        for _ in range(max(1, tries)):
            try:
                r = router.complete(messages, system=system, schema=schema,
                                    min_rung=rung, max_tokens=max_tokens)
            except Exception as exc:
                last = {"__error__": str(exc)}
                continue
            d = _loads(r.completion.text)
            if not isinstance(d, dict):
                # the model sometimes returns a bare JSON string/list — that
                # crashed the whole attempt ("'str' object has no attribute
                # 'get'") instead of counting as one more flaky try
                last = {"__error__": f"non-dict completion ({type(d).__name__})"}
                continue
            if d.get(want_key):
                return d
            last = d
        return last

    def driver(prompt: str) -> str:
        # A) Plan: pick the FEWEST files that fully do it + a concrete intent.
        plan = _complete_json(
            [{"role": "user",
              "content": prompt + "\n\nHarness source files:\n" + _list_source() +
              "\n\nPick the FEWEST files (1-4) that FULLY accomplish one small, safe, "
              "concrete change — include a test file when the change should be "
              "tested — and state the intent. You may name ONE file that does "
              "not exist yet (a new anvil/*.py module or docs/*.md) when the "
              "change genuinely needs a new file; everything else must be an "
              "existing file from the list."}],
            "You are ANVIL improving your own harness. Choose the minimal set "
            "of files that together accomplish one small, safe, concrete change.",
            PLAN_SCHEMA, 400, "files")
        if plan.get("__error__"):
            return f"plan-error: {plan['__error__']}"
        intent = (plan.get("intent") or "").strip()
        rels: List[str] = []
        new_files: set = set()
        for f in (plan.get("files") or [])[:4]:
            rel = str(f).replace("\\", "/").strip().lstrip("./")
            if not _allowed(rel) or rel in rels:
                continue
            if (root / rel).exists():
                rels.append(rel)
            elif (rel.endswith((".py", ".md")) and not new_files
                  and (root / rel).parent.is_dir()):
                # A feature may START a file: ONE brand-new module/doc per
                # cycle, in an existing directory. (Issue #95's parser died
                # here — the plan chose anvil/imports.py and the .exists()
                # check vetoed creation outright, three attempts running.)
                rels.append(rel)
                new_files.add(rel)
        if not rels:
            # Say WHICH files the plan wanted — this string reaches the issue
            # escalation verbatim, so a structural block (file not in the
            # allowlist) reads as exactly that, not as a mystery to narrate over.
            wanted = ", ".join(str(f) for f in (plan.get("files") or [])[:4]) or "(none)"
            logger(f"[selfdev] refused: no editable file chosen (planned: {wanted})")
            return (f"refused: the plan chose files outside the allowlist "
                    f"(planned: {wanted}); editable = anvil/*.py, "
                    "anvil/hearth.html, docs/*.md, forge/*.md, top-level *.md "
                    "(ONE brand-new .py/.md per cycle may be created) — "
                    "never forge.py/selfdev.py/introspect.py")
        originals: dict = {}
        blobs: List[str] = []
        for rel in rels:
            if rel in new_files:
                originals[rel] = ""
                blobs.append(f"===== FILE {rel} (NEW — does not exist yet; "
                             "create it with ONE edit: find=\"\" and the full "
                             "file body as replace) =====\n")
                continue
            try:
                originals[rel] = (root / rel).read_text("utf-8", "replace")
            except OSError:
                continue
            # hearth.html is ~85KB of HTML+CSS+JS; the 30KB code cap would show
            # the model only the stylesheet and its edits would miss every time.
            cap_n = 100_000 if rel.endswith(".html") else _FILE_CAP
            blobs.append(f"===== FILE {rel} =====\n"
                         + _file_view(originals[rel], intent + " " + prompt[:2000],
                                      cap_n))
        if not originals:
            return "read-error: could not read any chosen file"

        # B) Produce a LIST of precise find/replace edits across those files.
        # Same retry-on-empty guard as the plan step — the edits call is the
        # same strict-JSON cloud path and flakes the same way.
        data = _complete_json(
            [{"role": "user",
              "content": f"INTENT: {intent}\n\n" + "\n\n".join(blobs) +
              "\n\nReturn JSON {\"edits\": [{\"file\", \"find\", \"replace\"}, ...]}. "
              "Each 'find' is an EXACT verbatim substring of THAT file (long enough "
              "to be unique); 'replace' is the new text. Use MULTIPLE edits when "
              "needed — e.g. add a function AND the line that calls it, or land code "
              "in one file AND a test in another. Keep encoding='utf-8' on file I/O; "
              "never weaken or delete an existing test — add tests. Make the SMALLEST "
              "set of exact edits that FULLY does the intent."}],
            "You output a JSON list of precise find/replace edits across one "
            "or more files. Every 'find' MUST appear verbatim in the file it "
            "names. Prefer several small exact edits over one giant rewrite.",
            EDITS_SCHEMA, 16000, "edits")   # a NEW module + its test as JSON
                                            # edits easily exceeds 3000 tokens —
                                            # truncated tool_use parsed to {} and
                                            # every retry hit the same wall
        if data.get("__error__"):
            return f"edit-error: {data['__error__']}"
        edits = data.get("edits") or []
        contents = dict(originals)
        matched = 0
        misses: List[str] = []
        for e in edits:
            if not isinstance(e, dict):      # a bare string in the edits list
                misses.append(f"non-dict edit entry ({str(e)[:40]!r})")
                continue
            rel = str(e.get("file", "")).replace("\\", "/").strip().lstrip("./")
            find = e.get("find") or ""
            replace = e.get("replace")
            if rel not in contents or not _allowed(rel) \
                    or replace is None or find == replace:
                misses.append(f"{rel or '?'}: find-text not verbatim "
                              f"({(find or '')[:60]!r})")
                continue
            if not find:
                # empty find = "write the whole file" — ONLY valid for a
                # still-empty NEW file, so it can never clobber real source
                if contents[rel] == "" and rel in new_files:
                    contents[rel] = replace
                    matched += 1
                else:
                    misses.append(f"{rel}: empty find is only valid for a "
                                  "NEW empty file")
                continue
            if find not in contents[rel]:
                misses.append(f"{rel or '?'}: find-text not verbatim "
                              f"({(find or '')[:60]!r})")
                continue
            contents[rel] = contents[rel].replace(find, replace, 1)  # compound same-file edits
            matched += 1
        # ALL-OR-NOTHING (review 1.6): a half-applied plan — code landed but its
        # test edit missed — is an untested change wearing a complete one's face,
        # and the gate can't tell them apart. If ANY edit misses, apply none; the
        # precise miss report becomes the retry signal.
        if misses:
            return ("no-op: " + ("all" if not matched else
                    f"{len(misses)} of {len(edits)}") + " edits failed to match — "
                    "nothing applied (a partial change would dodge its own tests). "
                    "Misses: " + "; ".join(misses[:4]))

        applied: List[str] = []
        for rel, text in contents.items():
            if text != originals[rel]:
                try:
                    (root / rel).write_text(text, encoding="utf-8")
                    applied.append(rel)
                except OSError as exc:
                    return f"write-error: {exc}"
        if not applied:
            return f"no-op: no edit matched verbatim ({len(edits)} proposed)"
        logger(f"[selfdev] edited {', '.join(applied)} "
               f"({matched} edit(s)): {str(data.get('reason', ''))[:70]}")
        return f"edited {', '.join(applied)}"

    return driver


# --------------------------------------------------------------------------- #
# The reviewer: an independent (higher-rung) approval of the diff
# --------------------------------------------------------------------------- #
def build_local_reviewer(cfg, min_rung: int = 1,
                         context: str = "") -> Callable[[str], Tuple[bool, str]]:
    """Independent (higher-rung) reviewer. When ``context`` is given (the ticket/issue
    the change is meant to satisfy) the reviewer judges the diff AGAINST that intent —
    'does this correctly and safely do what was asked' — instead of judging a diff
    blind, which makes it reject correct-but-unexplained changes."""
    from .router import Router
    router = Router(cfg, plane="selfdev")
    goal = ("\n\nThe change is meant to accomplish:\n" + context.strip()[:1500] +
            "\n\nApprove if the diff CORRECTLY and safely does that." if context.strip()
            else "")

    def reviewer(diff: str) -> Tuple[bool, str]:
        if not diff.strip():
            return False, "empty diff"
        # MECHANICAL guard, before any model judgment: a hearth.html change
        # without a UI_BUILD bump ships an update installed apps never fetch
        # (hearth is cached at import + clients poll /api/version). The model
        # reviewer missed this on #64 — a deterministic rule shouldn't be
        # entrusted to judgment at all.
        if "anvil/hearth.html" in diff and "UI_BUILD" not in diff:
            return False, ("hearth.html changed but UI_BUILD in anvil/server.py "
                           "was not bumped — installed apps would never fetch "
                           "this update. Add the bump and resubmit.")
        # A diff too large to review in full is rejected outright (truncate-and-
        # approve was the old behavior — approving a sprawl on its first N chars,
        # precisely where skepticism matters). The cap is sized for the CLAUDE
        # reviewer era: a legitimate new-module-plus-test cycle runs 8-12K chars
        # and Sonnet reviews that in full — the old 8000 rejected GREEN, correct
        # work (issue #109: 8224 and 9707 both vetoed unread).
        if len(diff) > 24000:
            return False, f"diff too large to review ({len(diff)} chars > 24000)"
        try:
            rr = router.complete(
                [{"role": "user",
                  "content": "Review this diff to ANVIL's own harness. Approve if it is "
                  "a correct, small, safe change that does not weaken or delete tests, "
                  "touch secrets, or break style/encoding — a small correct docs/help "
                  "or comment fix is fine." + goal + "\n\nDiff:\n\n" + diff}],
                system="You are an independent code reviewer, not the author. Be "
                       "skeptical of unsafe, sprawling, or incorrect changes, but do "
                       "APPROVE a change that correctly and safely does what was asked.",
                schema=REVIEW_SCHEMA, min_rung=min_rung, max_tokens=300)
        except Exception as exc:
            return False, f"reviewer-error: {exc}"
        d = _loads(rr.completion.text)
        if "approve" in d:
            return bool(d.get("approve")), str(d.get("reason", ""))[:200]
        # Model ignored the schema and returned prose. This gate stands between
        # model-written code and (with auto_promote) an unattended release — an
        # unparseable verdict is NOT an approval. Re-ask once with the schema
        # restated; still unparseable -> reject.
        try:
            rr2 = router.complete(
                [{"role": "user",
                  "content": "Your previous review reply was not valid JSON. Answer "
                  "ONLY with JSON matching {\"approve\": true|false, \"reason\": \"...\"} "
                  "for this diff:\n\n" + diff}],
                system="You are an independent code reviewer. JSON only.",
                schema=REVIEW_SCHEMA, min_rung=min_rung, max_tokens=200)
            d2 = _loads(rr2.completion.text)
            if "approve" in d2:
                return bool(d2.get("approve")), str(d2.get("reason", ""))[:200]
        except Exception:
            pass
        return False, "reviewer verdict unparseable (twice) — fail-to-reject"

    return reviewer


# --------------------------------------------------------------------------- #
# One guarded self-dev cycle (what the sleep cycle calls)
# --------------------------------------------------------------------------- #
_STATE = ROOT / "dev-reports" / "selfdev-state.json"
FORGE_QUEUE = ROOT / "forge" / "queue"


def _read_ticket(path: Path) -> Tuple[dict, str]:
    text = path.read_text("utf-8", "replace")
    fm: dict = {}
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for ln in lines[1:]:
            if ln.strip() == "---":
                break
            if ":" in ln:
                k, v = ln.split(":", 1)
                fm[k.strip().lower()] = v.strip()
    return fm, text


def _set_status(path: Path, text: str, status: str) -> None:
    import re
    new = re.sub(r"(?m)^status:.*$", f"status: {status}", text, count=1)
    try:
        path.write_text(new, encoding="utf-8")
    except OSError:
        pass


def claim_ticket() -> Tuple[Optional[Path], Optional[str]]:
    """Claim the top queued/ready council ticket for internal building, marking it
    ``status: building-internal`` so the external council skips it (WIP=1). The
    internal self-dev and the external council thus share ONE backlog — queued
    work is never orphaned when the council is stopped. Returns (path, body)."""
    if not FORGE_QUEUE.exists():
        return None, None
    cands = []
    for p in sorted(FORGE_QUEUE.glob("TASK-*.md")):
        try:
            fm, text = _read_ticket(p)
        except OSError:
            continue
        if fm.get("status", "").lower() in ("queued", "ready"):
            try:
                pri = int(fm.get("priority", "3"))
            except ValueError:
                pri = 3
            cands.append((pri, fm.get("created", ""), p.name, p, text))
    if not cands:
        return None, None
    cands.sort(key=lambda t: (t[0], t[1], t[2]))
    _, _, _, path, text = cands[0]
    _set_status(path, text, "building-internal")
    return path, text


def resolve_ticket(path: Optional[Path], kept: bool) -> None:
    """Mark a claimed ticket done (kept) or return it to the queue (so the local
    model's misses fall back to the external cloud council)."""
    if not path or not path.exists():
        return
    _, text = _read_ticket(path)
    _set_status(path, text, "done" if kept else "queued")


def _daily_count() -> int:
    try:
        d = json.loads(_STATE.read_text("utf-8"))
        return int(d.get("count", 0)) if d.get("date") == date.today().isoformat() else 0
    except (OSError, ValueError):
        return 0


def _bump_daily() -> None:
    try:
        _STATE.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"date": date.today().isoformat(),
                              "count": _daily_count() + 1})
        _STATE.write_text(payload, encoding="utf-8")
    except OSError:
        pass


def coding_rungs(cfg) -> list:
    """The rung ladder the coder tries, in order. With ``selfdev_cloud_first`` code on
    a RELIABLE structured-output frontier rung (cloud-open / glm) — NOT the very top
    reasoning rung (deepseek), whose think+stream+strict-JSON combo returns truncated
    or empty find/replace edits, so nothing ever lands. The top rung stays as a second
    attempt. Without cloud-first: local first, cloud second."""
    top = len(cfg.ladder) - 1
    if getattr(cfg, "selfdev_cloud_first", False) and top > 0:
        co = cfg.rung_by_name("cloud-open")
        if co is not None:
            # Only cloud-open (glm). Do NOT escalate to the top reasoning rung
            # (deepseek): its think+stream+strict-JSON combo returns empty/truncated
            # edits ("refused target: ''"), which only wastes a rung and masks the
            # real reason. issue-work already retries the whole cycle up to 3x.
            return [co]
        return [top]
    return [0] if top <= 0 else [0, top]


def pick_review_rung(cfg) -> int:
    """Rung the independent reviewer runs on — a DIFFERENT frontier model than the
    coder, so review is never self-approval. The coder rides cloud-open (glm), so
    review rides cloud-heavy (kimi): different model family, correlated blind spots
    broken. Review is a plain JSON judgement (no find/replace emission), so the
    structured-edit weakness that rules the reasoning rungs out for CODING does not
    apply here."""
    if getattr(cfg, "selfdev_cloud_first", False):
        coder = cfg.rung_by_name("cloud-open")
        for name in ("cloud-heavy", "cloud-logic"):
            idx = cfg.rung_by_name(name)
            if idx is not None and idx != coder:
                return idx
        if coder is not None:            # single-cloud-rung install: better than local
            return coder
    return 1


def drive_forge(f, cfg, logger: Callable[[str], None] = print, n: int = 1):
    """Run the escalation ladder against an already-built Forge ``f`` (work_item set):
    try each coding rung in turn, stop as soon as a change lands. Shared by self-dev
    and issue-work so both get cloud-first coding + escalation, not a weak rung-0 try."""
    r = None
    rungs = coding_rungs(cfg)
    for attempt, rung in enumerate(rungs, 1):
        f.driver = build_local_driver(cfg, logger, rung=rung,
                                      root=getattr(f, "work_root", None))
        where = cfg.ladder[rung].name if rung < len(cfg.ladder) else f"rung{rung}"
        logger(f"[selfdev] attempt {attempt}/{len(rungs)} on {where}")
        r = f.cycle(n)
        if r.kept:
            break
        # Iterate-on-failure (review 1.6): the next attempt sees what the last
        # one changed and why the gate rejected it, instead of re-deriving the
        # same mistake blind from an identical prompt.
        if getattr(r, "fail_diff", ""):
            f.retry_context = (f"REASON REVERTED: {r.reason}\n"
                               f"WHAT THE TESTS SAID:\n{r.fail_out}\n"
                               f"THE DIFF THAT FAILED:\n{r.fail_diff}")[:4000]
        if attempt < len(rungs):
            logger(f"[selfdev] attempt did not land ({r.reason}) — escalating")
    # retry_context deliberately stays on f: issue-work retries the SAME work
    # item with the same Forge, and attempt 2 must see attempt 1's failure. A
    # fresh Forge (new cycle / new issue) starts clean by construction.
    return r


def run_one_cycle(cfg, logger: Callable[[str], None] = print) -> Dict[str, object]:
    """Run a single git-guarded self-dev cycle, LOCAL FIRST then escalating to the
    Ollama Cloud rung for help if the local attempt doesn't land a kept change.

    Each attempt is fully git-guarded (build -> test -> independent review ->
    commit-or-revert), so a failed local attempt is reverted before the cloud
    attempt starts from a clean tree. Skips (never disrupts the operator) if the
    working tree is dirty on a branch other than ``forge-auto``, or if the daily
    cap is reached. A ticket only leaves the queue as ``done`` when a change is
    actually kept; otherwise it returns to ``queued`` for a later, or attended,
    attempt.
    """
    from .forge import Forge

    # Re-entrancy guard: the forge's test gate shells out to `anvil doctor`; if we're
    # ALREADY inside one, running the forge would spawn another doctor -> fork bomb.
    import os
    if os.environ.get("ANVIL_IN_DOCTOR"):
        return {"ran": False, "reason": "skipped: inside anvil doctor"}
    cap = int(getattr(cfg, "self_dev_daily_cap", 3))
    if _daily_count() >= cap:
        return {"ran": False, "reason": f"daily cap {cap} reached"}

    # Single-writer: only one forge/self-dev/issue-work run may touch the tree.
    from .forge import forge_lock
    with forge_lock() as _lock_ok:
        if not _lock_ok:
            return {"ran": False, "reason": "busy (another forge run in progress)"}
        # Independent reviewer. With cloud-first, review on a DIFFERENT frontier
        # model than the coder (e.g. glm reviews kimi's diff) — diverse perspectives
        # catch what a model reviewing its own style would wave through.
        push_remote = (getattr(cfg, "forge_push_remote", "gitea")
                       if getattr(cfg, "forge_push", False) else None)
        f = Forge(branch=getattr(cfg, "forge_branch", "test"),
                  reviewer=build_local_reviewer(cfg, min_rung=pick_review_rung(cfg)),
                  logger=logger,
                  push_remote=push_remote, push_token=getattr(cfg, "gitea_token", None))
        # Don't hijack active human work: NEVER run on a dirty tree, even on the forge
        # branch. The operator now shares `test` with the forge, and ensure_repo/revert
        # would swallow their uncommitted edits. Wait until the tree is clean.
        _, branch = f._git("rev-parse", "--abbrev-ref", "HEAD")
        if f._dirty():
            return {"ran": False, "reason": f"working tree dirty on '{branch.strip()}'"}

        f.ensure_repo()
        ticket_path, ticket_text = claim_ticket()
        if ticket_text:
            f.work_item = ticket_text[:2500]
            logger(f"[selfdev] claimed {ticket_path.name}")

        # Escalation ladder (cloud-first when configured) — shared with issue-work.
        n = _daily_count() + 1
        r = drive_forge(f, cfg, logger, n=n)
        _bump_daily()
        if ticket_path:
            resolve_ticket(ticket_path, r.kept)
        logger(f"[selfdev] cycle: {'KEPT' if r.kept else 'reverted'} ({r.reason})")
        return {"ran": True, "kept": r.kept, "reason": r.reason,
                "ticket": ticket_path.name if ticket_path else None}
