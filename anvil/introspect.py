"""Harness self-awareness — notice when the process goes wrong, decide whether it's
the HARNESS's own fault, and file it so the self-dev loop fixes it.

Anything that misfires in Lara's process — a tool call erroring, a self-dev cycle that
can't land a change, a caught exception, bad output a gate rejected — is written to an
*incident log* via ``record()`` (cheap, never raises). In deep sleep, ``triage()``
clusters recurring incidents, asks whether each is a defect in the HARNESS ITSELF (its
own tools / self-dev driver / review gate / code) versus a transient outage or a genuinely
hard task, and files a deduped, rate-limited ``harness-bug`` issue for the harness ones —
assigned to the operator, because self-surgery deserves a second set of eyes.

This turns the thing a human reviewer does by hand ("that failure smells like OUR bug —
log it") into a capability the harness runs on itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCIDENTS = ROOT / "dev-reports" / "incidents.jsonl"       # append-only incident log
_DECIDED = ROOT / "dev-reports" / "incidents-decided.json"  # sigs we've already triaged
HARNESS_LABEL = "harness-bug"
HIGH_LABEL = "priority/high"    # only HIGH-priority harness bugs wait for a human sign-off
_MAX_FILE_PER_RUN = 3            # cap new harness issues filed per triage (anti-flood)
_MAX_CLASSIFY_PER_RUN = 6       # cap model classifications per triage (cost)


def _sig(kind: str, where: str, detail: str) -> str:
    """Stable signature so the same recurring failure dedups. Numbers/hex are
    normalised so '#3' vs '#7' or two different hashes collapse to one signature."""
    norm = re.sub(r"[0-9a-f]{6,}|\d+", "#", str(detail).lower())[:200]
    return hashlib.sha1(f"{kind}|{where}|{norm}".encode("utf-8")).hexdigest()[:12]


def record(kind: str, where: str, detail: str = "", evidence: str = "") -> None:
    """Log ONE process incident. Cheap, never raises, and a no-op inside `anvil doctor`
    (so the test suite can't spam incidents or fork-bomb via triage->forge->doctor)."""
    if os.environ.get("ANVIL_IN_DOCTOR"):
        return
    try:
        from datetime import datetime, timezone
        INCIDENTS.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now(timezone.utc).isoformat(),
               "kind": str(kind), "where": str(where)[:120],
               "detail": str(detail)[:500], "evidence": str(evidence)[:1500],
               "sig": _sig(kind, where, detail)}
        with INCIDENTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass                      # self-awareness must never break the thing it watches


def _load(limit: int = 800) -> list:
    if not INCIDENTS.exists():
        return []
    out = []
    try:
        for line in INCIDENTS.read_text("utf-8", "replace").splitlines()[-limit:]:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _decided() -> set:
    try:
        return set(json.loads(_DECIDED.read_text("utf-8")))
    except Exception:
        return set()


def _mark_decided(sigs) -> None:
    try:
        _DECIDED.parent.mkdir(parents=True, exist_ok=True)
        _DECIDED.write_text(json.dumps(sorted(_decided() | set(sigs))), encoding="utf-8")
    except Exception:
        pass


def pending() -> int:
    """Cheap work check for the autopilot: how many NEW incident signatures
    await triage? Local file reads only — no model, no network."""
    if not INCIDENTS.exists():
        return 0
    decided = _decided()
    return len({i.get("sig") for i in _load() if i.get("sig")} - decided)


_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "harness_bug": {"type": "boolean"},
        "high_priority": {"type": "boolean"},
        "title": {"type": "string"},
        "component": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["harness_bug"],
}


def triage(cfg, logger=print) -> str:
    """Cluster new incidents, classify each as a harness defect or not, and file a
    deduped, rate-limited ``harness-bug`` issue for the confirmed harness ones."""
    if os.environ.get("ANVIL_IN_DOCTOR"):
        return "skipped (in doctor)"
    from .gitea import GiteaClient, GiteaError
    from .router import Router
    gitea = GiteaClient(cfg)
    if not gitea.ok:
        return "gitea not configured"
    incidents = _load()
    if not incidents:
        return "no incidents"

    decided = _decided()
    clusters: dict = {}
    for inc in incidents:
        sig = inc.get("sig")
        if sig and sig not in decided:
            clusters.setdefault(sig, []).append(inc)
    if not clusters:
        return "no new incidents"

    # Dedup against harness issues already open: exact via the sig marker in the body,
    # and SEMANTIC via their titles (shown to the classifier so it won't re-file the
    # same bug worded differently).
    try:
        open_issues = gitea.list_issues(labels=[HARNESS_LABEL], state="open")
    except GiteaError:
        open_issues = []
    open_bodies = "\n".join((i.get("body") or "") for i in open_issues)
    known_titles = [str(i.get("title", "")) for i in open_issues if i.get("title")]

    router = Router(cfg, plane="selfdev")
    rung = cfg.rung_by_name("cloud-open")
    if rung is None:
        rung = 0
    operator = getattr(cfg, "issue_operator", "bytesnap")
    actor = getattr(cfg, "issue_actor", "lara")
    filed = 0
    classified = 0
    # Most-recurring signatures first — the loudest problems get looked at first.
    for sig, group in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        if filed >= _MAX_FILE_PER_RUN or classified >= _MAX_CLASSIFY_PER_RUN:
            break
        if f"incident-sig:{sig}" in open_bodies:      # already filed & still open
            _mark_decided([sig])
            continue
        s = group[-1]
        classified += 1
        dup_note = ("\n\nAlready-open harness issues (set harness_bug=false if this is "
                    "ALREADY covered by one of them — do NOT file a duplicate):\n"
                    + "\n".join(f"- {t}" for t in known_titles)) if known_titles else ""
        try:
            r = router.complete(
                [{"role": "user",
                  "content": f"A recurring process incident (seen {len(group)}x):\n"
                  f"kind = {s.get('kind')}\nwhere = {s.get('where')}\n"
                  f"detail = {s.get('detail')}\nevidence = {s.get('evidence','')[:900]}\n\n"
                  "Is this a DEFECT IN THE HARNESS ITSELF — its own tools, self-dev driver, "
                  "review gate, or code we can fix — as opposed to a transient outage, bad "
                  "user input, or a genuinely hard task? If harness_bug, give a crisp issue "
                  "title, the likely component, and a short summary. Also set high_priority: "
                  "true ONLY when the defect is severe — it breaks Lara's core loop, corrupts "
                  "state or data, is a security/safety hole, or repeatedly derails many flows. "
                  "A cosmetic, single-flow, or easily-recoverable bug is NOT high_priority."
                  + dup_note + "\n\nReturn JSON."}],
                system="You are ANVIL diagnosing your OWN failures. Be conservative: "
                       "harness_bug=true ONLY when the evidence points at a real defect in "
                       "the harness's own code/tools/process — never for an outage, the "
                       "environment, the difficulty of the task, or something ALREADY "
                       "covered by an open harness issue. Reserve high_priority=true for "
                       "genuinely severe defects; routine bugs are normal priority.",
                schema=_TRIAGE_SCHEMA, min_rung=rung, max_tokens=500)
            verdict = json.loads(r.completion.text)
        except Exception:
            continue                                   # leave undecided; retry next run
        _mark_decided([sig])                           # decided either way -> no re-triage churn
        if not verdict.get("harness_bug"):
            continue
        high = bool(verdict.get("high_priority"))
        title = (verdict.get("title")
                 or f"harness: {s.get('kind')} in {s.get('where')}").strip()[:120]
        # HIGH-priority harness bugs wait for a human sign-off (assigned to the operator);
        # routine ones go straight to Lara to work autonomously. The _SELF_CRITICAL lock
        # still keeps her off the machinery files regardless of priority.
        gate_note = ("_Filed by ANVIL's self-awareness triage. **High priority** — assigned "
                     "to the operator to sanity-check before Lara works it, since it touches "
                     "the harness's own code._"
                     if high else
                     "_Filed by ANVIL's self-awareness triage. Routine priority — assigned to "
                     "Lara to work autonomously (the self-critical machinery stays human-only)._")
        body = (f"{verdict.get('summary','')}\n\n"
                f"**Component:** {verdict.get('component','?')}\n"
                f"**Seen:** {len(group)}x. Latest signal:\n```\n"
                f"kind={s.get('kind')} where={s.get('where')}\n"
                f"detail={s.get('detail')}\nevidence={s.get('evidence','')[:800]}\n```\n\n"
                f"{gate_note}\n"
                f"<!-- incident-sig:{sig} -->")
        labels = ["selfdev", HARNESS_LABEL] + ([HIGH_LABEL] if high else [])
        assignee = operator if high else actor
        try:
            iss = gitea.create_issue(title, body, labels=labels)
            gitea.set_assignees(iss["number"], [assignee])
            logger(f"[introspect] filed {HARNESS_LABEL}"
                   f"{' (HIGH)' if high else ''} #{iss.get('number')}"
                   f" -> {assignee}: {title}")
            filed += 1
            known_titles.append(title)     # so a later cluster this run sees it as a dup
        except GiteaError:
            continue
    return f"triaged {classified}, filed {filed} harness-bug issue(s)"
