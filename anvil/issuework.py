"""Lara works the Gitea issue queue — with judgment, in the open.

For each open work-labelled issue she:
  1. **Checks for blockers** — if this issue can't be done until another is fixed
     first, she notes the dependency and works the BLOCKER first.
  2. **Checks she has enough** to confidently trace/fix it. If not, she asks the
     SPECIFIC questions she needs (in her own voice) and hands it to the creator.
  3. **Sanity-checks the idea** with a small council (technical / value / cost). If
     the idea itself is unsound she pushes back — with reasons — to the creator, and
     opens a rebuttal window: if they reply in time she gauges whether the rebuttal
     truly addresses the concerns, and if so takes it back to the council for a second
     hearing. A second denial closes the issue; a reversal goes to the operator; no
     rebuttal within the window lets the decision stand (closed).
  4. When it's clear and sound, she **works it on `test`** via the git-guarded forge,
     retrying up to 3 times with her full toolset. On success she posts ONE
     substantive update showing what she did; after 3 real failures she assigns the
     operator and explains what she tried, why it failed, and what she needs.

Signalling is by ASSIGNEE, not label churn — the assignee IS "whose turn is it":
  * assigned to Lara      -> she's on it
  * assigned to creator   -> she needs their answer (a question or a push-back)
  * assigned to operator  -> she's stuck after 3 tries and needs help
The one state that lives on a label is ``on-test`` — a fix landed on `test`, pending
the reviewed promotion to `main` (that's what Phase 3 keys off of). See docs/cicd.md.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import List, Optional

from .gitea import GiteaClient, GiteaError
from .router import Router

DONE_LABEL = "on-test"          # a fix landed on `test`, pending promotion to `main`
HIGH_LABEL = "priority/high"    # a HIGH-priority harness bug — needs a human sign-off first
MAX_WORK_ATTEMPTS = 3           # try this hard before asking a human for help

# Invisible markers (HTML comments) that let Lara reconstruct the council process
# across runs from her own comment history — no state labels needed.
COUNCIL_MARK = "<!-- lara:council-declined -->"   # a first-round council push-back
ESCALATED_MARK = "<!-- lara:council-approved-2 -->"  # council reversed; operator has it

_CLARITY_SCHEMA = {
    "type": "object",
    "properties": {
        "confident": {"type": "boolean"},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["confident"],
}
_VOTE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["ok", "reason"],
}
_DEP_SCHEMA = {
    "type": "object",
    "properties": {"blocked_by": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["blocked_by"],
}
_GAUGE_SCHEMA = {
    "type": "object",
    "properties": {"plausible": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["plausible"],
}

# Independent lenses — a majority "not ok" is a push-back on the IDEA.
COUNCIL_LENSES = [
    ("technical", "Is this technically sound and SAFE to implement in this codebase? "
                  "Consider correctness, side effects, and whether it could break things."),
    ("value", "Is this genuinely worth doing for a local-first family household "
              "assistant? Does it serve the users, or is it scope-creep / a bad idea?"),
    ("cost", "Is the effort and risk proportionate to the benefit? Would it add "
             "fragility or maintenance burden out of line with the payoff?"),
]

# How Lara should write a comment: real, intentional, and the RIGHT length — more
# words only when they add substance, never padding. Between a chat line and an email.
_VOICE = (
    "You are Lara, ANVIL's resident engineer, writing a Gitea issue comment in your "
    "OWN voice — a real, thoughtful collaborator, not a bot. Be specific and honest. "
    "Length sits between an instant message and an email: SAY everything that needs to "
    "be said, asked, or clarified — being brief is only right when there is genuinely "
    "nothing more to add. Never pad for length, but never clip a comment short at the "
    "cost of leaving something unsaid; every sentence earns its place. Use Markdown "
    "lightly. Write ONLY the comment body and sign off '— Lara'.")


class IssueWorker:
    def __init__(self, cfg, logger=print):
        self.cfg = cfg
        self.log = logger
        self.gitea = GiteaClient(cfg)
        self.router = Router(cfg, plane="selfdev")
        self.label = getattr(cfg, "issue_work_label", "selfdev")
        self.actor = getattr(cfg, "issue_actor", "lara")       # Lara's own login
        self.operator = getattr(cfg, "issue_operator", "bytesnap")   # human to escalate to
        # Judge on a strong (cloud) rung when available — assessment quality matters.
        self._rung = cfg.rung_by_name("cloud-open")
        if self._rung is None:
            self._rung = 0

    # -- model helpers --------------------------------------------------- #
    @staticmethod
    def _json_loose(text: str) -> dict:
        """Parse a decision even when the model ignored the JSON schema (cloud
        rungs do): strip code fences and surrounding prose, then take the first
        balanced {...} block. Raises ValueError when there's nothing to parse."""
        t = (text or "").strip()
        if not t:
            raise ValueError("empty response")
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        i = t.find("{")
        while i != -1:
            depth = 0
            for j in range(i, len(t)):
                if t[j] == "{":
                    depth += 1
                elif t[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[i:j + 1])
                        except json.JSONDecodeError:
                            break
            i = t.find("{", i + 1)
        raise ValueError("no JSON object in response")

    def _decide(self, system: str, user: str, schema: dict) -> dict:
        """One structured decision. Cloud models routinely ignore the schema=
        parameter (empty text, prose around the JSON, fenced blocks), so parse
        loosely and, if that still fails, retry ONCE with the schema restated
        in-band — only a failure of BOTH is a process incident worth recording."""
        last_exc: Exception = ValueError("no attempt")
        for attempt in (0, 1):
            try:
                u = user
                if attempt:
                    u += ("\n\nYour previous reply was not parseable. Respond with "
                          "ONLY a JSON object matching this schema — no prose, no "
                          "code fences:\n" + json.dumps(schema))
                r = self.router.complete([{"role": "user", "content": u}],
                                         system=system, schema=schema,
                                         min_rung=self._rung, max_tokens=700)
                return self._json_loose(r.completion.text)
            except Exception as exc:
                last_exc = exc
        # A decision (clarity / council / dependency) that can't be computed is a
        # process incident — a recurring one points at a schema/parse/prompt defect.
        try:
            from . import introspect
            introspect.record("decide-failed", "issuework._decide",
                              f"{type(last_exc).__name__}: {last_exc} (after in-band retry)",
                              evidence=system[:200])
        except Exception:
            pass
        return {}          # fail-open: a model/parse failure never blocks the queue

    def _compose(self, instruction: str, facts: str, fallback: str) -> str:
        """Write a comment in Lara's voice from the given facts; fall back to a plain
        line if the model is unavailable, so a comment always gets posted."""
        try:
            r = self.router.complete(
                [{"role": "user", "content": facts}],
                system=_VOICE + " " + instruction,
                min_rung=self._rung, max_tokens=500)
            t = (r.completion.text or "").strip()
            return t if t else fallback
        except Exception:
            return fallback

    def _issue_text(self, issue: dict) -> str:
        text = f"ISSUE #{issue.get('number')}: {issue.get('title','')}\n\n" \
               f"{(issue.get('body') or '').strip()}"
        try:
            comments = self.gitea.list_comments(issue["number"])
        except GiteaError:
            comments = []
        # Only genuine HUMAN discussion informs the assessment — feeding Lara's own
        # status comments back to her makes her ask the human to explain her own notes.
        human = [c for c in comments if not self._is_mine(c)]
        if human:
            text += "\n\nDISCUSSION SO FAR:\n" + "\n".join(
                f"- {(c.get('user') or {}).get('login','?')}: {(c.get('body') or '')[:400]}"
                for c in human[-6:])
        return text

    def _is_mine(self, comment: dict) -> bool:
        # Login ONLY. The old "— Lara" body-signature fallback made any human
        # comment that quoted her (or addressed her by name with a dash) invisible
        # as discussion/rebuttal — a timely rebuttal could be missed and the issue
        # auto-closed. The login check protects everything the signature did.
        return (comment.get("user") or {}).get("login") == self.actor

    # -- the queue ------------------------------------------------------- #
    def _next_actionable(self) -> Optional[dict]:
        try:
            issues = self.gitea.list_issues(labels=[self.label], state="open")
        except GiteaError:
            return None
        for iss in issues:
            names = {l.get("name") for l in (iss.get("labels") or [])}
            if DONE_LABEL in names:
                continue                            # landed, awaiting promotion
            if "harness-bug" in names and HIGH_LABEL in names:
                # Self-surgery on a HIGH-priority harness bug is never auto-picked: a human
                # must sanity-check it and hand it to her (assign her explicitly). Routine
                # harness bugs flow like any other issue. Either way the self-critical guard
                # keeps her off the machinery files (forge/selfdev/introspect).
                assignees = {(a or {}).get("login") for a in (iss.get("assignees") or [])}
                if self.actor not in assignees:
                    continue
            pending = self._council_pending(iss)
            if pending is not None:
                marker, rebuttals = pending
                if rebuttals:
                    return iss                      # a rebuttal to weigh
                age = self._comment_age_days(marker)
                window = int(getattr(self.cfg, "council_rebuttal_days", 7))
                if age is not None and age >= window:
                    return iss                      # window elapsed — time to close it
                continue                            # still open for rebuttal
            if self._waiting_on_human(iss):
                continue                            # ball is with a human right now
            if self._has_open_blocker(iss):
                continue                            # blocked by an unfinished dependency
            return iss
        return None

    def _has_open_blocker(self, issue: dict) -> bool:
        """True if this issue has an OPEN Gitea dependency — don't work it (or churn
        on its blocker) until the blocker is resolved; the blocker gets worked on its
        own turn."""
        try:
            return any(d.get("state") == "open"
                       for d in self.gitea.list_dependencies(issue["number"]))
        except GiteaError:
            return False

    def _waiting_on_human(self, issue: dict) -> bool:
        """True when the issue is assigned to a human AND they haven't replied since
        Lara last spoke — i.e. it's genuinely their turn. Once they comment back, it
        becomes actionable again and Lara re-assesses with their answer in hand."""
        assignees = [(a or {}).get("login") for a in (issue.get("assignees") or [])]
        humans = [a for a in assignees if a and a != self.actor]
        if not humans:
            return False
        try:
            comments = self.gitea.list_comments(issue["number"])
        except GiteaError:
            return True
        if not comments:
            return True
        return self._is_mine(comments[-1])          # last word was Lara's -> still waiting

    def run_once(self) -> str:
        # Re-entrancy guard: the forge's gate shells out to `anvil doctor`; running the
        # forge from inside one would fork-bomb (doctor -> forge -> doctor -> ...).
        import os
        if os.environ.get("ANVIL_IN_DOCTOR"):
            return "skipped (inside anvil doctor)"
        if not self.gitea.ok:
            return "gitea not configured"
        iss = self._next_actionable()
        if not iss:
            return "no actionable issues"
        # Single-writer: never work an issue while another forge/self-dev run holds
        # the tree — a concurrent run would fight over `test` and falsely fail/escalate.
        from .forge import forge_lock
        with forge_lock() as got:
            if not got:
                return "busy (another forge run in progress)"
            try:
                return self.work_issue(iss)
            except Exception as exc:      # a bug here must never kill the sleep loop
                self.log(f"[issuework] #{iss.get('number')} errored: "
                         f"{type(exc).__name__}: {exc}")
                return f"#{iss.get('number')}: errored"

    # -- judgment + work ------------------------------------------------- #
    def work_issue(self, issue: dict, _depth: int = 0) -> str:
        num = issue["number"]
        creator = (issue.get("user") or {}).get("login") or ""
        # Who to hand a question / push-back to: the creator, or the operator if Lara
        # filed it herself (no point asking herself).
        asker = creator if creator and creator != self.actor else self.operator
        text = self._issue_text(issue)

        # 0a. council rebuttal window — if the council previously declined this idea,
        # this is the deliberative second round (rebuttal / re-hearing / close).
        pending = self._council_pending(issue)
        if pending is not None:
            return self._council_second_round(issue, pending, asker, text)

        # 0. dependency — if a hard blocker exists, DEFER: record the dependency and
        # step back. The blocker gets worked on its own turn, and _next_actionable then
        # skips THIS issue (open blocker) until it's resolved — no recursion, no churn.
        blocker = self._blocking_dep(issue, text)
        if blocker is not None:
            self._comment(num, self._compose(
                "Explain that this can't be done safely until the blocker is fixed, "
                "so you're setting it aside and will pick it up once the blocker lands. "
                "Keep it brief.",
                f"Issue #{num} is blocked by issue #{blocker['number']} "
                f"('{blocker.get('title','')}') — that has to be fixed first. You've "
                "recorded the dependency and will work it once the blocker is resolved.",
                f"This is blocked by #{blocker['number']} — I've marked the dependency "
                "and I'll pick this up once that's resolved. — Lara"))
            return f"#{num}: deferred (blocked by #{blocker['number']})"

        # 1. claim it — assignment is the signal she's on it (no "on it" comment).
        self._assign(num, [self.actor])

        # 2. clarity gate — ask only what she genuinely can't determine herself.
        clar = self._decide(
            "You are Lara, a careful engineer triaging a work item for your OWN "
            "codebase (a local-first family assistant) — you can read every file and "
            "make normal engineering decisions yourself. Decide whether the issue gives "
            "you enough to act: a clear goal and a way to tell when it's done. You will "
            "read the code and match existing style, naming, placement, and wording on "
            "your own — do NOT ask about those. Set confident=false ONLY when something "
            "essential is missing or ambiguous that ONLY the creator can resolve "
            "(unclear intent, missing requirements, conflicting goals, or you can't tell "
            "what 'done' means). Default to confident=true and proceed; asking is rare.",
            text + "\n\nReturn JSON.", _CLARITY_SCHEMA)
        if clar.get("confident") is False and clar.get("questions"):
            qs = "\n".join(f"- {q}" for q in clar["questions"][:5])
            self._comment(num, self._compose(
                "Thank them for filing it, then ask these specific questions so you can "
                "trace and fix it confidently. Keep your framing tight; let the "
                "questions carry the length.",
                f"Questions you need answered before you can work issue #{num}:\n{qs}",
                f"Thanks for filing this. Before I dig in I need a bit more:\n\n{qs}\n\n"
                "Add that when you can and I'll pick it back up. — Lara"))
            self._assign(num, [asker])
            return f"#{num}: asked for clarification"

        # 3. council — is the IDEA itself sound? Push back to the creator if not.
        # A vote that could not be computed (model outage, parse failure) is NOT a
        # verdict in either direction: defer the whole stage and retry next tick.
        # (The old default counted an outage as an approval — three failed calls
        # was unanimous consent.)
        votes = []
        for name, lens in COUNCIL_LENSES:
            v = self._decide(
                "You are a member of ANVIL's review council. Judge the proposal "
                f"through THIS lens only: {lens} Be honest — say ok=false if you have "
                "real concerns.", text + "\n\nReturn JSON.", _VOTE_SCHEMA)
            if "ok" not in v:
                return f"#{num}: council deferred — {name} lens returned no verdict (retry next pass)"
            votes.append((name, bool(v.get("ok")), v.get("reason", "")))
        against = [(n, r) for n, ok, r in votes if ok is False]
        if len(against) > len(votes) / 2:
            reasons = "\n".join(f"- **{n}**: {r}" for n, r in against if r)
            window = int(getattr(self.cfg, "council_rebuttal_days", 7))
            body = self._compose(
                "You took this to the council and the majority has real concerns about "
                "the IDEA itself. Push back honestly but respectfully, own that you "
                f"might have misread the intent, and invite a rebuttal within {window} "
                "days — if it addresses their concerns you'll take it back to them.",
                f"Council concerns about issue #{num}:\n{reasons}",
                f"I took this to the council and the majority has real concerns:\n\n"
                f"{reasons}\n\nIf you think I've misread the intent or can address these, "
                f"reply within {window} days and I'll take it back to the council. — Lara")
            self._comment(num, body + "\n\n" + COUNCIL_MARK)   # mark the first ruling
            self._assign(num, [asker])
            return f"#{num}: council pushed back ({len(against)}/{len(votes)})"

        # 4. work it on `test`, retrying with the full toolset before asking for help.
        return self._work(num, text, issue.get("title", ""))

    def _blocking_dep(self, issue: dict, text: str) -> Optional[dict]:
        """Return the issue THIS one is blocked by (must be fixed first), or None.
        Explicit Gitea dependency links win; otherwise Lara looks for an implicit hard
        dependency and, if she finds one, RECORDS it as a real Gitea dependency."""
        num = issue.get("number")
        # 1. An explicit Gitea dependency link is authoritative — honour it.
        try:
            for dep in self.gitea.list_dependencies(num):
                if dep.get("state") == "open":
                    return dep
        except GiteaError:
            pass
        # 2. Otherwise, look for an implicit hard dependency among the OTHER open issues.
        try:
            others = [i for i in self.gitea.list_issues(labels=[self.label], state="open")
                      if i.get("number") != num]
        except GiteaError:
            return None
        if not others:
            return None
        catalog = "\n".join(f"#{i['number']}: {i.get('title','')}" for i in others[:20])
        d = self._decide(
            "You are Lara triaging a work item. Given the OTHER open issues, decide if "
            "THIS issue is BLOCKED — it cannot be correctly done until another specific "
            "issue is fixed first. Only a genuine hard dependency counts, not vague "
            "relatedness. Return blocked_by = the blocking issue number, or 0 if none.",
            text + "\n\nOTHER OPEN ISSUES:\n" + catalog + "\n\nReturn JSON.", _DEP_SCHEMA)
        try:
            b = int(d.get("blocked_by") or 0)
        except (TypeError, ValueError):
            b = 0
        blocker = next((i for i in others if i.get("number") == b), None) if b else None
        if blocker:                          # record the discovered dependency in Gitea
            self.gitea.add_dependency(num, blocker["number"])
        return blocker

    # -- council: the deliberative second round -------------------------- #
    def _council_pending(self, issue):
        """If this issue is in the post-decline rebuttal window, return
        (first_ruling_comment, [rebuttal_comments_since]) else None."""
        try:
            comments = self.gitea.list_comments(issue["number"])
        except GiteaError:
            return None
        marker = None
        for c in comments:
            if not self._is_mine(c):
                continue
            b = c.get("body") or ""
            if ESCALATED_MARK in b:
                return None                 # already reversed + escalated; done here
            if COUNCIL_MARK in b:
                marker = c                  # remember the LATEST first-ruling marker
        if marker is None:
            return None
        after = comments[comments.index(marker) + 1:]
        rebuttals = [c for c in after if not self._is_mine(c)]
        return (marker, rebuttals)

    def _comment_age_days(self, comment) -> Optional[float]:
        ts = (comment.get("created_at") or "").strip()
        if not ts:
            return None
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        except Exception:
            return None

    def _council_second_round(self, issue, pending, asker, text) -> str:
        num = issue["number"]
        marker, rebuttals = pending
        first_ruling = marker.get("body", "")
        window = int(getattr(self.cfg, "council_rebuttal_days", 7))

        # No rebuttal yet: if the window has elapsed, the decision stands -> close.
        if not rebuttals:
            age = self._comment_age_days(marker)
            if age is None or age < window:
                return f"#{num}: awaiting rebuttal"
            self._comment(num, self._compose(
                f"A full {window} days passed with no rebuttal to the council's "
                "concerns, so the decision stands and you're closing this. Be gracious "
                "and leave the door open to reopen with a stronger case.",
                f"Issue #{num}: no rebuttal in {window} days; closing. The council's "
                f"concerns were:\n{first_ruling}",
                f"It's been {window} days with no rebuttal to the council's concerns, "
                "so I'll close this for now. Reopen anytime with a case that speaks to "
                "them. — Lara"))
            self._close(num)
            return f"#{num}: closed (no rebuttal in {window}d)"

        rtext = "\n".join(f"- {(c.get('user') or {}).get('login','?')}: "
                          f"{(c.get('body') or '')[:500]}" for c in rebuttals[-4:])

        # Gauge the rebuttal YOURSELF first — don't waste the council's time on one
        # that doesn't actually engage their concerns.
        gauge = self._decide(
            "You are Lara. The council previously declined this idea for the concerns "
            "below, and the creator has rebutted. Judge HONESTLY and in context whether "
            "the rebuttal genuinely ADDRESSES the council's specific concerns — enough "
            "to be worth a second hearing. Don't advocate; assess. plausible=true only "
            "if it actually engages the concerns.",
            f"COUNCIL'S FIRST RULING:\n{first_ruling}\n\nREBUTTAL:\n{rtext}\n\nReturn JSON.",
            _GAUGE_SCHEMA)
        # No verdict computed (outage/parse failure) -> say NOTHING and retry next
        # tick. Telling a human their argument "doesn't address the concerns" with an
        # empty reason — because a model call timed out — is a fabricated public
        # judgment, the worst possible place to fail open.
        if "plausible" not in gauge:
            return f"#{num}: rebuttal gauge returned no verdict (retry next pass)"
        if not gauge.get("plausible"):
            self._comment(num, self._compose(
                "The rebuttal doesn't yet address the council's concerns. Explain "
                "specifically what still isn't answered — honestly and kindly — and "
                "leave the door open to try again within the window.",
                f"Rebuttal on #{num} doesn't resolve the council's concerns "
                f"({gauge.get('reason','')}). Concerns:\n{first_ruling}",
                "I hear you, but this doesn't yet address the council's concerns "
                f"({gauge.get('reason','')}). Speak to those directly and I'll take it "
                "back to them. — Lara"))
            self._assign(num, [asker])
            return f"#{num}: rebuttal insufficient"

        # Plausible -> re-present to the council WITH the first ruling + the rebuttal.
        votes = []
        for name, lens in COUNCIL_LENSES:
            v = self._decide(
                "You are a member of ANVIL's review council RECONSIDERING a proposal you "
                f"previously declined. Judge through THIS lens: {lens} You have your "
                "first ruling and the creator's rebuttal — weigh whether the rebuttal "
                "resolves your concern. Be honest; ok=true only if it is genuinely "
                "addressed now.",
                f"{text}\n\nYOUR FIRST RULING:\n{first_ruling}\n\nREBUTTAL:\n{rtext}"
                "\n\nReturn JSON.", _VOTE_SCHEMA)
            if "ok" not in v:              # outage is not a vote — retry next pass
                return f"#{num}: re-hearing deferred — {name} lens returned no verdict"
            votes.append((name, bool(v.get("ok")), v.get("reason", "")))
        against = [(n, r) for n, ok, r in votes if ok is False]
        if len(against) > len(votes) / 2:
            reasons = "\n".join(f"- **{n}**: {r}" for n, r in against if r)
            self._comment(num, self._compose(
                "You took the rebuttal back to the council and, having weighed it, they "
                "still can't get behind the idea. Close the issue, conveying the "
                "second-round reasoning respectfully — this is the final call.",
                f"Second council review of #{num} still declines:\n{reasons}",
                "I took your rebuttal back to the council and, having considered it, "
                f"they still can't get behind this:\n\n{reasons}\n\nI'll close it "
                "here. — Lara"))
            self._close(num)
            return f"#{num}: council declined 2nd time, closed"

        # Council reversed -> a human should make the final call before building it.
        self._comment(num, self._compose(
            "On reconsideration WITH the rebuttal, the council came around and now "
            "supports this. Because that reverses their first ruling, you're looping in "
            "the operator for the final go-ahead before you build it. Keep it brief.",
            f"Council reversed on #{num} after the rebuttal and now supports it; "
            "handing to the operator for the final call.",
            "Good news — with your rebuttal in hand the council came around and now "
            "supports this. Since that reverses their first call, I'm looping in "
            f"{self.operator} for the final go-ahead. — Lara") + "\n\n" + ESCALATED_MARK)
        self._assign(num, [self.operator])
        return f"#{num}: council approved on 2nd round, escalated to {self.operator}"

    def _work(self, num: int, text: str, title: str) -> str:
        from . import selfdev
        from .forge import Forge
        attempts: List[str] = []
        for attempt in range(1, MAX_WORK_ATTEMPTS + 1):
            edits: List[str] = []

            def cap(msg):                    # capture what she edited, still log through
                s = str(msg)
                if "edited " in s:
                    edits.append(s.split("edited ", 1)[1])
                self.log(msg)
            try:
                f = Forge(branch=getattr(self.cfg, "forge_branch", "test"),
                          reviewer=selfdev.build_local_reviewer(
                              self.cfg, min_rung=selfdev.pick_review_rung(self.cfg),
                              context=text),
                          push_remote=(getattr(self.cfg, "forge_push_remote", "gitea")
                                       if getattr(self.cfg, "forge_push", False) else None),
                          push_token=getattr(self.cfg, "gitea_token", None),
                          logger=cap)
                f.work_item = (f"Work Gitea issue #{num}: {title}.\n\n{text}\n\nMake the "
                               "SMALLEST correct change; keep every test green; never "
                               "weaken tests.")
                res = selfdev.drive_forge(f, self.cfg, cap, n=attempt)
            except Exception as exc:
                attempts.append(f"attempt {attempt}: setup error ({type(exc).__name__})")
                continue

            if res and res.kept:
                # Shipped to `test` = the issue is DONE -> close it (proper etiquette;
                # no waiting). It still reaches `main` later via the reviewed promotion,
                # but that's a pipeline concern, not the issue's open/closed state. The
                # 7-day wait is ONLY for a council NO, never for a shipped fix.
                self._comment(num, self._compose(
                    "You just fixed this and shipped it to `test`, and you're CLOSING "
                    "the issue now — it's done. Post ONE update: what the issue was, "
                    "what you changed and briefly why, the result, and that you're "
                    "closing it (it'll still reach `main` via the reviewed promotion, "
                    "but the issue itself is done). Substance over length.",
                    f"Fixed and shipped issue #{num}: '{title}' to `test`.\nWhat you "
                    f"changed: {'; '.join(edits) or 'a small correct change'}.\nResult: "
                    f"tests green ({res.before_failed}->{res.after_failed} failures), "
                    f"{self._head_summary()} Closing the issue.",
                    f"Done — shipped to `test`, tests green. {self._head_summary()} "
                    "Closing this; it'll reach `main` on the next reviewed promotion. "
                    "— Lara"))
                self._add_label(num, DONE_LABEL)     # marker for the promotion pipeline
                self._close(num)                     # shipped -> close (also unassigns)
                return f"#{num}: worked, shipped to test, closed (attempt {attempt})"

            reason = getattr(res, "reason", "no change") or "no change"
            # ENVIRONMENTAL blocks are not Lara's failure and must NOT burn her
            # attempts: (a) a gated GREEN change that couldn't merge because the
            # operator was mid-edit ("live tree busy"), and (b) the forge
            # refusing to even start on a dirty tree ("uncommitted changes" /
            # "worktree unavailable") — both are the operator's transient tree
            # state colliding with her tick. Burning 3 tries on these
            # mis-escalated real, workable issues (happened live on #26, and on
            # #62 where three "uncommitted changes" skips forced a false
            # escalation). Defer the whole issue, uncounted, and retry next tick.
            deferrable = ("live tree busy", "uncommitted changes",
                          "worktree unavailable", "cap-hit", "month-cap")
            if any(d in reason for d in deferrable):
                return (f"#{num}: not workable this tick ({reason}) — "
                        "environmental (busy tree or exhausted budget), "
                        "deferred uncounted; will retry next pass")
            attempts.append(f"attempt {attempt}: {reason}")

        # 3 real tries and it still won't land cleanly — record it (a repeated failure
        # to land can be a driver/harness limit worth filing) and hand it to the operator.
        try:
            from . import introspect
            introspect.record("selfdev-cant-land", "issuework._work",
                              f"issue #{num} failed {MAX_WORK_ATTEMPTS}x",
                              evidence="; ".join(attempts))
        except Exception:
            pass
        tried = "\n".join(f"- {a}" for a in attempts)
        self._comment(num, self._compose(
            "You genuinely tried to fix this several times and couldn't land it. Ask "
            "the operator for help. STRICT GROUNDING: describe the failures ONLY by "
            "quoting the attempt log below — include each attempt's reason VERBATIM. "
            "Never invent specifics (test names, assertion details, error messages) "
            "that are not literally in that log; a plausible story that isn't in the "
            "log is worse than saying 'the log doesn't tell me why'. If a reason says "
            "'refused: the plan chose files outside the allowlist', the ask is a "
            "structural one — name the files and ask whether they should be editable.",
            f"Issue #{num}: '{title}'. You tried {MAX_WORK_ATTEMPTS} times on `test` "
            f"and it wouldn't land. THE ATTEMPT LOG (quote this, add nothing):\n{tried}",
            f"I've tried this {MAX_WORK_ATTEMPTS} times and can't land a change that "
            f"keeps every test green:\n{tried}\n\nCould you take a look with me? — Lara"))
        self._assign(num, [self.operator])
        return f"#{num}: escalated to {self.operator} after {MAX_WORK_ATTEMPTS} tries"

    # -- gitea side effects (never crash the loop) ----------------------- #
    def _comment(self, num: int, body: str) -> None:
        try:
            self.gitea.comment_issue(num, body)
        except GiteaError:
            pass

    def _assign(self, num: int, logins: List[str]) -> None:
        try:
            self.gitea.set_assignees(num, logins)
        except GiteaError:
            pass

    def _add_label(self, num: int, name: str) -> None:
        try:
            self.gitea.add_labels(num, [name])
        except GiteaError:
            pass

    def _close(self, num: int) -> None:
        try:
            self.gitea.set_assignees(num, [])
            self.gitea.close_issue(num)
        except GiteaError:
            pass

    def _head_summary(self) -> str:
        try:
            out = subprocess.run(["git", "log", "-1", "--format=%h %s"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            return f"Commit `{out}`." if out else ""
        except Exception:
            return ""
