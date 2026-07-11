"""ANVIL's mind — heartbeat, two-tier memory, and a sleep/dream cycle.

This is the autonomy layer. It gives ANVIL:

* **Short-term memory (STM)** — a rolling, high-detail buffer of recent events,
  observations, and its own thoughts (``memory/short_term.jsonl``, capped).
* **Long-term memory (LTM)** — the durable, distilled note store (``memory.py``).
* **Heartbeat (``think``)** — a cheap periodic "tick" that produces a spontaneous,
  useful thought from what's recently on its mind. This is the spark of autonomy.
* **Sleep / dream (``dream``)** — periodic consolidation: it reviews STM, distills
  durable *lessons* and *facts about the operator* into LTM, raises open
  *questions*, and writes concrete *self-improvement proposals* — then prunes STM,
  decays old memories, and lets its persona evolve. This is self-teaching from
  experience.
* **Pulse (``pulse``)** — the circadian loop: frequent heartbeats, periodic dreams.

Safety: dreams write self-improvement ideas to ``test-reports/proposals/`` for
human review. The mind never edits ANVIL's own code.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import persona


# A memory is unfaithful if it claims ANVIL took an action (it has read-only
# senses — it observes, it doesn't switch/configure/deploy). Catches both the
# first-person accomplishment ("I've implemented…", "we switched…") and the
# accomplishment-log opener ("Configured…", "Switched X to…").
_FP_ACTION = re.compile(
    r"\b(i|i'?ve|i'?m|we|we'?ve)\b[^.]{0,32}\b(implement|configur|deploy|install"
    r"|automat|integrat|snapshot|debounc|establish|enabl|disabl|refin|set\s?up"
    r"|switch|rout|pin|creat|built|wrote|fix|updat|chang|add)\w*", re.I)
_START_ACTION = re.compile(
    r"^\W*(implement|configur|deploy|install|automat|integrat|snapshot|establish"
    r"|enabl|disabl|set\s?up|switch|rout|pin|creat|built|fix)\w*\b", re.I)


def _faithful(note: str, source_terms: set, action_terms=frozenset()) -> bool:
    """Keep a candidate memory only if it's grounded in the source events, and —
    if it claims ANVIL took an action — only if that action matches a real logged
    execution (``action_terms``, drawn from 'did …' STM records). With no logged
    actions, every action-claim is fabrication and is dropped; once ANVIL actually
    acts, the true ones pass. Errs toward keeping honest observations."""
    low = (note or "").lower()
    words = set(re.findall(r"[a-z0-9]{4,}", low))
    if not words:
        return False
    if _FP_ACTION.search(low) or _START_ACTION.search(low):
        if len(words & action_terms) < 2:      # action claimed but not logged
            return False
    grounded = len(words & source_terms)
    return grounded >= 2 or grounded / len(words) >= 0.30

ROOT = Path(__file__).resolve().parent.parent
PROPOSALS = ROOT / "test-reports" / "proposals"

# A spontaneous thought is worth a push only if it reads as something the operator
# would want surfaced: a question, a suggestion, or a flagged pattern. Most idle
# heartbeats are mundane and should stay silent, so we gate hard.
_NOTABLE = re.compile(
    r"\?|\b(should i|want me|i could|i can|i'd|i would|suggest|recommend|remind"
    r"|noticed|noticing|it looks like|you (usually|often|tend)|worth|heads up"
    r"|might want|consider)\b", re.I)


def _notable(text: str) -> bool:
    t = (text or "").strip()
    return len(t) > 12 and bool(_NOTABLE.search(t))


def _quiet_now(cfg, hour: Optional[int] = None) -> bool:
    """Inside the operator's quiet window? Ambient pushes (thoughts/dreams) hold
    overnight; direct interactions (answers, approvals) are never gated — if
    you're chatting at 2am you obviously want the reply."""
    try:
        start = int(getattr(cfg, "push_quiet_start", 22))
        end = int(getattr(cfg, "push_quiet_end", 7))
    except (TypeError, ValueError):
        return False
    if start == end:                      # equal bounds = feature disabled
        return False
    h = datetime.now().hour if hour is None else int(hour)
    if start < end:                       # same-day window (e.g. 13..15)
        return start <= h < end
    return h >= start or h < end          # overnight wrap (e.g. 22..7)


def _push(cfg, title: str, body: str, tag: str) -> None:
    """Best-effort push from the mind loop; never let a notification failure
    disturb the heartbeat/dream cycle."""
    if tag in ("thought", "dream") and _quiet_now(cfg):
        return                            # ambient — respect quiet hours
    try:
        from . import push
        push.notify(cfg, title, body[:180], url="/", tag=tag)
    except Exception:
        pass


def _similarity(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' word sets — cheap, dependency-free."""
    wa = set(re.findall(r"[a-z0-9]{4,}", (a or "").lower()))
    wb = set(re.findall(r"[a-z0-9]{4,}", (b or "").lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _is_repeat(thought: str, recent_thoughts, threshold: float = 0.5) -> bool:
    """True if this thought substantially echoes one ANVIL just had. Stops the
    heartbeat from ruminating on the same idea (which then compounds: the echo
    fills STM, the next heartbeat reads it back, dreams over-strengthen it, and
    the operator gets pinged with near-duplicate notifications)."""
    return any(_similarity(thought, prev) >= threshold for prev in recent_thoughts)


# A thought that claims to have OBSERVED the house ("I've noticed the dishwasher
# trips the UPS...") is a factual claim and must be held to evidence, exactly like
# dream lessons. Ideas/questions that don't claim observation stay ungated.
_OBS_CLAIM = re.compile(
    r"\b(i[' ]?ve noticed|i noticed|i keep noticing|i saw|i[' ]?ve seen"
    r"|i[' ]?ve been (seeing|watching|tracking|noticing|monitoring)"
    r"|every time|each time|keeps? (firing|dropping|kicking|spiking|cycling"
    r"|turning|tripping|running))\b", re.I)


def _grounded_thought(thought: str, evidence_terms: set) -> bool:
    """A heartbeat thought that claims an observation must trace to real evidence
    (house events, chats, logged actions — NEVER prior thoughts, or fabrications
    would license more fabrications). Non-observational ideas pass freely."""
    if not _OBS_CLAIM.search(thought or ""):
        return True
    return _faithful(thought, evidence_terms)

# The journal gains a line per heartbeat and dream; on a long-running ANVIL it
# would grow without bound. Like STM (capped) and the proposal archive
# (_prune_proposals), give it a ceiling — but only pay the read+rewrite when the
# file actually crosses a byte threshold, then keep the most recent lines.
JOURNAL_MAX_BYTES = 1_000_000
JOURNAL_KEEP_LINES = 2000

HEARTBEAT_SYS = (
    "You are idling, letting your mind wander over what's actually happened "
    "recently. Produce ONE short, grounded thought — a pattern in the home, "
    "something the operator seems to want, a question worth asking, or a small "
    "idea to improve ANVIL. Your ONE rule is FAITHFULNESS: the events you are "
    "shown are your ONLY knowledge of the house. NEVER invent sensors, devices, "
    "rooms, schedules, or patterns that do not appear in those events — no "
    "imagined dishwashers, thermostats, motion sensors, or routines. Never say "
    "'I noticed X' unless X is literally in the events. Don't claim you did "
    "something unless it actually happened (actions are logged). You CANNOT run "
    "tools or take any action during idle thinking — never promise 'I'll pull/"
    "draft/track/set up X now'; phrase ideas as suggestions or questions for "
    "the operator instead. If the events show nothing noteworthy, reply with "
    "exactly 'pass'. 1-2 sentences, first person, no preamble."
)
DREAM_SYS = (
    "You are consolidating memory during sleep. Your ONE rule is FAITHFULNESS: "
    "record only what is directly supported by the recent events you are given. "
    "Never invent details, numbers, device features, or configuration that are "
    "not in those events. You can OBSERVE the home and also ACT on it, but only "
    "through approved tools — and every action you take is logged as an event that "
    "starts with 'did …'. So you may record an action ONLY if there is a matching "
    "'did …' record in the events below; NEVER claim you did, changed, switched, "
    "configured, set up, or enabled something that isn't logged. Otherwise "
    "describe what you OBSERVED or what the operator SAID, tentatively ('it looks "
    "like…', 'the operator mentioned…', 'X often…'), not as accomplishments. If "
    "little is clearly supported, keep the lists short or empty."
)
DREAM_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {"type": "array", "items": {"type": "string"}},
        "profile_facts": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["lessons"],
}


# --------------------------------------------------------------------------- #
# Short-term memory
# --------------------------------------------------------------------------- #
class ShortTerm:
    def __init__(self, path: Path, cap: int = 300):
        self.path = Path(path)
        self.cap = cap
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, text: str, meta: Optional[dict] = None) -> dict:
        rec = {"ts": time.time(),
               "iso": datetime.now().isoformat(timespec="seconds"),
               "kind": kind, "text": text, "meta": meta or {}}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            # The fast append can't heal a read-only file (AV / sync tool lock);
            # fall back to the atomic, read-only-tolerant rewrite so a single
            # locked thought stream never crashes the heartbeat / observe loop.
            self._write(self.all() + [rec])
        self._prune()
        return rec

    def all(self) -> List[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text("utf-8", "replace").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def recent(self, n: int = 30) -> List[dict]:
        # Hot path: think() calls this every heartbeat. Read only a bounded
        # window from the END of the file (same tail-read discipline as
        # _tail_lines) instead of pulling the whole STM into memory, then
        # JSON-parse from the tail until we have n valid records. Same result
        # as all()[-n:] — malformed lines are skipped either way — but the I/O
        # stays cheap as STM fills toward (a large) cap.
        if not self.path.exists() or n <= 0:
            return []
        try:
            size = self.path.stat().st_size
            window = min(size, max(4096, n * 512))
            with self.path.open("rb") as fh:
                fh.seek(size - window)
                raw = fh.read()
            lines = raw.decode("utf-8", "replace").splitlines()
            partial = window < size   # we may have sliced through a leading line
            if partial and len(lines) > 1:
                lines = lines[1:]     # drop the partial fragment of a sliced line
        except OSError:
            lines = self.path.read_text("utf-8", "replace").splitlines()
            partial = False
        out: List[dict] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= n:
                break
        # A short window may not have held n valid records; if it didn't cover
        # the whole file, fall back to a full read so we still match all()[-n:].
        if partial and len(out) < n:
            return self.all()[-n:]
        out.reverse()
        return out

    def size(self) -> int:
        # Hot path: pulse() calls this every heartbeat tick to decide whether to
        # dream. Counting raw lines skips the full JSON parse of every record
        # (same discipline as _prune / recent). Under normal operation the raw
        # count equals the record count -- prune rewrites only valid records --
        # and a raw count never under-counts, so the dream trigger stays correct.
        return self._raw_line_count()

    def _write(self, items: List[dict]) -> None:
        body = "\n".join(json.dumps(x) for x in items)
        # Atomic + read-only-tolerant, like every other state file in ANVIL:
        # a crash mid-rewrite (prune/clear) can't truncate the thought stream.
        from . import config as cfgmod
        cfgmod.atomic_write(self.path, body + ("\n" if body else ""))

    def _raw_line_count(self) -> int:
        # Hot path: _prune() runs this on every append and size() on every pulse
        # tick. Counting newline bytes avoids decoding the whole file to UTF-8 and
        # building a stripped per-line list just to check the cap. Every record is
        # written newline-terminated (append adds "\n"; _write joins with "\n" +
        # a trailing "\n"), so the newline count equals the record count; a final
        # line missing its newline (an interrupted/external write) is still counted
        # via the trailing check rather than silently dropped.
        if not self.path.exists():
            return 0
        try:
            raw = self.path.read_bytes()
        except OSError:
            return 0
        if not raw:
            return 0
        return raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)

    def _prune(self) -> None:
        # Hot path: append runs this every tick. Parsing every JSON record just
        # to check the cap is wasteful, so count raw lines first and only do the
        # full parse + rewrite when we're actually over cap. A raw count >= the
        # parsed count (malformed lines are dropped by all()), so an under-cap
        # raw count guarantees we're under cap and can skip the work safely.
        if self._raw_line_count() <= self.cap:
            return
        items = self.all()
        if len(items) > self.cap:
            self._write(items[-self.cap:])

    def clear_to_tail(self, keep: int = 15) -> int:
        items = self.all()
        dropped = max(0, len(items) - keep)
        self._write(items[-keep:] if keep else [])
        return dropped

    def clear_consolidated(self, groups, before: float, keep: int = 15) -> int:
        """Watermark-based prune after a dream: drop ONLY records that were
        actually consolidated — i.e. belong to a group whose consolidation
        SUCCEEDED and existed when the dream started (`before`). A group whose
        model call failed keeps its records for the next dream; events that
        arrived mid-dream survive; and the newest `keep` records stay as
        conversational context regardless. Consolidation must never destroy
        what it failed to consolidate."""
        items = self.all()
        done = set(groups)
        kept: List[dict] = []
        for i, r in enumerate(items):
            if i >= len(items) - keep:
                kept.append(r)                       # the tail always survives
                continue
            m = r.get("meta") or {}
            actor = str(m.get("actor") or "") if isinstance(m, dict) else ""
            if actor not in done or float(r.get("ts", 0)) >= before:
                kept.append(r)                       # unconsolidated -> keep
        self._write(kept)
        return len(items) - len(kept)


# --------------------------------------------------------------------------- #
# The mind
# --------------------------------------------------------------------------- #
class Mind:
    def __init__(self, cfg, router=None, memory=None):
        self.cfg = cfg
        from .router import Router
        from .memory import MemoryStore
        # The mind is all BACKGROUND (heartbeat thoughts, dreams, sensing) — tag
        # its spend so tiered budget throttles it before foreground chat.
        self.router = router or Router(cfg, plane="dreams")
        # Same embedder the Pipeline builds — WITHOUT one, every dream-written
        # lesson/fact was permanently invisible to semantic recall (the largest
        # weight in scoring), silently demoting the most durable memory tier.
        emb = None
        if getattr(cfg, "use_embeddings", True):
            prov = (getattr(self.router, "providers", None) or {}).get("ollama_local")
            if prov is not None:
                emb = lambda text: prov.embed(cfg.embed_model, text)
        self.ltm = memory or MemoryStore(cfg, embedder=emb)
        self.stm = ShortTerm(Path(cfg.memory_dir) / "short_term.jsonl",
                             cap=int(getattr(cfg, "stm_cap", 300)))
        self.journal = Path(cfg.memory_dir) / "journal.md"
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self._persona = persona.load()

    # -- helpers ------------------------------------------------------- #
    def _sys(self, role: str) -> str:
        return persona.preamble(self._persona) + "\n\n" + role

    _JOURNAL_LOCK = threading.Lock()   # mind thread + request threads both journal

    def _journal(self, mark: str, text: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        line = f"- {stamp} {mark} {text}\n"
        with self._JOURNAL_LOCK:
            try:
                with self.journal.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError:
                # The journal can be locked read-only by AV / a sync tool; the fast
                # append can't heal that, so fall back to the atomic, read-only-
                # tolerant rewrite (like STM) so a single locked journal never
                # crashes the heartbeat / dream loop. Locked so two fallback
                # rewrites can't overwrite each other's line.
                from . import config as cfgmod
                prev = self.journal.read_text("utf-8", "replace") \
                    if self.journal.exists() else ""
                cfgmod.atomic_write(self.journal, prev + line)
            self._cap_journal()

    def _cap_journal(self, max_bytes: int = JOURNAL_MAX_BYTES,
                     keep: int = JOURNAL_KEEP_LINES) -> None:
        """Trim the unbounded heartbeat/dream journal so it can't grow forever.

        Mirrors the disk hygiene already applied to STM (capped) and the
        proposal archive (``_prune_proposals``). The common append path pays
        only a cheap ``stat()`` — the full read + atomic rewrite happens only
        once the file crosses ``max_bytes``, after which the most recent
        ``keep`` lines are retained. Atomic + read-only tolerant so a locked
        journal never crashes the heartbeat loop.
        """
        try:
            if not self.journal.exists() \
                    or self.journal.stat().st_size <= max_bytes:
                return
            lines = self.journal.read_text("utf-8", "replace").splitlines()
            if len(lines) <= keep:
                return
            from . import config as cfgmod
            cfgmod.atomic_write(self.journal, "\n".join(lines[-keep:]) + "\n")
        except OSError:
            pass

    def observe(self, text: str, kind: str = "observation",
                meta: Optional[dict] = None) -> None:
        self.stm.append(kind, text, meta)

    # -- ambient senses ------------------------------------------------ #
    # Household domains worth watching for rhythm/security — deliberately NOT
    # binary_sensor (motion toggles constantly) so STM isn't flooded.
    _HOUSE_DOMAINS = ("person", "device_tracker", "media_player", "lock",
                      "cover", "alarm_control_panel", "climate")

    def sense_house(self) -> Optional[str]:
        """Look at the home and remember what CHANGED since last glance.

        Snapshots notable Home Assistant state (presence, media, locks, doors,
        climate) and records only the *changes* into STM as ``house`` events, so
        dreams can learn the household's rhythm over time without STM drowning in
        unchanged state. First look establishes a silent baseline. Best-effort:
        HA absent, unreachable, or empty -> records nothing, never raises."""
        try:
            from . import homeassistant as ha
            client = ha.HomeAssistant(timeout=getattr(self.cfg, "request_timeout", 5))
            if not client.is_configured:
                return None
            states = client.states()
            if not states:
                return None
            snap = {}
            for e in states:
                eid = e.get("entity_id", "")
                if eid.split(".", 1)[0] in self._HOUSE_DOMAINS:
                    snap[eid] = str(e.get("state", ""))
            prev = getattr(self, "_house_snap", None)
            self._house_snap = snap
            if prev is None:
                return None                      # establish baseline silently
            changed = [(k, prev[k], v) for k, v in snap.items()
                       if k in prev and prev[k] != v]
            if not changed:
                return None
            # PERMANENT event log (index.py): each transition lands in SQLite so
            # dreams/heartbeats can later ask REAL aggregate questions ("front
            # door opened after midnight 3x this week" from a count, not a
            # model's imagination). STM prose gets consolidated away; this
            # doesn't. Best-effort like everything else in sensing.
            try:
                from .index import SearchIndex
                ix = SearchIndex(self.cfg)
                for k, old, new in changed[:16]:
                    ix.record_event(k.split(".", 1)[1], old, new)
            except Exception:
                pass
            changes = [f"{k.split('.', 1)[1]} {old}->{new}"
                       for k, old, new in changed]
            digest = "; ".join(changes[:8])
            self.observe("house: " + digest, kind="house")
            self._journal("[sense]", digest[:120])
            return digest
        except Exception:
            return None

    def sense_weather(self) -> Optional[str]:
        """Check NWS for NEW severe weather at home and raise the alarm.

        Deterministic, evidence-first: the alert itself is the observation (a
        real API response, recorded to STM as a ``house`` event) — no model in
        the loop, so nothing to fabricate. Severe/Extreme alerts push
        immediately with tag 'alert', which deliberately BYPASSES quiet hours:
        a tornado warning at 3am should wake the operator. Best-effort like
        sense_house: no home location or no network -> silently does nothing."""
        try:
            from . import weather as wx
            w = wx.Weather(timeout=int(getattr(self.cfg, "request_timeout", 10) or 10),
                           contact=str(getattr(self.cfg, "push_contact", "") or ""))
            latlon = wx.home_latlon(self.cfg, w)
            if not latlon:
                return None
            seen = getattr(self, "_wx_alerts_seen", set())
            self._wx_alerts_seen = seen
            fired = []
            for a in w.alerts(latlon[0], latlon[1]):
                aid = a.get("id") or (a.get("event", "") + a.get("ends", ""))
                if not aid or aid in seen:
                    continue
                seen.add(aid)
                line = f"{a.get('event', 'weather alert')}: {a.get('headline', '')}"
                self.observe("house: weather alert — " + line, kind="house")
                self._journal("[weather]", line[:140])
                if a.get("severity") in ("Severe", "Extreme"):
                    _push(self.cfg, "Weather alert", line, tag="alert")
                fired.append(line)
            return "; ".join(fired) if fired else None
        except Exception:
            return None

    # -- heartbeat ----------------------------------------------------- #
    def think(self) -> str:
        recent = self.stm.recent(30)
        # What ANVIL recently thought AND asked — both feed the repeat gate.
        # Questions must be included: the operator rarely answers the thought
        # stream, so without this the same 'rain alerts: hard cutoff or
        # nudges?' gets re-asked every few heartbeats, forever.
        recent_thoughts = [str(r.get("text", "")) for r in recent
                           if r.get("kind") in ("thought", "question")][-10:]
        # Evidence = what actually happened (house observations, operator chat,
        # logged actions). Prior thoughts are deliberately EXCLUDED: a fabricated
        # thought must never count as evidence for the next one.
        evidence = " ".join(str(r.get("text", "")) for r in recent
                            if r.get("kind") in ("house", "chat", "action"))
        evidence_terms = set(re.findall(r"[a-z0-9]{4,}", evidence.lower()))
        if recent:
            # A valid-JSON-but-wrong-shape record (externally written, a future
            # schema, or a non-string text) survives all()/recent() parsing; use
            # .get()+str() so building the digest can't KeyError/TypeError and
            # crash the heartbeat (and the pulse loop) before the guarded call.
            # iso[11:16] is the "HH:MM" the event happened — without it every
            # digest line reads as "now" and thoughts misplace events in time.
            ctx = "\n".join(
                f"[{r.get('kind', '?')} {str(r.get('iso', ''))[11:16]}] "
                f"{str(r.get('text', ''))[:160]}" for r in recent)
            avoid = ""
            if recent_thoughts:
                avoid = ("\n\nYou have ALREADY had these thoughts recently — do "
                         "NOT repeat or reword them; think about something else "
                         "(a different room, task, person, or idea), or stay "
                         "silent by replying with just 'pass':\n- "
                         + "\n- ".join(t[:140] for t in recent_thoughts))
            prompt = ("Recently on your mind:\n" + ctx + avoid +
                      "\n\nHave one useful, NEW spontaneous thought now.")
        else:
            prompt = ("Your short-term memory is quiet. Raise one concrete idea "
                      "to improve ANVIL or help your operator, or a curiosity "
                      "worth exploring.")
        try:
            r = self.router.complete([{"role": "user", "content": prompt}],
                                     system=self._sys(HEARTBEAT_SYS),
                                     min_rung=0, max_tokens=140)
            thought = (r.completion.text or "").strip()
        except Exception as exc:
            thought = f"(heartbeat skipped: {exc})"
        if thought and not thought.startswith("(heartbeat skipped"):
            # Drop a heartbeat that just echoes a recent one: don't store it (so
            # STM/dreams stop compounding the loop) and don't ping the operator.
            if thought.lower().strip(".!") == "pass" or _is_repeat(thought, recent_thoughts):
                self._journal("[think]", "(quiet — nothing new to add)")
                return thought
            # Faithfulness gate (same standard as dreams): a thought claiming an
            # observation must trace to real evidence, or it's fabrication — not
            # stored (so it can't seed the next heartbeat), not pushed.
            if not _grounded_thought(thought, evidence_terms):
                self._journal("[think]", "(dropped an ungrounded observation claim)")
                return thought
            self.stm.append("thought", thought)
            self._journal("[think]", thought)
            if _notable(thought):        # only surface genuine questions/suggestions
                _push(self.cfg, "Lara", thought, tag="thought")
        return thought

    # -- sleep / dream ------------------------------------------------- #
    def _consolidate_group(self, items: List[dict], owner: str) -> Dict[str, Any]:
        """Consolidate ONE profile's events into memory OWNED by them (owner="" =
        shared household/ambient). Same faithfulness gate as before, applied per
        group so one person's activity never lands in another's — or the shared —
        memory. Returns the raw lists + stored counts for the caller to aggregate."""
        empty = {"lessons": [], "facts": [], "questions": [], "improvements": [],
                 "stored_lessons": 0, "stored_facts": 0, "summary": "", "ok": False}
        # Tolerate a valid-JSON-but-wrong-shape record (see think()): .get()+str()
        # so the digest can't KeyError/TypeError and abort the consolidation.
        # Dreams span a whole day, so the stamp keeps the date too
        # (iso[5:16] = "MM-DD HH:MM") — a consolidated memory that can't say
        # WHEN something happened invents a when.
        digest = "\n".join(
            f"[{r.get('kind', '?')} {str(r.get('iso', ''))[5:16]}] "
            f"{str(r.get('text', ''))[:200]}" for r in items[-120:])
        if not owner:
            # Ambient group: append MEASURED house activity (SQL counts over the
            # event log) so any 'pattern' the dream keeps traces to real
            # transitions — the digest is the faithfulness gate's source_terms,
            # so grounding starts here, structurally.
            try:
                from .index import SearchIndex
                pat = SearchIndex(self.cfg).house_patterns()
                if pat:
                    digest += "\n[measured] " + pat
            except Exception:
                pass
        who = ("this family member's" if owner else "the household's")
        prompt = (
            f"Below are {who} recent REAL events — observations, home-state "
            "changes, and conversation. Consolidate ONLY what these events "
            "actually support; do not add anything not present here.\n\n"
            "Return JSON:\n"
            "- 'lessons': durable, tentative observations grounded in the events "
            "(about the home, the person's patterns, or how ANVIL itself "
            "behaves). No invented specifics. No claims that you took an action.\n"
            "- 'profile_facts': things the person stated, or that are clearly "
            "true about them from the events.\n"
            "- 'questions': open questions worth exploring next.\n"
            "- 'improvements': concrete ideas to improve ANVIL's own code or "
            "behaviour (these are proposals — fine to be forward-looking).\n"
            "- 'summary': one honest sentence about this session.\n\n"
            "RECENT EVENTS:\n" + digest)
        data = {}
        try:
            r = self.router.complete([{"role": "user", "content": prompt}],
                                     system=self._sys(DREAM_SYS),
                                     schema=DREAM_SCHEMA, min_rung=0,
                                     max_tokens=800)
            data = _loads(r.completion.text)
            if not data:
                r = self.router.complete([{"role": "user", "content": prompt}],
                                         system=self._sys(DREAM_SYS),
                                         min_rung=0, max_tokens=800)
                data = _loads(r.completion.text) or {}
        except Exception as exc:
            self._journal("[dream]", f"consolidation skipped: {exc}")
            return empty

        # Faithfulness gate: durable memory (lessons + facts about the person)
        # must be grounded in the source events and must not claim ANVIL acted.
        # Questions/improvements are forward-looking, so they skip the gate.
        src_terms = set(re.findall(r"[a-z0-9]{4,}", digest.lower()))
        action_src = " ".join(str(r.get("text", "")) for r in items[-120:]
                              if r.get("kind") == "action").lower()
        action_terms = set(re.findall(r"[a-z0-9]{4,}", action_src))
        lessons = [x for x in data.get("lessons", [])
                   if isinstance(x, str) and len(x) > 6
                   and _faithful(x, src_terms, action_terms)]
        facts = [x for x in data.get("profile_facts", [])
                 if isinstance(x, str) and len(x) > 6
                 and _faithful(x, src_terms, action_terms)]
        questions = [x for x in data.get("questions", []) if isinstance(x, str) and len(x) > 4]
        improvements = [x for x in data.get("improvements", []) if isinstance(x, str) and len(x) > 6]

        # Persist defensively (a single locked/failed write must not abort the
        # dream), and attribute every note to THIS group's owner.
        stored_lessons = stored_facts = 0
        for l in lessons:
            try:
                self.ltm.write(l, type="project", tags=["lesson", "dream"], owner=owner)
                stored_lessons += 1
            except Exception:
                continue
        for f in facts:
            try:
                self.ltm.write(f, type="profile", tags=["about-you", "dream"], owner=owner)
                stored_facts += 1
            except Exception:
                continue
        return {"lessons": lessons, "facts": facts, "questions": questions,
                "improvements": improvements, "stored_lessons": stored_lessons,
                "stored_facts": stored_facts, "summary": data.get("summary", ""),
                "ok": True}

    def dream(self) -> Dict[str, Any]:
        items = self.stm.all()
        if not items:
            self._journal("[dream]", "nothing to consolidate")
            return {"consolidated": 0, "lessons": 0, "facts": 0,
                    "questions": 0, "proposals": 0}
        # Group events by the profile they belong to. Chat turns carry
        # meta.actor; ambient events (observations, home-state, Lara's own
        # thoughts) have none -> the shared "" group. Consolidating each group
        # separately, into memory owned by that person, keeps one family
        # member's activity out of another's (and out of the shared pool).
        groups: Dict[str, List[dict]] = {}
        for r in items:
            m = r.get("meta") or {}
            a = str(m.get("actor") or "") if isinstance(m, dict) else ""
            groups.setdefault(a, []).append(r)
        # Always do the shared/ambient group, then up to a few person groups
        # (cap bounds the per-dream LLM cost on a busy day). Person groups are
        # ordered LEAST-recently-consolidated first (persisted clock) — the old
        # alphabetical order silently starved the same 5th+ family member on
        # every single dream, forever.
        from .schedule import Schedule
        sched = Schedule()
        start_ts = time.time()               # the consolidation watermark
        people = [a for a in groups if a]
        people.sort(key=lambda a: -(sched.elapsed(f"dream-group:{a}") or float("inf")))
        order = [""] + people[:4]
        lessons, questions, improvements = [], [], []
        stored_lessons = stored_facts = 0
        summary = ""
        done_groups = []                     # groups whose consolidation SUCCEEDED
        for a in order:
            gi = groups.get(a)
            if not gi:
                continue
            res = self._consolidate_group(gi, a)
            if res.get("ok"):
                done_groups.append(a)
                sched.mark(f"dream-group:{a}")
            lessons += res["lessons"]
            questions += res["questions"]
            improvements += res["improvements"]
            stored_lessons += res["stored_lessons"]
            stored_facts += res["stored_facts"]
            summary = summary or res["summary"]
        data = {"summary": summary}
        # Cap the question flood (a dream once wrote 5 at a time), and don't
        # re-ask what's already sitting unanswered in STM.
        open_qs = [str(r.get("text", "")) for r in items
                   if r.get("kind") == "question"]
        stored_qs = 0
        for q in questions:
            if stored_qs >= 3 or _is_repeat(q, open_qs):
                continue
            try:
                self.stm.append("question", q)
                open_qs.append(q)
                stored_qs += 1
            except Exception:
                continue

        if improvements:
            try:
                PROPOSALS.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                body = ("# Dream self-improvement proposals — " +
                        datetime.now().isoformat(timespec="seconds") + "\n\n" +
                        "These are ideas ANVIL generated while consolidating memory. "
                        "Review before acting; nothing here was applied automatically.\n\n" +
                        "\n".join(f"- {i}" for i in improvements) + "\n")
                (PROPOSALS / f"dream-{stamp}.md").write_text(body, encoding="utf-8")
                _prune_proposals(PROPOSALS)
            except Exception:
                pass

        # housekeeping: decay LTM, then RECONSOLIDATE it against the day's
        # activity (strengthen what proved relevant, forget what faded), evolve
        # persona, and prune STM (sleep clears the day's buffer).
        try:
            self.ltm.consolidate()
        except Exception:
            pass
        # Self-healing: vector any notes that missed their embedding (endpoint
        # down at write time, or written by an embedder-less store) so semantic
        # recall covers the whole store, not just the lucky notes.
        try:
            healed = self.ltm.backfill_embeddings()
            if healed:
                self._journal("[dream]", f"backfilled {healed} note embedding(s)")
        except Exception:
            pass
        # Keep the search index warm off-path (index.py): conversation turns and
        # the journal sync here so search_chats never pays the first-index cost
        # on a live turn. Mtime-diffed — a no-op when nothing changed.
        try:
            from .index import SearchIndex
            ix = SearchIndex(self.cfg)
            ix.sync_turns(Path(getattr(self.cfg, "conversations_dir", "conversations")))
            ix.sync_journal(self.journal)
        except Exception:
            pass
        # Cheapest insurance: once a day, snapshot the family's irreplaceable data
        # (memory + skills + chats) so a disk hiccup or bad edit is recoverable.
        try:
            from . import backup
            if backup.maybe_daily(self.cfg):
                self._journal("[dream]", "backed up memory + skills + chats")
        except Exception:
            pass
        # Keep the family-docs RAG index fresh so new papers are answerable.
        try:
            from . import docs
            root = docs.docs_dir(self.cfg)
            if root.exists() and any(root.rglob("*")):
                def _embed(t):
                    return self.router.providers["ollama_local"].embed(self.cfg.embed_model, t)
                emb = _embed if getattr(self.cfg, "use_embeddings", True) else None
                r = docs.DocStore(self.cfg, embedder=emb).reindex()
                if r.get("indexed"):
                    self._journal("[dream]", f"indexed {r['indexed']} family doc(s)")
        except Exception:
            pass
        # Refresh the operator card when new facts about them landed this dream
        # (or none exists yet) — the distilled 'who I'm helping' synthesis. With
        # a family, build ONE CARD PER PROFILE from each person's own facts, each
        # scoped by cfg._actor so cards never blend family members together.
        try:
            from . import usercard, profiles
            actors = ([p.name for p in profiles.load(self.cfg).values()]
                      if profiles.auth_on(self.cfg) else [""])
            built = 0
            for actor in actors:
                try:
                    self.cfg._actor = actor
                except Exception:
                    pass
                if stored_facts or not usercard.load(self.cfg):
                    if usercard.build(self.cfg, self.router, self.ltm):
                        built += 1
            try:
                self.cfg._actor = ""      # leave the background cfg neutral
            except Exception:
                pass
            if built:
                self._journal("[dream]", f"refreshed operator card ({built})")
        except Exception:
            pass
        refl = {"strengthened": 0, "forgotten": 0}
        try:
            recent_texts = [str(r.get("text", "")) for r in items[-80:]] + lessons
            refl = self.ltm.reflect(
                recent_texts,
                floor=float(getattr(self.cfg, "ltm_forget_floor", 0.08)))
        except Exception:
            pass
        try:
            persona.evolve(self._persona, self.router, self.ltm)
        except Exception:
            pass
        # Watermark prune: drop ONLY what was successfully consolidated. A group
        # whose model call failed keeps its events for the next dream (the old
        # unconditional clear deleted a whole day on an Ollama outage), and
        # anything that arrived mid-dream survives untouched.
        dropped = self.stm.clear_consolidated(done_groups, before=start_ts, keep=15)

        summary = data.get("summary") or (
            f"consolidated {len(items)} items: {stored_lessons} lessons, "
            f"{stored_facts} facts, {len(questions)} questions")
        self._journal("[dream]", summary +
                      f" | LTM +{refl.get('strengthened', 0)} strengthened, "
                      f"-{refl.get('forgotten', 0)} forgotten")
        # Surface a dream only when it actually learned something durable — a
        # top lesson if there is one, otherwise stay quiet (no "consolidated N
        # items" noise).
        if stored_lessons and lessons:
            _push(self.cfg, "Lara reflected", lessons[0], tag="dream")
        return {"consolidated": len(items), "lessons": stored_lessons,
                "facts": stored_facts, "questions": stored_qs,
                "proposals": len(improvements), "pruned": dropped,
                "strengthened": refl.get("strengthened", 0),
                "forgotten": refl.get("forgotten", 0),
                "summary": summary}

    # -- autopilot: think & dream on its own, all day ------------------ #
    def autopilot(self, interval_min: Optional[int] = None,
                  max_ticks: Optional[int] = None,
                  sleep_fn=time.sleep, schedule=None) -> int:
        """Run the mind unattended: a spontaneous thought every
        ``heartbeat_interval_min`` minutes; a consolidation ``dream`` when STM
        fills to ``dream_after`` items (or ``dream_max_age_hours`` passes); and
        the deep-sleep dev stages (triage / promote / issue-work / self-dev)
        WORK-driven each quiet tick — see ``_deep_sleep``. Designed to run in a
        daemon thread while the server is up.

        Cadence is wall-clock and persisted (anvil/schedule.py), not tick
        counters — a restart resumes the schedule instead of resetting it.

        Every tick is fully guarded — a model timeout or a locked file logs to
        the journal and the loop keeps breathing. ``max_ticks``/``sleep_fn``/
        ``schedule`` exist for testing; in production it loops for the life of
        the process.
        """
        from .schedule import Schedule
        interval = int(interval_min if interval_min is not None
                       else getattr(self.cfg, "heartbeat_interval_min", 15))
        dream_after = int(getattr(self.cfg, "dream_after", 40))
        dream_age_h = float(getattr(self.cfg, "dream_max_age_hours", 6.0))
        sched = schedule if schedule is not None else Schedule()
        self._journal("[autopilot]",
                      f"awake — heartbeat {interval}m, dream at {dream_after} STM "
                      f"items or {dream_age_h:g}h, dev stages work-driven")
        cooldown = int(getattr(self.cfg, "chat_quiet_cooldown_min", 5)) * 60
        tick = 0
        while max_ticks is None or tick < max_ticks:
            tick += 1
            try:
                # Sensing is model-free — always run it, even mid-conversation.
                if getattr(self.cfg, "sense_house", True):
                    self.sense_house()          # look at the home, note what changed
                    self.sense_weather()        # new severe alert -> record + push
                # But thinking/dreaming/self-dev all hit the single local model.
                # If the family was just chatting, yield the GPU to them and skip
                # this tick's model work (STM keeps filling; it consolidates once
                # the house goes quiet).
                if cooldown > 0 and self._chat_active(cooldown):
                    self._journal("[autopilot]", "yielding — operator was just chatting")
                else:
                    self.think()
                    # Dream when there's enough to consolidate, with a max-age
                    # backstop so a quiet house still consolidates a few times a day.
                    if (self.stm.size() >= dream_after
                            or sched.due("dream", hours=dream_age_h)):
                        self.dream()
                        sched.mark("dream")
                    self._deep_sleep(sched)     # work-driven; cheap when idle
            except Exception as exc:
                try:
                    self._journal("[autopilot]", f"tick {tick} error: {exc}")
                except Exception:
                    pass
            if max_ticks is not None and tick >= max_ticks:
                break
            # Sense BETWEEN thinks (review 2.3 interim): sensing is model-free,
            # so it must not ride the think cadence — a door opening at 2am was
            # seen up to a full heartbeat (35 min) late, or not at all if it
            # closed again in between. Slice the inter-tick sleep and diff the
            # house each slice; thinking still happens once per heartbeat.
            sense_s = max(30, int(getattr(self.cfg, "sense_interval_s", 60)))
            remaining = max(60, interval * 60)
            while remaining > 0:
                step = min(sense_s, remaining)
                sleep_fn(step)
                remaining -= step
                if remaining > 0 and getattr(self.cfg, "sense_house", True):
                    try:
                        self.sense_house()
                    except Exception:
                        pass
        return tick

    def _chat_active(self, cooldown_s: float) -> bool:
        """Did the operator chat within the cooldown? Read from STM (the shared
        on-disk event stream), so the autopilot thread sees chats logged by the
        separate request threads — no in-process signal needed."""
        now = time.time()
        for r in self.stm.recent(12):
            if r.get("kind") == "chat" and (now - float(r.get("ts", 0))) < cooldown_s:
                return True
        return False

    def _deep_sleep(self, schedule=None) -> Optional[dict]:
        """The dev stages of sleep, WORK-driven: each stage runs when there is
        actually work for it (checked cheaply, model-free), throttled by its own
        wall-clock dial persisted across restarts (anvil/schedule.py):

          * triage      — when NEW incidents await, at most every triage_debounce_min
          * promote     — when `test` is ahead of `main`, at most every
                          promote_debounce_min (so batches can accumulate)
          * issue-work  — every quiet tick; "is there work?" is one Gitea API call
          * self-dev    — the speculative improve-the-harness crawl, the only truly
                          periodic job: every selfdev_interval_hours

        Called every quiet autopilot tick — cheap when idle. Fully guarded and
        capped: a failure or a no-op never disturbs the loop, git guarantees the
        worst case, and everything stays behind the chat-cooldown GPU yield."""
        from .schedule import Schedule
        sched = schedule if schedule is not None else Schedule()
        out = {}
        # Curator: age out skills nothing recalls anymore (deterministic, no
        # model call) so the flywheel's output doesn't sprawl into clutter.
        if sched.due("curator", hours=24):
            try:
                from .skills import SkillStore
                pruned = SkillStore(self.cfg).prune(
                    stale_days=int(getattr(self.cfg, "skill_stale_days", 30)),
                    archive_days=int(getattr(self.cfg, "skill_archive_days", 90)))
                sched.mark("curator")
                if pruned.get("stale") or pruned.get("archived"):
                    self._journal("[curator]", f"skills {pruned}")
            except Exception:
                pass
        # NEVER launch the forge from inside the test suite: `anvil doctor` runs tests
        # that reach deep sleep, and the forge's gate shells out to `anvil doctor` — so
        # real self-dev here would spawn doctor -> forge -> doctor, a fork bomb.
        # (selfdev/issuework also self-guard; this keeps _deep_sleep from even calling.)
        import os as _os
        if _os.environ.get("ANVIL_IN_DOCTOR"):
            return None
        # Self-awareness: when new process incidents await, cluster + classify them
        # and file harness-bug issues for the ones that are our own fault.
        try:
            from . import introspect
            if (introspect.pending()
                    and sched.due("triage", minutes=int(getattr(
                        self.cfg, "triage_debounce_min", 30)))):
                t = introspect.triage(
                    self.cfg, logger=lambda m: self._journal("[introspect]", str(m)))
                sched.mark("triage")
                self._journal("[deep-sleep]", f"introspect: {t}")
                out["triage"] = t
        except Exception:
            pass
        # Phase 3: when `test` is ahead, promote it (opens a reviewed PR; merges only
        # if auto_promote). Debounced so several fixes can ride one release.
        try:
            from . import promote as _promote
            if (_promote.pending(self.cfg)
                    and sched.due("promote", minutes=int(getattr(
                        self.cfg, "promote_debounce_min", 20)))):
                p = _promote.promote(
                    self.cfg, logger=lambda m: self._journal("[promote]", str(m)))
                sched.mark("promote")
                self._journal("[deep-sleep]", f"promote: {p}")
                out["promote"] = p
        except Exception:
            pass
        try:
            # Work the Gitea issue queue (assess -> clarify / push-back / fix on
            # `test`, in the open). Every quiet tick — an issue filed at 9am gets
            # picked up within one heartbeat, not on a multi-hour fuse. When it
            # actually worked something, that's enough model load for this tick.
            if getattr(self.cfg, "issue_work", False):
                from . import issuework
                r = issuework.IssueWorker(
                    self.cfg,
                    logger=lambda m: self._journal("[issuework]", str(m))).run_once()
                if r != "no actionable issues":
                    self._journal("[deep-sleep]", f"issue-work: {r}")
                    sched.mark("issuework")
                    out["issue_work"] = r
                    return out
            # Speculative self-dev: the improve-the-harness crawl. Periodic, not
            # event-driven — there's always *something* it could try, so the dial
            # is how often we let it (daily cap still enforced inside).
            if (getattr(self.cfg, "self_dev_in_sleep", True)
                    and sched.due("selfdev", hours=float(getattr(
                        self.cfg, "selfdev_interval_hours", 12.0)))):
                from . import selfdev
                res = selfdev.run_one_cycle(
                    self.cfg, logger=lambda m: self._journal("[selfdev]", str(m)))
                sched.mark("selfdev")
                self._journal("[deep-sleep]", f"self-dev: {res}")
                out["selfdev"] = res
        except Exception as exc:
            # A caught exception in the sleep loop is almost certainly OUR bug — record it.
            try:
                from . import introspect
                introspect.record("exception", "mind._deep_sleep",
                                  f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            try:
                self._journal("[deep-sleep]", f"dev stage failed: {exc}")
            except Exception:
                pass
        return out or None

    # -- circadian loop ------------------------------------------------ #
    def pulse(self, minutes: int = 480, interval: int = 10,
              dream_every: int = 6) -> None:
        end = time.time() + minutes * 60
        tick = 0
        dream_after = int(getattr(self.cfg, "dream_after", 40))
        print(f"[anvil pulse] heartbeat every {interval} min for {minutes} min; "
              f"dream every {dream_every} ticks or {dream_after} STM items")
        while True:
            tick += 1
            thought = self.think()
            print(f"  tick {tick} @ {datetime.now().strftime('%H:%M')} 💭 {thought[:80]}")
            if tick % max(1, dream_every) == 0 or self.stm.size() >= dream_after:
                d = self.dream()
                print(f"  💤 dream: {d.get('summary', d)}")
            if time.time() >= end:
                break
            time.sleep(max(5, interval * 60))
        print(f"[anvil pulse] rested after {tick} ticks")


def _prune_proposals(directory: Path, keep: int = 20) -> int:
    """Cap the dream-proposal archive so it can't grow without bound.

    ``dream()`` writes a fresh ``dream-*.md`` every consolidation; over a
    long-running ANVIL these accumulate forever. Keep only the most recent
    ``keep`` (the timestamp filenames sort chronologically) and drop the rest.
    Best-effort and scoped to ``dream-*.md`` only: a locked/undeletable file or
    an unrelated proposals file (e.g. soak triage notes) is left untouched, and
    a failed unlink never aborts the dream / crashes the pulse loop.
    """
    if keep < 0:
        return 0
    try:
        stale = sorted(directory.glob("dream-*.md"))[:-keep] if keep \
            else sorted(directory.glob("dream-*.md"))
    except OSError:
        return 0
    dropped = 0
    for old in stale:
        try:
            old.unlink()
            dropped += 1
        except OSError:
            continue
    return dropped


def _loads(text: str) -> dict:
    t = (text or "").strip()
    import re
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.DOTALL)
    if m:
        t = m.group(1)
    candidates = [t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""]
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    # Local models routinely emit trailing commas (`[1, 2,]` / `{"a": 1,}`),
    # which are invalid JSON and would otherwise lose a whole dream
    # consolidation. As a last resort, strip a comma before a closing ] or }
    # and retry — same lenient-fallback discipline as config loading.
    for cand in candidates:
        if not cand:
            continue
        repaired = re.sub(r",(\s*[}\]])", r"\1", cand)
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


def record(cfg, kind: str, text: str, meta: Optional[dict] = None) -> None:
    """Lightweight STM append for callers that don't want a full Mind."""
    try:
        ShortTerm(Path(cfg.memory_dir) / "short_term.jsonl",
                  cap=int(getattr(cfg, "stm_cap", 300))).append(kind, text, meta)
    except Exception:
        pass


def read_stm(cfg, n: int = 40) -> List[dict]:
    st = ShortTerm(Path(cfg.memory_dir) / "short_term.jsonl")
    # Carry meta (esp. meta.actor) so the Pulse view can scope the thought stream
    # to the viewing profile. The UI renders only kind/text, never meta.
    return [{"iso": e.get("iso", ""), "kind": e.get("kind", ""),
             "text": e.get("text", ""), "meta": e.get("meta") or {}}
            for e in st.recent(n)]


def _tail_lines(path: Path, n: int) -> List[str]:
    """Return roughly the last ``n`` lines without loading the whole file.

    The journal grows unbounded over a long-running ANVIL (a line per heartbeat
    and dream), but the UI only ever wants the tail. Read a bounded window from
    the end instead of pulling the entire file into memory on every request —
    same tail-read discipline as ``ShortTerm.recent()``.
    """
    if n <= 0 or not path.exists():
        return []
    try:
        size = path.stat().st_size
        # A window generous enough to hold n lines at any realistic length, but
        # capped so a huge journal stays cheap to tail.
        window = min(size, max(4096, n * 512))
        with path.open("rb") as fh:
            fh.seek(size - window)
            raw = fh.read()
    except OSError:
        return path.read_text("utf-8", "replace").splitlines()[-n:]
    lines = raw.decode("utf-8", "replace").splitlines()
    # If we started mid-file, the first line is likely a partial fragment from a
    # line we sliced through — drop it so only whole lines are returned.
    if window < size and len(lines) > 1:
        lines = lines[1:]
    # The window is sized for typical line lengths; very long lines (a verbose
    # dream summary, say) can leave it holding fewer than n whole lines. If we
    # didn't cover the whole file, fall back to a full read so the caller still
    # gets the true last n — same fallback discipline as ShortTerm.recent().
    if window < size and len(lines) < n:
        return path.read_text("utf-8", "replace").splitlines()[-n:]
    return lines[-n:]


def read_journal(cfg, n: int = 40) -> List[str]:
    return _tail_lines(Path(cfg.memory_dir) / "journal.md", n)
