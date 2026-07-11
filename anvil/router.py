"""The escalation engine + cost ledger — the heart of "smart token use".

The router picks the cheapest rung that can plausibly do a step, runs it, and
escalates one rung only when a concrete signal fires (low self-reported
confidence, schema-validation failure, an explicit floor, or a critic veto).
Every call is logged to ``ledger.jsonl`` with an estimated cost so the ladder
can be tuned from real data and a daily spend cap can throttle escalation.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .providers import (Completion, ContextOverflow, ProviderError,
                        build_providers, is_transient, jittered_backoff)


def _shrink_for_retry(messages: List[dict], cap: int = 4000) -> List[dict]:
    """Make an over-long message list fit, for a one-shot retry after a context
    overflow. Drops the middle of the history (keeps the oldest for continuity
    and the most recent turns) and caps each message's text. The LAST message
    keeps its TAIL — the actual task lives at the end of the primed user turn,
    so trimming from the front would cut the question itself."""
    if not messages:
        return messages
    msgs = messages if len(messages) <= 5 else [messages[0]] + messages[-4:]
    out, last = [], len(msgs) - 1
    for i, m in enumerate(msgs):
        c = m.get("content")
        if isinstance(c, str) and len(c) > cap:
            m = dict(m)
            m["content"] = (c[-cap:] if i == last else c[:cap]) + "\n…[trimmed]"
        out.append(m)
    # De-orphan tool calls/responses broken by dropping the middle. Keeping
    # messages[0] + the tail can sever an assistant tool_calls turn from its
    # 'tool' responses (or leave a leading 'tool' response with no producing
    # call) — both of which the native tool endpoints reject with a 400, turning
    # a recoverable overflow into a hard failure on exactly the tool-heavy
    # sessions most likely to overflow. Same guard _compact_live applies.
    while len(out) > 1 and out[1].get("role") == "tool":
        del out[1]                    # tool response whose call was dropped
    for i, m in enumerate(out):
        if m.get("tool_calls"):
            nxt = out[i + 1] if i + 1 < len(out) else None
            if not (nxt and nxt.get("role") == "tool"):
                out[i] = dict(m)
                out[i].pop("tool_calls", None)
    return out

def _estimate_prompt_tokens(messages: List[dict], system: str = "") -> int:
    """Conservative prompt-size estimate (chars/4 ≈ English tokens). Used for
    the pre-flight window check and to detect SILENT truncation: local Ollama
    trims an oversized prompt from the front and returns 200, so 'measured ≪
    estimated' is the only tell that the system prompt fell out the window."""
    total = len(system or "")
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            total += sum(len(str(p.get("text", ""))) for p in c
                         if isinstance(p, dict))
    return max(1, total // 4)


# spent_today() used to re-read and re-parse the ENTIRE ledger.jsonl on every
# single model call — a per-request tax that grows forever on a 24/7 install.
# Cache the day's total per ledger path, updated incrementally by record();
# invalidated when the date rolls over or the file changes size underneath us
# (external truncation/edit). Module-level so every CostLedger instance (one is
# built per request) shares it.
_SPENT_LOCK = threading.Lock()
_SPENT_CACHE: Dict[str, Dict[str, Any]] = {}   # path -> {date, total, size}

# Rotation: the ledger gains a line per model call, forever. Past ~2MB, trim to
# the most recent lines — but never drop any of TODAY's records, or a restart
# would under-count spend against the daily cap.
_LEDGER_MAX_BYTES = 2_000_000
_LEDGER_KEEP_LINES = 4000


@dataclass
class RouteResult:
    completion: Completion
    rung_index: int
    rung_name: str
    escalations: List[str] = field(default_factory=list)
    est_cost_usd: float = 0.0


class CostLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rung, comp: Completion, trigger: str = "",
               plane: str = "chat") -> float:
        cost = self._estimate(rung, comp)
        rec = {
            "ts": time.time(),
            "date": date.today().isoformat(),
            "rung": rung.name,
            "provider": comp.provider,
            "model": comp.model,
            "in_tok": comp.input_tokens,
            "cached_tok": comp.cached_input_tokens,
            "out_tok": comp.output_tokens,
            "est_cost": round(cost, 6),
            "trigger": trigger,
            "plane": plane,            # chat | dreams | selfdev | hive
        }
        if getattr(comp, "web_searches", 0):
            rec["searches"] = comp.web_searches
        if (getattr(comp, "raw", None) or {}).get("batched"):
            rec["batched"] = True      # priced at the 50% batch rate

        with _SPENT_LOCK:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            # Keep the day-total cache coherent with what we just appended.
            key = str(self.path)
            c = _SPENT_CACHE.get(key)
            if c and c.get("date") == rec["date"]:
                c["total"] += rec["est_cost"]
                try:
                    c["size"] = self.path.stat().st_size
                except OSError:
                    _SPENT_CACHE.pop(key, None)
            self._rotate(rec["date"], key)
        return cost

    def _rotate(self, today: str, key: str) -> None:
        """Trim an oversized ledger, keeping the recent tail AND all of today
        (cap accounting must survive a restart). Called under _SPENT_LOCK."""
        try:
            if self.path.stat().st_size <= _LEDGER_MAX_BYTES:
                return
            lines = self.path.read_text("utf-8", "replace").splitlines()
            marker = f'"date": "{today}"'
            first_today = next((i for i, ln in enumerate(lines) if marker in ln),
                               len(lines))
            start = max(0, min(len(lines) - _LEDGER_KEEP_LINES, first_today))
            if start <= 0:
                return
            from . import config as cfgmod
            cfgmod.atomic_write(self.path, "\n".join(lines[start:]) + "\n")
            c = _SPENT_CACHE.get(key)
            if c:
                c["size"] = self.path.stat().st_size
        except OSError:
            pass

    def spent_today_planes(self, planes) -> float:
        """Today's spend for a SUBSET of planes (issue #99: the background
        soft cap must measure background spend, not the whole day — foreground
        chat was starving the self-dev/dream/hive budget). No cache: callers
        are low-rate background dispatches and the ledger file is size-capped."""
        if not self.path.exists():
            return 0.0
        today = date.today().isoformat()
        total = 0.0
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("date") == today and rec.get("plane") in planes:
                        total += float(rec.get("est_cost", 0.0) or 0.0)
        except OSError:
            return 0.0
        return total

    @staticmethod
    def _estimate(rung, comp: Completion) -> float:
        if not rung.is_paid:
            return 0.0
        fresh_in = max(0, comp.input_tokens - comp.cached_input_tokens)
        cost = (
            fresh_in / 1e6 * rung.cost_in
            + comp.cached_input_tokens / 1e6 * rung.cache_read
            + comp.output_tokens / 1e6 * rung.cost_out
        )
        # Message Batches price every token (fresh, cached, output) at 50%.
        if (getattr(comp, "raw", None) or {}).get("batched"):
            cost *= 0.5
        # Anthropic server-side web search: $10 per 1,000 searches (never
        # discounted — it's a per-search fee, not a token price).
        return cost + getattr(comp, "web_searches", 0) * 0.01

    def spent_today(self) -> float:
        if not self.path.exists():
            return 0.0
        today = date.today().isoformat()
        key = str(self.path)
        try:
            size = self.path.stat().st_size
        except OSError:
            size = -1
        with _SPENT_LOCK:
            c = _SPENT_CACHE.get(key)
            if c and c.get("date") == today and c.get("size") == size:
                return c["total"]
        # Cache miss (first call today, or the file changed underneath us):
        # do the full scan once, then serve from the cache.
        total = 0.0
        for line in self.path.read_text("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("date") == today:
                total += rec.get("est_cost", 0.0)
        with _SPENT_LOCK:
            _SPENT_CACHE[key] = {"date": today, "total": total, "size": size}
        return total

    def spent_by_plane(self) -> Dict[str, float]:
        """Today's spend split by plane (chat/dreams/selfdev/hive) — for the
        PWA governance line. Not cached (called rarely, from the dashboard)."""
        out: Dict[str, float] = {}
        if not self.path.exists():
            return out
        today = date.today().isoformat()
        for line in self.path.read_text("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("date") == today:
                p = rec.get("plane", "chat")
                out[p] = round(out.get(p, 0.0) + rec.get("est_cost", 0.0), 6)
        return out

    def spent_last_days(self, days: int = 30) -> float:
        """Rolling N-day spend total — the monthly-cap denominator. Cheap tail
        read is unnecessary; the ledger is small at household rates."""
        if not self.path.exists() or days <= 0:
            return 0.0
        from datetime import timedelta
        cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
        total = 0.0
        for line in self.path.read_text("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("date", "")) >= cutoff:
                total += float(rec.get("est_cost", 0.0) or 0.0)
        return round(total, 6)


# Minimal JSON-schema-shaped response used to read back a confidence signal.
CONFIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["answer", "confidence"],
}


_CAP_NOTIFIED: Dict[str, str] = {}   # ledger path -> date we last pushed a hard-cap alert


# ---- rung health: a tiny three-state circuit breaker (review 1.3) ---------- #
# Routers are built per request, so without module-level health memory every
# call relearned "this provider is down" by paying the full timeout ladder.
# Keyed provider:model (same failure domain), guarded like _SPENT_CACHE.
_HEALTH_LOCK = threading.Lock()
_RUNG_HEALTH: Dict[str, Dict[str, float]] = {}   # key -> {"fails": n, "opened": ts}
_BREAK_AFTER = 3          # consecutive transport failures -> open
_BREAK_FOR_S = 60.0       # open this long, then half-open (one probe allowed)


def _breaker_state(key: str) -> str:
    """closed (healthy) | open (skip if possible) | half-open (probe carefully)."""
    if os.environ.get("ANVIL_IN_DOCTOR"):
        return "closed"                  # hermetic: tests must not cross-poison
    with _HEALTH_LOCK:
        rec = _RUNG_HEALTH.get(key)
        if not rec or rec.get("fails", 0) < _BREAK_AFTER:
            return "closed"
        age = time.time() - rec.get("opened", 0)
        return "open" if age < _BREAK_FOR_S else "half-open"


def _breaker_failure(key: str) -> None:
    if os.environ.get("ANVIL_IN_DOCTOR"):
        return
    with _HEALTH_LOCK:
        rec = _RUNG_HEALTH.setdefault(key, {"fails": 0, "opened": 0.0})
        rec["fails"] = rec.get("fails", 0) + 1
        if rec["fails"] >= _BREAK_AFTER:
            rec["opened"] = time.time()      # (re)open — half-open probe failed too


def _breaker_success(key: str) -> None:
    with _HEALTH_LOCK:
        _RUNG_HEALTH.pop(key, None)


def _preflight_ok(base_url: str, timeout: float = 5.0) -> bool:
    """Cheap TCP connect check before probing a suspect rung — a black-holed
    host costs 5s here instead of the full request timeout (120s)."""
    try:
        import socket
        from urllib.parse import urlsplit
        u = urlsplit(base_url or "")
        host = u.hostname or "localhost"
        port = u.port or (443 if u.scheme == "https" else 80)
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except Exception:
        return False


class Router:
    # Planes whose paid-rung spend is throttled at the SOFT (background) cap so a
    # runaway can't exhaust the day's quota. Foreground "chat" is protected up to
    # the hard cap.
    _BACKGROUND = frozenset({"dreams", "selfdev", "hive", "scribe"})
    # Planes whose paid calls may route through the Message Batches API (50%
    # price, minutes of latency). Deliberately NOT "hive": drones answer live
    # chat turns, and never "chat" — someone is waiting on those.
    _BATCH_PLANES = frozenset({"dreams", "selfdev", "scribe"})

    def __init__(self, cfg, providers: Optional[Dict[str, Any]] = None,
                 plane: str = "chat"):
        self.cfg = cfg
        self.providers = providers or build_providers(cfg)
        self.ledger = CostLedger(cfg.ledger_path)
        self.plane = plane or "chat"

    def _use_batch(self, provider, rung) -> bool:
        """Route this call through the async Message Batches API? Only when
        the operator opted in (batch_background), only on latency-insensitive
        background planes, only on paid rungs of a provider that supports it.
        Foreground chat/approvals and hive drones always stay on the live
        API — batching structurally cannot touch a family-facing call."""
        return (bool(getattr(self.cfg, "batch_background", False))
                and self.plane in self._BATCH_PLANES
                and bool(getattr(rung, "is_paid", False))
                and hasattr(provider, "chat_batch"))

    def _cap_for_plane(self) -> float:
        hard = float(getattr(self.cfg, "daily_cost_cap_usd", 5.0))
        if self.plane in self._BACKGROUND:
            soft = float(getattr(self.cfg, "background_cost_cap_usd", hard))
            return min(soft, hard)
        return hard

    def _notify_hard_cap(self, spent: float) -> None:
        """Tell the operator ONCE per day when foreground work hits the hard cap
        and drops to local-only — so a silent quality drop doesn't surprise them."""
        try:
            key = str(self.cfg.ledger_path)
            today = date.today().isoformat()
            if _CAP_NOTIFIED.get(key) == today:
                return
            _CAP_NOTIFIED[key] = today
            from . import push
            push.notify(self.cfg, "Cloud budget reached",
                        f"Spent ${spent:.2f} today — Lara is on local models "
                        "until tomorrow. Raise the cap in Server settings if needed.",
                        url="/", tag="budget")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def complete(self, messages: List[dict], *, system: Optional[str] = None,
                 min_rung: int = 0, max_rung: Optional[int] = None,
                 schema: Optional[dict] = None,
                 want_confidence: bool = False,
                 tools: Optional[List[dict]] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 2048,
                 on_token=None, cancel=None, think=None,
                 on_think=None) -> RouteResult:
        """Run a step, escalating up the ladder on concrete failure signals."""
        max_rung = len(self.cfg.ladder) - 1 if max_rung is None else max_rung
        escalations: List[str] = []
        rung_i = max(min_rung, 0)
        last_exc: Optional[Exception] = None
        last_result: Optional[RouteResult] = None

        eff_schema = schema or (CONFIDENCE_SCHEMA if want_confidence else None)
        # Cloud-proxied models often IGNORE response_format (json_schema/strict) and
        # emit prose, which then fails to parse and silently misfires — e.g. a reviewer
        # that plainly said "approve" reads as a reject. When the CALLER asked for a
        # schema, also instruct the model in-band to return ONLY matching JSON. Scoped
        # to an explicit schema, so normal prose answers and the confidence read are
        # untouched.
        sys_for_call = system
        if eff_schema is not None:
            # Review 2.9: applies to EVERY effective schema — the confidence
            # read used to get neither server enforcement (native path) nor
            # this insurance, so its parse failures escalated to paid rungs.
            sys_for_call = (system or "") + (
                "\n\nIMPORTANT: Respond with ONLY a single valid JSON object matching "
                "this schema. No prose, no markdown, no code fences:\n"
                + json.dumps(eff_schema))
        overflow_retried = False
        degraded = False                       # cap-degrade to a free rung fired
        rung_attempts: Dict[int, int] = {}     # transient retries used per rung
        # Attempt-scoped token sink (review 2.7): a retry or escalation after
        # tokens already streamed used to CONCATENATE the new attempt onto the
        # doomed partial in the live chat bubble. Wrap the sink to count what
        # was emitted; before any re-attempt, tell it to reset (the server
        # clears the partial; the UI redraws clean). Tokens reach the user
        # from at most ONE attempt.
        emitted = {"n": 0}
        caller_sink = on_token
        if caller_sink is not None:
            def on_token(delta, _cs=caller_sink):
                emitted["n"] += 1
                _cs(delta)

        def _reset_sink():
            if emitted["n"] and caller_sink is not None:
                try:
                    getattr(caller_sink, "reset", lambda: None)()
                except Exception:
                    pass
                emitted["n"] = 0

        while rung_i <= max_rung:
            rung = self.cfg.rung(rung_i)
            key = f"{rung.provider}:{rung.model}"
            # Tiered cap guard: refuse to *start* a paid rung once over the cap
            # that applies to THIS plane. Background work (dreams/self-dev/hive)
            # stops at the soft cap so it can't starve foreground chat, which runs
            # up to the hard daily cap (then drops to local + notifies once).
            if rung.is_paid:
                # Background planes are throttled by BACKGROUND spend (#99) —
                # the operator's foreground day must not starve the self-dev/
                # dream budget. Foreground keeps the whole-day total: the hard
                # daily cap is about the household's total bill.
                spent = (self.ledger.spent_today_planes(self._BACKGROUND)
                         if self.plane in self._BACKGROUND
                         else self.ledger.spent_today())
                cap = self._cap_for_plane()
                # Rolling monthly ceiling (the metered-Claude guard): if the last
                # 30 days crossed it, treat this paid rung exactly like a daily
                # cap-hit — degrade to free, notify once. Keeps the bill bounded.
                mcap = float(getattr(self.cfg, "monthly_cost_cap_usd", 0.0) or 0.0)
                if mcap > 0 and self.ledger.spent_last_days(30) >= mcap:
                    spent = max(spent, cap)      # force the cap-hit branch below
                    escalations.append(f"month-cap@{rung.name}")
                if spent >= cap:
                    escalations.append(f"cap-hit@{rung.name}:{self.plane}")
                    if (self.plane not in self._BACKGROUND
                            and spent >= float(getattr(self.cfg,
                                              "daily_cost_cap_usd", 5.0))):
                        self._notify_hard_cap(spent)
                    # A paid FLOOR (min_rung) can skip past every free rung. If
                    # we haven't produced any result yet, degrade to the highest
                    # reachable free rung below us instead of aborting the whole
                    # turn with a misleading 'no model reachable' error — a local
                    # model can still answer. Degrade at MOST once: if that free
                    # rung then fails (e.g. local Ollama down), rung_i bumps back
                    # up to this capped paid floor — without the guard we'd
                    # degrade to the same failing free rung again and spin
                    # forever (free rung fails -> bump to paid -> cap -> degrade
                    # -> ...), hanging the request thread with no exit.
                    if last_result is None and not degraded:
                        free_i = next(
                            (i for i in range(rung_i - 1, -1, -1)
                             if not self.cfg.rung(i).is_paid), None)
                        if free_i is not None:
                            degraded = True
                            escalations.append(
                                f"degrade-to-free@{self.cfg.rung(free_i).name}")
                            rung_i = free_i
                            continue
                    break               # stay at the highest free rung we have
            provider = self.providers.get(rung.provider)
            if provider is None:
                # A rung references a provider we can't build (e.g. an anthropic
                # rung but ANTHROPIC_API_KEY was pulled). Skip it gracefully so
                # the ladder degrades to whatever IS configured, never crashes.
                escalations.append(f"no-provider@{rung.name}:{rung.provider}")
                if rung_i < max_rung:
                    rung_i += 1
                    _reset_sink()
                    continue
                raise ProviderError(
                    f"no provider '{rung.provider}' for rung '{rung.name}' "
                    "(is the API key set?)")
            # Pre-flight window budget (review 2.8): finally USE rung.max_context.
            # Compact deterministically BEFORE dispatch instead of letting the
            # provider silently truncate the head (or 400 on the cloud side).
            maxctx = int(getattr(rung, "max_context", 0) or 0)
            if maxctx and not overflow_retried:
                est = _estimate_prompt_tokens(messages, sys_for_call)
                if est > maxctx * 0.9:
                    overflow_retried = True
                    messages = _shrink_for_retry(messages)
                    escalations.append(f"ctx-preflight-compacted@{rung.name}")
            # Circuit breaker (review 1.3): a rung that just failed repeatedly is
            # SKIPPED instantly instead of re-paying the full connect-timeout tax
            # on every call — the old behavior relearned "cloud is down" from
            # scratch, ~6 minutes per turn, forever. The LAST reachable rung is
            # always probed (never strand a turn with zero attempts), but through
            # a 5s connect preflight and without in-rung retries, so probing a
            # black-holed host costs seconds, not minutes.
            state = _breaker_state(key)
            probing = state != "closed"
            if state == "open" and rung_i < max_rung:
                escalations.append(f"breaker-open@{rung.name}")
                rung_i += 1
                continue
            if probing and not _preflight_ok(getattr(provider, 'base_url', '')):
                _breaker_failure(key)
                escalations.append(f"breaker-preflight-fail@{rung.name}")
                last_exc = ProviderError(f"{rung.name} unreachable (preflight)")
                rung_i += 1
                continue
            # A rung can force reasoning ('thinking') ON — e.g. local-reason =
            # qwen3.6 with think=true — so escalating to it turns reasoning on
            # WITHOUT swapping the resident model. When the RUNG (not the caller)
            # forces it, also lift the tool-precision temperature so the forced
            # reasoning doesn't ruminate at a greedy temp.
            rung_forces_think = bool(getattr(rung, "think", False))
            rung_think = True if rung_forces_think else think
            rung_temp = temperature
            if rung_forces_think and not think and temperature <= 0.3:
                rung_temp = 0.6
            # Batch lane (the overnight 50% lever): background planes trade
            # latency for half-price tokens. Same payload, same parsing —
            # just the async transport underneath.
            batched = self._use_batch(provider, rung)
            chat_kw = {}
            if batched:
                escalations.append(f"batched@{rung.name}")
                chat_kw = {"wait_s": int(getattr(self.cfg, "batch_wait_s",
                                                 3600) or 3600),
                           "poll_s": int(getattr(self.cfg, "batch_poll_s",
                                                 20) or 20)}
            chat_fn = provider.chat_batch if batched else provider.chat
            try:
                comp = chat_fn(
                    rung.model, messages, schema=eff_schema,
                    temperature=rung_temp, system=sys_for_call,
                    max_tokens=max_tokens, tools=tools, on_token=on_token,
                    cancel=cancel, think=rung_think, on_think=on_think,
                    **chat_kw,
                )
            except ContextOverflow as exc:
                # The prompt didn't fit. Compact it once and retry the SAME rung
                # before giving up — the operator never sees the error.
                if not overflow_retried:
                    overflow_retried = True
                    _reset_sink()
                    messages = _shrink_for_retry(messages)
                    escalations.append(f"ctx-overflow-compacted@{rung.name}")
                    continue
                last_exc = exc
                escalations.append(f"error@{rung.name}:ContextOverflow")
                rung_i += 1
                continue
            except ProviderError as exc:
                _breaker_failure(key)
                # Transient (rate limit / 5xx / transport)? Retry the SAME rung a
                # couple times with jittered backoff before escalating — a brief
                # cloud hiccup shouldn't bump the whole turn to a costlier rung,
                # and the jitter stops concurrent hive drones from stampeding.
                # (Not when probing a broken rung — one attempt is the probe.)
                rung_tries = rung_attempts.get(rung_i, 0)
                if (is_transient(exc) and rung_tries < 2 and not probing
                        and not (cancel and cancel())):
                    rung_attempts[rung_i] = rung_tries + 1
                    delay = jittered_backoff(rung_tries + 1)
                    escalations.append(f"retry@{rung.name}:{delay:.1f}s")
                    _reset_sink()
                    time.sleep(delay)
                    continue                      # same rung, no rung_i bump
                last_exc = exc
                escalations.append(f"error@{rung.name}:{type(exc).__name__}")
                rung_i += 1
                _reset_sink()
                continue
            _breaker_success(key)

            # Silent-truncation detection (review 2.8): local Ollama does NOT
            # error on an oversized prompt — it truncates the FRONT (the system
            # prompt falls out first: persona, safety rules, schema directive)
            # and returns 200. The router used to read that as success. Compare
            # what the model says it ingested (prompt_eval_count) to a
            # conservative size estimate; a big shortfall means the window ate
            # the head — compact once and retry the SAME rung, like overflow.
            est_tokens = _estimate_prompt_tokens(messages, sys_for_call)
            if (not overflow_retried and comp.input_tokens > 0
                    and est_tokens > 2000
                    and comp.input_tokens < est_tokens * 0.5):
                overflow_retried = True
                _reset_sink()
                messages = _shrink_for_retry(messages)
                escalations.append(
                    f"ctx-truncated@{rung.name}:{comp.input_tokens}/{est_tokens}")
                continue

            trigger = self._should_escalate(comp, eff_schema, want_confidence)
            cost = self.ledger.record(rung, comp, trigger or "ok", plane=self.plane)
            last_result = RouteResult(
                completion=comp, rung_index=rung_i, rung_name=rung.name,
                escalations=escalations, est_cost_usd=cost,
            )

            if trigger and rung_i < max_rung:
                escalations.append(f"{trigger}@{rung.name}")
                rung_i += 1
                _reset_sink()                     # next rung rewrites from scratch
                continue

            return last_result

        if last_result:
            return last_result
        if last_exc:
            raise last_exc
        # Cap hit with no usable result.
        raise ProviderError(
            f"No rung produced a result (escalations: {escalations})"
        )

    # ------------------------------------------------------------------ #
    def _should_escalate(self, comp: Completion, schema: Optional[dict],
                         want_confidence: bool) -> Optional[str]:
        if comp.tool_calls:
            return None  # a native tool call is a valid, complete step
        if not comp.text.strip():
            return "empty-output"
        if schema:
            parsed = _try_json(comp.text)
            if parsed is None:
                return "schema-fail"
            if want_confidence:
                conf = parsed.get("confidence")
                if isinstance(conf, (int, float)) and conf < self.cfg.confidence_floor:
                    return "low-confidence"
        return None


def _try_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Tolerate models that wrap JSON in ``` fences.
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        # Last resort: grab the first {...} block.
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None
