"""ANVIL local web server — a launchable browser interface.

Zero dependencies: built on the standard-library ``http.server``. Three tabs:
Setup (Ollama key + models + persona), Chat (an IRC-style client that talks to
the pipeline), and Status. On first launch it shows a persona wizard so you can
name the agent and give it an initial personality before chatting.
"""

from __future__ import annotations

import json
import sys
import threading as _threading
import uuid
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from . import config as cfgmod
from . import persona
from . import mind

ROOT = Path(__file__).resolve().parent.parent
TOML_PATH = ROOT / "anvil.toml"
ENV_PATH = ROOT / ".env"


# --------------------------------------------------------------------------- #
# .env helpers
# --------------------------------------------------------------------------- #
def load_env_file(path: Optional[Path] = None) -> Dict[str, str]:
    path = path or ENV_PATH
    out: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def apply_env_to_process(env: Dict[str, str]) -> None:
    import os
    for k, v in env.items():
        if v:
            os.environ[k] = v


def write_env_updates(updates: Dict[str, str], path: Optional[Path] = None) -> None:
    path = path or ENV_PATH
    existing = load_env_file(path)
    for k, v in updates.items():
        if v is not None and v != "":
            existing[k] = v
    lines = ["# ANVIL secrets — managed by the Setup tab. Do not commit.", ""]
    # Preserve EVERY key already in .env (HA_URL/HA_TOKEN, custom vars, ...), plus
    # ensure the known ones exist. A save must never silently drop credentials.
    known = ("OLLAMA_API_KEY", "TAVILY_API_KEY",
             "DISCORD_WEBHOOK_URL", "DISCORD_BOT_TOKEN")
    for k in known:
        existing.setdefault(k, "")
    for k in list(known) + [x for x in sorted(existing) if x not in known]:
        lines.append(f"{k}={existing.get(k, '')}")
    cfgmod.atomic_write(path, "\n".join(lines) + "\n")
    apply_env_to_process(existing)


# --------------------------------------------------------------------------- #
# TOML rendering
# --------------------------------------------------------------------------- #
# Top-level keys render_toml manages explicitly. Anything ELSE found in the
# live anvil.toml (hand-tuned extras like vision_rung, heartbeat_interval_min)
# is preserved verbatim on save — a Setup-tab save must never wipe hand edits.
_MANAGED_KEYS = {
    "ollama_local_url", "ollama_cloud_url", "embed_model",
    "use_embeddings", "confidence_floor", "note_token_budget",
    "daily_cost_cap_usd", "background_cost_cap_usd", "request_timeout", "planner_rung", "critic_rung",
    "server_port", "bind_host", "autonomy", "searxng_url", "synthesis_mode", "home_address", "push_quiet_start", "push_quiet_end",
    "memory_dir", "jobs_dir", "ledger_path", "ladder",
}


def _toml_extras() -> Dict[str, Any]:
    """Hand-tuned scalar keys in the live TOML that Setup doesn't manage."""
    try:
        import tomllib
        raw = tomllib.loads(TOML_PATH.read_text("utf-8")) if TOML_PATH.exists() else {}
        return {k: v for k, v in raw.items()
                if k not in _MANAGED_KEYS and not isinstance(v, (dict, list))}
    except Exception:
        return {}


def _toml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_toml(c: Dict[str, Any], extras: Optional[Dict[str, Any]] = None) -> str:
    rungs = c.get("ladder", [])
    L = []
    A = L.append
    A("# ANVIL configuration — managed by the Setup tab (Ollama-only mode).")
    A("# Secrets live in .env, never here.\n")
    for key in ("ollama_local_url", "ollama_cloud_url", "embed_model"):
        A(f'{key} = "{c[key]}"')
    A(f"use_embeddings = {str(bool(c['use_embeddings'])).lower()}")
    A("")
    A(f"confidence_floor   = {c['confidence_floor']}")
    A(f"note_token_budget  = {int(c['note_token_budget'])}")
    A(f"daily_cost_cap_usd = {c['daily_cost_cap_usd']}")
    A(f"background_cost_cap_usd = {c.get('background_cost_cap_usd', 3.0)}")
    A(f"request_timeout    = {int(c['request_timeout'])}")
    A(f'planner_rung = "{c["planner_rung"]}"')
    A(f'critic_rung  = "{c["critic_rung"]}"')
    A(f"server_port  = {int(c['server_port'])}")
    A(f'bind_host    = "{c.get("bind_host", "127.0.0.1")}"')
    A(f'autonomy     = "{c.get("autonomy", "trusted")}"')
    A(f'searxng_url  = "{c.get("searxng_url", "")}"')
    A(f'synthesis_mode = "{c.get("synthesis_mode", "balanced")}"')
    A(f'home_address   = "{c.get("home_address", "")}"')
    A(f"push_quiet_start = {int(c.get('push_quiet_start', 22))}")
    A(f"push_quiet_end   = {int(c.get('push_quiet_end', 7))}")
    A("")
    A('memory_dir  = "memory"')
    A('jobs_dir    = "jobs"')
    A('ledger_path = "ledger.jsonl"')
    A("")
    if extras:
        A("# hand-tuned extras (preserved by Setup saves)")
        for k in sorted(extras):
            A(f"{k} = {_toml_scalar(extras[k])}")
        A("")
    for r in rungs:
        A("[[ladder]]")
        A(f'name = "{r["name"]}"')
        A(f'provider = "{r["provider"]}"')
        A(f'model = "{r["model"]}"')
        if r.get("cost_in"):
            A(f"cost_in = {r['cost_in']}")
        if r.get("cost_out"):
            A(f"cost_out = {r['cost_out']}")
        if r.get("cache_read"):
            A(f"cache_read = {r['cache_read']}")
        A(f"max_context = {int(r.get('max_context', 64000))}")
        A("")
    return "\n".join(L).rstrip() + "\n"


def current_config_dict() -> Dict[str, Any]:
    cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
    env = load_env_file()
    return {
        "ollama_local_url": cfg.ollama_local_url,
        "ollama_cloud_url": cfg.ollama_cloud_url,
        "embed_model": cfg.embed_model,
        "use_embeddings": cfg.use_embeddings,
        "confidence_floor": cfg.confidence_floor,
        "note_token_budget": cfg.note_token_budget,
        "daily_cost_cap_usd": cfg.daily_cost_cap_usd,
        "background_cost_cap_usd": getattr(cfg, "background_cost_cap_usd", 3.0),
        "request_timeout": cfg.request_timeout,
        "planner_rung": cfg.planner_rung,
        "critic_rung": cfg.critic_rung,
        "server_port": getattr(cfg, "server_port", 8765),
        "bind_host": getattr(cfg, "bind_host", "127.0.0.1"),
        "autonomy": getattr(cfg, "autonomy", "trusted"),
        "searxng_url": getattr(cfg, "searxng_url", ""),
        "synthesis_mode": getattr(cfg, "synthesis_mode", "balanced"),
        "home_address": getattr(cfg, "home_address", ""),
        "push_quiet_start": getattr(cfg, "push_quiet_start", 22),
        "push_quiet_end": getattr(cfg, "push_quiet_end", 7),
        "ladder": [
            {"name": r.name, "provider": r.provider, "model": r.model,
             "cost_in": r.cost_in, "cost_out": r.cost_out,
             "cache_read": r.cache_read, "max_context": r.max_context}
            for r in cfg.ladder
        ],
        "ollama_api_key_set": bool(env.get("OLLAMA_API_KEY")),
        "tavily_key_set": bool(env.get("TAVILY_API_KEY")),
        "discord_webhook_set": bool(env.get("DISCORD_WEBHOOK_URL")),
    }


def save_config_from_ui(payload: Dict[str, Any]) -> Dict[str, Any]:
    cur = current_config_dict()
    for k in ("ollama_local_url", "ollama_cloud_url", "embed_model",
              "use_embeddings", "confidence_floor", "note_token_budget",
              "daily_cost_cap_usd", "background_cost_cap_usd",
              "request_timeout", "planner_rung", "critic_rung", "server_port",
              "bind_host", "autonomy", "searxng_url", "synthesis_mode", "home_address", "push_quiet_start", "push_quiet_end"):
        if k in payload:
            cur[k] = payload[k]
    if str(cur.get("autonomy", "trusted")) not in ("ask", "trusted", "auto"):
        cur["autonomy"] = "trusted"
    if str(cur.get("synthesis_mode", "balanced")) not in ("local", "balanced", "cloud"):
        cur["synthesis_mode"] = "balanced"
    models = payload.get("models", {})
    for r in cur["ladder"]:
        if r["name"] in models and models[r["name"]]:
            r["model"] = models[r["name"]]
    cfgmod.atomic_write(TOML_PATH, render_toml(cur, _toml_extras()))
    env_updates = {}
    if payload.get("ollama_api_key"):
        env_updates["OLLAMA_API_KEY"] = payload["ollama_api_key"].strip()
    if payload.get("tavily_api_key"):
        env_updates["TAVILY_API_KEY"] = payload["tavily_api_key"].strip()
    if payload.get("discord_webhook_url") is not None:
        env_updates["DISCORD_WEBHOOK_URL"] = payload["discord_webhook_url"].strip()
    if env_updates:
        write_env_updates(env_updates)
    return current_config_dict()


# --------------------------------------------------------------------------- #
# Persona
# --------------------------------------------------------------------------- #
def persona_public(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "configured": bool(p.get("configured")),
        "name": p.get("name"),
        "base_prompt": p.get("base_prompt"),
        "traits": p.get("traits", []),
        "interactions": p.get("interactions", 0),
        "evolve_every": p.get("evolve_every", 12),
    }


def save_persona_from_ui(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = persona.load()
    if "name" in payload and payload["name"].strip():
        p["name"] = payload["name"].strip()
    if "base_prompt" in payload and payload["base_prompt"].strip():
        p["base_prompt"] = payload["base_prompt"].strip()
    if "evolve_every" in payload:
        try:
            p["evolve_every"] = max(1, int(payload["evolve_every"]))
        except (TypeError, ValueError):
            pass
    if payload.get("reset_traits"):
        p["traits"] = []
    if payload.get("configured"):
        p["configured"] = True
    persona.save(p)
    out = persona_public(persona.load())  # re-read to verify it stuck
    out["persisted"] = bool(out.get("configured")) if payload.get("configured") else True
    return out


# In-memory approval sessions: token -> {messages, pending}
SESSIONS: Dict[str, Dict[str, Any]] = {}
# Which family profile a chat session (sid) is acting as, and — for adults with a
# PIN — until when they're verified. {sid: {"name","role","adult_until": ts}}.
# The danger gate reads this: only a verified adult can auto-run/approve danger.
SESSION_PROFILE: Dict[str, Dict[str, Any]] = {}
_PROFILE_GUARD = _threading.Lock()
_ADULT_TTL = 12 * 3600           # a PIN unlock keeps this device adult for 12h


# Real login sessions: a random cookie token -> the authenticated profile.
# Auth is OPT-IN — it engages only once an adult sets a password (profiles.auth_on).
# Until then ANVIL runs with no login wall, exactly as before.
AUTH: Dict[str, Dict[str, Any]] = {}
_AUTH_TTL = 30 * 24 * 3600       # remember-me: a family device stays in ~30 days
_AUTH_TTL_SESSION = 12 * 3600    # no remember-me: server-side cap for session cookies
_COOKIE = "anvil_auth"
# Login rate limiting: username -> [failure timestamps] (15-min window, 5 max).
_LOGIN_FAILS: Dict[str, list] = {}


def _mint_auth(name: str, role: str, remember: bool = True) -> str:
    import time as _t
    tok = uuid.uuid4().hex
    with _PROFILE_GUARD:
        AUTH[tok] = {"name": name, "role": role, "ts": _t.time(),
                     "remember": bool(remember)}
    _persist_auth()          # a release restart must not log the family out
    return tok


def _host_is_ours(host: str) -> bool:
    """Is this a name/address this box legitimately answers to? localhost, any IP
    literal (LAN + tailscale IPs — an IP can't DNS-rebind), the machine's own
    hostname, or a tailnet name. An attacker's rebinding domain matches none."""
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        import ipaddress
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host.endswith(".ts.net"):
        return True
    # A deliberately configured public domain (the VPS front door) is ours
    # too — without this, every POST arriving as Host: lara.example.com
    # would be blocked as a suspected rebinding page.
    try:
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        pub = str(getattr(cfg, "public_host", "") or "").strip().lower()
        if pub and host == pub:
            return True
    except Exception:
        pass
    import socket as _sock
    name = _sock.gethostname().lower()
    return host == name or host.startswith(name + ".")


def _cookie_header(handler) -> str:
    import time as _t
    raw = handler.headers.get("Cookie", "") or ""
    for part in raw.split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            if k == _COOKIE:
                with _PROFILE_GUARD:
                    rec = AUTH.get(v)
                # Session (no remember-me) tokens expire server-side in 12h even
                # if the browser kept the cookie alive; remember-me gets 30 days.
                ttl = _AUTH_TTL if (rec or {}).get("remember", True) else _AUTH_TTL_SESSION
                if rec and _t.time() - rec.get("ts", 0) < ttl:
                    return v
    return ""


def _authed(handler):
    """The logged-in profile record for this request, or None."""
    tok = _cookie_header(handler)
    if not tok:
        return None
    with _PROFILE_GUARD:
        return dict(AUTH.get(tok) or {}) or None


def _resolve_profile(cfg, sid, handler=None):
    """Who is acting? A live login cookie wins (real identity); otherwise the
    per-sid selection; otherwise the fail-safe default."""
    from . import profiles
    if handler is not None:
        a = _authed(handler)
        if a and a.get("name"):
            prof = profiles.get(cfg, a["name"])
            if prof:
                return prof, {"name": prof.name, "role": prof.role, "authed": True}
    with _PROFILE_GUARD:
        rec = dict(SESSION_PROFILE.get(sid) or {})
    name = rec.get("name") or profiles.default_name(cfg)
    prof = profiles.get(cfg, name) or profiles.get(cfg, profiles.default_name(cfg))
    return prof, rec


def _session_admin(cfg, sid, handler=None) -> bool:
    """Is this request acting as the household ADMIN (the first profile, always
    an adult)? Admin gates profile management + system settings — distinct from
    plain adult-ness, which any adult has (e.g. to approve a child's action)."""
    prof, _ = _resolve_profile(cfg, sid, handler)
    if not getattr(prof, "is_admin", False):
        return False
    return _session_adult(cfg, sid, handler)   # admin is adult; verify the login/PIN


def _trusted_proxies(cfg) -> set:
    raw = str(getattr(cfg, "trusted_proxy", "") or "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def _client_ip(cfg, handler) -> str:
    """The REAL client address. X-Forwarded-For is only believed when the
    request physically came from a configured trusted proxy — anyone else
    claiming a forwarded identity is lying, and gets their socket address."""
    peer = ""
    try:
        peer = handler.client_address[0]
    except Exception:
        pass
    if peer and peer in _trusted_proxies(cfg):
        xff = (handler.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        if xff:
            return xff
    return peer


def _via_https(cfg, handler) -> bool:
    """True when the request reached us over HTTPS via the trusted proxy —
    the signal for marking session cookies Secure."""
    try:
        peer = handler.client_address[0]
    except Exception:
        return False
    return (peer in _trusted_proxies(cfg)
            and (handler.headers.get("X-Forwarded-Proto") or "").lower() == "https")


def _approvals_log_path(cfg) -> Path:
    return Path(getattr(cfg, "memory_dir", "memory")) / "approvals_log.jsonl"


def _log_approval_decision(cfg, rec: dict) -> None:
    """Append one decided approval to the family-visible audit log (#62),
    trimming to the last 200 lines so it never grows unbounded."""
    p = _approvals_log_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    try:
        lines = p.read_text("utf-8", "replace").splitlines()
        if len(lines) > 200:
            cfgmod.atomic_write(p, "\n".join(lines[-200:]) + "\n")
    except OSError:
        pass


def _read_approvals_log(cfg, n: int = 30) -> list:
    p = _approvals_log_path(cfg)
    if not p.exists():
        return []
    out = []
    for line in p.read_text("utf-8", "replace").splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    out.reverse()                      # newest first for the card
    return out


def _actor_name(cfg, sid, handler=None) -> str:
    """The acting profile's name for data-isolation, or "" for a single-user
    install (no login configured). Private data — conversations, personal memory
    — is namespaced by this so nothing leaks between family members."""
    from . import profiles
    if not profiles.auth_on(cfg):
        return ""                       # single-user: one flat namespace, as before
    prof, _ = _resolve_profile(cfg, sid, handler)
    return getattr(prof, "name", "") or ""


def _bind_actor(cfg, sid, handler=None) -> str:
    """Stamp the acting profile onto this request's cfg so cfg-only code paths
    (Conversations, the memory tools) scope to the right person. Returns the
    actor name."""
    actor = _actor_name(cfg, sid, handler)
    try:
        cfg._actor = actor
    except Exception:
        pass
    return actor


def _session_adult(cfg, sid, handler=None) -> bool:
    """Is this request acting as a VERIFIED adult? A logged-in adult (real auth)
    is verified by their login. Without auth: an adult-with-PIN must have
    unlocked within the TTL; an adult with no PIN is trusted ONLY when the
    household has no minor (single-user / all-adult default stays frictionless);
    a minor is never adult."""
    import time as _t
    from . import profiles
    prof, rec = _resolve_profile(cfg, sid, handler)
    if prof is None or not prof.is_adult:
        return False
    if rec.get("authed"):
        return True                       # logged in as this adult = verified
    if not prof.has_pin:
        # A PIN-less adult is auto-trusted only in a household with NO minor.
        # Once a minor exists, the fail-safe default (session -> minor) would
        # otherwise be trivially defeated by selecting a PIN-less adult profile
        # with no credential — re-opening the "kid unlocks the door" hole. In a
        # mixed household an adult must prove identity (PIN unlock or login).
        return not profiles.any_minor(cfg)
    return bool(rec.get("name") == prof.name and rec.get("adult_until", 0) > _t.time())
# Live "what am I doing" trail per in-flight chat request (rid -> ordered list of
# distinct phases), so the UI can poll /api/progress, catch even momentary phases,
# and play them out with a minimum linger instead of a static spinner.
PROGRESS: Dict[str, list] = {}
# Live partial answer per in-flight chat request (rid -> text streamed so far), so
# the UI can poll /api/progress and render Lara's reply token-by-token as it lands.
STREAM: Dict[str, str] = {}
# Live deliberation per in-flight request: a reasoning model's "thinking" deltas
# stream on their own channel, so the UI can show WHAT Lara is mulling over
# during the otherwise-silent reasoning phase instead of looking hung.
THINKING: Dict[str, str] = {}
# Live context size per in-flight request (rid -> prompt tokens of the latest
# model call), so the UI can show context building up in real time.
CTX: Dict[str, int] = {}
# Live tool events per in-flight request ({tool, args, out, t} in call order):
# the Viewer's terminal feed — shell/ssh calls render as a typed live session.
TOOLS_LIVE: Dict[str, list] = {}
# SSE subscribers per in-flight request (rid -> [Queue]). The push replacement
# for the 500ms /api/progress poll: each live sink nudges a throttled snapshot
# into every subscriber's queue; /api/stream drains one queue per client.
# Polling stays as the fallback transport — the dicts above remain canonical.
SSE_SUBS: Dict[str, list] = {}
SSE_LOCK = _threading.Lock()


def _live_snapshot(rid: str) -> Dict[str, Any]:
    """The same payload /api/progress serves — one shape for both transports."""
    return {"steps": PROGRESS.get(rid, []), "partial": STREAM.get(rid, ""),
            "thinking": THINKING.get(rid, ""), "ctx": CTX.get(rid, 0),
            "tools": TOOLS_LIVE.get(rid, [])}


def _sse_push(rid: str, kind: str, data) -> None:
    with SSE_LOCK:
        subs = list(SSE_SUBS.get(rid, ()))
    for q in subs:
        try:
            q.put_nowait((kind, data))
        except Exception:
            pass                       # a full/slow client just misses a frame

# One ask at a time PER CONVERSATION: without this, a quick follow-up ("Hello?")
# runs on a parallel thread, slips between the long first task's model calls,
# and gets answered first — out-of-order replies in the same chat. Different
# conversations still run in parallel.
_ASK_LOCKS: Dict[str, Any] = {}
_ASK_LOCKS_GUARD = _threading.Lock()

# Cooperative cancel: rids the operator hit Stop on. Checked in the token sink
# (aborts mid-generation) and between agent-loop steps.
CANCELLED: set = set()
_CANCEL_GUARD = _threading.Lock()
# Poll/stream liveness. This USED to abandon-cancel a turn when the phone went
# silent — but that guaranteed a locked phone lost its answer. Now that every
# turn is hard-bounded by ask_time_budget_s (240s wrap-up) and the finished
# answer both persists to the conversation and PUSHES to the asker, generation
# always runs to completion: lock your phone mid-question, the answer is
# waiting in the chat (and on your lock screen) when you come back. LAST_POLL
# stays as a liveness record only — nothing cancels off it anymore.
LAST_POLL: Dict[str, float] = {}


def _cancel_add(rid: str) -> None:
    with _CANCEL_GUARD:
        CANCELLED.add(rid)


def _is_cancelled(rid: str) -> bool:
    with _CANCEL_GUARD:
        return rid in CANCELLED


def _cancel_clear(rid: str) -> None:
    with _CANCEL_GUARD:
        CANCELLED.discard(rid)


def _sid_lock(sid: str):
    with _ASK_LOCKS_GUARD:
        # Every 'New chat' mints a sid; drop idle locks so the dict can't grow
        # forever on a long-running server (an unheld lock is safely re-mintable).
        if len(_ASK_LOCKS) > 64:
            for k in [k for k, l in list(_ASK_LOCKS.items())
                      if k != sid and not l.locked()]:
                _ASK_LOCKS.pop(k, None)
        if sid not in _ASK_LOCKS:
            _ASK_LOCKS[sid] = _threading.Lock()
        return _ASK_LOCKS[sid]


def _snippet(text: str, limit: int = 140) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _adult_profiles(cfg) -> set:
    """Names of the adult profiles — the push target for approvals a child raised."""
    try:
        from . import profiles
        return {p.name for p in profiles.load(cfg).values() if p.is_adult}
    except Exception:
        return set()


def _push_answer(cfg, answer: str, to=None, sid: str = "") -> None:
    """Notify that Lara finished answering (the service worker suppresses this
    when the app is already open/visible, so it only lands when you're away).
    Routed to the ASKER's own devices only when ``to`` is given (a profile name).
    ``sid`` deep-links the notification straight to THAT conversation, so a tap
    lands you in the chat you asked from — not a cold open on the last one."""
    if (answer or "").strip():
        try:
            from . import push
            from urllib.parse import quote
            tgt = {to} if to else None
            url = "/?chat=" + quote(sid, safe="") if sid else "/"
            push.notify(cfg, "Lara answered", _snippet(answer),
                        url=url, tag="chat", to=tgt)
        except Exception:
            pass


def _pending_summary(pending: dict) -> str:
    """A short human phrase for what a pending danger action would do."""
    tool = (pending or {}).get("tool", "an action")
    args = (pending or {}).get("args", {}) or {}
    hint = (args.get("entity_id") or args.get("cmd") or args.get("service")
            or args.get("url") or args.get("path") or "")
    return f"{tool}" + (f": {hint}" if hint else "")


def _push_approval(cfg, pending: dict, who: str = "", why: str = "") -> None:
    summary = _pending_summary(pending)
    # Cross-device routing: an adult on another device should see WHO is asking
    # and WHY, straight from the notification, before they even open the app.
    lead = (f"{who} wants to run {summary}" if who else f"Wants to run {summary}")
    body = lead + (f" — “{_snippet(why, 80)}”" if why else "") + " — tap to review"
    try:
        from . import push, profiles
        # An approval can only be granted by an adult — push it to the ADULTS'
        # devices (not the whole household). No auth yet -> broadcast as before.
        to = _adult_profiles(cfg) if profiles.auth_on(cfg) else None
        # Deep-link straight to the pending list, so a tap opens the approvals
        # sheet (no hunting for the badge).
        push.notify(cfg, "Lara needs approval", body,
                    url="/?approvals=1", tag="approval", to=to)
    except Exception:
        pass


# When an adult resolves a child's request on THEIR device, the child's device
# (still showing the pending card) polls here to pick up the outcome and route
# the answer back. Kept small + TTL-pruned alongside SESSIONS.
RESOLVED: Dict[str, Dict[str, Any]] = {}


def _prune_resolved(now: Optional[float] = None) -> None:
    import time as _t
    now = now or _t.time()
    for tok in [t for t, r in list(RESOLVED.items())
                if now - r.get("ts", now) > SESSION_TTL_S]:
        RESOLVED.pop(tok, None)


# Approval requests wait for a human; one that's been sitting for an hour is
# stale — approving it days later by accident (a forgotten "unlock the door")
# would be genuinely dangerous now that Lara can act on the house. Expired
# tokens answer with the existing "unknown or expired approval token" message.
SESSION_TTL_S = 3600


def _prune_sessions(now: Optional[float] = None) -> None:
    import time as _t
    now = now or _t.time()
    for tok in [t for t, s in list(SESSIONS.items())
                if now - s.get("ts", now) > SESSION_TTL_S]:
        SESSIONS.pop(tok, None)
        _unpersist_approval(tok)


# ---- durable serving state (review 1.1) ----------------------------------- #
# Approvals and logins must survive a restart: this harness restarts ITSELF to
# deploy, and the cross-device flow makes approvals long-lived (an adult answers
# minutes later). Losing SESSIONS mid-wait silently swallowed a parked danger
# request; losing AUTH logged every family device out on every release.
_APPROVALS_DIR = ROOT / "memory" / "approvals"
_AUTH_FILE = ROOT / "memory" / "auth_sessions.json"


def _persist_approval(token: str, sess: Dict[str, Any]) -> None:
    import os as _os
    if _os.environ.get("ANVIL_IN_DOCTOR"):
        return          # hermetic: tests must never park REAL approval files
    try:
        _APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
        cfgmod.atomic_write(_APPROVALS_DIR / f"{token}.json",
                            json.dumps(sess, default=str))
    except Exception:
        pass                              # durability is best-effort, never fatal


def _unpersist_approval(token: str) -> None:
    try:
        (_APPROVALS_DIR / f"{token}.json").unlink()
    except OSError:
        pass


def _persist_auth() -> None:
    import os as _os
    if _os.environ.get("ANVIL_IN_DOCTOR"):
        return
    try:
        with _PROFILE_GUARD:
            snap = dict(AUTH)
        cfgmod.atomic_write(_AUTH_FILE, json.dumps(snap))
    except Exception:
        pass


def _load_serving_state() -> None:
    """Reload approvals + logins on boot; TTLs still apply, so anything stale
    just prunes instead of resurrecting."""
    import time as _t
    now = _t.time()
    try:
        for f in _APPROVALS_DIR.glob("*.json"):
            try:
                sess = json.loads(f.read_text("utf-8"))
                if now - float(sess.get("ts", 0)) <= SESSION_TTL_S:
                    SESSIONS.setdefault(f.stem, sess)
                else:
                    f.unlink()
            except Exception:
                continue
    except OSError:
        pass
    try:
        data = json.loads(_AUTH_FILE.read_text("utf-8"))
        with _PROFILE_GUARD:
            for tok, rec in data.items():
                if now - float(rec.get("ts", 0)) <= _AUTH_TTL:
                    AUTH.setdefault(tok, rec)
    except Exception:
        pass


def _step_public(s: Dict[str, Any]) -> Dict[str, Any]:
    """The wire copy of one agent step: everything the transparency timeline
    needs (tool, args, result excerpt, the thought that led to it), capped so
    a huge observation can't bloat the response."""
    try:
        args = json.dumps(s.get("args", {}), default=str)[:600]
    except Exception:
        args = str(s.get("args", ""))[:600]
    return {"tool": s.get("tool", ""), "args": args,
            "observation": str(s.get("observation", ""))[:2000],
            "danger": bool(s.get("danger")),
            "thought": str(s.get("thought", ""))[:4000],
            # narration she wrote WHILE deciding to make this call — without
            # it a reread of the conversation loses the story between actions
            "said": str(s.get("said", ""))[:1500]}


def _turn_meta(res: Dict[str, Any]) -> Dict[str, Any]:
    """The transparency payload worth KEEPING with the transcript turn — the
    same capped shapes the live response carries, trimmed to sane counts so
    the 'how' trace, artifact chips, and folded shell survive a reload."""
    steps = [_step_public(s) for s in (res.get("steps") or [])[:16]]
    return {"steps": steps,
            "trace": (res.get("trace") or [])[:60],
            "rung": res.get("rung", ""),
            "final_thought": str(res.get("final_thought", ""))[:4000]}


def _agent_response(res: Dict[str, Any], sid: str = "", task: str = "",
                    user_recorded: bool = False,
                    requested_by: str = "") -> Dict[str, Any]:
    out = {"mode": "agent", "status": res["status"],
           "steps": [_step_public(s) for s in res.get("steps", [])],
           "rung": res.get("rung", ""),
           "recalled": res.get("recalled", 0), "evolved": res.get("evolved"),
           "ctx": res.get("ctx", 0), "trace": res.get("trace", []),
           "final_thought": str(res.get("final_thought", ""))[:4000]}
    if res["status"] == "approve":
        import time as _t
        _prune_sessions()                 # opportunistic cleanup, no timer thread
        token = uuid.uuid4().hex
        adult_required = bool(res.get("adult_required"))
        SESSIONS[token] = {"messages": res["messages"], "pending": res["pending"],
                           "sid": sid, "task": task, "ts": _t.time(),
                           "user_recorded": user_recorded,
                           "adult_required": adult_required,
                           "requested_by": requested_by}
        _persist_approval(token, SESSIONS[token])   # survives a self-deploy restart
        out["token"] = token
        out["pending"] = res["pending"]
        # Tell the client to demand an adult PIN before this can be approved
        # (a minor/unverified session raised a danger action).
        out["adult_required"] = adult_required
        out["requested_by"] = requested_by
    else:
        out["answer"] = res.get("answer", "")
    return out


def _job_public(job) -> Dict[str, Any]:
    return {"name": job.name, "cron": job.cron,
            "prompt": job.inputs.get("prompt", ""),
            "notify": job.notify or "", "min_rung": job.min_rung,
            "enabled": job.enabled, "owner": getattr(job, "owner", "") or ""}


# --------------------------------------------------------------------------- #
# Probes / status
# --------------------------------------------------------------------------- #
def ollama_models(base_url: str, timeout: int = 5) -> Dict[str, Any]:
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name") for m in data.get("models", [])]
        return {"reachable": True, "models": [n for n in names if n]}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return {"reachable": False, "models": []}


def build_status() -> Dict[str, Any]:
    cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
    probe = ollama_models(cfg.ollama_local_url)
    from .scheduler import Scheduler
    from .memory import MemoryStore
    from .router import CostLedger
    p = persona.load()
    return {
        "agent_name": p.get("name"),
        "ladder": [f"{r.name} ({r.model})" for r in cfg.ladder],
        "ollama_reachable": probe["reachable"],
        "installed_models": probe["models"],
        "cloud_key_set": bool(load_env_file().get("OLLAMA_API_KEY")),
        "jobs": len(Scheduler(cfg).load_jobs()),
        "notes": len(MemoryStore(cfg).all_notes()),
        "spent_today": round(CostLedger(cfg.ledger_path).spent_today(), 4),
        "embeddings": cfg.use_embeddings,
        "traits": len(p.get("traits", [])),
    }


def _forge_commits(n: int = 8):
    import subprocess
    try:
        out = subprocess.run(["git", "log", "forge-auto", "--oneline", f"-{n}"],
                             cwd=str(ROOT), capture_output=True, text=True,
                             timeout=5).stdout
        return [ln for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def _queue_summary():
    from collections import Counter
    q = ROOT / "forge" / "queue"
    counts: Counter = Counter()
    items = []
    if q.exists():
        for p in sorted(q.glob("TASK-*.md")):
            try:
                text = p.read_text("utf-8", "replace")
            except OSError:
                continue
            st = title = ""
            for ln in text.splitlines()[1:]:
                if ln.strip() == "---":
                    break
                low = ln.lower()
                if low.startswith("status:"):
                    st = ln.split(":", 1)[1].strip()
                elif low.startswith("title:"):
                    title = ln.split(":", 1)[1].strip()
            counts[st or "?"] += 1
            items.append({"id": "-".join(p.name.split("-")[:2]),
                          "status": st, "title": title[:70]})
    return {"counts": dict(counts), "items": items}


def build_pulse(cfg=None, is_admin: bool = True) -> Dict[str, Any]:
    """Everything the Pulse dashboard shows: autonomy state, thoughts, dreams,
    self-dev activity, and the shared council backlog.

    Scoped to the viewer: the thought stream shows only THIS person's activity
    (their meta.actor) plus Lara's ambient thoughts — never another family
    member's. Development views (self-dev, the council backlog + commits) are
    ADMIN-only and withheld from a non-admin's payload entirely."""
    from datetime import date
    from .memory import MemoryStore
    from .router import CostLedger
    if cfg is None:
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
    actor = getattr(cfg, "_actor", "")
    stm = mind.read_stm(cfg, 60)

    def _mine(r) -> bool:
        if not actor:
            return True                    # single-user / admin-neutral: all
        m = r.get("meta") or {}
        return (m.get("actor") or "") in ("", actor)   # ambient or mine
    thoughts = [r for r in stm if _mine(r)][-30:]

    jlines = mind.read_journal(cfg, 250) if is_admin else []
    dreams = [ln for ln in jlines if "[dream]" in ln][-8:]
    selfdev_log = [ln for ln in jlines if "[selfdev]" in ln or "[deep-sleep]" in ln][-10:]
    sd_state = {}
    if is_admin:
        try:
            sd_state = json.loads((ROOT / "dev-reports" / "selfdev-state.json")
                                  .read_text("utf-8"))
        except Exception:
            pass
    today = sd_state.get("count", 0) if sd_state.get("date") == date.today().isoformat() else 0
    return {
        "is_admin": bool(is_admin),
        "auto_pulse": bool(getattr(cfg, "auto_pulse", True)),
        "heartbeat_min": getattr(cfg, "heartbeat_interval_min", 15),
        "dream_after": getattr(cfg, "dream_after", 40),
        "dream_max_age_h": getattr(cfg, "dream_max_age_hours", 6.0),
        "self_dev_in_sleep": bool(getattr(cfg, "self_dev_in_sleep", True)),
        "selfdev_interval_h": getattr(cfg, "selfdev_interval_hours", 12.0),
        "self_dev_today": today,
        "self_dev_cap": getattr(cfg, "self_dev_daily_cap", 3),
        "ladder": [r.name for r in cfg.ladder],
        "spent_today": round(CostLedger(cfg.ledger_path).spent_today(), 4),
        "cost_cap": float(getattr(cfg, "daily_cost_cap_usd", 5.0)),
        "cost_cap_bg": float(getattr(cfg, "background_cost_cap_usd", 3.0)),
        "spent_by_plane": (CostLedger(cfg.ledger_path).spent_by_plane()
                           if is_admin else {}),
        "notes": len(MemoryStore(cfg).visible_notes()),
        "stm_size": len(thoughts),
        "thoughts": thoughts,
        # Dreams are Lara's ambient reflections; dev views are admin-only.
        "dreams": dreams if is_admin else [],
        "selfdev_log": selfdev_log,
        "queue": _queue_summary() if is_admin else {},
        "commits": _forge_commits(8) if is_admin else [],
    }


_VIEW_CAP = 512 * 1024          # viewer payload cap — plenty for text, sane for images
_IMG_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml"}


def build_file(cfg, rel: str, admin: bool) -> Dict[str, Any]:
    """The Viewer's data source: ONE workspace file, resolved through the same
    sandbox as Lara's file tools (so every path in her trace opens here).
    Admins see the whole workspace; family accounts only its family-facing
    corners (docs/, recipes/, shared/)."""
    from . import tools
    try:
        p = tools._safe_path(cfg, rel)
    except Exception:
        return {"error": "that path isn't in the workspace"}
    if not admin:
        root = tools.workspace(cfg)
        fam = [(root / d).resolve() for d in ("docs", "recipes", "shared")]
        if not any(f == p or f in p.parents for f in fam):
            return {"error": "that file isn't shared with the family"}
    if not p.exists() or not p.is_file():
        return {"error": f"no such file: {rel}"}
    size = p.stat().st_size
    ext = p.suffix.lower()
    if ext in _IMG_EXT:
        if size > _VIEW_CAP * 4:
            return {"error": "that image is too large to preview"}
        import base64
        return {"ok": True, "name": p.name, "path": rel, "kind": "image",
                "mime": _IMG_EXT[ext], "size": size,
                "b64": base64.b64encode(p.read_bytes()).decode("ascii")}
    text = p.read_text("utf-8", "replace")
    truncated = len(text) > _VIEW_CAP
    kind = ("markdown" if ext in (".md", ".markdown")
            else "html" if ext in (".html", ".htm") else "text")
    # show_map documents render as an interactive map, not raw JSON
    if ext == ".json" and rel.replace("\\", "/").lstrip("./").startswith(
            "shared/maps/"):
        kind = "map"
    return {"ok": True, "name": p.name, "path": rel, "kind": kind, "size": size,
            "content": text[:_VIEW_CAP], "truncated": truncated}


def build_memory(cfg=None) -> Dict[str, Any]:
    from .memory import MemoryStore
    if cfg is None:
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
    store = MemoryStore(cfg)
    # Only what the acting profile may SEE — shared household notes + their own.
    notes = store.visible_notes()
    me = store.actor
    rows = [{"name": n.name, "type": n.type, "salience": round(n.salience, 3),
             "created": n.created, "tags": n.tags, "body": n.body[:400],
             "owner": n.owner, "shared_with": n.shared_with,
             # classify for the viewer: their own, household/common, or shared
             # to them by someone else (can't be resharing that one).
             "mine": bool(me) and n.owner == me,
             "household": n.owner == "",
             "personal": bool(n.owner)}
            for n in notes]
    rows.sort(key=lambda r: (-r["salience"], r["name"]))
    from .skills import SkillStore
    skills = [{"name": s.name, "description": s.description, "when": s.when,
               "body": s.body[:600]} for s in SkillStore(cfg).all()]
    return {"count": len(rows), "notes": rows,
            "skills": skills, "skill_count": len(skills),
            "actor": store.actor}


def build_ha() -> Dict[str, Any]:
    from collections import Counter
    cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
    apply_env_to_process(load_env_file())
    from . import homeassistant as ha
    client = ha.HomeAssistant(timeout=getattr(cfg, "request_timeout", 5))
    if not client.is_configured:
        return {"configured": False}
    states = client.states()
    if not states:
        return {"configured": True, "healthy": client.health_check(), "count": 0}

    def pick(*domains):
        out = []
        for e in states:
            dom = e.get("entity_id", "").split(".", 1)[0]
            if dom in domains:
                out.append({"id": e.get("entity_id"), "state": str(e.get("state")),
                            "name": (e.get("attributes") or {}).get("friendly_name", "")})
        return out

    doms = Counter(e.get("entity_id", "").split(".", 1)[0] for e in states)
    return {
        "configured": True, "healthy": client.health_check(), "count": len(states),
        "domains": dict(doms),
        "people": pick("person"),
        "media": pick("media_player"),
        "controls": pick("light", "switch", "fan")[:20],
        "sensors": pick("binary_sensor", "sensor")[:20],
    }


# --------------------------------------------------------------------------- #
# Request handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "ANVIL/0.2"

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str,
              cache: str = "no-store, no-cache, must-revalidate",
              extra_headers=None):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            if cache.startswith("no-store"):
                self.send_header("Pragma", "no-cache")
            for k, v in (extra_headers or []):
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # The client hung up before we finished (backgrounded tab, navigated
            # away, or a slow reply it gave up on). There's nothing to send to a
            # dead socket — swallow it so we don't crash the worker thread or
            # double-fault trying to write an error response to the same socket.
            self.close_connection = True

    def _json(self, obj: Any, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _origin_blocked(self) -> Optional[str]:
        """CSRF / drive-by / DNS-rebinding defense for every state-changing request.
        Auth is opt-in, so without this ANY web page a family device visits could
        POST /api/ask (or worse) straight into Lara. Three checks, browser-shaped:

        1. A request body must be JSON — a cross-site <form> cannot send
           application/json without a CORS preflight, which we never grant.
        2. When the browser supplies Origin/Referer, its host must equal the Host
           we were addressed as — a drive-by page's origin never matches.
           (curl/CLI send neither header and pass untouched.)
        3. The Host itself must look like OURS (localhost, an IP literal, this
           machine's hostname, or a tailnet name) — Origin==Host alone would pass
           a DNS-rebinding page, whose attacker DNS name matches on both sides.
        Returns a short reason when blocked, None when fine."""
        n = int(self.headers.get("Content-Length", 0) or 0)
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if n and ctype != "application/json":
            return f"content-type {ctype or '(none)'} not accepted"
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        for h in ("Origin", "Referer"):
            v = (self.headers.get(h) or "").strip()
            if v and v.lower() != "null":
                try:
                    from urllib.parse import urlsplit
                    ohost = (urlsplit(v).hostname or "").strip("[]").lower()
                except ValueError:
                    return f"unparseable {h} header"
                if ohost != host:
                    return f"cross-origin {h.lower()} ({ohost or '?'} != {host or '?'})"
                break
            if v:                        # "null" Origin: sandboxed/data: pages
                return "null origin"
        if host and not _host_is_ours(host):
            return f"host {host!r} is not this server"
        return None

    # Paths reachable without a login (the shell, static assets, the login flow).
    _PUBLIC = {"/", "/index.html", "/api/version", "/api/login", "/api/me",
               "/api/setup",
               "/manifest.webmanifest", "/sw.js"}

    def _auth_gate(self, path) -> bool:
        """Return True if the request should be BLOCKED (401). Only when auth is
        configured, the path isn't public/static, and there's no valid login."""
        if (path in self._PUBLIC or path.startswith("/icons/")
                or path.startswith("/vendor/")):
            return False
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        from . import profiles
        if not profiles.auth_on(cfg):
            return False
        return _authed(self) is None

    def do_GET(self):
        # Strip the cache-buster (?_=...) so query-stringed GETs still match.
        path = self.path.split("?", 1)[0]
        if self._auth_gate(path):
            return self._json({"error": "auth", "need_login": True}, 401)
        if path in ("/", "/index.html"):
            # HEARTHLIGHT (the new family UI) fronts the app; the classic
            # console survives at /classic as the Workshop while the overhaul
            # phases move its surfaces across. Falls back to classic if the
            # hearth asset is missing (a broken deploy must not brick the UI).
            html = HEARTH_HTML or INDEX_HTML
            html = html.replace("__UI_BUILD__", str(UI_BUILD))
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/classic":
            html = INDEX_HTML.replace("__UI_BUILD__", str(UI_BUILD))
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/version":
            self._json({"build": UI_BUILD})
        elif path == "/manifest.webmanifest":
            self._send(200, MANIFEST_JSON.encode("utf-8"),
                       "application/manifest+json; charset=utf-8")
        elif path == "/sw.js":
            # Served from the root so its scope covers the whole app.
            self._send(200, SW_JS.encode("utf-8"),
                       "application/javascript; charset=utf-8")
        elif path.startswith("/icons/flame-") and path.endswith(".png"):
            from . import icon
            try:
                size = int(path[len("/icons/flame-"):-len(".png")].split("-")[0])
            except ValueError:
                size = 192
            size = max(16, min(1024, size))
            maskable = path.endswith("-maskable.png")
            # Icons are deterministic — let the phone cache them for a day
            # instead of re-downloading on every PWA launch (everything else
            # stays no-store so UI/SW updates always land).
            self._send(200, icon.render_png(size, maskable), "image/png",
                       cache="public, max-age=86400")
        elif path.startswith("/vendor/"):
            # Vendored third-party assets (no CDNs — the PWA is self-contained).
            # Strict allowlist: nothing else under anvil/ is ever served raw.
            _VENDOR = {"leaflet.js": "application/javascript; charset=utf-8",
                       "leaflet.css": "text/css; charset=utf-8"}
            name = path[len("/vendor/"):]
            f = Path(__file__).parent / "vendor" / name
            if name in _VENDOR and f.exists():
                self._send(200, f.read_bytes(), _VENDOR[name],
                           cache="public, max-age=86400")
            else:
                self._json({"error": "not found"}, 404)
        elif path == "/api/push/config":
            self._json(self._push_config())
        elif path == "/api/config":
            self._json(current_config_dict())
        elif path == "/api/persona":
            self._json(persona_public(persona.load()))
        elif path == "/api/status":
            self._json(build_status())
        elif path == "/api/jobs":
            from urllib.parse import urlparse, parse_qs
            sid = (parse_qs(urlparse(self.path).query).get("sid") or [""])[0]
            self._json(self._jobs_list(sid))
        elif path == "/api/journal":
            from urllib.parse import urlparse, parse_qs
            sid = (parse_qs(urlparse(self.path).query).get("sid") or [""])[0]
            self._json(self._mind_view(sid))
        elif path == "/api/me":
            from . import profiles
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            a = _authed(self)
            name = (a or {}).get("name", "")
            me_prof = profiles.get(cfg, name) if name else None
            self._json({"auth_on": profiles.auth_on(cfg),
                        "needs_setup": profiles.needs_setup(cfg),
                        "authed": bool(a),
                        "name": name,
                        "username": getattr(me_prof, "username", ""),
                        "has_password": bool(getattr(me_prof, "has_password", False)),
                        "role": (a or {}).get("role", ""),
                        "admin": bool(getattr(me_prof, "is_admin", False)),
                        "profiles": [p.public() for p in profiles.load(cfg).values()]})
        elif path == "/api/profiles":
            from urllib.parse import urlparse, parse_qs
            from . import profiles
            sid = (parse_qs(urlparse(self.path).query).get("sid") or [""])[0]
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            prof, _ = _resolve_profile(cfg, sid, self)
            self._json({"profiles": [p.public() for p in profiles.load(cfg).values()],
                        "default": profiles.default_name(cfg),
                        "any_minor": profiles.any_minor(cfg),
                        "auth_on": profiles.auth_on(cfg),
                        "admin_name": profiles.admin_name(cfg),
                        "current": {"name": getattr(prof, "name", ""),
                                    "role": getattr(prof, "role", "adult"),
                                    "admin": bool(getattr(prof, "is_admin", False)),
                                    "adult": _session_adult(cfg, sid, self),
                                    "is_admin_session": _session_admin(cfg, sid, self)}})
        elif path == "/api/approvals":
            # Cross-device queue: the pending danger requests a child raised,
            # visible ONLY to a verified adult so they can approve remotely with
            # full context. Anyone else gets an empty list (never leak the queue).
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            if not _session_adult(cfg, "", self):
                self._json({"pending": [], "adult": False})
            else:
                _prune_sessions()
                items = [{"token": t, "who": s.get("requested_by", ""),
                          "why": s.get("task", ""),
                          "summary": _pending_summary(s.get("pending") or {}),
                          "tool": (s.get("pending") or {}).get("tool", ""),
                          "ts": s.get("ts", 0)}
                         for t, s in sorted(SESSIONS.items(),
                                            key=lambda kv: kv[1].get("ts", 0))
                         if s.get("adult_required")]
                self._json({"pending": items, "adult": True})
        elif path == "/api/approval/poll":
            # The requester's device polls to learn if an adult resolved this on
            # another device, so the answer routes back to where it was asked.
            from urllib.parse import urlparse, parse_qs
            tok = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
            r = RESOLVED.get(tok)
            still = tok in SESSIONS
            if r:
                self._json({"resolved": True, "decision": r.get("decision"),
                            "status": r.get("status"), "answer": r.get("answer", ""),
                            "by": r.get("by", "")})
            else:
                self._json({"resolved": False, "pending": still})
        elif path == "/api/pulse":
            from urllib.parse import urlparse, parse_qs
            sid = (parse_qs(urlparse(self.path).query).get("sid") or [""])[0]
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            _bind_actor(cfg, sid, self)      # thought stream scoped to this person
            self._json(build_pulse(cfg, is_admin=_session_admin(cfg, sid, self)))
        elif path == "/api/memory":
            from urllib.parse import urlparse, parse_qs
            sid = (parse_qs(urlparse(self.path).query).get("sid") or [""])[0]
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            _bind_actor(cfg, sid, self)      # show only THIS profile's visible memory
            self._json(build_memory(cfg))
        elif path == "/api/file":
            from urllib.parse import urlparse, parse_qs, unquote
            qs = parse_qs(urlparse(self.path).query)
            sid = (qs.get("sid") or [""])[0]
            rel = unquote((qs.get("path") or [""])[0])
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            _bind_actor(cfg, sid, self)
            from . import profiles as _pm
            full = (not _pm.auth_on(cfg)) or _session_admin(cfg, sid, self)
            self._json(build_file(cfg, rel, admin=full))
        elif path == "/api/lists":
            # Shared family lists (groceries etc.) — household state, visible
            # to every logged-in profile.
            from . import lists as listsmod
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            self._json({"ok": True, "lists": listsmod.all_lists(cfg)})
        elif path == "/api/approvals/log":
            # The decided-approvals audit trail (#62) — family-visible.
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            self._json({"ok": True, "log": _read_approvals_log(cfg)})
        elif path == "/api/ha":
            self._json(build_ha())
        elif path == "/api/progress":
            from urllib.parse import urlparse, parse_qs
            import time as _t
            rid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
            if rid:
                LAST_POLL[rid] = _t.time()     # liveness: someone is listening
            self._json(_live_snapshot(rid))
        elif path == "/api/stream":
            # SSE: the pushed twin of /api/progress. One event per throttled
            # snapshot, ': ping' comments keep proxies awake, 'done' ends the
            # stream. Each connection holds one thread — fine at family scale.
            from urllib.parse import urlparse, parse_qs
            import queue as _q
            import time as _t
            rid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
            if not rid:
                return self._json({"error": "missing id"}, 400)
            q: "_q.Queue" = _q.Queue(maxsize=200)
            with SSE_LOCK:
                SSE_SUBS.setdefault(rid, []).append(q)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                def _frame(kind, data):
                    self.wfile.write(("data: " + json.dumps(
                        {"kind": kind, "data": data}) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                _frame("snap", _live_snapshot(rid))     # current state right away
                while True:
                    LAST_POLL[rid] = _t.time()          # stream = liveness too
                    try:
                        kind, data = q.get(timeout=15)
                    except _q.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    _frame(kind, data)
                    if kind == "done":
                        break
            except (BrokenPipeError, ConnectionError, OSError):
                pass                                    # client went away — fine
            finally:
                with SSE_LOCK:
                    try:
                        SSE_SUBS.get(rid, []).remove(q)
                    except ValueError:
                        pass
        elif path == "/api/tailscale":
            self._json(self._ts_summary())
        elif path == "/api/backups":
            from urllib.parse import urlparse, parse_qs
            from . import backup, profiles as _pm
            sid = (parse_qs(urlparse(self.path).query).get("sid") or [""])[0]
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            # Admin-scoped like the /api/backup create counterpart: never leak
            # snapshot names/sizes/timestamps to non-admin (incl. minor) sessions.
            if _pm.auth_on(cfg) and not _session_admin(cfg, sid, self):
                self._json({"backups": []})
            else:
                self._json({"backups": backup.list_backups(cfg)})
        elif path == "/api/conversation":
            from urllib.parse import urlparse, parse_qs
            from .conversations import Conversations
            import time as _t
            sid = (parse_qs(urlparse(self.path).query).get("sid") or [""])[0]
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            _bind_actor(cfg, sid, self)      # isolate to the acting profile
            # Pending approvals BELONG to the conversation: without this, the
            # approval card only existed on the device that asked — navigate to
            # the chat from a push on another device and there was nothing to
            # tap. Any device restoring this sid now gets the live card(s).
            pend = []
            if sid:
                for tok, sess in list(SESSIONS.items()):
                    if sess.get("sid") == sid and sess.get("pending"):
                        pend.append({
                            "token": tok,
                            "pending": sess["pending"],
                            "task": sess.get("task", ""),
                            "adult_required": bool(sess.get("adult_required")),
                            "requested_by": sess.get("requested_by", ""),
                            "age_s": round(_t.time() - sess.get("ts", _t.time()))})
            self._json({"turns": Conversations(cfg).history(sid) if sid else [],
                        "pending": pend})
        elif path == "/api/conversations":
            from urllib.parse import urlparse, parse_qs
            from .conversations import Conversations
            qs = parse_qs(urlparse(self.path).query)
            q = (qs.get("q") or [""])[0]
            sid = (qs.get("sid") or [""])[0]
            cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
            _bind_actor(cfg, sid, self)      # sidebar shows only THIS profile's chats
            c = Conversations(cfg)
            self._json({"chats": c.search(q) if q.strip() else c.list()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            blocked = self._origin_blocked()
            if blocked:
                return self._json({"error": f"request blocked: {blocked}"}, 403)
            if self._auth_gate(self.path):
                return self._json({"error": "auth", "need_login": True}, 401)
            if self.path == "/api/login":
                return self._handle_login(self._read_json())
            if self.path == "/api/setup":
                return self._handle_setup(self._read_json())
            if self.path == "/api/password":
                return self._handle_password(self._read_json())
            if self.path == "/api/logout":
                tok = _cookie_header(self)
                if tok:
                    with _PROFILE_GUARD:
                        AUTH.pop(tok, None)
                    _persist_auth()
                return self._send(200, b'{"ok":true}', "application/json",
                                  extra_headers=[("Set-Cookie",
                                   f"{_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax")])
            if self.path in ("/api/config", "/api/persona"):
                # Server settings are ADMIN-only once login is configured.
                body = self._read_json()
                cfg0 = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
                from . import profiles as _pm
                if _pm.auth_on(cfg0) and not _session_admin(cfg0, (body.get("sid") or ""), self):
                    return self._json({"error": "only the admin can change server settings"}, 403)
                if self.path == "/api/config":
                    self._json(save_config_from_ui(body))
                else:
                    self._json(save_persona_from_ui(body))
            elif self.path == "/api/backup":
                body = self._read_json()
                cfg0 = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
                from . import profiles as _pm, backup
                if _pm.auth_on(cfg0) and not _session_admin(cfg0, (body.get("sid") or ""), self):
                    return self._json({"error": "only the admin can run backups"}, 403)
                p = backup.create_backup(cfg0)
                self._json({"ok": bool(p), "name": (p.name if p else ""),
                            "backups": backup.list_backups(cfg0)})
            elif self.path == "/api/ask":
                self._handle_ask(self._read_json())
            elif self.path == "/api/note":
                self._handle_note(self._read_json())
            elif self.path == "/api/remember":
                self._handle_remember(self._read_json())
            elif self.path == "/api/memory/share":
                self._json(self._memory_share(self._read_json()))
            elif self.path == "/api/memory/delete":
                self._json(self._memory_delete(self._read_json()))
            elif self.path == "/api/profile/select":
                self._json(self._profile_select(self._read_json()))
            elif self.path == "/api/profiles/save":
                self._json(self._profiles_save(self._read_json()))
            elif self.path == "/api/profile/push_allow":
                self._json(self._profile_push_allow(self._read_json()))
            elif self.path in ("/api/lists/add", "/api/lists/done",
                               "/api/lists/remove"):
                # Shared family lists: any logged-in profile may edit; the
                # session's profile name is recorded as `by` on additions.
                from . import lists as listsmod
                body = self._read_json()
                cfg0 = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
                who = _actor_name(cfg0, (body.get("sid") or ""), self)
                name = str(body.get("list") or "groceries")
                try:
                    if self.path.endswith("/add"):
                        items = listsmod.add_item(cfg0, body.get("text", ""),
                                                  by=who, name=name)
                    elif self.path.endswith("/done"):
                        items = listsmod.set_done(cfg0, int(body.get("index", -1)),
                                                  bool(body.get("done")), name=name)
                    else:
                        items = listsmod.remove_item(cfg0, int(body.get("index", -1)),
                                                     name=name)
                    self._json({"ok": True, "items": items})
                except (ValueError, TypeError) as exc:
                    self._json({"error": str(exc)})
            elif self.path == "/api/approve":
                self._handle_approve(self._read_json())
            elif self.path == "/api/jobs/save":
                self._json(self._job_save(self._read_json()))
            elif self.path == "/api/jobs/delete":
                self._json(self._job_delete(self._read_json()))
            elif self.path == "/api/jobs/toggle":
                self._json(self._job_toggle(self._read_json()))
            elif self.path == "/api/jobs/run":
                self._json(self._job_run(self._read_json()))
            elif self.path in ("/api/tailscale/up", "/api/tailscale/bind"):
                # Network binding is a SYSTEM setting: /api/tailscale/bind
                # persists cfg.bind_host (same field /api/config admin-gates) and
                # /api/tailscale/up changes network state. Gate both to the admin
                # so a non-admin can't expose the server on the tailnet or flip
                # the bind host out from under the household manager.
                body = self._read_json()
                cfg0 = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
                from . import profiles as _pm
                if _pm.auth_on(cfg0) and not _session_admin(cfg0, (body.get("sid") or ""), self):
                    return self._json({"error": "only the admin can change network binding"}, 403)
                if self.path == "/api/tailscale/up":
                    self._json(self._ts_up())
                else:
                    self._json(self._ts_bind(body))
            elif self.path == "/api/cancel":
                rid = (self._read_json().get("rid") or "").strip()
                if rid:
                    _cancel_add(rid)
                self._json({"ok": True})
            elif self.path in ("/api/think", "/api/dream"):
                # A mind cycle is a SYSTEM/cross-profile operation: think() reads
                # every profile's STM and dream() consolidates each family
                # member's STM into their own memory and writes the ADMIN-only
                # journal (which _mind_view/build_pulse only let the admin READ).
                # Gate the write side to the admin too, mirroring
                # /api/config, /api/backup and /api/tailscale/*.
                body = self._read_json()
                cfg0 = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
                from . import profiles as _pm
                if _pm.auth_on(cfg0) and not _session_admin(cfg0, (body.get("sid") or ""), self):
                    return self._json({"error": "only the admin can run the mind cycle"}, 403)
                if self.path == "/api/think":
                    self._json(self._mind_think())
                else:
                    self._json(self._mind_dream())
            elif self.path == "/api/conversation/rename":
                self._json(self._conv_action("rename", self._read_json()))
            elif self.path == "/api/conversation/delete":
                self._json(self._conv_action("delete", self._read_json()))
            elif self.path == "/api/conversation/pin":
                self._json(self._conv_action("pin", self._read_json()))
            elif self.path == "/api/push/subscribe":
                self._json(self._push_subscribe(self._read_json()))
            elif self.path == "/api/push/unsubscribe":
                self._json(self._push_unsubscribe(self._read_json()))
            elif self.path == "/api/push/test":
                self._json(self._push_test())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 200)

    def _conv_action(self, action: str, payload) -> Dict[str, Any]:
        from .conversations import Conversations
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        sid = (payload.get("sid") or "").strip()
        if not sid:
            return {"error": "sid required"}
        _bind_actor(cfg, sid, self)      # act only within the caller's namespace
        c = Conversations(cfg)
        if action == "rename":
            c.set_title(sid, payload.get("title") or "")
        elif action == "delete":
            c.clear(sid)
        elif action == "pin":
            c.set_pinned(sid, bool(payload.get("pinned")))
        return {"ok": True}

    # ------------------------------------------------------------------ #
    # Web Push (PWA notifications)
    # ------------------------------------------------------------------ #
    def _push_config(self):
        from . import push
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        enabled = bool(getattr(cfg, "push_enabled", True)) and push.available()
        return {
            "enabled": enabled,
            "available": push.available(),
            "public_key": push.public_key(cfg) if enabled else "",
            "subscribed": push.subscription_count(cfg),
        }

    def _push_subscribe(self, payload):
        from . import push
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        if not push.available():
            return {"ok": False, "error": "push unavailable (cryptography missing)"}
        # Tag the device with the profile logged in on it, so personal pushes only
        # reach that person. 'sticky' = the 'keep me logged in on this device' opt.
        actor = _actor_name(cfg, (payload.get("sid") or ""), self)
        sticky = bool(payload.get("sticky", True))
        ok = push.add_subscription(cfg, payload.get("subscription") or payload,
                                   profile=actor, sticky=sticky)
        return {"ok": ok, "subscribed": push.subscription_count(cfg), "profile": actor}

    def _push_unsubscribe(self, payload):
        from . import push
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        ep = (payload.get("endpoint") or "").strip()
        if ep:
            push.remove_subscription(cfg, ep)
        return {"ok": True, "subscribed": push.subscription_count(cfg)}

    def _push_test(self):
        from . import push
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        # Test goes to the CURRENT profile's devices only (proves personal routing).
        actor = _actor_name(cfg, "", self)
        to = {actor} if actor else None
        return push.send(cfg, "Lara", "Notifications are live. \U0001F525",
                         url="/", tag="test", to=to)

    def _handle_ask(self, payload):
        task = (payload.get("task") or "").strip()
        if not task:
            return self._json({"error": "empty task"})
        from .pipeline import Pipeline
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        apply_env_to_process(load_env_file())
        # Resolve WHO is asking up front so their chat activity is tagged with
        # them in STM — the dream cycle then consolidates it into THEIR memory,
        # not the shared household pool (no activity spillover between profiles).
        actor = _bind_actor(cfg, (payload.get("sid") or ""), self)
        try:
            mind.record(cfg, "chat", f"operator asked: {task[:200]}",
                        meta={"actor": actor} if actor else None)
        except Exception:
            pass
        use_tools = payload.get("tools", getattr(cfg, "tools_enabled", True))
        rid = (payload.get("rid") or "").strip()
        sid = (payload.get("sid") or "").strip()
        geo = payload.get("geo") if isinstance(payload.get("geo"), dict) else None
        # Attached photos / video frames: base64 JPEG strings (client already
        # downscaled). Strip any dataURL header, cap count and per-image size.
        images = []
        for im in (payload.get("images") or [])[:6]:
            if not isinstance(im, str):
                continue
            if im.startswith("data:"):
                im = im.split(",", 1)[-1]
            if 0 < len(im) <= 12_000_000:
                images.append(im)
        from .conversations import Conversations
        # actor is already bound (above): cfg._actor scopes the transcript AND the
        # pipeline's cross-conversation memory tool + personal recall to this person.
        conv = Conversations(cfg) if sid else None

        # Permanent record of HOW the answer happened: every phase Lara went
        # through, with the deliberation she streamed during it. Returned with
        # the result so the UI can fold it into a clickable "how I got here"
        # block under the answer — transparency that survives the live view.
        import time as _time
        trace: list = []
        t_start = _time.time()

        def prog(msg):
            trace.append({"phase": msg, "t": round(_time.time() - t_start, 1)})
            if rid:
                trail = PROGRESS.setdefault(rid, [])
                if not trail or trail[-1] != msg:   # collapse repeats
                    trail.append(msg)
                # A new phase (recalling/sensing a tool/thinking again) means the
                # previous streamed prose was a dead end — clear it so the live
                # preview only ever shows the answer currently being written.
                STREAM[rid] = ""
                THINKING[rid] = ""   # fresh step -> fresh deliberation trace
                _sse_push(rid, "snap", _live_snapshot(rid))   # phases never coalesce

        # SSE side-channel: every sink below nudges a THROTTLED full snapshot to
        # any /api/stream subscribers, so pushed clients see exactly what the
        # 500ms poll would have shown — one payload shape, two transports.
        _emit_last = {"t": 0.0}

        def _emit(force: bool = False):
            if not rid:
                return
            now = _time.time()
            if not force and now - _emit_last["t"] < 0.15:
                return
            _emit_last["t"] = now
            _sse_push(rid, "snap", _live_snapshot(rid))

        def on_tool(name, args, obs):
            if not rid:
                return
            try:
                a = json.dumps(args, default=str)[:600]
            except Exception:
                a = str(args)[:600]
            TOOLS_LIVE.setdefault(rid, []).append(
                {"tool": name, "args": a, "out": str(obs)[:2000],
                 "t": round(_time.time() - t_start, 1)})
            _emit(force=True)          # a tool event must never be coalesced away

        def sink(delta):
            if rid:
                STREAM[rid] = STREAM.get(rid, "") + delta
                _emit()

        def _sink_reset():
            # Attempt-scoped streaming (review 2.7): the router calls this when
            # a retry/escalation abandons a partially-streamed answer — the
            # doomed partial used to stay in the bubble and the new attempt
            # CONCATENATED onto it. The client polls the full partial each
            # tick, so clearing here redraws the bubble clean.
            if rid:
                STREAM[rid] = ""
                _emit(force=True)
        sink.reset = _sink_reset

        def think_sink(delta):
            if trace:   # file the deliberation under the phase it belongs to
                row = trace[-1]
                if len(row.get("think", "")) < 8000:
                    row["think"] = (row.get("think", "") + delta)[:8000]
            if rid:
                THINKING[rid] = THINKING.get(rid, "") + delta
                _emit()

        def ctx_sink(tokens):
            if rid:
                CTX[rid] = int(tokens)
                _emit()

        # Checked on every streamed line (thinking + content) in the provider, so
        # Stop interrupts even during the long silent reasoning phase. ONLY the
        # operator's explicit Stop cancels — a locked/backgrounded phone must
        # never kill an answer (it finishes, persists, and pushes instead).
        cancel_fn = (lambda: _is_cancelled(rid)) if rid else None

        # Transcript gets a marker, never megabytes of base64.
        task_for_log = task + (f" [+{len(images)} image(s) attached]" if images else "")

        def record(answer, res=None):
            if conv and (answer or "").strip():
                conv.append(sid, "user", task_for_log)
                conv.append(sid, "assistant", answer,
                            meta=_turn_meta(res) if res else None)
                # Advance the DURABLE rolling summary off-path (review 1.2): a
                # no-op for short chats; for long ones it folds aged turns in
                # with one background call so recall never summarizes anything.
                def _roll():
                    try:
                        out = pipe.update_rolling_summary(conv, sid)
                        if out.startswith("folded"):
                            print(f"[conv] rolling summary {sid[:8]}: {out}")
                    except Exception:
                        pass
                _threading.Thread(target=_roll, name="anvil-roll", daemon=True).start()

        # Serialize asks within one conversation so replies arrive in order
        # (a quick follow-up used to overtake a long-running first question).
        lock = _sid_lock(sid or "anon")
        if not lock.acquire(blocking=False):
            prog("answering your previous message first")
            lock.acquire()
        from .providers import GenerationCancelled, ProviderError
        pipe = Pipeline(cfg)
        try:
            # History is read AFTER the lock: a queued follow-up must see the
            # exchange it just waited for. `packed` = stored rolling summary +
            # the verbatim turns it doesn't cover — ZERO model calls here (the
            # old path re-summarized the whole head on every turn of a long
            # chat, a multi-second model call before the first answer token).
            from .context import COMPACTION_PREAMBLE
            history = conv.packed(sid, COMPACTION_PREAMBLE) if conv else None
            if use_tools:
                adult = _session_adult(cfg, sid, self)
                res = pipe.agent_start(task, progress=prog, history=history,
                                       stream=sink, geo=geo, images=images,
                                       cancel=cancel_fn,
                                       think_stream=think_sink, on_ctx=ctx_sink,
                                       adult=adult, on_tool=on_tool)
                res["trace"] = trace
                if res.get("status") == "cancelled":
                    return self._json({"status": "cancelled"})
                if res.get("status") == "done":       # not paused for approval
                    record(res.get("answer", ""), res)
                    _push_answer(cfg, res.get("answer", ""), to=getattr(cfg, "_actor", ""), sid=sid)
                elif res.get("status") == "approve":
                    who = ""
                    try:
                        who = _resolve_profile(cfg, sid, self)[0].name
                    except Exception:
                        who = ""
                    _push_approval(cfg, (res.get("pending") or {}),
                                   who=who, why=task)
                    # Persist the operator's turn NOW: if they deny (or never
                    # answer) this approval, the exchange used to vanish from
                    # the transcript entirely.
                    if conv:
                        conv.append(sid, "user", task_for_log)
                return self._json(_agent_response(
                    res, sid=sid, task=task,
                    user_recorded=bool(conv and res.get("status") == "approve"),
                    requested_by=(_resolve_profile(cfg, sid, self)[0].name
                                  if res.get("status") == "approve" else "")))
            r = pipe.run(task, plan=bool(payload.get("plan")),
                         verify=bool(payload.get("verify", True)), progress=prog,
                         history=history, stream=sink, geo=geo, images=images)
            record(r.answer)
            _push_answer(cfg, r.answer, to=getattr(cfg, "_actor", ""), sid=sid)
            self._json({
                "mode": "chat", "status": "done", "answer": r.answer,
                "rung": r.rung_name, "verdict": r.critic_verdict,
                "escalations": r.escalations, "recalled": r.recalled,
                "notes_written": r.notes_written, "cost": r.est_cost_usd,
                "evolved": r.evolved,
            })
        except GenerationCancelled:
            # Stop pressed mid-generation: don't record, don't push. The lock
            # releases in finally, so the operator's NEXT prompt runs at once.
            self._json({"status": "cancelled"})
        except ProviderError as exc:
            # Graceful degradation, honestly WORDED for its actual cause: a
            # spent budget reads as a budget ("run ollama serve" advice from
            # the local era confused the operator when the real reason was the
            # daily cap). Never fabricate, never raw-traceback.
            if "cap-hit" in str(exc) or "month-cap" in str(exc):
                msg = ("Today's model budget is used up, so I have to sit the "
                       "rest of the day out — I'll be back after midnight. "
                       "(The admin can raise `daily_cost_cap_usd` in Setup if "
                       "this keeps happening.)")
            else:
                msg = ("I can't reach my language model right now — the cloud "
                       "may be having a brief outage. Try again in a moment.")
            self._json({"mode": "chat", "status": "done", "degraded": True,
                        "trace": trace, "answer": msg})
        finally:
            lock.release()
            if rid:
                # Tell pushed clients the turn is over (with the final snapshot)
                # BEFORE the state is torn down, then drop their queues.
                _sse_push(rid, "done", _live_snapshot(rid))
                with SSE_LOCK:
                    SSE_SUBS.pop(rid, None)
                PROGRESS.pop(rid, None)
                STREAM.pop(rid, None)
                LAST_POLL.pop(rid, None)
                THINKING.pop(rid, None)
                CTX.pop(rid, None)
                TOOLS_LIVE.pop(rid, None)
                _cancel_clear(rid)

    def _handle_login(self, payload):
        from . import profiles
        import time as _t
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        username = (payload.get("username") or payload.get("name") or "").strip()
        password = str(payload.get("password") or "")
        remember = bool(payload.get("remember"))
        # Rate limit BEFORE any verification: 5 failures per 15 minutes per
        # account name locks the form (brute-force / kid-curiosity defense),
        # PLUS 20 failures per 15 minutes per CLIENT IP across all usernames
        # (credential-stuffing defense for the public-front-door era; the IP
        # is proxy-aware via trusted_proxy + X-Forwarded-For).
        key = profiles._norm_username(username) or "?"
        ipkey = "ip:" + (_client_ip(cfg, self) or "?")
        now = _t.time()
        with _PROFILE_GUARD:
            fails = [t for t in _LOGIN_FAILS.get(key, []) if now - t < 900]
            _LOGIN_FAILS[key] = fails
            ipfails = [t for t in _LOGIN_FAILS.get(ipkey, []) if now - t < 900]
            _LOGIN_FAILS[ipkey] = ipfails
            if len(fails) >= 5 or len(ipfails) >= 20:
                first = fails[0] if len(fails) >= 5 else ipfails[0]
                wait = int(900 - (now - first)) + 1
                return self._json({"error": "too many attempts — try again in "
                                   f"{max(1, wait // 60)} min"}, 429)
        prof = profiles.authenticate(cfg, username, password)
        if not prof:
            with _PROFILE_GUARD:
                _LOGIN_FAILS.setdefault(key, []).append(now)
                _LOGIN_FAILS.setdefault(ipkey, []).append(now)
            # One constant message — never reveal whether the USERNAME exists.
            return self._json({"error": "wrong username or password"}, 401)
        with _PROFILE_GUARD:
            _LOGIN_FAILS.pop(key, None)
        tok = _mint_auth(prof.name, prof.role, remember=remember)
        # HttpOnly cookie — the token never touches JS; SameSite=Lax keeps it
        # same-site. REMEMBER ME decides the lifetime: a persistent Max-Age
        # cookie (~30 days) on a trusted family device, or a SESSION cookie
        # (dies with the browser; server-side TTL 12h) on a shared/guest one.
        lifetime = f" Max-Age={_AUTH_TTL};" if remember else ""
        # Behind the HTTPS front door the session cookie is marked Secure so
        # it can never leak over a plain-http downgrade.
        secure = " Secure;" if _via_https(cfg, self) else ""
        self._send(200, json.dumps({"ok": True, "name": prof.name,
                                    "role": prof.role,
                                    "username": prof.username,
                                    "needs_password": (prof.is_adult
                                                       and not prof.has_password)}).encode(),
                   "application/json",
                   extra_headers=[("Set-Cookie",
                    f"{_COOKIE}={tok}; Path=/;{lifetime}{secure} HttpOnly; SameSite=Lax")])

    def _handle_setup(self, payload):
        """First-run wizard: create the household roster in one shot. ONLY
        works while the install is genuinely fresh (no profiles) — once any
        account exists this endpoint is a 403, so it can never clobber or
        backdoor a real household. The admin is auto-signed-in on success."""
        from . import profiles
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        if not profiles.needs_setup(cfg):
            return self._json({"error": "setup is already complete"}, 403)
        admin = payload.get("admin") or {}
        pw = str(admin.get("password") or "")
        if len(pw) < 6:
            return self._json({"error": "password must be at least 6 characters"}, 400)
        if not (admin.get("name") or "").strip():
            return self._json({"error": "your name is required"}, 400)
        members = [m for m in (payload.get("members") or []) if isinstance(m, dict)]
        if not profiles.create_household(cfg, admin, members):
            return self._json({"error": "could not create the household"}, 400)
        prof = profiles.load(cfg)[(admin.get("name") or "").strip()]
        tok = _mint_auth(prof.name, prof.role, remember=True)
        secure = " Secure;" if _via_https(cfg, self) else ""
        self._send(200, json.dumps({"ok": True, "name": prof.name}).encode(),
                   "application/json",
                   extra_headers=[("Set-Cookie",
                    f"{_COOKIE}={tok}; Path=/; Max-Age={_AUTH_TTL};{secure} HttpOnly; SameSite=Lax")])

    def _handle_password(self, payload):
        """Change YOUR password (verify the current credential first), or —
        as the admin — set a family member's without one (kid forgot theirs)."""
        from . import profiles
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        a = _authed(self)
        if not a:
            return self._json({"error": "sign in first"}, 401)
        me = profiles.get(cfg, a.get("name", ""))
        target = (payload.get("name") or a.get("name", "")).strip()
        new = str(payload.get("new") or "")
        if len(new) < 6:
            return self._json({"error": "new password must be at least 6 characters"}, 400)
        if target != a.get("name"):
            if not (me and me.is_admin):
                return self._json({"error": "only the admin can set another "
                                   "member's password"}, 403)
        else:
            cur = str(payload.get("current") or "")
            has_cred = me and (me.has_password or me.has_pin)
            if has_cred and not profiles.authenticate(cfg, target, cur):
                return self._json({"error": "current password is wrong"}, 403)
        if not profiles.set_password(cfg, target, new):
            return self._json({"error": "unknown profile"}, 400)
        return self._json({"ok": True})

    def _profile_select(self, payload):
        """Bind a chat session to a family profile. Selecting an adult profile
        that has a PIN requires the correct PIN and grants adult authority for a
        TTL; a minor (or a PIN-less profile) binds immediately."""
        import time as _t
        from . import profiles
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        sid = (payload.get("sid") or "").strip()
        name = (payload.get("name") or "").strip()
        prof = profiles.get(cfg, name)
        if not sid or not prof:
            return {"error": "unknown profile"}
        rec = {"name": prof.name, "role": prof.role}
        if prof.is_adult and prof.has_pin:
            if not profiles.verify_pin(prof.pin_hash, str(payload.get("pin") or "")):
                return {"error": "wrong PIN"}
            rec["adult_until"] = _t.time() + _ADULT_TTL
        with _PROFILE_GUARD:
            SESSION_PROFILE[sid] = rec
        return {"ok": True, "name": prof.name, "role": prof.role,
                "adult": _session_adult(cfg, sid, self)}

    def _profiles_save(self, payload):
        """Create/replace the family profile list (Setup). Only an adult session
        may change profiles. New/changed PINs come in as plaintext -> hashed;
        an omitted PIN for an existing profile keeps its current hash."""
        from . import profiles
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        sid = (payload.get("sid") or "").strip()
        existing = profiles.load(cfg)
        # Guard: managing profiles is an ADMIN action. Once login is set up, only
        # the admin may change the family. (Fresh setup — no login yet — is the
        # bootstrap where the first profile/admin is created, so it's allowed.)
        if profiles.auth_on(cfg) and not _session_admin(cfg, sid, self):
            return {"error": "only the admin can manage profiles"}
        out = []
        for p in payload.get("profiles", []):
            if not isinstance(p, dict) or not (p.get("name") or "").strip():
                continue
            name = p["name"].strip()
            role = "minor" if p.get("role") == "minor" else "adult"
            pin = str(p.get("pin") or "")
            if role == "minor":
                pin_hash = ""
            elif pin:
                pin_hash = profiles.hash_pin(pin)
            else:
                pin_hash = existing[name].pin_hash if name in existing else ""
            out.append({"name": name, "role": role, "pin_hash": pin_hash})
        profiles.save(cfg, out, default=(payload.get("default") or "").strip())
        return {"ok": True, "profiles": [Prof.public() for Prof in profiles.load(cfg).values()]}

    def _handle_approve(self, payload):
        sess = SESSIONS.pop(payload.get("token"), None)
        if not sess:
            return self._json({"error": "unknown or expired approval token"})
        _unpersist_approval(payload.get("token") or "")   # consumed (re-persisted if put back)
        from .pipeline import Pipeline
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        apply_env_to_process(load_env_file())
        decision = payload.get("decision", "deny")
        sid = sess.get("sid", "")
        # Identity gate: a danger action raised by a minor/unverified session can
        # only be APPROVED by a verified adult. Accept either an already-adult
        # session or a correct adult PIN supplied with this approval.
        if decision in ("approve", "always") and sess.get("adult_required"):
            from . import profiles
            pin = str(payload.get("pin") or "")
            ok_adult = _session_adult(cfg, sid, self)
            if not ok_adult and pin:
                ok_adult = any(p.is_adult and p.has_pin and profiles.verify_pin(p.pin_hash, pin)
                               for p in profiles.load(cfg).values())
            if not ok_adult:
                SESSIONS[payload.get("token")] = sess     # put it back; not consumed
                _persist_approval(payload.get("token") or "", sess)
                return self._json({"error": "an adult must approve this",
                                   "adult_required": True})
        if decision == "always":
            # Operator taught us this exact command: remember it (shell only —
            # other danger tools have no stable "same action" identity), then
            # run it like a normal approval.
            try:
                from . import tools as toolsmod
                pending = sess.get("pending") or {}
                cmd = (pending.get("args") or {}).get("cmd") or ""
                if pending.get("tool") == "shell" and cmd:
                    toolsmod.allowlist_add(cfg, cmd)
            except Exception:
                pass
            decision = "approve"
        # AUDIT LOG (#62): every decision — allow or deny — lands in a durable
        # family-visible log the moment it's made: who asked, what would run,
        # who decided. Best-effort; a log failure must never block the action.
        try:
            import time as _time0
            pend0 = sess.get("pending") or {}
            _log_approval_decision(cfg, {
                "ts": _time0.time(),
                "requested_by": sess.get("requested_by", "") or "someone",
                "tool": pend0.get("tool", ""),
                "what": str((pend0.get("args") or {}).get("cmd")
                            or (pend0.get("args") or {}))[:160],
                "decision": "allowed" if decision == "approve" else "denied",
                "decided_by": _actor_name(cfg, (payload.get("sid") or ""), self)
                              or "someone",
            })
        except Exception:
            pass
        # Scope the resumed turn to the REQUESTER (who may be a child), not the
        # approving adult — the transcript + personal recall belong to whoever
        # asked, even when a different family member approved the danger action.
        from . import profiles as _profmod
        try:
            cfg._actor = sess.get("requested_by", "") if _profmod.auth_on(cfg) else ""
        except Exception:
            pass
        task = sess.get("task", "")
        # LIVE plumbing for the resumed turn (the fix for "approve, then stare
        # at a frozen card"): the client sends a rid and follows it over
        # SSE/poll exactly like a normal ask — the terminal shows the command
        # the moment it runs, and the continued answer streams as it's written.
        rid = str(payload.get("rid") or "")
        import time as _time
        t0 = _time.time()

        def _emit(force=False):
            if rid:
                _sse_push(rid, "snap", _live_snapshot(rid))

        def prog(msg):
            if rid:
                trail = PROGRESS.setdefault(rid, [])
                if not trail or trail[-1] != msg:
                    trail.append(msg)
                STREAM[rid] = ""
                _emit(True)

        def sink(delta):
            if rid:
                STREAM[rid] = STREAM.get(rid, "") + delta
                _emit()

        def _sink_reset():
            if rid:
                STREAM[rid] = ""
                _emit(True)
        sink.reset = _sink_reset

        def on_tool(name, args, obs):
            if not rid:
                return
            try:
                a = json.dumps(args, default=str)[:600]
            except Exception:
                a = str(args)[:600]
            TOOLS_LIVE.setdefault(rid, []).append(
                {"tool": name, "args": a, "out": str(obs)[:2000],
                 "t": round(_time.time() - t0, 1)})
            _emit(True)
        try:
            res = Pipeline(cfg).agent_resume(
                sess["messages"], sess["pending"], decision,
                adult=_session_adult(cfg, sid, self), task=task,
                progress=(prog if rid else None),
                stream=(sink if rid else None),
                on_tool=(on_tool if rid else None))
        finally:
            if rid:
                _sse_push(rid, "done", _live_snapshot(rid))
                with SSE_LOCK:
                    SSE_SUBS.pop(rid, None)
                PROGRESS.pop(rid, None)
                STREAM.pop(rid, None)
                TOOLS_LIVE.pop(rid, None)
                LAST_POLL.pop(rid, None)
        user_recorded = bool(sess.get("user_recorded"))
        if res.get("status") == "done" and (res.get("answer") or "").strip():
            if sid:
                from .conversations import Conversations
                c = Conversations(cfg)
                if not user_recorded:     # already persisted at approval time
                    c.append(sid, "user", task)
                c.append(sid, "assistant", res.get("answer", ""),
                         meta=_turn_meta(res))
            _push_answer(cfg, res.get("answer", ""), to=getattr(cfg, "_actor", ""), sid=sid)
        elif res.get("status") == "approve":       # chained: needs another OK
            _push_approval(cfg, (res.get("pending") or {}),
                           who=sess.get("requested_by", ""), why=task)
        # Cross-device routing: stamp the outcome under the ORIGINAL token so the
        # requester's device (which may be a child's, still showing the pending
        # card) can poll and pick up what the adult decided.
        import time as _t
        _prune_resolved()
        RESOLVED[payload.get("token")] = {
            "decision": decision, "status": res.get("status"),
            "answer": res.get("answer", ""),
            "by": (_resolve_profile(cfg, sid, self)[0].name
                   if sess.get("adult_required") else ""),
            "ts": _t.time()}
        self._json(_agent_response(res, sid=sid, task=task,
                                   user_recorded=user_recorded,
                                   requested_by=sess.get("requested_by", "")))

    def _memory_share(self, payload):
        """Set who one of MY notes is shared with. Only the owner may reshare
        their own memory. `with`: list of profile names, or ["*"] for everyone,
        or [] to make it private again."""
        from .memory import MemoryStore
        from . import profiles
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        actor = _bind_actor(cfg, (payload.get("sid") or ""), self)
        if not actor:
            return {"error": "sign in to share memory"}
        name = (payload.get("name") or "").strip()
        raw = payload.get("with") or []
        if not isinstance(raw, list):
            return {"error": "bad request"}
        valid = {p.name for p in profiles.load(cfg).values()}
        targets = [w for w in raw if w == "*" or w in valid]
        note = MemoryStore(cfg).share(name, targets, by_actor=actor)
        if note is None:
            return {"error": "not found, or it isn't yours to share"}
        return {"ok": True, "name": note.name, "shared_with": note.shared_with}

    def _profile_push_allow(self, payload):
        """The acting profile sets who may send THEM reminders (their own choice —
        not the admin's). Allowing someone surfaces you as a push target for them."""
        from . import profiles
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        me = _actor_name(cfg, (payload.get("sid") or ""), self)
        if not me:
            return {"error": "sign in to change this"}
        allow = payload.get("allow") or []
        if not isinstance(allow, list):
            return {"error": "bad request"}
        if not profiles.set_push_allow(cfg, me, [str(a) for a in allow]):
            return {"error": "profile not found"}
        p = profiles.get(cfg, me)
        return {"ok": True, "push_allow": list(getattr(p, "push_allow", []))}

    def _memory_delete(self, payload):
        """Forget one of MY notes — the memory-mirror right. Only the note's owner
        may delete it. A single-user install (no login) can delete anything."""
        from .memory import MemoryStore
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        actor = _bind_actor(cfg, (payload.get("sid") or ""), self)
        name = (payload.get("name") or "").strip()
        ok = MemoryStore(cfg).delete(name, by_actor=(actor or None))
        return {"ok": bool(ok)} if ok else {"error": "not found, or not yours to delete"}

    def _handle_remember(self, payload):
        text = (payload.get("text") or "").strip()
        if not text:
            return self._json({"error": "empty"})
        from .memory import MemoryStore
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        # "Remember (about you)" is a PERSONAL fact — bind the actor so it's
        # owned by (and private to) whoever asked, not dropped in shared memory.
        _bind_actor(cfg, (payload.get("sid") or ""), self)
        note = MemoryStore(cfg).write(text, type="profile")
        self._json({"ok": True, "name": note.name})

    def _sched(self):
        from .scheduler import Scheduler
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        return Scheduler(cfg), cfg

    def _job_owner_ok(self, cfg, job, sid):
        """May the acting session manage ``job``? Yes when login is off
        (single-user), or the actor owns the job, or the actor is the household
        admin. Mirrors the ownership/admin gating on the memory + system paths so
        a minor can't touch another member's scheduled jobs."""
        from . import profiles as _pm
        if not _pm.auth_on(cfg):
            return True
        owner = getattr(job, "owner", "") or ""
        return owner == _actor_name(cfg, sid, self) or _session_admin(cfg, sid, self)

    def _jobs_list(self, sid=""):
        from . import profiles as _pm
        sched, cfg = self._sched()
        jobs = sched.load_jobs()
        if _pm.auth_on(cfg) and not _session_admin(cfg, sid, self):
            me = _actor_name(cfg, sid, self)
            jobs = [j for j in jobs if (getattr(j, "owner", "") or "") == me]
        return {"jobs": [_job_public(j) for j in jobs]}

    def _job_save(self, payload):
        from .scheduler import Scheduler, Job, Cron
        name = (payload.get("name") or "").strip().replace(" ", "-")
        if not name:
            return {"error": "name is required"}
        cron = (payload.get("cron") or "").strip()
        try:
            Cron.parse(cron)
        except Exception as exc:
            return {"error": f"invalid cron '{cron}': {exc}"}
        sched, cfg0 = self._sched()
        sid = (payload.get("sid") or "")
        owner = _actor_name(cfg0, sid, self)
        # Block clobbering someone else's job: if a job of this name already
        # exists and belongs to another member, only its owner or the admin may
        # overwrite it (matches the delete/toggle/run gate).
        existing = next((j for j in sched.load_jobs() if j.name == name), None)
        if existing is not None and not self._job_owner_ok(cfg0, existing, sid):
            return {"error": "not your job"}
        # notify targeting honors each person's push_allow — the same consent
        # gate as the remind tool, so the raw API can't route pushes to someone
        # who hasn't allowed you (parent -> child stays automatic in can_remind).
        notify = (payload.get("notify") or "").strip()
        from . import profiles as _pm
        if (notify and _pm.auth_on(cfg0) and notify != owner
                and not _pm.can_remind(cfg0, owner, notify)
                and not _session_admin(cfg0, sid, self)):
            return {"error": f"{notify} hasn't allowed notifications from you"}
        job = Job(name=name, cron=cron,
                  inputs={"prompt": payload.get("prompt", "")},
                  notify=(payload.get("notify") or None),
                  min_rung=int(payload.get("min_rung", 0) or 0),
                  enabled=bool(payload.get("enabled", True)),
                  owner=owner)
        sched.write_job(job)
        return {"ok": True, "jobs": [_job_public(j) for j in sched.load_jobs()]}

    def _job_delete(self, payload):
        sched, cfg = self._sched()
        name = (payload.get("name") or "").strip()
        sid = (payload.get("sid") or "")
        job = next((j for j in sched.load_jobs() if j.name == name), None)
        if job is not None and not self._job_owner_ok(cfg, job, sid):
            return {"error": "not your job"}
        try:
            sched.remove_job(name)
        except OSError as exc:
            return {"error": f"could not delete: {exc}",
                    "jobs": [_job_public(j) for j in sched.load_jobs()]}
        return {"ok": True, "jobs": [_job_public(j) for j in sched.load_jobs()]}

    def _job_toggle(self, payload):
        from .scheduler import Job
        sched, cfg = self._sched()
        name = (payload.get("name") or "").strip()
        sid = (payload.get("sid") or "")
        for j in sched.load_jobs():
            if j.name == name:
                if not self._job_owner_ok(cfg, j, sid):
                    return {"error": "not your job"}
                j.enabled = bool(payload.get("enabled", not j.enabled))
                sched.write_job(j)
                break
        return {"ok": True, "jobs": [_job_public(x) for x in sched.load_jobs()]}

    def _job_run(self, payload):
        from .pipeline import Pipeline
        from .comms import notify
        sched, cfg = self._sched()
        name = (payload.get("name") or "").strip()
        job = next((j for j in sched.load_jobs() if j.name == name), None)
        if not job:
            return {"error": f"no job named {name}"}
        if not self._job_owner_ok(cfg, job, (payload.get("sid") or "")):
            return {"error": "not your job"}
        apply_env_to_process(load_env_file())
        prompt = job.inputs.get("prompt") or job.name
        # Run the job AS its owner (mirror _run_scheduled_job): the manual
        # UI 'run' path was actorless, so identity-aware age gating and the
        # operator card differed from the scheduled path for the same job.
        if getattr(job, "owner", ""):
            cfg._actor = job.owner
        res = Pipeline(cfg).run(prompt, min_rung=job.min_rung)
        try:
            mind.record(cfg, "action",
                        f"did job '{job.name}' -> {(res.answer or '')[:140]}")
        except Exception:
            pass
        if job.notify == "discord":
            notify(cfg.discord_webhook_url, f"**{job.name}**\n{res.answer}")
        return {"ok": True, "answer": res.answer, "rung": res.rung_name}

    def _ts_summary(self):
        from . import tailscale as ts
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        s = ts.Tailscale().summary()
        s["bind_host"] = getattr(cfg, "bind_host", "127.0.0.1")
        s["server_port"] = getattr(cfg, "server_port", 8765)
        return s

    def _ts_up(self):
        from . import tailscale as ts
        return ts.Tailscale().up()

    def _ts_bind(self, payload):
        from . import tailscale as ts
        host = (payload.get("host") or "").strip()
        if not host:
            host = ts.Tailscale().self_ip()
        if not host:
            return {"error": "no Tailscale IP found — sign in to Tailscale first"}
        save_config_from_ui({"bind_host": host})
        localish = host in ("127.0.0.1", "localhost")
        return {"ok": True, "bind_host": host, "note": (
            "restart ANVIL to bind to " + host if not localish
            else "restart ANVIL to return to localhost-only")}

    def _mind_cfg(self):
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        apply_env_to_process(load_env_file())
        return cfg

    def _mind_view(self, sid: str = ""):
        # Same privacy gate as build_pulse: the journal (dreams + self-dev inner
        # monologue) is ADMIN-only, and the STM thought stream is scoped to the
        # viewing profile (their meta.actor + ambient) so one family member's
        # activity never leaks to another. Without this, any authenticated
        # non-admin could read the full journal + everyone's raw STM here.
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        actor = _actor_name(cfg, sid, self)
        is_admin = _session_admin(cfg, sid, self)

        def _mine(r) -> bool:
            if not actor:
                return True                    # single-user / admin-neutral: all
            m = r.get("meta") or {}
            return (m.get("actor") or "") in ("", actor)

        stm = [r for r in mind.read_stm(cfg, 60) if _mine(r)][-40:]
        journal = mind.read_journal(cfg, 60) if is_admin else []
        return {"journal": journal, "stm": stm}

    def _mind_think(self):
        return {"thought": mind.Mind(self._mind_cfg()).think()}

    def _mind_dream(self):
        return mind.Mind(self._mind_cfg()).dream()

    def _handle_note(self, payload):
        text = (payload.get("text") or "").strip()
        if not text:
            return self._json({"error": "empty note"})
        from .memory import MemoryStore
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        # A quick /note is treated as household by default (shared_house), but
        # bind the actor so 'private' scope still walls it to this person.
        _bind_actor(cfg, (payload.get("sid") or ""), self)
        note = MemoryStore(cfg).write(text, type="user")
        self._json({"ok": True, "name": note.name})


def persist_selfcheck() -> list:
    """Round-trip-write each state file to heal read-only flags and report
    whether persistence actually works. Returns [(label, ok, detail), ...]."""
    import os
    results = []
    targets = [("persona.json", persona.PERSONA_PATH),
               (".env", ENV_PATH), ("anvil.toml", TOML_PATH)]
    for label, path in targets:
        path = Path(path)
        try:
            if path.exists():
                cfgmod.atomic_write(path, path.read_text("utf-8", "replace"))
                results.append((label, True, "writable"))
            elif cfgmod.dir_writable(path.parent):
                results.append((label, True, "will be created on first save"))
            else:
                results.append((label, False, "folder not writable"))
        except OSError as exc:
            results.append((label, False, f"{type(exc).__name__}: {exc}"))
    return results


class _QuietServer(ThreadingHTTPServer):
    """Don't dump a traceback when a client simply hangs up mid-response — a
    normal event (backgrounded PWA, slow reply the phone abandoned). Real errors
    still surface."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionAbortedError, ConnectionResetError,
                            BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def serve(port: Optional[int] = None, open_browser: bool = True) -> None:
    apply_env_to_process(load_env_file())
    _load_serving_state()    # pending approvals + logins survive a restart
    cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
    if getattr(cfg, "ha_stream", False):
        # Real-time house sensing (kill-switched in anvil.toml). Kept events
        # land in STM through the same evidence-gated write path the
        # snapshot-diff sensing uses.
        try:
            from .ha_stream import HAStream
            from . import mind as _mind

            def _house_evt(evt):
                try:
                    _mind.record(cfg, "house", evt.get("desc", ""))
                except Exception:
                    pass
            HAStream(cfg, on_event=_house_evt).start()
            print("[anvil] HA websocket stream: ON")
        except Exception as exc:
            print(f"[anvil] HA stream failed to start ({exc}) — "
                  "snapshot-diff sensing remains active")
    port = port or getattr(cfg, "server_port", 8765)
    host = getattr(cfg, "bind_host", "127.0.0.1") or "127.0.0.1"
    httpd = _QuietServer((host, port), Handler)
    # If bound to a specific off-box IP (e.g. the Tailscale 100.x address), point
    # the browser/URL at that host; localhost only works when bound to it.
    url_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{url_host}:{port}"
    if host not in ("127.0.0.1", "localhost"):
        print(f"[anvil] bound to {host} — reachable off this machine (e.g. over Tailscale)")
    print("[anvil] persistence check (these must say WRITABLE to retain settings):")
    for label, ok, detail in persist_selfcheck():
        print(f"   {label:<12} {'WRITABLE' if ok else 'NOT WRITABLE'}  ({detail})")
        if not ok:
            print(f"   ^ delete '{label}' in Explorer, then save again; ANVIL will recreate it.")
    print(f"[anvil] web interface at {url}  (Ctrl-C to stop)")
    if getattr(cfg, "auto_pulse", True):
        _start_autopilot(cfg)
    _start_scheduler()
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[anvil] stopped")


def _start_autopilot(cfg) -> None:
    """Run ANVIL's mind (think + dream) in a background daemon thread so it keeps
    thinking and consolidating on its own the whole time the server is up."""
    import threading
    from . import mind as mindmod

    def _run():
        try:
            mindmod.Mind(cfg).autopilot()
        except Exception:
            pass  # never let the mind thread take down the server

    threading.Thread(target=_run, name="anvil-autopilot", daemon=True).start()
    iv = getattr(cfg, "heartbeat_interval_min", 15)
    print(f"[anvil] autopilot on — thinking every {iv} min, dreaming as it goes "
          f"(set auto_pulse=false to disable)")


def _run_scheduled_job(job) -> None:
    """Execute one due cron job through the normal pipeline, leave evidence in
    STM (so the mind can faithfully remember it acted), and notify."""
    cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
    apply_env_to_process(load_env_file())
    from .pipeline import Pipeline
    prompt = job.inputs.get("prompt") or job.name
    # Run the job AS its owner: the operator card and identity-aware gating
    # inside the job's own pipeline run were previously actorless, so a
    # reminder job ('tell the operator ...') couldn't tell who the operator was.
    if getattr(job, "owner", ""):
        cfg._actor = job.owner
    res = Pipeline(cfg).run(prompt, min_rung=job.min_rung)
    try:
        mind.record(cfg, "action",
                    f"did job '{job.name}' -> {(res.answer or '')[:140]}")
    except Exception:
        pass
    if job.notify == "discord":
        try:
            from .comms import notify as discord_notify
            discord_notify(cfg.discord_webhook_url, f"**{job.name}**\n{res.answer}")
        except Exception:
            pass
    try:                       # scheduled results always deliver (user asked
        from . import push     # for this job explicitly — not quiet-gated)
        to = {job.owner} if getattr(job, "owner", "") else None   # its creator's devices
        push.notify(cfg, job.name, _snippet(res.answer or ""),
                    url="/", tag="job", to=to)
    except Exception:
        pass


def _start_scheduler() -> None:
    """Run due cron jobs inside serve-web. The Jobs UI has always PROMISED
    'jobs run automatically when ANVIL is left running' — but the runner only
    existed in the separate `python -m anvil serve` CLI mode, so in the normal
    web setup jobs silently never fired. This closes the gap: the same
    crash-proof run_pending() loop, on a daemon thread."""
    import threading
    import time as _t
    from .scheduler import Scheduler

    def _loop():
        cfg = cfgmod.load(TOML_PATH if TOML_PATH.exists() else None)
        sched = Scheduler(cfg)         # one instance: keeps the ran-this-minute stamps
        while True:
            try:
                sched.run_pending(_run_scheduled_job)
            except Exception:
                pass                   # never let the scheduler take down the server
            _t.sleep(30)

    threading.Thread(target=_loop, name="anvil-scheduler", daemon=True).start()
    print("[anvil] scheduler on — cron jobs run while the web app is up")


# --------------------------------------------------------------------------- #
# The single-page UI (inline; IRC-style chat + first-launch persona wizard).
# --------------------------------------------------------------------------- #
MANIFEST_JSON = json.dumps({
    "name": "ANVIL — Lara",
    "short_name": "Lara",
    "description": "Your household companion.",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "background_color": "#0b0d10",
    "theme_color": "#14171c",
    "orientation": "portrait-primary",
    "icons": [
        {"src": "/icons/flame-192.png", "sizes": "192x192", "type": "image/png",
         "purpose": "any"},
        {"src": "/icons/flame-512.png", "sizes": "512x512", "type": "image/png",
         "purpose": "any"},
        {"src": "/icons/flame-512-maskable.png", "sizes": "512x512",
         "type": "image/png", "purpose": "maskable"},
    ],
})


# Service worker: no offline shell (avoids stale UI during dev); it exists to
# receive Web Push and route notification taps back into the app. iOS requires a
# notification to be shown for every push, so we always display one.
SW_JS = r"""
self.addEventListener('install', function(e){ self.skipWaiting(); });
self.addEventListener('activate', function(e){ e.waitUntil(self.clients.claim()); });
// Offline shell: NETWORK-FIRST for the app itself, so a live server always
// serves the freshest build (the checkBuild self-update stays in charge) and
// only a dead network falls back to the cached copy. API calls are never
// cached — stale answers are worse than no answers.
var SHELL_CACHE = 'hearth-shell-v1';
self.addEventListener('fetch', function(e){
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.indexOf('/api/') === 0) return;      // live data only
  if (req.mode === 'navigate' || url.pathname === '/') {
    e.respondWith(fetch(req).then(function(r){
      if (r && r.ok) {
        var copy = r.clone();
        caches.open(SHELL_CACHE).then(function(c){ c.put('/', copy); });
      }
      return r;
    }).catch(function(){
      return caches.match('/').then(function(hit){
        return hit || new Response('offline', {status: 503});
      });
    }));
  }
});
self.addEventListener('push', function(event){
  var d = {};
  try { d = event.data ? event.data.json() : {}; } catch (e) { d = {}; }
  var title = d.title || 'Lara';
  var opts = {
    body: d.body || '',
    icon: '/icons/flame-192.png',
    badge: '/icons/flame-192.png',
    tag: d.tag || 'anvil',
    renotify: true,
    data: { url: d.url || '/' }
  };
  event.waitUntil((async function(){
    // Suppress the banner when the app is open and visible on screen — the
    // user is already looking at the answer. Only notify when every window is
    // hidden/backgrounded (or none is open).
    var wins = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (var i = 0; i < wins.length; i++) {
      if (wins[i].visibilityState === 'visible') return;
    }
    return self.registration.showNotification(title, opts);
  })());
});
self.addEventListener('notificationclick', function(event){
  event.notification.close();
  var url = (event.notification.data && event.notification.data.url) || '/';
  event.waitUntil((async function(){
    var all = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (var i = 0; i < all.length; i++) {
      var c = all[i];
      if ('focus' in c) { try { await c.navigate(url); } catch (e) {} return c.focus(); }
    }
    if (self.clients.openWindow) return self.clients.openWindow(url);
  })());
});
"""


# Bump on every shipped UI change. Served to the client, which polls
# /api/version and RELOADS itself when the number changes — the fix for an iOS
# standalone PWA holding stale JS (it doesn't re-fetch the shell on resume, so
# new features like the live thinking trace never appeared there). Keep the
# "ANVIL UI build N" marker below in sync (the doctor asserts it exists).
UI_BUILD = 87


def _load_hearth() -> str:
    """The Hearthlight UI ships as anvil/hearth.html (source-controlled).
    Missing/unreadable -> empty string and / serves the classic console."""
    try:
        p = Path(__file__).resolve().parent / "hearth.html"
        return p.read_text("utf-8")
    except Exception:
        return ""


HEARTH_HTML = _load_hearth()

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta http-equiv="Cache-Control" content="no-store, max-age=0">
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#0f141b" media="(prefers-color-scheme: dark)">
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Lara">
<link rel="apple-touch-icon" href="/icons/flame-180.png">
<link rel="icon" type="image/png" href="/icons/flame-192.png">
<title>ANVIL</title>
<style>
  /* ---- color + elevation tokens ----
     Dark-first, tonal-elevation system (2025 dashboard practice): depth comes
     from stepping surface LIGHTNESS across nested layers, not shadow soup.
     Ladder: bg (page) -> panel (card) -> elev (inset/nested) -> elev2 (raised).
     Neutrals carry a faint warm cast so they harmonize with the ember accent
     instead of reading cold-slate. Text lands ~90% white on a soft base, never
     pure #fff on pure #000. --acc-soft is a low-alpha ember wash for hovers,
     focus rings and active glows; --sh* are reserved for genuinely floating
     surfaces (drawer, overlay, composer) where tonal steps aren't enough. */
  :root{--bg:#0b0e12;--panel:#141922;--elev:#1c222c;--elev2:#252d39;--ircbg:#080b0f;
        --line:#242b35;--line2:#38424f;--ink:#e9eef4;--mut:#9aa4b1;--dim:#6f7a87;--acc:#f0883e;
        --acc-ink:#ffb877;--acc-soft:rgba(240,136,62,.13);--acc-line:rgba(240,136,62,.36);
        --acc-hi:#ffa057;--acc-ink-on:#1a1205;--acc-lineb:var(--acc-line);
        --blue:#5aa9ff;--ok:#3fb950;--bad:#f85149;--pur:#a371f7;--teal:#39c5cf;
        --ok-soft:rgba(63,185,80,.14);--bad-soft:rgba(248,81,73,.14);--blue-soft:rgba(90,169,255,.14);--pur-soft:rgba(163,113,247,.14);
        --ok-line:rgba(63,185,80,.4);--blue-line:rgba(90,169,255,.4);--bad-line:rgba(248,81,73,.4);--pur-line:rgba(163,113,247,.4);
        --codebg:#0c1116;--userbubble:#212936;--hdr:#0f141b;--sb:#38424f;--sbh:#586475;
        --sh1:0 1px 2px rgba(0,0,0,.45);--sh2:0 10px 28px -10px rgba(0,0,0,.6);
        --sh3:0 24px 60px -16px rgba(0,0,0,.66);--focus:0 0 0 3px var(--acc-soft);--focus-edge:0 0 0 1px var(--acc);
        /* Radius, spacing & motion scales — theme-agnostic, defined once here. */
        --r-xs:5px;--r-sm:7px;--r-md:10px;--r-lg:14px;--r-xl:18px;--r-pill:999px;
        --sp-1:4px;--sp-2:8px;--sp-3:12px;--sp-4:16px;--sp-5:20px;--sp-6:28px;
        --dur-1:.12s;--dur-2:.2s;--dur-3:.32s;--ease:cubic-bezier(.2,.6,.2,1);
        /* Type system — a ~1.2 (minor-third) modular scale, capped at a small set
           of named steps so density stays rhythmic instead of ad-hoc. Body sits at
           --fs-md; dense rows/labels drop to --fs-sm/--fs-xs; headings climb. */
        --fs-xs:11.5px;--fs-sm:12.5px;--fs-base:13.5px;--fs-md:15px;--fs-lg:17px;
        --fs-xl:20px;--fs-2xl:24px;
        /* System UI stack + a monospace stack, defined once. -apple-system/Blink
           first so iOS/macOS (the family's phones) render San Francisco crisply. */
        --font-ui:-apple-system,BlinkMacSystemFont,system-ui,"Segoe UI",Roboto,sans-serif;
        --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
  [data-theme="light"]{--bg:#f5f7fa;--panel:#ffffff;--elev:#eef1f6;--elev2:#e4e9f0;--ircbg:#f8fafc;
        --line:#e3e7ee;--line2:#ccd3dc;--ink:#1c2126;--mut:#565f6b;--dim:#7c8591;--acc:#c96209;
        --acc-ink:#a85400;--acc-soft:rgba(201,98,9,.11);--acc-line:rgba(201,98,9,.34);
        --acc-hi:#e0740c;--acc-ink-on:#241300;--acc-lineb:var(--acc-line);
        --blue:#0969da;--ok:#1a7f37;--bad:#cf222e;--pur:#8250df;--teal:#0e7490;
        --ok-soft:rgba(26,127,55,.11);--bad-soft:rgba(207,34,46,.11);--blue-soft:rgba(9,105,218,.11);--pur-soft:rgba(130,80,223,.11);
        --ok-line:rgba(26,127,55,.36);--blue-line:rgba(9,105,218,.36);--bad-line:rgba(207,34,46,.36);--pur-line:rgba(130,80,223,.36);
        --codebg:#eef1f5;--userbubble:#e6effb;--hdr:#ffffff;--sb:#c2cbd4;--sbh:#9aa6b2;
        --sh1:0 1px 2px rgba(16,24,40,.06);--sh2:0 10px 26px -10px rgba(16,24,40,.13);
        --sh3:0 24px 56px -16px rgba(16,24,40,.2);--focus:0 0 0 3px var(--acc-soft);--focus-edge:0 0 0 1px var(--acc)}
  *{box-sizing:border-box} html{height:100%}
  /* Stylized scrollbars: a slim rounded pill that floats with margin (the
     transparent border + padding-box clip), fading to warm accent on hover. */
  *{scrollbar-width:thin;scrollbar-color:var(--sb) transparent}
  ::-webkit-scrollbar{width:12px;height:12px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:var(--sb);border-radius:10px;
    border:3px solid transparent;background-clip:padding-box;transition:background var(--dur-2)}
  ::-webkit-scrollbar-thumb:hover{background:var(--sbh);background-clip:padding-box}
  ::-webkit-scrollbar-thumb:active{background:var(--acc);background-clip:padding-box}
  ::-webkit-scrollbar-corner{background:transparent}
  #msgs::-webkit-scrollbar{width:10px}
  /* Full-height app shell: header + nav are fixed, everything else flexes to
     fill the viewport so the message list is the ONLY thing that scrolls — no
     page-level scrollbar. 100dvh tracks the dynamic viewport on iOS/PWA. */
  body{margin:0;background:var(--bg);color:var(--ink);
    font:var(--fs-md)/1.55 var(--font-ui);
    -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
    text-rendering:optimizeLegibility;
    height:100vh;height:100dvh;display:flex;flex-direction:column;overflow:hidden}
  /* Tabular figures: same-width digits so numeric columns/metrics/timestamps
     stop shifting as values change. Opt-in via .tnum or the surfaces below. */
  .tnum{font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
  /* Pad the header for the iOS status bar / notch (safe-area). The app runs
     full-bleed under the status bar in standalone PWA mode, so without this the
     title + buttons collide with the clock and battery. The dark header bg still
     extends edge-to-edge; only the content sits below the inset. */
  header{flex:none;display:flex;align-items:center;gap:12px;
    padding:calc(12px + env(safe-area-inset-top,0px)) calc(22px + env(safe-area-inset-right,0px)) 12px calc(22px + env(safe-area-inset-left,0px));
    border-bottom:1px solid var(--line);background:var(--hdr)}
  header h1{font-size:var(--fs-lg);margin:0;letter-spacing:4px;font-weight:700}
  header .dot{width:9px;height:9px;border-radius:var(--r-pill);background:var(--dim)}
  header .dot.live{background:var(--ok);box-shadow:0 0 0 0 var(--ok-soft);animation:pulse 2.4s infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--ok) 45%,transparent)}70%{box-shadow:0 0 0 7px transparent}100%{box-shadow:0 0 0 0 transparent}}
  header .sub{color:var(--mut);font-size:var(--fs-base)}
  .themebtn{margin-left:auto;display:flex;align-items:center;gap:6px;background:var(--elev);border:1px solid var(--line);
    color:var(--mut);border-radius:var(--r-sm);padding:var(--sp-1) var(--sp-3);cursor:pointer;font-size:var(--fs-sm);
    transition:border-color var(--dur-1),color var(--dur-1),background var(--dur-1),transform var(--dur-1)}
  .themebtn:hover{border-color:var(--acc-line);color:var(--ink);background:var(--elev2)}
  .themebtn:focus-visible{outline:none;box-shadow:var(--focus-edge),var(--focus)} .themebtn svg{width:14px;height:14px;display:block}
  /* Chats sidebar. ONE control: the hamburger is FIXED at the top-left and
     never moves — the sidebar forms around it when open (the glyph morphs to
     an X), and the same button dismisses it. Phone: overlay + backdrop.
     Desktop: docked, content shifts right; open by default, collapsible. */
  .menubtn{position:fixed;top:calc(9px + env(safe-area-inset-top,0px));left:12px;
    z-index:42;margin:0;font-size:15px;padding:5px 10px;background:var(--hdr)}
  header{padding-left:calc(58px + env(safe-area-inset-left,0px))}
  .drawer-top{margin-left:46px}
  .drawer-bg{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:40}
  .drawer{position:fixed;top:0;left:0;bottom:0;width:min(320px,88vw);z-index:41;
    background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;
    padding:calc(10px + env(safe-area-inset-top,0px)) 10px calc(10px + env(safe-area-inset-bottom,0px));
    box-shadow:var(--sh3)}
  /* display:flex above beats the UA's [hidden]{display:none} — without this
     the 'hidden' drawer still rendered invisibly over the header and ATE the
     hamburger's clicks (same bug class as the wizard overlay). */
  .drawer[hidden]{display:none}
  @media (min-width: 900px){
    body.dopen header, body.dopen nav, body.dopen main{margin-left:320px}
    /* main declares width:100%, so the docked margin must come OUT of the
       width or the content overflows the viewport by the sidebar's width. */
    body.dopen main{width:calc(100% - 320px)}
    body.dopen header{padding-left:calc(22px + env(safe-area-inset-left,0px))}
    .drawer{box-shadow:none;z-index:30}
    .drawer-bg{display:none}
    header, nav, main{transition:margin-left var(--dur-3) var(--ease)}
  }
  .drawer-top{display:flex;gap:8px;margin-bottom:10px;align-items:center}
  .drawer-top input{flex:1;min-width:0}
  .chatlist{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:2px}
  .chatrow{display:flex;align-items:center;gap:8px;padding:9px 10px;border-radius:var(--r-sm);cursor:pointer;color:var(--ink)}
  .chatrow{transition:background var(--dur-1)}
  .chatrow:hover{background:var(--elev)} .chatrow.on{background:var(--acc-soft);outline:1px solid var(--acc-line)}
  .chatrow .t{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:var(--fs-base)}
  .chatrow .snip{display:block;color:var(--mut);font-size:var(--fs-xs);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .chatrow .when{flex:none;color:var(--dim);font-size:var(--fs-xs)}
  .chatrow .act{flex:none;display:none;gap:2px}
  .chatrow:hover .act{display:inline-flex}
  .chatrow .act button{background:none;border:none;color:var(--mut);cursor:pointer;padding:2px 4px;font-size:var(--fs-sm);border-radius:var(--r-xs)}
  .chatrow .act button:hover{color:var(--ink);background:var(--panel)}
  .chatrow .act button:focus-visible{outline:none;box-shadow:var(--focus)}
  .chatrow .pinmark{flex:none;color:var(--acc);font-size:var(--fs-xs)}
  .chatsec{color:var(--dim);font-size:var(--fs-xs);letter-spacing:.8px;font-weight:600;text-transform:uppercase;padding:8px 10px 3px}
  nav{flex:none;display:flex;gap:2px;border-bottom:1px solid var(--line);background:var(--panel);
    overflow-x:auto;padding:0 calc(16px + env(safe-area-inset-right,0px)) 0 calc(16px + env(safe-area-inset-left,0px))}
  nav button{background:none;border:none;color:var(--mut);padding:var(--sp-3) var(--sp-4);cursor:pointer;
    font-size:var(--fs-base);border-bottom:2px solid transparent;white-space:nowrap;display:flex;align-items:center;gap:7px;
    border-radius:var(--r-xs);transition:color var(--dur-1),border-color var(--dur-1)}
  nav button:hover{color:var(--ink)}
  /* Inset ring (nav clips overflow-x, so an outset ring would be cut off) — echoes
     the active-tab accent underline for keyboard users without spilling outside. */
  nav button:focus-visible{outline:none;color:var(--ink);box-shadow:inset 0 0 0 2px var(--acc-soft),inset 0 -2px 0 0 var(--acc)}
  /* Active tab: warm underline plus a faint, tightly-contained ember glow, so the
     current view reads intentionally at a glance without a heavy fill or spill. */
  nav button.on{color:var(--ink);border-bottom-color:var(--acc);
    box-shadow:0 4px 12px -8px var(--acc)}
  /* main fills the remaining height AND width — the app scales to the window
     (no fixed column cap). Chat fits exactly (no main scroll); long tabs like
     Setup overflow and scroll within main, not the page. */
  main{width:100%;margin:0 auto;flex:1 1 auto;min-height:0;
    display:flex;flex-direction:column;overflow-y:auto;
    padding:16px calc(22px + env(safe-area-inset-right,0px)) calc(18px + env(safe-area-inset-bottom,0px)) calc(22px + env(safe-area-inset-left,0px))}
  /* Card: top of the tonal ladder over the page bg, held by a hairline plus a
     whisper of shadow for lift (never a heavy drop shadow on the dark base). */
  .card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-lg);padding:var(--sp-4);margin-bottom:var(--sp-4);
    box-shadow:var(--sh1)}
  .card h3 .mut{font-weight:400}
  label{display:block;color:var(--mut);font-size:var(--fs-base);margin:12px 0 4px}
  input,select,textarea{width:100%;background:var(--bg);color:var(--ink);caret-color:var(--acc);border:1px solid var(--line2);
    border-radius:var(--r-sm);padding:9px 11px;font:inherit;outline:none;transition:border-color var(--dur-1),box-shadow var(--dur-1)}
  input:focus,select:focus,textarea:focus{border-color:var(--acc);box-shadow:var(--focus-edge),var(--focus)}
  input::placeholder,textarea::placeholder{color:var(--dim)}
  /* Shared disabled affordance: dim + not-allowed for every control locked by JS
     (forging button, locked profile fields, remind/share toggles). Opacity-based
     so both themes inherit it without hardcoded colors. */
  input:disabled,select:disabled,textarea:disabled,button:disabled{opacity:.5;cursor:not-allowed;box-shadow:none}
  button:disabled:hover{background:var(--acc);box-shadow:0 2px 10px -3px var(--acc)}
  button.ghost:disabled:hover,.themebtn:disabled:hover{background:var(--elev);border-color:var(--line)}
  textarea{min-height:96px;resize:vertical;line-height:1.5}
  /* Primary action: ember fill on near-black ink; a soft ember-tinted shadow
     grounds it and warms on hover. */
  button.go{background:var(--acc);color:var(--acc-ink-on);border:none;border-radius:var(--r-sm);padding:10px 16px;
    font-weight:600;cursor:pointer;margin-top:14px;box-shadow:0 2px 10px -3px var(--acc);
    transition:background var(--dur-1),box-shadow var(--dur-1),transform var(--dur-2) var(--ease)}
  button.go:hover{background:var(--acc-hi);box-shadow:0 4px 14px -3px var(--acc)}
  button.go:active{transform:translateY(1px)}
  button.go:focus-visible{outline:none;box-shadow:var(--focus-edge),var(--focus)}
  button.ghost{background:var(--elev);border:1px solid var(--line);color:var(--ink);border-radius:var(--r-sm);
    padding:7px 12px;cursor:pointer;font-size:var(--fs-base);transition:background var(--dur-1),border-color var(--dur-1),transform var(--dur-1)}
  button.ghost:hover{background:var(--elev2);border-color:var(--acc-line)}
  button.ghost:focus-visible{outline:none;box-shadow:var(--focus-edge),var(--focus)}
  button.ghost:active,.themebtn:active,.sendbtn:active,.attachbtn:active,.prompts button:active{transform:translateY(1px)}
  .pill{display:inline-block;padding:2px 9px;border-radius:var(--r-pill);font-size:12px;border:1px solid var(--line2);background:var(--elev);color:var(--mut)}
  .ok{color:var(--ok);border-color:var(--ok)} .bad{color:var(--bad);border-color:var(--bad)}
  .pill.ok{background:var(--ok-soft);border-color:var(--ok-line)} .pill.bad{background:var(--bad-soft);border-color:var(--bad-line)}
  .mut{color:var(--mut)} .dim{color:var(--dim)} .hint{color:var(--mut);font-size:var(--fs-sm);margin-top:4px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:var(--sp-3)}
  h3{margin:2px 0 8px;font-size:var(--fs-md);font-weight:600;letter-spacing:-.1px}
  /* ---- command-deck dashboard ---- */
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:var(--sp-3);margin:2px 0 4px}
  .tile{background:var(--elev);border:1px solid var(--line);border-radius:var(--r-md);padding:var(--sp-3);
    transition:border-color var(--dur-1),background var(--dur-1),transform var(--dur-2) var(--ease)}
  .tile:hover{border-color:var(--acc-line);background:var(--elev2);transform:translateY(-1px)}
  .tile .k{color:var(--mut);font-size:var(--fs-sm);letter-spacing:.4px;
    text-transform:uppercase;font-weight:500}
  .tile .v{font-size:var(--fs-2xl);font-weight:600;margin-top:4px;line-height:1.15;
    font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
  .tile .v small{font-size:var(--fs-base);color:var(--mut);font-weight:400}
  .statline{display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:var(--mut);font-size:var(--fs-base)}
  .sdot{width:8px;height:8px;border-radius:var(--r-pill);background:var(--dim);flex:none}
  .sdot.on{background:var(--ok)} .sdot.off{background:var(--dim)}
  .feed{font-family:var(--font-mono);font-size:var(--fs-sm);line-height:1.55;
    font-variant-numeric:tabular-nums;
    max-height:40vh;overflow-y:auto;background:var(--ircbg);border:1px solid var(--line);border-radius:var(--r-md);padding:11px 13px}
  .feed .row{padding:2px 0;white-space:pre-wrap;word-break:break-word;border-bottom:1px solid var(--line)}
  .feed .row:last-child{border-bottom:none}
  .feed .t{color:var(--dim);margin-right:8px}
  .tag{display:inline-block;min-width:64px;color:var(--teal)}
  .tag.think{color:var(--blue)} .tag.dream{color:var(--pur)} .tag.selfdev{color:var(--acc)}
  .tag.question{color:var(--teal)} .tag.event,.tag.chat,.tag.observation{color:var(--dim)}
  .qrow{display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--line);font-size:var(--fs-base);font-variant-numeric:tabular-nums}
  .qrow:last-child{border-bottom:none}
  .qstat{font-size:var(--fs-xs);padding:1px 8px;border-radius:var(--r-pill);border:1px solid var(--line2);color:var(--mut);flex:none}
  .qstat.done{color:var(--ok);border-color:var(--ok-line);background:var(--ok-soft)} .qstat.queued{color:var(--blue);border-color:var(--blue-line);background:var(--blue-soft)}
  .qstat.rework{color:var(--acc);border-color:var(--acc-line);background:var(--acc-soft)} .qstat.rejected{color:var(--bad);border-color:var(--bad-line);background:var(--bad-soft)}
  .qstat.building-internal{color:var(--pur);border-color:var(--pur-line);background:var(--pur-soft)}
  .bar{height:5px;background:var(--line);border-radius:var(--r-xs);overflow:hidden;flex:1;min-width:40px}
  .bar>i{display:block;height:100%;background:var(--acc)}
  .note{background:var(--elev);border:1px solid var(--line);border-radius:var(--r-md);padding:11px 13px;margin-bottom:9px}
  .note .h{display:flex;align-items:center;gap:8px;font-size:var(--fs-base);margin-bottom:5px}
  .note .b{color:var(--mut);font-size:var(--fs-base);white-space:pre-wrap;word-break:break-word}
  .entity{display:flex;align-items:center;gap:9px;padding:6px 0;border-bottom:1px solid var(--line);font-size:var(--fs-base)}
  .entity:last-child{border-bottom:none}
  .entity .st{margin-left:auto;font-weight:600;font-size:12px;color:var(--ink)}
  .entity .st.on{color:var(--ok)} .entity .st.off{color:var(--dim)}
  /* ---- chat (modern messages) ---- */
  #chat:not([hidden]){display:flex;flex-direction:column;flex:1 1 auto;min-height:0}
  #msgs{display:flex;flex-direction:column;gap:20px;flex:1;min-height:0;overflow-y:auto;padding:18px 6px 10px}
  .msg{display:flex;gap:11px;max-width:100%}
  .av{flex:none;width:30px;height:30px;border-radius:var(--r-pill);margin-top:1px;background:var(--acc-soft);
    border:1px solid var(--acc-line);display:flex;align-items:center;justify-content:center;color:var(--acc)}
  .av svg{width:17px;height:17px}
  .msg.ember .content{min-width:0;flex:1}
  .who{font-size:var(--fs-sm);color:var(--mut);margin-bottom:4px;font-weight:600;letter-spacing:.2px}
  .msg.user{justify-content:flex-end}
  .msg.user .bubble{background:var(--userbubble);border:1px solid var(--line2);border-radius:var(--r-lg) var(--r-lg) var(--r-xs) var(--r-lg);
    padding:10px 14px;max-width:80%;white-space:pre-wrap;word-break:break-word;font-size:var(--fs-md);line-height:1.5}
  .meta{color:var(--dim);font-size:var(--fs-xs);margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;
    font-variant-numeric:tabular-nums}
  .meta .d{width:3px;height:3px;border-radius:var(--r-pill);background:var(--dim)}
  .steps{display:flex;flex-direction:column;gap:5px;margin:0 0 9px}
  .step{display:flex;align-items:center;gap:8px;font-size:var(--fs-sm);color:var(--mut)}
  .step .sd{width:6px;height:6px;border-radius:var(--r-pill);background:var(--teal);flex:none}
  .step.denied .sd{background:var(--bad)}
  .step code{background:var(--codebg);border:1px solid var(--line);border-radius:var(--r-xs);padding:0 5px;font-size:12px}
  .sysmsg{align-self:center;color:var(--dim);font-size:12px;font-style:italic}
  .dots{display:inline-flex;gap:4px;align-items:center;height:20px}
  .dots i{width:6px;height:6px;border-radius:var(--r-pill);background:var(--mut);animation:blink 1.4s infinite}
  .dots i:nth-child(2){animation-delay:.2s} .dots i:nth-child(3){animation-delay:.4s}
  .wline{display:flex;align-items:center;gap:10px;color:var(--mut);font-size:13px}
  .wctx{color:var(--dim);font-size:11px}
  /* Live deliberation trace: dim, italic, shows the tail of what Lara is
     mulling over — proof of motion during the otherwise-silent thinking phase */
  .think{color:var(--dim);font-size:12px;font-style:italic;white-space:pre-wrap;
    max-height:110px;overflow-y:auto;border-left:2px solid var(--line2);
    padding:4px 10px;margin:8px 0;opacity:.85}
  /* "How I got here": the live subtext compacted into a clickable archive —
     phases, her thoughts per step, and each tool's log, all folded away */
  .trace{margin:2px 0 8px;font-size:12px}
  .trace>summary{cursor:pointer;color:var(--dim);user-select:none}
  .trace>summary:hover{color:var(--mut)}
  .trace[open]>summary{margin-bottom:4px}
  .trow{color:var(--mut);padding:2px 0 2px 16px}
  .tthink,.tstep{margin:2px 0 2px 16px}
  .tthink>summary,.tstep>summary{cursor:pointer;color:var(--dim);font-size:11.5px;user-select:none}
  .tthink .tfull{max-height:260px}
  .tstep summary .step{display:inline-flex}
  .tobs{color:var(--dim);font-size:var(--fs-xs);white-space:pre-wrap;font-family:var(--font-mono);
    max-height:220px;overflow-y:auto;border-left:2px solid var(--line2);
    padding:4px 10px;margin:4px 0 4px 14px}
  @keyframes blink{0%,65%,100%{opacity:.25;transform:translateY(0)}32%{opacity:1;transform:translateY(-2px)}}
  .approve{background:var(--elev);border:1px solid var(--line2);border-radius:var(--r-lg);padding:12px 14px;font-size:var(--fs-base);margin-top:4px;transition:opacity .2s,border-color .2s}
  .approve .r2{display:flex;gap:8px;margin-top:11px;align-items:center}
  .approve.resolved{opacity:.62;border-color:var(--line)}
  .approve.resolved .r2{margin-top:8px}
  .approve .verdict{font-weight:600;display:inline-flex;align-items:center;gap:5px}
  .approve .verdict.ok{color:var(--ok)} .approve .verdict.no{color:var(--bad)}
  .famreqs{margin:6px 0 10px}
  .famreq{background:var(--elev);border:1px solid var(--acc);border-radius:var(--r-lg);padding:11px 13px;font-size:13px;margin-bottom:8px}
  .famreq .who{font-weight:600;color:var(--acc)}
  .famreq .why{color:var(--dim);margin:4px 0 2px;font-style:italic}
  .famreq .r2{display:flex;gap:8px;margin-top:10px}
  .hello{margin:auto;text-align:center;color:var(--mut);padding:26px 10px}
  .hello .bigav{width:58px;height:58px;border-radius:var(--r-pill);margin:0 auto 15px;background:var(--acc-soft);
    border:1px solid var(--acc-line);display:flex;align-items:center;justify-content:center;color:var(--acc);
    box-shadow:0 8px 28px -10px var(--acc)}
  .hello .bigav svg{width:32px;height:32px}
  .hello h2{color:var(--ink);font-weight:600;margin:0 0 6px;font-size:var(--fs-xl);letter-spacing:0}
  .prompts{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:18px}
  .prompts button{background:var(--panel);border:1px solid var(--line2);color:var(--ink);border-radius:var(--r-pill);
    padding:8px 15px;font-size:13px;cursor:pointer;transition:border-color var(--dur-1),background var(--dur-1),color var(--dur-1),transform var(--dur-1)}
  .prompts button:hover{border-color:var(--acc-line);background:var(--acc-soft);color:var(--acc-ink)}
  .prompts button:focus-visible{outline:none;box-shadow:var(--focus-edge),var(--focus)}
  /* A two-tone I-beam (black core + white outline) as a custom cursor, so the
     text cursor stays visible over the message box in BOTH themes — a white
     Windows pointer scheme otherwise vanishes against the light-mode white
     composer. Falls back to the native text I-beam if the data URI is unused. */
  .composer,#task{cursor:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='16'%20height='26'%3E%3Cpath%20d='M4%203h8v2h-3v16h3v2h-8v-2h3v-16h-3z'%20fill='%23000'%20stroke='%23fff'%20stroke-width='1.3'%20stroke-linejoin='round'/%3E%3C/svg%3E") 8 13,text}
  .composer{display:flex;align-items:flex-end;gap:9px;background:var(--panel);border:1px solid var(--line2);
    border-radius:var(--r-lg);padding:8px 8px 8px 16px;box-shadow:var(--sh2);transition:border-color var(--dur-1),box-shadow var(--dur-1)}
  .composer:focus-within{border-color:var(--acc);box-shadow:var(--sh2),var(--focus-edge),var(--focus)}
  #task{border:none;background:none;outline:none;flex:1;color:var(--ink);caret-color:var(--acc);resize:none;overflow-y:auto;
    font:inherit;font-size:var(--fs-md);line-height:1.5;padding:8px 0;margin:0;min-height:26px;max-height:40vh;display:block}
  .sendbtn{flex:none;width:38px;height:38px;border-radius:var(--r-pill);border:none;cursor:pointer;background:var(--acc);
    color:var(--acc-ink-on);display:flex;align-items:center;justify-content:center;box-shadow:0 2px 10px -3px var(--acc);
    transition:background var(--dur-1),box-shadow var(--dur-1),transform var(--dur-1)}
  .sendbtn:hover{background:var(--acc-hi);box-shadow:0 4px 14px -3px var(--acc)}
  .sendbtn:focus-visible{outline:none;box-shadow:var(--focus-edge),var(--focus)}
  .sendbtn.stopping{background:var(--bad)}   /* Stop mode: red square, click or Esc to abort */
  .attachbtn{flex:none;width:34px;height:34px;border-radius:var(--r-pill);border:1px solid var(--line2);cursor:pointer;
    background:none;color:var(--mut);font-size:20px;line-height:1;display:flex;align-items:center;justify-content:center;margin-bottom:2px;
    transition:border-color var(--dur-1),color var(--dur-1),transform var(--dur-1)}
  .attachbtn:hover{border-color:var(--acc-line);color:var(--ink)}
  .attachrow{display:flex;gap:8px;flex-wrap:wrap;padding:8px 4px 6px}
  .attachrow .chip{position:relative;width:64px;height:64px;border-radius:var(--r-md);overflow:hidden;border:1px solid var(--line2)}
  .attachrow .chip img{width:100%;height:100%;object-fit:cover;display:block}
  .attachrow .chip .n{position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.55);color:#fff;font-size:9px;padding:1px 3px;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .attachrow .chip button{position:absolute;top:2px;right:2px;width:16px;height:16px;border-radius:50%;border:none;cursor:pointer;
    background:rgba(0,0,0,.65);color:#fff;font-size:10px;line-height:1;display:flex;align-items:center;justify-content:center}
  .msg .imgs{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
  .msg .imgs img{max-width:140px;max-height:140px;border-radius:var(--r-sm);border:1px solid var(--line2)}
  .cbar{display:flex;align-items:center;gap:16px;margin:6px 0 9px;padding:0 4px;color:var(--mut);font-size:12px}
  .cbar label{display:inline-flex;align-items:center;gap:5px;margin:0} .cbar input{width:auto}
  /* First-visit notification ask (replaces the old bell button) */
  .notifask{flex:none;display:flex;align-items:center;justify-content:space-between;gap:12px;
    background:var(--elev);border:1px solid var(--line2);border-radius:var(--r-lg);padding:11px 14px;margin:0 0 12px}
  .notifask .na-t{color:var(--ink);font-size:var(--fs-base);line-height:1.4}
  .notifask .na-b{display:flex;gap:8px;flex:none}
  .notifask .na-b button{padding:6px 14px;font-size:var(--fs-base);border-radius:var(--r-sm);margin:0}
  code{background:var(--codebg);border:1px solid var(--line);border-radius:var(--r-xs);padding:1px 5px}
  /* ---- rendered markdown (agent answers) ---- */
  .md{white-space:normal;font-family:var(--font-ui);font-size:var(--fs-base);line-height:1.62;margin-top:2px}
  .md.streaming{white-space:pre-wrap}
  .md>:first-child{margin-top:0} .md>:last-child{margin-bottom:0}
  .md h1,.md h2,.md h3,.md h4,.md h5,.md h6{margin:12px 0 5px;line-height:1.3;font-weight:600}
  .md h1{font-size:var(--fs-lg)} .md h2{font-size:var(--fs-lg)} .md h3{font-size:var(--fs-md)} .md h4,.md h5,.md h6{font-size:var(--fs-base)}
  .md p{margin:7px 0} .md ul,.md ol{margin:6px 0;padding-left:22px} .md li{margin:3px 0}
  .md code{background:var(--codebg);border:1px solid var(--line2);border-radius:var(--r-xs);padding:1px 5px;
    font-family:var(--font-mono);font-size:var(--fs-sm)}
  .md pre{background:var(--codebg);border:1px solid var(--line);border-radius:var(--r-sm);padding:11px 13px;overflow-x:auto;margin:9px 0}
  .md pre code{background:none;border:none;padding:0;font-size:var(--fs-sm);color:var(--ink)}
  .md blockquote{border-left:3px solid var(--acc);margin:9px 0;padding:2px 0 2px 13px;color:var(--mut)}
  .md a{color:var(--blue);text-decoration:none;text-underline-offset:2px;transition:color var(--dur-1)} .md a:hover{color:var(--acc-ink);text-decoration:underline} .md strong{font-weight:600} .md em{font-style:italic}
  .md hr{border:none;border-top:1px solid var(--line);margin:11px 0}
  .md table{border-collapse:collapse;margin:8px 0;font-size:var(--fs-base);font-variant-numeric:tabular-nums}
  .md th,.md td{border:1px solid var(--line2);padding:5px 9px;text-align:left}
  /* ---- wizard overlay ---- */
  .overlay{position:fixed;inset:0;background:rgba(4,7,11,.7);backdrop-filter:blur(4px);
    -webkit-backdrop-filter:blur(4px);display:flex;align-items:center;
    justify-content:center;z-index:50;padding:16px}
  .overlay[hidden]{display:none}
  .wiz{width:min(580px,94vw);background:var(--panel);border:1px solid var(--line);border-radius:var(--r-xl);padding:24px;
    box-shadow:var(--sh3)}
  .wiz h2{margin:0 0 4px;letter-spacing:2px} .wiz p{color:var(--mut);margin:0 0 6px;font-size:14px}
  .traitlist{margin:6px 0 0;padding:0;list-style:none}
  .traitlist li{padding:6px 10px;background:var(--bg);border:1px solid var(--line);border-radius:var(--r-sm);
    margin-bottom:6px;font-size:13px}
  /* Respect an OS request for reduced motion: neutralize the pulse/blink/dots
     keyframes and every transition without removing any element or handler. */
  @media (prefers-reduced-motion: reduce){
    *,*::before,*::after{animation-duration:.001ms !important;animation-iteration-count:1 !important;
      transition-duration:.001ms !important;scroll-behavior:auto !important}
  }
</style></head>
<body>
<button id="menubtn" class="themebtn menubtn" onclick="toggleChats()" title="Chats" aria-label="Toggle chat list">&#9776;</button>
<header>
  <h1>ANVIL</h1><span class="dot" id="live" title="autopilot"></span>
  <button id="whoBtn" class="themebtn" style="display:none;margin-left:auto" onclick="openProfiles()" title="Who's using Lara">👤 <span id="whoName">—</span></button>
  <button id="themebtn" class="themebtn" onclick="cycleTheme()" title="Theme (Auto / Light / Dark)"></button></header>
<div id="drawerBg" class="drawer-bg" hidden onclick="toggleChats()"></div>
<aside id="drawer" class="drawer" hidden>
  <div class="drawer-top">
    <input id="chatSearch" placeholder="Search all chats…" oninput="chatsRefresh()">
    <button class="go" style="margin:0;white-space:nowrap" onclick="drawerNewChat()">New chat</button>
  </div>
  <div id="chatList" class="chatlist"></div>
</aside>
<nav>
  <button data-t="chat" class="on">Chat</button>
  <button data-t="pulse">Pulse</button>
  <button data-t="memory">Memory</button>
  <button data-t="shared" id="sharedTab" hidden>Shared</button>
  <button data-t="jobs">Jobs</button>
  <button data-t="profile" id="profileTab">Profile</button>
  <button data-t="setup" id="serverTab">Server</button>
</nav>
<main>
  <section id="chat">
    <div id="famReqs" class="famreqs" hidden></div>
    <div id="msgs"></div>
    <div id="attachRow" class="attachrow" hidden></div>
    <div class="composer">
      <button class="attachbtn" onclick="$('fileIn').click()" title="Attach photos or videos" aria-label="Attach">+</button>
      <input id="fileIn" type="file" accept="image/*,video/*" multiple hidden onchange="filesPicked(this.files); this.value='';">
      <textarea id="task" rows="1" autocomplete="off" placeholder="Message Lara…"></textarea>
      <button id="send" class="sendbtn" onclick="onSend()" aria-label="Send" title="Send (Enter)">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M3.4 20.4 21 12 3.4 3.6 3.4 10l12 2-12 2z"/></svg>
      </button>
    </div>
  </section>

  <section id="setup" hidden>
    <div class="card">
      <h3>Persona</h3>
      <label>Name</label><input id="p_name">
      <label>Personality (initial prompt — it evolves from here)</label>
      <textarea id="p_prompt" style="min-height:120px"></textarea>
      <div class="hint">Personality now evolves on its own during the Mind tab's dream cycle.</div>
      <button class="go" onclick="savePersona()">Save persona</button>
      <button class="ghost" style="margin-left:8px" onclick="resetTraits()">Reset evolved traits</button>
      <span id="pState" class="mut" style="margin-left:10px"></span>
      <div style="margin-top:14px"><div class="mut" style="font-size:13px">Evolved traits (self-developed over time):</div>
        <ul class="traitlist" id="traitlist"></ul></div>
    </div>
    <div class="card">
      <h3>Ollama Cloud</h3>
      <label>Ollama Cloud API key <span class="mut" id="keyState"></span></label>
      <input id="ollama_api_key" type="password" placeholder="paste key to enable the cloud rung (blank keeps current)">
      <div class="hint">Create at ollama.com → Settings → API keys. Stored in .env, never in the config file.</div>
      <label>Ollama Cloud URL</label><input id="ollama_cloud_url">
    </div>
    <div class="card">
      <h3>Local Ollama</h3>
      <label>Local Ollama URL</label><input id="ollama_local_url">
      <div class="hint" id="localState"></div>
    </div>
    <div class="card">
      <h3>Web search <span class="mut" id="tavilyState"></span></h3>
      <div class="hint">Tiered, cheapest-first: self-hosted SearXNG → Tavily → DuckDuckGo. Each tier is used only if the ones above it are unset or come up empty.</div>
      <label>SearXNG URL (self-hosted, primary — free &amp; private)</label>
      <input id="searxng_url" placeholder="http://archive:8088  (blank to skip)">
      <div class="hint">Your own metasearch instance. Must have the JSON format enabled.</div>
      <label style="margin-top:12px">Tavily API key (escalation)</label>
      <input id="tavily_api_key" type="password" placeholder="paste tvly-... to enable (blank keeps current)">
      <div class="hint">Free key at tavily.com → dashboard (~1,000 searches/month). Stored in .env, never in the config file.</div>
    </div>
    <div class="card">
      <h3>Tailscale <span class="mut" id="tsState"></span></h3>
      <div class="hint">Reach Lara from anywhere over your private tailnet, and let her sense your other devices.</div>
      <div id="tsBody" class="mut" style="margin-top:10px">loading…</div>
      <div id="tsActions" style="margin-top:12px"></div>
      <div id="tsMsg" class="hint" style="margin-top:8px"></div>
    </div>
    <div class="card">
      <h3>Backups <span class="mut" id="bkState"></span></h3>
      <div class="hint">Snapshots of the family's irreplaceable data — memory, learned skills, and every conversation. One is taken automatically each day (during sleep) and the last 14 are kept in <code>backups/</code>. Git only protects the code.</div>
      <div style="margin-top:10px"><button class="go" onclick="backupNow()">Back up now</button>
        <span id="bkMsg" class="mut" style="margin-left:10px"></span></div>
      <div id="bkList" class="feed" style="max-height:22vh;margin-top:10px">loading…</div>
    </div>
    <div class="card">
      <h3>Models per rung</h3>
      <div class="hint">Rung 0 is the local workhorse; the cloud rung also runs the Planner &amp; Critic.</div>
      <label>Rung 0 — local-fast</label><input id="m_local-fast" list="localmodels">
      <label>Rung 1 — local-reason</label><input id="m_local-reason" list="localmodels">
      <label>Rung 2 — cloud-open (Ollama Cloud)</label><input id="m_cloud-open">
      <datalist id="localmodels"></datalist>
    </div>
    <div class="card">
      <h3>Behaviour</h3>
      <div class="grid">
        <div><label>Confidence floor (escalate below)</label><input id="confidence_floor" type="number" step="0.05"></div>
        <div><label>Note token budget</label><input id="note_token_budget" type="number"></div>
        <div><label>Request timeout (s)</label><input id="request_timeout" type="number"></div>
        <div><label>Embeddings (semantic recall)</label>
          <select id="use_embeddings"><option value="true">on</option><option value="false">off</option></select></div>
        <div><label>Daily cloud budget ($ hard cap)</label><input id="daily_cost_cap_usd" type="number" step="0.5"></div>
        <div><label>Background budget ($ soft cap)</label><input id="background_cost_cap_usd" type="number" step="0.5"></div>
      </div>
      <div class="hint">Background work (dreams, self-dev, hive) stops using the cloud past the soft cap — protecting your chat up to the hard cap. At the hard cap Lara drops to local models and pushes you once.</div>
      <label>Lara's privileges (how much she may do without asking)</label>
      <select id="autonomy">
        <option value="ask">Ask first — every action needs your approval</option>
        <option value="trusted">Trusted — read-only commands run free, the rest asks (recommended)</option>
        <option value="auto">Autonomous — never asks (hard denylist still applies)</option>
      </select>
      <div class="hint">"Always allow" on an approval card teaches Trusted mode new commands, one at a time.</div>
      <label>Answer synthesis (local model always fronts the chat &amp; runs tools)</label>
      <select id="synthesis_mode">
        <option value="local">Local only — qwen writes every answer (fastest, fully private)</option>
        <option value="balanced">Balanced — cloud writes substantial answers, house/quick stay local (recommended)</option>
        <option value="cloud">Cloud-heavy — cloud writes everything but trivial (house data may leave the box)</option>
      </select>
      <div class="hint">Balanced keeps house/family data on your machine; only Cloud-heavy lets it reach the cloud experts.</div>
      <div class="grid">
        <div><label>Quiet hours start (0–23)</label><input id="push_quiet_start" type="number" min="0" max="23"></div>
        <div><label>Quiet hours end (0–23)</label><input id="push_quiet_end" type="number" min="0" max="23"></div>
      </div>
      <div class="hint">During quiet hours, ambient pushes (Lara's thoughts &amp; dreams) hold; approvals, answers, and severe-weather alerts still come through.</div>
      <label>Home location (for weather &amp; local info)</label>
      <input id="home_address" placeholder="city, state or full address — leave blank to use Home Assistant's">
      <div class="hint">Sets where “what's the weather” means. If blank, Lara uses Home Assistant's home — but HA ships with Amsterdam as a placeholder, so set this if you're not there.</div>
      <label>Discord webhook (optional)</label>
      <input id="discord_webhook_url" placeholder="https://discord.com/api/webhooks/...">
      <button class="go" onclick="saveCfg()">Save configuration</button>
      <span id="saveState" class="mut" style="margin-left:10px"></span>
    </div>
  </section>

  <section id="profile" hidden>
    <div class="card">
      <h3>Family profiles</h3>
      <div class="hint">The <b>first profile is the admin</b> (household manager — always an adult; only they can manage profiles &amp; server settings). Everyone else is a user account that's an <b>adult</b> or a <b>child</b>. Give adults a PIN — then only a verified adult can approve dangerous actions (unlock the door, run commands, change the house); a child can chat and check the house but never approve those.</div>
      <div id="profAdminNote" class="hint" style="color:var(--acc)"></div>
      <div id="profRows" style="margin-top:10px"></div>
      <button class="ghost" id="addProfBtn" style="margin-top:6px" onclick="addProfRow()">+ Add person</button>
      <div style="margin-top:12px"><button class="go" id="saveProfBtn" onclick="saveProfiles()">Save profiles</button>
        <span id="profState" class="mut" style="margin-left:10px"></span></div>
    </div>
    <div class="card">
      <h3>Notifications <span class="mut" id="npState"></span></h3>
      <div class="hint">Push to <b>this device</b> when Lara answers you, needs approval, or has a notable thought. Each family member enables this on their own phone. Ambient pushes hold during quiet hours (10pm&ndash;7am).</div>
      <div id="npActions" style="margin-top:12px">
        <button class="go" style="margin:0" onclick="npEnable()">Enable on this device</button>
        <button class="ghost" onclick="npDisable()">Disable</button>
        <button class="ghost" onclick="npTest()">Send test</button>
      </div>
      <div id="npMsg" class="hint" style="margin-top:8px"></div>
      <label style="display:flex;align-items:center;gap:8px;margin-top:12px;cursor:pointer">
        <input type="checkbox" id="pushSticky" onchange="setSticky(this.checked)"> Keep me logged in on this device
      </label>
      <div class="hint">On: this device stays yours — your notifications keep arriving here even after you close the app. Off: logging out unbinds it.</div>
    </div>
    <div class="card" id="remindCard" hidden>
      <h3>Reminders from family</h3>
      <div class="hint">Choose who can have Lara send YOU a reminder ("Lara, remind me…" on their end). Allowing someone makes your phone a target for their reminders. Your kids can always be reached by you.</div>
      <div id="remindRows" style="margin-top:10px" class="mut">loading…</div>
      <div style="margin-top:10px"><button class="go" onclick="savePushAllow()">Save</button>
        <span id="remindMsg" class="mut" style="margin-left:10px"></span></div>
    </div>
  </section>

  <section id="pulse" hidden>
    <div class="card">
      <h3>Autopilot <span class="mut" id="apState"></span></h3>
      <div class="mut" id="personaLine" style="margin-bottom:10px">loading…</div>
      <div class="statline" id="apLine" style="margin-bottom:14px">loading…</div>
      <div class="tiles" id="pulseTiles"></div>
      <div class="mut" id="pulseGov" style="font-size:12px;margin-top:10px"></div>
      <div style="margin-top:14px">
        <button class="go" onclick="thinkNow()">Think now</button>
        <button class="ghost" style="margin-left:8px" onclick="dreamNow()">Dream now</button>
        <span id="mState" class="mut" style="margin-left:10px"></span>
      </div>
    </div>
    <div class="grid" id="pulseDreamGrid">
      <div class="card" id="dreamCard"><h3>Dreams <span class="mut">— consolidation</span></h3>
        <div class="feed" id="dreamFeed" style="max-height:26vh">loading…</div></div>
      <div class="card" id="devSelfCard"><h3>Deep sleep <span class="mut">— self-development</span></h3>
        <div class="feed" id="selfdevFeed" style="max-height:26vh">loading…</div></div>
    </div>
    <div class="card" id="devCouncilCard"><h3>Council backlog <span class="mut" id="qCounts"></span></h3>
      <div class="hint">One shared queue: ANVIL builds tickets itself during deep sleep (local &rarr; cloud); heavier work waits for the external cloud council.</div>
      <div id="queueList" style="margin-top:10px" class="mut">loading…</div>
      <div class="hint" style="margin-top:12px">Recent commits on <code>forge-auto</code>:</div>
      <div class="feed" id="commitFeed" style="max-height:20vh;margin-top:6px">—</div>
    </div>
    <div class="card"><h3>Thought stream <span class="mut" id="thoughtWho"></span></h3>
      <div class="feed" id="thoughtFeed" style="max-height:42vh">loading…</div></div>
  </section>

  <section id="memory" hidden>
    <div class="card"><h3>Skills <span class="mut" id="skillCount"></span></h3>
      <div class="hint">Procedures Lara has learned and can reuse verbatim. Unlike notes, skills never fade — a working recipe is kept until replaced.</div>
      <div id="skillList" class="mut" style="margin-top:10px">loading…</div>
    </div>
    <div class="card"><h3>Long-term memory <span class="mut" id="memCount"></span></h3>
      <div class="hint">What ANVIL has chosen to keep — lessons, facts about you, references. Strengthened when used, forgotten when they fade (during sleep).</div>
      <input id="memSearch" placeholder="filter notes by text or tag" oninput="renderMem()" style="margin:12px 0">
      <div id="memList" class="mut">loading…</div>
    </div>
  </section>


  <section id="shared" hidden>
    <div class="card"><h3>Shared with you <span class="mut" id="sharedCount"></span></h3>
      <div class="hint">Memories other family members chose to share with you. Lara uses these as context when she's helping you — just like your own memories.</div>
      <div id="sharedList" class="mut" style="margin-top:12px">loading…</div>
    </div>
  </section>

  <section id="jobs" hidden>
    <div class="card"><h3>Scheduled jobs</h3>
      <div class="hint">Jobs run automatically when ANVIL is left running (the scheduler also runs via <code>python -m anvil serve</code>). “Run” executes one now.</div>
      <div id="jobsList" style="margin-top:10px" class="mut">loading…</div>
    </div>
    <div class="card"><h3>New job</h3>
      <label>Name</label><input id="j_name" placeholder="cert-watch">
      <label>Schedule — cron (min hour day month weekday)</label><input id="j_cron" placeholder="0 7 * * 1">
      <div class="hint">examples: <code>0 7 * * *</code> daily 7am · <code>0 8 * * 1</code> Mondays 8am · <code>*/30 * * * *</code> every 30 min</div>
      <label>Prompt / task</label><textarea id="j_prompt" placeholder="Summarize overnight homelab health and flag anything that needs me."></textarea>
      <div class="grid">
        <div><label>Notify</label><select id="j_notify"><option value="">Push to my phone (default)</option><option value="discord">Push + Discord</option></select></div>
        <div><label>Start rung</label><select id="j_rung"><option value="0">local-fast</option><option value="1">local-reason</option><option value="2">cloud-open</option></select></div>
      </div>
      <div class="hint">Results always push to your phone (PWA). Discord is an optional extra.</div>
      <button class="go" onclick="saveJob()">Save job</button>
      <span id="jState" class="mut" style="margin-left:10px"></span>
    </div>
  </section>
</main>

<div class="overlay" id="wizard" hidden>
  <div class="wiz">
    <h2>FORGE YOUR AGENT</h2>
    <p>Give it a name and a personality. This is just the seed — it grows its own
       traits as you work together.</p>
    <label>Name</label><input id="w_name" value="Anvil">
    <label>Personality</label>
    <textarea id="w_prompt" style="min-height:150px"></textarea>
    <div><button class="go" id="forgeBtn" onclick="forge()">Forge it</button>
      <button class="ghost" style="margin-left:8px" onclick="closeWizard()">Skip for now</button>
      <span id="wState" class="mut" style="margin-left:10px"></span></div>
    <div class="hint" style="margin-top:10px">ANVIL UI build __UI_BUILD__</div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let AGENT="Anvil";
function freshSid(){ return 'c'+Date.now().toString(36)+Math.random().toString(36).slice(2,8); }
let SID=localStorage.getItem('anvil-sid'); if(!SID){ SID=freshSid(); localStorage.setItem('anvil-sid',SID); }
const FLAME='<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12.9 2.2c.35 2.3-.05 3.9-1 5.2-.6.8-1.3 1.5-1.8 2.5-.4.85-.5 1.8-.25 2.7-1-.35-1.75-1.25-2.1-2.55C6.6 11.1 6 12.5 6 14.2 6 17.4 8.7 20 12 20s6-2.6 6-5.8c0-2-.85-3.4-1.8-4.8-.9-1.3-1.9-2.5-2.4-4.4-.15-.9-.6-1.9-.9-2.8z"/><path fill="currentColor" opacity=".55" d="M12.2 12.3c.9.9.3 2.9-.9 3.2 1.2.35 2.6-.4 2.6-1.9 0-1-.5-1.7-1.1-2.4-.15.55-.35.9-.6 1.1z"/></svg>';
/* ---- theme (light / dark / auto) ---- */
const THEMES=['auto','light','dark'];
const _sun='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const _moon='<svg viewBox="0 0 24 24"><path fill="currentColor" d="M21 12.8A9 9 0 1111.2 3 7 7 0 0021 12.8z"/></svg>';
const _auto='<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path fill="currentColor" d="M12 3a9 9 0 010 18z"/></svg>';
function themeMode(){ return localStorage.getItem('anvil-theme')||'dark'; }
function effTheme(m){ return m==='auto' ? (matchMedia('(prefers-color-scheme: light)').matches?'light':'dark') : m; }
function applyTheme(m){ localStorage.setItem('anvil-theme',m); document.documentElement.dataset.theme=effTheme(m);
  const b=$('themebtn'); if(b){ const ic={auto:_auto,light:_sun,dark:_moon}[m]; const lb={auto:'Auto',light:'Light',dark:'Dark'}[m]; b.innerHTML=ic+'<span>'+lb+'</span>'; } }
function cycleTheme(){ applyTheme(THEMES[(THEMES.indexOf(themeMode())+1)%3]); }
applyTheme(themeMode());
try{ matchMedia('(prefers-color-scheme: light)').addEventListener('change',()=>{ if(themeMode()==='auto') applyTheme('auto'); }); }catch(e){}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  ['chat','pulse','memory','shared','jobs','profile','setup'].forEach(s=>$(s).hidden = s!==b.dataset.t);
  const t=b.dataset.t;
  if(t==='pulse') loadPulse();
  if(t==='memory') loadMemory();
  if(t==='shared') loadShared();
  if(t==='jobs') loadJobs();
  if(t==='profile'){ loadProfiles(); npRefresh(); loadRemind(); }
  if(t==='setup'){ loadTailscale(); loadBackups(); }
});
async function jget(u){const r=await fetch(u+(u.includes('?')?'&':'?')+'_='+Date.now(),{cache:'no-store'});
  if(r.status===401){showLogin();throw new Error('auth');} return r.json();}
async function jpost(u,d,signal){const r=await fetch(u,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(d),signal});
  if(r.status===401 && u!=='/api/login'){showLogin();throw new Error('auth');} return r.json();}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function now(){return new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});}

/* ---- modern message rendering ---- */
// Stick-to-bottom, driven by the reader's INTENT (scroll actions), not by
// measuring right after content lands. Measuring post-append is buggy: a chunk
// taller than the threshold opens a gap that reads as "scrolled up", so it
// stops following and never recovers. Instead STICK is recomputed only when the
// user actually scrolls; content growth alone never changes it.
let STICK=true;
function bindScrollStick(){
  const m=$('msgs'); if(!m||m._stickBound) return; m._stickBound=true;
  m.addEventListener('scroll',()=>{
    // Generous threshold: sitting anywhere near the live edge counts as stuck.
    STICK = m.scrollHeight-m.scrollTop-m.clientHeight < 80;
  }, {passive:true});
}
function scrollMsgs(force){
  const m=$('msgs'); if(!m) return;
  if(force){ STICK=true; m.scrollTop=m.scrollHeight; return; }
  if(STICK) m.scrollTop=m.scrollHeight;
}
function clearHello(){ const h=$('hello'); if(h) h.remove(); }
function msgSys(text){ clearHello(); const d=document.createElement('div'); d.className='sysmsg'; d.textContent=text; $('msgs').appendChild(d); scrollMsgs(); return d; }
function msgUser(text){ clearHello(); const d=document.createElement('div'); d.className='msg user'; const b=document.createElement('div'); b.className='bubble'; b.textContent=text; d.appendChild(b); $('msgs').appendChild(d); scrollMsgs(true); return d; }
function emberMsg(){ clearHello(); const d=document.createElement('div'); d.className='msg ember'; d.innerHTML='<div class="av" aria-hidden="true">'+FLAME+'</div><div class="content"><div class="who">Lara</div></div>'; $('msgs').appendChild(d); scrollMsgs(); return {el:d, content:d.querySelector('.content')}; }
function line(kind,nick,text){ if(kind==='sys') return msgSys(text); if(kind==='me') return msgUser(text); return lineMD(nick,text); }
/* ---- tiny self-contained markdown renderer (no external deps) ---- */
function mdInline(s){
  return s
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g,'$1<em>$2</em>')
    .replace(/(^|[^_])_([^_\n]+)_(?!_)/g,'$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function md(src){
  const lines=(src||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])).split('\n');
  let out='',i=0;
  const blockStart=/^(#{1,6}\s|```|&gt;\s?|\s*[-*+]\s+|\s*\d+\.\s+|---\s*$|\*\*\*\s*$|___\s*$)/;
  while(i<lines.length){
    let l=lines[i];
    if(/^```/.test(l)){ let code=[]; i++; while(i<lines.length&&!/^```/.test(lines[i])){code.push(lines[i]);i++;} i++; out+='<pre><code>'+code.join('\n')+'</code></pre>'; continue; }
    let h=l.match(/^(#{1,6})\s+(.*)$/); if(h){ const n=h[1].length; out+='<h'+n+'>'+mdInline(h[2])+'</h'+n+'>'; i++; continue; }
    if(/^(---|\*\*\*|___)\s*$/.test(l)){ out+='<hr>'; i++; continue; }
    if(/^&gt;\s?/.test(l)){ let q=[]; while(i<lines.length&&/^&gt;\s?/.test(lines[i])){q.push(lines[i].replace(/^&gt;\s?/,''));i++;} out+='<blockquote>'+mdInline(q.join(' '))+'</blockquote>'; continue; }
    if(/^\s*[-*+]\s+/.test(l)){ let it=[]; while(i<lines.length&&/^\s*[-*+]\s+/.test(lines[i])){it.push(lines[i].replace(/^\s*[-*+]\s+/,''));i++;} out+='<ul>'+it.map(x=>'<li>'+mdInline(x)+'</li>').join('')+'</ul>'; continue; }
    if(/^\s*\d+\.\s+/.test(l)){ let it=[]; while(i<lines.length&&/^\s*\d+\.\s+/.test(lines[i])){it.push(lines[i].replace(/^\s*\d+\.\s+/,''));i++;} out+='<ol>'+it.map(x=>'<li>'+mdInline(x)+'</li>').join('')+'</ol>'; continue; }
    if(/^\s*$/.test(l)){ i++; continue; }
    let para=[]; while(i<lines.length&&!/^\s*$/.test(lines[i])&&!blockStart.test(lines[i])){para.push(lines[i]);i++;}
    out+='<p>'+mdInline(para.join('<br>'))+'</p>';
  }
  return out;
}
function lineMD(nick,text){ const m=emberMsg(); const b=document.createElement('div'); b.className='md'; b.innerHTML=md(text); m.content.appendChild(b); scrollMsgs(); return m; }
let PENDING=null, PENDINGEL=null, PENDING_ADULT=false;
// In-flight requests in SUBMISSION order. [0] is the primary — the one the
// server is actually running (it holds the per-conversation lock); the rest are
// queued behind it. Stop always targets the primary, never the newest.
let INFLIGHTS=[];
function sendIcon(stop){ return stop
  ? '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/></svg>'
  : '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M3.4 20.4 21 12 3.4 3.6 3.4 10l12 2-12 2z"/></svg>'; }
function refreshSendBtn(){ const b=$('send'); if(!b) return;
  const on=INFLIGHTS.length>0;
  b.classList.toggle('stopping', on);
  b.title=on?(INFLIGHTS.length>1?'Stop current task ('+INFLIGHTS.length+' queued)':'Stop'):'Send (Enter)';
  b.setAttribute('aria-label', on?'Stop':'Send'); b.innerHTML=sendIcon(on); }
function onSend(){ if(INFLIGHTS.length) stopAsk(); else ask(); }
async function stopAsk(){
  const f=INFLIGHTS[0]; if(!f) return;   // the PRIMARY running task, not the newest
  f.stop();                              // immediate UI: remove its bubble, 'stopped'
  try{ f.ctrl.abort(); }catch(e){}       // unblock the client for that request
  try{ await fetch('/api/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rid:f.rid})}); }catch(e){}
}
async function ask(){
  let t=$('task').value.trim();
  if(!t && !ATTACH.length) return;
  if(!t) t='What do you see in this?';
  // Slash commands only when it's a SINGLE-LINE message starting with one '/'
  // — not '//' (comments/URLs) and not a multi-line paste (config, code).
  if(t[0]==='/' && t[1]!=='/' && t.indexOf('\n')<0){ $('task').value=''; grow(); return handleCommand(t); }
  $('task').value=''; grow();
  const sending=ATTACH.slice(); ATTACH=[]; attachRender();
  const um=msgUser(t);
  if(sending.length){ const g=document.createElement('div'); g.className='imgs';
    sending.forEach(a=>{ const im=document.createElement('img'); im.src=a.thumb; g.appendChild(im); });
    um.appendChild(g); scrollMsgs(); }
  const rid=Math.random().toString(36).slice(2);
  const wm=emberMsg();
  const w=document.createElement('div'); w.className='wline';
  w.innerHTML='<span class="wtext">working</span><span class="dots"><i></i><i></i><i></i></span><span class="wctx"></span>';
  wm.content.appendChild(w); const wtext=w.querySelector('.wtext'), wctx=w.querySelector('.wctx'); scrollMsgs();
  const nap=ms=>new Promise(r=>setTimeout(r,ms));
  const ctrl=new AbortController();
  let live=true, q=[], seen=0, sEl=null, tEl=null;
  const entry={rid, ctrl};
  const drop=()=>{ live=false; INFLIGHTS=INFLIGHTS.filter(x=>x!==entry); refreshSendBtn(); };
  entry.stop=()=>{ if(!live) return; drop(); try{wm.el.remove();}catch(e){} msgSys('stopped'); };
  INFLIGHTS.push(entry); refreshSendBtn();
  (async()=>{ while(live){ try{ const p=await jget('/api/progress?id='+rid);
      const st=(p&&p.steps)||[]; while(seen<st.length) q.push(st[seen++]);
      if(p&&p.ctx) wctx.textContent='ctx '+(p.ctx>=1000?(p.ctx/1000).toFixed(1)+'k':p.ctx);
      const part=(p&&p.partial)||'';
      const tk=(p&&p.thinking)||'';
      // Live deliberation: show the reasoning trace while it streams; fold it
      // away the moment the actual answer starts (or this step's trace ends).
      if(tk && !part){ if(!tEl){ tEl=document.createElement('div'); tEl.className='think'; wm.content.appendChild(tEl); }
        const atTail=tEl.scrollHeight-tEl.scrollTop-tEl.clientHeight<24;
        tEl.textContent=tk; if(atTail) tEl.scrollTop=tEl.scrollHeight; scrollMsgs(); }
      else if(tEl){ tEl.remove(); tEl=null; }
      if(part){ if(!sEl){ sEl=document.createElement('div'); sEl.className='md streaming'; wm.content.appendChild(sEl); }
        sEl.textContent=part; w.style.display='none'; scrollMsgs(); }
      else if(sEl){ sEl.textContent=''; w.style.display=''; }
    }catch(e){} await nap(250); } })();
  (async()=>{ while(live){ if(q.length){ const s=q.shift();
      if(wtext&&live) wtext.textContent=s; await nap(700);
    } else await nap(120); } })();
  let r;
  try{ r=await jpost('/api/ask',{task:t,rid:rid,sid:SID,geo:geoNow(),images:sending.map(a=>a.b64),tools:true}, ctrl.signal); }
  catch(e){ if(live) drop(); return; }   // aborted by Stop (entry.stop already cleaned up) or network
  if(!live) return;                       // was stopped while we awaited
  drop(); try{wm.el.remove();}catch(e){}
  if(r && r.status==='cancelled'){ msgSys('stopped'); if(chatsOpen()) chatsRefresh(); return; }
  renderResult(r);
  if(chatsOpen()) chatsRefresh();   // docked sidebar: keep titles/times fresh
}
async function restoreChat(){
  let h; try{ h=await jget('/api/conversation?sid='+SID); }catch(e){ h=null; }
  const turns=(h&&h.turns)||[];
  if(!turns.length){ greet(); return; }
  greeted=true; $('msgs').innerHTML='';
  turns.forEach(t=>{ if(t.role==='user') msgUser(t.content); else lineMD(AGENT,t.content); });
  scrollMsgs(true);
}
function newChat(){ SID=freshSid(); localStorage.setItem('anvil-sid',SID); $('msgs').innerHTML=''; greeted=false; greet(); $('task').focus(); }
// Cross-device approval routing. An adult device polls for the danger actions a
// child raised and can approve/deny them remotely with full context (who + why).
let FAMREQ_SIG='';
async function pollApprovals(){
  const box=$('famReqs'); if(!box) return;
  let r; try{ r=await jget('/api/approvals'); }catch(e){ return; }
  if(!r.adult || !(r.pending||[]).length){ box.hidden=true; box.innerHTML=''; FAMREQ_SIG=''; return; }
  const sig=r.pending.map(p=>p.token).join(',');
  if(sig===FAMREQ_SIG){ return; }   // no change — don't stomp a click mid-render
  FAMREQ_SIG=sig; box.hidden=false;
  box.innerHTML=r.pending.map(p=>{ const tk=JSON.stringify(p.token).replace(/"/g,'&quot;');
    return '<div class="famreq"><div><span class="who">'+esc(p.who||'Someone')+'</span> wants to run <code>'+
      esc(p.summary||p.tool)+'</code></div>'+
      (p.why?'<div class="why">“'+esc(p.why)+'”</div>':'')+
      '<div class="r2"><button class="go" style="margin:0" onclick="resolveFamReq('+tk+',\'approve\')">Approve</button>'+
      '<button class="ghost" onclick="resolveFamReq('+tk+',\'deny\')">Deny</button></div></div>';
  }).join('');
}
async function resolveFamReq(token, decision){
  await jpost('/api/approve',{token:token,decision:decision});
  FAMREQ_SIG=''; pollApprovals();       // refresh the queue immediately
}
// The requester's own device keeps watching its pending card — if an adult
// resolves it elsewhere, the outcome routes back here and the card updates.
function watchPending(token, el){
  const iv=setInterval(async()=>{
    if(PENDING!==token){ clearInterval(iv); return; }   // resolved on this device
    let r; try{ r=await jget('/api/approval/poll?token='+encodeURIComponent(token)); }catch(e){ return; }
    if(r.resolved){
      clearInterval(iv); PENDING=null; PENDINGEL=null;
      const ok=r.decision!=='deny', r2=el.querySelector('.r2');
      if(r2) r2.innerHTML='<span class="verdict '+(ok?'ok':'no')+'">'+
        (ok?'✓ Approved':'✕ Denied')+(r.by?' by '+esc(r.by):'')+'</span>';
      el.classList.add('resolved');
      if(r.answer) renderResult({status:r.status,answer:r.answer});
    } else if(!r.pending){ clearInterval(iv); }   // expired server-side
  }, 3000);
}
function renderResult(r){
  if(r.error){ msgSys('error: '+r.error); return; }
  const m=emberMsg();
  const tr=r.trace||[], st=r.steps||[];
  // The live "working" subtext compacts into this once the answer lands:
  // a one-line human summary that expands into phases, thoughts and tool logs.
  if(st.length || tr.some(x=>x.think) || tr.length>2) m.content.appendChild(traceEl(r));
  if(r.status==='approve'){ m.content.appendChild(approvalEl(r)); scrollMsgs(); return; }
  if(r.answer){ const b=document.createElement('div'); b.className='md'; b.innerHTML=md(r.answer); m.content.appendChild(b); }
  const parts=['via '+(r.rung||'?')];
  if(r.verdict) parts.push(r.verdict);
  if(r.escalations&&r.escalations.length) parts.push('escalated '+r.escalations.length+'x');
  if(r.recalled) parts.push(r.recalled+' notes recalled');
  if(r.ctx) parts.push('ctx '+(r.ctx>=1000?(r.ctx/1000).toFixed(1)+'k':r.ctx)+' tok');
  if(typeof r.cost==='number'&&r.cost>0) parts.push('$'+r.cost);
  const mv=document.createElement('div'); mv.className='meta'; mv.innerHTML=parts.map(esc).join('<span class="d"></span>'); m.content.appendChild(mv);
  if(r.evolved){ const e=document.createElement('div'); e.className='meta'; e.textContent='developed a new trait: '+r.evolved; m.content.appendChild(e); }
  scrollMsgs();
}
function stepEl(s){
  const a=s.args||{}; const arg=a.cmd||a.path||a.url||a.query||a.entity_id||a.domain||a.host||'';
  const d=document.createElement('div'); d.className='step'+(s.approved===false?' denied':'');
  d.innerHTML='<span class="sd"></span>'+(s.approved===false?'skipped':'used')+' <b>'+esc(s.tool)+'</b>'+(arg?' <code>'+esc(String(arg).slice(0,90))+'</code>':'')+
    (s.auto&&s.danger?' <span class="mut" title="allowed by your privilege setting">· ran without asking</span>':'');
  return d;
}
function traceEl(r){
  const tr=r.trace||[], st=r.steps||[];
  const d=document.createElement('details'); d.className='trace';
  const sum=document.createElement('summary');
  const secs=tr.length?tr[tr.length-1].t:0;
  const bits=[];
  if(st.length) bits.push(st.length+' action'+(st.length>1?'s':''));
  if(tr.some(x=>x.think)) bits.push('thought it through');
  if(secs>=1) bits.push(secs+'s');
  sum.textContent='how I got here'+(bits.length?' · '+bits.join(' · '):'');
  d.appendChild(sum);
  let lastPhase='';
  tr.forEach(x=>{
    if(x.phase!==lastPhase){ lastPhase=x.phase;
      const row=document.createElement('div'); row.className='trow';
      row.textContent=x.phase+(x.t>=1?' · '+x.t+'s':''); d.appendChild(row); }
    if(x.think){
      const td=document.createElement('details'); td.className='tthink';
      const ts=document.createElement('summary'); ts.textContent='her thoughts here'; td.appendChild(ts);
      const body=document.createElement('div'); body.className='think tfull'; body.textContent=x.think;
      td.appendChild(body); d.appendChild(td); }
  });
  st.forEach(s=>{
    const sd=document.createElement('details'); sd.className='tstep';
    const ss=document.createElement('summary'); ss.appendChild(stepEl(s)); sd.appendChild(ss);
    const obs=document.createElement('div'); obs.className='tobs';
    obs.textContent=String(s.observation||'(no output)').slice(0,2000);
    sd.appendChild(obs); d.appendChild(sd);
  });
  return d;
}
function approvalEl(r){
  PENDING=r.token; PENDING_ADULT=!!r.adult_required;
  const a=r.pending.args||{}; const arg=a.cmd||a.path||a.url||JSON.stringify(a);
  const d=document.createElement('div'); d.className='approve';
  const always=r.pending.tool==='shell'
    ? '<button class="ghost" onclick="decide(\'always\')" title="Run it now and never ask again for this exact command">Always allow</button>' : '';
  // A minor / unverified session raised a danger action — an adult PIN is required.
  const pinField=r.adult_required
    ? '<div class="hint" style="margin:6px 0 2px;color:var(--acc)">An adult must approve this.</div>'
      +'<input id="approvePin" type="password" inputmode="numeric" placeholder="adult PIN" style="max-width:160px">' : '';
  d.innerHTML='Lara wants to run <b>'+esc(r.pending.tool)+'</b>: <code>'+esc(arg)+'</code>'+pinField+
    '<div class="r2"><button class="go" style="margin:0" onclick="decide(\'approve\')">Approve</button>'+always+
    '<button class="ghost" onclick="decide(\'deny\')">Deny</button></div>';
  PENDINGEL=d;
  // If a child raised this, an adult may resolve it on another device — watch
  // for that so the answer routes back to this screen too.
  if(r.adult_required && r.token) watchPending(r.token, d);
  return d;
}
async function decide(decision){
  if(!PENDING) return;                       // already resolved — nothing to do
  const pinEl=PENDINGEL&&PENDINGEL.querySelector('#approvePin');
  const pin=pinEl?pinEl.value.trim():'';
  if(PENDING_ADULT && decision!=='deny' && !pin){ if(pinEl){pinEl.focus();pinEl.style.borderColor='var(--bad)';} return; }
  const tok=PENDING, el=PENDINGEL, wasAdult=PENDING_ADULT;
  // Send first when a PIN gate is involved so a wrong PIN can re-prompt without
  // having destroyed the card; otherwise resolve-in-place immediately as before.
  if(!wasAdult){
    PENDING=null; PENDINGEL=null;
    if(el){ const r2=el.querySelector('.r2'); const ok=decision!=='deny';
      if(r2) r2.innerHTML='<span class="verdict '+(ok?'ok':'no')+'">'+
        (decision==='always'?'✓ Approved · always':(ok?'✓ Approved':'✕ Denied'))+'</span>';
      el.classList.add('resolved'); }
  }
  const r=await jpost('/api/approve',{token:tok,decision:decision,pin:pin});
  if(r&&r.adult_required&&r.error){        // wrong/absent PIN — keep the card live to retry
    if(pinEl){pinEl.value='';pinEl.style.borderColor='var(--bad)';pinEl.placeholder='wrong PIN — try again';pinEl.focus();}
    return;
  }
  if(wasAdult){ PENDING=null; PENDINGEL=null;
    if(el){ const r2=el.querySelector('.r2'); const ok=decision!=='deny';
      if(r2) r2.innerHTML='<span class="verdict '+(ok?'ok':'no')+'">'+
        (decision==='always'?'✓ Approved · always':(ok?'✓ Approved':'✕ Denied'))+'</span>';
      el.classList.add('resolved'); } }
  renderResult(r);
}
async function handleCommand(t){
  const parts=t.slice(1).split(' '); const cmd=parts[0].toLowerCase(); const arg=parts.slice(1).join(' ').trim();
  if(cmd==='help'){ line('sys',null,'commands: /help · /note <text> · /remember <fact about you> · /status · /reset · /approve · /deny'); return; }
  if(cmd==='reset'||cmd==='new'){ newChat(); return; }
  if(cmd==='approve'||cmd==='deny'){ if(PENDING) decide(cmd); else line('sys',null,'nothing pending to '+cmd); return; }
  if(cmd==='note'){ if(!arg) return line('sys',null,'usage: /note <text>'); await jpost('/api/note',{text:arg}); line('sys',null,'noted.'); return; }
  if(cmd==='remember'){ if(!arg) return line('sys',null,'usage: /remember <fact about you>'); await jpost('/api/remember',{text:arg}); line('sys',null,'remembered (about you): '+arg); return; }
  if(cmd==='status'){ const s=await jget('/api/status'); line('sys',null,'ladder '+s.ladder.join(' → ')+' · ollama '+(s.ollama_reachable?'up':'down')+' · '+s.notes+' notes · '+s.traits+' traits'); return; }
  line('sys',null,'unknown command /'+cmd+' (try /help)');
}
function grow(){ const t=$('task'); t.style.height='auto'; t.style.height=Math.min(t.scrollHeight,Math.round(innerHeight*0.4))+'px'; }
$('task').addEventListener('input',grow);
bindScrollStick();
$('task').addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask();} });
document.addEventListener('keydown',e=>{ if(e.key==='Escape'&&INFLIGHTS.length){ e.preventDefault(); stopAsk(); } });

/* ---- persona ---- */
async function loadPersona(allowWizard=true){
  const p=await jget('/api/persona');
  AGENT=p.name||'Anvil';
  $('p_name').value=p.name||''; $('p_prompt').value=p.base_prompt||'';
  const tl=$('traitlist'); tl.innerHTML='';
  (p.traits||[]).forEach(t=>{const li=document.createElement('li');li.textContent=t;tl.appendChild(li);});
  const pl=$('personaLine'); if(pl){ const n=(p.traits||[]).length;
    pl.textContent=AGENT+' · '+(n?n+' evolved trait'+(n===1?'':'s')+' (see Setup for details)':'fresh persona'); }
  if(allowWizard && !p.configured){ $('w_name').value=p.name||'Anvil'; $('w_prompt').value=p.base_prompt||''; $('wizard').hidden=false; }
  else { restoreChat(); }
}
let greeted=false;
function greet(){ if(greeted) return; greeted=true;
  const d=document.createElement('div'); d.className='hello'; d.id='hello';
  d.innerHTML='<div class="bigav" aria-hidden="true">'+FLAME+'</div><h2>'+esc(AGENT)+'</h2>'+
    '<div>Your household companion, on your own hardware.</div>'+
    '<div class="prompts">'+
    '<button onclick="usePrompt(this)">What is happening in the house?</button>'+
    '<button onclick="usePrompt(this)">Summarize my day</button>'+
    '<button onclick="usePrompt(this)">What have you learned lately?</button>'+
    '</div>';
  $('msgs').appendChild(d);
}
function usePrompt(b){ $('task').value=b.textContent; grow(); $('task').focus(); }
function closeWizard(){ $('wizard').hidden=true; greeted=false; greet(); }
async function forge(){
  const btn=$('forgeBtn'); btn.disabled=true; $('wState').textContent='forging…';
  let p=null;
  try{ p=await jpost('/api/persona',{name:$('w_name').value,base_prompt:$('w_prompt').value,configured:true}); }
  catch(e){ p={error:String(e)}; }
  // Always close the wizard — never trap the user.
  $('wizard').hidden=true; $('wState').textContent=''; btn.disabled=false;
  if(p && !p.error){
    AGENT=p.name||'Anvil';
    try{ await loadPersona(false); }catch(e){}
    line('sys',null,'forged '+AGENT+' — say hi, it grows its own personality as you go');
  } else {
    AGENT=(($('w_name').value||'Anvil').trim())||'Anvil';
    line('sys',null,'⚠ saved for this session only ('+((p&&p.error)||'unknown')+'). To persist, make persona.json writable.');
  }
  greeted=false; greet();
}
async function savePersona(){
  $('pState').textContent='saving…';
  const p=await jpost('/api/persona',{name:$('p_name').value,base_prompt:$('p_prompt').value,configured:true});
  if(p.error){ $('pState').textContent='error: '+p.error; return; }
  AGENT=p.name;
  $('pState').textContent='saved ✓'; setTimeout(()=>$('pState').textContent='',2500);
  loadPersona(false);
}
async function resetTraits(){
  await jpost('/api/persona',{reset_traits:true,configured:true});
  $('pState').textContent='traits cleared'; setTimeout(()=>$('pState').textContent='',2500); loadPersona();
}

/* ---- config ---- */
let CFG=null;
/* ---- family profiles (identity for the danger gate) ---- */
let PROFILES=null;
// Login wall — only appears once an adult sets a password (auth_on). Until then
// the app is open (single-user friendly). The session lives in an HttpOnly cookie
// the server sets on /api/login, so nothing sensitive touches JS.
let ME=null;
async function checkAuth(){
  try{ ME=await (await fetch('/api/me?_='+Date.now(),{cache:'no-store'})).json(); }
  catch(e){ return true; }
  if(ME.needs_setup){ showSetupWizard(); return false; }
  if(ME.auth_on && !ME.authed){ showLogin(); return false; }
  return true;
}
let LOGIN_SHOWN=false;
async function showLogin(){
  if(LOGIN_SHOWN || $('loginOv') || $('setupOv')) return;
  LOGIN_SHOWN=true;                 // sync guard: a burst of 401s builds ONE overlay
  try{ ME=await (await fetch('/api/me?_='+Date.now(),{cache:'no-store'})).json(); }catch(e){}
  if(ME && ME.needs_setup){ showSetupWizard(); return; }
  const ov=document.createElement('div'); ov.className='overlay'; ov.id='loginOv';
  ov.style.zIndex=9999;
  ov.innerHTML='<div class="card" role="dialog" aria-modal="true" aria-label="Sign in" style="max-width:360px;margin:auto"><h3>Sign in to Lara</h3>'+
    '<label for="loginUser">Username</label>'+
    '<input id="loginUser" type="text" autocomplete="username" autocapitalize="none" spellcheck="false" style="width:100%;font-size:16px">'+
    '<label for="loginPwIn" style="margin-top:8px;display:block">Password</label>'+
    '<input id="loginPwIn" type="password" autocomplete="current-password" style="width:100%;font-size:16px">'+
    '<label style="display:flex;align-items:center;gap:8px;margin-top:10px;cursor:pointer">'+
    '<input id="loginRemember" type="checkbox" checked style="width:18px;height:18px"> '+
    '<span>Remember me on this device</span></label>'+
    '<button class="go" style="margin-top:12px;width:100%;min-height:44px" id="loginGo">Sign in</button>'+
    '<div class="hint" style="margin-top:8px">Kids without a password sign in with just their username. '+
    'Had a PIN before passwords existed? It still works here until you set a password.</div>'+
    '<div id="loginErr" class="mut" style="color:var(--bad);margin-top:6px" aria-live="polite"></div></div>';
  document.body.appendChild(ov);
  const go=()=>doLogin($('loginUser').value, $('loginPwIn').value, $('loginRemember').checked);
  $('loginGo').onclick=go;
  $('loginPwIn').onkeydown=e=>{ if(e.key==='Enter') go(); };
  $('loginUser').onkeydown=e=>{ if(e.key==='Enter') $('loginPwIn').focus(); };
  $('loginUser').focus();
}
async function doLogin(user, pw, remember){
  $('loginErr').textContent='';
  const r=await (await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:user,password:pw,remember:!!remember})})).json();
  if(r.ok){ location.reload(); }
  else{ $('loginErr').textContent=r.error||'sign-in failed';
    const inp=$('loginPwIn'); if(inp){ inp.value=''; inp.focus(); } }
}
function showSetupWizard(){
  if($('setupOv')) return;
  const ov=document.createElement('div'); ov.className='overlay'; ov.id='setupOv';
  ov.style.zIndex=9999;
  ov.innerHTML='<div class="card" role="dialog" aria-modal="true" aria-label="Set up Lara" style="max-width:420px;margin:auto;max-height:88vh;overflow-y:auto">'+
    '<h3>Welcome — let’s set up Lara</h3>'+
    '<div class="hint">You’re the first person here, so you’ll be the household admin. '+
    'Add the rest of the family now, or later from the Profile tab.</div>'+
    '<label for="suName" style="margin-top:10px;display:block">Your name</label>'+
    '<input id="suName" type="text" autocomplete="name" style="width:100%;font-size:16px" placeholder="Alex">'+
    '<label for="suUser" style="margin-top:8px;display:block">Username <span class="mut">(for signing in)</span></label>'+
    '<input id="suUser" type="text" autocomplete="username" autocapitalize="none" spellcheck="false" style="width:100%;font-size:16px" placeholder="alex">'+
    '<label for="suPw" style="margin-top:8px;display:block">Password <span class="mut">(6+ characters)</span></label>'+
    '<input id="suPw" type="password" autocomplete="new-password" style="width:100%;font-size:16px">'+
    '<label for="suPin" style="margin-top:8px;display:block">Quick PIN <span class="mut">(optional — fast adult approval for kid requests)</span></label>'+
    '<input id="suPin" type="password" inputmode="numeric" autocomplete="off" style="width:100%;font-size:16px" placeholder="4–6 digits">'+
    '<div style="margin-top:14px"><b>Family members</b> <span class="mut">(optional)</span></div>'+
    '<div id="suMembers"></div>'+
    '<button class="ghost" id="suAdd" style="margin-top:6px;min-height:36px">+ Add a family member</button>'+
    '<button class="go" id="suGo" style="margin-top:14px;width:100%;min-height:44px">Create household</button>'+
    '<div id="suErr" class="mut" style="color:var(--bad);margin-top:6px" aria-live="polite"></div></div>';
  document.body.appendChild(ov);
  $('suName').oninput=()=>{ const u=$('suUser'); if(!u.dataset.touched) u.value=$('suName').value.toLowerCase().replace(/[^a-z0-9._-]/g,''); };
  $('suUser').oninput=()=>{ $('suUser').dataset.touched=1; };
  $('suAdd').onclick=()=>{
    const row=document.createElement('div');
    row.className='sumem'; row.style.cssText='display:flex;gap:6px;margin-top:6px;flex-wrap:wrap';
    row.innerHTML='<input class="smName" placeholder="Name" style="flex:2;min-width:90px;font-size:16px">'+
      '<select class="smRole" style="flex:1;min-width:80px"><option value="adult">adult</option><option value="minor">child</option></select>'+
      '<input class="smPw" type="password" placeholder="Password (kids can skip)" style="flex:2;min-width:120px;font-size:16px">';
    $('suMembers').appendChild(row);
  };
  $('suGo').onclick=async()=>{
    $('suErr').textContent='';
    const members=[...document.querySelectorAll('.sumem')].map(r=>({
      name:r.querySelector('.smName').value.trim(),
      role:r.querySelector('.smRole').value,
      password:r.querySelector('.smPw').value})).filter(m=>m.name);
    const r=await (await fetch('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({admin:{name:$('suName').value.trim(),username:$('suUser').value.trim(),
        password:$('suPw').value,pin:$('suPin').value},members:members})})).json();
    if(r.ok){ location.reload(); }
    else{ $('suErr').textContent=r.error||'setup failed'; }
  };
  $('suName').focus();
}
async function logout(){ await jpost('/api/logout',{}); location.reload(); }

async function refreshWho(){
  try{ PROFILES=await jget('/api/profiles?sid='+SID); }catch(e){ return; }
  const chip=$('whoBtn'), name=$('whoName');
  // Only surface the chip once a family actually has multiple people / a minor —
  // a single-user install stays uncluttered.
  const show=(PROFILES.profiles||[]).length>1 || PROFILES.any_minor;
  chip.style.display=show?'':'none';
  const cur=PROFILES.current||{};
  name.textContent=(cur.name||'—')+(cur.adult?'':' 🔒');
}
function openProfiles(){
  if(!PROFILES) return;
  const ov=document.createElement('div'); ov.className='overlay'; ov.id='whoOv';
  const list=(PROFILES.profiles||[]).map(p=>
    '<button class="ghost" style="display:block;width:100%;text-align:left;margin:4px 0" '+
    'onclick="pickProfile('+JSON.stringify(p.name).replace(/"/g,'&quot;')+','+p.has_pin+')">'+
    esc(p.name)+' <span class="mut">· '+p.role+(p.has_pin?' 🔒':'')+'</span></button>').join('');
  ov.innerHTML='<div class="card" style="max-width:340px;margin:auto"><h3>Who\'s using Lara?</h3>'+
    '<div class="hint">Danger actions (unlock, shell, house changes) need a verified adult.</div>'+
    list+'<div id="whoPin" style="margin-top:8px"></div>'+
    ((ME&&ME.auth_on&&ME.authed)?'<button class="ghost" style="margin-top:8px" onclick="logout()">Sign out'+
      (ME.name?' ('+esc(ME.name)+')':'')+'</button>':'')+
    '<button class="ghost" style="margin-top:8px" onclick="document.getElementById(\'whoOv\').remove()">Close</button></div>';
  document.body.appendChild(ov);
}
async function pickProfile(name, hasPin){
  let pin='';
  if(hasPin){
    const box=$('whoPin');
    box.innerHTML='<input id="whoPinIn" type="password" inputmode="numeric" placeholder="PIN for '+esc(name)+'" style="max-width:200px"> '+
      '<button class="go" style="margin:0" onclick="submitProfile('+JSON.stringify(name).replace(/"/g,'&quot;')+')">Unlock</button>';
    $('whoPinIn').focus(); return;
  }
  submitProfile(name, '');
}
async function submitProfile(name){
  const inp=$('whoPinIn'); const pin=inp?inp.value.trim():'';
  const r=await jpost('/api/profile/select',{sid:SID,name:name,pin:pin});
  if(r.error){ if(inp){inp.value='';inp.style.borderColor='var(--bad)';inp.placeholder='wrong PIN';} return; }
  const ov=$('whoOv'); if(ov) ov.remove();
  await refreshWho();
}
function profRowHtml(p){
  p=p||{name:'',role:'adult',has_pin:false,admin:false};
  // The admin's role is fixed (always adult) and the row can't be deleted; a
  // user account chooses adult vs child and is removable.
  const roleCell = p.admin
    ? '<span class="pill" style="background:var(--acc);color:#fff;flex:none">admin · adult</span>'
    : '<select class="pf-role"><option value="adult"'+(p.role!=='minor'?' selected':'')+'>adult</option>'+
      '<option value="minor"'+(p.role==='minor'?' selected':'')+'>child</option></select>';
  const del = p.admin ? '' :
    '<button class="ghost pf-del" style="padding:4px 8px" onclick="this.closest(\'.profrow\').remove()">✕</button>';
  return '<div class="profrow" data-admin="'+(p.admin?1:0)+'" style="display:flex;gap:6px;align-items:center;margin:4px 0;flex-wrap:wrap">'+
    '<input class="pf-name" placeholder="name" value="'+esc(p.name||'')+'" style="max-width:130px">'+
    roleCell+
    '<input class="pf-pin" type="password" inputmode="numeric" placeholder="'+(p.has_pin?'PIN set — leave blank to keep':'set PIN (adults)')+'" style="max-width:170px">'+
    del+'</div>';
}
async function loadProfiles(){
  let d; try{ d=await jget('/api/profiles?sid='+SID); }catch(e){ return; }
  const box=$('profRows'); if(!box) return;
  box.innerHTML=(d.profiles||[]).map(profRowHtml).join('') || profRowHtml({admin:!d.auth_on});
  // Only the ADMIN may manage the family. Everyone else sees it read-only.
  const canEdit = !d.auth_on || (d.current && d.current.is_admin_session);
  box.querySelectorAll('input, select, .pf-del').forEach(el=>{ el.disabled=!canEdit; if(el.classList.contains('pf-del')) el.style.display=canEdit?'':'none'; });
  if($('addProfBtn')) $('addProfBtn').style.display=canEdit?'':'none';
  if($('saveProfBtn')) $('saveProfBtn').style.display=canEdit?'':'none';
  if($('profAdminNote')) $('profAdminNote').textContent = canEdit ? '' :
    ('Only the admin'+(d.admin_name?' ('+d.admin_name+')':'')+' can manage profiles.');
}
function addProfRow(){ $('profRows').insertAdjacentHTML('beforeend', profRowHtml()); }
async function saveProfiles(){
  const rows=[...document.querySelectorAll('#profRows .profrow')].map(r=>{
    const roleEl=r.querySelector('.pf-role');   // admin row has no select -> adult
    return {name:r.querySelector('.pf-name').value.trim(),
            role: roleEl ? roleEl.value : 'adult',
            pin:r.querySelector('.pf-pin').value.trim()};
  }).filter(p=>p.name);
  $('profState').textContent='saving…';
  const r=await jpost('/api/profiles/save',{sid:SID,profiles:rows});
  $('profState').textContent=r.error?('FAILED: '+r.error):'saved ✓';
  setTimeout(()=>$('profState').textContent='',2500);
  loadProfiles(); refreshWho();
}
function setSticky(on){ localStorage.setItem('anvil-push-sticky', on?'1':'0'); retagPush(); }
async function loadRemind(){
  if($('pushSticky')) $('pushSticky').checked = pushSticky();
  const card=$('remindCard'); if(!card) return;
  if(!(ME && ME.auth_on)){ card.hidden=true; return; }
  card.hidden=false;
  let me; try{ me=await (await fetch('/api/me?_='+Date.now(),{cache:'no-store'})).json(); }catch(e){ return; }
  const profs=me.profiles||[], mine=profs.find(p=>p.name===me.name)||{}, allow=mine.push_allow||[];
  const others=profs.filter(p=>p.name!==me.name);
  $('remindRows').innerHTML = others.length ? others.map(p=>{
    const auto = (mine.role==='minor' && p.role==='adult');   // a parent always reaches their child
    const checked = auto || allow.indexOf(p.name)>=0;
    return '<label style="display:block;margin:3px 0"><input type="checkbox" class="ra" value="'+esc(p.name)+'" '+
      (checked?'checked':'')+(auto?' disabled':'')+'> '+esc(p.name)+
      ' <span class="mut">· '+esc(p.role)+(auto?' (always — your parent)':'')+'</span></label>';
  }).join('') : '<span class="mut">no other family members yet</span>';
}
async function savePushAllow(){
  const allow=[...document.querySelectorAll('#remindRows .ra')].filter(c=>c.checked && !c.disabled).map(c=>c.value);
  $('remindMsg').textContent='saving…';
  const r=await jpost('/api/profile/push_allow',{sid:SID,allow:allow});
  $('remindMsg').textContent = r&&r.ok ? 'saved ✓' : ('failed'+(r&&r.error?': '+r.error:''));
  setTimeout(()=>$('remindMsg').textContent='',2500);
}
async function loadCfg(){
  CFG=await jget('/api/config');
  $('ollama_cloud_url').value=CFG.ollama_cloud_url; $('ollama_local_url').value=CFG.ollama_local_url;
  $('confidence_floor').value=CFG.confidence_floor; $('note_token_budget').value=CFG.note_token_budget;
  if($('daily_cost_cap_usd')) $('daily_cost_cap_usd').value=CFG.daily_cost_cap_usd;
  if($('background_cost_cap_usd')) $('background_cost_cap_usd').value=CFG.background_cost_cap_usd;
  $('request_timeout').value=CFG.request_timeout; $('use_embeddings').value=String(CFG.use_embeddings);
  $('autonomy').value=CFG.autonomy||'trusted';
  $('synthesis_mode').value=CFG.synthesis_mode||'balanced';
  CFG.ladder.forEach(r=>{const el=$('m_'+r.name); if(el) el.value=r.model;});
  $('keyState').textContent=CFG.ollama_api_key_set?'✓ set':'— not set';
  $('searxng_url').value=CFG.searxng_url||'';
  if($('home_address')) $('home_address').value=CFG.home_address||'';
  if($('push_quiet_start')) $('push_quiet_start').value=CFG.push_quiet_start;
  if($('push_quiet_end')) $('push_quiet_end').value=CFG.push_quiet_end;
  const tiers=[]; if(CFG.searxng_url) tiers.push('SearXNG'); if(CFG.tavily_key_set) tiers.push('Tavily'); tiers.push('DuckDuckGo');
  $('tavilyState').textContent=tiers.join(' → ');
}
async function loadModels(){
  const s=await jget('/api/status');
  const dl=$('localmodels'); dl.innerHTML='';
  (s.installed_models||[]).forEach(m=>{const o=document.createElement('option');o.value=m;dl.appendChild(o);});
  $('localState').innerHTML = s.ollama_reachable
    ? '<span class="pill ok">local Ollama reachable</span> '+s.installed_models.length+' models installed'
    : '<span class="pill bad">local Ollama not reachable</span> start it with: ollama serve';
}
async function saveCfg(){
  $('saveState').textContent='saving…';
  const payload={ollama_cloud_url:$('ollama_cloud_url').value,ollama_local_url:$('ollama_local_url').value,
    confidence_floor:parseFloat($('confidence_floor').value),note_token_budget:parseInt($('note_token_budget').value),
    request_timeout:parseInt($('request_timeout').value),use_embeddings:$('use_embeddings').value==='true',
    daily_cost_cap_usd:parseFloat($('daily_cost_cap_usd').value),background_cost_cap_usd:parseFloat($('background_cost_cap_usd').value),
    autonomy:$('autonomy').value,searxng_url:$('searxng_url').value.trim(),synthesis_mode:$('synthesis_mode').value,
    discord_webhook_url:$('discord_webhook_url').value,home_address:$('home_address').value.trim(),
    push_quiet_start:parseInt($('push_quiet_start').value),push_quiet_end:parseInt($('push_quiet_end').value),
    models:{'local-fast':$('m_local-fast').value,'local-reason':$('m_local-reason').value,'cloud-open':$('m_cloud-open').value}};
  const k=$('ollama_api_key').value.trim(); if(k) payload.ollama_api_key=k;
  const tv=$('tavily_api_key').value.trim(); if(tv) payload.tavily_api_key=tv;
  const r=await jpost('/api/config',payload); $('ollama_api_key').value=''; $('tavily_api_key').value='';
  if(r.error){ $('saveState').textContent='SAVE FAILED: '+esc(r.error); return; }
  if(k && !r.ollama_api_key_set){ $('saveState').textContent='SAVE FAILED - could not write .env (folder not writable). Run: python -m anvil doctor'; return; }
  $('saveState').textContent='saved ✓'; setTimeout(()=>$('saveState').textContent='',2500); loadCfg();
}
async function loadStatus(){
  const s=await jget('/api/status');
  $('statusBox').innerHTML=
    '<div>Agent: <b>'+esc(s.agent_name||'Anvil')+'</b> · '+s.traits+' evolved traits</div>'+
    '<div style="margin-top:8px">Ladder: '+s.ladder.map(esc).join(' → ')+'</div>'+
    '<div style="margin-top:8px">Local Ollama: '+(s.ollama_reachable?'<span class="pill ok">reachable</span>':'<span class="pill bad">offline</span>')+
    '  Cloud key: '+(s.cloud_key_set?'<span class="pill ok">set</span>':'<span class="pill bad">not set</span>')+'</div>'+
    '<div style="margin-top:8px">Installed models: '+(s.installed_models.length?s.installed_models.map(esc).join(', '):'<span class="mut">none / unreachable</span>')+'</div>'+
    '<div style="margin-top:8px">Jobs: '+s.jobs+' · Notes: '+s.notes+' · Spent today: $'+s.spent_today+' · Embeddings: '+(s.embeddings?'on':'off')+'</div>';
}
async function loadJobs(){ const r=await jget('/api/jobs'); renderJobs(r.jobs); }
function renderJobs(jobs){
  const el=$('jobsList');
  if(!jobs||!jobs.length){ el.innerHTML='<span class="mut">no jobs yet — add one below</span>'; return; }
  el.innerHTML=jobs.map(j=>'<div style="display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--line)">'+
    '<input type="checkbox" '+(j.enabled?'checked':'')+' onchange="toggleJob(\''+esc(j.name)+'\',this.checked)" style="width:auto">'+
    '<code>'+esc(j.cron)+'</code> <b>'+esc(j.name)+'</b>'+(j.notify?' <span class="pill">'+esc(j.notify)+'</span>':'')+
    '<span class="mut" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc(j.prompt)+'</span>'+
    '<button class="ghost" onclick="runJob(\''+esc(j.name)+'\')">Run</button>'+
    '<button class="ghost" onclick="deleteJob(\''+esc(j.name)+'\')">Delete</button></div>').join('');
}
async function saveJob(){
  $('jState').textContent='saving…';
  const r=await jpost('/api/jobs/save',{name:$('j_name').value,cron:$('j_cron').value,prompt:$('j_prompt').value,
    notify:$('j_notify').value,min_rung:parseInt($('j_rung').value)||0,enabled:true});
  if(r.error){ $('jState').textContent='error: '+r.error; return; }
  $('jState').textContent='saved ✓'; setTimeout(()=>$('jState').textContent='',2000);
  $('j_name').value='';$('j_cron').value='';$('j_prompt').value=''; renderJobs(r.jobs);
}
async function toggleJob(name,enabled){ renderJobs((await jpost('/api/jobs/toggle',{name:name,enabled:enabled})).jobs); }
async function deleteJob(name){ renderJobs((await jpost('/api/jobs/delete',{name:name})).jobs); }
async function runJob(name){ line('sys',null,'running job '+name+'…'); const r=await jpost('/api/jobs/run',{name:name});
  if(r.error){ line('sys',null,'job error: '+r.error); } else { lineMD(AGENT,'**['+name+']** '+r.answer); } }
/* ---- Pulse dashboard ---- */
function feedRows(lines,tagOf){
  if(!lines||!lines.length) return '<div class="row mut">nothing yet</div>';
  return lines.map(l=>{const g=tagOf(l);return '<div class="row"><span class="tag '+g.cls+'">'+esc(g.tag)+'</span>'+esc(g.text)+'</div>';}).join('');
}
function jtail(line){ // strip "- 2026-07-01 17:39 [tag] text"
  const m=(line||'').match(/^-?\s*([\d-]+ [\d:]+)?\s*\[([^\]]+)\]\s*(.*)$/);
  if(m) return {ts:(m[1]||'').slice(11,16),tag:m[2],text:m[3]};
  return {ts:'',tag:'',text:line};
}
async function loadPulse(){
  const p=await jget('/api/pulse?sid='+SID);
  setLive(p.auto_pulse);
  const admin=!!p.is_admin;
  // Development views (self-dev, council backlog + commits) are admin-only.
  ['devSelfCard','devCouncilCard','dreamCard'].forEach(id=>{ const el=$(id); if(el) el.hidden=!admin; });
  $('apState').innerHTML = p.auto_pulse
    ? '<span class="pill ok">running</span>' : '<span class="pill bad">paused</span>';
  $('apLine').innerHTML =
    '<span class="sdot '+(p.auto_pulse?'on':'off')+'"></span> thinks every <b>'+p.heartbeat_min+' min</b>'+
    ' &nbsp;·&nbsp; dreams at <b>'+p.dream_after+'</b> STM items or <b>'+p.dream_max_age_h+'h</b>'+
    (admin?' &nbsp;·&nbsp; self-dev crawl every <b>'+p.selfdev_interval_h+'h</b> (issues &amp; releases work-driven)'+
      (p.self_dev_in_sleep?'':' <span class="pill bad">off</span>'):'');
  const tiles=[['memories',p.notes],['short-term',p.stm_size]];
  if(admin){
    const pct = p.cost_cap ? Math.round(100*p.spent_today/p.cost_cap) : 0;
    const over = p.spent_today>=p.cost_cap, softOver = p.spent_today>=(p.cost_cap_bg||0);
    tiles.push(['self-dev today',p.self_dev_today+'<small>/'+p.self_dev_cap+'</small>'],
      ['cloud budget','<span style="color:'+(over?'var(--bad)':softOver?'var(--acc)':'inherit')+'">$'+p.spent_today+'<small>/$'+p.cost_cap+' · '+pct+'%</small></span>']);
    // per-plane split of today's cloud spend (governance transparency)
    const bp=p.spent_by_plane||{}; const parts=Object.keys(bp).sort((a,b)=>bp[b]-bp[a]).map(k=>k+' $'+bp[k].toFixed(2));
    const gl=$('pulseGov'); if(gl) gl.innerHTML = parts.length
      ? 'cloud spend by plane: '+parts.join(' · ')+(softOver&&!over?' — <span style="color:var(--acc)">background paused (soft cap)</span>':'')+(over?' — <span style="color:var(--bad)">hard cap: local-only</span>':'')
      : 'no cloud spend today';
  }
  tiles.push(['ladder',(p.ladder||[]).length+'<small> rungs</small>']);
  $('pulseTiles').innerHTML=tiles.map(t=>'<div class="tile"><div class="k">'+t[0]+'</div><div class="v">'+t[1]+'</div></div>').join('');
  const who=$('thoughtWho'); if(who) who.textContent = (MEM_ME||(ME&&ME.name)) ? '— '+esc(MEM_ME||ME.name)+"'s activity" : '';
  $('dreamFeed').innerHTML=feedRows((p.dreams||[]).slice().reverse(),l=>{const j=jtail(l);return {cls:'dream',tag:j.ts||'dream',text:j.text};});
  $('selfdevFeed').innerHTML=feedRows((p.selfdev_log||[]).slice().reverse(),l=>{const j=jtail(l);return {cls:'selfdev',tag:j.ts||'dev',text:j.text};});
  $('commitFeed').innerHTML=feedRows(p.commits||[],l=>({cls:'',tag:(l.split(' ')[0]||'').slice(0,7),text:l.split(' ').slice(1).join(' ')}));
  const q=p.queue||{counts:{},items:[]};
  $('qCounts').textContent='— '+Object.entries(q.counts).map(([k,v])=>v+' '+k).join(', ');
  const active=(q.items||[]).filter(i=>i.status!=='done'&&i.status!=='rejected');
  $('queueList').innerHTML = active.length ? active.map(i=>
    '<div class="qrow"><span class="qstat '+esc(i.status)+'">'+esc(i.status||'?')+'</span>'+
    '<span class="mut" style="flex:none">'+esc(i.id)+'</span>'+
    '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc(i.title||'')+'</span></div>').join('')
    : '<span class="mut">backlog clear</span>';
  const stm=(p.thoughts||[]).slice().reverse();
  $('thoughtFeed').innerHTML=feedRows(stm.map(e=>e),()=>({}))==='' ? '' :
    (stm.length? stm.map(e=>'<div class="row"><span class="t">'+esc((e.iso||'').slice(11,16))+'</span><span class="tag '+esc(e.kind||'')+'">'+esc(e.kind||'')+'</span>'+esc(e.text||'')+'</div>').join('')
      : '<div class="row mut">quiet — no recent thoughts</div>');
}
async function thinkNow(){ $('mState').textContent='thinking…'; await jpost('/api/think',{}); $('mState').textContent=''; loadPulse(); }
async function dreamNow(){ $('mState').textContent='dreaming…'; const r=await jpost('/api/dream',{}); $('mState').textContent='dreamed: '+esc(r.summary||JSON.stringify(r)); setTimeout(()=>$('mState').textContent='',7000); loadPulse(); }

/* ---- Memory ---- */
let MEM=[], MEM_ME='';
async function loadMemory(){ const r=await jget('/api/memory?sid='+SID); MEM=r.notes||[]; MEM_ME=r.actor||'';
  $('memCount').textContent='— '+r.count+' notes'+(r.actor?' · '+esc(r.actor)+"'s + household":''); renderMem();
  const sk=r.skills||[]; $('skillCount').textContent='— '+(r.skill_count||0);
  $('skillList').innerHTML = sk.length ? sk.map(s=>
    '<div class="note"><div class="h"><span class="pill">skill</span><b>'+esc(s.name)+'</b>'+
    (s.when?'<span class="dim" style="font-size:12px">when: '+esc(s.when)+'</span>':'')+'</div>'+
    '<div class="mut" style="font-size:12.5px;margin:2px 0 5px">'+esc(s.description)+'</div>'+
    '<div class="b" style="white-space:pre-wrap">'+esc(s.body)+'</div></div>').join('')
    : '<span class="mut">no skills learned yet — Lara saves one when she works out a repeatable procedure</span>';
}
function shareState(n){
  const sw=n.shared_with||[];
  if(!sw.length) return 'private';
  if(sw.indexOf('*')>=0) return 'shared with everyone';
  return 'shared: '+sw.join(', ');
}
function renderMem(){
  const q=($('memSearch').value||'').toLowerCase();
  const rows=MEM.filter(n=>!q || (n.body+' '+n.name+' '+(n.tags||[]).join(' ')).toLowerCase().includes(q));
  $('memList').innerHTML = rows.length ? rows.map(n=>{
    const badge = n.household
      ? '<span class="mut" style="flex:none;font-size:11px">household</span>'
      : n.mine
        ? '<span class="pill" style="background:var(--acc);color:#fff">yours · '+esc(shareState(n))+'</span>'
        : '<span class="mut" style="flex:none;font-size:11px">shared with you</span>';
    const tk=JSON.stringify(n.name).replace(/"/g,'&quot;');
    const share = n.mine
      ? '<button class="ghost" style="padding:2px 8px;font-size:11px;flex:none" onclick="openShare('+tk+')">Share</button>'
        +'<button class="ghost" style="padding:2px 8px;font-size:11px;flex:none" onclick="forgetNote('+tk+')" title="Delete this memory — Lara forgets it">Forget</button>' : '';
    return '<div class="note" data-name="'+esc(n.name)+'"><div class="h"><span class="pill">'+esc(n.type)+'</span>'+badge+
      '<span class="mut" style="flex:none;font-size:12px">sal '+n.salience+'</span>'+
      '<span class="bar"><i style="width:'+Math.round(n.salience*100)+'%"></i></span>'+share+
      (n.tags&&n.tags.length?'<span class="dim" style="font-size:12px">'+esc(n.tags.join(' '))+'</span>':'')+'</div>'+
      '<div class="sharebox" style="display:none"></div>'+
      '<div class="b">'+esc(n.body)+'</div></div>';
  }).join('') : '<span class="mut">no matching notes</span>';
}
function openShare(name){
  const note=MEM.find(n=>n.name===name); if(!note) return;
  const row=document.querySelector('.note[data-name="'+CSS.escape(name)+'"]');
  const box=row&&row.querySelector('.sharebox'); if(!box) return;
  if(box.style.display!=='none'){ box.style.display='none'; box.innerHTML=''; return; }
  const others=((ME&&ME.profiles)||[]).filter(p=>p.name!==MEM_ME);
  const sw=note.shared_with||[], everyone=sw.indexOf('*')>=0;
  box.innerHTML='<div style="margin:6px 0;padding:8px;border:1px solid var(--line2);border-radius:8px">'+
    '<div class="mut" style="font-size:12px;margin-bottom:4px">Share this memory with:</div>'+
    '<label style="display:block;font-size:13px"><input type="checkbox" class="sh-all" '+(everyone?'checked':'')+'> Everyone</label>'+
    (others.length?others.map(p=>'<label style="display:block;font-size:13px"><input type="checkbox" class="sh-p" value="'+esc(p.name)+'" '+((everyone||sw.indexOf(p.name)>=0)?'checked':'')+(everyone?' disabled':'')+'> '+esc(p.name)+'</label>').join('')
      :'<div class="mut" style="font-size:12px">no other profiles yet</div>')+
    '<button class="go" style="margin-top:6px;padding:3px 10px" onclick="saveShare('+JSON.stringify(name).replace(/"/g,'&quot;')+')">Save</button>'+
    ' <span class="mut sh-msg" style="font-size:12px"></span></div>';
  box.style.display='';
  const all=box.querySelector('.sh-all');
  all.onchange=()=>box.querySelectorAll('.sh-p').forEach(c=>{ if(all.checked) c.checked=true; c.disabled=all.checked; });
}
async function forgetNote(name){
  if(!confirm('Forget this memory? Lara will delete it permanently.')) return;
  const r=await jpost('/api/memory/delete',{sid:SID,name:name});
  if(r&&r.ok) loadMemory(); else alert((r&&r.error)||'could not delete');
}
async function saveShare(name){
  const row=document.querySelector('.note[data-name="'+CSS.escape(name)+'"]');
  const box=row&&row.querySelector('.sharebox'); if(!box) return;
  const all=box.querySelector('.sh-all').checked;
  const withList = all ? ['*'] : [...box.querySelectorAll('.sh-p')].filter(c=>c.checked).map(c=>c.value);
  const msg=box.querySelector('.sh-msg'); if(msg) msg.textContent='saving…';
  const r=await jpost('/api/memory/share',{sid:SID,name:name,'with':withList});
  if(r&&r.ok){ if(msg) msg.textContent='saved ✓'; setTimeout(loadMemory,600); }
  else if(msg){ msg.textContent=(r&&r.error)||'failed'; }
}
async function loadShared(){
  let r; try{ r=await jget('/api/memory?sid='+SID); }catch(e){ return; }
  // Shared WITH me = visible notes that aren't mine and aren't household; the
  // owner is who shared them. (Same recall the pipeline injects as my context.)
  const notes=(r.notes||[]).filter(n=>!n.mine && !n.household && n.owner);
  $('sharedCount').textContent='— '+notes.length;
  if(!notes.length){ $('sharedList').innerHTML='<span class="mut">Nothing has been shared with you yet. When a family member shares a memory with you, it appears here — and Lara uses it as context when helping you.</span>'; return; }
  const byOwner={};
  notes.forEach(n=>{ (byOwner[n.owner]=byOwner[n.owner]||[]).push(n); });
  $('sharedList').innerHTML=Object.keys(byOwner).sort().map(owner=>
    '<div style="margin-bottom:16px"><div class="mut" style="font-size:12px;margin-bottom:5px">shared by <b style="color:var(--acc)">'+esc(owner)+'</b> · '+byOwner[owner].length+'</div>'+
    byOwner[owner].map(n=>'<div class="note"><div class="h"><span class="pill">'+esc(n.type)+'</span>'+
      (n.tags&&n.tags.length?'<span class="dim" style="font-size:12px">'+esc(n.tags.join(' '))+'</span>':'')+
      '<span class="mut" style="flex:none;font-size:12px">sal '+n.salience+'</span></div>'+
      '<div class="b">'+esc(n.body)+'</div></div>').join('')+'</div>').join('');
}

function setLive(on){ const d=$('live'); if(d){ d.className='dot'+(on?' live':''); } }
/* ---- Tailscale ---- */
function fmtBytes(n){ if(n<1024) return n+' B'; if(n<1048576) return (n/1024).toFixed(0)+' KB'; return (n/1048576).toFixed(1)+' MB'; }
async function loadBackups(){
  let r; try{ r=await jget('/api/backups'); }catch(e){ return; }
  const b=r.backups||[]; if($('bkState')) $('bkState').textContent='— '+b.length+' kept';
  const el=$('bkList'); if(!el) return;
  el.innerHTML = b.length ? b.map(x=>'<div class="row"><span class="t">'+esc(x.name.replace('anvil-backup-','').replace('.zip',''))+'</span>'+fmtBytes(x.size)+'</div>').join('')
    : '<div class="row mut">no backups yet — one is taken automatically each day, or tap “Back up now”.</div>';
}
async function backupNow(){
  $('bkMsg').textContent='backing up…';
  const r=await jpost('/api/backup',{sid:SID});
  $('bkMsg').textContent = r&&r.ok ? ('saved '+esc(r.name)) : ('failed'+(r&&r.error?': '+r.error:''));
  setTimeout(()=>$('bkMsg').textContent='',4000);
  loadBackups();
}
async function loadTailscale(){
  const b=$('tsBody'),act=$('tsActions'),st=$('tsState'); if(!b) return; $('tsMsg').textContent='';
  let s; try{ s=await jget('/api/tailscale'); }catch(e){ s={installed:false}; }
  if(!s.installed){ st.innerHTML='<span class="pill bad">not installed</span>';
    b.innerHTML='Install Tailscale on this machine (tailscale.com/download), then reload.'; act.innerHTML=''; return; }
  if(!s.running){ st.innerHTML='<span class="pill bad">signed out</span>';
    b.innerHTML='Tailscale is installed but not connected yet.';
    act.innerHTML='<button class="go" style="margin:0" onclick="tsConnect()">Connect / sign in</button>'; return; }
  st.innerHTML='<span class="pill ok">connected</span>';
  const boundHere=(s.bind_host===s.ip);
  const peers=(s.peers||[]).map(p=>'<div class="entity"><span>'+esc(p.name||p.host||'?')+'</span>'+
    '<span class="mut" style="font-size:11px">'+esc(p.ip||'')+(p.os?' · '+esc(p.os):'')+'</span>'+
    '<span class="st '+(p.online?'on':'off')+'">'+(p.online?'online':'offline')+'</span></div>').join('');
  b.innerHTML='This node: <b>'+esc(s.name||'')+'</b> <code>'+esc(s.ip||'')+'</code> on tailnet <b>'+esc(s.tailnet||'')+'</b><br>'+
    'ANVIL is bound to <code>'+esc(s.bind_host||'127.0.0.1')+'</code> '+(boundHere?'<span class="pill ok">reachable over tailnet</span>':'<span class="mut">(localhost only)</span>')+
    '<div style="margin-top:12px">'+peers+'</div>';
  if(boundHere) act.innerHTML='<button class="ghost" onclick="tsBind(\'127.0.0.1\')">Unbind (localhost only)</button>'+
    ' <span class="mut" style="margin-left:8px">Reach Lara at <code>'+esc(s.ip)+':'+s.server_port+'</code></span>';
  else act.innerHTML='<button class="go" style="margin:0" onclick="tsBind(\''+esc(s.ip)+'\')">Bind ANVIL to '+esc(s.ip)+'</button>';
}
async function tsConnect(){ $('tsMsg').textContent='starting sign-in…'; let r; try{ r=await jpost('/api/tailscale/up',{}); }catch(e){ r={error:String(e)}; }
  if(r.auth_url){ $('tsMsg').innerHTML='Sign in here: <a href="'+esc(r.auth_url)+'" target="_blank" rel="noopener">'+esc(r.auth_url)+'</a>'; try{ window.open(r.auth_url,'_blank'); }catch(e){} setTimeout(loadTailscale,4000); }
  else if(r.ok){ $('tsMsg').textContent='connected.'; setTimeout(loadTailscale,1200); }
  else { $('tsMsg').textContent='error: '+(r.error||'could not start sign-in'); } }
async function tsBind(host){ $('tsMsg').textContent='saving…'; const r=await jpost('/api/tailscale/bind',{host:host});
  if(r.error){ $('tsMsg').textContent='error: '+r.error; return; }
  $('tsMsg').textContent=r.note||('bound to '+r.bind_host); loadTailscale(); }
/* ---- attachments: photos as-is (downscaled), videos as sampled frames ---- */
let ATTACH=[];   // [{b64 (no dataURL header), thumb, label}]
function attachRender(){
  const row=$('attachRow'); row.innerHTML=''; row.hidden=!ATTACH.length;
  ATTACH.forEach((a,i)=>{
    const d=document.createElement('div'); d.className='chip';
    const im=document.createElement('img'); im.src=a.thumb; d.appendChild(im);
    if(a.label){ const n=document.createElement('span'); n.className='n'; n.textContent=a.label; d.appendChild(n); }
    const x=document.createElement('button'); x.textContent='✕';
    x.onclick=()=>{ ATTACH.splice(i,1); attachRender(); };
    d.appendChild(x); row.appendChild(d);
  });
}
function canvasJpeg(source,w,h){
  const MAX=1344, s=Math.min(1, MAX/Math.max(w,h));
  const c=document.createElement('canvas'); c.width=Math.round(w*s); c.height=Math.round(h*s);
  c.getContext('2d').drawImage(source,0,0,c.width,c.height);
  return c.toDataURL('image/jpeg',0.85);
}
function addImageFile(file,label){
  return new Promise(res=>{
    const im=new Image();
    im.onload=()=>{ const url=canvasJpeg(im,im.naturalWidth,im.naturalHeight);
      ATTACH.push({b64:url.split(',')[1], thumb:url, label:label||''});
      URL.revokeObjectURL(im.src); attachRender(); res(); };
    im.onerror=()=>res();
    im.src=URL.createObjectURL(file);
  });
}
function addVideoFile(file){
  return new Promise(res=>{
    const v=document.createElement('video');
    v.muted=true; v.preload='auto'; v.src=URL.createObjectURL(file);
    v.onloadedmetadata=async ()=>{
      const dur=v.duration||0, nFrames=Math.min(6, Math.max(2, Math.round(dur/5)));
      for(let i=0;i<nFrames && ATTACH.length<6;i++){
        const t=dur*(i+0.5)/nFrames;
        await new Promise(ok=>{ v.onseeked=ok; v.currentTime=t; });
        const url=canvasJpeg(v, v.videoWidth, v.videoHeight);
        ATTACH.push({b64:url.split(',')[1], thumb:url,
                     label:file.name.slice(0,14)+' @'+Math.round(t)+'s'});
      }
      URL.revokeObjectURL(v.src); attachRender(); res();
    };
    v.onerror=()=>{ msgSys('could not read that video'); res(); };
  });
}
async function filesPicked(files){
  for(const f of files){
    if(ATTACH.length>=6){ msgSys('attachment limit: 6 images (video frames count)'); break; }
    if(f.type.startsWith('video/')) await addVideoFile(f);
    else if(f.type.startsWith('image/')) await addImageFile(f);
  }
}
document.addEventListener('paste', e=>{
  const items=[...(e.clipboardData?.items||[])].filter(i=>i.type.startsWith('image/'));
  if(items.length){ e.preventDefault(); filesPicked(items.map(i=>i.getAsFile()).filter(Boolean)); }
});
document.addEventListener('dragover', e=>{ e.preventDefault(); });
document.addEventListener('drop', e=>{ e.preventDefault();
  if(e.dataTransfer?.files?.length) filesPicked(e.dataTransfer.files); });

/* ---- Chats drawer: bounce between conversations (each keeps its context) ---- */
function ago(ts){ const s=(Date.now()/1000)-ts; if(s<90) return 'now';
  if(s<3600) return Math.round(s/60)+'m'; if(s<86400) return Math.round(s/3600)+'h';
  return Math.round(s/86400)+'d'; }
function isDesktop(){ try{ return matchMedia('(min-width: 900px)').matches; }catch(e){ return false; } }
function chatsOpen(){ return !$('drawer').hidden; }
function applyChats(open){
  $('drawer').hidden=!open;
  $('drawerBg').hidden=!open || isDesktop();      // no backdrop when docked
  document.body.classList.toggle('dopen', open && isDesktop());
  localStorage.setItem('anvil-chats-open', open?'1':'0');
  // The one fixed button morphs: hamburger when closed, X when the sidebar
  // has formed around it.
  const mb=$('menubtn');
  if(mb){ mb.textContent = open ? '✕' : '☰';
          mb.title = open ? 'Close chats' : 'Chats'; }
  if(open){ chatsRefresh(); }
}
function toggleChats(){ if(!chatsOpen()) $('chatSearch').value=''; applyChats(!chatsOpen()); }
function closeChatsIfMobile(){ if(!isDesktop()) applyChats(false); }
function drawerNewChat(){ newChat(); closeChatsIfMobile(); chatsRefresh(); }
/* Default: desktop shows the docked sidebar (unless the user collapsed it);
   phone starts with the full screen as chat. */
(function(){
  const open = isDesktop() && localStorage.getItem('anvil-chats-open')!=='0';
  if(open) setTimeout(()=>applyChats(true), 0);
  try{ matchMedia('(min-width: 900px)').addEventListener('change', ()=>applyChats(chatsOpen())); }catch(e){}
})();
async function chatsRefresh(){
  const q=$('chatSearch').value.trim();
  let r; try{ r=await jget('/api/conversations'+(q?'?q='+encodeURIComponent(q):'')); }catch(e){ r={chats:[]}; }
  const chats=(r&&r.chats)||[];
  const box=$('chatList'); box.innerHTML='';
  if(!chats.length){ box.innerHTML='<div class="mut" style="padding:12px 10px">'+(q?'no chats mention that':'no chats yet')+'</div>'; return; }
  let lastSec='';
  chats.forEach(c=>{
    if(!q){ const sec=c.pinned?'pinned':'recent';
      if(sec!==lastSec){ lastSec=sec; const h=document.createElement('div'); h.className='chatsec'; h.textContent=sec; box.appendChild(h); } }
    const d=document.createElement('div'); d.className='chatrow'+(c.sid===SID?' on':'');
    const title=document.createElement('span'); title.className='t';
    title.textContent=c.title||'New chat';
    if(c.snippet){ const s=document.createElement('span'); s.className='snip'; s.textContent=c.snippet; title.appendChild(s); }
    d.appendChild(title);
    if(c.pinned&&!q){ const p=document.createElement('span'); p.className='pinmark'; p.textContent='pinned'; d.appendChild(p); }
    const when=document.createElement('span'); when.className='when'; when.textContent=ago(c.ts); d.appendChild(when);
    const act=document.createElement('span'); act.className='act';
    act.innerHTML='<button title="Rename">✎</button><button title="'+(c.pinned?'Unpin':'Pin')+'">'+(c.pinned?'⊖':'⊕')+'</button><button title="Delete">🗑</button>';
    const [rn,pn,del]=act.querySelectorAll('button');
    rn.onclick=async ev=>{ ev.stopPropagation();
      const t=prompt('Rename chat:', c.title||''); if(t===null) return;
      await jpost('/api/conversation/rename',{sid:c.sid,title:t}); chatsRefresh(); };
    pn.onclick=async ev=>{ ev.stopPropagation();
      await jpost('/api/conversation/pin',{sid:c.sid,pinned:!c.pinned}); chatsRefresh(); };
    del.onclick=async ev=>{ ev.stopPropagation();
      if(!confirm('Delete this chat? Lara forgets its transcript.')) return;
      await jpost('/api/conversation/delete',{sid:c.sid});
      if(c.sid===SID){ newChat(); } chatsRefresh(); };
    d.appendChild(act);
    d.onclick=()=>{ switchChat(c.sid); };
    box.appendChild(d);
  });
}
function switchChat(sid){
  if(sid!==SID){ SID=sid; localStorage.setItem('anvil-sid',SID);
    $('msgs').innerHTML=''; greeted=false; restoreChat(); }
  closeChatsIfMobile(); chatsRefresh(); $('task').focus();
}

/* ---- device geolocation (phones): Lara answers weather/location questions
   for wherever you actually are. Only auto-requested on mobile / the installed
   app — the desktop at home doesn't need it. Position is cached and refreshed
   when the app comes back to the foreground. ---- */
let GEO=null;
function isMobileish(){
  try{ return matchMedia('(display-mode: standalone)').matches ||
              /iPhone|iPad|Android/i.test(navigator.userAgent); }catch(e){ return false; }
}
function geoRefresh(){
  if(!('geolocation' in navigator) || !window.isSecureContext || !isMobileish()) return;
  try{ navigator.geolocation.getCurrentPosition(function(p){
      GEO={lat:+p.coords.latitude.toFixed(4), lon:+p.coords.longitude.toFixed(4), ts:Date.now()};
    }, function(){}, {maximumAge:600000, timeout:8000, enableHighAccuracy:false}); }catch(e){}
}
function geoNow(){ return (GEO && (Date.now()-GEO.ts) < 15*60*1000) ? {lat:GEO.lat, lon:GEO.lon} : null; }
geoRefresh();
document.addEventListener('visibilitychange', function(){ if(!document.hidden) geoRefresh(); });

/* ---- PWA install + Web Push (auto-ask once on first visit, then remember) ---- */
let SWREG=null, PUSHCFG=null;
function b64ToU8(b64){ const pad='='.repeat((4-b64.length%4)%4); const s=(b64+pad).replace(/-/g,'+').replace(/_/g,'/');
  const raw=atob(s); const out=new Uint8Array(raw.length); for(let i=0;i<raw.length;i++) out[i]=raw.charCodeAt(i); return out; }
async function initPush(){
  if(!('serviceWorker' in navigator)) return;
  try{ SWREG=await navigator.serviceWorker.register('/sw.js'); }catch(e){ return; }
  try{ PUSHCFG=await jget('/api/push/config'); }catch(e){ PUSHCFG=null; }
  maybeAskNotify();
}
async function currentSub(){ try{ return SWREG?await SWREG.pushManager.getSubscription():null; }catch(e){ return null; } }
function pushSticky(){ return localStorage.getItem('anvil-push-sticky')!=='0'; }  // default: keep logged in
async function doSubscribe(test){
  try{
    const sub=await SWREG.pushManager.subscribe({userVisibleOnly:true,
      applicationServerKey:b64ToU8(PUSHCFG.public_key)});
    const r=await jpost('/api/push/subscribe',{subscription:sub.toJSON(),sid:SID,sticky:pushSticky()});
    if(r.ok && test){ try{ await jpost('/api/push/test',{sid:SID}); }catch(e){} }
    return !!(r&&r.ok);
  }catch(e){ return false; }
}
// Re-tag this device's existing subscription with the current profile (so pushes
// route to the right person after login) — cheap, runs on load when subscribed.
async function retagPush(){
  try{ const sub=await currentSub();
    if(sub) await jpost('/api/push/subscribe',{subscription:sub.toJSON(),sid:SID,sticky:pushSticky()});
  }catch(e){}
}
// First visit: proactively ask. iOS requires a user gesture to raise the OS
// permission dialog, so we show a one-tap in-app prompt whose Enable button IS
// that gesture. The answer is remembered in localStorage so we never nag again.
async function maybeAskNotify(){
  const canPush = PUSHCFG && PUSHCFG.enabled && ('PushManager' in window) &&
                  ('Notification' in window) && window.isSecureContext && SWREG;
  if(!canPush) return;
  if(await currentSub()) return;                          // already subscribed
  const perm=Notification.permission;
  if(perm==='granted'){ doSubscribe(false); return; }     // permission stands — resubscribe silently
  if(perm==='denied') return;                             // OS-blocked; can't re-ask
  if(localStorage.getItem('anvil-notify-asked')) return;  // already answered — respect it
  showNotifyPrompt();
}
function showNotifyPrompt(){
  if(document.getElementById('notifask')) return;
  const chat=$('chat'); if(!chat) return;
  const d=document.createElement('div'); d.id='notifask'; d.className='notifask';
  d.innerHTML='<span class="na-t">Let Lara notify you when she answers or needs you?</span>'+
    '<span class="na-b"><button class="go" onclick="notifyAllow()">Enable</button>'+
    '<button class="ghost" onclick="notifyDismiss()">Not now</button></span>';
  chat.insertBefore(d, chat.firstChild);
}
function removeNotifyPrompt(){ const d=document.getElementById('notifask'); if(d) d.remove(); }
async function notifyAllow(){
  let perm=Notification.permission;                        // this click is the required gesture
  if(perm!=='granted'){ try{ perm=await Notification.requestPermission(); }catch(e){} }
  localStorage.setItem('anvil-notify-asked','1');
  removeNotifyPrompt();
  if(perm==='granted'){ if(await doSubscribe(true)) msgSys('notifications on — Lara will ping this device'); }
}
function notifyDismiss(){ localStorage.setItem('anvil-notify-asked','1'); removeNotifyPrompt(); }
/* Setup-tab Notifications card: the permanent switch (the chat banner only asks once). */
async function npRefresh(){
  const st=$('npState'); if(!st) return;
  if(!PUSHCFG||!PUSHCFG.enabled||!window.isSecureContext||!('PushManager' in window)){
    st.innerHTML='<span class="pill bad">unavailable here</span>';
    $('npMsg').textContent = window.isSecureContext ? 'push is not enabled on the server'
      : 'needs the HTTPS address (open via your tailnet name, and on iPhone use the installed app)';
    return;
  }
  const sub=await currentSub();
  const on=!!sub && Notification.permission==='granted';
  st.innerHTML = on ? '<span class="pill ok">on — this device</span>'
                    : '<span class="pill">off on this device</span>';
}
async function npEnable(){
  localStorage.removeItem('anvil-notify-asked');   // permanent switch overrides "Not now"
  await notifyAllow(); npRefresh();
}
async function npDisable(){
  const sub=await currentSub();
  if(sub){ try{ await sub.unsubscribe(); }catch(e){}
    try{ await jpost('/api/push/unsubscribe',{endpoint:sub.endpoint}); }catch(e){} }
  localStorage.setItem('anvil-notify-asked','1');  // don't re-prompt on next visit
  $('npMsg').textContent='notifications off on this device'; npRefresh();
}
async function npTest(){
  const r=await jpost('/api/push/test',{});
  $('npMsg').textContent = r.sent ? 'test sent — check your devices'
    : 'nothing sent ('+(r.subs||0)+' subscription(s); reason: '+(r.reason||'device not subscribed or push rejected')+')';
}
initPush().then?.(npRefresh);
setTimeout(npRefresh, 1200);
checkAuth().then(ok=>{ if(ok){
  loadPersona(); loadCfg().then(loadModels); jget('/api/pulse').then(p=>setLive(p.auto_pulse)).catch(()=>{});
  refreshWho(); loadProfiles();
  // The Shared tab only makes sense with a family (more than one profile).
  if($('sharedTab')) $('sharedTab').hidden = !(ME && ME.auth_on);
  // Server settings are ADMIN-only: hide the tab once login is on and you're
  // not the admin. (Single-user / no login: you ARE the admin, so it shows.)
  if($('serverTab')) $('serverTab').hidden = !!(ME && ME.auth_on && !ME.admin);
  setTimeout(retagPush, 1500);   // bind this device's push subscription to the profile
  pollApprovals(); setInterval(pollApprovals, 8000);   // adult devices watch the family queue
}});

// PWA self-update: an installed iOS PWA holds its old JS and never re-fetches
// the shell on resume, so new UI (like the live thinking trace) never showed.
// Poll the server's build number; when it changes, reload to the fresh app.
const MY_BUILD='__UI_BUILD__';
let updating=false;
async function checkBuild(){
  if(updating) return;
  try{
    const v=await jget('/api/version');
    if(v && String(v.build)!==MY_BUILD && !INFLIGHTS.length){
      updating=true; location.reload(true);
    }
  }catch(e){}
}
document.addEventListener('visibilitychange',()=>{ if(document.visibilityState==='visible') checkBuild(); });
setInterval(checkBuild, 90000);   // also catch a long-open session
</script>
</body></html>"""
