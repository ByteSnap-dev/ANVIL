"""The hive — Lara's council of specialist drones.

One persistent mind (Lara: memory, dreams, senses, persona, voice) sits out
front. Behind her is a bench of EXPERTS, each a disposable drone bound to the
model that is genuinely best at its domain and a scoped set of read-only tools.

* Lara routes each delegated sub-task to the CLOSEST expert (cheap path).
* If that lead expert comes back weak or failed, the hive convenes a COUNCIL:
  the lead plus the 1-2 most relevant backups take the task in parallel and a
  neutral aggregator synthesises their reports (the "lead the charge with
  backup" escalation).
* Workers are ephemeral, carry NO persona, write NO long-term memory (Lara
  distils the result through her own scribe/dream path), and get a NARROWED
  read-only tool allowlist — danger tools are UNREACHABLE, not merely gated.
* The HOME expert is privacy-pinned to the LOCAL model: house state never
  ships to a cloud model. The rest run on their cloud rung (Ollama Max), so
  the council works in parallel while Lara keeps the local GPU for the family.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

# Read-only senses only. No shell, no ssh, no write_file, no ha_service, no
# schedule, no save_skill — a drone observes and reports, never acts or learns.
SAFE_TOOLS = frozenset({
    "search", "web_fetch", "weather", "ha_list", "ha_get", "ha_search",
    "search_chats", "list_skills", "list_jobs", "read_file", "list_dir",
    "tailscale_status",
})

# House-touching cues → the HOME expert, which is privacy-pinned local.
HOUSE_CUES = (
    "home assistant", "ha_", "the house", "my house", "who is home", "who's home",
    "presence", "room", "light", "lights", "sensor", "thermostat", "camera",
    "door", "lock", "garage", "media player", "device in", "entity", "living room",
    "kitchen", "bedroom", "is anyone", "at home", "the home",
)

# ---------------------------------------------------------------------------
# The specialist bench. Each expert = a domain persona + the model genuinely
# best at it (by rung name; falls back to cloud-open then local if absent) +
# a scoped read-only toolset (always a subset of SAFE_TOOLS). ``local`` pins
# the expert to the local model regardless of rung (privacy).
# ---------------------------------------------------------------------------
SPECIALISTS: Dict[str, dict] = {
    "home": {
        "blurb": ("the household expert. You know this family's home through "
                  "Home Assistant. Answer from live entity state via your tools; "
                  "never guess about the house."),
        "rung": None, "local": True,
        "cues": HOUSE_CUES,
        "tools": {"ha_list", "ha_get", "ha_search", "weather", "search_chats",
                  "tailscale_status", "list_jobs"},
    },
    "code": {
        "blurb": ("an expert software engineer. Be precise about languages, "
                  "APIs, errors and fixes; read files with your tools before "
                  "asserting how this codebase works."),
        "rung": "cloud-heavy",
        "cues": ("code", "coding", "function", "bug", "python", "javascript",
                 "typescript", "script", "refactor", "stack trace", "traceback",
                 "api", "regex", "git ", "compile", "error:", "exception",
                 "install", "dependency", "docker", "config file", "syntax"),
        "tools": {"read_file", "list_dir", "search", "web_fetch"},
    },
    "logic": {
        "blurb": ("an expert in math, logic, probability and step-by-step "
                  "reasoning. Show the reasoning that leads to the number; "
                  "watch for the intuitive-but-wrong answer."),
        "rung": "cloud-logic",
        "cues": ("calculate", "how much", "how many", "math", "equation",
                 "probability", "odds", "percent", "optimi", "algorithm",
                 "logic", "prove", "statistic", "estimate", "budget", "cost of",
                 "convert", "ratio", "average"),
        "tools": {"search", "web_fetch"},
    },
    "research": {          # the generalist / default expert
        "blurb": ("an expert researcher. Find primary sources with your tools, "
                  "prefer recent ones, and report the key facts WITH the urls "
                  "they came from. Say plainly when the sources don't answer it."),
        "rung": "cloud-open",
        "cues": (),
        "tools": SAFE_TOOLS,
        "default": True,
    },
}

# Legacy role names still accepted by delegate(role=...) — map onto experts.
_ROLE_ALIASES = {"worker": None, "researcher": "research", "checker": "research",
                 "summarizer": "research"}

WORKER_SYS = (
    "You are {name}, a specialist agent in ANVIL's hive: {blurb}\n"
    "One focused task, done well, then you're gone. Use ONLY the tools you were "
    "given (read-only senses — you cannot act on anything). Be factual; if a "
    "tool fails or you can't find the answer, SAY SO plainly instead of "
    "guessing. Reply with your findings in compact plain prose — it goes to the "
    "coordinating agent, not a human, so skip pleasantries."
)

# A lead expert's result is 'weak' (→ convene the council) if it failed or the
# text hedges. Conservative: only genuine uncertainty escalates.
_UNCERTAIN = (
    "i'm not sure", "im not sure", "not sure", "couldn't find", "could not find",
    "unable to", "no results", "uncertain", "don't have enough", "insufficient",
    "i don't know", "i do not know", "unclear", "not certain", "can't determine",
    "cannot determine", "no information", "failed", "i couldn't", "i could not",
)


def _rung_idx(cfg, name: Optional[str]) -> int:
    """Ladder index for a worker rung. The role table's Ollama-era names
    (cloud-heavy/cloud-logic/cloud-open) only bind when they exist EXACTLY in
    the ladder — otherwise the operator's hive_worker_rung wins. Without this,
    the legacy tier aliases quietly routed every drone to Sonnet/Opus while
    hive_worker_rung said Haiku (Lara's issue #100)."""
    exact = {r.name for r in (cfg.ladder or [])}
    if name and name in exact:
        return cfg.rung_by_name(name)
    hw = getattr(cfg, "hive_worker_rung", "") or ""
    if hw in exact:
        return cfg.rung_by_name(hw)
    if not name:
        return 0
    idx = cfg.rung_by_name(name)          # legacy alias, last resort
    if idx is not None:
        return idx
    alt = cfg.rung_by_name("cloud-open")
    return alt if alt is not None else 0


# ---------------------------------------------------------------------------
# Local lane: Ollama batches concurrent requests to the one loaded model, so a
# couple of local drones can run at once. The HOME expert shares these slots.
# ---------------------------------------------------------------------------
_LANE_LOCK = threading.Lock()
_LOCAL_SEM: Optional[threading.BoundedSemaphore] = None


def _local_sem(cfg) -> Optional[threading.BoundedSemaphore]:
    global _LOCAL_SEM
    slots = int(getattr(cfg, "hive_local_slots", 2))
    if slots <= 0:
        return None
    with _LANE_LOCK:
        if _LOCAL_SEM is None:
            _LOCAL_SEM = threading.BoundedSemaphore(slots)
        return _LOCAL_SEM


def pick_specialist(task: str) -> str:
    """Route a task to the closest expert by cue score; default = research."""
    low = (task or "").lower()
    if any(c in low for c in HOUSE_CUES):
        return "home"
    best, best_score = "research", 0
    for name, spec in SPECIALISTS.items():
        score = sum(1 for c in spec["cues"] if c in low)
        if score > best_score:
            best, best_score = name, score
    return best


def _backups_for(lead: str, task: str, n: int = 2) -> List[str]:
    """The n most relevant OTHER experts to back the lead up. Always includes
    research (the generalist) unless it's the lead; fills by cue relevance."""
    low = (task or "").lower()
    others = [x for x in SPECIALISTS if x != lead]
    scored = sorted(others, key=lambda x: (
        sum(1 for c in SPECIALISTS[x]["cues"] if c in low),
        x == "research"), reverse=True)
    return scored[:max(1, n)]


def _run_expert(cfg, task: str, name: str, deadline: float = None) -> Dict:
    """Run one expert drone. Never raises — a dead drone is a report.
    ``deadline`` (monotonic) propagates the delegation's shared budget all the
    way into the agent loop's cancel check (review 2.4): before, a timeout only
    changed the COLLECTOR's error message while the drone ran on, holding GPU
    slots for work nobody would read."""
    import time as _time
    from .pipeline import Pipeline
    spec = SPECIALISTS.get(name) or SPECIALISTS["research"]
    is_local = bool(spec.get("local"))
    min_rung = 0 if is_local else _rung_idx(cfg, spec.get("rung"))
    allowed = frozenset(spec["tools"]) & SAFE_TOOLS
    sys_prompt = WORKER_SYS.format(name=name, blurb=spec["blurb"])
    sem = _local_sem(cfg) if is_local else None
    held = bool(sem and sem.acquire(timeout=60))
    lane = "local" if (is_local or min_rung == 0) else "cloud"
    if sem is not None and not held:
        # The lane gate is BINDING now: an un-acquired slot used to run the
        # drone anyway, piling local models onto the single GPU alongside the
        # foreground chat — exactly what the lane exists to prevent. (Privacy-
        # pinned experts never fall through to cloud; busy is just busy.)
        return {"task": task, "specialist": name, "lane": lane, "ok": False,
                "answer": "(local lane busy — re-delegate this task or answer "
                          "from other findings)"}
    cancel = ((lambda: _time.monotonic() > deadline) if deadline else None)
    try:
        res = Pipeline(cfg, plane="hive")._agent_loop(
            [{"role": "user", "content": task}],
            min_rung=min_rung, system_prompt=sys_prompt,
            allowed=allowed, scribe=False, cancel=cancel,
            think=False if lane == "local" else None)
        ok = res.get("status") == "done" and bool((res.get("answer") or "").strip())
        return {"task": task, "specialist": name, "lane": lane, "ok": ok,
                "answer": (res.get("answer") or "(no answer)").strip()}
    except Exception as exc:
        return {"task": task, "specialist": name, "lane": lane, "ok": False,
                "answer": f"(worker failed: {type(exc).__name__}: {exc})"}
    finally:
        if held and sem:
            try:
                sem.release()
            except ValueError:
                pass


def _is_weak(result: Dict) -> bool:
    if not result.get("ok"):
        return True
    low = (result.get("answer") or "").lower()
    return len(low) < 24 or any(m in low for m in _UNCERTAIN)


def _synthesize(cfg, task: str, lead: Dict, panel: List[Dict]) -> str:
    """Neutral aggregator (glm) merges the lead + backups into one answer,
    weighing agreement and flagging conflicts. Plain completion, no tools."""
    from .pipeline import Pipeline
    reports = "\n\n".join(
        f"[{r['specialist']} expert]:\n{r['answer'][:1400]}"
        for r in [lead] + panel)
    sys = ("You are the hive's aggregator. Several specialist agents answered "
           "the same question. Merge their reports into ONE correct, complete "
           "answer for the coordinator: keep what they agree on, resolve or "
           "flag conflicts, drop hedging. Do not invent facts none of them "
           "gave. Compact prose.")
    try:
        r = Pipeline(cfg, plane="hive").router.complete(
            [{"role": "user", "content": f"QUESTION:\n{task}\n\nSPECIALIST "
              f"REPORTS:\n{reports}"}],
            system=sys, min_rung=_rung_idx(cfg, "cloud-open"),
            max_tokens=900, think=False)
        return r.completion.text.strip() or lead["answer"]
    except Exception:
        return lead["answer"]


def run_worker(cfg, task: str, role: str = "worker",
               deadline: float = None) -> Dict:
    """Route a task to its lead expert; if the lead comes back weak, convene a
    council (lead + relevant backups in parallel) and synthesise. ``role`` may
    force a specialist by name, else the task is auto-routed. Never raises."""
    import time as _time
    forced = role if role in SPECIALISTS else _ROLE_ALIASES.get(role, "MISS")
    lead_name = forced if forced in SPECIALISTS else pick_specialist(task)
    lead = _run_expert(cfg, task, lead_name, deadline=deadline)

    if not getattr(cfg, "hive_council", True) or not _is_weak(lead):
        return lead
    # A weak lead near the deadline just returns weak, flagged — convening a
    # 3-drone council the coordinator can never wait for wastes every lane.
    if deadline and _time.monotonic() > deadline - 30:
        return {**lead, "note": "weak answer; no time left for a council"}

    # Escalate: the lead leads the charge with backup from other experts.
    backups = _backups_for(lead_name, task)
    with ThreadPoolExecutor(max_workers=len(backups),
                            thread_name_prefix="council") as pool:
        panel = [f.result() for f in
                 [pool.submit(_run_expert, cfg, task, b, deadline) for b in backups]]
    panel = [p for p in panel if p.get("ok")]
    if not panel:                       # nobody could help — return lead as-is
        return {**lead, "council": [lead_name] + backups}
    answer = _synthesize(cfg, task, lead, panel)
    return {"task": task, "specialist": lead_name,
            "council": [lead_name] + [p["specialist"] for p in panel],
            "lane": lead["lane"], "ok": True, "answer": answer}


def delegate(cfg, tasks: List[str], role: str = "worker",
             run_fn=run_worker) -> List[Dict]:
    """Fan tasks out to parallel expert drones; results return in task order.
    Each task auto-routes to its lead expert and may escalate to a council.
    ``run_fn`` is injectable for offline tests."""
    cap = max(1, int(getattr(cfg, "hive_max_workers", 4)))
    raw = [str(t).strip()[:600] for t in tasks if str(t).strip()]
    dropped = len(raw) - cap
    tasks = raw[:cap]
    if not tasks:
        return []
    import time as _time
    from concurrent.futures import wait as _fwait
    budget = float(getattr(cfg, "ask_time_budget_s", 240)) + 90
    # ONE shared deadline for the whole delegation (review 2.4). The old code
    # applied the budget PER future in a sequential collect loop (worst case
    # N x budget) and the pool's context-manager exit then blocked until every
    # hung drone finished anyway — the timeout only changed the error message.
    deadline = _time.monotonic() + budget
    pool = ThreadPoolExecutor(max_workers=cap, thread_name_prefix="hive")
    futures = [pool.submit(run_fn, cfg, t, role, deadline=deadline) for t in tasks]
    _fwait(futures, timeout=max(1.0, deadline - _time.monotonic()))
    out = []
    for t, f in zip(tasks, futures):
        if f.done():
            try:
                out.append(f.result())
            except Exception as exc:
                out.append({"task": t, "specialist": "?", "ok": False,
                            "answer": f"(worker crashed: {exc})"})
        else:
            out.append({"task": t, "specialist": "?", "ok": False,
                        "answer": "(worker hit the delegation deadline)"})
    # Return NOW with partial results; stragglers see the deadline via their
    # cancel check and wind down on their own — nobody waits for them.
    pool.shutdown(wait=False, cancel_futures=True)
    if dropped > 0 and out:
        out[-1]["answer"] += (
            f"\nNOTE: {dropped} task(s) beyond hive_max_workers={cap} "
            "were not run.")
    return out
