"""Configuration loading for ANVIL.

Loads an ``anvil.toml`` (or ``anvil.json``) file and layers environment
variable overrides on top. No third-party YAML/TOML dependency is required:
TOML is read with the stdlib ``tomllib`` (Python 3.11+) when available, then a
``tomli`` backport, and finally a tiny built-in reader for ANVIL's own schema.

Secrets (API keys, webhook URLs) are *never* put in the config file; they are
read from the environment so the config can be committed to git safely.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Rung:
    """One rung of the escalation ladder."""

    name: str
    provider: str            # "ollama_local" | "ollama_cloud"
    model: str
    cost_in: float = 0.0
    cost_out: float = 0.0
    cache_read: float = 0.0
    max_context: int = 32_000
    # Force reasoning ('thinking') ON for this rung, regardless of the per-task
    # think decision. Lets a rung be "the same local model, but reasoning" — e.g.
    # local-reason = qwen3.6 with think=true: escalating from local-fast keeps the
    # model RESIDENT (no VRAM swap to a different model) and just turns thinking on.
    think: bool = False

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama_local"

    @property
    def is_paid(self) -> bool:
        return self.cost_in > 0 or self.cost_out > 0


@dataclass
class Config:
    ladder: List[Rung]
    memory_dir: Path = Path("memory")
    jobs_dir: Path = Path("jobs")
    ledger_path: Path = Path("ledger.jsonl")
    ollama_local_url: str = "http://localhost:11434"
    ollama_cloud_url: str = "https://ollama.com"
    embed_model: str = "nomic-embed-text"
    use_embeddings: bool = True
    confidence_floor: float = 0.6
    note_token_budget: int = 8_000
    recall_max_notes: int = 10         # non-profile notes injected per answer (relevance-floored)
    recall_max_profile: int = 8        # most-salient profile notes always loaded
    daily_cost_cap_usd: float = 5.0
    # Tiered spend governance: BACKGROUND cloud work (dreams, self-dev, hive)
    # stops using paid rungs once the day's spend crosses this SOFT cap, so a 3am
    # runaway can't exhaust the quota — foreground chat stays protected up to the
    # hard daily_cost_cap_usd. Keep below the hard cap. (0 = background never
    # uses paid rungs; >= daily cap = no separate throttle.)
    background_cost_cap_usd: float = 3.0
    request_timeout: int = 120
    # Which rung (by name) the Planner and Critic roles run at — point these at
    # "cloud-open" so the big cloud model plans/verifies independently of the
    # local worker. (Ollama-only stack: there are no hosted-API rungs.)
    planner_rung: str = "cloud-open"
    critic_rung: str = "cloud-open"
    # Floor for FOREGROUND chat turns only (ask + approval resume). Set to a
    # stronger rung (e.g. claude-sonnet) to give the family first-shot quality
    # while every background/automation role (planner, dreams, scribe, hive,
    # skill review, critic) stays on the cheap base rung. Empty = ladder base.
    chat_rung: str = ""
    server_port: int = 8765
    bind_host: str = "127.0.0.1"       # set to the Tailscale IP (or 0.0.0.0) to reach ANVIL off-box
    workspace_dir: Path = Path("workspace")
    tools_enabled: bool = True
    max_tool_steps: int = 6
    # Planning interval: on a long tool-using turn, every Nth step the model is
    # nudged to step back and re-plan (no tool call) so it doesn't tunnel down a
    # dead end. 0 = off. Only fires once several tool actions are already in.
    planning_interval: int = 4
    shell_timeout: int = 600           # shell tool budget (distinct from request_timeout)
    # Which rung handles requests with attached images. gemma4 ADVERTISES vision
    # but ignores images on Ollama 0.31.1 (verified 2026-07-02); qwen3.6 sees.
    vision_rung: str = "local-reason"
    # Reasoning-model 'thinking' for chat: 'auto' (reason only for hard prompts —
    # ~8x faster on simple ones), 'on' (always), 'off' (never). Autopilot dreams
    # keep thinking regardless (quality over latency, and they run in background).
    chat_think: str = "auto"
    # How much privilege Lara has (ask / trusted / auto):
    # 'ask' = every danger action needs approval; 'trusted' (default) =
    # read-only shell commands + operator-taught 'Always allow' commands run
    # free, everything else asks; 'auto' = nothing asks (hard denylist still
    # refuses catastrophes). Set from the Setup tab.
    autonomy: str = "trusted"
    # The skill flywheel: after substantive turns, a background reviewer updates
    # the skill library (patches the skill in play, or distils a new one) so
    # Lara gets better at your recurring tasks over time. Off = never learn
    # unprompted. Fires every Nth substantive turn, on a cheap local rung.
    skill_learning: bool = True
    skill_review_every: int = 3
    skill_review_rung: str = "local-fast"
    # Curator (runs in deep sleep): age out skills nothing recalls anymore.
    skill_stale_days: int = 30
    skill_archive_days: int = 90
    # Anti-rumination: cap on the thinking stream per model call (chars, ~1500
    # tokens). Past it the call is cancelled and re-asked with thinking off —
    # a looping deliberation ("but wait—" forever) gets cut, not waited out.
    think_budget_chars: int = 6000
    # Context watchdog: when a completion's true prompt size (prompt_eval_count)
    # crosses this, the live tool-session context is compacted deterministically
    # (old observations capped, middle dropped) — prompt processing stays fast
    # and nothing silently overflows the model's window.
    ctx_soft_limit_tokens: int = 12000
    # Self-hosted SearXNG instance URL (e.g. http://archive:8088). When set, it's
    # the PRIMARY search tier — free, private, 70+ engines — with Tavily and then
    # DuckDuckGo as fallbacks. Blank = skip straight to Tavily/DDG.
    searxng_url: str = ""
    # VPS/public-front-door hardening: the reverse proxy's IP(s), comma-
    # separated. ONLY requests arriving FROM these addresses get their
    # X-Forwarded-For (real client IP, feeds the login limiter) and
    # X-Forwarded-Proto (https -> Secure cookies) headers honored — anyone
    # else claiming a forwarded identity is ignored. Empty = no proxy.
    # Real-time HA sensing over the websocket event API (docs/
    # ha-websocket-design.md). OFF by default — snapshot-diff sensing stays
    # the engine until this is deliberately flipped after a soak.
    ha_stream: bool = False
    trusted_proxy: str = ""
    # The deliberately-published domain (e.g. lara.example.com) when a VPS
    # front door exists. Host-header requests for it pass the rebinding
    # check; everything else non-local is still refused. Empty = none.
    public_host: str = ""
    # On Anthropic rungs, swap the local `search` tool for Anthropic's
    # server-side web_search: the search runs INSIDE the API call (no client
    # round trip), returns cited results, and retires the scrape fragility.
    # $10 per 1,000 searches, priced into the ledger. Non-Anthropic rungs
    # keep the local SearXNG/Tavily/DDG stack either way.
    native_web_search: bool = True
    web_search_max_uses: int = 3    # per-request ceiling on server-side searches
    # Message Batches (the overnight 50% lever): paid Anthropic calls on the
    # latency-insensitive background planes (self-dev, dreams, scribe) go
    # through the async Batches API — half price on every token. Each call
    # becomes submit-then-poll (minutes, not seconds), which only the operator
    # watching the dev queue ever notices. Chat, approvals, and hive drones
    # stay on the live API unconditionally — batching can't touch the family.
    batch_background: bool = False
    batch_wait_s: int = 3600        # cancel a batch still queued after this
    batch_poll_s: int = 20          # seconds between status polls
    # Local model context window (0 = Ollama's server default, 32768 on 0.31.1).
    # Measured on the 24GB 7900 XTX with qwen3.6 Q4: 65536 still fits 100% on
    # GPU (24GB); 131072 spills to CPU and generation craters. The model itself
    # supports 262144 — that headroom needs a second card.
    local_num_ctx: int = 0
    # The hive: Lara's parallel worker drones (delegate tool). Cloud rung by
    # default — zero local VRAM, so workers truly run in parallel while Lara
    # keeps the GPU.
    hive_worker_rung: str = "cloud-open"
    hive_max_workers: int = 4
    # Local lane: up to this many hive drones may share the LOCAL model at once
    # (Ollama batches parallel requests). The privacy-pinned HOME expert runs
    # here. 0 = never use the local model for drones (cloud-only bench).
    hive_local_slots: int = 2
    # Council escalation: when a lead expert comes back weak/failed, convene the
    # lead + relevant backups and synthesise. Off = single expert, no fallback.
    hive_council: bool = True
    # Who writes the final answer. Local qwen ALWAYS fronts the chat + calls the
    # tools; this decides who SYNTHESISES:
    #   'local'    — qwen (fastest, fully private; current behaviour)
    #   'balanced' — cloud writes SUBSTANTIVE answers (research/reasoning); qwen
    #                keeps trivial + HOUSE turns (limits house data leaving box)
    #   'cloud'    — cloud writes everything but pure trivial (house included)
    # Cloud unreachable -> silently falls back to the local answer.
    synthesis_mode: str = "balanced"
    # Self-dev: code the change directly on the frontier cloud rung (the top of
    # the ladder — a strong coding model) instead of a local attempt first. On
    # an Ollama Max sub the cloud time is paid for and the patches are better.
    selfdev_cloud_first: bool = False
    # Auto-push: after each doctor-green forge/self-dev commit, push the forge
    # branch to the Gitea remote so autonomous self-improvement is backed up
    # off-box immediately. Off unless enabled AND a GITEA_TOKEN + remote exist.
    forge_push: bool = False
    forge_push_remote: str = "gitea"
    # Gitea CI/CD (issues, promotion PRs): API base + owner/repo. Blank => derived
    # from the gitea remote's URL. See docs/cicd.md.
    gitea_url: str = ""       # e.g. "http://archive:3000"
    gitea_repo: str = ""      # e.g. "bytesnap/ANVIL"
    # The working ('test') branch the forge/self-dev commits to; main is promoted-to.
    forge_branch: str = "test"
    # Lara works the Gitea issue queue autonomously (deep sleep): assess -> ask for
    # clarification / council push-back / work the fix on `test`, showing her work.
    issue_work: bool = False
    issue_work_label: str = "selfdev"
    ask_time_budget_s: int = 240       # wall-clock cap per chat turn's tool loop
    stm_cap: int = 300
    dream_after: int = 40
    # Conversation history: Lara remembers the current chat's turns.
    conversations_dir: str = "conversations"
    conv_token_budget: int = 6000      # recent turns kept verbatim up to this; older summarized
    conv_keep_recent: int = 8          # always keep at least this many recent turns verbatim
    conv_disk_cap: int = 400           # max turns retained per conversation on disk
    # Autopilot: ANVIL thinks & dreams on its own while the server is up.
    auto_pulse: bool = True
    sense_house: bool = True           # observe Home Assistant on each heartbeat
    heartbeat_interval_min: int = 15   # minutes between spontaneous thoughts
    chat_quiet_cooldown_min: int = 5   # skip model-heavy autopilot ticks this long after a chat (0=off)
    dream_every_ticks: int = 8         # `anvil pulse` only: dream after this many heartbeats
    ltm_forget_floor: float = 0.08     # notes decaying below this are forgotten in sleep
    # Memory dynamics (review 2.1): activation decays by WALL CLOCK at read time
    # (sidecar memory/notes/salience.json), not by dream count — and forgetting
    # requires BOTH low activation AND no actual recall within the window.
    salience_half_life_days: float = 14.0
    forget_unused_days: float = 21.0   # never forget a note recalled this recently
    # FTS index (anvil/index.py): below this many notes the full scan runs as
    # before; above it, BM25 narrows recall/dedup candidates (files stay canon).
    index_min_notes: int = 100
    # Autopilot cadence — wall-clock intervals persisted across restarts (see
    # anvil/schedule.py). Dreams are WORK-driven (STM fills to dream_after) with a
    # max-age backstop; the dev stages each have their own independent dial instead
    # of the old multiplied tick counters (which reset on every restart).
    dream_max_age_hours: float = 6.0   # dream at least this often, even if STM is quiet
    triage_debounce_min: int = 30      # min gap between incident-triage passes
    promote_debounce_min: int = 20     # min gap between promote attempts (batches accumulate)
    selfdev_interval_hours: float = 12.0  # speculative self-dev crawl: every this often
    # Deep sleep: ANVIL develops its own harness (git-guarded) during sleep.
    self_dev_in_sleep: bool = True     # run a self-dev cycle during deep sleep
    self_dev_daily_cap: int = 3        # max self-dev cycles per day (runaway guard)
    # Web Push (PWA notifications). VAPID keys are generated on first use and
    # stored under memory_dir; push_contact is the VAPID "sub" (a mailto/URL the
    # push service can reach you at). Requires a secure (HTTPS) origin on iOS.
    push_enabled: bool = True
    # Quiet hours for AMBIENT pushes only (spontaneous thoughts, dream insights);
    # chat answers and approval requests always deliver. Local hours, overnight
    # wrap supported; set both equal to disable.
    push_quiet_start: int = 22
    push_quiet_end: int = 7
    # Weather senses (NWS api.weather.gov — US only, keyless). Home location:
    # set home_lat/home_lon directly, or home_address (geocoded via the free
    # Census geocoder), or leave all empty to use Home Assistant's zone.home.
    home_address: str = ""
    home_lat: float = 0.0
    home_lon: float = 0.0
    # Folder of the family's own documents (manuals, warranties, medical/insurance
    # papers, recipes) that Lara indexes for RAG — see anvil/docs.py. Drop files
    # in; they're indexed during sleep (and on demand).
    family_docs_dir: str = "family_docs"
    # VAPID "sub": a contact the push service can reach the operator at. Apple
    # rejects non-real addresses (e.g. anything @localhost) with 403 BadJwtToken,
    # so this must be a valid mailto:/https: URI. Override in anvil.toml with your
    # own email if you want push-service notices.
    push_contact: str = "mailto:anvil@example.com"
    ollama_api_key: Optional[str] = field(default=None, repr=False)
    # Claude Messages API key — when present, the tiered Anthropic ladder
    # (Haiku -> Sonnet -> Opus -> Fable) lights up; absent, those rungs are
    # skipped and the stack stays Ollama-only. Read from ANTHROPIC_API_KEY.
    anthropic_api_key: Optional[str] = field(default=None, repr=False)
    # OpenAI key — used ONLY by the generate_image tool (gpt-image-1); chat
    # never routes here. Read from OPENAI_API_KEY. Absent = the tool refuses
    # with a clear message, everything else unaffected.
    openai_api_key: Optional[str] = field(default=None, repr=False)
    # Rolling 30-day spend ceiling (0 = off). Foreground chat is protected up to
    # daily_cost_cap_usd; this is the extra monthly guard the operator set so the
    # metered Claude bill can't drift past a known number.
    monthly_cost_cap_usd: float = 0.0
    gitea_token: Optional[str] = field(default=None, repr=False)  # GITEA_TOKEN, for forge auto-push
    issue_actor: str = "lara"      # Lara's Gitea login — her own comments are excluded from assessment
    issue_operator: str = "bytesnap"   # human to assign when Lara needs help (after 3 tries)
    council_rebuttal_days: int = 7     # window to rebut a council push-back before it stands
    main_branch: str = "main"          # the promotion TARGET — stable, reached only via a reviewed PR
    # Phase 3: after the test gate is green, promote() opens a `test`->`main` PR. When
    # auto_promote is on it also MERGES (fully autonomous release); off (default) it
    # leaves the PR for the operator to approve/merge.
    auto_promote: bool = False
    discord_webhook_url: Optional[str] = field(default=None, repr=False)
    discord_bot_token: Optional[str] = field(default=None, repr=False)

    def rung(self, index: int) -> Rung:
        index = max(0, min(index, len(self.ladder) - 1))
        return self.ladder[index]

    def rung_by_name(self, name: str) -> Optional[int]:
        for i, r in enumerate(self.ladder):
            if r.name == name:
                return i
        # Legacy-name TIER aliases: years of call sites reference the Ollama-era
        # rung names (local-fast/cloud-open/...). On a renamed ladder (e.g. the
        # Claude tiers) those lookups used to return None -> silently rung 0.
        # Resolve them by ROLE instead: base tier / workhorse+1 / review tier —
        # clamped to the ladder, and inert when the exact names still exist.
        tier = {"local-fast": 0, "local-reason": 0,
                "cloud-open": 1, "cloud-heavy": 2, "cloud-logic": 2}.get(name)
        if tier is not None and self.ladder:
            return min(tier, len(self.ladder) - 1)
        return None


DEFAULT_LADDER = [
    Rung("local-fast", "ollama_local", "qwen3-coder:30b", max_context=64_000),
    Rung("local-reason", "ollama_local", "qwen3-coder:30b", max_context=64_000, think=True),
    Rung("cloud-open", "ollama_cloud", "qwen3-coder:480b-cloud", max_context=256_000),
]


def default_config(base_dir: Optional[Path] = None) -> Config:
    cfg = Config(ladder=list(DEFAULT_LADDER))
    if base_dir:
        cfg = _resolve_paths(cfg, Path(base_dir))
    return from_env(cfg)


def _load_toml(path: Path) -> Dict[str, Any]:
    # Read tolerantly and strip NUL/control junk some tools inject, so a
    # corrupted file can never crash startup.
    text = path.read_text("utf-8", "replace").replace("\x00", "")
    try:
        import tomllib  # Python 3.11+
        try:
            return tomllib.loads(text)
        except Exception:
            return _mini_toml(text)            # strict-invalid -> lenient fallback
    except ModuleNotFoundError:
        pass
    try:
        import tomli  # backport
        try:
            return tomli.loads(text)
        except Exception:
            return _mini_toml(text)
    except ModuleNotFoundError:
        return _mini_toml(text)


def _coerce(val: str) -> Any:
    val = val.strip()
    if val and val[0] in "\"'" and val[-1] == val[0]:
        return val[1:-1]
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _mini_toml(text: str) -> Dict[str, Any]:
    """Minimal TOML reader for ANVIL's config (no third-party dep needed)."""
    root: Dict[str, Any] = {}
    cur: Dict[str, Any] = root
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "[[ladder]]":
            cur = {}
            root.setdefault("ladder", []).append(cur)
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = root
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            cur[k.strip()] = _coerce(v)
    return root


def load(path: Optional[str] = None) -> Config:
    """Load config from a file, applying defaults for anything unspecified."""
    p: Optional[Path] = Path(path) if path else None
    if p is None:
        for cand in (Path("anvil.toml"), Path("anvil.json")):
            if cand.exists():
                p = cand
                break
    if p is None or not p.exists():
        return default_config()

    raw = _load_toml(p) if p.suffix == ".toml" else json.loads(p.read_text("utf-8"))
    cfg = _from_dict(raw)
    cfg = _resolve_paths(cfg, p.parent)
    cfg = from_env(cfg)
    _validate(cfg)                 # coerce out-of-range enums (file/env) to safe defaults
    return cfg


_VALID_AUTONOMY = ("ask", "trusted", "auto")
_VALID_SYNTHESIS = ("local", "balanced", "cloud")
_VALID_CHAT_THINK = ("auto", "on", "off")


def _validate(cfg: Config) -> None:
    """Coerce out-of-range enum settings that came from the file or environment to
    their safe defaults — WARNING to stderr, never silently, so an operator typo in
    `anvil.toml` can't quietly change how much privilege Lara has. Runs in the load
    path (after the file and env are applied), which is the only place a bad value can
    enter — the Setup UI already validates its own writes.
    #6: bytesnap directed load-path validation + a warning; __post_init__ can't see the
    TOML value because load() sets autonomy after construction."""
    import sys
    if cfg.autonomy not in _VALID_AUTONOMY:
        print(f"[anvil] config: invalid autonomy {cfg.autonomy!r} — using 'trusted' "
              f"(valid: {', '.join(_VALID_AUTONOMY)})", file=sys.stderr)
        cfg.autonomy = "trusted"
    if not cfg.bind_host or not cfg.bind_host.strip():   # issue #7
        print(f"[anvil] config: empty bind_host {cfg.bind_host!r} — using '127.0.0.1' "
              f"(set to a Tailscale IP or 0.0.0.0 to reach ANVIL off-box)", file=sys.stderr)
        cfg.bind_host = "127.0.0.1"
    if cfg.synthesis_mode not in _VALID_SYNTHESIS:
        print(f"[anvil] config: invalid synthesis_mode {cfg.synthesis_mode!r} — using 'balanced' "
              f"(valid: {', '.join(_VALID_SYNTHESIS)})", file=sys.stderr)
        cfg.synthesis_mode = "balanced"
    if cfg.chat_think not in _VALID_CHAT_THINK:
        print(f"[anvil] config: invalid chat_think {cfg.chat_think!r} — using 'auto' "
              f"(valid: {', '.join(_VALID_CHAT_THINK)})", file=sys.stderr)
        cfg.chat_think = "auto"


def _rung_from_dict(r: Any) -> Optional[Rung]:
    """Build a Rung tolerantly: drop unknown keys, skip entries missing a
    required field. A stray/typo'd key in a ``[[ladder]]`` block can't crash
    startup — it's just ignored, same as every other corruption path here."""
    if not isinstance(r, dict):
        return None
    known = set(Rung.__dataclass_fields__)
    kwargs = {k: v for k, v in r.items() if k in known}
    try:
        return Rung(**kwargs)
    except TypeError:          # missing name/provider/model -> unusable rung
        return None


def _from_dict(raw: Dict[str, Any]) -> Config:
    ladder_raw = raw.get("ladder")
    ladder = [r for r in (_rung_from_dict(x) for x in (ladder_raw or [])) if r]
    if not ladder:
        ladder = list(DEFAULT_LADDER)   # tolerate a corrupt/empty ladder
    known = {f for f in Config.__dataclass_fields__ if f != "ladder"}
    kwargs = {k: v for k, v in raw.items() if k in known and k != "ladder"}
    for pf in ("memory_dir", "jobs_dir", "ledger_path", "workspace_dir"):
        if pf in kwargs:
            kwargs[pf] = Path(kwargs[pf])
    return Config(ladder=ladder, **kwargs)


def _resolve_paths(cfg: Config, base: Path) -> Config:
    base = base.resolve()
    for pf in ("memory_dir", "jobs_dir", "ledger_path", "workspace_dir"):
        val = getattr(cfg, pf)
        if not val.is_absolute():
            setattr(cfg, pf, base / val)
    return cfg


def atomic_write(path, text: str) -> None:
    """Write text robustly, even over a read-only/locked file (Windows-safe).

    Clears a read-only attribute, writes via a temp file + atomic replace, and
    falls back to unlink+write if the target is locked. Raises only if the
    directory itself is not writable.
    """
    import os
    import stat
    import time
    path = Path(path)
    text = text.replace("\x00", "")   # never write NUL corruption
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
    tmp = path.with_name(path.name + ".tmp")
    # Prefer the atomic temp-write + replace, retrying a few times before giving
    # up: on Windows a transient lock (AV scanner, sync tool, search indexer)
    # briefly holds the target and makes os.replace fail, but it clears within
    # milliseconds. Retrying keeps the fast atomic path instead of dropping to
    # the destructive unlink+rewrite below, which can lose data (or crash) if
    # the lock outlives the unlink. The success path pays no retry cost.
    for attempt in range(4):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            return
        except OSError:
            if attempt < 3:
                time.sleep(0.02 * (attempt + 1))
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
    path.write_text(text, encoding="utf-8")
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass


def dir_writable(path) -> bool:
    """True if we can create AND replace a file here (what atomic_write needs).

    Deliberately does not require unlink — some mounts block deletion but allow
    create+replace, which is all persistence needs.
    """
    import os
    base = Path(path)
    probe = base / ".anvil_write_probe"
    tmp = base / ".anvil_write_probe.tmp"
    ok = False
    try:
        probe.write_text("ok", encoding="utf-8")
        tmp.write_text("ok2", encoding="utf-8")
        os.replace(tmp, probe)               # the exact op atomic_write relies on
        ok = probe.read_text("utf-8") == "ok2"
    except OSError:
        ok = False
    for f in (probe, tmp):                    # best-effort cleanup
        try:
            if f.exists():
                f.unlink()
        except OSError:
            pass
    return ok


def _dir_writable_unused(path) -> bool:
    import os
    probe = Path(path) / ".anvil_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        os.replace(probe, probe)
        probe.unlink()
        return True
    except OSError:
        try:
            if probe.exists():
                probe.unlink()
        except OSError:
            pass
        return False


def from_env(cfg: Config) -> Config:
    """Layer environment-variable secrets and overrides onto a Config."""
    cfg.ollama_api_key = os.environ.get("OLLAMA_API_KEY", cfg.ollama_api_key)
    cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", cfg.anthropic_api_key)
    cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", cfg.openai_api_key)
    cfg.gitea_token = os.environ.get("GITEA_TOKEN", cfg.gitea_token)
    cfg.discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", cfg.discord_webhook_url)
    cfg.discord_bot_token = os.environ.get("DISCORD_BOT_TOKEN", cfg.discord_bot_token)
    if "OLLAMA_HOST" in os.environ:
        cfg.ollama_local_url = os.environ["OLLAMA_HOST"]
    if "ANVIL_DAILY_CAP" in os.environ:
        # Tolerate a malformed value (typo / corruption) the same way every
        # other load path here does: keep the default rather than crash startup.
        try:
            cfg.daily_cost_cap_usd = float(os.environ["ANVIL_DAILY_CAP"])
        except ValueError:
            pass
    return cfg
