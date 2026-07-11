"""Procedural memory — the library of skills Lara has learned.

A third memory tier, distinct from the other two:
  * LTM notes (memory.py) — facts & lessons, decay by salience.
  * conversations — the running transcript of a chat.
  * **skills (here)** — reusable PROCEDURES: how to do a recurring task
    reliably. A skill NEVER decays; a procedure that worked is worth keeping
    until explicitly replaced. (DreamEngine's insight: skill-memories must be
    exempt from the compaction that fades observations.)

Format borrowed from DreamEngine's skill clusters: one folder per skill with a
``SKILL.md`` — YAML-ish frontmatter (``name`` / ``description`` / ``when``) plus
a Markdown body. Greppable, git-able, one skill per file. Stored under
``memory/skills/<name>/SKILL.md``.

Recall is keyword-overlap against name+description+when (no embeddings needed):
the matching skills' bodies are injected into the agent's context, and a compact
catalog of all skill names+descriptions is always available so the model KNOWS
what it can do.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config as cfgmod

_LOCK = threading.Lock()

# Per-file parse cache for SkillStore.all(): path -> (mtime_ns, size, Skill).
# all() runs on EVERY chat turn — _skills_context() calls count()/recall()/
# catalog(), each of which re-globs and re-reads+re-parses every SKILL.md.
# A stat() per file is enough to know the cached parse is still valid; any
# rewrite (atomic_write replaces the file) changes mtime/size and misses the
# cache naturally — the same invariant memory.py's _NOTE_CACHE relies on.
_SKILL_CACHE: Dict[str, tuple] = {}

# The system-prompt skill index truncates each description hard at this length
# every session (Hermes' most-violated rule) — anything past it silently never
# routes. Enforced on write so a runaway description can't kill discoverability.
_MAX_DESC = 60


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return (s[:48] or "skill").strip("-")


class Skill:
    def __init__(self, name: str, description: str, body: str,
                 when: str = "", path: Optional[Path] = None,
                 requires: str = ""):
        self.name = name
        self.description = description
        self.when = when
        self.body = body
        self.path = path
        # Gating prerequisites: space/comma-separated 'kind:name' tokens —
        # bin:ffmpeg (on PATH), env:TAVILY_API_KEY (set), config:searxng_url
        # (truthy on cfg). A skill whose requirements aren't met is HIDDEN from
        # recall/catalog so the model never tries a procedure it can't run.
        self.requires = requires

    def _reqs(self) -> List[str]:
        return [t for t in re.split(r"[,\s]+", (self.requires or "").strip()) if t]

    def is_available(self, cfg) -> bool:
        for tok in self._reqs():
            kind, _, name = tok.partition(":")
            kind, name = kind.strip().lower(), name.strip()
            if not name:
                continue
            if kind == "bin" and shutil.which(name) is None:
                return False
            if kind == "env" and not os.environ.get(name):
                return False
            if kind == "config" and not getattr(cfg, name, None):
                return False
        return True

    def to_markdown(self) -> str:
        fm = [f"name: {self.name}", f"description: {self.description}"]
        if self.when:
            fm.append(f"when: {self.when}")
        if self.requires:
            fm.append(f"requires: {self.requires}")
        return "---\n" + "\n".join(fm) + "\n---\n\n" + (self.body or "").strip() + "\n"


def parse_skill(text: str, path: Optional[Path] = None) -> Skill:
    name = description = when = requires = ""
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip().lower(), v.strip()
                if k == "name":
                    name = v
                elif k == "description":
                    description = v
                elif k == "when":
                    when = v
                elif k == "requires":
                    requires = v
        body = m.group(2).strip()
    if not name and path is not None:
        name = path.parent.name
    return Skill(name=name, description=description, body=body, when=when,
                 path=path, requires=requires)


class SkillStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg.memory_dir) / "skills"

    def available(self) -> List[Skill]:
        """Skills whose gating prerequisites (requires:) are satisfied here."""
        return [s for s in self.all() if s.is_available(self.cfg)]

    def _path(self, name: str) -> Path:
        return self.dir / _slug(name) / "SKILL.md"

    def all(self) -> List[Skill]:
        out: List[Skill] = []
        if not self.dir.exists():
            return out
        with _LOCK:
            paths = sorted(self.dir.glob("*/SKILL.md"))
            for p in paths:
                try:
                    st = p.stat()
                    key = str(p)
                    hit = _SKILL_CACHE.get(key)
                    if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                        out.append(hit[2])
                        continue
                    sk = parse_skill(p.read_text("utf-8", "replace"), p)
                    _SKILL_CACHE[key] = (st.st_mtime_ns, st.st_size, sk)
                    out.append(sk)
                except Exception:
                    continue
            # Evict entries for skills whose files are gone (deleted/archived),
            # scoped to this store's dir so it can't grow stale.
            live = {str(p) for p in paths}
            for k in [k for k in _SKILL_CACHE if k not in live
                      and k.startswith(str(self.dir))]:
                _SKILL_CACHE.pop(k, None)
        return out

    def get(self, name: str) -> Optional[Skill]:
        p = self._path(name)
        if p.exists():
            try:
                return parse_skill(p.read_text("utf-8", "replace"), p)
            except Exception:
                return None
        return None

    def write(self, name: str, description: str, body: str,
              when: str = "", by: str = "agent", requires: str = "") -> Skill:
        name = (name or "").strip()
        if not name:
            raise ValueError("skill name is required")
        # Skills are the one write that crosses the web boundary INTO future
        # prompts: a recalled skill body is injected as Lara's own learned
        # procedure, every matching turn, forever. So the injection scan that
        # guards web content at fetch time must also guard what gets to
        # persist here — otherwise fetched attacker text is laundered into
        # permanently-trusted context by way of "learning".
        from . import safety
        findings = safety.scan(f"{name}\n{description}\n{when}\n{body}")
        if findings:
            raise ValueError(
                f"skill rejected — injection patterns found: {', '.join(findings)}")
        description = (description or "").strip()
        if len(description) > _MAX_DESC:               # hard cap — see _MAX_DESC
            description = description[:_MAX_DESC - 1].rstrip() + "…"
        p = self._path(name)
        existed = p.exists()
        sk = Skill(name=name, description=description, body=body, when=when,
                   path=p, requires=requires)
        with _LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            cfgmod.atomic_write(p, sk.to_markdown())
        self._record(_slug(name), "patch" if existed else "create", by=by)
        return sk

    def delete(self, name: str) -> bool:
        p = self._path(name)
        if p.exists():
            with _LOCK:
                try:
                    p.unlink()
                    try:
                        p.parent.rmdir()      # remove the now-empty skill folder
                    except OSError:
                        pass
                    return True
                except OSError:
                    return False
        return False

    def recall(self, query: str, limit: int = 2) -> List[Skill]:
        """Skills whose name/description/when overlap the query, best first.
        A recalled skill is injected into context = a USE, so we record it (the
        curator ages out skills nothing ever recalls)."""
        qwords = set(re.findall(r"[a-z0-9]{4,}", (query or "").lower()))
        if not qwords:
            return []
        scored = []
        for sk in self.available():          # never surface an ungated-out skill
            hay = f"{sk.name} {sk.description} {sk.when}".lower()
            hw = set(re.findall(r"[a-z0-9]{4,}", hay))
            score = len(qwords & hw)
            if score:
                scored.append((score, sk))
        scored.sort(key=lambda x: -x[0])
        hits = [sk for _, sk in scored[:limit]]
        for sk in hits:
            # SURFACED, not used (review 2.5): keyword-matching into a prompt
            # is exposure. Counting it as usage made promiscuous generic skills
            # immortal — their use_count climbed on every turn they merely
            # matched, so the inactivity curator could never touch them.
            self._record(_slug(sk.name), "surfaced")
        return hits

    # ---- usage telemetry sidecar (memory/skills/.usage.json) -------------- #
    def _usage_path(self) -> Path:
        return self.dir / ".usage.json"

    def _load_usage(self) -> Dict[str, dict]:
        try:
            return json.loads(self._usage_path().read_text("utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return {}

    def _record(self, slug: str, event: str, by: str = "agent") -> None:
        """event: create | patch | use | view. Never raises — telemetry must
        not break a skill write or recall."""
        try:
            with _LOCK:
                u = self._load_usage()
                now = time.time()
                r = u.get(slug) or {"created_by": by, "created_at": now,
                                    "use_count": 0, "view_count": 0,
                                    "patch_count": 0, "state": "active"}
                if event == "create":
                    r["created_by"], r["created_at"] = by, now
                elif event == "use":
                    r["use_count"] = r.get("use_count", 0) + 1
                elif event == "surfaced":
                    r["surfaced_count"] = r.get("surfaced_count", 0) + 1
                elif event == "view":
                    r["view_count"] = r.get("view_count", 0) + 1
                elif event == "patch":
                    r["patch_count"] = r.get("patch_count", 0) + 1
                if event in ("use", "view", "create", "patch", "surfaced"):
                    r["last_activity_at"] = now
                    r["state"] = "active"
                u[slug] = r
                self.dir.mkdir(parents=True, exist_ok=True)
                cfgmod.atomic_write(self._usage_path(), json.dumps(u, indent=1))
        except Exception:
            pass

    def record_view(self, name: str) -> None:
        self._record(_slug(name), "view")

    def record_use(self, name: str) -> None:
        """A REAL usage signal: the answer actually leaned on this skill (or
        the model deliberately opened it). This is what keeps a skill alive."""
        self._record(_slug(name), "use")

    # ---- curator: age out skills nothing recalls anymore ------------------ #
    def prune(self, stale_days: int = 30, archive_days: int = 90) -> dict:
        """Deterministic inactivity pass (no model call): mark long-unused
        skills 'stale', and ARCHIVE (move to .archive/) ones unused even
        longer. Agent-created only — a skill never touched again is clutter.
        Returns {stale:[...], archived:[...]}."""
        now = time.time()
        u = self._load_usage()
        stale, archived = [], []
        for sk in self.all():
            slug = _slug(sk.name)
            r = u.get(slug) or {}
            last = r.get("last_activity_at") or r.get("created_at") or now
            idle_days = (now - last) / 86400.0
            # Promiscuous-but-useless (review 2.5): surfaced into prompts many
            # times, never once actually USED. Under the old telemetry these
            # were the most immortal skills in the library — and the most
            # harmful, crowding the 2 injection slots out from under genuinely
            # useful procedures on every matching turn.
            surfaced = int(r.get("surfaced_count", 0))
            used = int(r.get("use_count", 0)) + int(r.get("view_count", 0))
            if surfaced >= 20 and used == 0:
                if self._archive(slug):
                    r["state"] = "archived"; r["archived_at"] = now
                    u[slug] = r; archived.append(sk.name)
                continue
            if idle_days >= archive_days:
                if self._archive(slug):
                    r["state"] = "archived"; r["archived_at"] = now
                    u[slug] = r; archived.append(sk.name)
            elif idle_days >= stale_days and r.get("state") != "stale":
                r["state"] = "stale"; u[slug] = r; stale.append(sk.name)
        if stale or archived:
            with _LOCK:
                self.dir.mkdir(parents=True, exist_ok=True)
                cfgmod.atomic_write(self._usage_path(), json.dumps(u, indent=1))
        return {"stale": stale, "archived": archived}

    def _archive(self, slug: str) -> bool:
        src = self.dir / slug
        if not src.exists():
            return False
        dst = self.dir / ".archive" / slug
        try:
            with _LOCK:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.move(str(src), str(dst))
            return True
        except OSError:
            return False

    def catalog(self) -> str:
        """One line per skill (name — description): so the model always knows
        what procedures it has, even when recall doesn't inject the full body."""
        lines = [f"- {s.name}: {s.description}"
                 for s in self.available() if s.name]
        return "\n".join(lines)

    def count(self) -> int:
        return len(self.all())
