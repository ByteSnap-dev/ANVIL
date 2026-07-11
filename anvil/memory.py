"""File-based notes & memory with context-budget-aware recall.

Notes are plain Markdown files with a small YAML-style frontmatter block —
greppable, git-able, no database. One fact per file. ``INDEX.md`` is the
always-loaded cheap map of what exists.

Recall ranks notes by semantic similarity (Ollama embeddings, optional) plus
tag overlap, recency, and salience, then packs the top results into a fixed
token budget so notes are always *available* but only *loaded* when they earn
their token cost.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from . import config as cfgmod
from . import context

# Serialize INDEX.md / embeddings writers across the request threads (scribe)
# and the mind thread (dreams) — both call MemoryStore.write() concurrently.
# RLock (not Lock) so write() can hold it across the whole dedup-through-file
# creation critical section while the helpers it calls (all_notes, _index_note,
# _embed_note) re-acquire the same lock without deadlocking.
_WRITE_LOCK = threading.RLock()
# Parse cache for the salience-dynamics sidecar (mtime+size keyed).
_DYN_CACHE: Dict[str, tuple] = {}
# Per-file parse cache for all_notes(): path -> (mtime_ns, size, Note).
_NOTE_CACHE: Dict[str, tuple] = {}
# Per-file parse cache for _load_embeddings(): path -> (mtime_ns, size, dict).
# recall() runs on EVERY chat turn and full-JSON-parses embeddings.jsonl each
# time; that file grows one line per note forever. Cache the parse keyed on
# stat() — any write (append or atomic_write replace) changes mtime/size and
# misses the cache naturally, same invariant as _NOTE_CACHE.
_EMBED_CACHE: Dict[str, tuple] = {}


# --------------------------------------------------------------------------- #
# Note model + tiny frontmatter parser (no PyYAML dependency)
# --------------------------------------------------------------------------- #
@dataclass
class Note:
    name: str
    body: str
    type: str = "reference"
    tags: List[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: date.today().isoformat())
    salience: float = 0.5
    path: Optional[Path] = None
    # Action-sensitivity (adapted from OpenClaw's active-memory framing):
    #   act = "never" — sensitive (a secret, a password); recall it for context
    #                   but NEVER act on it autonomously or repeat it aloud.
    #   act = "ask"   — actionable only with explicit operator confirmation.
    #   act = ""/"auto" — an ordinary fact.
    # expires = ISO date after which the fact is stale ("away until 2026-07-11")
    #                   and is dropped from recall.
    act: str = ""
    expires: str = ""
    # Family privacy. Memory is PRIVATE-BY-DEFAULT: a note is owned by the family
    # member who created it and no one else sees it — no leakage unless shared.
    #   owner = ""          -> HOUSEHOLD/ambient (Lara's own observations about
    #                          the home); common baseline, visible to everyone.
    #   owner = "<profile>" -> that person's PRIVATE memory.
    #   shared_with         -> extra profiles the owner chose to share it with
    #                          ("*" = everyone). See MemoryStore._visible / .share.
    owner: str = ""
    shared_with: List[str] = field(default_factory=list)

    def is_expired(self, today: Optional[str] = None) -> bool:
        if not self.expires:
            return False
        return (today or date.today().isoformat()) > self.expires.strip()

    def to_markdown(self) -> str:
        tags = "[" + ", ".join(self.tags) + "]"
        extra = ""
        if self.act:
            extra += f"act: {self.act}\n"
        if self.expires:
            extra += f"expires: {self.expires}\n"
        if self.owner:
            extra += f"owner: {self.owner}\n"
        if self.shared_with:
            extra += "shared_with: [" + ", ".join(self.shared_with) + "]\n"
        return (
            "---\n"
            f"name: {self.name}\n"
            f"type: {self.type}\n"
            f"tags: {tags}\n"
            f"created: {self.created}\n"
            f"salience: {self.salience}\n"
            f"{extra}"
            "---\n"
            f"{self.body.strip()}\n"
        )


_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_note(text: str, path: Path) -> Note:
    m = _FM_RE.match(text)
    if not m:
        return Note(name=path.stem, body=text.strip(), path=path)
    fm, body = m.group(1), m.group(2)
    meta: Dict[str, str] = {}
    for line in fm.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    tags_raw = meta.get("tags", "").strip().strip("[]")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    try:
        salience = float(meta.get("salience", 0.5))
    except ValueError:
        salience = 0.5
    return Note(
        name=meta.get("name", path.stem),
        body=body.strip(),
        type=meta.get("type", "reference"),
        tags=tags,
        created=meta.get("created", date.today().isoformat()),
        salience=salience,
        act=meta.get("act", "").strip().lower(),
        expires=meta.get("expires", "").strip(),
        owner=meta.get("owner", "").strip(),
        shared_with=[s.strip() for s in
                     meta.get("shared_with", "").strip().strip("[]").split(",")
                     if s.strip()],
        path=path,
    )


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or f"note-{int(time.time())}"


_SECRET_RE = re.compile(
    r"\b(password|passcode|passphrase|\bpin\b|api[\s_-]?key|secret|"
    r"access[\s_-]?token|private[\s_-]?key|credential|ssn|social security|"
    r"credit[\s_-]?card|cvv|routing[\s_-]?number|account[\s_-]?number)\b", re.I)


def _looks_secret(body: str) -> bool:
    """Does this note read like it contains a credential? Used to auto-mark it
    act=never so Lara treats it as reference-only and never repeats it."""
    return bool(_SECRET_RE.search(body or ""))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #
class MemoryStore:
    def __init__(self, cfg, embedder=None, actor=None):
        self.cfg = cfg
        self.dir = Path(cfg.memory_dir)
        self.notes_dir = self.dir / "notes"
        self.index_path = self.dir / "INDEX.md"
        self.embed_path = self.dir / "embeddings.jsonl"
        self.embedder = embedder  # callable(text) -> List[float], or None
        # Whose memory is this view acting as? Defaults off cfg (set per-request
        # by the server) so every call site — pipeline, scribe, dreams, the
        # Memory tab — stays scoped without new plumbing. actor="" = a single-
        # user install (no isolation) or a background/ambient writer.
        self.actor = (actor if actor is not None
                      else getattr(cfg, "_actor", "")) or ""
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def _visible(self, note: "Note") -> bool:
        """Can the acting profile SEE this note? Memory is PRIVATE-BY-DEFAULT —
        the security boundary, enforced at recall/display time. A single-user
        install (actor="") sees everything. Otherwise: household/ambient notes
        (owner="") are the common baseline; a person's own notes are theirs; and
        a note reaches others only if its owner explicitly shared it with them
        (or with "*" = everyone)."""
        if not self.actor:
            return True
        if note.owner in ("", self.actor):
            return True
        return self.actor in note.shared_with or "*" in note.shared_with

    def _owner_for(self, type: str, act: str) -> str:
        """Who owns a NEWLY written note: the acting profile. Everything a person
        creates is private to them (no leakage unless they later share it). A
        background/ambient writer with no actor (dreams over home-state) writes
        household notes (owner="")."""
        return self.actor

    # -- writing -------------------------------------------------------- #
    def write(self, body: str, tags: Optional[List[str]] = None,
              type: str = "reference", name: Optional[str] = None,
              salience: float = 0.5, act: str = "", expires: str = "",
              owner: Optional[str] = None) -> Note:
        act = (act or "").strip().lower()
        if not act and _looks_secret(body):
            act = "never"          # auto-flag passwords/keys so Lara never blurts them
        # An explicit owner (the dream cycle attributing a person's consolidated
        # activity to them) wins; otherwise derive it from type/scope/actor.
        owner = owner if owner is not None else self._owner_for(type, act)
        # Dedupe-on-write: dreams + the scribe keep re-learning the same lesson
        # in fresh words ('foil wrap cuts 30-45 min' x2, 'rooms pausing in sync'
        # x3...). A near-duplicate should STRENGTHEN the existing note — that's
        # what repetition means — not mint another file for reflect() to
        # over-strengthen forever. Scoped to the SAME owner so one person's fact
        # never merges into another's (a cross-profile leak) or gets silently
        # strengthened by someone else repeating it.
        #
        # The whole dedup-check-through-file-creation is one critical section:
        # unlocked, two concurrent writers (scribe on a request thread + dream on
        # the mind thread) rephrasing the same fact both miss _find_duplicate and
        # mint two files (defeating dedup), or both auto-slug to the same name,
        # both pass the exists() check, and one atomic_write clobbers the other
        # (a scribe fact silently lost). _WRITE_LOCK is an RLock, so the nested
        # acquisitions in all_notes/_index_note re-enter safely. (Embedding runs
        # AFTER this block so its blocking network call never holds the lock.)
        with _WRITE_LOCK:
            dup = self._find_duplicate(body, type, owner)
            if dup is not None:
                # Re-learning IS the strengthen signal — record it in the
                # dynamics sidecar; rewriting the note file for a float was
                # exactly the churn the sidecar exists to end.
                d = dict(self._load_dyn())
                rec = dict(d.get(dup.name) or {})
                rec["sal"] = round(min(1.0, float(rec.get("sal", dup.salience)) + 0.1), 3)
                rec["last"] = time.time()
                d[dup.name] = rec
                self._save_dyn(d)
                return dup
            if not name:
                # Auto-slugged names can collide with an existing DIFFERENT note
                # (e.g. same opening words, different type — the dedupe above only
                # merges within a type). Uniquify instead of silently overwriting.
                name = base = _slug(body.split("\n", 1)[0])
                i = 2
                while (self.notes_dir / f"{name}.md").exists():
                    name = f"{base}-{i}"
                    i += 1
            note = Note(name=name, body=body, type=type,
                        tags=tags or [], salience=salience,
                        act=act, expires=(expires or "").strip(), owner=owner)
            path = self.notes_dir / f"{name}.md"
            # Atomic + read-only-tolerant, like every other ANVIL state file: a
            # note locked read-only (AV / sync tool) or a crash mid-write must not
            # lose a dream's consolidated lesson, nor truncate an existing note.
            cfgmod.atomic_write(path, note.to_markdown())
            note.path = path
            self._index_note(note)
        # Embed OUTSIDE the critical section: self.embedder(...) is a blocking
        # Ollama network round-trip. Held under _WRITE_LOCK it would stall every
        # concurrent recall() (all_notes() takes the same lock on each chat turn)
        # for the duration of a scribe/dream's embed call. _embed_note re-acquires
        # the lock only for its own atomic file append, so nothing here needs the
        # outer lock — the note file + INDEX are already committed above.
        if self.embedder and self.cfg.use_embeddings:
            self._embed_note(note)
        return note

    # Calibrated against real duplicates from a day of live use: rephrasings of
    # the same fact land at ~0.43-0.50 Jaccard ('pull ribs off grill before
    # 4:30' vs 'by 4:30 to avoid incoming rain'), while related-but-distinct
    # notes stay <= 0.38. 0.42 splits them.
    DUP_THRESHOLD = 0.42

    def _find_duplicate(self, body: str, type: str,
                        owner: str = "") -> Optional[Note]:
        """An existing note of the same type AND owner whose words substantially
        overlap the candidate (Jaccard on words of 4+ chars), else None. Scoping
        by owner keeps one family member's notes from merging into another's."""
        words = set(re.findall(r"[a-z0-9]{4,}", (body or "").lower()))
        if len(words) < 4:
            return None
        cands = self.all_notes()
        # FTS narrowing: at scale, Jaccard only the BM25 top candidates instead
        # of every note — this scan runs under _WRITE_LOCK, where recalls wait.
        if len(cands) >= int(getattr(self.cfg, "index_min_notes", 100)):
            try:
                from .index import SearchIndex
                ix = SearchIndex(self.cfg)
                ix.sync_notes(self)
                hits = {h[0] for h in ix.search_notes(body, limit=15)}
                if hits:
                    cands = [n for n in cands if n.name in hits]
            except Exception:
                pass
        best, best_score = None, 0.0
        for n in cands:
            if n.type != type or n.owner != owner:
                continue
            other = set(re.findall(r"[a-z0-9]{4,}", n.body.lower()))
            if not other:
                continue
            score = len(words & other) / max(1, len(words | other))
            if score > best_score:
                best, best_score = n, score
        return best if best_score >= self.DUP_THRESHOLD else None

    def _index_note(self, note: Note) -> None:
        # Locked: two concurrent write() calls (scribe on a request thread +
        # dream on the mind thread) both read-modify-write INDEX.md; unlocked,
        # the second write silently drops the first's entry.
        with _WRITE_LOCK:
            line = f"- [{note.name}]({note.path.name}) — {note.body[:80].strip()}"
            existing = []
            if self.index_path.exists():
                existing = [
                    ln for ln in self.index_path.read_text("utf-8", "replace").splitlines()
                    if f"]({note.path.name})" not in ln and ln.strip()
                ]
            existing.append(line)
            cfgmod.atomic_write(self.index_path, "\n".join(sorted(existing)) + "\n")

    def _embed_note(self, note: Note) -> None:
        try:
            vec = self.embedder(note.body)
        except Exception:
            return
        rec = json.dumps({"name": note.name, "vec": vec})
        with _WRITE_LOCK:
            try:
                with self.embed_path.open("a", encoding="utf-8") as fh:
                    fh.write(rec + "\n")
            except OSError:
                # Read-only/locked embeddings.jsonl (AV / sync tool) must not crash
                # the enclosing write() — the note file + INDEX are already saved.
                # Fall back to the atomic read-modify rewrite every other ANVIL
                # state writer uses so the vector still lands when at all possible.
                try:
                    prev = (self.embed_path.read_text("utf-8", "replace")
                            if self.embed_path.exists() else "")
                    cfgmod.atomic_write(self.embed_path, prev + rec + "\n")
                except OSError:
                    return

    def backfill_embeddings(self, limit: int = 50) -> int:
        """Self-healing: embed any notes that have no vector yet (written while
        the embed endpoint was down, by a store built without an embedder, or
        before embeddings existed). Idempotent, bounded per call so a dream's
        housekeeping never stalls on a huge historical backlog."""
        if not (self.embedder and self.cfg.use_embeddings):
            return 0
        have = set(self._load_embeddings())
        done = 0
        for n in self.all_notes():
            if done >= limit:
                break
            if n.name in have:
                continue
            self._embed_note(n)              # appends on success, silent on failure
            if n.name not in self._load_embeddings():
                break                        # embedder down — stop; retry next dream
            done += 1
        return done

    # -- reading -------------------------------------------------------- #
    def all_notes(self) -> List[Note]:
        """All notes, with a per-file parse cache.

        This runs on EVERY recall (each chat message), plus scribe/dream/
        consolidate. Re-reading and re-parsing every .md each time scales with
        the note count forever. A stat() per file is enough to know whether the
        cached parse is still valid — any rewrite (atomic_write replaces the
        file) changes mtime/size and misses the cache naturally."""
        out = []
        live = set()
        with _WRITE_LOCK:
            for p in sorted(self.notes_dir.glob("*.md")):
                try:
                    st = p.stat()
                    key = str(p)
                    live.add(key)
                    hit = _NOTE_CACHE.get(key)
                    if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                        out.append(hit[2])
                        continue
                    note = _parse_note(p.read_text("utf-8", "replace"), p)
                    _NOTE_CACHE[key] = (st.st_mtime_ns, st.st_size, note)
                    out.append(note)
                except Exception:
                    continue
            # Drop cache entries for deleted notes so it can't grow stale.
            for k in [k for k in _NOTE_CACHE if k not in live
                      and k.startswith(str(self.notes_dir))]:
                _NOTE_CACHE.pop(k, None)
        return out

    def visible_notes(self) -> List["Note"]:
        """Notes the acting profile may SEE — household/ambient (owner=""), their
        own, plus anything explicitly shared WITH them. Backs the Memory tab so a
        family member never browses another's private memory."""
        return [n for n in self.all_notes() if self._visible(n)]

    def delete(self, name: str, by_actor: Optional[str] = None) -> bool:
        """Forget one note. The memory-mirror right: a family member can delete
        what Lara remembers about THEM (owner guard). Removes the file, its
        INDEX entry, and its embedding. Returns True if a note was deleted."""
        for n in self.all_notes():
            if n.name != name:
                continue
            if by_actor is not None and n.owner != by_actor:
                return False                # only your own memory is yours to forget
            try:
                if n.path:
                    n.path.unlink(missing_ok=True)
                    _NOTE_CACHE.pop(str(n.path), None)
            except OSError:
                return False
            self._rebuild_index()
            self._prune_embeddings({name})
            return True
        return False

    def _prune_embeddings(self, names) -> None:
        """Drop the embedding rows for these note names. embeddings.jsonl is keyed
        by name and only ever appended to, so any path that DELETES a note file
        (delete/consolidate/reflect) must also drop its vector — else the orphaned
        row lingers and a later note re-minting the same slug (its file gone, so
        the on-disk uniquify doesn't fire) inherits a STALE vector, silently
        scoring recall against the wrong note's embedding."""
        names = {n for n in names if n}
        if not names:
            return
        try:
            if not self.embed_path.exists():
                return
            with _WRITE_LOCK:
                keep = []
                for ln in self.embed_path.read_text("utf-8", "replace").splitlines():
                    if not ln.strip():
                        continue
                    try:
                        if json.loads(ln).get("name") in names:
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass   # keep undecodable lines untouched
                    keep.append(ln)
                cfgmod.atomic_write(self.embed_path,
                                    "\n".join(keep) + ("\n" if keep else ""))
        except Exception:
            pass

    def share(self, name: str, with_list: List[str],
              by_actor: Optional[str] = None) -> Optional["Note"]:
        """Set who a note is shared with. Only its OWNER may reshare it
        (``by_actor`` guard) — you can't share memory that isn't yours. Pass
        ``["*"]`` for everyone, ``[]`` to make it private again. Returns the
        updated Note, or None if not found / not permitted."""
        for n in self.all_notes():
            if n.name != name:
                continue
            if by_actor is not None and n.owner != by_actor:
                return None                 # not the owner — refuse
            clean, seen = [], set()
            for w in with_list:
                w = str(w or "").strip()
                if w and w != n.owner and w not in seen:
                    seen.add(w)
                    clean.append(w)
            n.shared_with = clean
            if n.path:
                cfgmod.atomic_write(n.path, n.to_markdown())
                _NOTE_CACHE.pop(str(n.path), None)   # bust the parse cache
            return n
        return None

    def index_text(self) -> str:
        return self.index_path.read_text("utf-8", "replace") if self.index_path.exists() else ""

    def _load_embeddings(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        if self.embed_path.exists():
            key = str(self.embed_path)
            try:
                st = self.embed_path.stat()
                hit = _EMBED_CACHE.get(key)
                if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                    return hit[2]
            except OSError:
                st = None
            for line in self.embed_path.read_text("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                # Tolerate corruption like all_notes()/STM.all() do: a single
                # truncated or malformed line (interrupted append, AV / sync-tool
                # lock) must not crash recall() — skip it and keep the rest.
                try:
                    rec = json.loads(line)
                    out[rec["name"]] = rec["vec"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
            if st is not None:
                _EMBED_CACHE[key] = (st.st_mtime_ns, st.st_size, out)
        return out

    # -- recall --------------------------------------------------------- #
    # ---- salience dynamics sidecar (review 2.1) ------------------------- #
    # Machine-owned scalars (activation, last access, use count) live in ONE
    # sidecar file, not in every note's frontmatter: the old design rewrote the
    # ENTIRE store twice per dream just to nudge floats — O(n) daily write
    # amplification that invalidated the parse cache, churned mtimes (breaking
    # the git/backup story files were chosen FOR), and decayed by dream-count,
    # so how fast Lara forgot depended on how busy the house was. Decay is now
    # wall-clock arithmetic at READ time; the only write is a touch on recall.
    def _dyn_path(self) -> Path:
        return self.notes_dir / "salience.json"

    def _load_dyn(self) -> Dict[str, dict]:
        p = self._dyn_path()
        try:
            st = p.stat()
            key = str(p)
            c = _DYN_CACHE.get(key)
            if c and c[0] == st.st_mtime_ns and c[1] == st.st_size:
                return c[2]
            d = json.loads(p.read_text("utf-8"))
            d = d if isinstance(d, dict) else {}
            _DYN_CACHE[key] = (st.st_mtime_ns, st.st_size, d)
            return d
        except Exception:
            return {}

    def _save_dyn(self, d: Dict[str, dict]) -> None:
        try:
            with _WRITE_LOCK:
                cfgmod.atomic_write(self._dyn_path(), json.dumps(d))
        except Exception:
            pass

    def touch(self, names) -> None:
        """Record that these notes were actually USED (packed into a prompt by
        recall). This is the usage signal the lifecycle never had — before,
        ranking read salience but nothing ever fed it back."""
        names = [n for n in names if n]
        if not names:
            return
        d = dict(self._load_dyn())
        now = time.time()
        for name in names:
            rec = dict(d.get(name) or {})
            rec["sal"] = round(min(1.0, float(rec.get("sal", 0.5)) + 0.05), 3)
            rec["last"] = now
            rec["uses"] = int(rec.get("uses", 0)) + 1
            d[name] = rec
        self._save_dyn(d)

    def eff_salience(self, note: Note, dyn: Dict[str, dict], now: float) -> float:
        """Activation at READ time: the stored value decayed by wall-clock age
        since the note was last used (or created). ACT-R-shaped: no write is
        needed for forgetting — it is just arithmetic on timestamps."""
        rec = dyn.get(note.name) or {}
        base = float(rec.get("sal", note.salience))
        last = rec.get("last")
        if last is None:
            try:
                last = time.mktime(time.strptime(note.created, "%Y-%m-%d"))
            except Exception:
                last = now
        half = max(1.0, float(getattr(self.cfg, "salience_half_life_days", 14.0)))
        age_d = max(0.0, now - float(last)) / 86400.0
        return base * (0.5 ** (age_d / half))

    def recall(self, query: str, budget: Optional[int] = None,
               query_tags: Optional[List[str]] = None) -> List[Note]:
        """Rank notes for ``query`` and pack the best into the token budget."""
        budget = budget or self.cfg.note_token_budget
        # Drop time-expired facts ("away until Friday") so a stale note never
        # gets recalled as if it were still true. VISIBILITY FILTER FIRST: a note
        # owned by another family member is never a recall candidate — this is
        # the privacy boundary, enforced before any ranking can surface it.
        today = date.today().isoformat()
        notes = [n for n in self.all_notes()
                 if self._visible(n) and not n.is_expired(today)]
        if not notes:
            return []
        query_tags = set(t.lower() for t in (query_tags or []))
        q_terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        now = time.time()

        qvec = None
        embeds = {}
        if self.embedder and self.cfg.use_embeddings:
            try:
                qvec = self.embedder(query)
                embeds = self._load_embeddings()
            except Exception:
                qvec = None

        # "About you" profile notes load first (identity context) — but only the
        # most salient few, not ALL of them (a growing store was flooding every
        # answer with the operator's entire biography).
        dyn = self._load_dyn()
        max_profiles = int(getattr(self.cfg, "recall_max_profile", 8))
        profiles = sorted([n for n in notes if n.type == "profile"],
                          key=lambda n: -self.eff_salience(n, dyn, now))[:max_profiles]
        others = [n for n in notes if n.type != "profile"]
        # FTS candidate narrowing (index.py): above the threshold, BM25 picks
        # the top candidates in ~1ms and the scoring loop below only touches
        # those — recall stops scaling with how much Lara remembers. Below the
        # threshold (or if the index is unavailable) the full scan runs exactly
        # as before; the files stay the source of truth either way.
        if len(others) >= int(getattr(self.cfg, "index_min_notes", 100)):
            try:
                from .index import SearchIndex
                ix = SearchIndex(self.cfg)
                ix.sync_notes(self)
                hits = {h[0] for h in ix.search_notes(query, limit=60)}
                if hits:
                    others = [n for n in others if n.name in hits]
            except Exception:
                pass                      # index trouble -> old full scan
        always = context.pack_to_budget(
            [(1.0, n.body, n) for n in profiles], max(1, budget // 2))
        # RELEVANCE FLOOR: without it, a note sharing ZERO words with the query
        # still scores on salience+recency and packs — and ~150 tangential
        # fragments per answer is a confabulation machine (the model weaves
        # unrelated memories into fictional 'projects'). A note must show actual
        # evidence of relevance (keyword/tag/semantic) to be recalled at all,
        # and only the best few make it in.
        max_notes = int(getattr(self.cfg, "recall_max_notes", 10))
        scored = []
        for n in others:
            # Term sets ride the note parse cache: re-tokenizing every body on
            # every turn was the biggest per-turn cost at scale (review 2.2).
            n_terms = getattr(n, "_terms", None)
            if n_terms is None:
                n_terms = set(re.findall(r"[a-z0-9]{3,}", n.body.lower()))
                try:
                    n._terms = n_terms
                except Exception:
                    pass
            kw_hits = len(q_terms & n_terms)
            tag_hits = len(query_tags & set(t.lower() for t in n.tags))
            sem = 0.0
            if qvec is not None and n.name in embeds:
                sem = _cosine(qvec, embeds[n.name])
            if kw_hits < 1 and tag_hits < 1 and sem < 0.35:
                continue
            scored.append((self._score(n, q_terms, query_tags, sem, n_terms, now,
                                       self.eff_salience(n, dyn, now)),
                           n.body, n))
        scored.sort(key=lambda t: -t[0])
        packed = context.pack_to_budget(scored[:max_notes],
                                        budget - always.used_tokens)
        included = always.included + packed.included
        self.touch([n.name for n in included])   # inclusion IS the usage signal
        return included

    def _score(self, note: Note, q_terms, query_tags, sem, n_terms, now,
               eff_sal: Optional[float] = None) -> float:
        # Semantic similarity (0..1) and note tokens are computed once by the
        # recall() candidate loop and passed in — recomputing them here would
        # tokenize the note body and iterate the full embedding vector twice per
        # candidate on every chat turn.
        # Keyword overlap fallback / supplement.
        kw = len(q_terms & n_terms) / (len(q_terms) + 1)
        # Tag overlap.
        tag = len(query_tags & set(t.lower() for t in note.tags)) * 0.5
        # Recency: decay over ~30 days.
        try:
            age_days = (now - time.mktime(time.strptime(note.created, "%Y-%m-%d"))) / 86400
        except Exception:
            age_days = 30
        recency = math.exp(-age_days / 30.0)
        # Exact-literal boost (Khoj hybrid recall): a DISTINCTIVE query token
        # appearing VERBATIM in the note is strong evidence — model numbers, med
        # names, IDs, error codes shouldn't whiff just because their embedding is
        # average. Distinctive = contains a digit, or is long (>=6 chars).
        exact = 0.0
        low = note.body.lower()
        for t in q_terms:
            if (any(c.isdigit() for c in t) or len(t) >= 6) and t in low:
                exact += 1.0
        sal = note.salience if eff_sal is None else eff_sal
        return ((2.0 * sem) + (1.5 * kw) + tag + (0.5 * sal)
                + (0.3 * recency) + (1.5 * min(exact, 2.0)))

    # -- maintenance ---------------------------------------------------- #
    def consolidate(self, decay: float = 0.95) -> Dict[str, int]:
        """Drop near-empty notes and rebuild the index. Salience decay is
        wall-clock arithmetic at READ time now (eff_salience) — the old
        per-dream multiply rewrote EVERY note file twice a night to nudge a
        float, defeating the parse cache, churning git/backup state, and tying
        the forgetting rate to how often the house dreams instead of to time."""
        notes = self.all_notes()
        removed = 0
        dropped: set = set()
        for n in notes:
            if not n.body.strip():
                if n.path:
                    n.path.unlink(missing_ok=True)
                    _NOTE_CACHE.pop(str(n.path), None)
                dropped.add(n.name)
                removed += 1
        if dropped:
            d = {k: v for k, v in self._load_dyn().items() if k not in dropped}
            self._save_dyn(d)
        self._prune_embeddings(dropped)   # deleted notes must lose their vectors too
        self._rebuild_index()
        return {"notes": len(notes), "removed": removed}

    def reflect(self, recent_texts: List[str], floor: float = 0.08) -> Dict[str, int]:
        """Sleep reconsolidation over LTM itself.

        Human sleep doesn't just file the day's events — it re-weights existing
        long-term memory: what proved relevant is strengthened, what has faded is
        let go. Here that's salience-only (no content rewriting, which is known to
        corrupt memories over repeated LLM edits), so it's safe and deterministic:

        * A note whose words overlap recent activity gets its salience boosted
          (reconsolidation — 'use it or lose it', used = kept).
        * A note that has decayed below ``floor`` and wasn't touched is forgotten.
        * ``profile`` notes (durable facts about the operator) are never forgotten.
        """
        terms: set = set()
        for t in recent_texts:
            terms |= set(re.findall(r"[a-z0-9]{4,}", (t or "").lower()))
        strengthened = forgotten = 0
        dropped: set = set()
        dyn = dict(self._load_dyn())
        now = time.time()
        keep_days = max(1.0, float(getattr(self.cfg, "forget_unused_days", 21.0)))
        for n in self.all_notes():
            body_terms = set(re.findall(r"[a-z0-9]{4,}", (n.body or "").lower()))
            if terms & body_terms:
                rec = dict(dyn.get(n.name) or {})
                rec["sal"] = round(min(1.0, float(rec.get("sal", n.salience)) + 0.15), 3)
                dyn[n.name] = rec              # sidecar only — the FILE never churns
                strengthened += 1
                continue
            if n.type == "profile" or not n.path:
                continue
            # Forgetting is irreversible, so it takes BOTH: activation decayed
            # below the floor AND no actual recall/use within the window. The
            # old rule deleted on the noisy overlap signal alone — a fact like
            # "the EpiPen expires 2026-09" died in days just for being quiet.
            rec = dyn.get(n.name) or {}
            last = rec.get("last")
            if last is None:
                try:
                    last = time.mktime(time.strptime(n.created, "%Y-%m-%d"))
                except Exception:
                    last = now
            unused_d = (now - float(last)) / 86400.0
            if self.eff_salience(n, dyn, now) < floor and unused_d > keep_days:
                try:
                    n.path.unlink(missing_ok=True)
                    _NOTE_CACHE.pop(str(n.path), None)
                    dyn.pop(n.name, None)
                    dropped.add(n.name)
                    forgotten += 1
                except OSError:
                    continue
        self._save_dyn(dyn)
        self._prune_embeddings(dropped)   # forgotten notes must lose their vectors too
        self._rebuild_index()
        return {"strengthened": strengthened, "forgotten": forgotten}

    def _rebuild_index(self) -> None:
        # Locked like _index_note: consolidate()/reflect()/delete() rebuild
        # INDEX.md while a scribe write() on a request thread may be doing a
        # read-modify-write of the same file via _index_note. Unlocked, the two
        # INDEX.md writers interleave and the rebuild (snapshotting all_notes()
        # before the fresh note landed) clobbers the just-appended entry —
        # leaving a note on disk with no INDEX map row (invisible to recall's
        # always-loaded index). RLock, so a caller already holding it re-enters.
        with _WRITE_LOCK:
            lines = []
            for n in self.all_notes():
                if n.path:
                    lines.append(f"- [{n.name}]({n.path.name}) — {n.body[:80].strip()}")
            cfgmod.atomic_write(self.index_path, "\n".join(sorted(lines)) + "\n")
