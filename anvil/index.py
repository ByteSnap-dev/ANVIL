"""The search index — SQLite FTS5 over ANVIL's file stores, plus the house
event history. One file: ``memory/index.db``.

DESIGN CONTRACT (review 2.2 follow-up): for text search, the Markdown/JSONL
files remain the SOURCE OF TRUTH and this database is a DERIVED, DISPOSABLE
cache — delete it and nothing is lost; it rebuilds from the files (mtime-diff
incremental, full rebuild when missing). That keeps memory greppable, git-able
and human-editable while retrieval stops scaling with its size: an inverted
index answers "which notes contain these words" in ~1ms whether there are 400
notes or 40,000, and BM25's IDF weighting finally makes a rare word (EpiPen)
outrank a common one (kitchen) — the thing bare set-intersection never could.

The ONE exception to "derived": the ``events`` table (house state transitions)
is a primary, append-only store. It is machine-generated time-series that no
human ever hand-edits, so the files-first rationale doesn't apply — and SQL
aggregates over it give dreams/heartbeats GROUNDED pattern evidence ("front
door opened after midnight 3x this week" from a real count, not a model's
imagination).

Every method degrades gracefully: no FTS5, a locked file, a corrupt db — the
callers fall back to their old full-scan paths and the harness keeps working.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_LOCK = threading.Lock()
_CONNS: Dict[str, sqlite3.Connection] = {}   # one connection per db path

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    name UNINDEXED, owner UNINDEXED, type UNINDEXED, body);
CREATE TABLE IF NOT EXISTS notes_meta(
    name TEXT PRIMARY KEY, mtime_ns INTEGER, owner TEXT, type TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    chunk_id UNINDEXED, source UNINDEXED, text);
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    sid UNINDEXED, owner UNINDEXED, turn_idx UNINDEXED, role UNINDEXED, content);
CREATE TABLE IF NOT EXISTS turns_meta(
    key TEXT PRIMARY KEY, mtime_ns INTEGER);
CREATE VIRTUAL TABLE IF NOT EXISTS journal_fts USING fts5(line_no UNINDEXED, text);
CREATE TABLE IF NOT EXISTS journal_meta(k TEXT PRIMARY KEY, v INTEGER);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, entity TEXT, old TEXT, new TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS events_entity ON events(entity, ts);
"""


def _match_query(text: str, max_terms: int = 12) -> str:
    """A safe FTS5 MATCH expression: each term quoted (no operator injection),
    OR-joined so any overlap ranks — BM25 sorts by how MUCH overlaps."""
    terms = re.findall(r"[a-z0-9]{3,}", (text or "").lower())[:max_terms]
    return " OR ".join(f'"{t}"' for t in terms)


class SearchIndex:
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = Path(getattr(cfg, "memory_dir", "memory")) / "index.db"
        self.ok = False
        try:
            self.db = self._connect()
            self.ok = True
        except Exception:
            self.db = None                # callers fall back to full scans

    def _connect(self) -> sqlite3.Connection:
        key = str(self.path)
        with _LOCK:
            conn = _CONNS.get(key)
            if conn is not None:
                return conn
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(key, check_same_thread=False, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")     # readers never block writer
            conn.executescript(_SCHEMA)
            _CONNS[key] = conn
            return conn

    # ---- notes (derived from memory/notes/*.md) --------------------------- #
    def sync_notes(self, store) -> int:
        """Incremental mtime-diff against the note files; returns rows changed.
        Cheap when nothing changed (one scandir + one small SELECT)."""
        if not self.ok:
            return 0
        try:
            with _LOCK:
                have = dict(self.db.execute(
                    "SELECT name, mtime_ns FROM notes_meta"))
            changed = 0
            seen = set()
            for n in store.all_notes():
                if not n.path:
                    continue
                seen.add(n.name)
                try:
                    mt = n.path.stat().st_mtime_ns
                except OSError:
                    continue
                if have.get(n.name) == mt:
                    continue
                with _LOCK:
                    self.db.execute("DELETE FROM notes_fts WHERE name=?", (n.name,))
                    self.db.execute(
                        "INSERT INTO notes_fts(name, owner, type, body) VALUES (?,?,?,?)",
                        (n.name, n.owner or "", n.type, n.body))
                    self.db.execute(
                        "INSERT OR REPLACE INTO notes_meta VALUES (?,?,?,?)",
                        (n.name, mt, n.owner or "", n.type))
                changed += 1
            gone = set(have) - seen
            if gone:
                with _LOCK:
                    for name in gone:
                        self.db.execute("DELETE FROM notes_fts WHERE name=?", (name,))
                        self.db.execute("DELETE FROM notes_meta WHERE name=?", (name,))
                changed += len(gone)
            if changed:
                with _LOCK:
                    self.db.commit()
            return changed
        except Exception:
            return 0

    def search_notes(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        """BM25-ranked candidate note NAMES (best first). Visibility/expiry
        stay the caller's job — this only narrows the candidate set."""
        if not self.ok:
            return []
        q = _match_query(query)
        if not q:
            return []
        try:
            with _LOCK:
                rows = self.db.execute(
                    "SELECT name, bm25(notes_fts) FROM notes_fts "
                    "WHERE notes_fts MATCH ? ORDER BY bm25(notes_fts) LIMIT ?",
                    (q, limit)).fetchall()
            return [(r[0], float(r[1])) for r in rows]
        except Exception:
            return []

    # ---- family docs (derived; DocStore hands us its chunks) -------------- #
    def sync_docs(self, chunks: List[dict]) -> int:
        """Full refresh from the DocStore's chunk list (it already owns
        incremental re-chunking; we just mirror the result)."""
        if not self.ok:
            return 0
        try:
            with _LOCK:
                self.db.execute("DELETE FROM docs_fts")
                for c in chunks:
                    self.db.execute(
                        "INSERT INTO docs_fts(chunk_id, source, text) VALUES (?,?,?)",
                        (str(c.get("id", "")), str(c.get("source", "")),
                         str(c.get("text", ""))[:8000]))
                self.db.commit()
            return len(chunks)
        except Exception:
            return 0

    def search_docs(self, query: str, limit: int = 20) -> List[Tuple[str, str, float]]:
        if not self.ok:
            return []
        q = _match_query(query)
        if not q:
            return []
        try:
            with _LOCK:
                rows = self.db.execute(
                    "SELECT chunk_id, source, bm25(docs_fts) FROM docs_fts "
                    "WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?",
                    (q, limit)).fetchall()
            return [(r[0], r[1], float(r[2])) for r in rows]
        except Exception:
            return []

    # ---- conversation turns (derived from conversations/**.jsonl) --------- #
    def sync_turns(self, conv_dir: Path) -> int:
        """Mtime-diff over transcript files (all owners' subdirs included —
        search results are owner-filtered at query time)."""
        if not self.ok:
            return 0
        changed = 0
        try:
            with _LOCK:
                have = dict(self.db.execute("SELECT key, mtime_ns FROM turns_meta"))
            files = list(Path(conv_dir).glob("*.jsonl")) + \
                list(Path(conv_dir).glob("u_*/*.jsonl"))
            seen = set()
            for p in files:
                owner = p.parent.name[2:] if p.parent.name.startswith("u_") else ""
                key = f"{owner}/{p.stem}"
                seen.add(key)
                try:
                    mt = p.stat().st_mtime_ns
                except OSError:
                    continue
                if have.get(key) == mt:
                    continue
                turns = []
                for i, line in enumerate(
                        p.read_text("utf-8", "replace").splitlines()):
                    try:
                        r = json.loads(line)
                        if r.get("role") in ("user", "assistant"):
                            turns.append((i, r["role"], str(r.get("content", ""))[:4000]))
                    except Exception:
                        continue
                with _LOCK:
                    self.db.execute("DELETE FROM turns_fts WHERE sid=? AND owner=?",
                                    (p.stem, owner))
                    for i, role, content in turns:
                        self.db.execute(
                            "INSERT INTO turns_fts(sid, owner, turn_idx, role, content) "
                            "VALUES (?,?,?,?,?)", (p.stem, owner, i, role, content))
                    self.db.execute("INSERT OR REPLACE INTO turns_meta VALUES (?,?)",
                                    (key, mt))
                changed += 1
            gone = set(have) - seen
            if gone:
                with _LOCK:
                    for key in gone:
                        owner, _, sid = key.partition("/")
                        self.db.execute("DELETE FROM turns_fts WHERE sid=? AND owner=?",
                                        (sid, owner))
                        self.db.execute("DELETE FROM turns_meta WHERE key=?", (key,))
                changed += len(gone)
            if changed:
                with _LOCK:
                    self.db.commit()
        except Exception:
            return changed
        return changed

    def search_turns(self, query: str, owner: str = "",
                     limit: int = 12) -> List[dict]:
        """Ranked turns for ONE owner — the privacy boundary is enforced here,
        exactly like note visibility: another member's chats never surface."""
        if not self.ok:
            return []
        q = _match_query(query)
        if not q:
            return []
        try:
            with _LOCK:
                rows = self.db.execute(
                    "SELECT sid, turn_idx, role, content, bm25(turns_fts) "
                    "FROM turns_fts WHERE turns_fts MATCH ? AND owner=? "
                    "ORDER BY bm25(turns_fts) LIMIT ?", (q, owner or "", limit)
                ).fetchall()
            return [{"sid": r[0], "idx": r[1], "role": r[2],
                     "content": r[3], "rank": float(r[4])} for r in rows]
        except Exception:
            return []

    # ---- journal (derived from memory/journal.md) -------------------------- #
    def sync_journal(self, journal_path: Path) -> int:
        """The journal is one capped file that only appends (until the cap
        trims it) — reindex whole when its mtime moves. Small by construction."""
        if not self.ok:
            return 0
        try:
            p = Path(journal_path)
            if not p.exists():
                return 0
            mt = p.stat().st_mtime_ns
            with _LOCK:
                row = self.db.execute(
                    "SELECT v FROM journal_meta WHERE k='mtime'").fetchone()
            if row and row[0] == mt:
                return 0
            lines = [(i, ln) for i, ln in
                     enumerate(p.read_text("utf-8", "replace").splitlines())
                     if ln.strip()]
            with _LOCK:
                self.db.execute("DELETE FROM journal_fts")
                for i, ln in lines:
                    self.db.execute(
                        "INSERT INTO journal_fts(line_no, text) VALUES (?,?)",
                        (i, ln[:2000]))
                self.db.execute(
                    "INSERT OR REPLACE INTO journal_meta VALUES ('mtime', ?)", (mt,))
                self.db.commit()
            return len(lines)
        except Exception:
            return 0

    def search_journal(self, query: str, limit: int = 10) -> List[str]:
        if not self.ok:
            return []
        q = _match_query(query)
        if not q:
            return []
        try:
            with _LOCK:
                rows = self.db.execute(
                    "SELECT text FROM journal_fts WHERE journal_fts MATCH ? "
                    "ORDER BY bm25(journal_fts) LIMIT ?", (q, limit)).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    # ---- house events (PRIMARY store — machine time-series) ---------------- #
    def record_event(self, entity: str, old: str, new: str) -> None:
        """Append one state transition. Never raises — sensing must not break."""
        if not self.ok:
            return
        try:
            with _LOCK:
                self.db.execute(
                    "INSERT INTO events(ts, entity, old, new) VALUES (?,?,?,?)",
                    (time.time(), str(entity)[:120], str(old)[:80], str(new)[:80]))
                # Retention: bounded by construction, oldest-first.
                self.db.execute(
                    "DELETE FROM events WHERE id <= ("
                    " SELECT id FROM events ORDER BY id DESC LIMIT 1 OFFSET 50000)")
                self.db.commit()
        except Exception:
            pass

    def house_patterns(self, days: float = 7.0, top: int = 8) -> str:
        """GROUNDED activity summary for dreams/thoughts: real SQL counts, so a
        'pattern' Lara mentions traces to actual transitions — the structural
        answer to fabricated observations. Empty string when nothing recorded."""
        if not self.ok:
            return ""
        try:
            since = time.time() - days * 86400
            with _LOCK:
                rows = self.db.execute(
                    "SELECT entity, count(*) FROM events WHERE ts > ? "
                    "GROUP BY entity ORDER BY count(*) DESC LIMIT ?",
                    (since, top)).fetchall()
                night = self.db.execute(
                    "SELECT entity, count(*) FROM events WHERE ts > ? AND "
                    "CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INT) < 5 "
                    "GROUP BY entity ORDER BY count(*) DESC LIMIT 3",
                    (since,)).fetchall()
            if not rows:
                return ""
            parts = [f"{e}: {c} change(s)" for e, c in rows]
            if night:
                parts.append("overnight (midnight-5am): "
                             + ", ".join(f"{e} x{c}" for e, c in night))
            return (f"MEASURED house activity, last {days:g} days "
                    f"(from the event log, real counts): " + "; ".join(parts))
        except Exception:
            return ""
