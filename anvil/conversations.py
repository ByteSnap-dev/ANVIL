"""Per-conversation short-term memory — the turns of the *current* chat.

Distinct from the long-term note store (``memory.py``, cross-session facts/lessons)
and STM (``mind.py``, the mind's own event stream). This is the running transcript
of a single conversation with the operator, so Lara actually remembers what was
just said instead of answering each message cold.

Stored one JSONL file per session under ``conversations/`` (git-ignored runtime
state, like ``memory/``): one line per turn ``{role, content, ts}``. The prompt
budget/compaction is handled at recall time by the pipeline (recent turns verbatim,
older ones summarized) — here we just persist the raw turns, capped on disk so a
long-running chat can't grow without bound.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List

from . import config as cfgmod

# Serialize writers: two request threads appending to the same transcript could
# race append vs. the _cap() read-trim-rewrite and drop a fresh turn. One
# process-wide lock is plenty at this scale (single operator, short writes).
_WRITE_LOCK = threading.Lock()


def _safe_sid(sid: str) -> str:
    """Sanitize a client session id into a safe filename (no path traversal)."""
    s = re.sub(r"[^A-Za-z0-9_-]", "", str(sid or ""))[:64]
    return s or "default"


def _safe_owner(owner: str) -> str:
    """Sanitize a profile name into a safe subdirectory component."""
    return re.sub(r"[^A-Za-z0-9_-]", "", str(owner or ""))[:48]


def when_stamp(ts: float, now: float | None = None) -> str:
    """A compact human 'when this was said' for prompt views: '2:41 PM' today,
    'Wed 2:41 PM' within the week, 'Jul 3, 9:04 AM' beyond (year added if it
    differs). Paired with the live clock in _date_context(), this is how the
    model tells this morning from five minutes ago."""
    from datetime import datetime
    now = time.time() if now is None else now
    d, n = datetime.fromtimestamp(ts), datetime.fromtimestamp(now)
    clock = d.strftime("%I:%M %p").lstrip("0")
    if d.date() == n.date():
        return clock
    if 0 <= (n.date() - d.date()).days < 7:
        return d.strftime("%a ") + clock
    day = d.strftime("%b ") + str(d.day)
    if d.year != n.year:
        day += " " + str(d.year)
    return day + ", " + clock


class Conversations:
    """Per-conversation transcript store, ISOLATED per family profile.

    ``owner`` is the acting profile's name. When set, every transcript, the
    sidebar list, search, and titles live under ``conversations/u_<owner>/`` —
    so one family member can never see, search, or open another's chats, and a
    guessed sid from another profile simply isn't found. ``owner=""`` (a single-
    user install with no login) keeps the flat root dir, exactly as before.

    ``owner`` defaults to ``cfg._actor`` (set per-request by the server/pipeline)
    so code paths that only carry ``cfg`` — notably the cross-conversation memory
    search tool — stay scoped to the right person without new plumbing.
    """

    def __init__(self, cfg, owner=None):
        base = Path(getattr(cfg, "conversations_dir", "conversations"))
        own = _safe_owner(owner if owner is not None
                          else getattr(cfg, "_actor", ""))
        self.owner = own
        self.dir = (base / ("u_" + own)) if own else base
        self.cap = int(getattr(cfg, "conv_disk_cap", 400))

    def _path(self, sid: str) -> Path:
        return self.dir / (_safe_sid(sid) + ".jsonl")

    def history(self, sid: str) -> List[Dict[str, str]]:
        """The conversation's turns as [{role, content, ts?}], oldest first.
        ``ts`` (epoch seconds — every persisted turn carries one) rides along
        so packed()/search_chats can stamp their prompt views with WHEN each
        thing was said; the model has no other way to place events in time."""
        p = self._path(sid)
        if not p.exists():
            return []
        out: List[Dict[str, str]] = []
        try:
            lines = p.read_text("utf-8", "replace").splitlines()
        except OSError:
            return []
        for line in lines:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                role, content = r.get("role"), r.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    turn: Dict[str, str] = {"role": role, "content": content}
                    if isinstance(r.get("ts"), (int, float)):
                        turn["ts"] = r["ts"]
                    if isinstance(r.get("meta"), dict):
                        turn["meta"] = r["meta"]   # how-the-answer-happened, for the UI
                    out.append(turn)
            except (ValueError, json.JSONDecodeError):
                continue
        return out

    # -- rolling summary (review 1.2) ------------------------------------ #
    # A DURABLE, incrementally-updated summary of the conversation's older
    # turns. Before this, every turn of a long chat re-summarized the whole
    # head with a blocking model call on the hot path and threw the result
    # away; and the disk cap deleted old turns UN-summarized (facts gone).
    # Now: {"summary": str, "covered": n} means turns[:n] are folded into
    # `summary`. The hot path just reads it; a background pass advances it.
    def _roll_path(self, sid: str) -> Path:
        return self.dir / (_safe_sid(sid) + ".roll.json")

    def rolling(self, sid: str) -> Dict[str, object]:
        try:
            d = json.loads(self._roll_path(sid).read_text("utf-8"))
            if isinstance(d, dict):
                return {"summary": str(d.get("summary", "")),
                        "covered": max(0, int(d.get("covered", 0)))}
        except Exception:
            pass
        return {"summary": "", "covered": 0}

    def set_rolling(self, sid: str, summary: str, covered: int) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            cfgmod.atomic_write(self._roll_path(sid), json.dumps(
                {"summary": str(summary)[:8000], "covered": max(0, int(covered))}))
        except Exception:
            pass

    def packed(self, sid: str, preamble: str = "") -> List[Dict[str, str]]:
        """The prompt-ready view: one summary pseudo-turn (when one exists)
        followed by the verbatim turns it doesn't cover. Zero model calls.
        User turns get a "[Wed 2:41 PM]" stamp (prompt view ONLY — disk and UI
        stay clean) so the model can tell this morning from five minutes ago;
        _date_context() tells it what the stamps mean."""
        turns = self.history(sid)
        roll = self.rolling(sid)
        covered = min(int(roll["covered"]), len(turns))
        now = time.time()
        view = []
        for t in turns:
            ts = t.get("ts")
            if t["role"] == "user" and isinstance(ts, (int, float)):
                view.append({"role": "user",
                             "content": "[" + when_stamp(float(ts), now) + "] "
                                        + t["content"]})
            else:
                view.append({"role": t["role"], "content": t["content"]})
        if not roll["summary"] or covered <= 0:
            return view
        tail = view[covered:]
        role = "assistant" if (tail and tail[0].get("role") == "user") else "user"
        head = {"role": role,
                "content": (preamble + "\n" if preamble else "") + str(roll["summary"])}
        return [head] + tail

    def append(self, sid: str, role: str, content: str, meta=None) -> None:
        if role not in ("user", "assistant") or not (content or "").strip():
            return
        p = self._path(sid)
        row = {"role": role, "content": content, "ts": time.time()}
        if isinstance(meta, dict) and meta:
            # "How the answer happened" (steps/trace/rung) so the transcript UI
            # can rebuild the transparency views after a reload. Hard-capped:
            # a runaway meta must never bloat the transcript file.
            blob = json.dumps(meta, default=str)
            if len(blob) <= 60_000:
                row["meta"] = meta
        rec = json.dumps(row)
        with _WRITE_LOCK:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                if not p.exists():          # brand-new conversation
                    self._prune_files()     # every turn is capped; cap COUNT too
                with p.open("a", encoding="utf-8") as fh:
                    fh.write(rec + "\n")
                    # fsync per turn (DreamEngine's crash-safety refinement): on a
                    # power cut a buffered write is lost; flushing bounds the loss
                    # to nothing. Cheap at household message rates. Best-effort —
                    # a filesystem that can't fsync must not break a chat turn.
                    try:
                        fh.flush()
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
            except OSError:
                # Read-only lock (AV / sync tool): fall back to the atomic rewrite
                # so a single locked transcript never crashes a chat turn.
                try:
                    prev = p.read_text("utf-8", "replace") if p.exists() else ""
                    cfgmod.atomic_write(p, prev + rec + "\n")
                except OSError:
                    return
            self._cap(sid)

    def _cap(self, sid: str) -> None:
        """Keep roughly the most recent ``cap`` turns on disk — but NEVER drop a
        turn the rolling summary hasn't folded in yet (that deleted facts
        permanently). Only summary-covered turns are trimmed; if the summarizer
        is behind, the file may briefly exceed the cap until it catches up."""
        p = self._path(sid)
        try:
            raw = p.read_bytes()
        except OSError:
            return
        if raw.count(b"\n") <= self.cap:
            return
        lines = raw.decode("utf-8", "replace").splitlines()
        overflow = len(lines) - self.cap
        if overflow <= 0:
            return
        roll = self.rolling(sid)
        drop = min(overflow, int(roll["covered"]))
        if drop <= 0:
            return                        # nothing safely droppable yet
        cfgmod.atomic_write(p, "\n".join(lines[drop:]) + "\n")
        self.set_rolling(sid, str(roll["summary"]), int(roll["covered"]) - drop)

    # -- chat management (sidebar) -------------------------------------- #
    def _meta_path(self) -> Path:
        return self.dir / "_meta.json"

    def _load_meta(self) -> Dict[str, dict]:
        try:
            m = json.loads(self._meta_path().read_text("utf-8"))
            return m if isinstance(m, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_meta(self, meta: Dict[str, dict]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            cfgmod.atomic_write(self._meta_path(), json.dumps(meta))
        except OSError:
            pass

    def set_title(self, sid: str, title: str) -> None:
        with _WRITE_LOCK:
            meta = self._load_meta()
            meta.setdefault(_safe_sid(sid), {})["title"] = (title or "").strip()[:80]
            self._save_meta(meta)

    def set_pinned(self, sid: str, pinned: bool) -> None:
        with _WRITE_LOCK:
            meta = self._load_meta()
            meta.setdefault(_safe_sid(sid), {})["pinned"] = bool(pinned)
            self._save_meta(meta)

    def list(self) -> List[Dict]:
        """All conversations for the sidebar: pinned first, then newest.
        Title = custom (renamed) title, else the first user message trimmed."""
        meta = self._load_meta()
        out = []
        for p in self.dir.glob("*.jsonl"):
            sid = p.stem
            m = meta.get(sid, {})
            title = (m.get("title") or "").strip()
            turns = self.history(sid)
            if not title:
                first = next((t["content"] for t in turns if t["role"] == "user"), "")
                title = " ".join(first.split())[:48] or "New chat"
            try:
                ts = p.stat().st_mtime
            except OSError:
                ts = 0
            out.append({"sid": sid, "title": title, "ts": ts,
                        "turns": len(turns), "pinned": bool(m.get("pinned"))})
        out.sort(key=lambda c: (not c["pinned"], -c["ts"]))
        return out

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """FULL-TEXT search across every transcript — the documented weak spot
        of the big three UIs (title-only / misses mid-thread content). Local
        files make this trivial: return the chat plus the matching line."""
        q = (query or "").strip().lower()
        if not q:
            return []
        hits = []
        for c in self.list():
            for t in self.history(c["sid"]):
                if q in t["content"].lower():
                    i = t["content"].lower().find(q)
                    lo = max(0, i - 40)
                    c["snippet"] = ("…" if lo else "") + \
                        " ".join(t["content"][lo:i + len(q) + 60].split())
                    hits.append(c)
                    break
            if len(hits) >= limit:
                break
        return hits

    def _prune_files(self, keep: int = 200) -> None:
        """Cap the NUMBER of transcripts: every 'New chat' makes a fresh file
        that would otherwise accumulate forever. Called under _WRITE_LOCK when
        a new conversation starts; keeps the most recently touched files.
        PINNED chats are 'keep this' contracts (set_pinned / list sorts them
        first): never prune a pinned transcript regardless of mtime — only the
        unpinned set is subject to the count cap."""
        try:
            meta = self._load_meta()
            pinned = {s for s, m in meta.items()
                      if isinstance(m, dict) and m.get("pinned")}
            files = sorted(self.dir.glob("*.jsonl"),
                           key=lambda f: f.stat().st_mtime, reverse=True)
            unpinned = [f for f in files if f.stem not in pinned]
            for old in unpinned[keep:]:
                try:
                    old.unlink()
                except OSError:
                    continue
        except OSError:
            pass

    def clear(self, sid: str) -> None:
        try:
            self._path(sid).unlink(missing_ok=True)
        except OSError:
            pass
        with _WRITE_LOCK:
            meta = self._load_meta()
            if meta.pop(_safe_sid(sid), None) is not None:
                self._save_meta(meta)
