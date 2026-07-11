"""Family-docs RAG — let Lara answer from the household's own documents.

Point ANVIL at a folder of the family's papers (manuals, warranties, medical
summaries, insurance, recipes, school forms) and it indexes them into embedded
chunks so Lara can ground answers in what the family actually has on file —
"what's our policy number", "how do I reset the router", "when's the warranty up".

Local + stdlib + numpy-free: chunks are stored with their embedding vectors in a
JSONL index next to memory. A content-hash cache skips unchanged files; files
that disappear are dropped from the index (delete-by-absence). Retrieval is plain
cosine over the chunk vectors — the same machinery ``memory.py`` already uses.

Scanned/image PDFs (pypdf returns ~no text) are detected and LOUDLY skipped, not
silently swallowed, so a medical/insurance scan never looks indexed when it isn't.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import config as cfgmod
from .memory import _cosine

_CHUNK_CHARS = 1200          # ~300 tokens; heading-aware split keeps ideas together
_OVERLAP = 180               # ~15% overlap so an answer isn't severed at a boundary


def docs_dir(cfg) -> Path:
    return Path(getattr(cfg, "family_docs_dir", "family_docs"))


def _index_path(cfg) -> Path:
    return Path(cfg.memory_dir) / "docs_index.jsonl"


def _read_text(p: Path) -> Tuple[str, bool]:
    """Return (text, ok). ok=False for a scanned/binary doc we couldn't read as
    text — the caller flags it rather than pretending it was indexed."""
    suf = p.suffix.lower()
    if suf in (".md", ".markdown", ".txt", ".text", ".rst"):
        try:
            return p.read_text("utf-8", "replace"), True
        except OSError:
            return "", False
    if suf == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(p))
            text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
            # A scanned PDF yields almost nothing from text extraction.
            return (text, True) if len(text.strip()) >= 40 else ("", False)
        except Exception:
            return "", False
    return "", False


def _chunk(text: str) -> List[str]:
    """Split on blank lines / headings into ~_CHUNK_CHARS chunks with overlap."""
    paras = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks, cur = [], ""
    for para in paras:
        if len(cur) + len(para) + 2 <= _CHUNK_CHARS:
            cur = (cur + "\n\n" + para) if cur else para
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-_OVERLAP:] + "\n\n" + para) if cur else para
            while len(cur) > _CHUNK_CHARS:      # a single huge paragraph
                chunks.append(cur[:_CHUNK_CHARS])
                cur = cur[_CHUNK_CHARS - _OVERLAP:]
    if cur.strip():
        chunks.append(cur)
    return chunks


def _file_hash(p: Path) -> str:
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()
    except OSError:
        return ""


class DocStore:
    def __init__(self, cfg, embedder: Optional[Callable[[str], List[float]]] = None):
        self.cfg = cfg
        self.embedder = embedder

    def _load(self) -> List[dict]:
        p = _index_path(self.cfg)
        if not p.exists():
            return []
        out = []
        for line in p.read_text("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _save(self, rows: List[dict]) -> None:
        p = _index_path(self.cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        cfgmod.atomic_write(p, "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))

    def reindex(self) -> Dict[str, object]:
        """(Re)build the index. Unchanged files (same content hash) keep their
        existing chunks+vectors; vanished files are dropped. Returns a summary
        including any files skipped because they weren't readable as text."""
        root = docs_dir(self.cfg)
        existing = self._load()
        by_file: Dict[str, List[dict]] = {}
        for r in existing:
            by_file.setdefault(r.get("file", ""), []).append(r)

        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        keep: List[dict] = []
        skipped: List[str] = []
        indexed = 0
        seen_rel = set()
        for p in files:
            rel = str(p.relative_to(root)).replace("\\", "/")
            seen_rel.add(rel)
            h = _file_hash(p)
            cached = by_file.get(rel)
            if cached and cached[0].get("hash") == h:
                keep.extend(cached)                 # unchanged — reuse
                continue
            # A file that is PRESENT but momentarily unreadable this pass (a
            # transient Windows lock from AV / a sync tool / an open editor makes
            # _file_hash return "" and _read_text fail) must NOT be silently
            # dropped from the index — that's a spurious delete-by-absence on a
            # doc that still exists. Keep its last-good chunks+vectors so a lock
            # blip can't un-index a family document; a real content change (hash
            # readable and differing) still reindexes normally below.
            if not h and cached:
                keep.extend(cached)
                continue
            text, ok = _read_text(p)
            if not ok:
                # Here the file WAS hashable (h is non-empty) but yields no usable
                # text: a genuinely scanned/binary doc or one whose content changed
                # to unreadable. Drop any stale chunks (don't keep phantom text) and
                # flag it — distinct from the transient-lock case handled above.
                skipped.append(rel)                 # scanned/binary — flag, don't fake
                continue
            for i, ch in enumerate(_chunk(text)):
                vec = []
                if self.embedder:
                    try:
                        vec = self.embedder(ch)
                    except Exception:
                        vec = []
                keep.append({"file": rel, "hash": h, "i": i, "text": ch, "vec": vec})
            indexed += 1
        # Root-level delete-by-absence guard, mirroring the per-file transient
        # protection above (line 141). If the docs ROOT itself vanished this pass
        # (a mount/network blip, a sync-tool folder rename, or a TOCTOU after the
        # caller's own exists()-check at mind.py:686), an empty scan must NOT
        # truncate a non-empty index and force a full re-embed of the whole corpus.
        # Guard on root.exists() (not on `keep`) so a legitimately EMPTIED but
        # present root still clears the index via normal delete-by-absence.
        if not root.exists() and existing:
            return {"files": 0, "chunks": len(existing),
                    "indexed": 0, "skipped": skipped}
        self._save(keep)
        # Mirror the chunks into the FTS index (index.py) so search can BM25-
        # narrow instead of scanning every chunk. Derived from the jsonl we
        # just saved — the index stays disposable.
        try:
            from .index import SearchIndex
            SearchIndex(self.cfg).sync_docs(
                [{"id": f"{r['file']}#{r['i']}", "source": r["file"],
                  "text": r["text"]} for r in keep])
        except Exception:
            pass
        return {"files": len(seen_rel), "chunks": len(keep),
                "indexed": indexed, "skipped": skipped}

    def search(self, query: str, k: int = 4) -> List[dict]:
        """Top-k chunks for the query by cosine (falls back to keyword overlap
        when embeddings are off), each with its source file."""
        rows = self._load()
        if not rows:
            return []
        # BM25 candidate narrowing at scale: manuals/insurance PDFs chunk into
        # far more text than notes do — rerank the FTS top hits instead of
        # cosining every chunk. Small corpora and index failures keep the
        # full scan below.
        if len(rows) >= 200:
            try:
                from .index import SearchIndex
                hits = {h[0] for h in SearchIndex(self.cfg).search_docs(query, limit=40)}
                if hits:
                    rows = [r for r in rows
                            if f"{r.get('file')}#{r.get('i')}" in hits]
            except Exception:
                pass
        qvec = None
        if self.embedder:
            try:
                qvec = self.embedder(query)
            except Exception:
                qvec = None
        q_terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        scored = []
        for r in rows:
            if qvec and r.get("vec"):
                s = _cosine(qvec, r["vec"])
            else:
                terms = set(re.findall(r"[a-z0-9]{3,}", r.get("text", "").lower()))
                s = len(q_terms & terms) / (len(q_terms) + 1)
            if s > 0:
                scored.append((s, r))
        scored.sort(key=lambda t: -t[0])
        return [{"file": r["file"], "text": r["text"], "score": round(s, 3)}
                for s, r in scored[:max(1, k)]]
