"""Nightly snapshots of ANVIL's irreplaceable data — memory + skills + chats.

Git guards the CODE, but the family's memory (notes, profiles, learned skills,
operator cards) and their conversations are runtime state that git IGNORES. A
disk hiccup, a bad edit, or a botched migration could lose them with no way back.
This makes a rotating, timestamped zip snapshot so there is always a recent
restore point — the cheapest insurance a home appliance can carry.

Backups land in ``backups/`` (git-ignored) and are pruned to the most recent N.
``maybe_daily`` is called from the sleep cycle and no-ops if today is already
captured, so it costs a directory glob on the vast majority of ticks.
"""

from __future__ import annotations

import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional


def _root(cfg) -> Path:
    return Path(getattr(cfg, "memory_dir", "memory")).resolve().parent


def backup_dir(cfg) -> Path:
    return _root(cfg) / "backups"


def _sources(cfg) -> List[Path]:
    # Everything precious + unrecoverable: the whole memory dir (notes, INDEX,
    # embeddings, profiles.json, operator cards, skills/) and the chat transcripts.
    return [Path(cfg.memory_dir), _root(cfg) / "conversations"]


def create_backup(cfg) -> Optional[Path]:
    """Write one snapshot zip; prune old ones. Returns the path, or None on I/O
    failure (never raises into the sleep loop)."""
    bdir = backup_dir(cfg)
    try:
        bdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = bdir / f"anvil-backup-{stamp}.zip"
    i = 2                                   # avoid overwrite on two runs in one second
    while out.exists():
        out = bdir / f"anvil-backup-{stamp}-{i}.zip"
        i += 1
    bkey = str(bdir.resolve())
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for src in _sources(cfg):
                if not src.exists():
                    continue
                base = src.parent
                for p in src.rglob("*"):
                    # Never recurse the backups dir into a backup.
                    if not p.is_file() or bkey in str(p.resolve()):
                        continue
                    try:
                        z.write(p, p.relative_to(base))
                    except OSError:
                        continue
    except OSError:
        try:
            out.unlink()
        except OSError:
            pass
        return None
    _rotate(cfg)
    return out


def _rotate(cfg, keep: int = 14) -> None:
    try:
        files = sorted(backup_dir(cfg).glob("anvil-backup-*.zip"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                continue
    except OSError:
        pass


def list_backups(cfg) -> List[dict]:
    out: List[dict] = []
    try:
        for p in sorted(backup_dir(cfg).glob("anvil-backup-*.zip"),
                        key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                st = p.stat()
            except OSError:
                continue
            out.append({"name": p.name, "size": st.st_size, "ts": st.st_mtime})
    except OSError:
        pass
    return out


def maybe_daily(cfg) -> Optional[Path]:
    """Create today's snapshot if one doesn't exist yet — at most one per day.
    Cheap glob check on every call; only writes once daily."""
    today = date.today().strftime("%Y%m%d")
    try:
        if any(backup_dir(cfg).glob(f"anvil-backup-{today}-*.zip")):
            return None
    except OSError:
        pass
    return create_backup(cfg)
