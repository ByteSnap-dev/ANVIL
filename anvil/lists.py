"""Shared family lists — the household's groceries (and later: errands,
packing, chores) as ONE family-visible store.

Deliberately simple: a single JSON file under ``cfg.memory_dir`` holding
named lists of ``{text, done, by, ts}`` items. Every profile sees and edits
the same lists (they're household state, like shared memory notes — the
default-private rule is for *memories*, not the groceries). Writes are
atomic so a crash mid-save can't eat the file; a missing or corrupt file
just starts empty.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

_MAX_ITEMS = 200          # per list — a grocery list, not a database


def _path(cfg) -> Path:
    return Path(getattr(cfg, "memory_dir", "memory")) / "shared_lists.json"


def _load(cfg) -> Dict[str, List[dict]]:
    p = _path(cfg)
    try:
        data = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, List[dict]] = {}
    for name, items in data.items():
        if not isinstance(items, list):
            continue
        clean = []
        for it in items:
            if isinstance(it, dict) and str(it.get("text", "")).strip():
                clean.append({"text": str(it["text"]),
                              "done": bool(it.get("done")),
                              "by": str(it.get("by", "")),
                              "ts": float(it.get("ts") or 0.0)})
        out[str(name)] = clean[:_MAX_ITEMS]
    return out


def _save(cfg, data: Dict[str, List[dict]]) -> None:
    from . import config as cfgmod
    p = _path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfgmod.atomic_write(p, json.dumps(data, indent=1))


def get_list(cfg, name: str = "groceries") -> List[dict]:
    return _load(cfg).get(name, [])


def all_lists(cfg) -> Dict[str, List[dict]]:
    data = _load(cfg)
    if "groceries" not in data:
        data["groceries"] = []       # the default list always exists
    return data


def add_item(cfg, text: str, by: str = "", name: str = "groceries") -> List[dict]:
    text = (text or "").strip()
    if not text:
        raise ValueError("item text is required")
    data = _load(cfg)
    items = data.setdefault(name, [])
    if len(items) >= _MAX_ITEMS:
        raise ValueError(f"list '{name}' is full ({_MAX_ITEMS} items)")
    items.append({"text": text[:300], "done": False,
                  "by": (by or "")[:40], "ts": time.time()})
    _save(cfg, data)
    return items


def set_done(cfg, index: int, done: bool, name: str = "groceries") -> List[dict]:
    data = _load(cfg)
    items = data.get(name, [])
    if not 0 <= int(index) < len(items):
        raise ValueError(f"no item {index} in '{name}'")
    items[int(index)]["done"] = bool(done)
    _save(cfg, data)
    return items


def remove_item(cfg, index: int, name: str = "groceries") -> List[dict]:
    data = _load(cfg)
    items = data.get(name, [])
    if not 0 <= int(index) < len(items):
        raise ValueError(f"no item {index} in '{name}'")
    items.pop(int(index))
    _save(cfg, data)
    return items
