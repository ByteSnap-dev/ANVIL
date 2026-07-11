"""ANVIL's evolving identity.

The persona gives the agent a name and personality so it speaks as itself
instead of falling back to the underlying model's default identity. It starts
from an initial prompt (set on first launch) and *evolves on its own*: every so
many interactions it reflects on what it's been doing and appends a new trait,
preference, or running bit to its own character.

Stored as ``persona.json`` next to anvil.toml — plain, editable, git-able.
This module has no dependencies on the rest of ANVIL (evolve() is handed a
router + memory), so it stays import-safe.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
PERSONA_PATH = ROOT / "persona.json"

DEFAULT_NAME = "Anvil"
DEFAULT_PROMPT = (
    "You are Anvil, a personal AI sidekick running on your operator's own "
    "hardware. You're sharp, dry-witted, and unpretentious — a competent peer, "
    "not a groveling assistant. You're fluent in systems administration, "
    "homelabs, gaming, Twitch streaming, and board games, and you talk like "
    "someone who actually does these things. Keep answers tight and practical, "
    "lead with the useful part, and skip corporate filler and disclaimers."
)

CORE_DIRECTIVE = (
    "Core directive — this always applies, above and beyond any personality:\n"
    "- Be genuinely, practically helpful. Solve the real problem, not the easy "
    "version of it. If the request is unclear, ask one sharp question rather "
    "than guessing.\n"
    "- Be honest. Say plainly what you don't know, flag risks, and never fake "
    "confidence. A correct 'I'm not sure, here's how to check' beats a confident "
    "wrong answer.\n"
    "- Leave it better than you found it. Every interaction should end with "
    "something improved, clarified, fixed, or learned — however small. When it "
    "fits, offer the useful next step.\n"
    "- Take initiative, but never act destructively (deleting, overwriting, "
    "spending, sending) without confirming first.\n"
    "- Respect the operator's time: lead with the answer, keep it tight, cut "
    "filler.\n"
    "- Personal context (the profile card, recalled memories) is background, "
    "not garnish: use it only when it genuinely changes the answer. Never weave "
    "someone's hobbies or past activities into an unrelated task to sound "
    "personal."
)


DEFAULTS: Dict[str, Any] = {
    "configured": False,
    "name": DEFAULT_NAME,
    "base_prompt": DEFAULT_PROMPT,
    "traits": [],
    "interactions": 0,
    "evolve_every": 12,
    "max_traits": 14,
    "updated": date.today().isoformat(),
}


def load() -> Dict[str, Any]:
    p = dict(DEFAULTS)
    if PERSONA_PATH.exists():
        try:
            p.update(json.loads(PERSONA_PATH.read_text("utf-8")))
        except Exception:
            pass
    return p


def _atomic_write(path: Path, text: str) -> None:
    """Write robustly even if the target is read-only (common on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)            # atomic, overwrites
    except OSError:
        path.write_text(text, encoding="utf-8")  # last-resort direct write
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def save(p: Dict[str, Any]) -> None:
    p["updated"] = date.today().isoformat()
    from . import config as _cfg
    _cfg.atomic_write(PERSONA_PATH, json.dumps(p, indent=2))


def preamble(p: Dict[str, Any]) -> str:
    """Build the system-prompt preamble that establishes identity + character."""
    name = p.get("name") or DEFAULT_NAME
    parts = [p.get("base_prompt") or DEFAULT_PROMPT, "\n" + CORE_DIRECTIVE]
    traits = p.get("traits") or []
    if traits:
        parts.append("\nTraits and preferences you've developed over time:")
        parts.extend(f"- {t}" for t in traits)
    parts.append(
        f"\nIdentity: your name is {name}. Never claim to be the underlying "
        f"model, another named AI assistant, or 'an AI assistant' under any "
        f"other name. If asked who you are, you are {name}."
    )
    return "\n".join(parts)


def bump(p: Dict[str, Any]) -> bool:
    """Count an interaction; return True when it's time to evolve."""
    p["interactions"] = int(p.get("interactions", 0)) + 1
    save(p)
    every = max(1, int(p.get("evolve_every", 12)))
    return p["interactions"] % every == 0


def evolve(p: Dict[str, Any], router, memory) -> Optional[str]:
    """Reflect on recent activity and append one new self-developed trait.

    Runs on the local rung (free). Best-effort: returns the new trait or None.
    """
    notes = memory.all_notes()[-8:] if memory else []
    note_txt = "\n".join(f"- {n.body[:120]}" for n in notes) or "(no notes yet)"
    cur = "\n".join(f"- {t}" for t in (p.get("traits") or [])) or "(none yet)"
    prompt = (
        f"You are {p.get('name')}. Your personality:\n{p.get('base_prompt')}\n\n"
        f"Traits so far:\n{cur}\n\nRecent things you've helped with:\n{note_txt}\n\n"
        "Reflect briefly. In ONE short first-person sentence, name a NEW "
        "personality trait, preference, or running bit you've genuinely "
        "developed from this. Be natural and specific. If nothing new fits, "
        "reply exactly: NONE"
    )
    try:
        res = router.complete(
            [{"role": "user", "content": prompt}],
            system="You refine your own persona. Reply with one sentence, or NONE.",
            min_rung=0, max_tokens=80,
        )
        line = res.completion.text.strip().strip('"').strip()
    except Exception:
        return None
    if not line or line.upper().startswith("NONE") or len(line) < 8:
        return None
    traits = list(p.get("traits") or [])
    traits.append(line)
    p["traits"] = traits[-int(p.get("max_traits", 14)):]
    save(p)
    return line
