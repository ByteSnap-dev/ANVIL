"""The operator card — a compact, coherent 'who the operator is' synthesis.

ANVIL's raw ``profile`` notes are a scattered pile of facts ("likes terse
answers", "has a Traeger", "wife is …"). Dumping N of them into every prompt is
noisy and incoherent. The operator card distils them (in the BACKGROUND — during
sleep, never on the hot path) into one tight paragraph the model can actually
use, refreshed as the facts grow.

The distillation is a 2-pass 'dialectic' adapted from Nous hermes-agent: a first
draft, then a self-audit/reconcile pass that tightens to only what helps ANVIL
be immediately useful. Grounded — it may only use the facts given, never invent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from . import config as cfgmod

_MAX_WORDS = 160

_DRAFT_SYS = (
    "You distil a coherent profile of the OPERATOR from scattered facts about "
    "them, for an AI household assistant. Use ONLY the facts given — never "
    "invent, guess, or pad. Cover who they are, their preferences and working "
    "style, how they like the assistant to behave, and any ongoing projects or "
    "context. Plain prose, no headers.")

_RECONCILE_SYS = (
    "You are tightening a draft operator profile. Reconcile any contradictions "
    "(prefer the more recent/specific fact), drop anything not useful for "
    "helping the operator day-to-day, and cut to the essentials. Return ONLY "
    f"the final profile, at most {_MAX_WORDS} words, plain prose. Invent nothing.")


def _actor(cfg) -> str:
    return getattr(cfg, "_actor", "") or ""


def _card_path(cfg) -> Path:
    """Per-family-member card when a profile is acting, so Lara never greets
    Sam with Alex's profile. Single-user (no actor) keeps the original file."""
    actor = _actor(cfg)
    if actor:
        slug = re.sub(r"[^A-Za-z0-9_-]", "", actor)[:48] or "op"
        return Path(cfg.memory_dir) / f"operator_card_{slug}.md"
    return Path(cfg.memory_dir) / "operator_card.md"


def load(cfg) -> str:
    """The acting profile's operator card text (empty string if none built yet)."""
    try:
        return _card_path(cfg).read_text("utf-8").strip()
    except OSError:
        return ""


def build(cfg, router, memory) -> str:
    """Distil the operator card from profile notes via a 2-pass dialectic.
    Runs in the background (sleep). Returns the card, or '' if too little to
    synthesise. Never raises into the caller's loop. Scoped to the acting
    profile's OWN facts (owner==actor) so cards never blend family members."""
    try:
        actor = _actor(cfg)
        notes = [n for n in memory.all_notes()
                 if n.type == "profile" and getattr(n, "owner", "") == actor]
        notes.sort(key=lambda n: -getattr(n, "salience", 0.5))
        facts = [n.body.strip() for n in notes[:40] if n.body.strip()]
        if len(facts) < 3:
            return ""                     # not enough to say anything coherent
        bullets = "\n".join(f"- {f}" for f in facts)

        draft = router.complete(
            [{"role": "user", "content": "Facts about the operator:\n" + bullets}],
            system=_DRAFT_SYS, min_rung=0, max_tokens=400, think=False)
        draft_text = draft.completion.text.strip()
        if not draft_text:
            return ""
        final = router.complete(
            [{"role": "user", "content":
              "Draft profile:\n" + draft_text
              + "\n\nOriginal facts (for reconciliation):\n" + bullets}],
            system=_RECONCILE_SYS, min_rung=0, max_tokens=350, think=False)
        card = final.completion.text.strip() or draft_text
        cfgmod.atomic_write(_card_path(cfg), card + "\n")
        return card
    except Exception:
        return ""
