"""Durable, actor-scoped Plan — the externalized to-do list that gives Lara
follow-through.

A ``Plan`` is an ordered list of ``Step``s, each ``pending | doing | done |
blocked``, persisted to ``memory_dir/plans/`` so a multi-step task survives
across turns, context compaction, and even sleep cycles. It is the artifact the
agent loop consults to tell a premature deferral ("what would you like to work
on?") from genuine completion, and to resume mid-plan.

Storage is **actor-scoped** exactly like conversations/memory: one active plan
per profile at ``plans/u_<actor>.json`` (``plans/household.json`` for the
single-user / ambient case), so one family member's plan never leaks into
another's. See docs/planning-followthrough.md.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional

from . import config as cfgmod

STATUSES = ("pending", "doing", "done", "blocked")


def _safe_actor(actor: str) -> str:
    # Same sanitization discipline as conversations._safe_owner.
    return re.sub(r"[^A-Za-z0-9_-]", "", str(actor or ""))[:48]


@dataclass
class Step:
    id: int
    text: str
    status: str = "pending"
    note: str = ""


@dataclass
class Plan:
    task: str = ""
    steps: List[Step] = field(default_factory=list)
    actor: str = ""
    created: str = field(default_factory=lambda: date.today().isoformat())
    updated: str = field(default_factory=lambda: date.today().isoformat())
    updated_ts: float = 0.0          # epoch seconds of last save (recency guard)

    def is_active(self, max_age_s: float = 43200.0,
                  now: Optional[float] = None) -> bool:
        """Should this plan drive follow-through RIGHT NOW? True only if it has
        open steps AND was touched recently. Recency stops a stale plan from a
        finished task silently hijacking an unrelated later turn (the completion
        gate keys off this). A plan with no timestamp (updated_ts=0, e.g. an old
        file) is treated as stale."""
        if not self.open_steps():
            return False
        return ((now if now is not None else time.time())
                - (self.updated_ts or 0.0)) <= max_age_s

    # -- queries the completion gate + resume rely on -------------------- #
    def open_steps(self) -> List[Step]:
        """Steps still to do — a blocked step is NOT open (it's waiting on the
        human), so a plan that is all done-or-blocked counts as complete."""
        return [s for s in self.steps if s.status in ("pending", "doing")]

    def next_step(self) -> Optional[Step]:
        """The step to act on next: a step already in progress wins, else the
        first pending one."""
        for s in self.steps:
            if s.status == "doing":
                return s
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def is_empty(self) -> bool:
        return not self.steps

    def is_complete(self) -> bool:
        """A non-empty plan with no open steps. An empty plan is NOT 'complete'
        (there's simply no plan) — the completion gate treats that as 'no plan',
        so one-shot Q&A never gets trapped."""
        return bool(self.steps) and not self.open_steps()

    def mark(self, sid: int, status: str, note: str = "") -> bool:
        status = (status or "").strip().lower()
        if status not in STATUSES:
            return False
        for s in self.steps:
            if s.id == sid:
                s.status = status
                if note:
                    s.note = note
                return True
        return False

    # -- (de)serialization ---------------------------------------------- #
    def to_dict(self) -> dict:
        return {"task": self.task, "actor": self.actor,
                "created": self.created, "updated": self.updated,
                "updated_ts": self.updated_ts,
                "steps": [asdict(s) for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        steps: List[Step] = []
        for i, sd in enumerate(d.get("steps") or []):
            try:
                st = str(sd.get("status", "pending")).strip().lower()
                steps.append(Step(id=int(sd.get("id", i + 1)),
                                  text=str(sd.get("text", "")).strip(),
                                  status=st if st in STATUSES else "pending",
                                  note=str(sd.get("note", ""))))
            except (ValueError, TypeError):
                continue
        try:
            uts = float(d.get("updated_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            uts = 0.0
        return cls(task=str(d.get("task", "")), actor=str(d.get("actor", "")),
                   created=str(d.get("created", date.today().isoformat())),
                   updated=str(d.get("updated", date.today().isoformat())),
                   updated_ts=uts, steps=steps)

    def render(self) -> str:
        """A compact text view for the model's context and the `plan` tool."""
        if not self.steps:
            return "(no active plan)"
        icon = {"pending": "[ ]", "doing": "[~]", "done": "[x]", "blocked": "[!]"}
        head = f"PLAN: {self.task}" if self.task else "PLAN:"
        lines = [head]
        for s in self.steps:
            extra = f"  ({s.note})" if s.note else ""
            lines.append(f"  {icon.get(s.status, '[ ]')} {s.id}. {s.text}{extra}")
        return "\n".join(lines)


class PlanStore:
    """One active Plan per profile, persisted actor-scoped like conversations."""

    def __init__(self, cfg, actor=None):
        self.cfg = cfg
        self.actor = _safe_actor(actor if actor is not None
                                 else getattr(cfg, "_actor", ""))
        self.dir = Path(cfg.memory_dir) / "plans"
        name = ("u_" + self.actor) if self.actor else "household"
        self.path = self.dir / (name + ".json")

    def load(self) -> Plan:
        try:
            p = Plan.from_dict(json.loads(self.path.read_text("utf-8")))
        except (OSError, ValueError):
            p = Plan(actor=self.actor)
        p.actor = self.actor
        return p

    def save(self, plan: Plan) -> None:
        plan.actor = self.actor
        plan.updated = date.today().isoformat()
        plan.updated_ts = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        cfgmod.atomic_write(self.path, json.dumps(plan.to_dict(), indent=2))

    def clear(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass

    def set_steps(self, task: str, texts: List[str]) -> Plan:
        """Replace the active plan with a fresh ordered step list (1-indexed)."""
        p = Plan(task=(task or "").strip(), actor=self.actor)
        p.steps = [Step(id=i + 1, text=t)
                   for i, t in enumerate(str(x).strip() for x in texts) if t]
        self.save(p)
        return p

    def update_step(self, sid: int, status: str, note: str = "") -> Plan:
        p = self.load()
        p.mark(int(sid), status, note)
        self.save(p)
        return p
