"""Zero-dependency cron scheduler + job runner.

Jobs are small spec files in the ``jobs/`` directory. The agent can create and
edit them itself (see ``write_job``), so when a conversation implies recurrence
the Planner just emits a spec — no human crontab editing.

A built-in 5-field cron parser (standard ``min hour dom month dow`` syntax with
``*``, ``,``, ``-`` and ``*/step``) keeps this dependency-free. If APScheduler
is installed the runner can hand off to it for sub-minute precision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Cron parsing
# --------------------------------------------------------------------------- #
def _parse_field(spec: str, lo: int, hi: int) -> set:
    out: set = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, step_s = part.split("/")
            step = int(step_s)
        # A non-positive step ('*/0', '*/-1') makes range() either raise or yield
        # NOTHING — the same silent never-run this parser guards against below.
        # Reject it explicitly so it surfaces as a clear error, not an empty set.
        if step <= 0:
            raise ValueError(f"step must be a positive integer, got {step!r}")
        if part in ("*", ""):
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-")
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        # Reject out-of-range LITERALS (e.g. minute '60', hour '25', dom '0').
        # Silently clamping them to an empty set is exactly the "looks valid but
        # never runs" failure this scheduler works to prevent: '60 14 * * *' would
        # parse without error yet match no real time, so the job would sit enabled
        # and never fire. Raise so _schedule's validation surfaces it to the model
        # and due_jobs warns, instead of a silent never-run.
        if not (lo <= start <= hi and lo <= end <= hi):
            raise ValueError(f"value out of range [{lo}-{hi}]: {part!r}")
        # A reversed range ('30-10') yields an empty range() — another silent
        # never-run. Require start <= end so it errors instead of vanishing.
        if start > end:
            raise ValueError(f"range start after end: {part!r}")
        out.update(range(start, end + 1, step))
    return {v for v in out if lo <= v <= hi}


@dataclass
class Cron:
    minute: set
    hour: set
    dom: set
    month: set
    dow: set  # 0=Sunday .. 6=Saturday
    dom_star: bool = True  # was the day-of-month field a literal '*'?
    dow_star: bool = True  # was the day-of-week field a literal '*'?

    @classmethod
    def parse(cls, expr: str) -> "Cron":
        f = expr.split()
        if len(f) != 5:
            raise ValueError(f"cron needs 5 fields, got {len(f)}: {expr!r}")
        # Day-of-week: standard cron accepts BOTH 0 and 7 for Sunday. Parse the
        # field allowing 7, then fold 7 -> 0, so a spec like '0 0 * * 7' (a common
        # way to write "Sunday") actually fires. Without the fold, 7 was dropped
        # as out-of-range, leaving an empty dow set that matched NOTHING — a job
        # that looked valid but silently never ran (the exact silent-never-run
        # failure this scheduler works to prevent).
        dow = _parse_field(f[4], 0, 7)
        if 7 in dow:
            dow = (dow - {7}) | {0}
        return cls(
            _parse_field(f[0], 0, 59),
            _parse_field(f[1], 0, 23),
            _parse_field(f[2], 1, 31),
            _parse_field(f[3], 1, 12),
            dow,
            dom_star=(f[2].strip() == "*"),
            dow_star=(f[4].strip() == "*"),
        )

    def matches(self, dt: datetime) -> bool:
        dow = (dt.weekday() + 1) % 7  # Python Mon=0 -> cron Sun=0
        # Standard Vixie/POSIX cron: when BOTH the day-of-month and day-of-week
        # fields are restricted (neither is a literal '*'), the job fires on the
        # UNION of the two — e.g. '0 9 1 * 1' means "9am on the 1st OR every
        # Monday", not "9am only when the 1st is a Monday". ANDing both here made
        # such a spec fire ~1-2x/year (a silent almost-never-run). Only OR when
        # both are restricted; otherwise (the common case, one field '*') AND.
        # A range like '1-7' is restricted (not a literal '*'), so it still ORs.
        if not self.dom_star and not self.dow_star:
            day_ok = (dt.day in self.dom) or (dow in self.dow)
        else:
            day_ok = (dt.day in self.dom) and (dow in self.dow)
        return (
            dt.minute in self.minute
            and dt.hour in self.hour
            and dt.month in self.month
            and day_ok
        )


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    name: str
    cron: str
    pipeline: str = "ask"
    inputs: Dict = field(default_factory=dict)
    notify: Optional[str] = None      # e.g. "discord"
    min_rung: int = 0
    enabled: bool = True
    owner: str = ""                   # profile whose devices get the result push
    once: bool = False                # one-shot: self-delete after it fires once
    path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "cron": self.cron, "pipeline": self.pipeline,
            "inputs": self.inputs, "notify": self.notify,
            "min_rung": self.min_rung, "enabled": self.enabled,
            "owner": self.owner, "once": self.once,
        }


class Scheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg.jobs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last_run: Dict[str, str] = {}
        self._warned_cron: Dict[str, str] = {}   # job -> bad spec already warned about

    # -- job CRUD (the agent can call these autonomously) -------------- #
    def write_job(self, job: Job) -> Path:
        path = self.dir / f"{job.name}.json"
        # Atomic + read-only-tolerant like every other ANVIL state file: a plain
        # write_text interrupted mid-write can leave a truncated/empty job file
        # that then silently never runs (observed: an empty '{}' job on disk).
        from . import config as cfgmod
        cfgmod.atomic_write(path, json.dumps(job.to_dict(), indent=2))
        job.path = path
        return path

    def remove_job(self, name: str) -> bool:
        path = self.dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def load_jobs(self) -> List[Job]:
        jobs = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                d = json.loads(p.read_text("utf-8"))
                jobs.append(Job(path=p, **d))
            except Exception as exc:
                # A malformed/empty job file (e.g. a truncated '{}') used to be
                # skipped in total silence — the operator couldn't tell a job had
                # gone bad. Warn ONCE so junk on disk is visible and fixable.
                if self._warned_cron.get("bad:" + p.name) != "1":
                    self._warned_cron["bad:" + p.name] = "1"
                    print(f"[anvil] job file '{p.name}' is malformed and ignored "
                          f"({type(exc).__name__}) — delete or fix it")
                continue
        return jobs

    # -- running ------------------------------------------------------- #
    def due_jobs(self, now: Optional[datetime] = None) -> List[Job]:
        now = now or datetime.now()
        stamp = now.strftime("%Y-%m-%d %H:%M")
        due = []
        for job in self.load_jobs():
            if not job.enabled:
                continue
            try:
                if Cron.parse(job.cron).matches(now) and self._last_run.get(job.name) != stamp:
                    due.append(job)
            except ValueError as exc:
                # A typo'd cron spec used to be skipped in total silence — the
                # job looked enabled but never fired. Warn once per bad spec so
                # the operator can actually see why their job isn't running.
                if self._warned_cron.get(job.name) != job.cron:
                    self._warned_cron[job.name] = job.cron
                    print(f"[anvil] job '{job.name}': invalid cron "
                          f"'{job.cron}' — never runs ({exc})")
                continue
        return due

    def run_pending(self, run_job: Callable[[Job], None],
                    now: Optional[datetime] = None) -> int:
        """Run every due job once, surviving job failures.

        TASK-0007: this used to be ``try/finally`` with no ``except`` inline in
        serve() — the FIRST exception from any job killed the entire scheduler
        loop permanently and silently. A failing job must never take down the
        other jobs (or the loop), so log it and keep going. The ``finally``
        stamp still guarantees a crashing job isn't retried in a tight loop."""
        now = now or datetime.now()
        stamp = now.strftime("%Y-%m-%d %H:%M")
        ran = 0
        for job in self.due_jobs(now):
            try:
                run_job(job)
                ran += 1
            except Exception as exc:
                print(f"[anvil] job '{job.name}' failed: {type(exc).__name__}: {exc}")
            finally:
                self._last_run[job.name] = stamp
                # A one-shot job (e.g. 'remind me in 1 minute') must fire ONCE and
                # then vanish — otherwise its cron nags forever. Delete it after it
                # runs, success or failure, so a crashing one-shot can't retry in a
                # tight loop either.
                if getattr(job, "once", False):
                    try:
                        self.remove_job(job.name)
                    except Exception:
                        pass
        return ran

    def serve(self, run_job: Callable[[Job], None], poll: int = 30) -> None:
        """Blocking loop: every ``poll`` seconds, run any due jobs once."""
        print(f"[anvil] scheduler up, watching {self.dir} (Ctrl-C to stop)")
        while True:
            self.run_pending(run_job)
            time.sleep(poll)
