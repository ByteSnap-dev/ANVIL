"""Durable wall-clock job schedule — the autopilot's restart-proof memory of
"when did I last do X?".

The old cadence chained everything to in-RAM tick counters (dream every N ticks,
self-dev every Nth dream), so every server restart reset the fuse and the
multiplied knobs (heartbeat x dream_every x self_dev_every) were impossible to
reason about. This replaces counters with timestamps persisted to disk:

  sched = Schedule()
  if sched.due("selfdev", hours=12):   # survives restarts
      run_it()
      sched.mark("selfdev")

Semantics chosen deliberately:
  * A job never seen before is NOT due — ``due()`` starts its clock and returns
    False. A restart (or a fresh install) therefore never triggers a burst of
    work; every interval is measured from first sighting.
  * ``mark()`` records a run. ``elapsed()`` reports seconds since the last mark
    (None if never marked) for debounce checks.
  * Never raises: a corrupt or unwritable file degrades to in-memory state, so
    the mind keeps breathing even if the disk misbehaves.

State lives in ``dev-reports/schedule.json`` (gitignored — runtime marks must
never dirty the tree, or the forge's dirty-tree guard would block self-dev).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "dev-reports" / "schedule.json"


class Schedule:
    def __init__(self, path: Path = None):
        self.path = Path(path) if path else DEFAULT_PATH
        # Under `anvil doctor` a DEFAULT-path Schedule is ephemeral (in-memory
        # only): tests must never advance or pollute the real runtime schedule.
        # An explicit path (tests, tools) always persists.
        self._ephemeral = path is None and bool(os.environ.get("ANVIL_IN_DOCTOR"))
        self._lock = threading.Lock()
        self._jobs = {} if self._ephemeral else self._read()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write(self) -> None:
        """Atomic write (tmp + replace) so a crash mid-write can't corrupt it."""
        if self._ephemeral:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._jobs, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            pass                      # disk trouble -> in-memory state still works

    def due(self, job: str, *, hours: float = 0.0, minutes: float = 0.0,
            seconds: float = 0.0) -> bool:
        """True when at least the given interval has passed since the last mark.
        First sighting starts the clock (records now, returns False) so restarts
        never cause a burst of immediately-due work."""
        interval = hours * 3600.0 + minutes * 60.0 + seconds
        with self._lock:
            rec = self._jobs.get(job)
            if not isinstance(rec, dict) or "last" not in rec:
                self._jobs[job] = {"last": time.time(), "runs": 0}
                self._write()
                return False
            return (time.time() - float(rec["last"])) >= interval

    def mark(self, job: str) -> None:
        """Record that the job just ran (resets its interval, bumps the counter)."""
        with self._lock:
            rec = self._jobs.get(job)
            runs = int(rec.get("runs", 0)) + 1 if isinstance(rec, dict) else 1
            self._jobs[job] = {"last": time.time(), "runs": runs}
            self._write()

    def elapsed(self, job: str):
        """Seconds since the job last ran / was first seen; None if never seen.
        For debounce checks ('has it been >= 20 min since the last promote?')."""
        with self._lock:
            rec = self._jobs.get(job)
            if not isinstance(rec, dict) or "last" not in rec:
                return None
            return max(0.0, time.time() - float(rec["last"]))

    def status(self) -> dict:
        """Snapshot for observability (journal / status UI)."""
        with self._lock:
            return {k: dict(v) for k, v in self._jobs.items()
                    if isinstance(v, dict)}
