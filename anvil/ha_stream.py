"""Real-time Home Assistant sensing over the websocket event API.

Implements docs/ha-websocket-design.md, migration step 1: the module ships
behind ``ha_stream = false`` and nothing regresses when it's off or the
socket can't stay up — snapshot-diff sensing remains the fallback engine.

The interesting logic (filter, per-entity throttle, transient debounce,
bounded ring) lives in :class:`EventSieve`, which is pure and fully
doctor-testable without a socket. :class:`HAStream` owns the connection
thread: connect → auth → subscribe_events(state_changed) → feed the sieve,
reconnecting with jittered backoff (HA restarts nightly; that's the common
path, not the edge case).
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

# Entity classes whose changes are worth remembering — mirrors what
# mind.sense_house() cares about today.
_WATCH_DOMAINS = ("person", "binary_sensor", "lock", "cover", "media_player",
                  "climate", "light", "switch")
_ALWAYS_RECORD = ("person", "lock")          # never throttled
_COOLDOWN_S = 300                            # per-entity, others
_DEBOUNCE_S = 30                             # a change that reverts = one event
_RING = 500


class EventSieve:
    """Turns the raw state_changed firehose into the few events worth
    keeping. Pure logic — inject a clock for tests."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.ring: deque = deque(maxlen=_RING)
        self._last_kept: Dict[str, float] = {}       # entity -> ts
        self._pending: Dict[str, dict] = {}          # entity -> maybe-transient

    @staticmethod
    def _domain(entity_id: str) -> str:
        return (entity_id or "").split(".", 1)[0]

    def _keep(self, evt: dict) -> None:
        self.ring.append(evt)
        self._last_kept[evt["entity_id"]] = evt["ts"]

    def feed(self, entity_id: str, old: Optional[str], new: Optional[str],
             ts: Optional[float] = None) -> None:
        """One state_changed event in; zero or one kept events out (into the
        ring). Attribute-only updates arrive with old == new and are dropped."""
        ts = self.clock() if ts is None else ts
        dom = self._domain(entity_id)
        if dom not in _WATCH_DOMAINS:
            return
        if old == new or new in (None, "", "unknown", "unavailable"):
            return
        # Transient debounce: if this REVERTS a still-pending change within
        # the window, collapse both into one compound event — signal the
        # snapshot-diff engine never had ("front door opened briefly").
        pend = self._pending.pop(entity_id, None)
        if pend and new == pend["old"] and ts - pend["ts"] <= _DEBOUNCE_S:
            self._keep({"entity_id": entity_id, "old": pend["old"],
                        "new": pend["new"], "ts": pend["ts"],
                        "transient": True,
                        "desc": f"{entity_id}: briefly {pend['new']} "
                                f"({int(ts - pend['ts'])}s), back to {new}"})
            return
        # Cooldown for chatty entities (never for person/lock).
        if dom not in _ALWAYS_RECORD:
            last = self._last_kept.get(entity_id, 0.0)
            if ts - last < _COOLDOWN_S:
                return
        evt = {"entity_id": entity_id, "old": old, "new": new, "ts": ts,
               "transient": False, "desc": f"{entity_id}: {old} -> {new}"}
        self._pending[entity_id] = evt
        self._keep(evt)

    def flush_pending(self) -> None:
        """Forget maybe-transient markers older than the window (their events
        are already kept; this only bounds the dict)."""
        now = self.clock()
        for k in [k for k, v in self._pending.items()
                  if now - v["ts"] > _DEBOUNCE_S]:
            self._pending.pop(k, None)

    def drain(self) -> List[dict]:
        out = list(self.ring)
        self.ring.clear()
        return out


class HAStream:
    """The connection thread. Start once; it reconnects forever until
    stop(). ``on_event`` receives each KEPT event dict (default: nothing —
    the ring is drained by whoever consumes sensing)."""

    def __init__(self, cfg, on_event: Optional[Callable[[dict], None]] = None):
        self.cfg = cfg
        self.sieve = EventSieve()
        self.on_event = on_event
        self.healthy = False
        self.events_seen = 0
        self.last_event_ts = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ha-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- connection loop ---------------------------------------------------- #
    def _ws_url(self) -> str:
        base = str(getattr(self.cfg, "ha_url", "") or "").rstrip("/")
        return (base.replace("https://", "wss://")
                    .replace("http://", "ws://") + "/api/websocket")

    def _run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self._session()
                backoff = 2.0                 # a clean session resets backoff
            except Exception:
                self.healthy = False
            if self._stop.is_set():
                return
            time.sleep(backoff + random.uniform(0, backoff / 2))
            backoff = min(backoff * 2, 60.0)

    def _session(self) -> None:
        import websocket                       # lazy: only when the stream runs
        token = str(getattr(self.cfg, "ha_token", "") or "")
        if not token or not getattr(self.cfg, "ha_url", ""):
            self._stop.set()                   # unconfigured: don't spin
            return
        ws = websocket.create_connection(self._ws_url(), timeout=35)
        try:
            json.loads(ws.recv())              # auth_required
            ws.send(json.dumps({"type": "auth", "access_token": token}))
            resp = json.loads(ws.recv())
            if resp.get("type") != "auth_ok":
                self._stop.set()               # bad token: surface, don't spin
                return
            ws.send(json.dumps({"id": 1, "type": "subscribe_events",
                                "event_type": "state_changed"}))
            json.loads(ws.recv())              # subscribe result
            self.healthy = True
            while not self._stop.is_set():
                raw = ws.recv()
                if not raw:
                    break
                msg = json.loads(raw)
                if msg.get("type") != "event":
                    continue
                data = (msg.get("event") or {}).get("data") or {}
                ent = data.get("entity_id") or ""
                old = ((data.get("old_state") or {}) or {}).get("state")
                new = ((data.get("new_state") or {}) or {}).get("state")
                before = len(self.sieve.ring)
                self.sieve.feed(ent, old, new)
                self.events_seen += 1
                self.last_event_ts = time.time()
                if self.on_event and len(self.sieve.ring) > before:
                    try:
                        self.on_event(self.sieve.ring[-1])
                    except Exception:
                        pass
        finally:
            self.healthy = False
            try:
                ws.close()
            except Exception:
                pass
