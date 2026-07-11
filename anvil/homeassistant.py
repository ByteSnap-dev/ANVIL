"""Read-only Home Assistant client — the senses keystone.

Reads entity/sensor states over HTTP using only the standard library
(``urllib``). The HTTP transport is dependency-injected so tests run
fully offline with no live HA required.

Config from environment:
  HA_URL        e.g. http://homeassistant.local:8123
  HA_TOKEN      long-lived access token (never committed)
  HA_TIMEOUT_S  request timeout in seconds (default 5)
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

# The full /api/states dump is fetched by the chat priming, the sense loop, AND
# every ha_* tool call — often several times within one agent turn (prime, then
# the model calls ha_list, then ha_search...). It's the same hundreds-of-KB
# payload each time. Cache it briefly per HA instance; 5s is far fresher than
# any human question needs, and a state-changing call_service invalidates it so
# an action's effect is visible to the very next read.
_STATES_TTL_S = 5.0
_STATES_LOCK = threading.Lock()
_STATES_CACHE: Dict[str, Any] = {}   # url -> {"ts": float, "states": list}


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _default_opener(url: str, headers: Dict[str, str], timeout: int,
                    data: Optional[bytes] = None) -> bytes:
    method = "POST" if data is not None else "GET"
    req = urllib.request.Request(url, headers=headers, data=data, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class HomeAssistant:
    """Minimal client for the Home Assistant REST API.

    Reads (states) are unrestricted; the one write path (``call_service``) is only
    ever reached through the approval-gated ``ha_service`` tool. Pass a custom
    ``opener`` callable ``(url, headers, timeout, data=None) -> bytes`` to stub
    out HTTP in tests; omit it to use real urllib.
    """

    def __init__(
        self,
        ha_url: Optional[str] = None,
        ha_token: Optional[str] = None,
        timeout: Optional[int] = None,
        opener: Optional[Callable[[str, Dict[str, str], int], bytes]] = None,
    ) -> None:
        self._url = (ha_url or os.environ.get("HA_URL", "")).rstrip("/")
        self._token = ha_token or os.environ.get("HA_TOKEN", "")
        self._timeout = timeout if timeout is not None else _env_int("HA_TIMEOUT_S", 5)
        self._opener = opener or _default_opener

    @property
    def is_configured(self) -> bool:
        return bool(self._url and self._token)

    def _get(self, path: str) -> Any:
        url = self._url + path
        headers = {"Authorization": f"Bearer {self._token}"}
        raw = self._opener(url, headers, self._timeout)
        return json.loads(raw.decode("utf-8"))

    def call_service(self, domain: str, service: str,
                     data: Optional[Dict[str, Any]] = None) -> Any:
        """Call a Home Assistant service (POST /api/services/<domain>/<service>),
        e.g. light.turn_off with {'entity_id': 'light.garage'}. Raises on failure
        so the caller (the approval-gated tool) can surface it."""
        if not self.is_configured:
            raise RuntimeError("Home Assistant is not configured")
        url = f"{self._url}/api/services/{domain}/{service}"
        headers = {"Authorization": f"Bearer {self._token}",
                   "Content-Type": "application/json"}
        body = json.dumps(data or {}).encode("utf-8")
        raw = self._opener(url, headers, self._timeout, body)
        # We just changed the house — drop the cached snapshot so the next read
        # (e.g. the model checking its own action) sees the new state.
        with _STATES_LOCK:
            _STATES_CACHE.pop(self._url, None)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            return []

    def states(self, fresh: bool = False) -> List[Dict[str, Any]]:
        """Return all entity states, or [] when not configured or on error.

        Served from a short (5s) shared cache unless ``fresh=True`` — see the
        module note; the dump is re-fetched constantly by independent callers."""
        if not self.is_configured:
            return []
        now = time.time()
        if not fresh:
            with _STATES_LOCK:
                c = _STATES_CACHE.get(self._url)
                if c and now - c["ts"] < _STATES_TTL_S:
                    return c["states"]
        try:
            result = self._get("/api/states")
            result = result if isinstance(result, list) else []
        except Exception:
            return []
        if result:                    # never cache a failure/empty as truth
            with _STATES_LOCK:
                _STATES_CACHE[self._url] = {"ts": now, "states": result}
        return result

    def state(self, entity_id: str) -> Dict[str, Any]:
        """Return one entity state dict, or {} when not configured or on error."""
        if not self.is_configured:
            return {}
        try:
            result = self._get(f"/api/states/{entity_id}")
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def who_is_home(self) -> List[Dict[str, Any]]:
        """Return a list of {entity_id, name, state} dicts for all person.*
        entities, or [] when not configured or on error."""
        try:
            people = []
            for ent in self.states():
                entity_id = ent.get("entity_id", "")
                if not entity_id.startswith("person."):
                    continue
                attrs = ent.get("attributes", {}) or {}
                people.append({
                    "entity_id": entity_id,
                    "name": attrs.get("friendly_name", entity_id),
                    "state": ent.get("state", ""),
                })
            return people
        except Exception:
            return []

    def health_check(self) -> bool:
        """True if HA responds with a well-formed API root, False otherwise."""
        if not self.is_configured:
            return False
        try:
            result = self._get("/api/")
            return isinstance(result, dict) and bool(result.get("message"))
        except Exception:
            return False
