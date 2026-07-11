"""US weather senses — National Weather Service + Census geocoder, keyless.

Two free, no-signup government APIs, stdlib-only, offline-testable via an
injected opener (same pattern as ``homeassistant.py``):

* Census geocoder  — "123 Main St, City ST" -> lat/lon   (geocoding.geo.census.gov)
* NWS api.weather.gov — point forecast, hourly, and ACTIVE ALERTS for a lat/lon

Caching: an address geocode and a point's forecast-URL set never change, so
they're cached forever; forecasts/alerts are cached briefly (5 min) since the
mind loop, chat priming, and the weather tool may all ask within one turn.

Where "here" is, in priority order (see ``resolve_latlon``):
  1. explicit lat/lon (e.g. the operator's phone geolocation)
  2. an explicit place string ("weather in Denver") -> geocoded
  3. cfg.home_lat/home_lon if set
  4. Home Assistant's zone.home entity (has latitude/longitude attributes)
  5. cfg.home_address -> geocoded
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

_UA = "ANVIL household assistant (self-hosted; contact: anvil@example.com)"
_FORECAST_TTL_S = 300.0

_LOCK = threading.Lock()
_GEO_CACHE: Dict[str, Tuple[float, float]] = {}       # address -> (lat, lon)
_POINT_CACHE: Dict[str, Dict[str, str]] = {}          # "lat,lon" -> point urls
_WX_CACHE: Dict[str, Dict[str, Any]] = {}             # url -> {ts, data}


def _default_opener(url: str, headers: Dict[str, str], timeout: int) -> bytes:
    # Reuse push's certifi-backed SSL context: the stdlib default context on
    # this Windows setup can't verify public CAs (the exact bug that silently
    # broke push delivery to web.push.apple.com).
    from .push import _SSL
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as resp:
        return resp.read()


class Weather:
    def __init__(self, timeout: int = 10,
                 opener: Optional[Callable[[str, Dict[str, str], int], bytes]] = None,
                 contact: str = ""):
        self._timeout = timeout
        self._opener = opener or _default_opener
        self._ua = _UA.replace("anvil@example.com", contact) if contact else _UA

    def _get(self, url: str, ttl: float = 0.0) -> Any:
        """GET JSON with optional shared TTL cache (0 = no cache)."""
        now = time.time()
        if ttl:
            with _LOCK:
                c = _WX_CACHE.get(url)
                if c and now - c["ts"] < ttl:
                    return c["data"]
        raw = self._opener(url, {"User-Agent": self._ua,
                                 "Accept": "application/geo+json,application/json"},
                           self._timeout)
        data = json.loads(raw.decode("utf-8"))
        if ttl:
            with _LOCK:
                _WX_CACHE[url] = {"ts": now, "data": data}
        return data

    # -- location ------------------------------------------------------- #
    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """US address/place -> (lat, lon). Cached forever (places don't move).

        Census geocoder first (authoritative for street addresses), then
        Open-Meteo's keyless place search as fallback — Census only matches
        full street addresses, so bare place names ('Miami FL', 'Denver')
        would otherwise resolve to nothing."""
        key = " ".join((address or "").lower().split())
        if not key:
            return None
        with _LOCK:
            if key in _GEO_CACHE:
                return _GEO_CACHE[key]
        latlon = None
        try:
            url = ("https://geocoding.geo.census.gov/geocoder/locations/"
                   "onelineaddress?benchmark=Public_AR_Current&format=json"
                   "&address=" + urllib.parse.quote(address))
            data = self._get(url)
            matches = (((data or {}).get("result") or {}).get("addressMatches")) or []
            if matches:
                c = matches[0].get("coordinates") or {}
                latlon = (round(float(c["y"]), 4), round(float(c["x"]), 4))
        except Exception:
            pass
        if latlon is None:
            # Open-Meteo matches bare place names only ('Miami', not
            # 'Miami FL'), so also try the name with a trailing state / comma
            # part stripped; prefer a US match when several places share a name.
            import re as _re
            variants = [address,
                        address.split(",")[0].strip(),
                        _re.sub(r"[,\s]+[A-Za-z]{2}\.?$", "", address).strip()]
            for v in dict.fromkeys(v for v in variants if v):
                try:
                    url = ("https://geocoding-api.open-meteo.com/v1/search?count=5"
                           "&name=" + urllib.parse.quote(v))
                    res = ((self._get(url) or {}).get("results")) or []
                    if not res:
                        continue
                    hit = next((r for r in res
                                if r.get("country_code") == "US"), res[0])
                    latlon = (round(float(hit["latitude"]), 4),
                              round(float(hit["longitude"]), 4))
                    break
                except Exception:
                    continue
        if latlon is None:
            return None
        with _LOCK:
            _GEO_CACHE[key] = latlon
        return latlon

    def _point(self, lat: float, lon: float) -> Dict[str, str]:
        """NWS point metadata (forecast URLs). Stable for a coordinate -> cached."""
        key = f"{round(lat, 4)},{round(lon, 4)}"
        with _LOCK:
            if key in _POINT_CACHE:
                return _POINT_CACHE[key]
        data = self._get(f"https://api.weather.gov/points/{key}")
        props = (data or {}).get("properties") or {}
        urls = {"forecast": props.get("forecast", ""),
                "hourly": props.get("forecastHourly", ""),
                "place": ((props.get("relativeLocation") or {}).get("properties")
                          or {}).get("city", "")}
        if urls["forecast"]:
            with _LOCK:
                _POINT_CACHE[key] = urls
        return urls

    # -- weather -------------------------------------------------------- #
    def forecast(self, lat: float, lon: float, periods: int = 6) -> List[dict]:
        """Upcoming forecast periods (Today/Tonight/...), compact dicts."""
        urls = self._point(lat, lon)
        if not urls.get("forecast"):
            return []
        data = self._get(urls["forecast"], ttl=_FORECAST_TTL_S)
        out = []
        for p in (((data or {}).get("properties") or {}).get("periods") or [])[:periods]:
            pop = ((p.get("probabilityOfPrecipitation") or {}).get("value"))
            out.append({
                "name": p.get("name", ""),
                "temp": f"{p.get('temperature')}°{p.get('temperatureUnit', 'F')}",
                "wind": f"{p.get('windSpeed', '')} {p.get('windDirection', '')}".strip(),
                "precip": f"{pop}%" if pop is not None else "",
                "forecast": p.get("shortForecast", ""),
                "detail": (p.get("detailedForecast") or "")[:200],
            })
        return out

    def hourly(self, lat: float, lon: float, hours: int = 12) -> List[dict]:
        """The next N hours: local time, temp, precip %, short forecast."""
        urls = self._point(lat, lon)
        if not urls.get("hourly"):
            return []
        data = self._get(urls["hourly"], ttl=_FORECAST_TTL_S)
        out = []
        for p in (((data or {}).get("properties") or {}).get("periods") or [])[:hours]:
            pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
            out.append({
                "time": _hh(p.get("startTime", "")),
                "temp": f"{p.get('temperature')}°{p.get('temperatureUnit', 'F')}",
                "precip": int(pop or 0),
                "forecast": p.get("shortForecast", ""),
            })
        return out

    def alerts(self, lat: float, lon: float) -> List[dict]:
        """Active NWS alerts for the point (warnings/watches/advisories)."""
        key = f"{round(lat, 4)},{round(lon, 4)}"
        data = self._get(f"https://api.weather.gov/alerts/active?point={key}",
                         ttl=_FORECAST_TTL_S)
        out = []
        for f in (data or {}).get("features") or []:
            p = f.get("properties") or {}
            out.append({"id": p.get("id") or f.get("id", ""),
                        "event": p.get("event", ""),
                        "severity": p.get("severity", ""),
                        "headline": p.get("headline", ""),
                        "ends": p.get("ends") or p.get("expires") or ""})
        return out

    def summary(self, lat: float, lon: float, periods: int = 4) -> str:
        """Compact human/model-readable snapshot: alerts first, then forecast."""
        lines: List[str] = []
        try:
            place = self._point(lat, lon).get("place", "")
        except Exception:
            place = ""
        try:
            for a in self.alerts(lat, lon):
                lines.append(f"!! ALERT [{a['severity']}] {a['event']}: "
                             f"{a['headline']}"[:200])
        except Exception:
            pass
        try:
            for p in self.forecast(lat, lon, periods=periods):
                bits = [p["name"] + ":", p["temp"], p["forecast"]]
                if p["precip"]:
                    bits.append(f"precip {p['precip']}")
                if p["wind"]:
                    bits.append(f"wind {p['wind']}")
                lines.append(" ".join(b for b in bits if b))
        except Exception:
            pass
        # Hourly precip timeline + a pre-computed rain outlook, so "when will
        # it rain?" is answerable with actual times, not a day-wide percentage.
        try:
            hrs = self.hourly(lat, lon, hours=12)
            if hrs:
                lines.append("Hourly precip next 12h: " +
                             " ".join(f"{h['time']} {h['precip']}%" for h in hrs))
                outlook = rain_windows(hrs)
                if outlook:
                    lines.append("Rain outlook: " + outlook)
        except Exception:
            pass
        if not lines:
            return ""
        head = f"WEATHER ({place or f'{lat},{lon}'}, via api.weather.gov):"
        return head + "\n" + "\n".join(lines)


def _hh(iso: str) -> str:
    """'2026-07-02T15:00:00-05:00' -> '3pm' (the location's local time)."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%I%p").lstrip("0").lower()
    except (ValueError, TypeError):
        return iso[:16]


def rain_windows(hours: List[dict], threshold: int = 30) -> str:
    """Turn hourly precip probabilities into a plain-language rain outlook.

    Deterministic (no model in the loop): consecutive hours at/over the
    threshold become windows like 'rain likely ~2pm-5pm (peaks 65% at 4pm)'.
    This is the line Lara actually needs to answer 'WHEN will it rain?' —
    handing her the analysis beats hoping she eyeballs 12 numbers correctly."""
    if not hours:
        return ""
    windows, cur = [], []
    for h in hours:
        if h.get("precip", 0) >= threshold:
            cur.append(h)
        elif cur:
            windows.append(cur)
            cur = []
    if cur:
        windows.append(cur)
    if not windows:
        peak = max(hours, key=lambda h: h.get("precip", 0))
        if peak.get("precip", 0) >= 15:
            return (f"rain unlikely in the next {len(hours)}h; highest chance "
                    f"{peak['precip']}% around {peak['time']}")
        return f"no rain expected in the next {len(hours)}h"
    parts = []
    for w in windows:
        peak = max(w, key=lambda h: h.get("precip", 0))
        span = w[0]["time"] if len(w) == 1 else f"{w[0]['time']}-{w[-1]['time']}"
        parts.append(f"~{span} (peaks {peak['precip']}% at {peak['time']})")
    return "rain likely " + ", then ".join(parts)


# --------------------------------------------------------------------------- #
# "Where is here?"
# --------------------------------------------------------------------------- #
def home_latlon(cfg, wx: Optional[Weather] = None) -> Optional[Tuple[float, float]]:
    """The household's coordinates: config lat/lon, else HA's zone.home, else a
    geocoded cfg.home_address. Best-effort; None when nothing is configured."""
    lat = float(getattr(cfg, "home_lat", 0.0) or 0.0)
    lon = float(getattr(cfg, "home_lon", 0.0) or 0.0)
    if lat or lon:
        return (lat, lon)
    try:
        from . import homeassistant as ha
        client = ha.HomeAssistant(timeout=getattr(cfg, "request_timeout", 5))
        if client.is_configured:
            for e in client.states():
                if e.get("entity_id") == "zone.home":
                    at = e.get("attributes") or {}
                    if at.get("latitude") and at.get("longitude"):
                        la = round(float(at["latitude"]), 4)
                        lo = round(float(at["longitude"]), 4)
                        # Home Assistant ships Amsterdam (52.3731, 4.8903) as the
                        # placeholder home location. If zone.home still reads that,
                        # HA is almost certainly UNCONFIGURED — don't silently give
                        # the family Amsterdam weather; fall through to
                        # home_address / None so they get a real location or an
                        # honest "not set" instead.
                        if not (abs(la - 52.3731) < 0.02 and abs(lo - 4.8903) < 0.02):
                            return (la, lo)
                    break
    except Exception:
        pass
    addr = (getattr(cfg, "home_address", "") or "").strip()
    if addr:
        return (wx or Weather()).geocode(addr)
    return None


def resolve_latlon(cfg, wx: Weather, lat=None, lon=None,
                   location: str = "", geo: Optional[dict] = None
                   ) -> Optional[Tuple[float, float]]:
    """Priority: explicit coords > named place > phone geolocation > home."""
    try:
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
    except (TypeError, ValueError):
        pass
    if (location or "").strip():
        hit = wx.geocode(location)
        if hit:
            return hit
    if geo and geo.get("lat") is not None and geo.get("lon") is not None:
        try:
            return (float(geo["lat"]), float(geo["lon"]))
        except (TypeError, ValueError):
            pass
    return home_latlon(cfg, wx)
