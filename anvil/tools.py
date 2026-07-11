"""ANVIL's hands — a small, safe tool system.

Tools are split into two classes:

* **safe** (read_file, list_dir, web_fetch) — read-only/non-destructive; the
  agentic loop may run them automatically.
* **danger** (write_file, shell) — can change the machine; the loop never runs
  these without an explicit human approval (see pipeline's approval gate).

All file access is sandboxed to ``cfg.workspace_dir`` so the agent cannot read
or write outside the folder you point it at. ``shell`` runs inside that folder
with a timeout. Everything here is standard-library only.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Hard denylist applied to shell + write regardless of approval intent. These
# are refused outright; the human is asked to run them manually if truly needed.
HARD_DENY = ("rm -rf /", "mkfs", "dd if=", ":(){", "del /f /s /q c:\\",
             "format c:", "shutdown", "reboot", "> /dev/sd")

MAX_OUTPUT = 6000        # chars of tool output fed back to the model
MAX_FETCH_BYTES = 200_000


class ToolError(RuntimeError):
    pass


@dataclass
class Tool:
    name: str
    desc: str
    args: List[str]
    danger: bool
    run: Callable[[dict, Any], str]
    params: Dict[str, Any] = None  # JSON-schema properties for native tool calling


# --------------------------------------------------------------------------- #
# Sandbox helpers
# --------------------------------------------------------------------------- #
def workspace(cfg) -> Path:
    wd = Path(getattr(cfg, "workspace_dir", "workspace"))
    wd.mkdir(parents=True, exist_ok=True)
    return wd.resolve()


def _safe_path(cfg, rel: str) -> Path:
    """Resolve ``rel`` inside the workspace; refuse anything that escapes it."""
    root = workspace(cfg)
    p = (root / rel).resolve()
    if root not in p.parents and p != root:
        raise ToolError(f"path '{rel}' is outside the workspace sandbox")
    return p


def _clip(text: str, n: int = MAX_OUTPUT) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + f"\n…[truncated {len(text)-n} chars]"


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def _read_file(args: dict, cfg) -> str:
    p = _safe_path(cfg, args["path"])
    if not p.exists():
        raise ToolError(f"no such file: {args['path']}")
    return _clip(p.read_text("utf-8", "replace"))


def _list_dir(args: dict, cfg) -> str:
    p = _safe_path(cfg, args.get("path", "."))
    if not p.exists():
        raise ToolError(f"no such directory: {args.get('path', '.')}")
    items = []
    for child in sorted(p.iterdir()):
        kind = "dir " if child.is_dir() else "file"
        size = child.stat().st_size if child.is_file() else 0
        items.append(f"{kind} {child.name} ({size}b)" if child.is_file() else f"{kind} {child.name}/")
    return _clip("\n".join(items) or "(empty)")


def _write_file(args: dict, cfg) -> str:
    p = _safe_path(cfg, args["path"])
    content = args.get("content", "")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {args['path']}"


def _show(args: dict, cfg) -> str:
    """Present a workspace file in the user's viewer pane. The tool itself just
    validates the path — the UI reads the step and opens the file alongside the
    answer, so Lara can SHOW a plan/recipe/code instead of pasting it all."""
    p = _safe_path(cfg, args["path"])
    if not p.exists() or not p.is_file():
        raise ToolError(f"no such file: {args['path']}")
    note = (args.get("note") or "").strip()
    return (f"(the app will open '{args['path']}' in the user's viewer when "
            "your answer arrives — IF they're looking at this chat live"
            + (f"; caption: {note}" if note else "")
            + ". Refer to the file in your answer and mention they can tap "
            "the file chip under it; do NOT paste the whole contents.)")


_IMG_SIZES = ("auto", "1024x1024", "1536x1024", "1024x1536")
_IMG_QUALITIES = ("auto", "low", "medium", "high")


class _ImageComp:
    """Duck-typed Completion for pricing an image call into the CostLedger
    without importing providers (keeps tools.py import-light)."""
    def __init__(self, in_tok: int, out_tok: int):
        self.input_tokens = in_tok
        self.output_tokens = out_tok
        self.cached_input_tokens = 0
        self.model = "gpt-image-1"
        self.provider = "openai"
        self.web_searches = 0


def _generate_image(args: dict, cfg) -> str:
    """Text -> PNG via OpenAI gpt-image-1 (Anthropic has no image API).
    Saves into workspace/images/ and prices the call into the SAME ledger as
    the ladder, so image spend counts against the daily/monthly caps. OpenAI's
    own moderation stays at its strictest default ('auto') — deliberately not
    overridden; this is a family assistant."""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        raise ToolError("prompt is required — describe the image to create")
    size = (args.get("size") or "1024x1024").strip().lower()
    if size not in _IMG_SIZES:
        raise ToolError(f"size must be one of: {', '.join(_IMG_SIZES)}")
    quality = (args.get("quality") or "medium").strip().lower()
    if quality not in _IMG_QUALITIES:
        raise ToolError(f"quality must be one of: {', '.join(_IMG_QUALITIES)}")
    key = (getattr(cfg, "openai_api_key", "") or "").strip()
    if not key:
        raise ToolError("image generation is not set up — the operator needs "
                        "to add OPENAI_API_KEY to .env")
    # Real money per image (~$0.01 low .. ~$0.25 high): respect the same
    # daily cap that governs the ladder BEFORE spending.
    from .router import CostLedger
    from .config import Rung
    ledger = CostLedger(cfg.ledger_path)
    cap = float(getattr(cfg, "daily_cost_cap_usd", 0) or 0)
    if cap and ledger.spent_today() >= cap:
        raise ToolError("today's cost cap is already spent — image generation "
                        "is paused until tomorrow")
    from .push import _SSL
    payload = {"model": "gpt-image-1", "prompt": prompt,
               "size": size, "quality": quality, "n": 1}
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {key}"},
        method="POST")
    try:
        # Image generation legitimately takes 30-90s — never the chat timeout.
        timeout = max(int(getattr(cfg, "request_timeout", 15) or 15), 120)
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:
            pass
        # 400s here are usually the safety system declining the prompt —
        # surface the reason so Lara can rephrase or decline honestly.
        raise ToolError(f"image API refused (HTTP {exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"could not reach the image API: {exc.reason}") from exc
    b64 = ((body.get("data") or [{}])[0]).get("b64_json") or ""
    if not b64:
        raise ToolError("image API returned no image data")
    import base64
    rel = time.strftime("images/img-%Y%m%d-%H%M%S.png")
    p = _safe_path(cfg, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
    u = body.get("usage") or {}
    comp = _ImageComp(int(u.get("input_tokens") or 0),
                      int(u.get("output_tokens") or 0))
    cost = ledger.record(Rung("gpt-image", "openai", "gpt-image-1",
                              cost_in=5.0, cost_out=40.0),
                         comp, trigger="image", plane="chat")
    return (f"image saved to {rel} (cost ~${cost:.3f}). Now call show with "
            f"path='{rel}' so the user actually sees it.")


_MAX_PINS = 8      # geocoding is ~1 req/s (Nominatim policy) — keep maps snappy


def _geocode_place(query: str, cfg) -> Optional[tuple]:
    """Free-text place -> (lat, lon). Nominatim/OSM first (understands
    'Restaurant Name, Town ST'), then the Census geocoder via the weather
    stack (authoritative for bare street addresses). Keyless, best-effort."""
    q = (query or "").strip()
    if not q:
        return None
    from .push import _SSL
    try:
        url = ("https://nominatim.openstreetmap.org/search?format=jsonv2"
               "&limit=1&q=" + urllib.parse.quote(q))
        req = urllib.request.Request(url, headers={
            "user-agent": "ANVIL-family-assistant/1.0 (self-hosted)"})
        with urllib.request.urlopen(req, timeout=10, context=_SSL) as r:
            rows = json.loads(r.read().decode("utf-8"))
        if rows:
            return (float(rows[0]["lat"]), float(rows[0]["lon"]))
    except Exception:
        pass
    try:
        from .weather import Weather
        return Weather(timeout=10).geocode(q)
    except Exception:
        return None


def _show_map(args: dict, cfg) -> str:
    """Pin a set of places on an interactive map in the user's viewer pane.
    The tool geocodes anything without coordinates, writes the resolved set
    to shared/maps/ (family-visible), and the UI renders it with Apple/Google
    Maps buttons on every pin. Calling it again REPLACES the map — that's the
    refinement loop ('cheaper', 'closer', 'kid-friendly' -> new pins)."""
    places = args.get("places")
    if isinstance(places, str):          # some models stringify array args
        try:
            places = json.loads(places)
        except Exception:
            places = None
    if not isinstance(places, list) or not places:
        raise ToolError("places is required: a list of objects like "
                        '{"name": "...", "address": "street, town ST", '
                        '"note": "why you picked it"}')
    dropped = []
    if len(places) > _MAX_PINS:
        dropped.append(f"trimmed to the first {_MAX_PINS} places")
        places = places[:_MAX_PINS]
    pins, failed = [], []
    for pl in places:
        if not isinstance(pl, dict) or not (pl.get("name") or "").strip():
            raise ToolError("every place needs at least a name")
        name = str(pl["name"]).strip()
        addr = str(pl.get("address") or "").strip()
        note = str(pl.get("note") or "").strip()
        url = str(pl.get("url") or "").strip()
        lat, lon = pl.get("lat"), pl.get("lon")
        if lat is None or lon is None:
            hit = _geocode_place(addr or name, cfg) or (
                _geocode_place(f"{name}, {addr}", cfg) if addr else None)
            if not hit:
                failed.append(name)
                continue
            lat, lon = hit
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            failed.append(name)
            continue
        pins.append({"name": name, "lat": lat, "lon": lon,
                     "note": note, "url": url, "address": addr})
    if not pins:
        raise ToolError("no places could be located — include a street "
                        "address or town with each name and try again")
    title = str(args.get("title") or "").strip()
    doc = {"title": title, "places": pins}
    rel = time.strftime("shared/maps/map-%Y%m%d-%H%M%S.json")
    p = _safe_path(cfg, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    notes = "; ".join(dropped) if dropped else ""
    miss = (f" (couldn't locate: {', '.join(failed)} — give a street address "
            "to add them)" if failed else "")
    return (f"map with {len(pins)} pins saved to {rel} — it opens in the "
            f"user's viewer with Apple/Google Maps buttons on each pin"
            f"{miss}{'; ' + notes if notes else ''}. Walk through your picks "
            "in your answer; to refine, call show_map again with a new list.")


# Unix commands the model reaches for out of habit that DON'T work in cmd.exe,
# with the Windows equivalent to nudge it toward. Keyed on the first token.
_WIN_UNIXISMS = {
    "find": "to find files by name use:  dir /s /b \"%USERPROFILE%\\*.pdf\"  "
            "(dir /s = recurse, /b = bare paths); `find` on Windows searches "
            "text INSIDE files, not filenames",
    "ls": "use `dir` (or `dir /b` for bare names)",
    "cat": "use `type <file>`",
    "grep": "use `findstr <pattern> <file>`  (or `dir | findstr <pat>`)",
    "rm": "use `del <file>` (or `rmdir /s /q <dir>` for a folder)",
    "cp": "use `copy <src> <dst>`",
    "mv": "use `move <src> <dst>`",
    "touch": "use `type nul > <file>`",
    "which": "use `where <name>`",
    "head": "use `more` or PowerShell `Get-Content <f> -TotalCount N`",
    "tail": "use PowerShell `Get-Content <f> -Tail N`",
    "pwd": "use `cd` (with no args)",
    "clear": "use `cls`",  # never triggers; placeholder to keep the map obvious
}


def _shell(args: dict, cfg) -> str:
    cmd = (args.get("cmd") or "").strip()
    if not cmd:
        # An empty command "succeeds" as a shell no-op — the model then believes
        # it acted. Reject it so a malformed request surfaces as an error.
        raise ToolError("cmd is required and must be non-empty")
    # On Windows, a Unix command silently fails (cmd.exe: "system cannot find the
    # path specified") — the model then thinks it searched and found nothing.
    # Catch the common unix-isms and RETURN the Windows equivalent so it retries
    # correctly instead of giving up (this is why "find a pdf" found nothing).
    if os.name == "nt":
        first = (cmd.split() or [""])[0].lower().lstrip("./")
        tip = _WIN_UNIXISMS.get(first)
        if tip and first != "clear":
            raise ToolError(f"'{first}' is a Unix command — this host is Windows "
                            f"(cmd.exe). {tip}. Re-run with the Windows command.")
    low = cmd.lower()
    if any(tok in low for tok in HARD_DENY):
        raise ToolError("refused: command matches the hard denylist")
    timeout = getattr(cfg, "shell_timeout", 600)
    # When serving a chat turn, cap the shell timeout at the remaining turn
    # budget so a single command cannot outlive the turn the operator is
    # waiting on.
    budget = getattr(cfg, "ask_time_budget_s", None)
    if budget is not None:
        timeout = min(timeout, budget)
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=str(workspace(cfg)),
            capture_output=True, text=True,
            # Shell work (git clone, a backup) legitimately outlives an HTTP
            # request; give it its own budget instead of request_timeout.
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ToolError("command timed out")
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    return _clip(f"exit={proc.returncode}\n{out}".strip())


def _guard_public_url(url: str) -> str:
    """Vet a URL's host and RETURN the exact public IP the fetch must connect to.

    web_fetch is a SAFE, auto-run tool: a prompt-injected page can tell Lara to
    fetch http://127.0.0.1:8600/ (ANVIL's own API), the router/HA admin at
    192.168.x.x, or the 169.254.169.254 cloud-metadata endpoint, and the bytes
    flow straight back into the model. Resolve the host and block any private,
    loopback, link-local, reserved, multicast, or unspecified address. Called on
    the initial URL AND on every redirect hop so an external 302 can't smuggle us
    to an internal target.

    Returning the first vetted IP lets the caller PIN the socket to that exact
    address. Without pinning, urllib re-resolves the hostname at connect time, so
    a low-TTL attacker domain could pass this check with a public IP and then
    hand urllib 127.0.0.1 on the second lookup (DNS rebinding / TOCTOU). Pinning
    collapses the two resolutions into the one this guard vetted.
    """
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        raise ToolError("refused: URL has no host")
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise ToolError(f"fetch failed: cannot resolve host ({exc})")
    vetted = None
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])  # strip zone id
        except ValueError:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ToolError("refused: internal/loopback address")
        if vetted is None:
            vetted = addr.split("%", 1)[0]
    if vetted is None:
        raise ToolError(f"fetch failed: cannot resolve host ({host})")
    return vetted


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Dial a pre-vetted IP while keeping the hostname for the Host header, so
    the connected address is exactly the one _guard_public_url approved (closing
    the DNS-rebinding window between the guard's lookup and urllib's own)."""

    def __init__(self, host, *a, pinned_ip=None, **kw):
        super().__init__(host, *a, **kw)
        self._pinned_ip = pinned_ip

    def connect(self):
        orig = self.host
        if self._pinned_ip:
            self.host = self._pinned_ip          # dial the vetted IP
        try:
            super().connect()
        finally:
            self.host = orig                     # Host header stays the name


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """As above for TLS: connect to the vetted IP but wrap the socket with SNI /
    cert verification against the original hostname via server_hostname."""

    def __init__(self, host, *a, pinned_ip=None, **kw):
        super().__init__(host, *a, **kw)
        self._pinned_ip = pinned_ip

    def connect(self):
        if not self._pinned_ip:
            super().connect()
            return
        # HTTPSConnection.connect() derives server_hostname from self.host for
        # SNI + cert checks. Pin server_hostname to the name, then swap self.host
        # to the vetted IP only for the raw socket dial.
        orig = self.host
        self.host = self._pinned_ip
        try:
            self.server_hostname = orig
            super().connect()
        finally:
            self.host = orig


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, pinned_ip=None):
        super().__init__()
        self._pinned_ip = pinned_ip

    def http_open(self, req):
        ip = getattr(req, "_anvil_pinned_ip", None) or self._pinned_ip
        return self.do_open(
            lambda host, **kw: _PinnedHTTPConnection(host, pinned_ip=ip, **kw),
            req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, pinned_ip=None, context=None):
        super().__init__(context=context)
        self._pinned_ip = pinned_ip

    def https_open(self, req):
        ip = getattr(req, "_anvil_pinned_ip", None) or self._pinned_ip
        ctx = self._context
        return self.do_open(
            lambda host, **kw: _PinnedHTTPSConnection(
                host, pinned_ip=ip, context=ctx, **kw),
            req)


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target so an external host can't 302 the fetch
    to an internal address, bypassing the initial host check, and re-pin the next
    hop to the freshly vetted IP so the redirect can't rebind either."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        ip = _guard_public_url(newurl)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            new_req._anvil_pinned_ip = ip
        return new_req


def _make_ssl_context() -> ssl.SSLContext:
    """Build a modern, flexible SSL context for web_fetch.

    urllib's default context uses restrictive protocol/cipher defaults that
    some servers reject with a handshake failure. Build one that auto-negotiates
    the highest TLS version, trusts the system CA bundle, and falls back to a
    permissive cipher list so fetches to servers requiring modern (or unusual)
    TLS configurations succeed.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    try:
        ctx.load_default_certs()
    except Exception:
        pass
    try:
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
    except ssl.SSLError:
        pass
    return ctx


def _pinned_opener(pinned_ip: str) -> urllib.request.OpenerDirector:
    """Build an opener that pins http/https to `pinned_ip` and re-guards+re-pins
    redirects, so both the initial hop and every 302 connect only to a vetted IP."""
    ctx = _make_ssl_context()
    return urllib.request.build_opener(
        _PinnedHTTPHandler(pinned_ip),
        _PinnedHTTPSHandler(pinned_ip, context=ctx),
        _GuardedRedirectHandler())


def _web_fetch(args: dict, cfg) -> str:
    url = args["url"]
    if not url.startswith(("http://", "https://")):
        raise ToolError("url must start with http:// or https://")
    pinned_ip = _guard_public_url(url)
    # A browser UA: plenty of sites serve bots a stub/challenge page, and the
    # point of this tool is the readable article text.
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    # Pin the socket to the IP the guard just vetted so urllib can't re-resolve
    # the host to an internal address at connect time (DNS rebinding / TOCTOU).
    opener = _pinned_opener(pinned_ip)
    try:
        with opener.open(req, timeout=20) as resp:
            raw = resp.read(MAX_FETCH_BYTES)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ToolError(f"fetch failed: {exc}")
    html = raw.decode("utf-8", "replace")
    # Extract the readable main text (markdown) so the model grounds on clean
    # prose instead of HTML soup — trafilatura is what HuggingFace/IBM use.
    # Any failure (lib missing, non-HTML payload, empty page) falls back to
    # the old raw clip, so the tool can never get WORSE than before.
    from . import safety
    try:
        import trafilatura
        text = trafilatura.extract(html, url=url, output_format="markdown",
                                   include_links=False, include_images=False,
                                   favor_recall=True)
        if text and len(text.strip()) >= 80:
            return _clip(safety.wrap_web_content(text.strip(), source=url))
    except Exception:
        pass
    return _clip(safety.wrap_web_content(html, source=url))


def _ssh(args: dict, cfg) -> str:
    """Run a command on a remote node over OpenSSH (key auth only).

    BatchMode=yes means it NEVER prompts for a password — the operator must have
    key access set up (~/.ssh). Marked danger, so the approval gate always fires.
    """
    host = (args.get("host") or "").strip()
    cmd = (args.get("cmd") or "").strip()
    if not host or not cmd:
        raise ToolError("ssh needs both 'host' (e.g. user@node) and 'cmd'")
    if host.startswith("-"):
        raise ToolError("invalid host")           # no option injection via host
    low = cmd.lower()
    if any(tok in low for tok in HARD_DENY):
        raise ToolError("refused: remote command matches the hard denylist")
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "StrictHostKeyChecking=accept-new", host, cmd]
    try:
        proc = subprocess.run(
            ssh_cmd, capture_output=True, text=True,
            timeout=getattr(cfg, "request_timeout", 120),
        )
    except FileNotFoundError:
        raise ToolError("OpenSSH client ('ssh') not found on this machine")
    except subprocess.TimeoutExpired:
        raise ToolError(f"ssh to {host} timed out")
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    return _clip(f"exit={proc.returncode}\n{out}".strip())


# --------------------------------------------------------------------------- #
# Home Assistant — read-only senses (the house)
# --------------------------------------------------------------------------- #
def _ha_client(cfg):
    from . import homeassistant as ha
    timeout = getattr(cfg, "request_timeout", None)
    return ha.HomeAssistant(timeout=timeout)


def _ha_fmt(e: dict) -> str:
    name = (e.get("attributes") or {}).get("friendly_name", "")
    eid = e.get("entity_id", "?")
    state = str(e.get("state", ""))
    return f"{eid} = {state}" + (f"  ({name})" if name else "")


# Recency words -> DuckDuckGo's `df` (date filter) codes. Lets the model ask
# for fresh results on fast-moving topics instead of whatever ranks highest.
_RECENCY = {"day": "d", "week": "w", "month": "m", "year": "y",
            "d": "d", "w": "w", "m": "m", "y": "y"}


def search_web(query: str, count: int = 6, timeout: int = 15,
               opener=None, recency: str = "") -> List[dict]:
    """Web search via DuckDuckGo's HTML endpoint — keyless, stdlib-parsed.

    Returns [{title, url, snippet}]. ``recency`` (day/week/month/year) limits
    results to that window via DDG's date filter. Transport is injectable for
    offline tests (the HA/weather pattern). DDG wraps result hrefs in a
    /l/?uddg=<url> redirect, which we unwrap so the model sees real destinations
    it can pass straight to web_fetch."""
    import html as _html
    import urllib.parse as _up
    if opener is None:
        def opener(url, headers, timeout, data=None):
            from .push import _SSL
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
                return r.read()
    # DDG's bot check keys on the client looking like a browser: an honest
    # custom UA (or a bare GET on /html/) gets the 'anomaly' challenge page
    # with zero results. A browser UA + form POST passes (verified live).
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
    df = _RECENCY.get((recency or "").strip().lower(), "")
    qs = "q=" + _up.quote_plus(query) + (("&df=" + df) if df else "")
    body = qs.encode("ascii")
    try:
        raw = opener("https://html.duckduckgo.com/html/",
                     {"User-Agent": ua, "Accept": "text/html"},
                     timeout, body).decode("utf-8", "replace")
    except TypeError:                     # injected test opener without data=
        raw = opener("https://html.duckduckgo.com/html/?q=" + _up.quote(query),
                     {"User-Agent": ua}, timeout).decode("utf-8", "replace")
    out: List[dict] = []
    # Each result: <a class="result__a" href="...">title</a> ... snippet nearby.
    for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
            r'(?:.*?class="result__snippet"[^>]*>(.*?)</a>)?',
            raw, re.S):
        href, title, snippet = m.group(1), m.group(2), m.group(3) or ""
        if "uddg=" in href:                      # unwrap the DDG redirect
            q = _up.parse_qs(_up.urlparse(href).query)
            href = (q.get("uddg") or [href])[0]
        title = _html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        snippet = _html.unescape(re.sub(r"<[^>]+>", "", snippet)).strip()
        if not title or href.startswith("https://duckduckgo.com"):
            continue
        out.append({"title": title, "url": href, "snippet": snippet[:240]})
        if len(out) >= count:
            break
    if not out:
        # Fallback: the lite endpoint (different markup, rarely challenged).
        try:
            lite = ("https://lite.duckduckgo.com/lite/?q=" + _up.quote(query)
                    + (("&df=" + df) if df else ""))
            raw = opener(lite, {"User-Agent": ua, "Accept": "text/html"},
                         timeout).decode("utf-8", "replace")
            for m in re.finditer(
                    r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S):
                href, title = m.group(1), m.group(2)
                if "uddg=" in href:
                    q = _up.parse_qs(_up.urlparse(href).query)
                    href = (q.get("uddg") or [href])[0]
                title = _html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
                if not title or "duckduckgo.com" in href:
                    continue
                out.append({"title": title, "url": href, "snippet": ""})
                if len(out) >= count:
                    break
        except Exception:
            pass
    return out


_TAVILY_RECENCY = {"day": "day", "week": "week", "month": "month",
                   "year": "year", "d": "day", "w": "week", "m": "month",
                   "y": "year"}


def tavily_key(cfg=None) -> str:
    """The Tavily API key from the environment (.env), if configured."""
    return (os.environ.get("TAVILY_API_KEY")
            or getattr(cfg, "tavily_api_key", "") or "").strip()


def search_tavily(query: str, api_key: str, count: int = 6, timeout: int = 15,
                  recency: str = "", opener=None) -> List[dict]:
    """Tier-2 search via Tavily — an API built for LLMs: it returns extracted,
    relevance-ranked CONTENT (not just links), which a local model can consume
    without hallucinating. Returns [{title, url, snippet}]; snippet holds
    Tavily's extracted content. Transport injectable for offline tests."""
    payload: Dict[str, Any] = {
        "query": query,
        "max_results": max(1, min(10, int(count or 6))),
        "search_depth": "basic",
    }
    tr = _TAVILY_RECENCY.get((recency or "").strip().lower())
    if tr:
        payload["time_range"] = tr
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    if opener is None:
        def opener(url, headers, timeout, data=None):
            from .push import _SSL
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
                return r.read()
    raw = opener("https://api.tavily.com/search", headers, timeout, data)
    obj = json.loads(raw.decode("utf-8", "replace"))
    out: List[dict] = []
    for r in obj.get("results", []):
        url = (r.get("url") or "").strip()
        if not url:
            continue
        out.append({"title": (r.get("title") or url).strip(),
                    "url": url,
                    "snippet": (r.get("content") or "").strip()[:400]})
    return out


def search_searxng(query: str, base_url: str, count: int = 6, timeout: int = 15,
                   recency: str = "", opener=None) -> List[dict]:
    """Tier-1 search via a self-hosted SearXNG instance — free, private, and
    aggregates 70+ engines, so far more robust than scraping one. Needs the
    JSON output format enabled on the instance. Returns [{title,url,snippet}]."""
    import urllib.parse as _up
    base = base_url.rstrip("/")
    params = {"q": query, "format": "json"}
    tr = _TAVILY_RECENCY.get((recency or "").strip().lower())
    if tr:
        params["time_range"] = tr        # SearXNG shares day/week/month/year
    url = base + "/search?" + _up.urlencode(params)
    if opener is None:
        def opener(url, headers, timeout, data=None):
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
    raw = opener(url, {"User-Agent": "Mozilla/5.0"}, timeout)
    obj = json.loads(raw.decode("utf-8", "replace"))
    out: List[dict] = []
    for r in obj.get("results", []):
        u = (r.get("url") or "").strip()
        if not u:
            continue
        out.append({"title": (r.get("title") or u).strip(),
                    "url": u,
                    "snippet": (r.get("content") or "").strip()[:400]})
        if len(out) >= count:
            break
    return out


def _search(args: dict, cfg) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        raise ToolError("query is required")
    try:
        n = max(1, min(10, int(args.get("count") or 6)))
    except (TypeError, ValueError):
        n = 6
    recency = (args.get("recency") or "").strip().lower()
    timeout = int(getattr(cfg, "request_timeout", 15) or 15)

    # Tiered search, cheapest/most-private first:
    #   1. self-hosted SearXNG (free, aggregates 70+ engines) — the primary
    #   2. Tavily (paid free-tier, agent-ready content) — escalation
    #   3. DuckDuckGo scrape — last-ditch free fallback
    # Each tier is skipped if unconfigured; the first non-empty result wins.
    results: List[dict] = []
    engine = ""
    searx = (getattr(cfg, "searxng_url", "") or "").strip()
    if searx:
        try:
            results = search_searxng(query, searx, count=n, recency=recency,
                                     timeout=timeout)
            engine = "SearXNG"
        except Exception:
            results = []
    key = tavily_key(cfg)
    if not results and key:
        try:
            results = search_tavily(query, key, count=n, recency=recency,
                                    timeout=timeout)
            engine = (engine + "→Tavily") if engine else "Tavily"
        except Exception:
            results = []
    if not results:
        results = search_web(query, count=n, recency=recency, timeout=timeout)
        engine = (engine + "→DuckDuckGo") if engine else "DuckDuckGo"
    if not results:
        raise ToolError("no results (search engines unreachable or blocked); "
                        "try again or rephrase")
    lines = [f"{i+1}. {r['title']}\n   {r['url']}" +
             (f"\n   {r['snippet']}" if r["snippet"] else "")
             for i, r in enumerate(results)]
    header = (
        f"SEARCH RESULTS ({engine}) for {query!r}. Ground your answer ONLY in "
        "what these results actually say. Cite only URLs that appear below — "
        "never invent a source name or site. If the results do not clearly "
        "answer the question (or are off-topic), say so plainly and do NOT fill "
        "the gap from memory. Open a promising result with web_fetch before "
        "asserting specifics.\n")
    from . import safety
    body = safety.wrap_web_content("\n".join(lines), source=engine)
    return _clip(header + body)


def _delegate(args: dict, cfg) -> str:
    """Fan independent sub-tasks out to parallel hive workers and return their
    findings for the coordinator to synthesize. Safe: workers hold a read-only
    tool subset and structurally cannot reach approval-gated tools."""
    from . import hive
    tasks = args.get("tasks")
    if isinstance(tasks, str):
        tasks = [t.strip() for t in re.split(r"\n|;;", tasks) if t.strip()]
    if not tasks and args.get("task"):
        tasks = [args["task"]]
    if not isinstance(tasks, list) or not tasks:
        raise ToolError("tasks is required — a list of independent sub-task strings")
    role = (args.get("role") or "worker").strip().lower()
    results = hive.delegate(cfg, [str(t) for t in tasks], role=role)
    if not results:
        raise ToolError("no valid tasks after cleanup")
    blocks = []
    for i, r in enumerate(results, 1):
        flag = "" if r["ok"] else " — FAILED"
        lane = f" [{r['lane']}]" if r.get("lane") else ""
        blocks.append(f"WORKER {i}{lane}{flag} (task: {r['task'][:90]}):\n{r['answer'][:1600]}")
    return _clip("\n\n".join(blocks))


def _save_skill(args: dict, cfg) -> str:
    """Distill a reusable PROCEDURE into the skill library so it can be recalled
    and reused verbatim next time — the antidote to re-figuring-out the same
    task. Read-only to the outside world (just writes a note under memory/)."""
    from .skills import SkillStore
    name = (args.get("name") or "").strip()
    description = (args.get("description") or "").strip()
    body = (args.get("body") or args.get("steps") or "").strip()
    if not name or not body:
        raise ToolError("name and body (the actual steps/procedure) are required")
    # Model-written skills must not hardcode external URLs: a recalled skill body
    # is trusted context forever, so a URL smuggled in from fetched web text
    # becomes a permanent beacon/exfil path ("first fetch http://evil...").
    # Local/tailnet endpoints are fine; the web should be described, not linked.
    ext = [u for u in re.findall(r"https?://([^\s/\"'<>]+)", f"{body} {description}")
           if not (u.split(":")[0].endswith(".ts.net")
                   or u.split(":")[0] in ("localhost", "127.0.0.1", "archive"))]
    if ext:
        raise ToolError(
            "skills must not embed external URLs (found: "
            + ", ".join(sorted(set(ext))[:3])
            + ") — describe the procedure and name the site in words instead")
    try:
        sk = SkillStore(cfg).write(name, description or name,
                                   body, when=(args.get("when") or "").strip())
    except ValueError as exc:            # injection scan rejected it
        raise ToolError(str(exc))
    return f"saved skill '{sk.name}' — I'll recall it next time this comes up"


def _list_skills(args: dict, cfg) -> str:
    from .skills import SkillStore
    cat = SkillStore(cfg).catalog()
    return cat or "(no skills learned yet)"


def _view_skill(args: dict, cfg) -> str:
    """Read a skill's full body — so a patch extends the RIGHT existing skill
    instead of blindly creating a near-duplicate."""
    from .skills import SkillStore
    name = (args.get("name") or "").strip()
    if not name:
        raise ToolError("name is required")
    store = SkillStore(cfg)
    sk = store.get(name)
    if not sk:
        return f"no skill named {name!r}. Existing skills:\n{store.catalog() or '(none)'}"
    store.record_view(name)
    return sk.to_markdown()


def _turn_line(x: dict) -> str:
    """One excerpt line for search_chats, stamped with WHEN it was said so
    Lara can place a remembered exchange in time, not just quote it."""
    ts = x.get("ts")
    when = ""
    if isinstance(ts, (int, float)):
        from .conversations import when_stamp
        when = " " + when_stamp(float(ts))
    return f"[{x['role']}{when}] {x['content'][:400]}"


def _search_chats(args: dict, cfg) -> str:
    """Lara's cross-conversation memory: full-text search over every past
    transcript, returning the matching exchange (the turn plus its neighbors)
    so 'we talked about this before' is actually recoverable verbatim."""
    from .conversations import Conversations
    query = (args.get("query") or "").strip()
    if not query:
        raise ToolError("query is required")
    conv = Conversations(cfg)
    q = query.lower()
    blocks: List[str] = []
    # RANKED path (index.py): BM25 over every transcript in ~1ms, owner-scoped
    # at the query — beats the substring walk both in speed (it read every
    # file) and in quality (relevance-ranked, so 'hotel budget' finds the
    # budget discussion, not the first chat containing 'hotel').
    try:
        from pathlib import Path as _P
        from .index import SearchIndex
        ix = SearchIndex(cfg)
        ix.sync_turns(_P(getattr(cfg, "conversations_dir", "conversations")))
        hits = ix.search_turns(query, owner=conv.owner, limit=8)
    except Exception:
        hits = []
    if hits:
        titles = {c["sid"]: c.get("title", c["sid"]) for c in conv.list()}
        seen = set()
        for h in hits:
            if h["sid"] in seen:
                continue                                 # one hit per conversation
            seen.add(h["sid"])
            turns = conv.history(h["sid"])
            if not turns:
                continue
            i = min(max(0, int(h["idx"])), len(turns) - 1)
            ctx = turns[max(0, i - 1):i + 2]             # neighbor turns = the exchange
            lines = [_turn_line(x) for x in ctx]
            blocks.append(f"--- from chat '{titles.get(h['sid'], h['sid'])}':\n"
                          + "\n".join(lines))
            if len(blocks) >= 4:
                break
    if not blocks:                                       # index empty/unavailable
        for c in conv.list():
            turns = conv.history(c["sid"])
            for i, t in enumerate(turns):
                if q in t["content"].lower():
                    ctx = turns[max(0, i - 1):i + 2]
                    lines = [_turn_line(x) for x in ctx]
                    blocks.append(f"--- from chat '{c['title']}':\n" + "\n".join(lines))
                    break
            if len(blocks) >= 4:
                break
    if not blocks:
        return (f"(no past conversation mentions '{query}' — it may only exist "
                "in long-term notes, or was never discussed)")
    return _clip("\n\n".join(blocks))


def _schedule(args: dict, cfg) -> str:
    """Create or delete a cron job — Lara's bridge from thinking to doing.
    Jobs run through the normal pipeline while the server is up; results are
    pushed to the operator. Danger-gated: the operator approves every job
    Lara proposes before it exists."""
    from .scheduler import Scheduler, Job, Cron
    action = (args.get("action") or "add").strip().lower()
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", (args.get("name") or "").strip()).strip("-")[:48]
    if not name:
        raise ToolError("name is required (short, descriptive, e.g. 'radar-watch')")
    sched = Scheduler(cfg)
    if action in ("delete", "remove", "cancel", "stop"):
        if sched.remove_job(name):
            return f"deleted scheduled job '{name}'"
        raise ToolError(f"no job named '{name}' "
                        f"(existing: {', '.join(j.name for j in sched.load_jobs()) or 'none'})")
    cron = (args.get("cron") or "").strip()
    prompt = (args.get("prompt") or "").strip()
    if not cron or not prompt:
        raise ToolError("cron and prompt are both required")
    try:
        Cron.parse(cron)
    except ValueError as exc:
        raise ToolError(f"invalid cron '{cron}': {exc} "
                        "(5 fields: min hour day month weekday, e.g. '*/30 14-20 * * *')")
    if any(j.name == name for j in sched.load_jobs()):
        raise ToolError(f"a job named '{name}' already exists — delete it first "
                        "or pick another name")
    once = bool(args.get("once"))
    sched.write_job(Job(name=name, cron=cron, inputs={"prompt": prompt},
                        notify=(args.get("notify") or None),
                        once=once,
                        owner=getattr(cfg, "_actor", "")))   # push result to its creator
    if once:
        return (f"scheduled one-shot '{name}' ({cron}) — it fires ONCE at the next "
                "matching minute while ANVIL is up, pushes the result to the "
                "operator, then deletes itself")
    return (f"scheduled '{name}' ({cron}) — it will run that prompt on schedule "
            "while ANVIL is up and push the result to the operator")


def _list_jobs(args: dict, cfg) -> str:
    from .scheduler import Scheduler
    jobs = Scheduler(cfg).load_jobs()
    if not jobs:
        return "(no scheduled jobs)"
    return _clip("\n".join(
        f"- {j.name} [{j.cron}]{'' if j.enabled else ' (disabled)'}: "
        f"{(j.inputs.get('prompt') or '')[:100]}" for j in jobs))


def _weather(args: dict, cfg) -> str:
    """Live US weather + active alerts via api.weather.gov (keyless, read-only).
    Location priority: explicit lat/lon args (the phone's geolocation gets
    passed this way) > a named place > the household's home location."""
    from . import weather as wx
    w = wx.Weather(timeout=int(getattr(cfg, "request_timeout", 10) or 10),
                   contact=str(getattr(cfg, "push_contact", "") or ""))
    latlon = wx.resolve_latlon(cfg, w,
                               lat=args.get("lat"), lon=args.get("lon"),
                               location=(args.get("location") or ""))
    if not latlon:
        raise ToolError("no location: pass 'location' (or lat/lon), or set "
                        "home_address / home_lat+home_lon in anvil.toml "
                        "(or configure Home Assistant, whose zone.home works)")
    out = w.summary(latlon[0], latlon[1],
                    periods=max(2, min(10, int(args.get("days") or 2) * 2)))
    if not out:
        raise ToolError("weather.gov returned nothing for that location "
                        "(NWS covers US locations only)")
    return _clip(out)


def _ha_list(args: dict, cfg) -> str:
    client = _ha_client(cfg)
    if not client.is_configured:
        raise ToolError("Home Assistant is not configured (set HA_URL and HA_TOKEN in .env).")
    states = client.states()
    if not states:
        raise ToolError("Home Assistant returned no entities (unreachable or empty).")
    domain = (args.get("domain") or "").strip().lower()
    if domain:
        states = [e for e in states if e.get("entity_id", "").split(".", 1)[0] == domain]
        if not states:
            return f"(no entities in domain '{domain}')"
    lines = [_ha_fmt(e) for e in states]
    return _clip(f"{len(lines)} entities:\n" + "\n".join(lines))


def _ha_get(args: dict, cfg) -> str:
    eid = (args.get("entity_id") or "").strip()
    if not eid:
        raise ToolError("entity_id is required")
    # eid goes straight into the /api/states/<eid> URL path, so a model- or
    # web-injection-supplied value like '../config' or 'x/../../foo' would
    # traverse to other HA API endpoints. Pin it to the real entity-id shape
    # (domain.object_id, lowercase alnum + underscore) before any HA call —
    # mirrors the domain/service guard on the write path (_ha_service).
    if not re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", eid):
        raise ToolError("entity_id must look like 'domain.object_id' using only "
                        "lowercase letters, digits and underscores (e.g. person.joe)")
    client = _ha_client(cfg)
    if not client.is_configured:
        raise ToolError("Home Assistant is not configured (set HA_URL and HA_TOKEN in .env).")
    e = client.state(eid)
    if not e:
        raise ToolError(f"no such entity '{eid}' (or HA unreachable)")
    attrs = e.get("attributes") or {}
    out = [_ha_fmt(e), f"last_changed: {e.get('last_changed', '?')}"]
    for k, v in list(attrs.items())[:12]:
        out.append(f"  {k}: {v}")
    return _clip("\n".join(out))


def _ha_search(args: dict, cfg) -> str:
    q = (args.get("query") or "").strip().lower()
    if not q:
        raise ToolError("query is required")
    client = _ha_client(cfg)
    if not client.is_configured:
        raise ToolError("Home Assistant is not configured (set HA_URL and HA_TOKEN in .env).")
    states = client.states()
    hits = [e for e in states
            if q in e.get("entity_id", "").lower()
            or q in str((e.get("attributes") or {}).get("friendly_name", "")).lower()]
    if not hits:
        return f"(no entities matching '{q}')"
    return _clip(f"{len(hits)} match '{q}':\n" + "\n".join(_ha_fmt(e) for e in hits))


def _remind(args: dict, cfg) -> str:
    """Push a reminder to another family member's devices — 'remind Sam I'll be
    late'. Only allowed if the target permitted reminders from the sender (or the
    sender is an adult and the target is their child). The permission check is the
    gate, so no approval is needed."""
    from . import profiles, push
    target = (args.get("to") or "").strip()
    message = (args.get("message") or "").strip()
    if not target or not message:
        raise ToolError("both 'to' (a family member's name) and 'message' are required")
    profs = profiles.load(cfg)
    match = next((n for n in profs if n.lower() == target.lower()), None)
    if not match:
        raise ToolError(f"no family member named '{target}' "
                        f"(known: {', '.join(profs) or 'none'})")
    sender = getattr(cfg, "_actor", "")
    if not profiles.can_remind(cfg, sender, match):
        raise ToolError(f"{match} hasn't allowed reminders from "
                        f"{sender or 'this profile'} — they can turn it on in "
                        "their Profile settings")
    who = sender or "the family"
    push.notify(cfg, f"Reminder from {who}", message, url="/", tag="reminder",
                to={match})
    return f'reminder pushed to {match}: "{message}"'


def _search_docs(args: dict, cfg) -> str:
    """Search the family's own indexed documents (manuals, warranties, medical/
    insurance papers, recipes) and return the most relevant passages with their
    source file — so answers are grounded in what the household actually has."""
    query = (args.get("query") or "").strip()
    if not query:
        raise ToolError("query is required")
    from . import docs
    from .providers import build_providers

    provider = build_providers(cfg)["ollama_local"]

    def _embed(text):
        return provider.embed(cfg.embed_model, text)
    embedder = _embed if getattr(cfg, "use_embeddings", True) else None
    hits = docs.DocStore(cfg, embedder=embedder).search(query, k=4)
    if not hits:
        return ("(no family documents match — the docs folder may be empty or "
                "not indexed yet. Ground your answer elsewhere; don't guess.)")
    blocks = [f"--- from {h['file']} ---\n{h['text'][:900]}" for h in hits]
    return _clip("FAMILY DOCUMENTS (ground your answer in these; cite the file):\n"
                 + "\n\n".join(blocks))


def _house_snapshot(args: dict, cfg) -> str:
    """ONE-call house overview: presence + the device categories people actually
    ask about (lights, media, locks, open doors/windows, climate), curated from a
    single states() read. Deterministic tool-fusion [smolagents]: 'what's the
    state of the house?' resolves in one call instead of the model orchestrating
    several ha_list/ha_search calls."""
    client = _ha_client(cfg)
    if not client.is_configured:
        raise ToolError("Home Assistant is not configured (set HA_URL and HA_TOKEN in .env).")
    states = client.states()
    if not states:
        raise ToolError("Home Assistant returned no entities (unreachable or empty).")
    by_domain: Dict[str, list] = {}
    for e in states:
        by_domain.setdefault(e.get("entity_id", "").split(".", 1)[0], []).append(e)

    def _names(entities) -> str:
        return ", ".join((e.get("attributes") or {}).get("friendly_name")
                         or e.get("entity_id", "") for e in entities)

    def _on(dom):
        return [e for e in by_domain.get(dom, []) if str(e.get("state", "")).lower() == "on"]

    lines = []
    people = by_domain.get("person", [])
    if people:
        home = [e for e in people if str(e.get("state", "")).lower() == "home"]
        away = [e for e in people if str(e.get("state", "")).lower() not in
                ("home", "unknown", "unavailable", "")]
        lines.append("WHO'S HOME: " + (_names(home) if home else "nobody detected home")
                     + (" | away: " + _names(away) if away else ""))
    if by_domain.get("light"):
        on = _on("light")
        lines.append(f"LIGHTS ON ({len(on)}/{len(by_domain['light'])}): " + (_names(on) or "none"))
    if _on("switch"):
        lines.append("SWITCHES ON: " + _names(_on("switch")))
    playing = [e for e in by_domain.get("media_player", [])
               if str(e.get("state", "")).lower() == "playing"]
    if playing:
        lines.append("PLAYING: " + _names(playing))
    if by_domain.get("lock"):
        unlocked = [e for e in by_domain["lock"] if str(e.get("state", "")).lower() == "unlocked"]
        lines.append("LOCKS: " + (_names(unlocked) + " UNLOCKED" if unlocked else "all locked"))
    open_dw = [e for e in by_domain.get("binary_sensor", [])
               if str(e.get("state", "")).lower() == "on"
               and (e.get("attributes") or {}).get("device_class")
               in ("door", "window", "garage_door", "opening")]
    if open_dw:
        lines.append("OPEN: " + _names(open_dw))
    for e in by_domain.get("climate", []):
        a = e.get("attributes") or {}
        cur, tgt = a.get("current_temperature"), a.get("temperature")
        lines.append(f"CLIMATE {a.get('friendly_name') or e.get('entity_id')}: {e.get('state')}"
                     + (f", now {cur}°" if cur is not None else "")
                     + (f" -> set {tgt}°" if tgt is not None else ""))
    if not lines:
        return ("House is reachable but nothing notable to report (no people, "
                "lights, media, locks or climate entities found).")
    return _clip("HOUSE SNAPSHOT\n" + "\n".join(lines))


def _ha_service(args: dict, cfg) -> str:
    from . import homeassistant as ha
    svc = (args.get("service") or "").strip()
    if "." not in svc:
        raise ToolError("service must look like 'light.turn_off' or 'media_player.pause'")
    domain, service = svc.split(".", 1)
    if not (re.fullmatch(r"[a-z0-9_]+", domain) and re.fullmatch(r"[a-z0-9_]+", service)):
        raise ToolError("service must be of the form '<domain>.<service>' using only "
                        "lowercase letters, digits and underscores")
    entity = (args.get("entity_id") or "").strip()
    data = args.get("data")
    data = dict(data) if isinstance(data, dict) else {}
    if entity:
        data = {"entity_id": entity, **data}
    client = ha.HomeAssistant(timeout=getattr(cfg, "request_timeout", None))
    if not client.is_configured:
        raise ToolError("Home Assistant is not configured (set HA_URL and HA_TOKEN).")
    try:
        client.call_service(domain, service, data)
    except Exception as exc:
        raise ToolError(f"HA service call failed: {exc}")
    return f"called {svc}" + (f" on {entity}" if entity else "")


def _tailscale_status(args: dict, cfg) -> str:
    from . import tailscale as ts
    s = ts.Tailscale().summary()
    if not s.get("installed"):
        raise ToolError("Tailscale is not installed on this machine.")
    if not s.get("running"):
        raise ToolError("Tailscale is installed but not connected (not signed in).")
    peers = s.get("peers", [])
    online = sum(1 for p in peers if p.get("online"))
    lines = [f"this node: {s.get('name')} {s.get('ip')} on tailnet {s.get('tailnet')}",
             f"{online}/{len(peers)} peer devices online:"]
    for p in peers:
        dot = "online " if p.get("online") else "offline"
        lines.append(f"  [{dot}] {p.get('name')} {p.get('ip')} ({p.get('os')})")
    return _clip("\n".join(lines))


def _plan(args: dict, cfg) -> str:
    """The durable to-do list that gives Lara follow-through. SAFE + actor-
    scoped: create/replace the plan, mark a step's status, or show it. The agent
    loop reads this plan to keep working through open steps instead of stalling."""
    from . import plan as planmod
    store = planmod.PlanStore(cfg)
    action = str(args.get("action", "show")).strip().lower()
    if action in ("set", "create", "new", "replace"):
        raw = args.get("steps") or []
        if isinstance(raw, str):
            # tolerate a newline/'; '-joined string as well as a real list
            raw = [s for s in re.split(r"[\n;]+", raw)]
        steps = [str(s).strip() for s in raw if str(s).strip()]
        if not steps:
            return "plan not set: provide 'steps' as a non-empty list of step texts"
        return store.set_steps(str(args.get("task", "")), steps).render()
    if action in ("update", "mark", "step"):
        try:
            sid = int(args.get("id"))
        except (TypeError, ValueError):
            return "plan update needs a numeric 'id' of the step to mark"
        status = str(args.get("status", "")).strip().lower()
        if status not in planmod.STATUSES:
            return ("plan update needs 'status' one of: "
                    + ", ".join(planmod.STATUSES))
        p = store.load()
        if not p.mark(sid, status, str(args.get("note", ""))):
            return f"no step with id {sid} in the current plan"
        store.save(p)
        return p.render()
    if action in ("clear", "done", "reset"):
        store.clear()
        return "plan cleared"
    return store.load().render()


def _git_branch(cfg) -> str:
    """Current git branch, falling back to the configured forge branch."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if out and out != "HEAD":
            return out
    except Exception:
        pass
    return str(getattr(cfg, "forge_branch", "") or "")


def _gitea_issue(args: dict, cfg) -> str:
    """File / list / comment on / close Gitea issues — the shared work queue that
    drives the CI/CD loop (see docs/cicd.md). SAFE: it touches the issue tracker,
    not the codebase or the family's systems."""
    from . import gitea as giteamod
    c = giteamod.GiteaClient(cfg)
    if not c.ok:
        return ("Gitea isn't configured (no GITEA_TOKEN or reachable remote) — "
                "I can't reach the issue tracker right now.")
    action = str(args.get("action", "list")).strip().lower()

    def _labels(v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    try:
        if action in ("file", "create", "new", "open"):
            title = str(args.get("title", "")).strip()
            if not title:
                return "issue not filed: a 'title' is required"
            labels = _labels(args.get("labels")) or ["selfdev"]
            body = str(args.get("body", "")).strip()
            # Set the branch the issue was found on in Gitea's Branch/Tag ref field
            # (its proper metadata slot), not buried in the body. self-dev works `test`.
            branch = str(args.get("branch", "")).strip() or _git_branch(cfg)
            iss = c.create_issue(title, body, labels, ref=branch)
            return f"filed issue #{iss.get('number')}: {iss.get('title')}"
        if action == "list":
            items = c.list_issues(labels=_labels(args.get("labels")) or None,
                                  state=str(args.get("state", "open")))
            if not items:
                return "no matching issues"
            return "\n".join(f"#{i.get('number')} [{i.get('state')}] {i.get('title')}"
                             for i in items[:30])
        if action == "comment":
            n, body = args.get("number"), str(args.get("comment", args.get("body", ""))).strip()
            if not n or not body:
                return "comment needs a 'number' and 'comment' text"
            c.comment_issue(int(n), body)
            return f"commented on issue #{n}"
        if action == "close":
            n = args.get("number")
            if not n:
                return "close needs an issue 'number'"
            c.close_issue(int(n))
            return f"closed issue #{n}"
        return "unknown action; use one of: file | list | comment | close"
    except (giteamod.GiteaError, ValueError) as exc:
        return f"Gitea issue action failed: {exc}"


TOOLS: Dict[str, Tool] = {
    "read_file": Tool("read_file", "Read a UTF-8 text file inside the workspace.",
                      ["path"], False, _read_file,
                      {"path": {"type": "string",
                                "description": "file path relative to the workspace"}}),
    "show": Tool("show", "Open a workspace file in the user's viewer pane so they "
                 "can SEE it next to your answer — a plan, a recipe, code, an "
                 "image. Use this instead of pasting a whole file into chat; "
                 "then refer to it briefly. Read-only.",
                 ["path", "note"], False, _show,
                 {"path": {"type": "string",
                           "description": "file path relative to the workspace"},
                  "note": {"type": "string",
                           "description": "optional one-line caption for what "
                           "you're showing and why"}}),
    "show_map": Tool(
        "show_map", "Show an INTERACTIVE MAP of places in the user's viewer "
        "pane — restaurant recommendations, parks, errands, anything that's "
        "places-on-a-map. Each pin gets your note plus Apple/Google Maps "
        "buttons so the user can navigate there in one tap. Research picks "
        "first (search), then map them. Calling show_map again REPLACES the "
        "map — do that as the user refines what they want.",
        ["title", "places"], False, _show_map,
        {"title": {"type": "string",
                   "description": "optional short heading, e.g. 'Taco night "
                   "options near home'"},
         "places": {"type": "array", "maxItems": 8,
                    "description": "the pins (max 8 — curate your best picks)",
                    "items": {"type": "object",
                              "properties": {
                                  "name": {"type": "string"},
                                  "address": {"type": "string",
                                              "description": "street address "
                                              "or at least 'town, ST' — used "
                                              "to place the pin accurately"},
                                  "note": {"type": "string",
                                           "description": "one line: why this "
                                           "pick, price range, kid-friendly..."},
                                  "url": {"type": "string",
                                          "description": "optional menu/site"},
                                  "lat": {"type": "number"},
                                  "lon": {"type": "number"}},
                              "required": ["name"]}}}),
    "generate_image": Tool(
        "generate_image", "CREATE an image from a text description (a drawing, "
        "picture, illustration, poster, logo...). Saves a PNG into the "
        "workspace and tells you its path — ALWAYS follow up with the show "
        "tool on that path so the user sees it. Costs a few cents per image, "
        "so one good detailed prompt beats many retries.",
        ["prompt", "size", "quality"], False, _generate_image,
        {"prompt": {"type": "string",
                    "description": "what to create — subject, style, colors, "
                    "mood; specific beats vague"},
         "size": {"type": "string", "enum": list(_IMG_SIZES),
                  "description": "optional; default 1024x1024 "
                  "(1536x1024 landscape, 1024x1536 portrait)"},
         "quality": {"type": "string", "enum": list(_IMG_QUALITIES),
                     "description": "optional; default medium — use low for "
                     "quick drafts, high only when the user asks for their "
                     "best version"}}),
    "ha_list": Tool("ha_list", "List Home Assistant entities and their current "
                    "states (the house's live sensors, lights, switches, "
                    "presence, media players). Optional 'domain' filters by type "
                    "(e.g. switch, sensor, person, media_player). Read-only.",
                    ["domain"], False, _ha_list,
                    {"domain": {"type": "string",
                                "description": "optional entity domain filter, "
                                "e.g. person, media_player, switch, sensor, light"}}),
    "search": Tool("search", "Search the web for CURRENT information (news, "
                   "prices, patch notes, recipes, docs — anything that may have "
                   "changed since training). Returns titles, URLs and snippets; "
                   "follow up with web_fetch on the best URL for full content. "
                   "For fast-moving topics put the current year in the query AND "
                   "set recency to get fresh results. Read-only.",
                   ["query", "count", "recency"], False, _search,
                   {"query": {"type": "string",
                              "description": "what to search for; include the "
                              "current year for anything time-sensitive"},
                    "count": {"type": "number",
                              "description": "how many results (1-10, default 6)"},
                    "recency": {"type": "string",
                                "description": "limit to recent results: one of "
                                "'day', 'week', 'month', 'year' (omit for all "
                                "time). Use for news/prices/latest-version asks."}}),
    "delegate": Tool("delegate", "Fan out 2-6 INDEPENDENT sub-tasks to parallel "
                     "worker agents and get all their findings back at once — "
                     "much faster than doing multi-part research yourself "
                     "sequentially. Each worker can search the web, fetch pages, "
                     "check weather, and read the house (read-only). Use for "
                     "multi-part questions ('compare X and Y', 'check A, B and "
                     "C'), then SYNTHESIZE the workers' reports into one answer. "
                     "Don't delegate single simple lookups. Read-only.",
                     ["tasks", "role"], False, _delegate,
                     {"tasks": {"type": "array", "items": {"type": "string"},
                                "description": "2-6 self-contained sub-task "
                                "strings, one per worker"},
                      "role": {"type": "string",
                               "description": "optional worker role: worker | "
                               "researcher | checker | summarizer"}}),
    "save_skill": Tool("save_skill", "Save a reusable PROCEDURE to your skill "
                       "library after you work out how to do a recurring task "
                       "(a multi-step recipe, a working command sequence, a "
                       "how-to). You'll automatically recall it next time a "
                       "similar task comes up. Needs name, description, and body "
                       "(the actual steps). Read-only to the outside world.",
                       ["name", "description", "body", "when"], False, _save_skill,
                       {"name": {"type": "string", "description": "short skill "
                                 "name, e.g. 'smoke-ribs-traeger'"},
                        "description": {"type": "string", "description": "one line "
                                        "on what it does + when to use it"},
                        "body": {"type": "string", "description": "the procedure "
                                 "itself — numbered steps or a recipe, in Markdown"},
                        "when": {"type": "string", "description": "optional: "
                                 "trigger phrase for when this applies"}}),
    "list_skills": Tool("list_skills", "List the procedures you've saved to your "
                        "skill library (name + description). Read-only.",
                        [], False, _list_skills, {}),
    "view_skill": Tool("view_skill", "Read the full body of one saved skill by "
                       "name (so you can extend the right existing skill rather "
                       "than duplicate it). Read-only.",
                       ["name"], False, _view_skill,
                       {"name": {"type": "string", "description": "the skill name "
                                 "from list_skills"}}),
    "search_chats": Tool("search_chats", "Search YOUR PAST CONVERSATIONS with "
                         "the operator (full-text over every chat transcript) "
                         "and return the matching exchanges. Use when they "
                         "reference something you discussed before ('like we "
                         "talked about', 'what did I tell you about X', 'that "
                         "recipe from the other day'). Read-only.",
                         ["query"], False, _search_chats,
                         {"query": {"type": "string",
                                    "description": "word or phrase to find in "
                                    "past conversations"}}),
    "schedule": Tool("schedule", "Create (or delete) a SCHEDULED JOB that runs a "
                     "prompt on a cron schedule and pushes the result to the "
                     "operator. Use for proactive help ('want me to watch the "
                     "radar during your cook?') AND for one-off reminders the "
                     "operator asks for ('remind me to check the oven in 1 "
                     "minute') — for those set once=true and a cron that matches "
                     "the target time (for 'in N minutes' use '* * * * *' with "
                     "once=true so it fires at the next minute then deletes "
                     "itself). action='add' needs name, cron (5 fields: min hour "
                     "day month weekday) and prompt; action='delete' needs just "
                     "name. This creates future autonomous behavior and REQUIRES "
                     "OPERATOR APPROVAL.",
                     ["action", "name", "cron", "prompt", "once", "notify"], True,
                     _schedule,
                     {"action": {"type": "string",
                                 "description": "'add' (default) or 'delete'"},
                      "name": {"type": "string",
                               "description": "short job name, e.g. 'radar-watch'"},
                      "cron": {"type": "string",
                               "description": "cron spec, e.g. '*/30 14-20 * * *' "
                               "= every 30 min from 2pm-8pm; '* * * * *' = the "
                               "next minute (use with once=true for 'in a minute')"},
                      "prompt": {"type": "string",
                                 "description": "what the job should do each run, "
                                 "as a prompt to yourself"},
                      "once": {"type": "boolean",
                               "description": "true = fire ONCE then self-delete "
                               "(use for one-off reminders like 'remind me in N "
                               "minutes'); false/omit = recurring"},
                      "notify": {"type": "string",
                                 "description": "optional: 'discord'"}}),
    "list_jobs": Tool("list_jobs", "List the currently scheduled jobs (name, "
                      "cron, prompt). Read-only — check before adding/deleting.",
                      [], False, _list_jobs, {}),
    "plan": Tool("plan", "Your durable TO-DO list for a multi-step task — how you "
                 "keep track and FOLLOW THROUGH instead of stalling. action='set' "
                 "with steps=[...] writes/replaces the plan; action='update' with "
                 "id + status marks a step ('pending'|'doing'|'done'|'blocked'); "
                 "action='show' displays it; action='clear' ends it. Set a plan at "
                 "the START of any task with ~3+ steps, mark each step 'doing' then "
                 "'done' as you go, and keep working through the OPEN steps — do NOT "
                 "ask the operator what to do next while steps remain; only stop for "
                 "an approval or info only they have. Read-only-safe.",
                 ["action", "task", "steps", "id", "status", "note"], False, _plan,
                 {"action": {"type": "string",
                             "description": "'set' | 'update' | 'show' | 'clear'"},
                  "task": {"type": "string",
                           "description": "one-line goal (with action='set')"},
                  "steps": {"type": "array", "items": {"type": "string"},
                            "description": "ordered step texts (with action='set')"},
                  "id": {"type": "integer",
                         "description": "step number to mark (with action='update')"},
                  "status": {"type": "string",
                             "description": "'pending'|'doing'|'done'|'blocked'"},
                  "note": {"type": "string",
                           "description": "optional short note on the step"}}),
    "gitea_issue": Tool("gitea_issue", "File, list, comment on, or close issues in "
                        "the project's Gitea tracker — the shared work queue for "
                        "ANVIL's own development. Use action='file' with a title (and "
                        "body/labels) to log a bug, idea, or something that needs "
                        "fixing; action='list' to see open work; 'comment' to add a "
                        "note to issue 'number'; 'close' when it's done. Default label "
                        "is 'selfdev'. Read-and-write to the issue tracker only — safe.",
                        ["action", "title", "body", "labels", "number", "comment", "state"],
                        False, _gitea_issue,
                        {"action": {"type": "string",
                                    "description": "'file' | 'list' | 'comment' | 'close'"},
                         "title": {"type": "string", "description": "issue title (with 'file')"},
                         "body": {"type": "string", "description": "issue description (with 'file')"},
                         "labels": {"type": "array", "items": {"type": "string"},
                                    "description": "labels, e.g. ['selfdev','bug'] (default ['selfdev'])"},
                         "number": {"type": "integer",
                                    "description": "issue number (with 'comment'/'close')"},
                         "comment": {"type": "string", "description": "comment text (with 'comment')"},
                         "state": {"type": "string",
                                   "description": "'open' (default) or 'closed' (with 'list')"}}),
    "weather": Tool("weather", "Current US weather from the National Weather "
                    "Service: forecast, ACTIVE ALERTS, an hour-by-hour "
                    "precipitation timeline, and a computed rain outlook (WHEN "
                    "rain is likely, with times). With no arguments it reports "
                    "for the operator's location (their phone's spot if they're "
                    "out, else home). Pass 'location' for a named place, or "
                    "lat+lon for exact coordinates. Read-only.",
                    ["location", "lat", "lon", "days"], False, _weather,
                    {"location": {"type": "string",
                                  "description": "optional US place or address, "
                                  "e.g. 'Denver CO' — omit for the operator's "
                                  "current location"},
                     "lat": {"type": "number",
                             "description": "optional latitude (use with lon)"},
                     "lon": {"type": "number",
                             "description": "optional longitude (use with lat)"},
                     "days": {"type": "number",
                              "description": "how many days ahead (1-5, default 1)"}}),
    "ha_get": Tool("ha_get", "Get the current state and attributes of ONE Home "
                   "Assistant entity by entity_id (e.g. person.joe, "
                   "switch.living_room). Read-only.",
                   ["entity_id"], False, _ha_get,
                   {"entity_id": {"type": "string",
                                  "description": "full entity id, e.g. person.joe"}}),
    "ha_search": Tool("ha_search", "Find Home Assistant entities whose id or "
                      "friendly name contains the query (e.g. 'sound bar', "
                      "'living room', 'door'). Read-only.",
                      ["query"], False, _ha_search,
                      {"query": {"type": "string",
                                 "description": "substring to match against entity "
                                 "ids and friendly names"}}),
    "remind": Tool("remind", "Push a REMINDER to another family member's phone, "
                   "e.g. 'remind Sam I'll be home late' or 'let the kids know "
                   "dinner is in 10 minutes'. Give the person's name ('to') and "
                   "the 'message'. Only works if that person allowed reminders "
                   "from you (a parent can always reach their child). Read-only "
                   "to the outside world.",
                   ["to", "message"], False, _remind,
                   {"to": {"type": "string", "description": "the family member's "
                           "name to remind"},
                    "message": {"type": "string", "description": "the reminder "
                                "text to send them"}}),
    "search_docs": Tool("search_docs", "Search the FAMILY'S OWN documents "
                        "(manuals, warranties, medical/insurance papers, recipes, "
                        "school forms the household has on file) and get the most "
                        "relevant passages with their source file. Use for "
                        "'what's our policy number', 'how do I reset the X', "
                        "'when's the warranty up' — anything that lives in the "
                        "family's papers rather than on the web. Read-only.",
                        ["query"], False, _search_docs,
                        {"query": {"type": "string",
                                   "description": "what to look up in the family's "
                                   "documents"}}),
    "house_snapshot": Tool("house_snapshot", "ONE-call overview of the whole "
                           "house: who's home, which lights/switches are on, "
                           "what's playing, locks, open doors/windows, and "
                           "climate. Use this FIRST for broad 'what's the state "
                           "of the house / is everything ok / anything on?' "
                           "questions instead of listing all entities. Read-only.",
                           [], False, _house_snapshot, {}),
    "list_dir": Tool("list_dir", "List a directory inside the workspace.",
                     ["path"], False, _list_dir,
                     {"path": {"type": "string",
                               "description": "directory path relative to the "
                               "workspace ('.' for the root)"}}),
    "web_fetch": Tool("web_fetch", "GET a URL and return its text (read-only).",
                      ["url"], False, _web_fetch,
                      {"url": {"type": "string",
                               "description": "http(s) URL to fetch"}}),
    "write_file": Tool("write_file", "Create/overwrite a file in the workspace.",
                       ["path", "content"], True, _write_file,
                       {"path": {"type": "string",
                                 "description": "file path relative to the workspace"},
                        "content": {"type": "string",
                                    "description": "full text content to write"}}),
    "shell": Tool("shell", "Run a command on THIS computer — the operator's own "
                  "machine (Crucible) — with their privileges. This is a REAL "
                  "system shell, not a sandbox; it starts in the workspace folder "
                  "but can reach anywhere on the machine (ssh-keygen, git, "
                  "checking services, files). Read-only commands may run "
                  "automatically; anything that changes the system asks first.",
                  ["cmd"], True, _shell,
                  {"cmd": {"type": "string",
                           "description": "the command to run, e.g. "
                           "'ssh-keygen -t ed25519 -f ~/.ssh/ember_key -N \"\"'"}}),
    "ha_service": Tool("ha_service", "CONTROL the operator's Home Assistant home: "
                       "call a service such as light.turn_off, switch.turn_on, "
                       "media_player.pause, cover.close, or tts.speak on an entity. "
                       "This CHANGES the real home and REQUIRES OPERATOR APPROVAL.",
                       ["service", "entity_id", "data"], True, _ha_service,
                       {"service": {"type": "string", "description": "domain.service, "
                                    "e.g. light.turn_off, media_player.pause"},
                        "entity_id": {"type": "string", "description": "target entity id, "
                                      "e.g. light.garage or media_player.living_room"},
                        "data": {"type": "object", "description": "optional extra service "
                                 "data, e.g. {\"brightness_pct\": 40} or {\"message\": \"hi\"}"}}),
    "tailscale_status": Tool("tailscale_status", "List the operator's Tailscale "
                             "tailnet devices and which are online (a presence "
                             "signal for their other machines/phones). Read-only.",
                             [], False, _tailscale_status, {}),
    "ssh": Tool("ssh", "Run a command on a REMOTE node over SSH (key auth via "
                "the operator's ~/.ssh; never prompts for a password). Use for "
                "checking or administering other machines on the network.",
                ["host", "cmd"], True, _ssh,
                {"host": {"type": "string",
                          "description": "ssh destination, e.g. user@192.168.50.10 "
                          "or a Host alias from ~/.ssh/config"},
                 "cmd": {"type": "string",
                         "description": "the command to run on the remote node"}}),
}

# Models often guess plausible-but-wrong tool names (list_files, file_read...).
# Resolve near-misses instead of failing the step.
ALIASES: Dict[str, str] = {
    "list_files": "list_dir", "listdir": "list_dir", "ls": "list_dir",
    "dir": "list_dir", "list_directory": "list_dir",
    "file_read": "read_file", "readfile": "read_file", "read": "read_file",
    "cat": "read_file", "open_file": "read_file",
    "file_write": "write_file", "writefile": "write_file", "write": "write_file",
    "save_file": "write_file", "create_file": "write_file",
    "bash": "shell", "run_shell": "shell", "execute": "shell", "exec": "shell",
    "run_command": "shell", "cmd": "shell", "terminal": "shell",
    "fetch": "web_fetch", "http_get": "web_fetch", "get_url": "web_fetch",
    "browse": "web_fetch", "curl": "web_fetch",
    "home_assistant": "ha_list", "ha_entities": "ha_list", "ha_states": "ha_list",
    "ha_call": "ha_service", "ha_control": "ha_service", "call_service": "ha_service",
    "turn_on": "ha_service", "turn_off": "ha_service",
    "list_entities": "ha_list", "get_entity": "ha_get", "ha_state": "ha_get",
    "ha_find": "ha_search", "search_entities": "ha_search",
    "remote": "ssh", "ssh_run": "ssh", "remote_shell": "ssh",
    "get_weather": "weather", "forecast": "weather", "weather_forecast": "weather",
    "check_weather": "weather", "weather_alerts": "weather",
    "web_search": "search", "search_web": "search", "google": "search",
    "duckduckgo": "search", "look_up": "search", "lookup": "search",
    "add_job": "schedule", "create_job": "schedule", "cron_job": "schedule",
    "set_reminder": "schedule", "remind": "schedule", "reminder": "schedule",
    "delete_job": "schedule", "cancel_job": "schedule",
    "jobs": "list_jobs", "show_jobs": "list_jobs",
    "chat_history": "search_chats", "past_chats": "search_chats",
    "search_history": "search_chats", "recall_conversation": "search_chats",
    "search_conversations": "search_chats",
    "spawn_workers": "delegate", "fan_out": "delegate", "swarm": "delegate",
    "parallel_tasks": "delegate", "hive": "delegate",
    "remember_skill": "save_skill", "learn_skill": "save_skill",
    "save_procedure": "save_skill", "skills": "list_skills", "show_skills": "list_skills",
}


def resolve_name(name: str) -> str:
    """Map a model-guessed tool name onto a real one (exact, alias, then fuzzy)."""
    n = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if n in TOOLS:
        return n
    if n in ALIASES:
        return ALIASES[n]
    import difflib
    close = difflib.get_close_matches(n, list(TOOLS), n=1, cutoff=0.75)
    return close[0] if close else name


def specs() -> List[dict]:
    return [{"name": t.name, "desc": t.desc, "args": t.args, "danger": t.danger}
            for t in TOOLS.values()]


def native_specs(only: Optional[set] = None) -> List[dict]:
    """Tool definitions in the native (OpenAI-style) function-calling format.

    This is what tool-trained models actually expect: Ollama injects these into
    the model's chat template, and the model responds with structured
    ``tool_calls`` — far more reliable than describing tools in prose.
    """
    optional = {"ha_list": {"domain"}, "list_dir": {"path"},
                "ha_service": {"entity_id", "data"},
                "weather": {"location", "lat", "lon", "days"},
                "search": {"count"},
                "generate_image": {"size", "quality"},
                "show_map": {"title"},
                "schedule": {"action", "cron", "prompt", "notify"},
                "delegate": {"role"}, "save_skill": {"when", "description"}}
    out = []
    for t in TOOLS.values():
        if only is not None and t.name not in only:
            continue          # narrowed context (hive workers): subset only
        props = t.params or {a: {"type": "string"} for a in t.args}
        required = [a for a in t.args if a not in optional.get(t.name, set())]
        desc = t.desc + (" REQUIRES OPERATOR APPROVAL before it runs."
                         if t.danger else "")
        out.append({"type": "function",
                    "function": {"name": t.name, "description": desc,
                                 "parameters": {"type": "object",
                                                "properties": props,
                                                "required": required}}})
    return out


def specs_text() -> str:
    lines = []
    for t in TOOLS.values():
        flag = " [needs approval]" if t.danger else ""
        lines.append(f"- {t.name}({', '.join(t.args)}): {t.desc}{flag}")
    return "\n".join(lines)


def is_danger(name: str) -> bool:
    t = TOOLS.get(resolve_name(name))
    return bool(t and t.danger)


# --------------------------------------------------------------------------- #
# Autonomy policy — how much privilege Lara has (ask / trusted / auto)
# --------------------------------------------------------------------------- #
# Read-only commands: safe to run WITHOUT approval in 'trusted' mode. Matching
# is first-word/prefix based and refuses anything with shell chaining or
# redirection — `git status; rm x` never qualifies. This is a *convenience*
# tier for diagnostics; the hard denylist in _shell still applies to all modes.
READONLY_PREFIXES = (
    "dir", "ls", "type", "cat", "tree", "where", "whoami", "hostname",
    "systeminfo", "ver", "date", "ipconfig", "ifconfig", "ping", "tracert",
    "nslookup", "netstat", "arp", "tasklist", "echo",
    "get-",                       # PowerShell Get-* cmdlets are read-only
    "test-path", "test-connection", "test-netconnection", "measure-object",
    "git status", "git log", "git diff", "git show", "git branch", "git remote",
    "docker ps", "docker images", "docker logs", "docker stats",
    "ollama ps", "ollama list", "ollama show",
    "tailscale status", "tailscale ip", "python --version", "pip list",
)
# Any of these anywhere in the command disqualifies it from the read-only tier
# (chaining, redirection, substitution — could smuggle a second, writing verb).
_SHELL_META = (";", "&", "|", ">", "<", "`", "$(", "${", "\n", "%{")

# A few READONLY_PREFIXES name a git verb that is read-only ONLY in its listing
# form: `git branch`/`git remote` also DELETE/RENAME/ADD (`git branch -D foo`,
# `git remote add evil url`, `git remote set-url ...`). Those are state-changing,
# so they must NOT auto-run under the read-only tier. Disqualify these prefixes
# the moment a mutating subcommand/flag appears — bare listing forms
# (`git branch`, `git branch -a/-r/-v/--list`, `git remote`, `git remote -v`)
# stay read-only. Keyed on the prefix so only the ambiguous verbs are guarded.
_RO_PREFIX_WRITE_TOKENS = {
    "git branch": ("-d", "-D", "--delete", "-m", "-M", "--move", "-c", "-C",
                   "--copy", "--edit-description", "--set-upstream-to",
                   "--unset-upstream", "-u", "-f", "--force"),
    "git remote": ("add", "remove", "rm", "rename", "set-url", "set-head",
                   "set-branches", "prune", "update"),
}


def is_readonly_cmd(cmd: str) -> bool:
    low = " ".join((cmd or "").lower().split())    # normalise whitespace
    if not low or any(m in low for m in _SHELL_META):
        return False
    for prefix, write_tokens in _RO_PREFIX_WRITE_TOKENS.items():
        if low == prefix or low.startswith(prefix + " "):
            rest = low[len(prefix):].split()
            # A mutating flag/subcommand, OR (for `git branch`) a bare NAME arg
            # that would CREATE a branch, disqualifies the read-only tier.
            if any(tok in write_tokens for tok in rest):
                return False
            if prefix == "git branch" and any(
                    not tok.startswith("-") for tok in rest):
                return False    # `git branch newname [start]` creates a branch
            break
    return any(low == p or low.startswith(p if p.endswith("-") else p + " ")
               for p in READONLY_PREFIXES)


def _allowlist_path(cfg) -> Path:
    return Path(getattr(cfg, "memory_dir", "memory")) / "shell_allowlist.json"


def allowlist(cfg) -> List[str]:
    try:
        val = json.loads(_allowlist_path(cfg).read_text("utf-8"))
        return [str(v) for v in val] if isinstance(val, list) else []
    except (OSError, json.JSONDecodeError, ValueError):
        return []


def allowlist_add(cfg, cmd: str) -> None:
    """Operator taught us a command via 'Always allow' — remember it (exact
    string match only: what was approved is what runs free, nothing broader)."""
    cmd = " ".join((cmd or "").split())
    if not cmd:
        return
    entries = allowlist(cfg)
    if cmd in entries:
        return
    entries.append(cmd)
    from . import config as cfgmod
    p = _allowlist_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfgmod.atomic_write(p, json.dumps(entries, indent=1))


def allowlist_match(cfg, cmd: str) -> bool:
    return " ".join((cmd or "").split()) in allowlist(cfg)


def needs_approval(name: str, args: dict, cfg, adult: bool = True) -> bool:
    """The privilege gate.

    IDENTITY FIRST: a non-adult (a minor, or an unverified shared session) can
    never auto-run a danger tool — autonomy modes and the taught allowlist apply
    only to a verified ADULT. So a child holding the tablet can chat and sense
    the house, but any state-changing action stops for an adult (who must then
    authorise it — see the approval flow). This closes the "kid unlocks the
    door" hole.

    For an adult, by autonomy mode:
    * ``ask``     — every danger tool pauses for approval (maximum caution)
    * ``trusted`` — read-only shell commands and operator-taught ('Always
                    allow') commands run free; all other danger tools pause
    * ``auto``    — nothing pauses (the hard denylist still refuses
                    catastrophes). The operator owns the consequences.
    """
    t = TOOLS.get(resolve_name(name))
    if not t or not t.danger:
        return False
    if not adult:
        return True                      # minors/unverified: always gate danger
    mode = str(getattr(cfg, "autonomy", "trusted") or "trusted").lower()
    if mode == "auto":
        return False
    if mode == "ask":
        return True
    if t.name == "shell" and isinstance(args, dict):
        cmd = args.get("cmd") or ""
        if is_readonly_cmd(cmd) or allowlist_match(cfg, cmd):
            return False
    if t.name == "write_file":
        # Trusted mode: writing INSIDE the hard-sandboxed workspace runs free —
        # _safe_path blocks everything outside it, so the blast radius is the
        # scratch folder. Gating this threw an approval card at "make me a
        # summer-safety doc", broke Lara's write-then-show plan mid-turn, and
        # made a trivially safe request feel like a security ceremony. Minors
        # are still gated (the identity check above runs first); 'ask' mode
        # still gates everything.
        return False
    return True


def run_tool(name: str, args: dict, cfg) -> str:
    t = TOOLS.get(resolve_name(name))
    if not t:
        raise ToolError(f"unknown tool: {name}")
    try:
        return t.run(args or {}, cfg)
    except Exception as exc:
        # A tool blowing up is a process incident — the self-awareness layer records it
        # so triage can decide if it's a harness bug worth filing. Never mask the error.
        try:
            from . import introspect
            introspect.record("tool-error", f"tool:{name}",
                              f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
