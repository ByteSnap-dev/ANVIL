"""Tailscale integration — reach Lara over the tailnet, and let her sense it.

A thin wrapper over the local ``tailscale`` CLI (no pip dependency). The command
runner is dependency-injected so tests run fully offline. Two jobs:

* **Connectivity** — read the node's tailnet IP so ANVIL can bind to it, and kick
  off login (returning the auth URL for a one-click sign-in).
* **Sensing** — list tailnet devices and who's online, so Lara can answer
  "is Joe's laptop on the tailnet?" the same way she reads the house.

Everything degrades gracefully: tailscale absent / not logged in / CLI error all
return a clear "not available" shape instead of raising.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from typing import Callable, Dict, List, Optional, Tuple

AUTH_RE = re.compile(r"https://login\.tailscale\.com/\S+")

Runner = Callable[[List[str], int], Tuple[int, str, str]]


def _default_runner(args: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    p = subprocess.run(["tailscale", *args], capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def extract_auth_url(text: str) -> str:
    m = AUTH_RE.search(text or "")
    return m.group(0) if m else ""


class Tailscale:
    def __init__(self, runner: Optional[Runner] = None):
        self._run = runner or _default_runner

    def is_installed(self) -> bool:
        if self._run is _default_runner:
            return shutil.which("tailscale") is not None
        try:
            code, _, _ = self._run(["version"], 5)
            return code == 0
        except Exception:
            return False

    def status(self) -> dict:
        """Parsed ``tailscale status --json``; {} when unavailable / not up."""
        try:
            code, out, err = self._run(["status", "--json"], 8)
        except Exception:
            return {}
        if code != 0 or not (out or "").strip():
            return {}
        try:
            v = json.loads(out)
            return v if isinstance(v, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}

    def summary(self) -> dict:
        """Operator/agent-facing snapshot of the tailnet."""
        if not self.is_installed():
            return {"installed": False, "running": False}
        st = self.status()
        if not st:
            return {"installed": True, "running": False}
        me = st.get("Self") or {}
        peers = []
        for p in (st.get("Peer") or {}).values():
            ips = p.get("TailscaleIPs") or []
            name = (p.get("DNSName") or "").split(".")[0] or p.get("HostName", "")
            peers.append({"name": name, "host": p.get("HostName", ""),
                          "ip": ips[0] if ips else "", "os": p.get("OS", ""),
                          "online": bool(p.get("Online"))})
        peers.sort(key=lambda d: (not d["online"], (d["name"] or "").lower()))
        my_ips = me.get("TailscaleIPs") or []
        return {
            "installed": True,
            "running": st.get("BackendState") == "Running",
            "state": st.get("BackendState", ""),
            "ip": my_ips[0] if my_ips else "",
            "name": (me.get("DNSName") or "").split(".")[0] or me.get("HostName", ""),
            "tailnet": (st.get("CurrentTailnet") or {}).get("Name", "")
            or st.get("MagicDNSSuffix", ""),
            "peers": peers,
        }

    def self_ip(self) -> str:
        return self.summary().get("ip", "")

    def up(self, timeout: int = 12) -> dict:
        """Start login. Returns {'ok':True} if already connected, or
        {'auth_url': ...} with the URL the operator must visit. Leaves the login
        running so it completes once they sign in."""
        s = self.summary()
        if not s.get("installed"):
            return {"error": "tailscale is not installed on this machine"}
        if s.get("running"):
            return {"ok": True, "already": True, "ip": s.get("ip", "")}
        try:
            proc = subprocess.Popen(["tailscale", "up"], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
        except Exception as exc:
            return {"error": str(exc)}
        end = time.time() + timeout
        while time.time() < end:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                url = extract_auth_url(line)
                if url:
                    return {"auth_url": url}
            elif proc.poll() is not None:
                break
            else:
                time.sleep(0.2)
        if proc.poll() == 0:
            return {"ok": True}
        return {"ok": True, "note": "login started; check status shortly"}
