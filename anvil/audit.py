"""Security posture audit — a checklist for the ways an ANVIL install can drift
into being exposed. Inspired by OpenClaw's ``security audit``: instead of a
one-time setup you hope stays right, a repeatable check you can run any time.

Reports findings at three levels — HIGH (fix now), WARN (know the trade-off),
OK (all good) — each with a concrete fix. Read-only: it never changes anything.
Run:  python -m anvil audit
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

# (level, title, detail, fix)  — level in {"HIGH","WARN","OK"}
Finding = Tuple[str, str, str, str]


def run_audit(cfg) -> List[Finding]:
    out: List[Finding] = []

    # 1) Off-box binding with no auth — anyone who can reach the port drives
    #    Lara (and can approve her danger tools). The tailnet is the only wall.
    bind = str(getattr(cfg, "bind_host", "127.0.0.1"))
    # Login is opt-in; if no adult has set a password, the port is still wide open.
    try:
        from . import profiles
        _auth_on = profiles.auth_on(cfg)
    except Exception:
        _auth_on = False
    _noauth = " (and no login is set — set a password in Setup → Family profiles)"
    if bind in ("0.0.0.0", "::"):
        lvl = "WARN" if _auth_on else "HIGH"
        out.append((lvl, "Server bound to all interfaces",
                    f"bind_host={bind!r} — anyone who can reach this port can load "
                    "ANVIL" + ("" if _auth_on else _noauth) + ".",
                    "Bind to 127.0.0.1 and reach it over Tailscale (tailscale "
                    "serve), or to your tailnet IP only — never a LAN/public IP."))
    elif bind not in ("127.0.0.1", "localhost", "::1"):
        out.append(("WARN", "Server bound off-loopback",
                    f"bind_host={bind!r}. Fine if that's a tailnet IP behind "
                    "Tailscale ACLs" + ("; a login is also set." if _auth_on
                    else " — and ANVIL has no login set yet."),
                    "Confirm this address is tailnet-only, not LAN-reachable."))
    else:
        out.append(("OK", "Server bound to loopback", f"bind_host={bind!r}.", ""))

    # 2) Autonomy = auto means danger tools run with NO approval prompt.
    autonomy = str(getattr(cfg, "autonomy", "trusted")).lower()
    if autonomy == "auto":
        out.append(("WARN", "Autonomy is 'auto'",
                    "Lara runs shell/HA-write/ssh with NO approval — only the "
                    "hard denylist stops a catastrophe.",
                    "Use 'trusted' (read-only commands free, the rest ask) unless "
                    "you've set up a low-privilege account for her shell."))
    else:
        out.append(("OK", f"Autonomy is '{autonomy}'",
                    "Danger actions are gated.", ""))

    # 3) Tailscale FUNNEL exposes a service to the PUBLIC internet — forbidden
    #    for ANVIL (serve = tailnet-only is fine; funnel is not).
    try:
        from . import tailscale
        status = tailscale.Tailscale().summary()
        raw = str(status).lower()
        if "funnel" in raw and "true" in raw:
            out.append(("HIGH", "Tailscale Funnel appears active",
                        "Funnel publishes to the PUBLIC internet — that would put "
                        "Lara (no auth) on the open web.",
                        "Use `tailscale serve` (tailnet-only), never `tailscale "
                        "funnel`. Run `tailscale funnel status` to check/disable."))
        else:
            out.append(("OK", "No Tailscale Funnel detected",
                        "Off-box access should be tailnet-only.", ""))
    except Exception:
        pass       # tailscale not installed / not reachable — nothing to assert

    # 4) .env (secrets) must be gitignored so a key never gets committed.
    root = Path(getattr(cfg, "memory_dir", "memory")).resolve().parent
    env_p = root / ".env"
    gi = root / ".gitignore"
    if env_p.exists():
        ignored = gi.exists() and any(
            ln.strip().rstrip("/") in (".env", "/.env", "*.env")
            for ln in gi.read_text("utf-8", "replace").splitlines())
        if not ignored:
            out.append(("HIGH", ".env not gitignored",
                        f"{env_p} holds API keys/tokens but isn't in .gitignore "
                        "— a commit could leak them.",
                        "Add '.env' to .gitignore and rotate any key already "
                        "committed."))
        else:
            out.append(("OK", ".env is gitignored", "Secrets stay out of git.", ""))

    # 4b) If minors exist, adults must have PINs — otherwise the identity gate
    #     is bypassable (a kid just picks an adult profile).
    try:
        from . import profiles
        profs = profiles.load(cfg)
        if any(not p.is_adult for p in profs.values()):
            unprotected = [p.name for p in profs.values() if p.is_adult and not p.has_pin]
            if unprotected:
                out.append(("HIGH", "Adult profile has no PIN (minors present)",
                            f"{', '.join(unprotected)} can be selected by anyone — a "
                            "child could pick it and gain danger-tool approval.",
                            "Set a PIN on every adult profile in Setup → Family "
                            "profiles so the identity gate actually holds."))
            else:
                out.append(("OK", "Family profiles: adults PIN-protected",
                            "Minors can't approve danger actions.", ""))
    except Exception:
        pass

    # 5) synthesis_mode = cloud ships HOUSE data to the cloud — a conscious
    #    trade-off worth surfacing, not a bug.
    synth = str(getattr(cfg, "synthesis_mode", "balanced")).lower()
    if synth == "cloud":
        out.append(("WARN", "Synthesis mode 'cloud'",
                    "House/family data can be sent to cloud models for answer "
                    "synthesis (you allowed this).",
                    "Use 'balanced' to keep house-touching turns local."))

    return out


def format_report(findings: List[Finding]) -> str:
    order = {"HIGH": 0, "WARN": 1, "OK": 2}
    findings = sorted(findings, key=lambda f: order.get(f[0], 3))
    icon = {"HIGH": "[HIGH]", "WARN": "[warn]", "OK": "[ ok ]"}
    lines = ["ANVIL security audit", "=" * 20]
    for level, title, detail, fix in findings:
        lines.append(f"{icon.get(level, '[?]')} {title}")
        if level != "OK":
            lines.append(f"        {detail}")
            if fix:
                lines.append(f"        fix: {fix}")
    highs = sum(1 for f in findings if f[0] == "HIGH")
    warns = sum(1 for f in findings if f[0] == "WARN")
    lines.append("-" * 20)
    lines.append(f"{highs} high, {warns} warnings, "
                 f"{sum(1 for f in findings if f[0]=='OK')} ok")
    return "\n".join(lines)
