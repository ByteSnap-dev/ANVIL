"""Family member profiles — the identity layer under the danger gate.

The PWA is shared by a whole family, but ANVIL's danger-tool approval was
identity-BLIND: whatever autonomy mode / taught allowlist an adult set applied
to *anyone* holding the tablet — a child could approve a shell command or
"unlock the front door". This module gives Lara a notion of WHO is acting.

A profile is ``{name, role: adult|minor, pin_hash}``. Adults may carry a PIN;
minors never can. The rule the danger gate enforces (see ``tools.needs_approval``
and the approval flow): **only a verified ADULT can auto-run or approve a danger
action.** A minor (or an unverified session) can chat, sense the house, and use
read-only tools — but every state-changing action stops and needs an adult.

Backward compatible: with no profiles.json, ANVIL synthesises a single adult
"operator" with no PIN, so an existing single-user install behaves exactly as
before. Family safety is opt-in — it engages the moment you add a minor profile.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config as cfgmod

_LOCK = threading.Lock()
_PBKDF2_ROUNDS = 120_000

# Per-file parse cache for _load_raw(): path -> (mtime_ns, size, raw_dict).
# A single chat turn resolves identity many times over (auth_on, _resolve_profile,
# _actor_name, _bind_actor, _session_adult, the safety block's minor check, the
# approval path) — each call read+parsed profiles.json AND re-ran the invariant
# pass from scratch, so a turn did ~8-12 full re-parses of the same small file.
# Cache the raw parse keyed on stat(); any write (save/set_push_allow go through
# atomic_write's os.replace, and a direct edit) changes mtime/size and misses the
# cache naturally — the exact invariant memory.py's _NOTE_CACHE and skills.py's
# _SKILL_CACHE already rely on. Callers only READ the returned dict (all mutation
# happens on the fresh Profile objects load() builds), so sharing it is safe.
_RAW_CACHE: Dict[str, tuple] = {}


def _path(cfg) -> Path:
    return Path(getattr(cfg, "memory_dir", "memory")) / "profiles.json"


def _norm_username(u: str) -> str:
    """Usernames are lowercase alphanumeric (dots/dashes/underscores kept):
    'Alex' and ' alex ' are the same account at the login form."""
    import re as _re
    return _re.sub(r"[^a-z0-9._-]", "", (u or "").strip().lower())[:32]


def hash_pin(pin: str) -> str:
    """pbkdf2-hmac-sha256; stored as pbkdf2$rounds$salt_hex$hash_hex."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_pin(stored: str, pin: str) -> bool:
    try:
        _, rounds, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"),
                                 bytes.fromhex(salt_hex), int(rounds))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


class Profile:
    """A family member. Two ORTHOGONAL axes:
      * admin vs user  — exactly ONE admin (the first profile ever made): the
        household manager who can create/edit profiles and system settings. An
        admin is ALWAYS an adult.
      * adult vs minor — applies to the user accounts; every non-admin is a user
        that is either an adult (can approve danger actions) or a child.
    """

    def __init__(self, name: str, role: str = "adult", pin_hash: str = "",
                 admin: bool = False, push_allow=None,
                 username: str = "", password_hash: str = ""):
        self.name = name
        self.admin = bool(admin)
        # An admin is always an adult; otherwise honour the stored role.
        self.role = "adult" if self.admin else (
            role if role in ("adult", "minor") else "adult")
        self.pin_hash = pin_hash
        # Accounts (2026-07): ``username`` + ``password_hash`` are for LOGIN;
        # the display ``name`` remains the ACTOR KEY everywhere (conversations,
        # memory owners, approvals) so account changes never orphan data. The
        # PIN survives as the QUICK adult elevation for danger approvals on a
        # shared/kid device — password signs you in, PIN vouches you're adult.
        self.username = _norm_username(username or name)
        self.password_hash = password_hash
        # Other profiles this person lets send them reminders via Lara (e.g.
        # "remind Sam I'll be late"). A minor implicitly allows every adult.
        self.push_allow = [str(x) for x in (push_allow or []) if str(x).strip()]

    @property
    def is_adult(self) -> bool:
        return self.role == "adult"

    @property
    def is_admin(self) -> bool:
        return self.admin

    @property
    def has_pin(self) -> bool:
        return bool(self.pin_hash)

    @property
    def has_password(self) -> bool:
        return bool(self.password_hash)

    def public(self) -> dict:
        return {"name": self.name, "role": self.role,
                "admin": self.admin, "has_pin": self.has_pin,
                "username": self.username, "has_password": self.has_password,
                "push_allow": list(self.push_allow)}


def _load_raw(cfg) -> dict:
    p = _path(cfg)
    key = str(p)
    try:
        st = p.stat()
    except OSError:
        _RAW_CACHE.pop(key, None)      # file gone -> single-user default path
        return {}
    with _LOCK:
        hit = _RAW_CACHE.get(key)
        if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
            return hit[2]
    try:
        raw = json.loads(p.read_text("utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    with _LOCK:
        _RAW_CACHE[key] = (st.st_mtime_ns, st.st_size, raw)
    return raw


def load(cfg) -> Dict[str, Profile]:
    raw = _load_raw(cfg)
    out: Dict[str, Profile] = {}
    for p in raw.get("profiles", []):
        if isinstance(p, dict) and p.get("name"):
            out[p["name"]] = Profile(p["name"], p.get("role", "adult"),
                                     p.get("pin_hash", ""), p.get("admin", False),
                                     push_allow=p.get("push_allow", []),
                                     username=p.get("username", ""),
                                     password_hash=p.get("password_hash", ""))
    if not out:
        # Backward-compatible default: one adult admin, no PIN. An existing
        # single-user install keeps behaving exactly as before.
        out["operator"] = Profile("operator", "adult", "", admin=True)
        return out
    # Enforce the invariant: EXACTLY ONE admin. Pre-admin installs (profiles.json
    # written before this feature) have none marked -> the FIRST profile (the one
    # made first, insertion order) becomes admin. Extra admins are demoted.
    admins = [p for p in out.values() if p.admin]
    if not admins:
        first = next(iter(out.values()))
        first.admin = True
        first.role = "adult"
    elif len(admins) > 1:
        kept = False
        for p in out.values():
            if p.admin:
                if kept:
                    p.admin = False
                else:
                    kept = True
    return out


def admin_name(cfg) -> str:
    """The single admin profile's name (the household manager)."""
    for n, p in load(cfg).items():
        if p.admin:
            return n
    return ""


def default_name(cfg) -> str:
    """The profile a fresh/unidentified session acts as. If any minor exists,
    default to the LEAST-privileged profile (fail safe) so a handed-over tablet
    doesn't start out with adult authority."""
    raw = _load_raw(cfg)
    d = raw.get("default")
    profs = load(cfg)
    if d and d in profs:
        return d
    minors = [n for n, p in profs.items() if not p.is_adult]
    if minors:
        return sorted(minors)[0]
    return sorted(profs)[0]


def get(cfg, name: str) -> Optional[Profile]:
    return load(cfg).get(name)


def save(cfg, profiles: List[dict], default: str = "") -> None:
    # The admin is NOT settable from the UI (no privilege escalation): it stays
    # with whoever is currently admin (matched by name). If none carries over
    # (fresh setup, or the admin was somehow dropped), the FIRST entry becomes
    # admin. Admin is forced adult; exactly one admin survives.
    cur_admin = admin_name(cfg) if _load_raw(cfg).get("profiles") else ""
    existing = load(cfg)                     # to carry over personal settings
    clean = []
    for p in profiles:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        name = p["name"]
        # push_allow / username / password_hash are PERSONAL settings each
        # member owns — the admin editing the family roster must not wipe
        # them. Preserve unless explicitly given.
        allow = p.get("push_allow")
        if allow is None:
            allow = list(existing[name].push_allow) if name in existing else []
        prev = existing.get(name)
        username = _norm_username(p.get("username") or
                                  (prev.username if prev else name))
        pw_hash = p.get("password_hash")
        if pw_hash is None:
            pw_hash = prev.password_hash if prev else ""
        clean.append({"name": name,
                      "role": "minor" if p.get("role") == "minor" else "adult",
                      "pin_hash": p.get("pin_hash", ""),
                      "admin": bool(cur_admin) and name == cur_admin,
                      "username": username, "password_hash": pw_hash,
                      "push_allow": [str(x) for x in allow if str(x).strip()]})
    if clean and not any(c["admin"] for c in clean):
        clean[0]["admin"] = True
    admin_seen = False
    for c in clean:
        if c["admin"]:
            if admin_seen:
                c["admin"] = False
            else:
                admin_seen = True
                c["role"] = "adult"        # admin is always an adult
    with _LOCK:
        _path(cfg).parent.mkdir(parents=True, exist_ok=True)
        cfgmod.atomic_write(_path(cfg),
                            json.dumps({"profiles": clean, "default": default},
                                       indent=1))


def set_push_allow(cfg, name: str, allow: List[str]) -> bool:
    """Set who may send THIS person reminders (their personal choice). Preserves
    everyone else's settings + roles. Returns True if the profile exists."""
    profs = load(cfg)
    if name not in profs:
        return False
    valid = set(profs)
    rows = []
    for n, p in profs.items():
        row = {"name": n, "role": p.role, "pin_hash": p.pin_hash,
               "push_allow": list(p.push_allow)}
        if n == name:
            row["push_allow"] = [a for a in allow if a in valid and a != name]
        rows.append(row)
    save(cfg, rows, default=_load_raw(cfg).get("default", ""))
    return True


def can_remind(cfg, sender: str, target: str) -> bool:
    """May ``sender`` have Lara push a reminder to ``target``? Yes if the target
    explicitly allowed the sender, OR the target is a MINOR and the sender is an
    adult (parents can always reach their kids). Never to yourself-only checks."""
    if not sender or not target or sender == target:
        return bool(sender and sender == target)   # reminding yourself is fine
    profs = load(cfg)
    tp, sp = profs.get(target), profs.get(sender)
    if not tp or not sp:
        return False
    if not tp.is_adult and sp.is_adult:
        return True                                 # parent -> child, automatic
    return sender in tp.push_allow


def remind_targets(cfg, sender: str) -> List[str]:
    """Profiles ``sender`` is allowed to remind (for the tool + UI)."""
    return sorted(n for n in load(cfg) if n != sender and can_remind(cfg, sender, n))


def any_minor(cfg) -> bool:
    return any(not p.is_adult for p in load(cfg).values())


def auth_on(cfg) -> bool:
    """Is login required? Opt-in: engages once an adult has a password (the
    account system) or a PIN (legacy). Until then ANVIL runs with no login
    wall (single-user friendly)."""
    return any(p.is_adult and (p.has_password or p.has_pin)
               for p in load(cfg).values())


def needs_setup(cfg) -> bool:
    """True on a genuinely FRESH install (no profiles file / empty roster) —
    the first visit should run the setup wizard, not a login form."""
    return not [p for p in _load_raw(cfg).get("profiles", [])
                if isinstance(p, dict) and p.get("name")]


def find_by_username(cfg, username: str):
    """Case/format-insensitive account lookup; falls back to display-name match
    so pre-account profiles ('Alex') can sign in as 'alex'."""
    u = _norm_username(username)
    if not u:
        return None
    for p in load(cfg).values():
        if p.username == u:
            return p
    for p in load(cfg).values():
        if _norm_username(p.name) == u:
            return p
    return None


def authenticate(cfg, name_or_username: str, password: str):
    """Return the Profile when the credential is valid, else None.

    Account order of proof: the PASSWORD when one is set; else the legacy PIN
    (transition path — the profile predates passwords, so their PIN still
    signs them in until they set a password); a credential-less MINOR signs
    in with an empty password. A credential-less ADULT can never log in.
    Timing: PBKDF2 verify runs on every real attempt; misses return via the
    same shape (no username enumeration in the error text — server's job)."""
    p = get(cfg, name_or_username) or find_by_username(cfg, name_or_username)
    if not p:
        return None
    if p.has_password:
        return p if verify_pin(p.password_hash, password or "") else None
    if p.has_pin:
        return p if verify_pin(p.pin_hash, password or "") else None
    return p if (not p.is_adult and not (password or "").strip()) else None


def set_password(cfg, name: str, password: str) -> bool:
    """Set/replace one member's login password (their other settings survive).
    Returns False for an unknown profile or an empty password."""
    if not (password or "").strip():
        return False
    profs = load(cfg)
    if name not in profs:
        return False
    rows = []
    for n, p in profs.items():
        row = {"name": n, "role": p.role, "pin_hash": p.pin_hash,
               "username": p.username, "password_hash": p.password_hash,
               "push_allow": list(p.push_allow)}
        if n == name:
            row["password_hash"] = hash_pin(password)
        rows.append(row)
    save(cfg, rows, default=_load_raw(cfg).get("default", ""))
    return True


def create_household(cfg, admin: dict, members=None) -> bool:
    """First-run wizard: create the roster in one shot — the FIRST entry is
    the admin (save() enforces admin=first + adult). Refuses to run when
    accounts already exist (the wizard must never clobber a real household)."""
    if not needs_setup(cfg):
        return False
    rows = []
    a_name = (admin.get("name") or "").strip()
    a_pw = admin.get("password") or ""
    if not a_name or not (a_pw or "").strip():
        return False
    rows.append({"name": a_name, "role": "adult",
                 "username": _norm_username(admin.get("username") or a_name),
                 "password_hash": hash_pin(a_pw),
                 "pin_hash": (hash_pin(admin["pin"])
                              if (admin.get("pin") or "").strip() else "")})
    for m in (members or []):
        n = (m.get("name") or "").strip()
        if not n or n == a_name:
            continue
        rows.append({"name": n,
                     "role": "minor" if m.get("role") == "minor" else "adult",
                     "username": _norm_username(m.get("username") or n),
                     "password_hash": (hash_pin(m["password"])
                                       if (m.get("password") or "").strip() else ""),
                     "pin_hash": ""})
    save(cfg, rows, default=a_name)
    return True
