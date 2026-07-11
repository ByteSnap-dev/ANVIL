"""Web Push (RFC 8291 + VAPID / RFC 8292) implemented on ``cryptography``.

iOS 16.4+ delivers notifications to an installed PWA through the standard Web
Push protocol: the browser hands us a *subscription* (an endpoint URL on Apple's
push service plus two keys), and to push we must encrypt an ``aes128gcm`` payload
to the subscription's key and authenticate the request with a signed VAPID JWT.

We do that with only ``cryptography`` (already a dependency) — no ``pywebpush`` —
so the harness keeps its light footprint. Everything degrades safely: if
``cryptography`` is missing the feature just reports disabled, and a failed push
never propagates into the request that triggered it.

Keys and subscriptions live under ``memory_dir`` (gitignored), so nothing secret
is committed.
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from . import config as cfgmod


def _ssl_context() -> Optional[ssl.SSLContext]:
    """A TLS context that can actually verify the push service's certificate.

    On Windows, Python's default context often can't find the CA chain for
    web.push.apple.com ('unable to get local issuer certificate'), so prefer the
    certifi bundle when it's installed. Falls back to the platform default."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return None


_SSL = _ssl_context()

try:  # cryptography ships the EC + AEAD + HKDF primitives we need
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    _HAVE_CRYPTO = True
except Exception:  # pragma: no cover - environment without cryptography
    _HAVE_CRYPTO = False

_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# base64url helpers (Web Push uses unpadded base64url everywhere)
# --------------------------------------------------------------------------- #
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(s: str) -> bytes:
    s = s.strip()
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# --------------------------------------------------------------------------- #
# feature + storage
# --------------------------------------------------------------------------- #
def available() -> bool:
    return _HAVE_CRYPTO


def _keys_path(cfg) -> Path:
    return Path(cfg.memory_dir) / "push_keys.json"


def _subs_path(cfg) -> Path:
    return Path(cfg.memory_dir) / "push_subscriptions.json"


def _contact(cfg) -> str:
    c = (getattr(cfg, "push_contact", "") or "").strip()
    if not c:
        return "mailto:anvil@localhost"
    return c if ":" in c else "mailto:" + c


def ensure_keys(cfg) -> Optional[Dict[str, str]]:
    """Load the VAPID keypair, generating + persisting one on first use."""
    if not _HAVE_CRYPTO:
        return None
    path = _keys_path(cfg)
    with _LOCK:
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if data.get("private") and data.get("public"):
                    return data
            except (ValueError, OSError):
                pass
        pk = ec.generate_private_key(ec.SECP256R1())
        priv_raw = pk.private_numbers().private_value.to_bytes(32, "big")
        pub_raw = pk.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint)
        data = {"private": _b64(priv_raw), "public": _b64(pub_raw)}
        # Atomic + read-only-tolerant like every other ANVIL state file: a plain
        # write_text interrupted mid-write (crash/power-cut) or blocked by a
        # transient AV/sync lock could truncate the keypair — and a corrupt
        # push_keys.json makes ensure_keys regenerate the VAPID pair, silently
        # invalidating every device's existing subscription.
        cfgmod.atomic_write(path, json.dumps(data))
        return data


def public_key(cfg) -> str:
    data = ensure_keys(cfg)
    return data["public"] if data else ""


def _load_subs(cfg) -> List[dict]:
    path = _subs_path(cfg)
    if not path.exists():
        return []
    try:
        subs = json.loads(path.read_text("utf-8"))
        return subs if isinstance(subs, list) else []
    except (ValueError, OSError):
        return []


def _save_subs(cfg, subs: List[dict]) -> None:
    # Atomic + read-only-tolerant like every other ANVIL state file: a plain
    # write_text interrupted mid-write leaves a truncated push_subscriptions.json
    # (which _load_subs then reads as empty, silently dropping every device), and
    # a transient AV/sync lock would raise OSError out through send()'s prune —
    # breaking its "never raises" contract. atomic_write heals both.
    path = _subs_path(cfg)
    cfgmod.atomic_write(path, json.dumps(subs))


def add_subscription(cfg, sub: dict, profile: str = "", sticky: bool = True) -> bool:
    """Register a device for push, TAGGED with the profile logged in on it, so a
    personal notification only reaches that person's own devices. ``sticky`` (the
    'keep me logged in on this device' option) keeps the binding after logout."""
    ep = (sub or {}).get("endpoint")
    keys = (sub or {}).get("keys") or {}
    if not ep or not keys.get("p256dh") or not keys.get("auth"):
        return False
    with _LOCK:
        subs = _load_subs(cfg)
        subs = [s for s in subs if s.get("endpoint") != ep]   # dedupe by endpoint
        subs.append({"endpoint": ep,
                     "keys": {"p256dh": keys["p256dh"], "auth": keys["auth"]},
                     "profile": profile or "", "sticky": bool(sticky)})
        _save_subs(cfg, subs)
    return True


def remove_subscription(cfg, endpoint: str) -> None:
    with _LOCK:
        subs = [s for s in _load_subs(cfg) if s.get("endpoint") != endpoint]
        _save_subs(cfg, subs)


def unbind_device(cfg, endpoint: str) -> None:
    """On logout, a NON-sticky device stops being that profile's push target
    (its binding is cleared -> broadcast-only). A sticky device keeps its owner."""
    if not endpoint:
        return
    with _LOCK:
        subs = _load_subs(cfg)
        changed = False
        for s in subs:
            if s.get("endpoint") == endpoint and not s.get("sticky"):
                if s.get("profile"):
                    s["profile"] = ""
                    changed = True
        if changed:
            _save_subs(cfg, subs)


def subscription_count(cfg) -> int:
    return len(_load_subs(cfg))


def _for_targets(subs: List[dict], to) -> List[dict]:
    """Filter subscriptions to a push target. ``to`` None = broadcast (all
    devices). Otherwise an iterable of profile names — only those profiles'
    devices receive it (untagged single-user devices match "")."""
    if to is None:
        return subs
    want = set(to)
    return [s for s in subs if s.get("profile", "") in want]


# --------------------------------------------------------------------------- #
# crypto: VAPID JWT + aes128gcm payload encryption
# --------------------------------------------------------------------------- #
def _vapid_jwt(aud: str, contact: str, priv_raw: bytes) -> str:
    pk = ec.derive_private_key(int.from_bytes(priv_raw, "big"), ec.SECP256R1())
    header = _b64(json.dumps({"typ": "JWT", "alg": "ES256"},
                             separators=(",", ":")).encode())
    claims = _b64(json.dumps({"aud": aud, "exp": int(time.time()) + 12 * 3600,
                              "sub": contact},
                             separators=(",", ":")).encode())
    signing_input = f"{header}.{claims}".encode()
    der = pk.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)                 # JOSE wants raw r||s, not DER
    sig = _b64(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    return f"{header}.{claims}.{sig}"


def _encrypt(payload: bytes, p256dh_b64: str, auth_b64: str) -> bytes:
    ua_pub_bytes = _unb64(p256dh_b64)                # client public key (65 bytes)
    auth = _unb64(auth_b64)                          # client auth secret (16 bytes)
    ua_pub = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), ua_pub_bytes)
    as_priv = ec.generate_private_key(ec.SECP256R1())
    as_pub_bytes = as_priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    shared = as_priv.exchange(ec.ECDH(), ua_pub)

    # RFC 8291 §3.4: fold the ECDH secret + auth into the input keying material.
    key_info = b"WebPush: info\x00" + ua_pub_bytes + as_pub_bytes
    ikm = HKDF(algorithm=hashes.SHA256(), length=32, salt=auth,
               info=key_info).derive(shared)
    salt = os.urandom(16)
    cek = HKDF(algorithm=hashes.SHA256(), length=16, salt=salt,
               info=b"Content-Encoding: aes128gcm\x00").derive(ikm)
    nonce = HKDF(algorithm=hashes.SHA256(), length=12, salt=salt,
                 info=b"Content-Encoding: nonce\x00").derive(ikm)

    # Single record: plaintext + 0x02 delimiter (last record, no padding).
    ct = AESGCM(cek).encrypt(nonce, payload + b"\x02", None)
    # aes128gcm header: salt(16) | rs(4) | idlen(1) | keyid(=server public key).
    header = salt + struct.pack(">I", 4096) + bytes([len(as_pub_bytes)]) + as_pub_bytes
    return header + ct


def _send_one(sub: dict, payload: bytes, priv_raw: bytes, pub_b64: str,
              contact: str, timeout: int) -> int:
    endpoint = sub["endpoint"]
    body = _encrypt(payload, sub["keys"]["p256dh"], sub["keys"]["auth"])
    u = urlparse(endpoint)
    jwt = _vapid_jwt(f"{u.scheme}://{u.netloc}", contact, priv_raw)
    headers = {
        "Authorization": f"vapid t={jwt}, k={pub_b64}",
        "Content-Encoding": "aes128gcm",
        "Content-Type": "application/octet-stream",
        "TTL": "86400",
    }
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code                       # 404/410 => subscription is dead
    except Exception:
        return 0                              # transient; keep the subscription


# --------------------------------------------------------------------------- #
# public send API
# --------------------------------------------------------------------------- #
def send(cfg, title: str, body: str = "", url: str = "/",
         tag: str = "anvil", to=None) -> dict:
    """Push to the target subscriptions; prune the ones the push service reports
    as gone (404/410). ``to`` None = broadcast to all devices; else an iterable
    of profile names (personal — only those people's devices). Never raises."""
    if not _HAVE_CRYPTO:
        return {"ok": False, "reason": "cryptography not available", "sent": 0}
    keys = ensure_keys(cfg)
    if not keys:
        return {"ok": False, "reason": "no vapid keys", "sent": 0}
    subs = _for_targets(_load_subs(cfg), to)
    if not subs:
        return {"ok": True, "sent": 0, "subs": 0}
    priv_raw = _unb64(keys["private"])
    pub_b64 = keys["public"]
    contact = _contact(cfg)
    timeout = int(getattr(cfg, "request_timeout", 10) or 10)
    payload = json.dumps({"title": title, "body": body, "url": url,
                          "tag": tag}).encode("utf-8")
    sent, dead = 0, []
    for sub in subs:
        try:
            code = _send_one(sub, payload, priv_raw, pub_b64, contact, timeout)
        except Exception:
            code = 0
        if code in (404, 410):
            dead.append(sub.get("endpoint"))
        elif 200 <= code < 300:
            sent += 1
    if dead:
        with _LOCK:
            keep = [s for s in _load_subs(cfg) if s.get("endpoint") not in dead]
            _save_subs(cfg, keep)
    return {"ok": True, "sent": sent, "pruned": len(dead), "subs": len(subs)}


def notify(cfg, title: str, body: str = "", url: str = "/",
           tag: str = "anvil", to=None) -> None:
    """Fire-and-forget: push on a background thread so we never block (or break)
    the chat request / mind loop that triggered the notification. ``to`` None =
    broadcast; else an iterable of profile names (personal)."""
    if not _HAVE_CRYPTO or not _load_subs(cfg):
        return
    threading.Thread(
        target=lambda: send(cfg, title, body, url, tag, to=to), daemon=True).start()
