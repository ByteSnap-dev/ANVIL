"""Procedural flame app icon — generated in pure stdlib, no image libraries.

A PWA needs real raster PNG icons (iOS home-screen tiles won't take an inline
SVG reliably). Rather than ship binary blobs in the repo, we draw ANVIL's flame
into an RGBA raster and PNG-encode it with ``zlib`` + CRC32 — both standard
library. The result is deterministic, so the same bytes come out every run and
can be cached in memory by the server.

The mark: a rounded ember/teardrop (rounded bottom bulb, pointed top) filled
with a warm bottom-hot gradient, on a dark charcoal full-bleed square so it
doubles as a ``maskable`` icon (the platform rounds the corners itself).
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import Dict, Tuple

# Brand palette (RGB).
_BG = (0x14, 0x17, 0x1C)          # charcoal, matches the command-deck UI
_TIP = (0xFF, 0x53, 0x1F)         # deep orange at the tip
_BASE = (0xFF, 0xD1, 0x4A)        # hot yellow at the bulb
_CORE = (0xFF, 0xF3, 0xD6)        # near-white inner core

_CACHE: Dict[Tuple[int, bool], bytes] = {}


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _flame_geom():
    """Teardrop = bulb circle (rounded bottom) ∪ tangent triangle to the tip."""
    r = 0.33                       # bulb radius in local units (height == 1.0)
    cy = 1.0 - r                   # bulb centre; bottom of circle sits at v=1.0
    d = cy                         # tip is at (0,0), straight above the centre
    phi = math.acos(max(-1.0, min(1.0, r / d)))
    # Tangent points where the cone meets the bulb.
    txr, tyr = math.sin(phi) * r, cy - math.cos(phi) * r
    return r, cy, (txr, tyr)


def _inside_tri(px, py, ax, ay, bx, by, cx, cy):
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def _sample(lx, lv, r, cy, tan):
    """Return (inside, core_t) for a local-space point, or (False, 0)."""
    txr, tyr = tan
    in_circle = (lx * lx + (lv - cy) ** 2) <= r * r
    in_tri = _inside_tri(lx, lv, 0.0, 0.0, txr, tyr, -txr, tyr)
    if not (in_circle or in_tri):
        return False, 0.0
    # Inner brightened core: same teardrop shrunk about a hot point low in the
    # flame, so the mark reads as glowing rather than flat.
    hx, hy = 0.0, 0.66
    s = 0.52
    cx2, cy2 = hx + (lx - hx) / s, hy + (lv - hy) / s
    in_core = ((cx2 * cx2 + (cy2 - cy) ** 2) <= r * r) or \
        _inside_tri(cx2, cy2, 0.0, 0.0, txr, tyr, -txr, tyr)
    return True, (1.0 if in_core else 0.0)


def render_png(size: int = 192, maskable: bool = False) -> bytes:
    key = (int(size), bool(maskable))
    if key in _CACHE:
        return _CACHE[key]

    n = int(size)
    ss = 2                                   # 2x supersampling for smooth edges
    pad = 0.20 if maskable else 0.16         # maskable keeps a safer margin
    span = 1.0 - 2 * pad                     # flame height as a fraction of n
    r, cy, tan = _flame_geom()

    # Precompute per-supersample-column/row local coords.
    px = [None] * (n * ss)
    for i in range(n * ss):
        px[i] = ((i + 0.5) / ss - n / 2) / (span * n)          # local x, centred
    pv = [None] * (n * ss)
    for j in range(n * ss):
        pv[j] = ((j + 0.5) / ss - pad * n) / (span * n)        # local v, 0 at top

    raw = bytearray()
    for j in range(n):
        raw.append(0)                         # PNG filter type 0 for this row
        for i in range(n):
            rr = gg = bb = 0
            for sj in range(ss):
                lv = pv[j * ss + sj]
                for si in range(ss):
                    lx = px[i * ss + si]
                    inside, core = _sample(lx, lv, r, cy, tan)
                    if inside:
                        t = max(0.0, min(1.0, lv))          # 0 tip .. 1 bottom
                        col = _lerp(_TIP, _BASE, t)
                        if core:
                            col = _lerp(col, _CORE, 0.55)
                    else:
                        col = _BG                           # full-bleed charcoal
                    rr += col[0]; gg += col[1]; bb += col[2]
            k = ss * ss
            raw += bytes((rr // k, gg // k, bb // k, 255))
    png = _encode_png(n, n, bytes(raw))
    _CACHE[key] = png
    return png


def _encode_png(w: int, h: int, raw_rgba: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + \
            struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)   # 8-bit RGBA
    idat = zlib.compress(raw_rgba, 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
