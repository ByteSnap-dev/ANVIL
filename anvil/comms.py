"""Discord liaison — reach Joe on the go.

Outbound notifications use a Discord *webhook* and need nothing but the stdlib
``urllib`` (zero dependencies). Two-way control (Joe issuing ``!ask``,
``!status``, ``!note``, ``!approve`` from his phone) is optional and only
activates if ``discord.py`` is installed and a bot token is configured.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Outbound: webhook push (zero-dep)
# --------------------------------------------------------------------------- #
def notify(webhook_url: Optional[str], content: str,
           username: str = "Anvil") -> bool:
    """POST a message to a Discord webhook. Returns True on success."""
    if not webhook_url:
        print(f"[anvil:discord disabled] {content}")
        return False
    # Discord hard-caps content at 2000 chars.
    if len(content) > 1900:
        content = content[:1897] + "..."
    payload = json.dumps({"content": content, "username": username}).encode()
    req = urllib.request.Request(
        webhook_url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.URLError as exc:
        print(f"[anvil:discord error] {exc}")
        return False


# --------------------------------------------------------------------------- #
# Inbound: optional two-way bot
# --------------------------------------------------------------------------- #
def run_bot(token: str, handlers: dict, prefix: str = "!") -> None:
    """Start a discord.py bot mapping ``!command`` -> handler(args)->str.

    ``handlers`` maps command name to a callable taking the argument string and
    returning a reply string. Raises a clear error if discord.py is absent.
    """
    try:
        import discord  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Two-way Discord control needs `pip install discord.py`. "
            "Outbound webhook notifications work without it."
        ) from exc

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_message(message):  # noqa: ANN001
        if message.author == client.user or not message.content.startswith(prefix):
            return
        body = message.content[len(prefix):].strip()
        cmd, _, args = body.partition(" ")
        handler = handlers.get(cmd)
        if not handler:
            await message.channel.send(
                f"Unknown command `{cmd}`. Try: {', '.join(handlers)}")
            return
        async with message.channel.typing():
            try:
                reply = handler(args.strip())
            except Exception as exc:  # surface errors to the phone
                reply = f"⚠️ {type(exc).__name__}: {exc}"
        for chunk in _chunk(reply or "(no output)", 1900):
            await message.channel.send(chunk)

    client.run(token)


def _chunk(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i:i + size]
