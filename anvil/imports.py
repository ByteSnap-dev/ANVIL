"""Import ChatGPT conversations.json exports into ANVIL's own turn format.

This module is intentionally defensive: real-world ChatGPT exports are large,
mutated across app versions, and frequently contain malformed or partial
entries (broken mapping links, non-text content parts, system/tool nodes,
etc). ``parse_chatgpt`` must never raise on a single bad conversation or a
single bad message -- it just skips the offending entry and keeps going.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _flatten_content(content: Any) -> Optional[str]:
    """Flatten a ChatGPT `content` blob down to plain text.

    ChatGPT messages carry a `content` dict with a `content_type` and either
    a `parts` list (text/multimodal_text) or other shapes we don't support.
    Non-text parts (images, tool blobs, dicts, etc.) are dropped silently;
    only string parts are kept and joined.
    """
    if content is None:
        return None
    if isinstance(content, str):
        text = content
    else:
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            return None
        texts = [p for p in parts if isinstance(p, str)]
        if not texts:
            return None
        text = "\n".join(texts)
    text = text.strip()
    return text or None


def _parse_message(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn one mapping node's `message` into a {role, content, ts} turn.

    Returns None (rather than raising) for system/tool roles, empty content,
    or any malformed shape.
    """
    try:
        msg = node.get("message")
        if not msg:
            return None
        author = msg.get("author") or {}
        role = author.get("role")
        if role not in ("user", "assistant"):
            return None
        text = _flatten_content(msg.get("content"))
        if not text:
            return None
        return {
            "role": role,
            "content": text,
            "ts": msg.get("create_time"),
        }
    except Exception:
        return None


def _walk_turns(mapping: Dict[str, Any], current_node: Optional[str]) -> List[Dict[str, Any]]:
    """Walk the mapping graph from current_node back to the root, then
    reverse it to get chronological turn order. Skips broken links/nodes.
    """
    chain: List[Dict[str, Any]] = []
    node_id = current_node
    seen = set()
    while node_id is not None and node_id not in seen:
        seen.add(node_id)
        try:
            node = mapping.get(node_id)
        except Exception:
            break
        if not isinstance(node, dict):
            break
        turn = _parse_message(node)
        if turn is not None:
            chain.append(turn)
        node_id = node.get("parent")
    chain.reverse()
    return chain


def _parse_conversation(conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        mapping = conv.get("mapping")
        if not isinstance(mapping, dict):
            return None
        current_node = conv.get("current_node")
        turns = _walk_turns(mapping, current_node)
        return {
            "id": conv.get("id") or conv.get("conversation_id"),
            "title": conv.get("title") or "",
            "created": conv.get("create_time"),
            "turns": turns,
        }
    except Exception:
        return None


def parse_chatgpt(path) -> List[Dict[str, Any]]:
    """Parse a ChatGPT `conversations.json` export into ANVIL's turn format.

    Returns a list of {id, title, created, turns:[{role, content, ts}]}
    dicts. Malformed conversations or messages are skipped rather than
    raising, so a single broken entry in a huge export can't blow up import.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        convs = data.get("conversations") or data.get("data") or []
    else:
        convs = data
    if not isinstance(convs, list):
        convs = []

    out: List[Dict[str, Any]] = []
    for conv in convs:
        if not isinstance(conv, dict):
            continue
        try:
            parsed = _parse_conversation(conv)
        except Exception:
            parsed = None
        if parsed is not None:
            out.append(parsed)
    return out


def _iso_to_epoch(value: Any) -> Optional[float]:
    """Best-effort ISO8601 -> epoch-seconds float. Never raises."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        # Python's fromisoformat doesn't accept a trailing 'Z' pre-3.11.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        from datetime import datetime
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _parse_claude_conversation(conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        msgs = conv.get("chat_messages")
        if not isinstance(msgs, list):
            msgs = []
        turns: List[Dict[str, Any]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            try:
                sender = m.get("sender")
                if sender == "human":
                    role = "user"
                elif sender == "assistant":
                    role = "assistant"
                else:
                    continue
                text = m.get("text")
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue
                turns.append({
                    "role": role,
                    "content": text,
                    "ts": _iso_to_epoch(m.get("created_at")),
                })
            except Exception:
                continue
        return {
            "id": conv.get("uuid"),
            "title": conv.get("name") or "",
            "created": _iso_to_epoch(conv.get("created_at")),
            "turns": turns,
        }
    except Exception:
        return None


def parse_claude(path) -> List[Dict[str, Any]]:
    """Parse a Claude.ai `conversations.json` export into ANVIL's turn format.

    Expects a JSON array of conversations, each with uuid/name/created_at and
    a flat `chat_messages` list (sender: human/assistant). Malformed entries
    are skipped rather than raising.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        convs = data.get("conversations") or data.get("data") or []
    else:
        convs = data
    if not isinstance(convs, list):
        convs = []

    out: List[Dict[str, Any]] = []
    for conv in convs:
        if not isinstance(conv, dict):
            continue
        try:
            parsed = _parse_claude_conversation(conv)
        except Exception:
            parsed = None
        if parsed is not None:
            out.append(parsed)
    return out


def _ci_get(d: Dict[str, Any], *keys: str) -> Any:
    """Case-insensitive dict lookup across candidate keys."""
    if not isinstance(d, dict):
        return None
    lower = {k.lower(): v for k, v in d.items() if isinstance(k, str)}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def _parse_gemini_conversation(conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        msgs = _ci_get(conv, "messages", "turns")
        if not isinstance(msgs, list):
            msgs = []
        turns: List[Dict[str, Any]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            try:
                author = _ci_get(m, "author", "role")
                if isinstance(author, str):
                    a = author.strip().lower()
                else:
                    a = ""
                if a in ("user", "human"):
                    role = "user"
                elif a in ("assistant", "model", "gemini", "bot"):
                    role = "assistant"
                else:
                    continue
                text = _ci_get(m, "text", "content")
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue
                turns.append({
                    "role": role,
                    "content": text,
                    "ts": _ci_get(m, "ts", "timestamp", "created", "create_time"),
                })
            except Exception:
                continue
        return {
            "id": _ci_get(conv, "id", "uuid", "conversation_id"),
            "title": _ci_get(conv, "title", "name") or "",
            "created": _ci_get(conv, "created", "created_at", "create_time"),
            "turns": turns,
        }
    except Exception:
        return None


def parse_gemini(path) -> List[Dict[str, Any]]:
    """Parse a Gemini conversations export into ANVIL's turn format.

    Tolerates either a bare JSON array of conversations or a dict with a
    'conversations' key. Messages may be under 'messages' or 'turns', with
    case-insensitive author/role and text/content fields. Malformed entries
    are skipped rather than raising.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        convs = data.get("conversations")
        if convs is None:
            convs = _ci_get(data, "conversations") or []
    else:
        convs = data
    if not isinstance(convs, list):
        convs = []

    out: List[Dict[str, Any]] = []
    for conv in convs:
        if not isinstance(conv, dict):
            continue
        try:
            parsed = _parse_gemini_conversation(conv)
        except Exception:
            parsed = None
        if parsed is not None:
            out.append(parsed)
    return out
