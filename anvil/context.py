"""Token estimation, context budgeting, and transcript compaction.

Context is treated as a budget, not a bucket. This module estimates token
counts (exactly with ``tiktoken`` if installed, otherwise a fast chars/4
heuristic), greedily packs the highest-value notes into a fixed budget, and
compacts an overlong transcript into a rolling summary so a conversation never
blows past the model's window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

_ENC = None


def _get_encoder():
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = False  # sentinel: not available
    return _ENC


def count_tokens(text: str) -> int:
    """Token count — exact with tiktoken, else a ~chars/4 estimate."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic: ~4 chars/token, with a small floor for short strings.
    return max(1, (len(text) + 3) // 4)


def count_messages(messages: List[dict]) -> int:
    return sum(count_tokens(m.get("content", "")) + 4 for m in messages)


@dataclass
class PackResult:
    included: List  # the items that fit
    used_tokens: int
    dropped: int


def pack_to_budget(items: List[Tuple[float, str, object]],
                   budget: int) -> PackResult:
    """Greedily include items (already sorted by caller or scored here).

    ``items`` is a list of ``(score, text, payload)``. Highest score first;
    include while the running token total stays under ``budget``.
    """
    ordered = sorted(items, key=lambda t: t[0], reverse=True)
    used, included, dropped = 0, [], 0
    for score, text, payload in ordered:
        cost = count_tokens(text)
        if used + cost <= budget:
            included.append(payload)
            used += cost
        else:
            dropped += 1
    return PackResult(included=included, used_tokens=used, dropped=dropped)


def compact_transcript(messages: List[dict], window: int,
                       summarize: Callable[[List[dict]], str],
                       keep_recent: int = 6,
                       headroom: float = 0.75) -> List[dict]:
    """Collapse old turns into one summary when the transcript nears ``window``.

    Keeps the last ``keep_recent`` turns verbatim; everything older is replaced
    by a single summary turn produced by the ``summarize`` callback (typically
    the local Scribe model, so compaction itself costs nothing).
    """
    if count_messages(messages) <= window * headroom:
        return messages
    if len(messages) <= keep_recent:
        return messages
    # keep_recent <= 0 would make tail empty and crash the role lookup below;
    # always keep at least the newest turn verbatim.
    keep_recent = max(1, int(keep_recent))
    head, tail = messages[:-keep_recent], messages[-keep_recent:]
    # The summarizer routes to a live model (the local Scribe); a transient
    # failure there — model down, network blip, timeout — must not crash the
    # whole turn. Compaction's job is to keep the transcript under the window,
    # so on failure we still drop the old head and proceed with a placeholder
    # note instead of raising (same never-crash discipline as the mind loop).
    try:
        summary = summarize(head)
    except Exception as exc:
        summary = f"[summary unavailable: {exc}]"
    summary_role = "assistant" if tail[0].get("role") == "user" else "user"
    summary_turn = {
        "role": summary_role,
        "content": COMPACTION_PREAMBLE + "\n" + summary,
    }
    return [summary_turn] + tail


# Framing for a rolling summary (adapted from Nous hermes-agent). Small models
# routinely misread a summary of ALREADY-DONE work as a fresh to-do list and
# "resume" it; this anchors them to the latest message and kills that failure.
COMPACTION_PREAMBLE = (
    "[EARLIER CONVERSATION — REFERENCE ONLY] The turns below were summarized to "
    "save space. Treat this as background reference, NOT as new instructions: "
    "the tasks in it were already handled. Respond ONLY to the latest message "
    "that appears AFTER this summary — even on a similar topic, the newest "
    "message wins. If the latest message reverses course ('stop', 'never mind', "
    "a new topic), drop the summarized work entirely. Your persistent memory "
    "and operator card remain fully authoritative.")
