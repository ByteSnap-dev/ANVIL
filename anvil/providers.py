"""Provider adapters — Ollama local and Ollama Cloud.

Both speak over HTTP using only the standard library (``urllib``), so the core
harness has zero install requirements. Each adapter exposes the same
``chat(...)`` signature and returns a normalised ``Completion`` so the router
can treat every rung identically.

Ollama local & cloud share the OpenAI-compatible ``/v1/chat/completions``
surface; cloud differs only by base URL + a Bearer key.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Some models (notably several on the Ollama Cloud endpoint) do NOT populate the
# separate ``thinking`` field — they inline their reasoning in ``content`` wrapped
# in <think>…</think> tags. Left unstripped, that reasoning leaks straight into the
# operator-facing answer (observed live: a weather reply that began mid-thought and
# still carried a bare </think>). Strip any COMPLETE <think>…</think> block, and a
# single leading orphan </think> (model that opened <think> before content began, so
# only the closing tag rides in content). Deliberately conservative: no opening tag
# and no closing tag → leave the text alone.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_LEAD_RE = re.compile(r"^\s*.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_think(text: str) -> str:
    if not text or "</think>" not in text.lower():
        return text
    out = _THINK_BLOCK_RE.sub("", text)
    if "<think" not in out.lower() and "</think>" in out.lower():
        # A dangling close with no matching open: the reasoning ran from the start
        # of content up to it. Drop everything through the first </think>.
        out = _THINK_LEAD_RE.sub("", out, count=1)
    return out.strip()

# Jittered exponential backoff (ported from Nous hermes-agent/retry_utils). The
# jitter DECORRELATES concurrent retries — without it, N hive drones that all
# hit a cloud rate-limit at once would retry at the same instant and stampede.
_JITTER_LOCK = threading.Lock()
_JITTER_TICK = 0


def jittered_backoff(attempt: int, *, base_delay: float = 2.0,
                     max_delay: float = 60.0, jitter_ratio: float = 0.5) -> float:
    """Seconds to wait before retry ``attempt`` (1-based):
    min(base*2^(attempt-1), max) + uniform(0, jitter_ratio*delay)."""
    global _JITTER_TICK
    with _JITTER_LOCK:
        _JITTER_TICK += 1
        tick = _JITTER_TICK
    exponent = max(0, attempt - 1)
    delay = max_delay if (exponent >= 63 or base_delay <= 0) \
        else min(base_delay * (2 ** exponent), max_delay)
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    return delay + random.Random(seed).uniform(0, jitter_ratio * delay)


def is_transient(exc: Exception) -> bool:
    """Worth a same-rung retry? Rate limits, 5xx, and transport hiccups are;
    auth/4xx/overflow are not (escalate or fail those)."""
    if isinstance(exc, ContextOverflow):
        return False
    msg = str(exc).lower()
    if any(s in msg for s in ("429", "rate limit", "overloaded", "timeout",
                              "timed out", "temporarily", "connection reset",
                              "connection aborted", "econnreset")):
        return True
    for code in (" 500", " 502", " 503", " 504", "http 5"):
        if code in msg:
            return True
    return False


def _estimate_tokens(msgs) -> int:
    """Conservative prompt-size estimate (chars/3) for when a provider's usage
    frame never arrives. The spend cap must fail CLOSED — better to over-count
    a broken stream than to record a paid call as free."""
    try:
        total = 0
        for m in msgs or []:
            c = m.get("content")
            if isinstance(c, str):
                total += len(c)
            elif isinstance(c, list):        # vision parts
                total += sum(len(str(p.get("text", ""))) for p in c
                             if isinstance(p, dict))
        return max(1, total // 3)
    except Exception:
        return 1


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    model: str = ""
    provider: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    # Native tool calls the model requested, as [{"name": str, "arguments": dict}].
    tool_calls: List[dict] = field(default_factory=list)
    # Server-side web searches Anthropic ran inside this call ($10/1k — the
    # ledger prices them; queries land in raw["search_queries"] for the trace).
    web_searches: int = 0


class ProviderError(RuntimeError):
    pass


class ContextOverflow(ProviderError):
    """The prompt exceeded the model's context window. Distinct from a generic
    ProviderError so the router can COMPACT the messages and retry the same rung
    instead of escalating or failing — the operator never sees the error."""


# Different backends phrase 'you sent too many tokens' differently; match on the
# stable substrings (Ollama / llama.cpp error formats all covered).
_OVERFLOW_MARKERS = (
    "context length", "context window", "maximum context", "context_length_exceeded",
    "too many tokens", "exceeds the maximum", "reduce the length", "input is too long",
    "prompt is too long", "requested tokens", "kv cache", "n_ctx",
)


def _looks_like_overflow(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _OVERFLOW_MARKERS)


class GenerationCancelled(Exception):
    """The operator hit Stop mid-generation. Raised from the token sink and
    allowed to propagate all the way out (unlike other sink errors) so the
    in-flight model call actually aborts and releases the model."""


def _emit(on_token, delta: str) -> None:
    """Forward a streamed token to an optional sink, swallowing sink errors so a
    flaky UI callback can never abort an in-flight generation — EXCEPT a
    deliberate cancel, which must propagate to stop the stream."""
    if on_token and delta:
        try:
            on_token(delta)
        except GenerationCancelled:
            raise
        except Exception:
            pass


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # surface the body for debugging
        body = exc.read().decode("utf-8", "replace")
        if _looks_like_overflow(body):
            raise ContextOverflow(body) from exc
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach {url}: {exc.reason}") from exc


def _get_json(url: str, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach {url}: {exc.reason}") from exc


def _get_text(url: str, headers: dict, timeout: int) -> str:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach {url}: {exc.reason}") from exc


def _post_json_stream(url: str, payload: dict, headers: dict, timeout: int,
                      on_line, cancel=None) -> None:
    """POST and consume the response one line at a time.

    The whole point: ``timeout`` bounds each individual socket read, and every
    line resets it. So a model that keeps emitting tokens never idle-times-out,
    no matter how long the *total* generation takes — the fix for the
    ``TimeoutError: timed out`` we used to hit when a slow local model needed
    more than ``request_timeout`` seconds to finish a big answer. The socket only
    times out if the model stalls completely (no output for ``timeout`` seconds),
    which is a genuine hang worth surfacing.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:  # iterates by line as bytes arrive off the socket
                # Check cancel on EVERY line — including the model's silent
                # 'thinking' lines — so Stop interrupts during the long reasoning
                # phase, not just once visible tokens start streaming.
                if cancel and cancel():
                    raise GenerationCancelled()
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    on_line(line)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if _looks_like_overflow(body):
            raise ContextOverflow(body) from exc
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach {url}: {exc.reason}") from exc


# --------------------------------------------------------------------------- #
# Ollama (local + cloud share this OpenAI-compatible adapter)
# --------------------------------------------------------------------------- #
class OllamaProvider:
    """Talks to any Ollama endpoint via its OpenAI-compatible /v1 surface."""

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 timeout: int = 120, name: str = "ollama",
                 num_ctx: int = 0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.name = name
        # Context window override (0 = server default). Measured on the 24GB
        # 7900 XTX with qwen3.6 Q4: 32k default = 23GB (100% GPU), 65536 =
        # 24GB (100% GPU, the single-card ceiling), 131072 = 26GB and SPILLS
        # 7% to CPU — generation craters. Don't raise past 65536 on one card.
        self.num_ctx = int(num_ctx or 0)

    def chat(self, model: str, messages: List[dict],
             schema: Optional[dict] = None, temperature: float = 0.2,
             tools: Optional[List[dict]] = None,
             system: Optional[str] = None, on_token=None, cancel=None,
             think: Optional[bool] = None, on_think=None, **kw) -> Completion:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        max_tokens = int(kw.get("max_tokens") or 0)

        # A conversation that already CARRIES tool calls / tool results must
        # stay on the native endpoint: it stores arguments as a JSON object,
        # which /api/chat expects but the /v1 OpenAI surface rejects (it wants
        # arguments as a *string*) with a 400. The "answer now" wrap-up calls
        # pass no tools, so without this they'd route to /v1 and crash mid-turn.
        carries_tools = any(m.get("tool_calls") or m.get("role") == "tool"
                            for m in msgs)
        if tools or carries_tools or any(m.get("images") for m in msgs):
            # Native /api/chat handles BOTH tool calling and vision: tools are
            # injected into the trained chat template, and per-message base64
            # "images" reach the model's vision encoder. The /v1 OpenAI surface
            # below would silently DROP images, so any sighted request must
            # come through here even when no tools are offered.
            return self._native_chat(model, msgs, tools, temperature, headers,
                                     on_token, cancel=cancel, think=think,
                                     on_think=on_think, max_tokens=max_tokens,
                                     schema=schema)

        # Everything streams: even schema/structured calls (where ``on_token`` is
        # None) go over the streaming transport so a slow generation can't trip
        # the socket's idle timeout. We just accumulate the deltas and hand back
        # the same assembled Completion the non-streaming path used to return.
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if schema:
            # Ollama enforces the JSON schema during decoding -> no retry loops.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "anvil", "schema": schema, "strict": True},
            }
        parts: List[str] = []
        usage: Dict[str, Any] = {}

        def handle(line: str) -> None:
            if not line.startswith("data:"):
                return
            chunk = line[len("data:"):].strip()
            if not chunk or chunk == "[DONE]":
                return
            try:
                obj = json.loads(chunk)
            except (json.JSONDecodeError, ValueError):
                return
            choice = (obj.get("choices") or [{}])[0]
            delta = (choice.get("delta") or {}).get("content") or ""
            if delta:
                parts.append(delta)
                _emit(on_token, delta)
            if obj.get("usage"):
                usage.update(obj["usage"])

        _post_json_stream(url, payload, headers, self.timeout, handle, cancel=cancel)
        raw_text = "".join(parts)
        # Don't touch schema-constrained output: it's pure JSON with no <think>
        # tags, and stripping could corrupt a valid document. Plain prose replies
        # may carry inline reasoning from models that ignore the thinking channel.
        text = raw_text if schema else _strip_think(raw_text)
        inp, out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        if not usage:
            # The usage frame never arrived (stream cut early, proxy dropped it).
            # A spend guard must fail CLOSED: estimate conservatively rather than
            # record a paid call as $0 — the failure is correlated with exactly
            # the moments the cap matters most.
            inp, out = _estimate_tokens(msgs), max(1, len(raw_text) // 3)
        cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                  if isinstance(usage.get("prompt_tokens_details"), dict) else 0)
        return Completion(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            cached_input_tokens=int(cached or 0),
            model=model,
            provider=self.name,
            raw={"usage": usage, "usage_estimated": not usage},
        )

    def _native_chat(self, model, msgs, tools, temperature, headers,
                     on_token=None, cancel=None,
                     think: Optional[bool] = None, on_think=None,
                     max_tokens: int = 0, schema: Optional[dict] = None) -> Completion:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": msgs,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if schema and not tools:
            # Review 2.9: this path silently DROPPED the caller's schema — a
            # structured verdict over a tool-bearing transcript (the natural
            # shape of "judge what the tools just did") got ZERO enforcement,
            # and the resulting parse failures escalated to paid rungs as
            # "schema-fail". Ollama's native `format` grammar-constrains
            # decoding exactly like /v1's response_format. (Not combinable
            # with live tool OFFERS — the model must stay free to emit a tool
            # call — but tool-HISTORY carrying calls is fine.)
            payload["format"] = schema
        if self.num_ctx:
            # One consistent window for every request keeps a single model
            # instance loaded (Ollama reuses a runner whose ctx >= the ask —
            # mixed sizes would thrash reloads on the one GPU).
            payload["options"]["num_ctx"] = self.num_ctx
        if tools:                     # vision-only calls carry no tools key
            payload["tools"] = tools
        if think is not None:
            # Ollama honours "think": false to SKIP a reasoning model's silent
            # <think> phase — measured ~8x faster (0.8s vs 6.9s) on qwen3.6 with
            # equal answer quality for ordinary questions. A model that doesn't
            # support thinking just ignores the flag.
            payload["think"] = bool(think)
        if max_tokens:
            # num_predict caps thinking + answer TOGETHER, so a turn that may
            # think gets headroom on top of the answer budget — otherwise a
            # long deliberation could eat the whole cap and yield no answer.
            # Either way generation is now bounded: a looping think phase can
            # no longer run forever.
            payload["options"]["num_predict"] = (
                max_tokens + (2048 if think is not False else 0))
        parts: List[str] = []
        calls: List[dict] = []
        meta: Dict[str, int] = {}

        def handle(line: str) -> None:
            try:
                obj = json.loads(line)  # /api/chat streams newline-delimited JSON
            except (json.JSONDecodeError, ValueError):
                return
            msg = obj.get("message", {}) or {}
            # Reasoning models stream their deliberation as separate "thinking"
            # deltas. Surface them on their own channel — never mixed into the
            # answer text — so the UI can show a live "thinking…" trace instead
            # of looking hung during the silent phase.
            thinking = msg.get("thinking") or ""
            if thinking:
                _emit(on_think, thinking)
            content = msg.get("content") or ""
            if content:
                parts.append(content)
                _emit(on_token, content)
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {}) or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"_raw": args}
                calls.append({"name": fn.get("name", ""), "arguments": args or {}})
            if obj.get("done"):
                meta["in"] = obj.get("prompt_eval_count", 0)
                meta["out"] = obj.get("eval_count", 0)

        _post_json_stream(url, payload, headers, self.timeout, handle, cancel=cancel)
        raw_text = "".join(parts)
        inp, out = meta.get("in", 0), meta.get("out", 0)
        if not meta:                     # done frame never arrived — fail CLOSED
            inp, out = _estimate_tokens(msgs), max(1, len(raw_text) // 3)
        return Completion(
            text=_strip_think(raw_text),
            input_tokens=inp,
            output_tokens=out,
            model=model,
            provider=self.name,
            raw={"usage_estimated": not meta},
            tool_calls=calls,
        )

    def embed(self, model: str, text: str) -> List[float]:
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = _post_json(url, {"model": model, "input": text},
                          headers, self.timeout)
        return resp["data"][0]["embedding"]


# --------------------------------------------------------------------------- #
# Anthropic (Claude) — the tiered cloud ladder: Haiku -> Sonnet -> Opus -> Fable
# --------------------------------------------------------------------------- #
class AnthropicProvider:
    """Talks to the Claude Messages API. Adapts the harness's OpenAI-style
    message list (system/user/assistant + OpenAI tool_calls + role:'tool'
    results) into Anthropic's content-block format, streams over SSE, forces
    JSON via a single tool when a schema is asked for, and reports real usage
    so the ledger prices each rung correctly."""

    API = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, timeout: int = 120, name: str = "anthropic",
                 max_tokens_default: int = 2048, native_web_search: bool = True,
                 web_search_max_uses: int = 3):
        self.api_key = api_key
        self.timeout = timeout
        self.name = name
        self.max_tokens_default = int(max_tokens_default or 2048)
        self.native_web_search = bool(native_web_search)
        self.web_search_max_uses = max(1, int(web_search_max_uses or 3))
        # The circuit breaker's half-open preflight probes provider.base_url;
        # without this it parsed '' -> localhost:443 -> refused, so an OPEN
        # anthropic rung could never close again (Lara's issue #89, 44x).
        self.base_url = self.API
        # Models that reject the `temperature` knob outright (Sonnet 5 400s
        # with "deprecated for this model") — learned on first refusal so
        # every later call skips the knob and the round trip.
        self._no_temp: set = set()

    # -- message-shape adapter ------------------------------------------- #
    @staticmethod
    def _to_anthropic(messages: List[dict]):
        """(system_str, [anthropic messages]). Pairs OpenAI tool_calls with the
        role:'tool' results that follow them, synthesizing the tool_use ids
        Anthropic requires (the harness/Ollama shape carries none)."""
        system_parts: List[str] = []
        out: List[dict] = []
        pending_ids: List[str] = []          # tool_use ids awaiting their result
        n = 0

        def _push_user(blocks):
            if out and out[-1]["role"] == "user":
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": "user", "content": blocks})

        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                if content:
                    system_parts.append(str(content))
                continue
            if role == "tool":
                tid = pending_ids.pop(0) if pending_ids else f"call_{n}"
                _push_user([{"type": "tool_result", "tool_use_id": tid,
                             "content": str(content or "")}])
                continue
            if role == "assistant":
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for c in (m.get("tool_calls") or []):
                    fn = (c.get("function") or c)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    tid = f"call_{n}"; n += 1
                    pending_ids.append(tid)
                    blocks.append({"type": "tool_use", "id": tid,
                                   "name": fn.get("name", ""), "input": args or {}})
                out.append({"role": "assistant",
                            "content": blocks or [{"type": "text", "text": ""}]})
                continue
            # user (may carry base64 images)
            blocks = []
            for img in (m.get("images") or []):
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": img}})
            blocks.append({"type": "text", "text": str(content or "")})
            _push_user(blocks)
        # Anthropic requires the first message to be 'user'.
        if not out or out[0]["role"] != "user":
            out.insert(0, {"role": "user", "content": [{"type": "text", "text": "."}]})
        return ("\n\n".join(system_parts), out)

    @staticmethod
    def _tools_to_anthropic(tools):
        out = []
        for t in (tools or []):
            fn = t.get("function", t)
            out.append({"name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters")
                        or {"type": "object", "properties": {}}})
        return out

    def _build_payload(self, model: str, messages: List[dict],
                       schema: Optional[dict], temperature: float,
                       tools: Optional[List[dict]], system: Optional[str],
                       think: Optional[bool], max_tokens: int):
        """Assemble the Messages API payload (separated from the network so the
        caching/thinking/forced-JSON shapes are unit-testable). Returns
        (payload, force_tool_name_or_None)."""
        sys_str, msgs = self._to_anthropic(messages)
        if system:
            sys_str = (system + ("\n\n" + sys_str if sys_str else ""))
        max_tokens = int(max_tokens or 0) or self.max_tokens_default
        # PROMPT CACHING (the dominant cost lever): three ephemeral breakpoints.
        # (a) the system prompt and (b) the tool defs are stable across every
        # step of a turn AND across turns; (c) a moving breakpoint on the last
        # block of the newest message caches the whole conversation prefix, so
        # each agent step re-reads prior context at ~10% of input price instead
        # of full price. Sub-minimum prompts just ignore the marker — harmless.
        if msgs and msgs[-1].get("content"):
            last_block = msgs[-1]["content"][-1]
            if isinstance(last_block, dict) and last_block.get("type") in (
                    "text", "tool_result", "image"):
                last_block["cache_control"] = {"type": "ephemeral"}
        payload: Dict[str, Any] = {
            "model": model, "messages": msgs,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": True,
        }
        if sys_str:
            payload["system"] = [{"type": "text", "text": sys_str,
                                  "cache_control": {"type": "ephemeral"}}]
        # Extended thinking (Sonnet/Opus/Fable): budget must be < max_tokens and
        # temperature must be 1 when thinking is on.
        if think:
            budget = max(1024, min(max_tokens - 256, 4096))
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["temperature"] = 1
            payload["max_tokens"] = max(max_tokens, budget + 512)
        force_tool = None
        if tools:
            # NATIVE WEB SEARCH: swap the harness's local `search` tool for
            # Anthropic's server-side one — the search executes inside this
            # API call (no client round trip, no tool-step budget spent) and
            # returns cited results. The server tool goes FIRST so the cache
            # breakpoint on the last function tool still covers every def.
            fn_tools = tools
            swap_search = (self.native_web_search
                           and any((t.get("function", t)).get("name") == "search"
                                   for t in tools))
            if swap_search:
                fn_tools = [t for t in tools
                            if (t.get("function", t)).get("name") != "search"]
            payload["tools"] = self._tools_to_anthropic(fn_tools)
            if payload["tools"]:
                payload["tools"][-1]["cache_control"] = {"type": "ephemeral"}
            if swap_search:
                payload["tools"].insert(0, {
                    "type": "web_search_20250305", "name": "web_search",
                    "max_uses": self.web_search_max_uses})
        elif schema:
            # No response_format on Anthropic: force a single tool whose input
            # IS the schema, then read the tool input back as the JSON answer.
            force_tool = "structured_output"
            payload["tools"] = [{"name": force_tool,
                                 "description": "Return the result in this exact schema.",
                                 "input_schema": schema}]
            payload["tool_choice"] = {"type": "tool", "name": force_tool}
            payload.pop("thinking", None)     # forced tool + thinking conflict
            # Forced JSON dies ugly when truncated (the tool_use block never
            # closes and NOTHING lands) — give structured calls output headroom
            # regardless of how small the caller's budget was.
            payload["max_tokens"] = max(payload["max_tokens"], 512)
        return payload, force_tool

    def chat(self, model: str, messages: List[dict],
             schema: Optional[dict] = None, temperature: float = 0.2,
             tools: Optional[List[dict]] = None,
             system: Optional[str] = None, on_token=None, cancel=None,
             think: Optional[bool] = None, on_think=None, **kw) -> Completion:
        # TLS or nothing: the key and the family's conversations must never
        # leave this machine in plain text. urlopen's default context already
        # verifies certs + hostname; this guard makes the https requirement
        # explicit so no future edit can silently downgrade it.
        if not self.API.lower().startswith("https://"):
            raise ProviderError(
                "refusing to send the Anthropic key over a non-TLS endpoint")
        payload, force_tool = self._build_payload(
            model, messages, schema, temperature, tools, system, think,
            int(kw.get("max_tokens") or 0))
        if model in self._no_temp:
            payload.pop("temperature", None)
        headers = {"content-type": "application/json",
                   "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}

        text_parts: List[str] = []
        tool_calls: List[dict] = []
        forced_json: List[str] = []
        search_queries: List[str] = []
        block: Dict[str, Any] = {}
        usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
                 "searches": 0}

        def handle(line: str) -> None:
            if not line.startswith("data:"):
                return
            chunk = line[len("data:"):].strip()
            if not chunk:
                return
            try:
                ev = json.loads(chunk)
            except (json.JSONDecodeError, ValueError):
                return
            et = ev.get("type")
            if et == "message_start":
                u = (ev.get("message") or {}).get("usage") or {}
                usage["input"] = int(u.get("input_tokens") or 0)
                usage["cache_read"] = int(u.get("cache_read_input_tokens") or 0)
                usage["cache_write"] = int(u.get("cache_creation_input_tokens") or 0)
            elif et == "content_block_start":
                block.clear()
                block.update(ev.get("content_block") or {})
                if block.get("type") in ("tool_use", "server_tool_use"):
                    block["_args"] = ""
            elif et == "content_block_delta":
                d = ev.get("delta") or {}
                dt = d.get("type")
                if dt == "text_delta":
                    t = d.get("text") or ""
                    text_parts.append(t); _emit(on_token, t)
                elif dt == "thinking_delta":
                    if on_think:
                        try:
                            on_think(d.get("thinking") or "")
                        except GenerationCancelled:
                            raise
                        except Exception:
                            pass
                elif dt == "input_json_delta":
                    block["_args"] = block.get("_args", "") + (d.get("partial_json") or "")
            elif et == "content_block_stop":
                if block.get("type") == "tool_use":
                    try:
                        args = json.loads(block.get("_args") or "{}")
                    except Exception:
                        args = {}
                    if force_tool and block.get("name") == force_tool:
                        forced_json.append(json.dumps(args))
                    else:
                        tool_calls.append({"name": block.get("name", ""),
                                           "arguments": args})
                elif block.get("type") == "server_tool_use":
                    # Server-side tool (web_search): resolved by Anthropic
                    # inside this call — record the query for the trace, but
                    # it is NOT a client tool call to execute.
                    try:
                        q = (json.loads(block.get("_args") or "{}")
                             ).get("query") or ""
                    except Exception:
                        q = ""
                    if block.get("name") == "web_search":
                        search_queries.append(q or "(unknown query)")
                block.clear()
            elif et == "message_delta":
                u = ev.get("usage") or {}
                if u.get("output_tokens"):
                    usage["output"] = int(u["output_tokens"])
                srv = u.get("server_tool_use") or {}
                if srv.get("web_search_requests"):
                    usage["searches"] = int(srv["web_search_requests"])

        try:
            _post_json_stream(self.API, payload, headers, self.timeout, handle,
                              cancel=cancel)
        except ProviderError as exc:
            # Newer models (Sonnet 5+) 400 on ANY `temperature` — the error
            # arrives before a single byte streams, so accumulators are clean:
            # drop the knob, remember the model, retry once.
            if ("temperature" in str(exc) and "deprecated" in str(exc)
                    and "temperature" in payload):
                self._no_temp.add(model)
                payload.pop("temperature", None)
                _post_json_stream(self.API, payload, headers, self.timeout,
                                  handle, cancel=cancel)
            else:
                raise
        # Truncation salvage: if a forced tool_use block never CLOSED (hit
        # max_tokens mid-JSON), surface the partial args — the callers' loose
        # parsers can often still read it, and an honest fragment beats ''.
        if force_tool and not forced_json and block.get("_args"):
            forced_json.append(block.get("_args", ""))
        text = ("".join(forced_json) if force_tool
                else _strip_think("".join(text_parts)))
        # Ledger semantics (OpenAI-style): input_tokens INCLUDES cached reads
        # and fresh = input - cached. Anthropic reports input/read/write as
        # DISJOINT, so rebuild the OpenAI shape: fold cache WRITES in at their
        # true 1.25x rate and add reads back so the ledger's subtraction prices
        # fresh at cost_in and reads at the rung's ~10% cache_read rate.
        billed_in = (usage["input"] + int(usage["cache_write"] * 1.25)
                     + usage["cache_read"])
        return Completion(
            text=text,
            input_tokens=billed_in, output_tokens=usage["output"],
            cached_input_tokens=usage["cache_read"],
            model=model, provider=self.name,
            tool_calls=tool_calls,
            web_searches=max(usage["searches"], len(search_queries)),
            raw={"usage": usage, "search_queries": search_queries})

    # -- Message Batches: the overnight 50% lever ------------------------ #
    BATCH_API = "https://api.anthropic.com/v1/messages/batches"

    @staticmethod
    def _parse_message(msg: dict, force_tool: Optional[str]):
        """Parse a NON-streaming Message object (the shape batch results carry)
        into the same accumulators the SSE handler builds, so both transports
        return identical Completions."""
        text_parts: List[str] = []
        tool_calls: List[dict] = []
        forced_json: List[str] = []
        search_queries: List[str] = []
        for block in (msg.get("content") or []):
            bt = block.get("type")
            if bt == "text":
                text_parts.append(block.get("text") or "")
            elif bt == "tool_use":
                args = block.get("input") or {}
                if force_tool and block.get("name") == force_tool:
                    forced_json.append(json.dumps(args))
                else:
                    tool_calls.append({"name": block.get("name", ""),
                                       "arguments": args})
            elif bt == "server_tool_use" and block.get("name") == "web_search":
                q = (block.get("input") or {}).get("query") or ""
                search_queries.append(q or "(unknown query)")
        u = msg.get("usage") or {}
        usage = {"input": int(u.get("input_tokens") or 0),
                 "output": int(u.get("output_tokens") or 0),
                 "cache_read": int(u.get("cache_read_input_tokens") or 0),
                 "cache_write": int(u.get("cache_creation_input_tokens") or 0),
                 "searches": int(((u.get("server_tool_use") or {})
                                  .get("web_search_requests")) or 0)}
        text = ("".join(forced_json) if force_tool
                else _strip_think("".join(text_parts)))
        return text, tool_calls, search_queries, usage

    def _batch_cancel(self, bid: str, headers: dict) -> None:
        try:
            _post_json(f"{self.BATCH_API}/{bid}/cancel", {}, headers,
                       self.timeout)
        except Exception:
            pass                    # best-effort: the 24h server TTL cleans up

    def _batch_round_trip(self, payload: dict, headers: dict,
                          wait_s: int, poll_s: int, cancel):
        """Submit ONE request as a batch, poll until it ends, return the
        succeeded Message dict — or the error string for the caller to
        classify (temperature retry / overflow / hard fail). Cancels the
        server-side batch on caller-cancel or wait-budget exhaustion so an
        abandoned request can't bill hours later."""
        created = _post_json(self.BATCH_API,
                             {"requests": [{"custom_id": "r0",
                                            "params": payload}]},
                             headers, self.timeout)
        bid = created.get("id") or ""
        status = created
        deadline = time.monotonic() + max(60, int(wait_s))
        while (status.get("processing_status") or "") != "ended":
            if cancel and cancel():
                self._batch_cancel(bid, headers)
                raise GenerationCancelled()
            if time.monotonic() >= deadline:
                self._batch_cancel(bid, headers)
                raise ProviderError(
                    f"batch {bid} still '{status.get('processing_status')}' "
                    f"after {wait_s}s — cancelled")
            time.sleep(max(2, int(poll_s)))
            status = _get_json(f"{self.BATCH_API}/{bid}", headers,
                               self.timeout)
        url = status.get("results_url") or f"{self.BATCH_API}/{bid}/results"
        for line in _get_text(url, headers, self.timeout).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            res = row.get("result") or {}
            if res.get("type") == "succeeded":
                return res.get("message") or {}
            err = res.get("error") or {}
            return json.dumps(err) if isinstance(err, dict) else str(err)
        return "batch ended with no results"

    def chat_batch(self, model: str, messages: List[dict],
                   schema: Optional[dict] = None, temperature: float = 0.2,
                   tools: Optional[List[dict]] = None,
                   system: Optional[str] = None, on_token=None, cancel=None,
                   think: Optional[bool] = None, on_think=None,
                   wait_s: int = 3600, poll_s: int = 20, **kw) -> Completion:
        """chat(), but through the Message Batches API — the same answer at
        HALF the token price (input AND output). The trade is latency: the
        request queues server-side and completes in minutes, not seconds,
        which the background planes that route here (self-dev, dreams,
        scribe) never notice. Nothing streams; a token sink gets the final
        text once. The Completion carries raw['batched']=True so the ledger
        prices it at the 50% batch rate."""
        if not self.API.lower().startswith("https://"):
            raise ProviderError(
                "refusing to send the Anthropic key over a non-TLS endpoint")
        payload, force_tool = self._build_payload(
            model, messages, schema, temperature, tools, system, think,
            int(kw.get("max_tokens") or 0))
        payload.pop("stream", None)     # batch entries are non-streaming
        if model in self._no_temp:
            payload.pop("temperature", None)
        headers = {"content-type": "application/json",
                   "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}
        msg: dict = {}
        for attempt in (0, 1):
            got = self._batch_round_trip(payload, headers, wait_s, poll_s,
                                         cancel)
            if isinstance(got, dict):
                msg = got
                break
            # Errored result: mirror chat()'s classification. The temperature
            # rejection arrives as a per-request error here, not an HTTP 400.
            if (attempt == 0 and "temperature" in got and "deprecated" in got
                    and "temperature" in payload):
                self._no_temp.add(model)
                payload.pop("temperature", None)
                continue
            if _looks_like_overflow(got):
                raise ContextOverflow(got)
            raise ProviderError(f"batch request errored: {got}")
        text, tool_calls, search_queries, usage = self._parse_message(
            msg, force_tool)
        _emit(on_token, text)
        billed_in = (usage["input"] + int(usage["cache_write"] * 1.25)
                     + usage["cache_read"])
        return Completion(
            text=text,
            input_tokens=billed_in, output_tokens=usage["output"],
            cached_input_tokens=usage["cache_read"],
            model=model, provider=self.name,
            tool_calls=tool_calls,
            web_searches=max(usage["searches"], len(search_queries)),
            raw={"usage": usage, "search_queries": search_queries,
                 "batched": True})

    def embed(self, model, text):   # Claude has no embeddings API
        raise ProviderError("anthropic provider has no embeddings endpoint")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_providers(cfg) -> Dict[str, Any]:
    """Return a provider keyed by the names used in config rungs."""
    provs = {
        "ollama_local": OllamaProvider(
            cfg.ollama_local_url, None, cfg.request_timeout, "ollama_local",
            num_ctx=int(getattr(cfg, "local_num_ctx", 0) or 0)),
        "ollama_cloud": OllamaProvider(
            cfg.ollama_cloud_url, cfg.ollama_api_key,
            cfg.request_timeout, "ollama_cloud"),
    }
    # The Claude tiered ladder lights up the moment a key is present — no key,
    # no anthropic provider, and any anthropic rung falls through gracefully.
    key = getattr(cfg, "anthropic_api_key", "") or ""
    if key:
        provs["anthropic"] = AnthropicProvider(
            key, cfg.request_timeout, "anthropic",
            native_web_search=getattr(cfg, "native_web_search", True),
            web_search_max_uses=getattr(cfg, "web_search_max_uses", 3))
    return provs
