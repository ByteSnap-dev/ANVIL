# Design: distilling imported chat history into memory (importer slice 3)

*(issue #95 slice 3 — design only. Slices 1–2 shipped: `anvil/imports.py`
parses ChatGPT/Claude/Gemini exports into a common turn format. This doc
gates slice 3's build; nothing writes to memory until this is agreed.)*

## The danger being designed against

Bulk-importing "facts" from years of old chats is exactly how confabulation
re-enters a memory system we spent weeks making faithful. Old chats contain:
stale facts (previous house, previous job), roleplay and hypotheticals,
OTHER people's info, the assistant's own hallucinations, and one-off context
that reads like preference ("I hate cilantro" said while ordering for a
friend). The evidence-aware gate exists because ambient memory once
fabricated; imports are that risk at 1000x volume.

## Principles

1. **Imports produce CANDIDATES, never memories.** Nothing lands in LTM
   directly from an import — everything goes through a review surface.
2. **Provenance is permanent.** Every distilled candidate carries
   `source: import/<provider>/<conversation-id>` and the quoted turn(s) it
   came from. A memory that can't show its quote doesn't get proposed.
3. **Owner-private by default.** Candidates belong to the profile who ran
   the import; the default-private sharing model applies unchanged.
4. **Recency-weighted skepticism.** Anything older than ~12 months is
   distilled as "was true as of <date>" phrasing or dropped; preferences
   need 2+ independent occurrences across conversations to qualify.

## Pipeline (slice 3 when built)

```
imports.parse_*()  ->  distill (Haiku, batched)  ->  candidates.jsonl
                                                        |
                              Me tab review sheet  <----+
                              [keep] [keep-private] [drop]
                                     |
                              MemoryStore.add(owner=importer, source=...)
```

- **Distill prompt** (per conversation, Haiku, plane=`selfdev`-style budget):
  extract ONLY durable, first-person facts/preferences about the importer —
  identity, recurring preferences, standing constraints. Explicitly refuse:
  third parties, one-offs, anything the assistant said (only USER turns are
  evidence), anything speculative. Output must quote the supporting user
  turn verbatim (the evidence gate, applied at distill time).
- **Dedup pass** against existing LTM (embedding or FTS match) — an import
  must not create 40 near-copies of "Alex likes brisket".
- **Volume guard**: cap candidates per import run (default 100) and total
  distill spend (uses the background budget; the plane-scoped cap applies).

## Review UX (the part that keeps trust)

A sheet on the Me tab: "From your ChatGPT import — keep what's true."
Each card: the proposed memory, the verbatim quote, the source
conversation title/date, buttons keep / drop. Bulk "drop all" always
visible. NOTHING is kept without a tap; closing the sheet keeps nothing.

## Explicitly rejected alternatives

- **Auto-import with salience decay** — rejected: decay doesn't fix wrong,
  it just makes wrong quieter. Human review is the gate.
- **Importing assistant turns as evidence** — rejected: that's importing
  another model's hallucinations as fact.
- **Family-shared candidates** — rejected for v1: sharing stays a
  deliberate per-memory act by the owner, same as today.

## Slice 4 (upload UI) prerequisite

The Setup-tab upload (admin-only) ships only after slice 3's review sheet
exists — an upload with nowhere safe to send its output would pressure
toward auto-import.
