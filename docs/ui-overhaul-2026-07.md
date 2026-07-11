# The Hearthlight Overhaul — UI/UX plan (2026-07-08)

Three parallel expert reviews (IA/flows, visual system, mobile/PWA/a11y) over the
full inline UI (server.py build 55), plus live desktop + 375px captures. Verdict,
unanimous: **the bones are excellent — cross-device approvals, actor-scoped
memory, streaming with thinking-trace, push, safe-areas — but they wear a
homelab dashboard's clothes.** Seven developer tabs, IRC feeds, "sal 0.42" and
"cloud spend by plane" on the family's phones; Lara's own answers are the
smallest, least-designed text in the app; and 2 of 7 tabs literally don't fit
on an iPhone.

## The organizing idea: Hearth & Workshop

Two registers, one token system. **Hearth** is what the family sees: warm,
glanceable, plain-spoken, thumb-first. **Workshop** is the admin's engine room:
dense, technical, earned (admin-only) — everything currently splattered across
Pulse/Jobs/Server survives there untouched. The fire heritage stays but the
metaphor shifts from *forge* (industrial) to *hearth* (kept-warm, domestic).

## Target information architecture

Bottom tab bar (phone) / left rail (desktop). Max 4 tabs + admin-only 5th:

| Tab | Contents | Child | Adult | Admin |
|---|---|---|---|---|
| **Home** (new, default) | house glance (the unused /api/ha!), today (weather, reminders, overnight digest in plain words), Lara presence strip, pending approvals, quick actions | read + safe controls | + approvals | + health chip |
| **Chat** | rebuilt flow (below) | approval = "waiting for a parent" | inline approve | full |
| **Family** | shared memories, shared lists (new), reminders-to-others, profiles, **approval audit log** (new) | lists, who's home | + approve/audit | + manage |
| **Me** | my memory (plain cards, no salience numbers), my reminders in natural language (cron demoted to "advanced"), notifications, password/sign-out | simplest screen in the app | richer | same |
| **System** (admin) | Pulse + Server + raw Jobs + HA setup UI (new) + persona + push diagnostics | — | — | everything |

Structural rules: ONE identity system (cookie when auth on — the in-app profile
picker dies; device-bound binding when auth off, never sid-bound so New Chat
can't demote a verified adult). ONE brand (Lara everywhere family-facing;
ANVIL retreats to System). Status is ambient (presence chip + approval badge in
the shell), never hunted. Errors are cards with verbs (Retry / Ask again),
never `error: TypeError`.

## Design language: Hearthlight

Evolves Warm Graphite (the token architecture held; the discipline didn't —
~120 inline styles, 3 pill systems, hard contrast failures). Full spec in the
visual review; the essentials:

- **Warm-neutral palette**, dark "Evening" (#131110 base — brown-black, not
  navy) + designed-not-derived light "Daylight" (#faf6f0 warm paper). Default
  theme: **auto**. Ember (#f0883e) kept but spent ONLY on action + presence;
  new `--warn` frees it from warning duty; per-person identity hues; every
  status ships -soft/-line pairs; contrast floor 4.5:1 asserted in a test.
- **Type on a 16px body**, true 1.2 scale. Lara's prose: 16px/1.65,
  max-width 68ch — her voice becomes the best-set text in the app (today:
  13.5px, unbounded line length, smaller than the user's own bubble).
- **One component library** (Button/Chip/Card/ListRow/Field/Switch/Sheet/
  Message/ApprovalCard/PresenceAvatar/EmptyState/Toast/ActivityList/Icon set)
  — JS composes components, never writes `style=` again. Enforced: a doctor
  check greps for inline styles / raw hex / raw px font-sizes and fails the
  build. Manifest/theme-color/icon hexes generated from the tokens (kills the
  three-different-blacks drift).

### Three signature moments
1. **The Kept Flame** — the flame avatar becomes Lara's living presence:
   resting (slow breathing glow) / listening (warms on composer focus) /
   thinking (flicker + rotating glow ring; the thinking-trace docks under it)
   / speaking (steady bright). Replaces the unexplained header dot AND the
   typing dots. Reduced-motion: static levels.
2. **Asking at the Door** — the approval card as the product's trust ceremony:
   who (person chip) / what ("Lara wants to unlock the front door", command
   behind a disclosure) / why (the child's ask, quoted); big Allow / Not now;
   live expiry countdown; per-card token (kills the single-slot PENDING
   global); child's device shows "Waiting for a parent — they've been
   notified" with resolved-by attribution — never a PIN field for kids to
   brute-force; terminal "expired — ask again" state.
3. **Warm Ink** — progressive block-level markdown while streaming (no
   plaintext→markdown flash), breathing ember caret, blocks settle with a
   150ms fade; the routing telemetry collapses to "answered in 8s ⌄"
   (Workshop register sees the full line).

## Mobile/PWA foundation (MUSTs)

Bottom nav ≥49pt; every target ≥44pt; no hover-gated actions (chat
rename/pin/delete are hover-only today = nonexistent on iPhones); inputs ≥16px
(kills iOS focus-zoom); visualViewport composer pinning; overscroll containment;
**offline shell** (SW precache keyed by UI_BUILD + offline banner + cached last
conversation); **SSE transport** replacing the 250ms/8s/90s poll trio;
**durable background asks** (locking the phone must never kill an answer —
finish, persist to the conversation, push; the 60→300s widening shipped as a
stopgap in PR #44); **deep-linked notifications** (payload carries
/chat/<sid> or /approvals/<token>; taps focus, not reload); update politeness
("Update ready" pill; auto-reload only when idle). A11y: real dialog semantics
+ focus traps, tablist nav, aria-live for answers/status, rem type with
Dynamic Type, kid mode (larger type, simplified copy, zero pipeline metadata).

## Phases

- **Phase 0 — Accounts (SHIPPED, PR #44):** username+password login,
  remember-me, rate limiting, first-run household wizard, legacy PIN
  migration. The wizard IS the new onboarding's first screen.
- **Phase 1 — Foundation:** split INDEX_HTML into static assets (also enables
  SW caching); Hearthlight tokens; component library + enforcement check;
  bottom-nav shell with role gating; type/contrast/target fixes ride along.
- **Phase 2 — Transport (SHIPPED, builds 71–72):** /api/stream SSE (pushed
  twin of /api/progress, 150ms-throttled snapshots, poll = automatic
  fallback); durable asks (abandon-cancel retired — only Stop cancels; a
  dropped client watches the transcript for the landed answer); durable
  transcript (per-turn meta rebuilds trace/chips/shell after reload);
  offline shell (network-first SW cache + offline bar). Remaining tail:
  notification deep links (filed as issue #65).
- **Phase 3 — Family surfaces (mostly SHIPPED):** Chat rebuilt (typewriter,
  ApprovalCard v2, expandable transparency trace, inline live shell, durable
  transcript); Me tab (password card, memory mirror + browser sheet, my
  scheduled jobs); the Home dashboard was CUT by operator decision — chat IS
  home, HA stays in Home Assistant. Remaining: Family audit log (#62) +
  shared lists (#63), filed as Lara issues.
- **Phase 3.5 — Showing, not just telling (the Viewer):** Lara gets a way to
  SHOW things, not just describe them. One responsive container, two forms:
  on desktop (>=900px) a right-side panel that the chat column yields to
  (55/45, draggable divider); on phones a full-height bottom sheet with
  drag-down-to-dismiss and the chat still reachable behind it. Same component,
  same content pipeline, CSS decides the form — no per-device code paths.
  Content sources, in build order:
  1. **Turn artifacts:** tool calls that touched files (read_file/write_file
     paths already flow through the trace) become tappable chips under the
     answer ("view forge/PLAN.md") that open the Viewer — read-only, rendered
     markdown / syntax-lit text / images.
  2. **A `show` tool:** Lara deliberately presents a file, image, or fragment
     while talking ("here's the plan:") — the Viewer opens alongside her
     message; the chip persists in the transcript so it survives reload.
  3. **Sandboxed HTML preview** (iframe, srcdoc, no network) for web things
     she builds — the Claude-style live preview.
  Server side: one /api/file endpoint, workspace-rooted (no traversal),
  size-capped, ROLE-GATED — admin sees repo files; family accounts only the
  family-facing dirs (docs/, recipes, shared lists). The viewer is also where
  background-work transparency lands eventually: tapping a running job's
  activity opens its live trace in the same panel.
- **Phase 4 — Roles & Workshop:** kid mode; System tab consolidation
  (Pulse/Server/Jobs/HA-setup); per-role trimming; brand unification.

**Quick wins already shipped:** abandon window 60→300s (PR #44; abandon then
fully retired in PR #61), login form 16px inputs (PR #44), polite updates
(hidden→silent swap, visible→"Update ready" pill).

**Deliberately NOT built — composer mic button:** `webkitSpeechRecognition`
is unsupported/unreliable in iOS *installed* PWAs (the family's primary
surface), while the iOS keyboard's built-in dictation already works in the
composer. A custom mic would be a dead control on the target devices;
revisit only if an Android/desktop user asks. **Quick-win batch for Lara (guided issues):** hover-gated
chat actions visible on coarse pointers; sub-44px padding fixes; --dim contrast
bump; orphaned "working…" bubble error card; poll relief (750ms after 5s,
pause on hidden); aria-live/dialog roles on overlays; notification pre-prompt
after first answer; geoRefresh deferred to first geo ask.

## Execution model

Same rhythm as everything else on this repo: each phase lands doctor-green on
`test` → auto-promotes. Phase 1's asset split is the enabling move and mine to
hand-build; the quick-win batch and many Phase 3/4 slices are Lara-shaped
guided issues once the component vocabulary exists. Screenshots + full review
texts referenced from this doc's reviews (2026-07-08 session).
