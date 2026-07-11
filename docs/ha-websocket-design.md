# Design: HA websocket event stream to replace snapshot-diff sensing

*(issue #42 — design only; no code changes ship with this doc)*

## Today: snapshot-diff

`mind.sense_house()` pulls a full `/api/states` snapshot (through the 5s
shared cache in `homeassistant.py`), diffs it against the previous snapshot
held on the mind object (`_house_snap`), and records only the changes into
STM as `house` events. It runs on the autopilot tick.

What's wrong with it, in practice:

- **Latency**: a change is only noticed on the next tick. "The dryer
  finished" can be minutes stale by the time it lands in STM.
- **Missed transients**: anything that changes and changes back between two
  ticks (a door opened and closed, a motion pulse) is invisible.
- **Cost per observation**: every tick transfers the WHOLE house state
  (hundreds of entities) to notice that usually nothing changed.
- **Ordering**: two changes between ticks arrive as one diff with no
  sequence — the story "motion, then light on" flattens to "both changed".

## Proposed: `/api/websocket` subscription

Home Assistant's websocket API pushes `state_changed` events as they
happen. One long-lived connection replaces the polling loop:

```
connect -> auth {access_token} -> subscribe_events(state_changed)
   -> event stream: {entity_id, old_state, new_state, time_fired}
```

### Architecture

- New module `anvil/ha_stream.py`: a daemon thread owning ONE websocket
  connection (stdlib-only is not possible here — add `websockets` or use a
  minimal hand-rolled client over `http.client` upgrade; prefer the tiny
  `websocket-client` package, sync API, no asyncio in the harness).
- The thread maintains a bounded in-memory ring (say 500 events) plus the
  same "record changes into STM as `house` events" write path sense_house
  uses today — throttled per entity (see below).
- `homeassistant.states()` keeps working exactly as-is for on-demand reads
  (tools, house snapshot); the stream only replaces the *ambient sensing*
  path.

### Filtering and throttle (the important part)

Raw `state_changed` is a firehose (power sensors update every few seconds).
STM must not drown:

- **Entity allow-classes**: person, binary_sensor (door/motion/presence
  classes), lock, cover, media_player, climate mode, light on/off (not
  brightness), switch — the same classes sense_house cares about today.
- **Attribute-only changes are dropped** unless the state string changed.
- **Per-entity cooldown**: at most one STM event per entity per N minutes
  (default 5), except person/lock/door which are always recorded.
- **Debounce transients**: a change that reverts within 30s is recorded as
  one compound event ("front door opened briefly") — this *adds* signal the
  snapshot-diff never had.

### Resilience

- Reconnect with jittered exponential backoff (2s → 60s cap), re-auth and
  re-subscribe on every connect. HA restarts nightly for updates — this is
  the common path, not the edge case.
- On (re)connect, do ONE snapshot diff against the last known state to
  catch events missed while disconnected — the old code path becomes the
  gap-filler instead of the engine.
- A `stream_healthy` flag; when False for >5 min, the autopilot tick falls
  back to today's snapshot-diff sensing (nothing regresses if the socket
  can't stay up).
- Auth failure (revoked token) disables the stream and surfaces ONCE in the
  system view, like other degradations.

### What changes where

| file | change |
| --- | --- |
| `anvil/ha_stream.py` | new: the connection thread, filter, throttle, ring |
| `anvil/mind.py` | `sense_house()` consumes the ring when the stream is healthy; falls back to snapshot-diff otherwise |
| `anvil/server.py` | start/stop the stream with the pulse; a line in the System view (connected / events seen / last event) |
| `anvil.toml` | `ha_stream = true` kill-switch, cooldown knobs |

### Migration steps

1. Ship `ha_stream.py` behind `ha_stream = false` (off), with a doctor test
   faking the socket (inject frames, assert filter/throttle/ring behavior).
2. Flip on live for a week with BOTH paths recording (stream events marked)
   — compare STM quality/volume.
3. Make the stream the default; snapshot-diff stays as reconnect gap-filler
   and fallback.

### Risks

- A new third-party dependency (`websocket-client`) in a stdlib-proud
  harness — smallest available, sync, vendorable if needed.
- Event storms during HA automations — the throttle is the guard; the ring
  is bounded either way.
- Thread lifetime across server restarts — the operator restarts the server
  a LOT; daemon thread + idempotent start makes that a non-issue.
