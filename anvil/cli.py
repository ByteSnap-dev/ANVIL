"""ANVIL command line — `python -m anvil <command>`.

Commands:
  ask "<task>"          Run the pipeline once and print the answer.
  note "<text>"         Write a note to memory.
  recall "<query>"      Show what memory would load for a query.
  status                Show today's spend, rung ladder, job count.
  schedule              List scheduled jobs.
  run <job>             Run one scheduled job now.
  serve                 Start the scheduler loop (and Discord bot if configured).
  serve-web [--port=N]  Launch the browser interface (Setup/Chat/Status).
  doctor [--live]       Run the self-test suite once and print a report.
  soak [--minutes=N]    Run the full battery on a loop (overnight churn).
  think                 One heartbeat tick — a spontaneous thought.
  dream                 One sleep/consolidation pass (STM -> LTM + lessons).
  pulse [--minutes=N]   The circadian loop: heartbeats + periodic dreams.
  forge [--cycles=N]    Autonomous git-guarded self-dev loop (Ollama).
  issues [--count=N]    Work the Gitea issue queue (assess -> clarify / fix).
  audit                 Report the install's security posture.
  consolidate           Decay salience, prune notes, rebuild the index.
"""

from __future__ import annotations

import sys

# Windows consoles default to cp1252, which crashes when the model replies with
# an emoji or other non-Latin-1 text. Force UTF-8 so printing answers never dies.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from . import config as cfgmod
from .comms import notify, run_bot
from .memory import MemoryStore
from .pipeline import Pipeline
from .router import Router
from .scheduler import Scheduler


def _load():
    return cfgmod.load()


def cmd_ask(args):
    cfg = _load()
    task = " ".join(args) or sys.stdin.read()
    plan = "--plan" in args
    res = Pipeline(cfg).run(task.replace("--plan", "").strip(), plan=plan)
    print(res.answer)
    print(f"\n- rung={res.rung_name} cost=${res.est_cost_usd:.4f} "
          f"recalled={res.recalled} verdict={res.critic_verdict or 'n/a'}")
    if res.escalations:
        print(f"  escalations: {', '.join(res.escalations)}")
    if res.notes_written:
        print(f"  notes: {', '.join(res.notes_written)}")


def cmd_note(args):
    cfg = _load()
    text = " ".join(args)
    note = MemoryStore(cfg).write(text, type="user")
    print(f"wrote {note.path}")


def cmd_recall(args):
    cfg = _load()
    store = MemoryStore(cfg)
    notes = store.recall(" ".join(args))
    if not notes:
        print("(nothing recalled)")
    for n in notes:
        print(f"[{n.name}] {n.body[:120]}")


def cmd_status(args):
    cfg = _load()
    led = Router(cfg).ledger
    sched = Scheduler(cfg)
    print("ANVIL status")
    print(f"  ladder:    {' -> '.join(r.name for r in cfg.ladder)}")
    print(f"  spent today: ${led.spent_today():.4f} / cap ${cfg.daily_cost_cap_usd:.2f}")
    print(f"  jobs:      {len(sched.load_jobs())} in {cfg.jobs_dir}")
    print(f"  memory:    {len(MemoryStore(cfg).all_notes())} notes in {cfg.memory_dir}")
    print(f"  forge branch: {getattr(cfg, 'forge_branch', 'test')}")
    print(f"  autonomy:  {getattr(cfg, 'autonomy', 'trusted')}")
    print(f"  bind host: {getattr(cfg, 'bind_host', '127.0.0.1')}")
    print(f"  web:       {getattr(cfg, 'bind_host', '127.0.0.1')}:{getattr(cfg, 'server_port', 8765)}")
    print(f"  ollama key: {'configured' if cfg.ollama_api_key else 'local-only'}")
    print(f"  discord:   {'webhook set' if cfg.discord_webhook_url else 'off'}")


def cmd_schedule(args):
    cfg = _load()
    for job in Scheduler(cfg).load_jobs():
        flag = "" if job.enabled else " (disabled)"
        print(f"  {job.cron:<16} {job.name}{flag} -> {job.pipeline}")


def _run_job(cfg, job):
    print(f"[anvil] running job: {job.name}")
    task = job.inputs.get("prompt") or job.inputs.get("command") or job.name
    res = Pipeline(cfg).run(task, min_rung=job.min_rung)
    if job.notify == "discord":
        notify(cfg.discord_webhook_url, f"**{job.name}**\n{res.answer}")
    print(res.answer)
    return res


def cmd_run(args):
    cfg = _load()
    sched = Scheduler(cfg)
    target = args[0] if args else ""
    for job in sched.load_jobs():
        if job.name == target:
            _run_job(cfg, job)
            return
    print(f"no job named {target!r}")


def cmd_consolidate(args):
    cfg = _load()
    stats = MemoryStore(cfg).consolidate()
    print(f"consolidated: {stats}")


def cmd_serve_web(args):
    from . import server
    port = None
    for a in args:
        if a.startswith("--port="):
            port = int(a.split("=", 1)[1])
    server.serve(port=port, open_browser="--no-browser" not in args)


def cmd_serve(args):
    cfg = _load()
    sched = Scheduler(cfg)
    if cfg.discord_bot_token and "--no-bot" not in args:
        import threading
        handlers = _bot_handlers(cfg)
        threading.Thread(
            target=run_bot, args=(cfg.discord_bot_token, handlers),
            daemon=True).start()
        print("[anvil] discord bot started")
    sched.serve(lambda job: _run_job(cfg, job))


def _bot_handlers(cfg):
    def ask(args):
        return Pipeline(cfg).run(args).answer

    def status(_):
        return f"spent today ${Router(cfg).ledger.spent_today():.4f}"

    def note(args):
        MemoryStore(cfg).write(args, type="user")
        return "noted."
    return {"ask": ask, "status": status, "note": note}


def cmd_think(args):
    from . import selftest, mind
    selftest._load_env()
    print(mind.Mind(_load()).think())


def cmd_dream(args):
    from . import selftest, mind
    selftest._load_env()
    import json as _j
    print(_j.dumps(mind.Mind(_load()).dream(), indent=2))


def cmd_pulse(args):
    from . import selftest, mind
    selftest._load_env()
    def opt(name, d):
        for a in args:
            if a.startswith(name + "="):
                return int(a.split("=", 1)[1])
        return d
    mind.Mind(_load()).pulse(minutes=opt("--minutes", 480),
                             interval=opt("--interval", 10),
                             dream_every=opt("--dream-every", 6))


def cmd_forge(args):
    from .forge import Forge
    from . import selftest, selfdev
    selftest._load_env()
    def opt(name, d):
        for a in args:
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return d
    cfg = _load()
    kw = {}
    if "--dry-run" in args:               # verify the loop+git without a model
        kw["driver"] = lambda prompt: None
    # else: cmd builds the Forge first, then attaches a driver pointed at its
    # surgery worktree (the driver must never edit the live checkout).
    if getattr(cfg, "forge_push", False) and getattr(cfg, "gitea_token", None):
        kw["push_remote"] = getattr(cfg, "forge_push_remote", "gitea")
        kw["push_token"] = cfg.gitea_token
    f = Forge(branch=opt("--branch", getattr(cfg, "forge_branch", "test")), **kw)
    if "--dry-run" not in args:           # code with ANVIL's OWN Ollama ladder
        f.driver = selfdev.build_local_driver(cfg, root=f.work_root)
    f.run(cycles=int(opt("--cycles", "20")), minutes=int(opt("--minutes", "0")))


def cmd_issues(args):
    """Lara works the Gitea issue queue: assess -> clarify / push-back / fix on test."""
    from . import selftest, issuework
    selftest._load_env()
    count = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--count=")), 1)
    w = issuework.IssueWorker(_load())
    for _ in range(max(1, count)):
        print(w.run_once())


def cmd_promote(args):
    """Reviewed promotion test -> main: open a PR (merge it too with --approve)."""
    from . import selftest, promote
    selftest._load_env()
    approve = True if "--approve" in args else None
    print(promote.promote(_load(), approve=approve))


def cmd_doctor(args):
    from . import selftest
    # Mark that we're inside the test suite so autonomous self-modification (deep
    # sleep -> self-dev / issue-work -> forge) NEVER runs here. The forge's test gate
    # shells out to `anvil doctor`; without this a test that touches deep sleep would
    # spawn a forge that spawns another `anvil doctor` — an exponential fork bomb.
    import os as _os
    _os.environ["ANVIL_IN_DOCTOR"] = "1"
    only = next((a.split("=", 1)[1] for a in args if a.startswith("--only=")), None)
    rep = selftest.doctor(live="--live" in args, only=only)
    raise SystemExit(0 if rep.green else 1)


def cmd_soak(args):
    from . import selftest
    def opt(name, default):
        for a in args:
            if a.startswith(name + "="):
                return int(a.split("=", 1)[1])
        return default
    selftest.soak(minutes=opt("--minutes", 480), interval=opt("--interval", 15),
                  live="--no-live" not in args)


def cmd_audit(args):
    """Report the install's security posture (bind host, autonomy, funnel, ...)."""
    from . import audit
    cfg = _load()
    findings = audit.run_audit(cfg)
    print(audit.format_report(findings))
    return 0


COMMANDS = {
    "ask": cmd_ask, "note": cmd_note, "recall": cmd_recall,
    "status": cmd_status, "schedule": cmd_schedule, "run": cmd_run,
    "serve": cmd_serve, "serve-web": cmd_serve_web,
    "doctor": cmd_doctor, "soak": cmd_soak, "audit": cmd_audit,
    "think": cmd_think, "dream": cmd_dream, "pulse": cmd_pulse,
    "forge": cmd_forge, "issues": cmd_issues, "promote": cmd_promote,
    "consolidate": cmd_consolidate,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    fn = COMMANDS.get(cmd)
    if not fn:
        print(f"unknown command: {cmd}\n")
        print(__doc__)
        return 2
    fn(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
