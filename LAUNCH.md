# Launching ANVIL (Ollama + Ollama Cloud only)

ANVIL now runs as a small **local web app**: double-click a launcher, a browser
tab opens, and you configure everything — including your Ollama Cloud key and
which model each rung uses — right there in the **Setup** tab. No external API
key required; this build is wired for **local Ollama + Ollama Cloud only**.

## One-time prerequisites

1. **Python 3.10+** — install from python.org and tick *“Add Python to PATH.”*
2. **Ollama** running locally — install from ollama.com, then pull the models:
   ```powershell
   ollama pull qwen3-coder:30b      # rung 0 — local workhorse
   ollama pull qwen3.6:27b          # rung 1 — heavier local reasoning
   ollama pull nomic-embed-text     # semantic note recall (optional)
   ```
   (Use `ollama list` to see exactly what tags you have — the Setup tab reads
   this list and offers them as suggestions.)
3. **Ollama Cloud key** *(optional but recommended)* — create one at
   ollama.com → Settings → API keys. You paste it into the Setup tab; you do
   **not** edit any files. Without it, ANVIL simply stays on the two local
   rungs and the cloud rung is skipped.

## Launch

**Easiest:** double-click **`Start-Anvil.bat`**.

Or from PowerShell:
```powershell
./Start-Anvil.ps1            # default port 8765, opens your browser
./Start-Anvil.ps1 -Port 9000 -NoBrowser
./Start-Anvil.ps1 -SkipInstall   # skip the optional-extras step
```

On WSL/Linux/macOS: `./start-anvil.sh`

The first run creates a local `.venv` and installs the *optional* extras
(`tiktoken`, `discord.py`, …). The core runs on the Python standard library
alone, so if that step fails it’s just a warning — ANVIL still launches.

## First-run setup (in the browser)

1. Open the **Setup** tab.
2. Paste your **Ollama Cloud API key** (or leave blank for local-only).
3. Confirm the **local Ollama URL** (`http://localhost:11434` by default) — the
   page shows a green pill when it can reach Ollama and lists your models.
4. Pick the **model per rung** (the local ones autocomplete from `ollama list`;
   the cloud one defaults to `qwen3-coder:480b-cloud`).
5. **Save configuration.** This writes `anvil.toml` and `.env` for you.
6. Switch to **Chat** and ask something.

## How it behaves in Ollama-only mode

- **Worker** runs on rung 0 (your local `qwen3-coder:30b`) — fast, free,
  private. It does the bulk of every task.
- It **escalates** to rung 1 (`qwen3.6:27b`) and then to the **Ollama Cloud**
  model only on a concrete signal: low self-reported confidence, a JSON-schema
  failure, or a Critic veto.
- The **Planner** and **Critic** run on the Ollama Cloud model (set via
  `planner_rung` / `critic_rung` in `anvil.toml`) so they reason independently
  of the local Worker — the cloud model becomes your verifier. If you have no
  cloud key, they fall back to the local rungs automatically.
- A hard **destructive-command denylist** (`rm -rf`, `mkfs`, `dd if=`, fork
  bombs, …) vetoes regardless of what any model says.

## Run it as a background server (optional)

To keep ANVIL serving without a window, run it as a scheduled task or service:

```powershell
# Headless (no browser pop-up), survives logout via Task Scheduler:
./Start-Anvil.ps1 -NoBrowser
```
Point Windows **Task Scheduler** at `Start-Anvil.bat` with *“Run whether user
is logged on or not”* to have the interface (and the job scheduler, via
`python -m anvil serve`) come up on boot.

## Command line (no browser)

Everything in the UI is also a CLI command:
```powershell
python -m anvil status
python -m anvil ask "find the 10 largest files under /var"
python -m anvil note "anvil-01 needs pve-firewall restart after bridge changes"
python -m anvil serve-web --port 8765
```
