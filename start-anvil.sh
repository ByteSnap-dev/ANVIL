#!/usr/bin/env bash
# Launcher for ANVIL on Linux / WSL / macOS. Mirrors Start-Anvil.ps1.
set -euo pipefail
cd "$(dirname "$0")"

PY="$(command -v python3 || command -v python)"
[ -z "$PY" ] && { echo "Python 3.10+ not found. Install it first."; exit 1; }

if [ "${1:-}" != "--skip-install" ]; then
  if [ ! -d .venv ]; then
    echo "[anvil] creating .venv (first run only)..."
    "$PY" -m venv .venv
  fi
  # shellcheck disable=SC1091
  . .venv/bin/activate
  pip install --quiet --upgrade pip || true
  pip install --quiet -r requirements.txt || echo "[anvil] optional extras skipped; core still runs."
  PY="python"
fi

# Load .env if present.
if [ -f .env ]; then set -a; . ./.env; set +a; echo "[anvil] loaded .env"; fi

echo "[anvil] starting web interface..."
exec "$PY" -m anvil serve-web "$@"
