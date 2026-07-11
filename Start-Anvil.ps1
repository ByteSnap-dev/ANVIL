<#
  Start-Anvil.ps1 - one-step launcher for the ANVIL web interface on Windows.
    1. Finds Python (py launcher, or python/python3 on PATH).
    2. First run: creates .venv and installs OPTIONAL extras (failures are
       warnings, not errors - the core needs none of them).
    3. Loads .env, starts the server, opens your browser.

  Usage:   right-click -> Run with PowerShell,  or:   ./Start-Anvil.ps1
  Options: -Port 8765   -NoBrowser   -SkipInstall
#>
param(
  [int]$Port = 0,
  [switch]$NoBrowser,
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Get-BasePython {
  if (Get-Command py -ErrorAction SilentlyContinue) { return @{ Exe = "py"; Pre = @("-3") } }
  foreach ($c in @("python", "python3")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) { return @{ Exe = $c; Pre = @() } }
  }
  throw "Python 3.10+ was not found. Install it from https://www.python.org/downloads/ (tick 'Add Python to PATH')."
}

$base = Get-BasePython
Write-Host "[anvil] using Python: $($base.Exe) $($base.Pre -join ' ')" -ForegroundColor DarkGray

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not $SkipInstall) {
  try {
    if (-not (Test-Path $venvPy)) {
      Write-Host "[anvil] creating .venv (first run only)..." -ForegroundColor Cyan
      & $base.Exe @($base.Pre) -m venv .venv
    }
    Write-Host "[anvil] installing optional extras (tiktoken, discord.py, ...)" -ForegroundColor Cyan
    & $venvPy -m pip install --quiet --upgrade pip | Out-Null
    & $venvPy -m pip install --quiet -r requirements.txt
  } catch {
    Write-Warning "[anvil] optional extras skipped ($($_.Exception.Message)). Core still runs."
  }
}

if (Test-Path $venvPy) { $exe = $venvPy; $pre = @() } else { $exe = $base.Exe; $pre = $base.Pre }

# Load .env into this process (so the Ollama Cloud key is available).
if (Test-Path ".env") {
  Get-Content ".env" | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
      $k, $v = $line.Split("=", 2)
      if ($v) { [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process") }
    }
  }
  Write-Host "[anvil] loaded .env" -ForegroundColor DarkGray
}

$argv = @("-m", "anvil", "serve-web")
if ($Port -gt 0)  { $argv += "--port=$Port" }
if ($NoBrowser)   { $argv += "--no-browser" }

Write-Host "[anvil] starting web interface..." -ForegroundColor Green
& $exe @($pre + $argv)
