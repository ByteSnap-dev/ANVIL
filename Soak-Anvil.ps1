<#
  Soak-Anvil.ps1 - overnight test-and-triage loop for ANVIL.

  Runs the full self-test battery (unit + regression + LIVE tests against your
  local Ollama and Ollama Cloud) on a cycle, logging a report each pass to
  test-reports\. On any failure it asks a reachable model to triage the cause
  and writes suggestions to test-reports\proposals\ - it never edits your code.

  Start it before bed:   right-click -> Run with PowerShell,  or:  ./Soak-Anvil.ps1
  Options: -Minutes 480   -Interval 15   -NoLive
  In the morning, read:   test-reports\soak-summary.md
#>
param(
  [int]$Minutes = 480,
  [int]$Interval = 15,
  [switch]$NoLive
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Get-BasePython {
  if (Get-Command py -ErrorAction SilentlyContinue) { return @{ Exe = "py"; Pre = @("-3") } }
  foreach ($c in @("python", "python3")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) { return @{ Exe = $c; Pre = @() } }
  }
  throw "Python 3.10+ not found. Install from https://www.python.org/downloads/ (tick 'Add to PATH')."
}
$base = Get-BasePython
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $exe = $venvPy; $pre = @() } else { $exe = $base.Exe; $pre = $base.Pre }

# Load .env so live tests can reach Ollama Cloud.
if (Test-Path ".env") {
  Get-Content ".env" | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
      $k, $v = $line.Split("=", 2)
      if ($v) { [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process") }
    }
  }
}
$argv = @("-m", "anvil", "soak", "--minutes=$Minutes", "--interval=$Interval")
if ($NoLive) { $argv += "--no-live" }
Write-Host "[anvil] soak starting - $Minutes min, every $Interval min. Reports in test-reports\" -ForegroundColor Green
Write-Host "[anvil] leave this window open overnight; read test-reports\soak-summary.md in the morning." -ForegroundColor DarkGray
& $exe @($pre + $argv)
