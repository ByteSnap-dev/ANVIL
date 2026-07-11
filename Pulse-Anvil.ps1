<#
  Pulse-Anvil.ps1 - ANVIL's heartbeat + sleep/dream loop (autonomy).

  Runs a "heartbeat" every few minutes (a spontaneous thought from recent
  short-term memory) and periodically "dreams" - consolidating short-term memory
  into lasting lessons, questions, and self-improvement proposals
  (test-reports\proposals\). Thoughts appear in the web UI's Mind tab and in
  memory\journal.md.

  Start it before bed:  ./Pulse-Anvil.ps1
  Options: -Minutes 480   -Interval 10   -DreamEvery 6
#>
param([int]$Minutes = 480, [int]$Interval = 10, [int]$DreamEvery = 6)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
function Get-BasePython {
  if (Get-Command py -ErrorAction SilentlyContinue) { return @{ Exe = "py"; Pre = @("-3") } }
  foreach ($c in @("python","python3")) { if (Get-Command $c -ErrorAction SilentlyContinue) { return @{ Exe=$c; Pre=@() } } }
  throw "Python 3.10+ not found."
}
$base = Get-BasePython
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $exe=$venvPy; $pre=@() } else { $exe=$base.Exe; $pre=$base.Pre }
if (Test-Path ".env") { Get-Content ".env" | ForEach-Object {
  $l=$_.Trim(); if ($l -and -not $l.StartsWith("#") -and $l.Contains("=")) { $k,$v=$l.Split("=",2); if ($v){ [Environment]::SetEnvironmentVariable($k.Trim(),$v.Trim(),"Process") } } } }
Write-Host "[anvil] pulse starting - heartbeat every $Interval min, dream every $DreamEvery ticks." -ForegroundColor Green
& $exe @($pre + @("-m","anvil","pulse","--minutes=$Minutes","--interval=$Interval","--dream-every=$DreamEvery"))
