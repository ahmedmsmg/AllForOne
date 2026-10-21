# AllForOne installer for Windows 10 and 11. In PowerShell:
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/ahmedmsmg/AllForOne/main/install.ps1))) <join-code>
#
# The join code is a Tailscale auth key (tskey-auth-...) from whoever hosts the group. Without
# one, this machine keeps whatever Tailscale network it is already on.
#
# What it does:
#   1. installs the AllForOne programs that match this PC's GPU into %USERPROFILE%\.allforone\bin
#   2. installs Tailscale if needed and joins the group's network with the join code
#   3. allows AllForOne through Windows Firewall for Tailscale peers only (one admin prompt)
#   4. starts the AllForOne agent in the background, now and at every login
param([string]$JoinCode = "")

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is many times slower with a progress bar
$Repo = "ahmedmsmg/AllForOne"
$Dir = Join-Path $env:USERPROFILE ".allforone"
$Bin = Join-Path $Dir "bin"

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# --- which build ---------------------------------------------------------------------------
# NVIDIA builds are made per GPU generation (compute capability) with CUDA 12.8, so they need a
# matching card and a driver that supports CUDA 12.8 or newer.
$variant = "windows-x86_64-cpu"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $smi = (nvidia-smi) -join "`n"
    $cap = ((nvidia-smi --query-gpu=compute_cap --format=csv,noheader) | Select-Object -First 1).Trim().Replace(".", "")
    if ($smi -match "CUDA Version:\s*(\d+)\.(\d+)" -and [version]"$($Matches[1]).$($Matches[2])" -ge [version]"12.8") {
        if (@("75", "80", "86", "89", "90", "120") -contains $cap) { $variant = "windows-x86_64-cuda-sm$cap" }
        else { Say "No GPU build for compute capability $cap; using the CPU build." }
    } else { Say "The NVIDIA driver is older than CUDA 12.8; using the CPU build (update the driver to use the GPU)." }
}

function Get-Build($name) {
    $url = "https://github.com/$Repo/releases/latest/download/afo-$name.zip"
    $zip = Join-Path $env:TEMP "afo-$name.zip"
    Say "Downloading AllForOne ($name)"
    try { Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing } catch { return $false }
    Get-Process afo, afo-node -ErrorAction SilentlyContinue | Stop-Process -Force
    if (Test-Path $Bin) { Remove-Item -Recurse -Force $Bin }
    New-Item -ItemType Directory -Force $Bin | Out-Null
    Expand-Archive -Path $zip -DestinationPath $Bin -Force
    Remove-Item $zip
    return $true
}
if (-not (Get-Build $variant)) {
    if ($variant -like "*-cuda-*") {
        Say "No GPU build available; using the CPU build"
        $variant = "windows-x86_64-cpu"
        if (-not (Get-Build $variant)) { throw "could not download the AllForOne build" }
    } else { throw "could not download the AllForOne build" }
}

# Put the programs on PATH for new terminals, and for this one.
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not ($userPath -split ";" | Where-Object { $_ -eq $Bin })) {
    [Environment]::SetEnvironmentVariable("Path", "$Bin;$userPath", "User")
}
$env:Path = "$Bin;$env:Path"

# --- Tailscale ------------------------------------------------------------------------------
$tsDir = Join-Path $env:ProgramFiles "Tailscale"
if (-not (Get-Command tailscale -ErrorAction SilentlyContinue) -and -not (Test-Path (Join-Path $tsDir "tailscale.exe"))) {
    Say "Installing Tailscale"
    winget install --id Tailscale.Tailscale -e --silent --accept-source-agreements --accept-package-agreements
}
if (Test-Path $tsDir) { $env:Path = "$env:Path;$tsDir" }

if ($JoinCode) {
    Say "Joining the group's network"
    $hostname = ($env:COMPUTERNAME.ToLower() -replace "[^a-z0-9-]", "-")
    tailscale up --auth-key=$JoinCode --hostname=$hostname --unattended
    if ($LASTEXITCODE -ne 0) { throw "tailscale up failed; check the join code" }
} else {
    tailscale status *> $null
    if ($LASTEXITCODE -ne 0) { Say "Tailscale isn't connected yet. Run this installer with the group's join code, or run: tailscale up" }
}

# --- firewall: only Tailscale peers may reach the agent and nodes ----------------------------
$rules = @"
Get-NetFirewallRule -DisplayName 'AllForOne *' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
foreach (`$exe in 'afo.exe', 'afo-node.exe', 'afo-client.exe') {
    New-NetFirewallRule -DisplayName "AllForOne `$exe" -Direction Inbound -Action Allow -Profile Any ``
        -Program (Join-Path '$Bin' `$exe) -RemoteAddress 100.64.0.0/10 | Out-Null
}
"@
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
Say "Allowing AllForOne through Windows Firewall (Tailscale peers only)"
if ($admin) { Invoke-Expression $rules }
else {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($rules))
    Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList "-NoProfile", "-EncodedCommand", $encoded
}

# --- agent in the background ----------------------------------------------------------------
New-Item -ItemType Directory -Force (Join-Path $Dir "logs") | Out-Null
$log = Join-Path $Dir "logs\agent.log"
$afo = Join-Path $Bin "afo.exe"
$arg = "-NoProfile -WindowStyle Hidden -Command `"`$env:Path = '$Bin;' + `$env:Path + ';$tsDir'; & '$afo' agent *> '$log'`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "AllForOne Agent" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Stop-ScheduledTask -TaskName "AllForOne Agent" -ErrorAction SilentlyContinue
Get-Process afo -ErrorAction SilentlyContinue | Stop-Process -Force
Start-ScheduledTask -TaskName "AllForOne Agent"

Say "Done. This PC is now offering itself to the group ($variant)."
Write-Host "    To start a session from this PC:  afo host   (in a new terminal)"
Write-Host "    Agent log:                         $log"
