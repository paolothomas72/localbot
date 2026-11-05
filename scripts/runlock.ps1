<#
.SYNOPSIS
  The run lock for Windows PowerShell 5.1: the same lock as scripts\runlock.py (audit E-3).

.DESCRIPTION
  Dot-source this file, then hold the lock for the whole job:

    . (Join-Path $PSScriptRoot "runlock.ps1")
    $lock = Enter-RunLock -Name "gpu_batch"
    if (-not $lock) { "busy: " + (Get-RunLockHolder); exit 3 }
    try { <the job> } finally { Exit-RunLock $lock }

  The lock is a byte-range lock on byte 0 of <main checkout>\state\run.lock ($env:LOCALBOT_RUNLOCK_PATH
  overrides the path): FileStream.Lock(0, 1), the byte scripts\runlock.py locks with msvcrt.locking,
  so the two exclude each other. Windows drops the lock when the process dies, so a stale lock never
  blocks. Who holds it goes next to it in run.lock.json (pid, name, cmd, start, start_epoch); that
  file is information only. Never waits: busy returns $null at once.
#>

# The main checkout: the parent of git's common dir, so every worktree of the repo finds the same
# folder. Without git: this repo.
function Get-MainCheckout {
    $ErrorActionPreference = "Continue"
    $repo = Split-Path -Parent $PSScriptRoot
    $main = $repo
    Push-Location -LiteralPath $repo
    try {
        $global:LASTEXITCODE = 0
        $common = @(& git rev-parse --path-format=absolute --git-common-dir 2>$null)
        if ($LASTEXITCODE -eq 0 -and $common.Count -gt 0 -and "$($common[0])".Trim()) {
            $main = Split-Path -Parent ([System.IO.Path]::GetFullPath("$($common[0])".Trim()))
        }
    } catch {
        $main = $repo
    } finally {
        Pop-Location
    }
    return $main
}

# The main checkout's state\run.lock (or $env:LOCALBOT_RUNLOCK_PATH).
function Get-RunLockPath {
    if ($env:LOCALBOT_RUNLOCK_PATH) { return $env:LOCALBOT_RUNLOCK_PATH }
    return (Join-Path (Get-MainCheckout) "state\run.lock")
}

# The holder file's text (one line of JSON), or $null when there is none.
function Get-RunLockHolder {
    param([string]$Path = "")
    if (-not $Path) { $Path = Get-RunLockPath }
    try {
        return ([System.IO.File]::ReadAllText($Path + ".json")).Trim()
    } catch {
        return $null
    }
}

function Write-RunLockHolder([string]$File, $Info) {
    $text = ($Info | ConvertTo-Json -Compress) + "`n"
    $tmp = "$File.$PID.tmp"
    [System.IO.File]::WriteAllText($tmp, $text, (New-Object System.Text.UTF8Encoding($false)))
    try {
        if (Test-Path -LiteralPath $File) {
            [System.IO.File]::Replace($tmp, $File, [NullString]::Value)
        } else {
            [System.IO.File]::Move($tmp, $File)
        }
    } catch {
        [System.IO.File]::Copy($tmp, $File, $true)
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

# Take the lock: returns the open FileStream (keep it until Exit-RunLock), or $null when busy.
function Enter-RunLock {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$Path = "",
        [string]$Cmd = ""
    )
    if (-not $Path) { $Path = Get-RunLockPath }
    $Path = [System.IO.Path]::GetFullPath($Path)
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite,
                                 [System.IO.FileShare]::ReadWrite)
    $locked = $false
    for ($i = 0; $i -lt 3 -and -not $locked; $i++) {
        # a status probe holds the lock for an instant; a couple of quick retries ride over it
        try { $fs.Lock(0, 1); $locked = $true } catch { Start-Sleep -Milliseconds 50 }
    }
    if (-not $locked) {
        $fs.Dispose()
        return $null
    }
    if (-not $Cmd) { $Cmd = [Environment]::CommandLine }
    $now = [DateTime]::UtcNow
    $info = [ordered]@{
        pid = $PID
        name = $Name
        cmd = $Cmd
        start = $now.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", [System.Globalization.CultureInfo]::InvariantCulture)
        start_epoch = [Math]::Round(($now - [DateTime]::new(1970, 1, 1, 0, 0, 0, [DateTimeKind]::Utc)).TotalSeconds, 3)
    }
    try { Write-RunLockHolder ($Path + ".json") $info } catch { }  # information only; the lock is held
    return $fs
}

# Let the lock go. The holder file is removed first, and only if it is still this process's.
function Exit-RunLock {
    param($Lock)
    if (-not $Lock) { return }
    try {
        $file = $Lock.Name + ".json"
        $h = $null
        try { $h = [System.IO.File]::ReadAllText($file) | ConvertFrom-Json } catch { }
        if ($h -and $h.pid -eq $PID) { Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue }
    } finally {
        try { $Lock.Unlock(0, 1) } catch { }
        $Lock.Dispose()
    }
}
