# Installs the `localbot` command for the current user (C3, C3b).
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_command.ps1
#
# What it does, and nothing else:
#   1. checks that the repo's Python can start Local Bot from this repo (nothing is installed): it runs
#      "<python>" -P -c "import localbot.chat" from the temp folder with PYTHONPATH=<repo>. If that fails it stops
#      with a plain message and changes nothing.
#   2. writes %USERPROFILE%\.localbot\bin\localbot.cmd. The command sets PYTHONPATH to this repo and runs
#      "<python>" -P -m localbot %*   (-P: the folder you are in is never put first on Python's import path, so a
#      file such as json.py there cannot replace real code). Running the script again rewrites the file.
#   3. adds that one bin folder to the USER path, once (running it twice leaves one entry). The path is edited as
#      the raw registry value: %NAME% parts stay as they are and the value keeps its type (expandable or plain).
#      Then Windows is told the environment changed (SendMessageTimeout with WM_SETTINGCHANGE "Environment"), so a
#      new PowerShell window sees the change.
# It does not put the virtual environment's own folder on the path.
param([string]$Python = "")

function Assert-CmdSafe {
    # cmd.exe would misread a % (variable) or a " (quote) inside the file's lines.
    param([Parameter(Mandatory = $true)][string]$What, [Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Contains('%') -or $Value.Contains('"')) {
        throw "The $What path contains a % or a double quote ($Value), which a .cmd file cannot hold safely. Move it to a folder without those characters, then run this again. Nothing was changed."
    }
}

function Get-LocalbotCmdText {
    param([Parameter(Mandatory = $true)][string]$Python, [Parameter(Mandatory = $true)][string]$Repo)
    Assert-CmdSafe -What "Python" -Value $Python
    Assert-CmdSafe -What "repo" -Value $Repo
    return "@echo off`r`nsetlocal`r`nset `"PYTHONPATH=$Repo`"`r`n`"$Python`" -P -m localbot %*`r`n"
}

function Write-LocalbotCmd {
    param([Parameter(Mandatory = $true)][string]$BinDir, [Parameter(Mandatory = $true)][string]$Python,
          [Parameter(Mandatory = $true)][string]$Repo)
    $text = Get-LocalbotCmdText -Python $Python -Repo $Repo
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $file = Join-Path $BinDir "localbot.cmd"
    # cmd.exe reads the file in the console's code page: no BOM, no UTF-8 marker.
    Set-Content -LiteralPath $file -Value $text -Encoding OEM -NoNewline
    return $file
}

function Test-LocalbotImport {
    # True when $Python can import $Module with PYTHONPATH=$Repo, run from the temp folder with -P (so neither the
    # current folder nor anything outside the repo and Python's own packages is a way in). Nothing is installed.
    param([Parameter(Mandatory = $true)][string]$Python, [Parameter(Mandatory = $true)][string]$Repo,
          [string]$Module = "localbot.chat")
    $old = $env:PYTHONPATH
    Push-Location $env:TEMP
    try {
        $env:PYTHONPATH = $Repo
        & $Python -P -c "import $Module" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    catch { return $false }
    finally {
        if ($null -eq $old) { Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $old }
        Pop-Location
    }
}

function Add-LocalbotPathEntry {
    # Returns the path with $Folder added once. The rest is kept as it was (nothing expanded, nothing reordered);
    # when the folder is already there (any letter case, with or without a closing backslash) the path comes back as it was.
    param([string]$Current = "", [Parameter(Mandatory = $true)][string]$Folder)
    $want = $Folder.TrimEnd('\').ToLowerInvariant()
    $parts = @($Current -split ';' | Where-Object { $_ -ne '' })
    foreach ($p in $parts) {
        if ($p.TrimEnd('\').ToLowerInvariant() -eq $want) { return $Current }
    }
    return (($parts + $Folder) -join ';')
}

function Send-LocalbotEnvironmentChange {
    # The same message Windows' own Environment dialog sends, so new windows re-read the user's variables.
    try {
        if (-not ('LocalbotWin.Native' -as [type])) {
            Add-Type -Namespace LocalbotWin -Name Native -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Auto)]
public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);
"@
        }
        $result = [UIntPtr]::Zero
        # HWND_BROADCAST 0xFFFF, WM_SETTINGCHANGE 0x001A, SMTO_ABORTIFHUNG 0x2, 5 seconds
        [void][LocalbotWin.Native]::SendMessageTimeout([IntPtr]0xFFFF, 0x1A, [UIntPtr]::Zero, "Environment", 2, 5000, [ref]$result)
    }
    catch {
        Write-Warning "The path was saved, but Windows could not be told about it. Open a new PowerShell window; if the command is still not found, sign out and in again."
    }
}

function Set-LocalbotUserPath {
    # Adds $Folder once to the user's Path in the registry key $Key, as the RAW value: %NAME% parts stay as written
    # and the value keeps its type (a missing value is made expandable). Nothing is written when the folder is there.
    param([Parameter(Mandatory = $true)][string]$Folder, [string]$Key = 'HKCU:\Environment')
    $item = Get-Item -LiteralPath $Key
    $has = @($item.GetValueNames()) -contains 'Path'
    $current = ''
    $kind = 'ExpandString'
    if ($has) {
        $current = [string]$item.GetValue('Path', '', 'DoNotExpandEnvironmentNames')
        $found = $item.GetValueKind('Path').ToString()
        if ($found -eq 'String') { $kind = 'String' }
    }
    $new = Add-LocalbotPathEntry -Current $current -Folder $Folder
    if ($has -and $new -eq $current) { return }
    if ($has) {
        Set-ItemProperty -LiteralPath $Key -Name Path -Value $new -Type $kind
    }
    else {
        New-ItemProperty -LiteralPath $Key -Name Path -Value $new -PropertyType $kind | Out-Null
    }
    Send-LocalbotEnvironmentChange
}

function Install-LocalbotCommand {
    param([string]$Python = "", [string]$EnvKey = 'HKCU:\Environment')
    $repo = Split-Path -Parent $PSScriptRoot
    if (-not $Python) { $Python = Join-Path $repo ".venv\Scripts\python.exe" }
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "No Python found at $Python. The repo's .venv must exist (this script installs nothing); or pass one with -Python. Nothing was changed."
    }
    $bin = Join-Path $env:USERPROFILE ".localbot\bin"
    [void](Get-LocalbotCmdText -Python $Python -Repo $repo)    # refuses a path cmd.exe would misread, before anything is written
    if (-not (Test-LocalbotImport -Python $Python -Repo $repo)) {
        throw "$Python cannot start Local Bot from $repo (it needs Python 3.11 or later, and the packages Local Bot uses in that Python). Nothing was changed."
    }
    $file = Write-LocalbotCmd -BinDir $bin -Python $Python -Repo $repo
    Set-LocalbotUserPath -Folder $bin -Key $EnvKey
    Write-Host "Wrote $file"
    Write-Host "open a new PowerShell window, then type: localbot chat"
}

if ($MyInvocation.InvocationName -ne '.' -and -not $env:LOCALBOT_INSTALL_TEST) {
    try { Install-LocalbotCommand -Python $Python }
    catch { Write-Host $_.Exception.Message; exit 1 }
}
