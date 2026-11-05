<#
.SYNOPSIS
  Stage 1 GPU batch (docs/STAGE1_SPEC.md section 8): G5 red team, G6 roster smoke, nemotron G3N/G4N,
  the full-model gates G3/G4 (qwen-ft + Ornith), and W2S (weak to strong, exam + product profiles).
  -Models overrides a gate's model list.

.DESCRIPTION
  The owner runs this after the merge, from the checkout that holds the merged code
  (Windows PowerShell 5.1):

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stage1_gpu_batch.ps1 -Commit <reviewed sha>

  Add -DryRun first to see the preflight and the plan without loading any model. -ListOnly prints
  the plan alone: no run lock, no preflight, no model tool (ollama, nvidia-smi, podman, docker), no
  file written; it only reads (git rev-parse to find the main checkout's venv, the venv python to
  read a done run's commit).

  W2S (BRIEF_H1J J1) is 24 pack runs over two evenings, so it resumes: a W2S-only batch gets the
  fixed tag s1gpu_w2s (an INJ-only batch: s1gpu_inj, same rules), and running the same command again
  skips every (model, pack, profile) whose
  W2S row in this batch's summary is "ok" and whose result file is still there (FAILED rows run
  again). Resume is per gate, holds for any gate when -Tag names an existing batch, and never crosses
  commits: a done run whose result file was made at another commit than -Commit stops the batch.
  docs/runs.json takes the W2S files from this batch's summary (gate W2S, status ok); after the
  evenings they are frozen into that entry's "files" list, in the same commit as the result files.

  Run lock (audit E-3): after the parameter checks the batch takes the one machine-wide run lock
  (scripts\runlock.ps1, the same lock watch_run.py, g2_run.py and runlock.py take) and holds it for
  the whole batch; busy = "STOP: busy: <holder>", exit 3. Children get LOCALBOT_RUNLOCK_HELD=1, so a
  runner or watch_run started by the batch does not fight it. -DryRun checks the lock, does not hold it.

  Preflight (any failure stops the batch before a model is loaded):
    git rev-parse HEAD must start with -Commit (7 to 40 hex chars); podman info and docker info
    exit 0; nvidia-smi exits 0; ollama list exits 0 and is saved to state\runs\<Tag>_ollama_list.txt;
    ollama ps exits 0 and shows no model loaded (else: ollama stop it first); the python in -Python
    exists. Recorded only: git status --short, :8090 listening or not.

  Runs, in gate order (one model per process, one Tee-Object log per run in state\runs\):
    G5   abliterated model (ID 9ead0fce8d3f), packs privacy then trust, profile core4, n=3
    G6   each roster model (docs/G6_ROSTER.md, rows -G6First..-G6Last), pack privacy, core4, n=3
    G3N  nemotron-3-nano:4b, profile parity_lab6, packs work, trust, privacy, n=3
    G4N  nemotron-3-nano:4b, profile core4, packs work, trust, privacy, n=3
    G3   qwen-ft + Ornith, profile parity_lab6, packs work, trust, privacy, n=3 (selectable, not in default)
    G4   qwen-ft + Ornith, profile core4, packs work, trust, privacy, n=3 (selectable, not in default)
    W2S  llama3.2:1b, nemotron-3-nano:4b, qwen-ft, Ornith, the abliterated model (red team),
         gpt-oss:20b (spill lane: timing only); profiles core4 then core4_product, packs privacy then
         trust, n=3: 24 pack runs, model by model (selectable, not in default). Same ID pins as above.
    INJ  the live injection pack (BRIEF_H1K K2): llama3.2:1b, qwen-ft, Ornith (weak, mid, strong),
         pack injection, profile core4, n=3: 3 pack runs (selectable, not in default).
  -Models "<tag,tag>" replaces the model list of every gate in the run (e.g. G6 on two roster models).
  G3/G4 models are pinned by tag (the only ID the result files carry): a model present under the tag
  runs, with no digest pin (G5/G6/G3N/G4N keep their exact digest pin). An override tag not otherwise
  known runs the same way (present under the tag is enough).
  Runner flags are the ones localbot/exam/runner.py main() takes:
    python -m localbot.exam.runner --pack work|trust|privacy|all --profile parity_lab6|core4|core4_product --n 3 MODEL
  Pack order work, trust, privacy is the runner's PACKS order (what --pack all runs; G4 run 3's log
  shows that order). G5 runs privacy then trust, as the owner's prompt asked.
  Per-run logs: <Tag>_<gate>_<pack>_<model>.log, with _<profile> after the gate for a gate with two
  profiles (W2S); a log name already on disk gets _2, _3, ..., so a failed attempt's log is kept.

  A model whose ID in ollama list is not the one in docs/G6_ROSTER.md is not run (recorded).
  A run that exits non-zero or prints no "wrote ..." line is recorded as FAILED, its model is
  unloaded with "ollama stop", and the batch goes on. Summary: state\runs\<Tag>_summary.tsv.
  Exit code: 0 all ran ok, 1 something failed or was skipped, 2 preflight stopped the batch,
  3 the run lock is busy (a suite, G2 or another batch is running).

.PARAMETER Commit
  The reviewed commit. HEAD must start with it.
.PARAMETER Gates
  Comma-separated subset of G5,G6,G3N,G4N,G3,G4,W2S,INJ (default: G5,G6,G3N,G4N, in that order).
.PARAMETER Models
  Comma-separated model tags that override the model list of every gate in the run (default: the
  gate's own models). Example: -Gates G6 -Models "llama3.2:1b,llama3.1:8b-instruct-q8_0".
.PARAMETER G6First
  First G6 roster row to run (1-20, default 1).
.PARAMETER G6Last
  Last G6 roster row to run (1-20, default 20).
.PARAMETER Python
  The pinned venv python (default: .venv\Scripts\python.exe of this checkout if it has one, else of
  the main checkout, found through git's common dir).
.PARAMETER Tag
  Prefix of every file this batch writes in state\runs\ (default s1gpu_<local time>; s1gpu_w2s for a
  W2S-only batch, s1gpu_inj for an INJ-only one). $env:LOCALBOT_RUNS_DIR replaces state\runs\ (tests).
.PARAMETER DryRun
  Preflight + check the run lock (busy = exit 3; not held) + print the plan. No model is loaded.
.PARAMETER ListOnly
  Print the plan (with what a resume would skip) and stop: no lock, no tool, no file written.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Commit,
    [string]$Gates = "G5,G6,G3N,G4N",
    [string]$Models = "",
    [int]$G6First = 1,
    [int]$G6Last = 20,
    [string]$Python = "",
    [string]$Tag = "",
    [switch]$DryRun,
    [switch]$ListOnly
)

# Native tools write to stderr; in Windows PowerShell 5.1 "Stop" would turn that into a crash.
$ErrorActionPreference = "Continue"

$Abliterated = @{ Tag = "hf.co/OBLITERATUS/Gemma-4-12B-OBLITERATED:Q4_K_M"; Id = "9ead0fce8d3f" }
$Nemotron = @{ Tag = "nemotron-3-nano:4b"; Id = "6cc467f05439" }
# G3/G4 full-model gates. The result files carry only the tag, no ollama digest, so these are
# pinned by tag (Id = "" = no digest pin: a model present under the tag runs).
$QwenFt = @{ Tag = "qwen3-8b-ft-agentic-v1:latest"; Id = "" }
$Ornith = @{ Tag = "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest"; Id = "" }
# docs/G6_ROSTER.md, same rows and order (smallest weights first). Keep the two in step.
$Roster = @(
    @{ Tag = "llama32-1b-localharness-poc:latest"; Id = "0f8ca00cd2d4" },                    # 1
    @{ Tag = "llama3.2:1b"; Id = "baf6a787fdff" },                                           # 2
    @{ Tag = "granite4:tiny-h"; Id = "566b725534ea" },                                       # 3
    @{ Tag = "hermes3:8b"; Id = "4f6b83f30b62" },                                            # 4
    @{ Tag = "hermes3-8b-mvp64k:latest"; Id = "4330032e0976" },                              # 5
    @{ Tag = "llama3.1:8b"; Id = "46e0c10c039e" },                                           # 6
    @{ Tag = "qwen3-8b-finetuned-test:latest"; Id = "67c203ac8cc9" },                        # 7
    @{ Tag = "qwen3-8b-gold-v2:latest"; Id = "8844d74b9f2f" },                               # 8
    @{ Tag = "lfm2.5:8b"; Id = "9cf756159fc2" },                                             # 9
    @{ Tag = "qwen3:8b"; Id = "500a1f067a9f" },                                              # 10
    @{ Tag = "qwen3-vl:8b"; Id = "901cae732162" },                                           # 11
    @{ Tag = "gemma4:12b"; Id = "6114515d63c1" },                                            # 12
    @{ Tag = "JetBrains/mellum2-thinking-q4_k_m:latest"; Id = "36a1b2f1712f" },              # 13
    @{ Tag = "llama3.1:8b-instruct-q8_0"; Id = "b158ded76fa0" },                             # 14 tight
    @{ Tag = "cogito:14b"; Id = "d0cac86a2347" },                                            # 15 tight
    @{ Tag = "qwen2.5-mvp64k:latest"; Id = "35f995f3957b" },                                 # 16 tight
    @{ Tag = "hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M"; Id = "ce5cb56a7898" }, # 17 tight
    @{ Tag = "gemma4:e4b"; Id = "c6eb396dbd59" },                                            # 18 tight
    @{ Tag = "ServiceNow-AI/Apriel-1.6-15b-Thinker:Q4_K_M"; Id = "13e8c240b206" },           # 19 tight
    @{ Tag = "gpt-oss:20b"; Id = "17052f91a42e" }                                            # 20 spill lane: timing only
)

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
# The gates come first: a W2S-only or INJ-only batch gets a fixed tag, so the same command resumes it.
$gateList = @($Gates -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { $_.ToUpper() })
$FixedTags = @{ "W2S" = "s1gpu_w2s"; "INJ" = "s1gpu_inj" }  # fixed tags so the same command resumes a W2S-only or INJ-only batch; docs/runs.json records that batch's frozen files
if (-not $Tag) {
    $Tag = if ($gateList.Count -eq 1 -and $FixedTags.ContainsKey($gateList[0])) { $FixedTags[$gateList[0]] }
           else { "s1gpu_" + (Get-Date -Format "yyyyMMddTHHmmss") }
}
$Runs = if ($env:LOCALBOT_RUNS_DIR) { $env:LOCALBOT_RUNS_DIR } else { Join-Path $Repo "state\runs" }
if (-not $ListOnly) { New-Item -ItemType Directory -Force -Path $Runs | Out-Null }
$BatchLog = Join-Path $Runs "${Tag}_batch.log"
$PreLog = Join-Path $Runs "${Tag}_preflight.log"
$ListFile = Join-Path $Runs "${Tag}_ollama_list.txt"
if (Test-Path -LiteralPath $ListFile) {  # a resumed batch keeps the first evening's list too
    $ListFile = Join-Path $Runs ("{0}_ollama_list_{1}.txt" -f $Tag, (Get-Date -Format "yyyyMMddTHHmmss"))
}
$Summary = Join-Path $Runs "${Tag}_summary.tsv"

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8  # read python's UTF-8 output as UTF-8

function Note([string]$Line) {
    $s = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Line
    Write-Host $s
    if (-not $ListOnly) { Add-Content -Path $BatchLog -Value $s -Encoding UTF8 }
}

# Run a native tool, return its exit code and its output lines (stderr merged).
function Invoke-Native([string]$Exe, [string[]]$ArgList) {
    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) {
        return [pscustomobject]@{ Rc = 127; Lines = @("not found: $Exe") }
    }
    $global:LASTEXITCODE = 0
    $lines = @(& $Exe @ArgList 2>&1 | ForEach-Object { "$_" })
    return [pscustomobject]@{ Rc = $LASTEXITCODE; Lines = $lines }
}

# One preflight check: output goes to the preflight log; a required check that fails stops here.
function Test-Pre([string]$Name, [string]$Exe, [string[]]$ArgList, [bool]$Required, [string]$SaveTo) {
    $r = Invoke-Native $Exe $ArgList
    Add-Content -Path $PreLog -Value "===== $Name (exit $($r.Rc))" -Encoding UTF8
    if ($r.Lines.Count -gt 0) { Add-Content -Path $PreLog -Value $r.Lines -Encoding UTF8 }
    if ($SaveTo) { $r.Lines | Out-File -FilePath $SaveTo -Encoding utf8 }
    Note ("preflight {0}: exit {1}" -f $Name, $r.Rc)
    if ($Required -and $r.Rc -ne 0) {
        Note "STOP: $Name failed (exit $($r.Rc)); details in $PreLog"
        exit 2
    }
    return $r
}

function Get-Slug([string]$Model) { return ($Model -replace '[^A-Za-z0-9._-]', '_') }

function Add-Summary([string]$Gate, [string]$Pack, [string]$Profile, [string]$Model, [string]$Status,
                     [string]$Rc, [string]$Secs, [string]$Result, [string]$Log) {
    $row = @((Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Gate, $Pack, $Profile, $Model, $Status, $Rc, $Secs,
             $Result, $Log) -join "`t"
    Add-Content -Path $Summary -Value $row -Encoding UTF8
}

. (Join-Path $PSScriptRoot "runlock.ps1")  # Enter-RunLock / Exit-RunLock / Get-RunLockHolder / Get-MainCheckout

# No fixed personal path: this checkout's venv, else the main checkout's (worktrees have none).
if (-not $Python) {
    $Python = Join-Path $Repo ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) { $Python = Join-Path (Get-MainCheckout) ".venv\Scripts\python.exe" }
}

# --- parameters ------------------------------------------------------------------------------

Note "batch $Tag in $Repo"
foreach ($g in $gateList) {
    if (@("G5", "G6", "G3N", "G4N", "G3", "G4", "W2S", "INJ") -notcontains $g) {
        Note "STOP: unknown gate '$g' (use G5,G6,G3N,G4N,G3,G4,W2S,INJ)"; exit 2
    }
}
if ($G6First -lt 1 -or $G6Last -gt $Roster.Count -or $G6First -gt $G6Last) {
    Note "STOP: -G6First/-G6Last must satisfy 1 <= first <= last <= $($Roster.Count)"; exit 2
}
if ($Commit -notmatch '^[0-9a-fA-F]{7,40}$') { Note "STOP: -Commit must be 7 to 40 hex characters"; exit 2 }

# --- plan ----------------------------------------------------------------------------------
# Built before the lock and the preflight, so -ListOnly can print it without touching anything.

$plan = New-Object System.Collections.ArrayList
function Add-Plan([string]$Gate, [hashtable]$M, [string[]]$Packs, [string[]]$Profiles) {
    foreach ($prof in $Profiles) {
        foreach ($p in $Packs) {
            [void]$plan.Add([pscustomobject]@{ Gate = $Gate; Model = $M.Tag; Id = $M.Id; Pack = $p; Profile = $prof
                                               Log = "" })
        }
    }
}
# Each gate's profiles and packs; the model list comes from the switch below (or -Models override).
$GateProfiles = @{ "G5" = @("core4"); "G6" = @("core4"); "G3N" = @("parity_lab6"); "G4N" = @("core4")
                   "G3" = @("parity_lab6"); "G4" = @("core4"); "W2S" = @("core4", "core4_product")
                   "INJ" = @("core4") }
$GatePacks = @{
    "G5" = @("privacy", "trust"); "G6" = @("privacy")
    "G3N" = @("work", "trust", "privacy"); "G4N" = @("work", "trust", "privacy")
    "G3" = @("work", "trust", "privacy"); "G4" = @("work", "trust", "privacy")
    "W2S" = @("privacy", "trust"); "INJ" = @("injection")
}
# W2S, weak to strong (BRIEF_H1J J1); gpt-oss:20b is the spill lane: its scores are timing only.
$W2SModels = @("llama3.2:1b", $Nemotron.Tag, $QwenFt.Tag, $Ornith.Tag, $Abliterated.Tag, "gpt-oss:20b")
# INJ, the live injection pack (BRIEF_H1K K2): weak, mid, strong.
$INJModels = @("llama3.2:1b", $QwenFt.Tag, $Ornith.Tag)
$KnownIds = @{}
foreach ($m in (@($Abliterated, $Nemotron, $QwenFt, $Ornith) + $Roster)) { $KnownIds[$m.Tag] = $m.Id }
$modelOverride = @($Models -split '[,\s]+' | Where-Object { $_ })

function Resolve-Model([string]$Tag) {
    $id = ""
    if ($KnownIds.ContainsKey($Tag)) { $id = $KnownIds[$Tag] }
    return @{ Tag = $Tag; Id = $id }
}
# Build each gate's model list into an ArrayList (every entry stays one hashtable, no array
# unwrapping). -Models, when given, replaces the model list of every gate in the run.
# NB: the local list is $gateModels, not $models — PowerShell variable names are case-insensitive,
# so $models would alias the [string]$Models parameter and be coerced to a string.
foreach ($g in $gateList) {
    $gateModels = New-Object System.Collections.ArrayList
    if ($modelOverride.Count -gt 0) {
        foreach ($ovTag in $modelOverride) { [void]$gateModels.Add((Resolve-Model $ovTag)) }  # not $tag: aliases $Tag
    } else {
        switch ($g) {
            "G5" { [void]$gateModels.Add($Abliterated) }
            "G6" { for ($i = $G6First; $i -le $G6Last; $i++) { [void]$gateModels.Add($Roster[$i - 1]) } }
            "G3N" { [void]$gateModels.Add($Nemotron) }
            "G4N" { [void]$gateModels.Add($Nemotron) }
            "G3" { [void]$gateModels.Add($QwenFt); [void]$gateModels.Add($Ornith) }
            "G4" { [void]$gateModels.Add($QwenFt); [void]$gateModels.Add($Ornith) }
            "W2S" { foreach ($w in $W2SModels) { [void]$gateModels.Add((Resolve-Model $w)) } }
            "INJ" { foreach ($w in $INJModels) { [void]$gateModels.Add((Resolve-Model $w)) } }
        }
    }
    foreach ($m in $gateModels) { Add-Plan $g $m $GatePacks[$g] $GateProfiles[$g] }
}

# One log per run, never an old one: a name already on disk (a failed attempt's evidence) or already
# planned gets _2, _3, ... A gate with two profiles has the profile in the name.
$usedLogs = @{}
function Get-FreeLog([string]$Base) {
    $name = "$Base.log"
    $i = 1
    while ((Test-Path -LiteralPath (Join-Path $Runs $name)) -or $usedLogs.ContainsKey($name)) {
        $i++
        $name = "${Base}_$i.log"
    }
    $usedLogs[$name] = $true
    return (Join-Path $Runs $name)
}
foreach ($r in $plan) {
    $profPart = if ($GateProfiles[$r.Gate].Count -gt 1) { "_" + $r.Profile } else { "" }
    $r.Log = Get-FreeLog ("{0}_{1}{2}_{3}_{4}" -f $Tag, $r.Gate.ToLower(), $profPart, $r.Pack, (Get-Slug $r.Model))
}

# Resume, per gate: a (model, pack, profile) this batch's summary marks "ok" for this gate, with its
# result file still on disk, is done; a FAILED or SKIPPED row, an ok row whose file is gone, or
# another gate's row runs again (docs/s10_recompute.py counts a run's rows by its gate). The result
# column is the runner's "wrote <path>  <pack> ..." line; the path may hold spaces.
$WroteRe = '^wrote (.+?\.json)(?:\s{2}|\s*$)'
$done = @{}
if (Test-Path -LiteralPath $Summary) {
    foreach ($line in @(Get-Content -LiteralPath $Summary -Encoding UTF8 | Select-Object -Skip 1)) {
        $c = @("$line" -split "`t")
        if ($c.Count -lt 10 -or $c[5] -ne "ok") { continue }
        if ($c[8] -match $WroteRe -and (Test-Path -LiteralPath $Matches[1])) {
            $done["$($c[1].ToUpper())`t$($c[4])`t$($c[2])`t$($c[3])"] = $Matches[1]
        }
    }
}
function Get-DoneKey($Row) { return "$($Row.Gate)`t$($Row.Model)`t$($Row.Pack)`t$($Row.Profile)" }

# A batch never resumes across commits: every done run of the planned gates must carry a
# run_env.commit that starts with -Commit, or evening 2 would mix code in one table. Read with the
# venv python (Windows PowerShell's ConvertFrom-Json refuses big files); an unreadable or missing
# commit counts as another commit.
$planKeys = @{}
foreach ($r in $plan) { $planKeys[(Get-DoneKey $r)] = $true }
$doneFiles = @($done.Keys | Where-Object { $planKeys.ContainsKey($_) } | ForEach-Object { $done[$_] } | Sort-Object -Unique)
if ($doneFiles.Count -gt 0) {
    $readCommit = "import json, sys`nfor p in sys.argv[1:]:`n    try:`n        c = json.load(open(p, encoding='utf-8')).get('run_env', {}).get('commit') or ''`n    except (OSError, ValueError, AttributeError):`n        c = ''`n    print((c or 'unknown') + '\t' + p)"
    $global:LASTEXITCODE = 0
    $seenCommits = @(& $Python -I -c $readCommit @doneFiles 2>&1 | ForEach-Object { "$_" })
    $other = @($seenCommits | Where-Object { -not ("$_".ToLower().StartsWith($Commit.ToLower())) })
    if ($LASTEXITCODE -ne 0 -or $seenCommits.Count -ne $doneFiles.Count -or $other.Count -gt 0) {
        Note ("STOP: this batch ({0}) already has runs from another commit (or an unknown one) than -Commit {1}:" -f
              $Tag, $Commit)
        foreach ($o in @($other + $(if ($LASTEXITCODE -ne 0) { $seenCommits } else { @() }))) { Note "  $o" }
        Note "Finish it at that commit, or start a new batch with -Tag <new tag> (and its own docs/runs.json entry)."
        exit 2
    }
}
function Get-CmdText($Row) {
    # a real PowerShell command a person can paste: '...' never expands $ or `, and a ' in the path is doubled
    $q = "'" + $Python.Replace("'", "''") + "'"
    return "& $q -m localbot.exam.runner --pack $($Row.Pack) --profile $($Row.Profile) --n 3 $($Row.Model)"
}
$doneBefore = @($plan | Where-Object { $done.ContainsKey((Get-DoneKey $_)) }).Count
$rows = if ($modelOverride.Count -gt 0) { ", models override ($($modelOverride.Count))" }
        elseif ($gateList -contains "G6") { ", G6 rows $G6First-$G6Last" } else { "" }
Note ("plan: {0} runs, gates {1}{2}; {3} done before in this batch (resume)" -f $plan.Count, ($gateList -join ","),
      $rows, $doneBefore)

if ($ListOnly) {
    foreach ($r in $plan) {
        $k = Get-DoneKey $r
        if ($done.ContainsKey($k)) { Note "done before [$($r.Gate)] $($r.Pack) $($r.Profile) $($r.Model): $($done[$k])" }
        else { Note "would run [$($r.Gate)] $(Get-CmdText $r) -> $($r.Log)" }
    }
    Note "list only: no run lock taken, no tool started, no file written"
    exit 0
}

# --- run lock (audit E-3): one long job at a time on this PC, from any worktree --------------
# Taken before anything can load a model and held for the whole batch, so no suite, G2 or other
# batch can run beside it. Windows drops it if this process dies.
$RunLock = Enter-RunLock -Name "gpu_batch" -Cmd ("{0} (tag {1})" -f [Environment]::CommandLine, $Tag)
if (-not $RunLock) {
    $holderText = Get-RunLockHolder
    if (-not $holderText) { $holderText = "(holder unknown)" }
    Note "STOP: busy: $holderText"
    exit 3
}
$heldBefore = $env:LOCALBOT_RUNLOCK_HELD
if ($DryRun) {
    Exit-RunLock $RunLock  # a dry run only checks the lock; it does not hold it while printing the plan
    $RunLock = $null
    Note "run lock: free (dry run: checked, not held)"
} else {
    $env:LOCALBOT_RUNLOCK_HELD = "1"  # children (runner, watch_run) see it held by this batch and do not fight it
    Note ("run lock: held by this batch (pid {0}) until it ends: {1}" -f $PID, $RunLock.Name)
}
# The rest of the batch runs inside this try (not indented, to keep the diff small); the finally at
# the end lets the lock go on every exit, including a preflight exit 2 and Ctrl+C.
try {

$head = Test-Pre "git rev-parse HEAD" "git" @("rev-parse", "HEAD") $true $null
$headSha = ($head.Lines | Select-Object -First 1).Trim().ToLower()
if (-not $headSha.StartsWith($Commit.ToLower())) {
    Note "STOP: HEAD is $headSha, not the reviewed commit $Commit"
    exit 2
}
Note "HEAD $headSha matches -Commit $Commit"
[void](Test-Pre "git status --short" "git" @("status", "--short") $false $null)
[void](Test-Pre "podman info" "podman" @("info") $true $null)
[void](Test-Pre "docker info" "docker" @("info") $true $null)
[void](Test-Pre "nvidia-smi" "nvidia-smi" @() $true $null)
$list = Test-Pre "ollama list" "ollama" @("list") $true $ListFile
$psNow = Test-Pre "ollama ps" "ollama" @("ps") $true $null
$loadedNow = @($psNow.Lines | Select-Object -Skip 1 | Where-Object { "$_".Trim() } |
               ForEach-Object { ("$_".Trim() -split '\s+')[0] })
if ($loadedNow.Count -gt 0) {
    Note ("STOP: ollama ps shows a model loaded: {0}. Unload it first (ollama stop {1}), then start the batch again" -f
          ($loadedNow -join ", "), ($loadedNow -join "; ollama stop "))
    exit 2
}
Note "ollama ps: no model loaded"
[void](Test-Pre "python --version" $Python @("--version") $true $null)
$listening8090 = $null -ne (Get-NetTCPConnection -LocalPort 8090 -State Listen -ErrorAction SilentlyContinue)
Note ("llama-server :8090 listening: {0} (not touched; if true, expect more spill)" -f $listening8090)

$ids = @{}
foreach ($line in ($list.Lines | Select-Object -Skip 1)) {
    $parts = @($line -split '\s+' | Where-Object { $_ })
    if ($parts.Count -ge 2) { $ids[$parts[0]] = $parts[1] }
}

if (-not $DryRun -and -not (Test-Path $Summary)) {
    Add-Content -Path $Summary -Value (@("time", "gate", "pack", "profile", "model", "status", "exit", "seconds",
                                         "result", "log") -join "`t") -Encoding UTF8
}

$failed = 0
foreach ($r in $plan) {
    $log = $r.Log
    $key = Get-DoneKey $r
    if ($done.ContainsKey($key)) {  # resume
        Note "done before [$($r.Gate)] $($r.Pack) $($r.Profile) $($r.Model): $($done[$key])"
        continue
    }
    $seen = $ids[$r.Model]
    $idOk = if ($r.Id) { $seen -eq $r.Id } else { [bool]$seen }  # no pinned digest -> present is enough
    if (-not $idOk) {
        $why = if ($seen) { "ID is $seen, roster says $($r.Id)" } else { "not in ollama list" }
        Note "SKIP $($r.Gate) $($r.Pack) $($r.Model): $why"
        if (-not $DryRun) { Add-Summary $r.Gate $r.Pack $r.Profile $r.Model "SKIPPED" "" "" $why "" }
        $failed++
        continue
    }
    $cmdText = Get-CmdText $r
    if ($DryRun) { Note "would run [$($r.Gate)] $cmdText -> $log"; continue }

    Note "START [$($r.Gate)] $cmdText"
    $t0 = Get-Date
    $global:LASTEXITCODE = 0
    & $Python -m localbot.exam.runner --pack $r.Pack --profile $r.Profile --n 3 $r.Model 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $log
    $rc = $LASTEXITCODE
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    $wrote = Select-String -Path $log -Pattern '^wrote ' | Select-Object -Last 1
    $result = if ($wrote) { $wrote.Line.Trim() } else { "no 'wrote' line" }
    if ($rc -eq 0 -and $wrote) {
        $status = "ok"
        if ($result -match $WroteRe) { $done[$key] = $Matches[1] }
    } else {
        $status = "FAILED"
        $failed++
        [void](Invoke-Native "ollama" @("stop", $r.Model))  # unload only; the server keeps running
    }
    Add-Summary $r.Gate $r.Pack $r.Profile $r.Model $status "$rc" "$secs" $result $log
    Note ("END [{0}] {1} {2} {3}: {4} exit {5} in {6}s :: {7}" -f $r.Gate, $r.Pack, $r.Profile, $r.Model, $status, $rc,
          $secs, $result)
    $ps = Invoke-Native "ollama" @("ps")
    $loaded = @($ps.Lines | Select-Object -Skip 1)
    Note ("ollama ps after run: {0}" -f ($loaded -join " | "))
    if ($loaded | Where-Object { ($_ -split '\s+')[0] -eq $r.Model }) {
        Note "still loaded after the run: ollama stop $($r.Model)"
        [void](Invoke-Native "ollama" @("stop", $r.Model))
    }
}

if ($DryRun) { Note "dry run: no model was loaded; $failed planned run(s) would be skipped"; exit 0 }
Note "done: $($plan.Count) planned, $failed failed or skipped; summary $Summary"
if ($failed -gt 0) { exit 1 }
exit 0
} finally {
    if ($RunLock) { Exit-RunLock $RunLock }
    $env:LOCALBOT_RUNLOCK_HELD = $heldBefore
}
