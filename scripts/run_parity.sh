#!/usr/bin/env bash
# G3 parity driver (docs/STAGE1_SPEC.md §8, S6): Local Bot's runner with parity_lab6 + emulated
# shell on real disk, n=3, WORK 12 / TRUST 10 / privacy 8, one model at a time.
#
#   bash scripts/run_parity.sh OUT_DIR MODEL [MODEL...]
#
# Env: LOCALBOT_PY (python, default <repo>/.venv/Scripts/python.exe, else the main checkout's
# venv), LOCALBOT_WT (the checkout to run, default: this script's repo), PACKS (default
# "work trust privacy"), CAP_S (per-pack timeout, default 3300).
# Each pack: hard timeout; a failure gets ONE retry, a hang none; /api/ps logged and the model
# unloaded between models; nvidia-smi sampled every 5 s into OUT_DIR/nvsmi.csv. Never touches
# llama-server :8090 (only reads its /health). Results land in <checkout>/results/stage1/.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WT="${LOCALBOT_WT:-$HERE}"
PY="${LOCALBOT_PY:-}"
if [ -z "$PY" ]; then
  # The main checkout = the parent of git's common dir (a worktree has no venv of its own).
  MAIN="$(cd "$HERE" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
  MAIN="${MAIN%/*}"  # drop "/.git" (no dirname: a bash started outside Git Bash has no /usr/bin)
  for c in "$WT/.venv/Scripts/python.exe" "$HERE/.venv/Scripts/python.exe" ${MAIN:+"$MAIN/.venv/Scripts/python.exe"}; do
    [ -x "$c" ] && PY="$c" && break
  done
fi
OUT="$1"; shift
PACKS="${PACKS:-work trust privacy}"; CAP_S="${CAP_S:-3300}"
mkdir -p "$OUT"; LOG="$OUT/parity.log"; START=$(date +%s)
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ps_now() { curl -s --max-time 5 http://127.0.0.1:11434/api/ps; }
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -l 5 >> "$OUT/nvsmi.csv" 2>&1 &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT
cd "$WT" || exit 2
log "python: $("$PY" -c 'import sys;print(sys.version.split()[0])')  checkout: $WT  commit: $(git -C "$WT" rev-parse --short HEAD 2>/dev/null || cat "$WT/COMMIT" 2>/dev/null)  8090: $(curl -s --max-time 3 http://127.0.0.1:8090/health || echo down)  ps: $(ps_now)"
run_pack() {
  local label="$1"; shift
  for attempt in 1 2; do
    log "START $label attempt=$attempt: python -m localbot.exam.runner $*"
    local t0=$(date +%s)
    timeout --kill-after=30 "$CAP_S" "$PY" -m localbot.exam.runner "$@" > "$OUT/$label.a$attempt.out" 2>&1
    local rc=$? dt=$(( $(date +%s) - t0 ))
    log "END $label attempt=$attempt rc=$rc ${dt}s  $(grep -E '^wrote ' "$OUT/$label.a$attempt.out" | sed 's/^wrote .*stage1.//' | tr '\n' ' ')"
    [ $rc -eq 0 ] && return 0
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then log "HANG $label (timeout ${CAP_S}s): stopping, no retry"; return 2; fi
    log "FAIL $label rc=$rc"; tail -8 "$OUT/$label.a$attempt.out" | tee -a "$LOG"
  done
  log "FAIL twice: $label: stopping"; return 1
}
unload() { curl -s --max-time 30 -d "{\"model\":\"$1\",\"keep_alive\":0,\"prompt\":\"\"}" http://127.0.0.1:11434/api/generate > /dev/null; log "unload $1 -> ps: $(ps_now)"; }
i=0
for M in "$@"; do
  i=$((i+1))
  log "=== MODEL $M (elapsed $(( $(date +%s) - START ))s) ps: $(ps_now)  vram: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
  for P in $PACKS; do
    run_pack "${P}_m$i" --pack "$P" --profile parity_lab6 --n 3 "$M" || { unload "$M"; exit 3; }
  done
  unload "$M"
done
log "ALL DONE in $(( $(date +%s) - START ))s"
