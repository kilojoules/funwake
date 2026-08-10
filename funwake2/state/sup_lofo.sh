#!/usr/bin/env bash
# Persistent detached supervisor for the LOFO CV (launch ONCE, detached). Keeps the
# gbar worker alive (resubmits on expiry) and relaunches the lofo_run.sh meta-loop if
# it dies (resumable). Exits when all 5 fold result files exist.
set -u
cd /Users/julianquick/working/funwake
BASE=funwake2/state/lofo
LOG="$BASE/lofo.log"; SLOG="$BASE/persist.log"; LOCK="$BASE/persist.lock"
mkdir -p "$BASE"
GBAR="ssh -o BatchMode=yes -o ConnectTimeout=25 gbar"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[sup:lofo] already live"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
done_folds() { ls "$BASE"/*_heldout.json 2>/dev/null | wc -l | tr -d ' '; }
while :; do
  df=$(done_folds)
  if [ "$df" -ge 5 ]; then echo "[sup:lofo] $(date '+%m-%d %H:%M') ALL 5 FOLDS DONE" >> "$SLOG"; break; fi
  if ! $GBAR 'bash -lc "bjobs -J funwake_wk 2>/dev/null | grep -qE \"RUN|PEND\""' 2>/dev/null; then
    echo "[sup:lofo] $(date '+%m-%d %H:%M') worker down -> resubmit" >> "$SLOG"
    $GBAR 'bash -lc "cd ~/funwake && bsub < funwake2/gbar_worker.sh"' >> "$SLOG" 2>&1
  fi
  if ! pgrep -f "lofo_run.sh" >/dev/null 2>&1; then
    echo "[sup:lofo] $(date '+%m-%d %H:%M') meta-loop down -> relaunch (folds=$df)" >> "$SLOG"
    nohup bash funwake2/state/lofo_run.sh >> "$LOG" 2>&1 &
    disown 2>/dev/null || true
  fi
  echo "[sup:lofo] $(date '+%m-%d %H:%M') folds_done=$df/5" >> "$SLOG"
  sleep 300
done
