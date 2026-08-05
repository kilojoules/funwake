#!/usr/bin/env bash
# Persistent detached supervisor for the gbar-backed portfolio run (launch ONCE,
# detached). Keeps BOTH alive: (1) the local run_port_loop.sh mutation/dispatch
# loop, and (2) the gbar bsub worker (resubmits it when its wall-time expires).
set -u
cd /Users/julianquick/working/funwake
TAG=port_gbar
BASE="funwake2/state/explore_${TAG}"
SUM="$BASE/summary.json"; LOG="$BASE/run.log"; SLOG="$BASE/persist.log"; LOCK="$BASE/persist.lock"
ITERS=200
GBAR="ssh -o BatchMode=yes -o ConnectTimeout=20 gbar"
mkdir -p "$BASE"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[sup:port_gbar] already live (pid $(cat "$LOCK"))"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }
stat() { python3 - "$SUM" "$ITERS" <<'PY' 2>/dev/null || echo "n=0 (no data yet)"
import json,sys
d=json.load(open(sys.argv[1]));t=d["trajectory"];b=d.get("best_feasible") or {}
print("n=%d/%s best=%+.4f%%(it%s worst%+.4f%%) allfeas=%d/%d"%(len(t),sys.argv[2],
      b.get("pct",0),b.get("iter","?"),b.get("worst",0) or 0,
      sum(1 for x in t if x.get("feasible")),len(t)))
PY
}
while :; do
  n=$(count)
  if [ "$n" -ge "$ITERS" ]; then echo "[sup:port_gbar] $(date '+%m-%d %H:%M:%S') DONE $(stat)" >> "$SLOG"; break; fi
  # (1) keep the gbar worker alive (resubmit if no RUN/PEND funwake_wk job)
  if ! $GBAR 'bjobs -J funwake_wk 2>/dev/null | grep -qE "RUN|PEND"' 2>/dev/null; then
    echo "[sup:port_gbar] $(date '+%m-%d %H:%M:%S') gbar worker down -> resubmit" >> "$SLOG"
    $GBAR 'cd ~/funwake && bsub < funwake2/gbar_worker.sh' >> "$SLOG" 2>&1
  fi
  # (2) keep the local mutation/dispatch loop alive
  if ! pgrep -f "run_port_loop.sh" >/dev/null 2>&1; then
    echo "[sup:port_gbar] $(date '+%m-%d %H:%M:%S') loop DEAD -> relaunch detached" >> "$SLOG"
    nohup bash funwake2/state/run_port_loop.sh >> "$LOG" 2>&1 &
    disown 2>/dev/null || true
  fi
  echo "[sup:port_gbar] $(date '+%m-%d %H:%M:%S') $(stat)" >> "$SLOG"
  sleep 300
done
