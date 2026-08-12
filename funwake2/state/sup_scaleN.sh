#!/usr/bin/env bash
# Persistent detached supervisor for the large-portfolio scaling run (tag scaleN).
# Keeps the 14-cell gbar worker (funwake_wkN) alive + the local scaling loop.
set -u
cd /Users/julianquick/working/funwake
BASE=funwake2/state/explore_scaleN
SUM="$BASE/summary.json"; LOG="$BASE/run.log"; SLOG="$BASE/persist.log"; LOCK="$BASE/persist.lock"
ITERS=80
GBAR="ssh -o BatchMode=yes -o ConnectTimeout=25 gbar"
mkdir -p "$BASE"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then echo "already live"; exit 0; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }
stat() { python3 - "$SUM" "$ITERS" <<'PY' 2>/dev/null || echo "n=0"
import json,sys
d=json.load(open(sys.argv[1]));t=d["trajectory"];b=d.get("best_feasible") or {}
print("n=%d/%s best=%+.4f%%(it%s worst%+.4f%%) allfeas=%d/%d"%(len(t),sys.argv[2],
      b.get("pct",0),b.get("iter","?"),b.get("worst",0) or 0,
      sum(1 for x in t if x.get("feasible")),len(t)))
PY
}
while :; do
  n=$(count)
  if [ "$n" -ge "$ITERS" ]; then echo "[sup:scaleN] $(date '+%m-%d %H:%M') DONE $(stat)" >> "$SLOG"; break; fi
  if ! $GBAR 'bash -lc "bjobs -J funwake_wkN 2>/dev/null | grep -qE \"RUN|PEND\""' 2>/dev/null; then
    echo "[sup:scaleN] $(date '+%m-%d %H:%M') big worker down -> resubmit" >> "$SLOG"
    $GBAR 'bash -lc "cd ~/funwake && bsub < funwake2/gbar_worker_bigN.sh"' >> "$SLOG" 2>&1
  fi
  if ! pgrep -f "run_scaleN_loop.sh" >/dev/null 2>&1; then
    echo "[sup:scaleN] $(date '+%m-%d %H:%M') loop down -> relaunch" >> "$SLOG"
    nohup bash funwake2/state/run_scaleN_loop.sh >> "$LOG" 2>&1 &
    disown 2>/dev/null || true
  fi
  echo "[sup:scaleN] $(date '+%m-%d %H:%M') $(stat)" >> "$SLOG"
  sleep 300
done
