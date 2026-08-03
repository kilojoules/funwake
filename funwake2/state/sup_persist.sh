#!/usr/bin/env bash
# PERSISTENT self-perpetuating supervisor for the long Claude->1000 run.
# Unlike sup_both (a 7-tick tracked heartbeat that needs relaunching), this runs
# an INFINITE loop and is meant to be launched ONCE, DETACHED (nohup, reparents
# to init/PPID=1), so it survives the environment's task-reaper the same way the
# run_gen loops do — no babysitting. Every 5 min it ensures the detached
# run_gen:claude150 loop is alive (respawns it from checkpoint if a sweep kills
# it) and appends a status line. Exits only when the run reaches its target.
set -u
cd /Users/julianquick/working/funwake
TAG=claude150; ENG=claude; ITERS=1000
BASE="funwake2/state/explore_${TAG}"
SUM="$BASE/summary.json"; LOG="$BASE/run.log"; SLOG="$BASE/persist.log"; LOCK="$BASE/persist.lock"

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[persist] already live (pid $(cat "$LOCK"))"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }
stat() { python3 - "$SUM" "$ITERS" <<'PY' 2>/dev/null || echo "n=? best=?"
import json,sys
d=json.load(open(sys.argv[1]));t=d["trajectory"];b=d.get("best_feasible") or {}
print("n=%d/%s best=%+.4f%%(it%s) feas=%d/%d"%(len(t),sys.argv[2],b.get("pct",0),
      b.get("iter","?"),sum(1 for x in t if x.get("feasible")),len(t)))
PY
}

while :; do
  n=$(count)
  if [ "$n" -ge "$ITERS" ]; then echo "[persist] $(date '+%m-%d %H:%M:%S') DONE $(stat)" >> "$SLOG"; break; fi
  if ! pgrep -f "run_gen.sh $TAG " >/dev/null 2>&1; then
    echo "[persist] $(date '+%m-%d %H:%M:%S') loop DEAD -> relaunch detached" >> "$SLOG"
    nohup bash funwake2/state/run_gen.sh "$TAG" "$ENG" "$ITERS" >> "$LOG" 2>&1 &
    disown 2>/dev/null || true
  fi
  echo "[persist] $(date '+%m-%d %H:%M:%S') $(stat)" >> "$SLOG"
  sleep 300
done
