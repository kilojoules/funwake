#!/usr/bin/env bash
# Heartbeat/supervisor for the detached Claude-150 run. Runs as a TRACKED task so
# its exit notifies the session. Idempotent via a lockfile: a second instance exits
# immediately if one is already live (prevents duplicate loop-relaunch races).
# Every 5 min: if the detached run_loop.sh is gone (and we're not done), relaunch it
# detached. After a ~35 min window it exits so the session starts a fresh supervisor.
set -u
cd /Users/julianquick/working/funwake
BASE=funwake2/state/explore_claude150
SUM=$BASE/summary.json
LOG=$BASE/run.log
LOCK=$BASE/hb.lock
TARGET=150

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[supervisor] already live (pid $(cat "$LOCK")) — exit"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

stat() { python3 - "$SUM" <<'PY' 2>/dev/null || echo "n=? best=?"
import json,sys
d=json.load(open(sys.argv[1]));t=d["trajectory"];b=d.get("best_feasible") or {}
print("n=%d best=%+.4f%%(it%s) feas=%d/%d"%(len(t),b.get("pct",0),b.get("iter","?"),
      sum(1 for x in t if x.get("feasible")),len(t)))
PY
}
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }

for tick in $(seq 1 7); do
  n=$(count)
  if [ "$n" -ge "$TARGET" ]; then echo "[supervisor] DONE $(stat)"; exit 0; fi
  if ! pgrep -f "run_loop.sh" >/dev/null 2>&1; then
    echo "[supervisor] $(date +%H:%M:%S) loop DEAD -> relaunch detached" >> "$LOG"
    nohup bash "$BASE/run_loop.sh" >> "$LOG" 2>&1 &
    disown 2>/dev/null || true
  fi
  echo "[supervisor] $(date +%H:%M:%S) tick=$tick $(stat)"
  sleep 300
done
echo "[supervisor] window elapsed $(stat) — relaunch me"
