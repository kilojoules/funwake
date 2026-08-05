#!/usr/bin/env bash
# Persistent detached supervisor for the portfolio run (launch ONCE, detached).
# Keeps run_port_loop.sh alive across sweeps; logs n/best to persist.log.
set -u
cd /Users/julianquick/working/funwake
BASE=funwake2/state/explore_port_claude
SUM="$BASE/summary.json"; LOG="$BASE/run.log"; SLOG="$BASE/persist.log"; LOCK="$BASE/persist.lock"
ITERS=200
mkdir -p "$BASE"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[sup:port] already live (pid $(cat "$LOCK"))"; exit 0
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
  if [ "$n" -ge "$ITERS" ]; then echo "[sup:port] $(date '+%m-%d %H:%M:%S') DONE $(stat)" >> "$SLOG"; break; fi
  if ! pgrep -f "run_port_loop.sh" >/dev/null 2>&1; then
    echo "[sup:port] $(date '+%m-%d %H:%M:%S') loop DEAD -> relaunch detached" >> "$SLOG"
    nohup bash funwake2/state/run_port_loop.sh >> "$LOG" 2>&1 &
    disown 2>/dev/null || true
  fi
  echo "[sup:port] $(date '+%m-%d %H:%M:%S') $(stat)" >> "$SLOG"
  sleep 300
done
