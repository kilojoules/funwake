#!/usr/bin/env bash
# Generic tracked heartbeat/supervisor. Args: <tag> <engine> <iters>
# Idempotent per-tag via a lockfile; keeps the detached run_gen.sh:<tag> loop alive;
# prints n/best each tick; exits after ~35 min so the session relaunches it.
set -u
cd /Users/julianquick/working/funwake
TAG="$1"; ENGINE="$2"; ITERS="$3"
BASE="funwake2/state/explore_${TAG}"
SUM="$BASE/summary.json"; LOG="$BASE/run.log"; LOCK="$BASE/hb.lock"
mkdir -p "$BASE"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[sup:$TAG] already live (pid $(cat "$LOCK")) — exit"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

stat() { python3 - "$SUM" "$ITERS" <<'PY' 2>/dev/null || echo "n=? best=?"
import json,sys
d=json.load(open(sys.argv[1]));t=d["trajectory"];b=d.get("best_feasible") or {}
print("n=%d/%s best=%+.4f%%(it%s) feas=%d/%d"%(len(t),sys.argv[2],b.get("pct",0),
      b.get("iter","?"),sum(1 for x in t if x.get("feasible")),len(t)))
PY
}
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }

for tick in $(seq 1 7); do
  n=$(count)
  if [ "$n" -ge "$ITERS" ]; then echo "[sup:$TAG] DONE $(stat)"; exit 0; fi
  if ! pgrep -f "run_gen.sh $TAG " >/dev/null 2>&1; then
    echo "[sup:$TAG] $(date +%H:%M:%S) loop DEAD -> relaunch detached" >> "$LOG"
    nohup bash funwake2/state/run_gen.sh "$TAG" "$ENGINE" "$ITERS" >> "$LOG" 2>&1 &
    disown 2>/dev/null || true
  fi
  echo "[sup:$TAG] $(date +%H:%M:%S) tick=$tick $(stat)"
  sleep 300
done
echo "[sup:$TAG] window elapsed $(stat) — relaunch me"
