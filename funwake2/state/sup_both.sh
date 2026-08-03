#!/usr/bin/env bash
# Combined tracked heartbeat for two concurrent explore runs (Claude->1000 and
# Antigravity->150). Idempotent via a lockfile; keeps each detached run_gen.sh
# loop alive; prints both runs' n/best each tick; exits after ~35 min so the
# session relaunches it. Each run is process-isolated by its tag in pgrep.
set -u
cd /Users/julianquick/working/funwake
LOCK=funwake2/state/hb_both.lock
# tag:engine:iters
RUNS=("claude150:claude:1000" "antigravity150:antigravity:150")

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[sup] already live (pid $(cat "$LOCK")) — exit"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

ncount() { python3 -c "import json;print(len(json.load(open('funwake2/state/explore_$1/summary.json'))['trajectory']))" 2>/dev/null || echo 0; }
statline() { python3 - "funwake2/state/explore_$1/summary.json" "$1" "$2" <<'PY' 2>/dev/null || echo "$1 n=?"
import json,sys
try:
    d=json.load(open(sys.argv[1]));t=d["trajectory"];b=d.get("best_feasible") or {}
    print("%s n=%d/%s best=%+.4f%%(it%s) feas=%d/%d"%(sys.argv[2],len(t),sys.argv[3],
          b.get("pct",0),b.get("iter","?"),sum(1 for x in t if x.get("feasible")),len(t)))
except Exception:
    print("%s n=0/%s (no data yet)"%(sys.argv[2],sys.argv[3]))
PY
}

for tick in $(seq 1 7); do
  alldone=1
  for r in "${RUNS[@]}"; do
    IFS=: read -r tag eng iters <<< "$r"
    n=$(ncount "$tag")
    if [ "$n" -lt "$iters" ]; then
      alldone=0
      if ! pgrep -f "run_gen.sh $tag " >/dev/null 2>&1; then
        echo "[sup] $(date +%H:%M:%S) $tag loop DEAD -> relaunch detached" >> "funwake2/state/explore_$tag/run.log"
        mkdir -p "funwake2/state/explore_$tag"
        nohup bash funwake2/state/run_gen.sh "$tag" "$eng" "$iters" >> "funwake2/state/explore_$tag/run.log" 2>&1 &
        disown 2>/dev/null || true
      fi
    fi
    echo "[sup] $(date +%H:%M:%S) tick=$tick $(statline "$tag" "$iters")"
  done
  if [ "$alldone" = 1 ]; then echo "[sup] ALL DONE"; exit 0; fi
  sleep 300
done
echo "[sup] window elapsed — relaunch me"
