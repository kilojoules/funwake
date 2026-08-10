#!/usr/bin/env bash
# Leave-one-farm-out cross-validation of portfolio training. For each held-out farm:
# train a portfolio search on the OTHER 4 farms (farm-balanced, gbar backend), then
# evaluate the fold-winner ONLY on the held-out farm. The held-out farm is never in
# the fold's fitness/feedback, so no cell is both trained and tested. Sequential +
# resumable (skips folds whose result file exists; resumes a fold's training).
set -u
cd /Users/julianquick/working/funwake
FARMS=(dei_n50 dei_n80_omnidir dei_n50_uniform parque_n20 parque_n10_omnidir)
ENGINE=codex
ITERS=60
SEEDS="0 1 2"
RESDIR=funwake2/state/lofo
mkdir -p "$RESDIR"
GBAR="ssh -o BatchMode=yes -o ConnectTimeout=25 gbar"
count() { python3 -c "import json,os;p='$1';print(len(json.load(open(p))['trajectory']) if os.path.exists(p) else 0)" 2>/dev/null || echo 0; }

# wait for the gbar worker to be RUNNING before dispatching (avoids wasted eval
# timeouts while it queues)
for i in $(seq 1 120); do
  $GBAR 'bash -lc "bjobs -J funwake_wk 2>/dev/null | grep -q RUN"' 2>/dev/null && { echo "[lofo] worker RUNNING"; break; }
  echo "[lofo] waiting for gbar worker (poll $i)..."; sleep 30
done

for HO in "${FARMS[@]}"; do
  TAG=lofo_${HO}
  SUM=funwake2/state/explore_${TAG}/summary.json
  RES=$RESDIR/${HO}_heldout.json
  [ -f "$RES" ] && { echo "[lofo] $HO already done"; continue; }
  TRAIN=""
  for f in "${FARMS[@]}"; do [ "$f" != "$HO" ] && TRAIN="$TRAIN $f"; done
  echo "[lofo] $(date '+%m-%d %H:%M') FOLD held-out=$HO train=$TRAIN"

  while [ "$(count "$SUM")" -lt "$ITERS" ]; do
    pixi run python funwake2/run_portfolio_explore.py --tag "$TAG" --cells $TRAIN \
        --engine "$ENGINE" --eval-backend gbar --iters "$ITERS" --seeds $SEEDS \
        --steps 8000 --resume
    for _ in $(seq 1 20); do :; done
  done

  BEST_IT=$(python3 -c "import json;print((json.load(open('$SUM')).get('best_feasible') or {}).get('iter','none'))")
  if [ "$BEST_IT" = "none" ]; then
    echo "{\"heldout\":\"$HO\",\"status\":\"no-feasible-winner\"}" > "$RES"; continue; fi
  WIN=funwake2/state/explore_${TAG}/iter_$(printf %03d "$BEST_IT").py
  scp -q -o BatchMode=yes "$WIN" gbar:'~/funwake/gbar_queue/.lofo_win.py' 2>/dev/null
  $GBAR "bash -lc 'cd ~/funwake && export PIXI_CACHE_DIR=~/.cache/rattler && pixi run python funwake2/gbar_eval.py --schedule gbar_queue/.lofo_win.py --cells $HO --seeds $SEEDS --steps 8000 --out ~/funwake/gbar_queue/lofo_${HO}_win.json --jobs 3'" >/dev/null 2>&1
  $GBAR "cat ~/funwake/gbar_queue/lofo_${HO}_win.json" > /tmp/lofo_${HO}_win.json 2>/dev/null
  python3 - "$HO" "$SEEDS" "$BEST_IT" > "$RES" <<'PY'
import json,sys,statistics,subprocess
ho,seeds,bit=sys.argv[1],sys.argv[2].split(),sys.argv[3]
win=json.load(open(f"/tmp/lofo_{ho}_win.json"))["cells"][ho]
nat=json.loads(subprocess.run(["ssh","-o","BatchMode=yes","gbar","cat ~/funwake/gbar_native_baselines.json"],
    capture_output=True,text=True).stdout)["cells"][ho]
wm=statistics.fmean(win[s]["aep"] for s in seeds); nm=statistics.fmean(nat[s]["aep"] for s in seeds)
feas=sum(1 for s in seeds if win[s]["feasible"])
print(json.dumps({"heldout":ho,"best_iter":bit,"heldout_delta_pct":round(100*(wm-nm)/nm,4),
    "heldout_feasible":f"{feas}/{len(seeds)}","win_mean":round(wm,4),"native_mean":round(nm,4)},indent=2))
PY
  echo "[lofo] $HO done -> $(python3 -c "import json;d=json.load(open('$RES'));print(d.get('heldout_delta_pct'),d.get('heldout_feasible'))" 2>/dev/null)"
done
echo "[lofo] ALL FOLDS DONE $(date '+%m-%d %H:%M')"
