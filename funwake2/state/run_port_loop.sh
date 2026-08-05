#!/usr/bin/env bash
# Self-healing loop for the gbar-backed portfolio run. The heavy 5x3 eval runs on
# the gbar worker (compute node); this loop only does LLM mutation + dispatch.
set -u
cd /Users/julianquick/working/funwake
TAG=port_gbar
ENGINE=claude
ITERS=200
SUM="funwake2/state/explore_${TAG}/summary.json"
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }
while :; do
  n=$(count)
  echo "[port_gbar] $(date +%H:%M:%S) attempts=$n/$ITERS"
  if [ "$n" -ge "$ITERS" ]; then echo "[port_gbar] DONE $n"; break; fi
  pixi run python funwake2/run_portfolio_explore.py \
      --engine "$ENGINE" --tag "$TAG" --resume \
      --iters "$ITERS" --seeds 0 1 2 --steps 8000 --eval-backend gbar
  echo "[port_gbar] python rc=$? at $(date +%H:%M:%S) n=$(count)"
  for _ in $(seq 1 30); do :; done
done
