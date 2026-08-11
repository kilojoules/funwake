#!/usr/bin/env bash
set -u
cd /Users/julianquick/working/funwake
TAG=scale14
SUM="funwake2/state/explore_${TAG}/summary.json"
ITERS=80
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }
while :; do
  n=$(count)
  echo "[scale14] $(date +%H:%M:%S) attempts=$n/$ITERS"
  [ "$n" -ge "$ITERS" ] && { echo "[scale14] DONE $n"; break; }
  pixi run python funwake2/run_portfolio_explore.py --engine codex --tag "$TAG" --resume \
      --iters "$ITERS" --seeds 0 1 2 --steps 8000 --eval-backend gbar --feas-slack 3 \
      --cells dei_n30_rosedei dei_n30_roseuniform dei_n40_roseomnidir dei_n40_roserowp dei_n50_rosedei dei_n50_roseuniform dei_n60_roserowp dei_n60_roseomnidir dei_n70_rosedei dei_n80_roseomnidir dei_n80_roseuniform parque_n10_omnidir parque_n20 parque_n30_uniform
  echo "[scale14] python rc=$? at $(date +%H:%M:%S) n=$(count)"
  for _ in $(seq 1 20); do :; done
done
