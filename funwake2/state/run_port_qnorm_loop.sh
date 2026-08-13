#!/usr/bin/env bash
# Fresh portfolio search under the QUAD-NORM spacing form (5-farm portfolio, codex).
set -u
cd /Users/julianquick/working/funwake
TAG=port_qnorm
SUM="funwake2/state/explore_${TAG}/summary.json"
ITERS=120
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }
while :; do
  n=$(count)
  echo "[port_qnorm] $(date +%H:%M:%S) attempts=$n/$ITERS"
  [ "$n" -ge "$ITERS" ] && { echo "[port_qnorm] DONE $n"; break; }
  pixi run python funwake2/run_portfolio_explore.py --engine codex --tag "$TAG" --resume \
      --iters "$ITERS" --seeds 0 1 2 --steps 8000 --eval-backend gbar --feas-slack 1 \
      --cells dei_n50 dei_n80_omnidir dei_n50_uniform parque_n20 parque_n10_omnidir
  echo "[port_qnorm] python rc=$? at $(date +%H:%M:%S) n=$(count)"
  for _ in $(seq 1 20); do :; done
done
