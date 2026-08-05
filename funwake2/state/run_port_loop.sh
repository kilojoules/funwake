#!/usr/bin/env bash
# Self-healing loop for the portfolio (multi-farm) run. Resumes until ITERS real
# attempts are checkpointed; re-resumes on any python crash/kill. Starts fresh from
# native on the first run (--resume no-ops when there's no checkpoint yet).
set -u
cd /Users/julianquick/working/funwake
TAG=port_claude
ENGINE=claude
ITERS=200
SUM="funwake2/state/explore_${TAG}/summary.json"
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }
while :; do
  n=$(count)
  echo "[port] $(date +%H:%M:%S) attempts=$n/$ITERS"
  if [ "$n" -ge "$ITERS" ]; then echo "[port] DONE $n"; break; fi
  pixi run python funwake2/run_portfolio_explore.py \
      --engine "$ENGINE" --tag "$TAG" --resume \
      --iters "$ITERS" --seeds 0 1 2 --steps 8000
  echo "[port] python rc=$? at $(date +%H:%M:%S) n=$(count)"
  for _ in $(seq 1 30); do :; done
done
