#!/usr/bin/env bash
# Self-healing driver for the Claude 150-iteration explore run.
# Loops `--resume` until 150 attempts are checkpointed; re-resumes on any
# python crash/kill. Wrapped as a tracked background task so an environment
# kill of the whole loop still notifies the session for relaunch.
set -u
cd /Users/julianquick/working/funwake
SUM=funwake2/state/explore_claude150/summary.json
TARGET=150

count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }

while :; do
  n=$(count)
  echo "[loop] $(date +%H:%M:%S) attempts=$n/$TARGET"
  if [ "$n" -ge "$TARGET" ]; then echo "[loop] DONE at $n"; break; fi
  pixi run python funwake2/run_codex_explore.py \
      --engine claude --tag claude150 --resume \
      --iters "$TARGET" --seeds 0 1 2 3 4 --steps 8000
  rc=$?
  echo "[loop] python exited rc=$rc at $(date +%H:%M:%S); n=$(count)"
  # tiny guard so a fast-failing python can't spin the loop hot
  for _ in $(seq 1 30); do :; done
done
