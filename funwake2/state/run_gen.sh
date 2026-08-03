#!/usr/bin/env bash
# Generic self-healing explore loop. Args: <tag> <engine> <iters>
# Loops `--resume` until <iters> attempts are checkpointed in explore_<tag>;
# re-resumes on any python crash/kill. Process signature includes the tag so
# concurrent runs (e.g. claude1000 + gemini150) don't match each other's pgrep.
set -u
cd /Users/julianquick/working/funwake
TAG="$1"; ENGINE="$2"; ITERS="$3"
SUM="funwake2/state/explore_${TAG}/summary.json"
count() { python3 -c "import json;print(len(json.load(open('$SUM'))['trajectory']))" 2>/dev/null || echo 0; }
while :; do
  n=$(count)
  echo "[loop:$TAG] $(date +%H:%M:%S) attempts=$n/$ITERS"
  if [ "$n" -ge "$ITERS" ]; then echo "[loop:$TAG] DONE $n"; break; fi
  pixi run python funwake2/run_codex_explore.py \
      --engine "$ENGINE" --tag "$TAG" --resume \
      --iters "$ITERS" --seeds 0 1 2 3 4 --steps 8000
  echo "[loop:$TAG] python rc=$? at $(date +%H:%M:%S) n=$(count)"
  for _ in $(seq 1 30); do :; done   # brief guard against hot-spin
done
