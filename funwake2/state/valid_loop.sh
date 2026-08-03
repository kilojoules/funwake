#!/usr/bin/env bash
# Paired validation of the Claude-150 winner (iter_109) vs native c*D.
# 10 seeds on the two decision-critical cells (dei_n50, held-out rowp_n74);
# 5 seeds on the four generalization cells (matches the iter_04 protocol).
# run_validation.py is resumable (per cell@nseeds), so a kill just re-resumes.
set -u
cd /Users/julianquick/working/funwake
CAND=funwake2/state/seed_claude150_iter109.py
OUT=funwake2/state/validation/iter109.json
mkdir -p funwake2/state/validation
ncells() { python3 -c "import json;print(len(json.load(open('$OUT'))['cells']))" 2>/dev/null || echo 0; }
while :; do
  n=$(ncells)
  echo "[valid] $(date +%H:%M:%S) cells_done=$n/6"
  if [ "$n" -ge 6 ]; then echo "[valid] DONE $n/6"; break; fi
  pixi run python funwake2/run_validation.py --cand "$CAND" --out "$OUT" \
      --cells dei_n50 rowp_n74 --seeds 0 1 2 3 4 5 6 7 8 9
  pixi run python funwake2/run_validation.py --cand "$CAND" --out "$OUT" \
      --cells parque_n20 parque_n10_omnidir dei_n80_omnidir dei_n50_uniform --seeds 0 1 2 3 4
  for _ in $(seq 1 30); do :; done
done
