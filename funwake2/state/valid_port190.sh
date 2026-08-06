#!/usr/bin/env bash
# Validate the portfolio generalist (it190) on the Mac, same 6-cell protocol as the
# single-farm specialists (dei_n50 + held-out rowp_n74 @ 10 seeds; four others @ 5),
# so select_deploy.py compares them apples-to-apples. Resumable per cell@nseeds.
set -u
cd /Users/julianquick/working/funwake
CAND=funwake2/state/seed_port_gbar_iter190_clean.py
OUT=funwake2/state/validation/port190.json
mkdir -p funwake2/state/validation
ncells() { python3 -c "import json,os;print(len(json.load(open('$OUT'))['cells']) if os.path.exists('$OUT') else 0)" 2>/dev/null || echo 0; }
while :; do
  n=$(ncells)
  echo "[valid190] $(date +%H:%M:%S) cells=$n/6"
  if [ "$n" -ge 6 ]; then echo "[valid190] DONE"; break; fi
  pixi run python funwake2/run_validation.py --cand "$CAND" --out "$OUT" \
      --cells dei_n50 rowp_n74 --seeds 0 1 2 3 4 5 6 7 8 9
  pixi run python funwake2/run_validation.py --cand "$CAND" --out "$OUT" \
      --cells parque_n20 parque_n10_omnidir dei_n80_omnidir dei_n50_uniform --seeds 0 1 2 3 4
  for _ in $(seq 1 30); do :; done
done
