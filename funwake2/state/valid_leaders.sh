#!/usr/bin/env bash
# Paired 6-cell validation of the current per-engine leaders vs native c*D, same
# protocol as the Claude iter_109 validation: dei_n50 + held-out rowp_n74 at 10
# seeds; parque_n20, parque_n10_omnidir, dei_n80_omnidir, dei_n50_uniform at 5.
# run_validation.py is resumable per cell@nseeds, so a kill just re-resumes.
# Self-healing internal loop: exits when both candidates have all 6 cells.
set -u
cd /Users/julianquick/working/funwake
mkdir -p funwake2/state/validation
# "seed-file-basename:out-json-basename"
PAIRS=("seed_antigravity150_iter488:antigravity488" "seed_codex1000_iter021:codex021")
ncells() { python3 -c "import json,os;p='$1';print(len(json.load(open(p))['cells']) if os.path.exists(p) else 0)" 2>/dev/null || echo 0; }

while :; do
  done_ct=0
  for pair in "${PAIRS[@]}"; do
    cand="funwake2/state/${pair%%:*}.py"
    out="funwake2/state/validation/${pair##*:}.json"
    n=$(ncells "$out")
    echo "[valid] $(date +%H:%M:%S) ${pair##*:} cells=$n/6"
    if [ "$n" -ge 6 ]; then done_ct=$((done_ct+1)); continue; fi
    pixi run python funwake2/run_validation.py --cand "$cand" --out "$out" \
        --cells dei_n50 rowp_n74 --seeds 0 1 2 3 4 5 6 7 8 9
    pixi run python funwake2/run_validation.py --cand "$cand" --out "$out" \
        --cells parque_n20 parque_n10_omnidir dei_n80_omnidir dei_n50_uniform --seeds 0 1 2 3 4
  done
  if [ "$done_ct" -ge 2 ]; then echo "[valid] ALL DONE"; break; fi
  for _ in $(seq 1 30); do :; done
done
