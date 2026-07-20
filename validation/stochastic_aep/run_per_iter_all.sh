#!/bin/bash
# Task 1 — per-iter AEP runs for mechanism (3 cells × ES off/on-runmax) and
# convergence (DEI N=50 training + ROWP N=80 held-out, ES-off only).
set -e
cd /Users/julianquick/portfolio_copy/funwake
export PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep

# Mechanism cells: top 3 H2-largest-negative ΔAEP
MECH_CELLS=(
    "results/matrix/problem_rowp_n80_roserowp.json"
    "results/matrix/problem_rowp_n80_roseomnidir.json"
    "results/matrix/problem_rowp_n70_roserowp.json"
)

for cell in "${MECH_CELLS[@]}"; do
    name=$(basename "$cell" .json | sed 's/problem_//')
    for es in off on_runmax; do
        out="validation/stochastic_aep/per_iter_${name}_${es}.json"
        echo "=== $name $es ==="
        pixi run python validation/stochastic_aep/run_per_iter_aep.py \
            --cell "$cell" --schedule claude_iter192 --es-mode "$es" \
            --sample-seeds 100000 200000 300000 \
            --probe-every 200 --out "$out"
    done
done

# Convergence cells (ES-off only): training + held-out
CONV_CELLS=(
    "results/matrix/problem_dei_n50_rosedei.json"
)
for cell in "${CONV_CELLS[@]}"; do
    name=$(basename "$cell" .json | sed 's/problem_//')
    out="validation/stochastic_aep/per_iter_${name}_off.json"
    echo "=== $name off (convergence) ==="
    pixi run python validation/stochastic_aep/run_per_iter_aep.py \
        --cell "$cell" --schedule claude_iter192 --es-mode off \
        --sample-seeds 100000 200000 300000 \
        --probe-every 200 --out "$out"
done

echo "Done."
