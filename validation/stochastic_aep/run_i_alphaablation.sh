#!/bin/bash
# Task I — iter_192-baseline-α: iter_192 lr+bumps, baseline α profile.
set -e
cd /Users/julianquick/portfolio_copy/funwake
export PYTHONPATH=playground/pixwake/src:validation/stochastic_aep

for cell_name in rowp_n80_roserowp rowp_n80_roseomnidir rowp_n70_roserowp; do
    out="validation/stochastic_aep/gb_${cell_name}_iter192_baselinealpha_off.json"
    if [ ! -f "$out" ]; then
        echo "=== I: $cell_name iter192_baseline_alpha off ==="
        pixi run python validation/stochastic_aep/run_gradient_balance.py \
            --cell "results/matrix/problem_${cell_name}.json" \
            --schedule claude_iter192_baseline_alpha --es-mode off \
            --sample-seeds 100000 200000 300000 --probe-every 200 --out "$out"
    fi
done
echo "Done."
