#!/bin/bash
# Task G — iter_192-no-bumps on 3 mechanism cells, ES-off, per-iter AEP + grad norms
set -e
cd /Users/julianquick/portfolio_copy/funwake
export PYTHONPATH=playground/pixwake/src:validation/stochastic_aep

for cell_name in rowp_n80_roserowp rowp_n80_roseomnidir rowp_n70_roserowp; do
    out="validation/stochastic_aep/gb_${cell_name}_iter192_nobumps_off.json"
    if [ ! -f "$out" ]; then
        echo "=== G: $cell_name iter192_no_bumps off ==="
        pixi run python validation/stochastic_aep/run_gradient_balance.py \
            --cell "results/matrix/problem_${cell_name}.json" \
            --schedule claude_iter192_no_bumps --es-mode off \
            --sample-seeds 100000 200000 300000 --probe-every 200 --out "$out"
    fi
done
echo "Done."
