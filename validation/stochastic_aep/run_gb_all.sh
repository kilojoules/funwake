#!/bin/bash
set -e
cd /Users/julianquick/portfolio_copy/funwake
export PYTHONPATH=playground/pixwake/src:validation/stochastic_aep

run_pair() {
    local cell_name="$1"
    local eta="$2"
    local cell="results/matrix/problem_${cell_name}.json"

    for kind in "iter192_off:claude_iter192:off" "iter192_on:claude_iter192:on_runmax" \
                "baseline_eta${eta}_off:decay_es_eta_${eta}:off" \
                "baseline_eta${eta}_on:decay_es_eta_${eta}:on_runmax"; do
        IFS=":" read -r tag sched es <<< "$kind"
        out="validation/stochastic_aep/gb_${cell_name}_${tag}.json"
        if [ ! -f "$out" ]; then
            echo "=== gb $cell_name $tag ==="
            pixi run python validation/stochastic_aep/run_gradient_balance.py \
                --cell "$cell" --schedule "$sched" --es-mode "$es" \
                --sample-seeds 100000 200000 300000 --probe-every 200 --out "$out"
        fi
    done
}

run_pair rowp_n80_roserowp 5.0
run_pair rowp_n80_roseomnidir 1.0
run_pair rowp_n70_roserowp 1.0

echo "Done."
