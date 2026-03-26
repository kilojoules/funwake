#!/usr/bin/env bash
set -e

WIND_CSV="${1:-$HOME/clusters/energy_island_10y_daily_av_wind.csv}"
PIXWAKE_REPO="${2:-https://github.com/kilojoules/cluster-tradeoffs.git}"
PIXWAKE_COMMIT="${3:-b8e905a}"

cd "$(dirname "$0")"

# 1. Clone pixwake playground
if [ ! -d "playground/pixwake" ]; then
    echo "Cloning pixwake playground..."
    mkdir -p playground
    git clone "$PIXWAKE_REPO" playground/pixwake
    cd playground/pixwake && git checkout "$PIXWAKE_COMMIT" && cd ../..
else
    echo "Playground exists."
fi

# 2. Run baselines on all farms (500 multi-starts, max_iter=4000, const_lr=2000)
echo ""
echo "Running baselines on all 10 farms (500 multi-starts — this takes ~2.5 hours)..."
PYTHONPATH="playground/pixwake/src:$PYTHONPATH" \
JAX_ENABLE_X64=True \
python benchmarks/dei_layout.py --wind-csv "$WIND_CSV" baseline-all --output results

echo ""
echo "Setup complete."
echo "Training baselines: results/baselines.json (farms 1-9)"
echo "Test baseline:      results/baselines.json (farm 0)"
echo "Problem info:       results/problem_farm{0..9}.json"
