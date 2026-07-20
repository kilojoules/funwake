#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

WIND_CSV="${1:-dependencies/pixwake/energy_island_10y_daily_av_wind.csv}"
# pixwake source repo (only needed for the fallback re-clone below).
PIXWAKE_REPO="${2:-https://github.com/kilojoules/cluster-tradeoffs.git}"
PIXWAKE_COMMIT="${3:-b8e905a}"

# 1. pixwake engine
# Vendored in-repo at dependencies/pixwake/ (the pixwake package from
# cluster-tradeoffs @ b8e905a, used on PYTHONPATH — see dependencies/README.md).
# No clone needed. Fallback re-clone only if the vendored copy is missing.
if [ ! -d "dependencies/pixwake/src/pixwake" ]; then
    echo "Vendored pixwake missing — re-cloning from source..."
    git clone "$PIXWAKE_REPO" _pixwake_tmp
    cd _pixwake_tmp && git checkout "$PIXWAKE_COMMIT" && cd ..
    mkdir -p dependencies/pixwake
    cp -R _pixwake_tmp/src dependencies/pixwake/src
    cp _pixwake_tmp/energy_island_10y_daily_av_wind.csv _pixwake_tmp/pyproject.toml dependencies/pixwake/ 2>/dev/null || true
    rm -rf _pixwake_tmp
else
    echo "pixwake vendored at dependencies/pixwake — OK."
fi

# 2. Run baselines on all farms (500 multi-starts, max_iter=4000, const_lr=2000)
echo ""
echo "Running baselines on all 10 farms (500 multi-starts — this takes ~2.5 hours)..."
PYTHONPATH="dependencies/pixwake/src:$PYTHONPATH" \
JAX_ENABLE_X64=True \
python benchmarks/dei_layout.py --wind-csv "$WIND_CSV" baseline-all --output results

echo ""
echo "Setup complete."
echo "Training baselines: results/baselines.json (farms 1-9)"
echo "Test baseline:      results/baselines.json (farm 0)"
echo "Problem info:       results/problem_farm{0..9}.json"
