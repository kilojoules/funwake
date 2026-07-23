#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Args: an optional wind CSV path (defaults to the tracked one) and an optional
# --recompute-baselines flag to force the (expensive) baseline recompute.
WIND_CSV="dependencies/energy_island_10y_daily_av_wind.csv"
RECOMPUTE_BASELINES=0
for arg in "$@"; do
    case "$arg" in
        --recompute-baselines) RECOMPUTE_BASELINES=1 ;;
        *.csv) WIND_CSV="$arg" ;;
    esac
done
# pixwake source repo (only needed for the clone below).
PIXWAKE_REPO="https://github.com/kilojoules/cluster-tradeoffs.git"
PIXWAKE_COMMIT="b8e905a"

# 1. pixwake engine
# NOT vendored (private — see dependencies/README.md). Used on PYTHONPATH at
# dependencies/pixwake/src (gitignored). Cloned here when missing.
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

# 2. Baselines (500-multistart best-feasible AEP per farm; max_iter=4000,
#    additional_constant_lr_iterations=2000). These are COMMITTED
#    (results/baselines.json, results/baseline_rowp.json), so a fresh clone and
#    CI never recompute. Recomputing is ~2.6 h/farm (~26 h for all 10 farms; the
#    per-farm elapsed_s is recorded inside the JSONs). Force with
#    --recompute-baselines.
if [ -f "results/baselines.json" ] && [ "$RECOMPUTE_BASELINES" -eq 0 ]; then
    echo ""
    echo "Baselines present (results/baselines.json) — skipping recompute."
    echo "Pass --recompute-baselines to force (~2.6 h/farm, ~26 h for all 10)."
else
    echo ""
    echo "Recomputing baselines (500 multi-starts/farm — ~2.6 h/farm)..."
    PYTHONPATH="dependencies/pixwake/src:$PYTHONPATH" \
    JAX_ENABLE_X64=True \
    python benchmarks/dei_layout.py --wind-csv "$WIND_CSV" baseline-all --output results
fi

echo ""
echo "Setup complete."
echo "Baselines: results/baselines.json (farm 1 = train, farm 0 = test)."
echo "Problems:  results/problem_farm{0..9}.json"
