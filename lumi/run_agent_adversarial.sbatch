#!/bin/bash
# Adversarial-selection schedule-only agent run.
#
# Every attempt is scored on BOTH the training cell (DEI N=50, observed
# rose) AND the stress cell (DEI N=50, uniform rose) — the documented
# failure mode in §2.3 of the paper. Deployment criterion is
# min(train_gap, stress_gap), so the agent is forced to find schedules
# that don't break on uniform wind.
#
# Usage:
#   sbatch lumi/run_agent_adversarial.sbatch
#
# Requires: Claude Code CLI installed and authenticated on LUMI for user.
# If that's not available, run on a machine where it is (e.g. Sophia or
# local) — the agent loop itself doesn't need GPU; only the matrix
# replay (lumi/replay_candidates_matrix.sbatch) does.
#
#SBATCH --account=project_465002609
#SBATCH --job-name=funwake-adv
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=lumi/logs/agent_adversarial_%j.out
#SBATCH --error=lumi/logs/agent_adversarial_%j.err

set -e
cd /scratch/project_465002609/julian/funwake_adv
mkdir -p lumi/logs

# Ensure pixwake is cloned (idempotent)
if [ ! -d "playground/pixwake" ]; then
    mkdir -p playground
    git clone https://github.com/kilojoules/cluster-tradeoffs.git playground/pixwake
    cd playground/pixwake && git checkout b8e905a && cd ../..
fi

# Stress cell: same N=50 DEI polygon, uniform wind rose.
# Baseline AEP from results/matrix/baselines_matrix.json (dei_n50_roseuniform.best_aep).
STRESS_PROBLEM="results/matrix/problem_dei_n50_roseuniform.json"
STRESS_BASELINE=5599.79
OUTDIR="results_agent_schedule_adversarial"

echo "=== FunWake adversarial agent run ==="
echo "Start: $(date)"
echo "Stress: $STRESS_PROBLEM (baseline $STRESS_BASELINE GWh)"
echo "Output: $OUTDIR"

pixi run python agent_cli.py \
    --provider claude-code \
    --schedule-only \
    --wind-csv playground/pixwake/energy_island_10y_daily_av_wind.csv \
    --hot-start results/seed_schedule.py \
    --output-dir "$OUTDIR" \
    --time-budget 18000 \
    --timeout-per-run 90 \
    --adversarial-stress "$STRESS_PROBLEM" \
    --adversarial-stress-baseline "$STRESS_BASELINE"

echo "Done: $(date)"
echo "Saved scripts: $(ls $OUTDIR/iter_*.py 2>/dev/null | wc -l)"
