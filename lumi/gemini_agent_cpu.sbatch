#!/bin/bash
# Run a Gemini API agent on a LUMI CPU node.
# The optimizer evals (JAX SGD) run on CPU; Gemini API calls go over HTTPS.
# Rate-limit retry logic handles 429/503 errors gracefully.
#
# Usage:
#   TRAIN_PROBLEM=results/matrix/problem_dei_n50_roseuniform.json \
#   BASELINES=results/baselines_with_uniform.json \
#   TRAIN_FARM=uniform \
#   MAX_ATTEMPTS=50 \
#   sbatch --export=ALL lumi/gemini_agent_cpu.sbatch

#SBATCH --job-name=gemini-agent
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=lumi/logs/gemini_agent_%j.log
#SBATCH --error=lumi/logs/gemini_agent_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs

export JAX_ENABLE_X64=True
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

# Configurable via environment
TRAIN_PROBLEM=${TRAIN_PROBLEM:-playground/problem.json}
BASELINES=${BASELINES:-results/baselines.json}
TRAIN_FARM=${TRAIN_FARM:-1}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-50}
HOT_START=${HOT_START:-results/seed_optimizer.py}
SCHEDULE_ONLY=${SCHEDULE_ONLY:-1}
OUTPUT_DIR=${OUTPUT_DIR:-results_agent_gemini_cpu_${SLURM_JOB_ID}}
MODEL=${MODEL:-gemini-2.5-flash}

CLI_ARGS=(
    --provider gemini
    --model "$MODEL"
    --wind-csv playground/pixwake/energy_island_10y_daily_av_wind.csv
    --time-budget 43200
    --max-attempts "$MAX_ATTEMPTS"
    --hot-start "$HOT_START"
    --train-problem "$TRAIN_PROBLEM"
    --baselines "$BASELINES"
    --train-farm "$TRAIN_FARM"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "$SCHEDULE_ONLY" ]; then
    CLI_ARGS+=(--schedule-only)
fi

echo "=== Gemini Agent on LUMI CPU ==="
echo "  Problem:      $TRAIN_PROBLEM"
echo "  Baselines:    $BASELINES"
echo "  Train farm:   $TRAIN_FARM"
echo "  Max attempts: $MAX_ATTEMPTS"
echo "  Hot start:    $HOT_START"
echo "  Output:       $OUTPUT_DIR"
echo "  Model:        $MODEL"
echo ""

pixi run python agent_cli.py "${CLI_ARGS[@]}"
