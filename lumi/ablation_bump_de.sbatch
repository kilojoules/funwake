#!/bin/bash
# Bump-family differential-evolution ablation.
# Single GCD (DE is inherently sequential: workers=1 in the optimizer).
# Expected wall: popsize 15 * maxiter 20 = ~300 evaluations * ~75s
# (train + post-hoc ROWP backfill) ~ 5-6 hours.
#
# Usage:
#   sbatch lumi/ablation_bump_de.sbatch

#SBATCH --job-name=bump-de
#SBATCH --account=project_465002609
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=lumi/logs/bump_de_%j.log
#SBATCH --error=lumi/logs/bump_de_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs

MAX_ITER=${MAX_ITER:-20}
POPSIZE=${POPSIZE:-15}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-results_bump_opt}
INIT=${INIT:-claude}  # "claude" or "baseline" for the fair ablation

export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=True
export PYTHONUNBUFFERED=1

echo "=== Bump-family DE Ablation ==="
echo "popsize=$POPSIZE maxiter=$MAX_ITER seed=$SEED init=$INIT"
echo "output=$OUTPUT_DIR"
echo ""

pixi run python tools/optimize_bump_family.py \
    --max-iter "$MAX_ITER" \
    --popsize "$POPSIZE" \
    --seed "$SEED" \
    --init "$INIT" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "DE done. Best schedule in $OUTPUT_DIR/bump_opt_log.json"
