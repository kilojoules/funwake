#!/bin/bash
# Smoke test for --stress-problem flag. Scores seed_schedule.py on
# the training cell AND the uniform stress cell; should print JSON
# with stress_aep_gwh, stress_gap, min_gap fields.
#
#SBATCH --account=project_465002609
#SBATCH --job-name=funwake-smoke
#SBATCH --partition=small
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=lumi/logs/smoke_stress_%j.out
#SBATCH --error=lumi/logs/smoke_stress_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs

export JAX_PLATFORMS=cpu
LOG="lumi/logs/smoke_stress_attempt_${SLURM_JOB_ID}.json"

echo "=== Smoke test: --stress-problem ==="
echo "Start: $(date)"

pixi run python tools/run_optimizer.py \
    results/seed_schedule.py \
    --schedule-only \
    --problem playground/problem.json \
    --stress-problem results/matrix/problem_dei_n50_roseuniform.json \
    --stress-baseline 5599.79 \
    --timeout 180 \
    --log "$LOG"

echo ""
echo "=== Log entry ==="
cat "$LOG"
echo ""
echo "Done: $(date)"
