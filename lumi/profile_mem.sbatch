#!/bin/bash
# Profile peak RSS of a single schedule evaluation (with and without stress)
# to size memory for local runs.
#
#SBATCH --account=project_465002609
#SBATCH --job-name=funwake-mem
#SBATCH --partition=small
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=lumi/logs/profile_mem_%j.out
#SBATCH --error=lumi/logs/profile_mem_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=== Single train-only eval ==="
/usr/bin/time -v pixi run python tools/run_optimizer.py \
    results/seed_schedule.py \
    --schedule-only \
    --problem playground/problem.json \
    --timeout 180 \
    --log /tmp/mem_${SLURM_JOB_ID}.json 2>&1 | tail -25

echo ""
echo "=== Train + stress eval ==="
/usr/bin/time -v pixi run python tools/run_optimizer.py \
    results/seed_schedule.py \
    --schedule-only \
    --problem playground/problem.json \
    --stress-problem results/matrix/problem_dei_n50_roseuniform.json \
    --stress-baseline 5599.79 \
    --timeout 180 \
    --log /tmp/mem_stress_${SLURM_JOB_ID}.json 2>&1 | tail -25

echo ""
echo "Done: $(date)"
