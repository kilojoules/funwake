#!/bin/bash
# Run a single baseline start as part of a SLURM array.
#
# Usage:
#   PROBLEM=results/problem_dei_n50.json sbatch --array=0-499 lumi/baseline_array.sbatch
#
# Each array task runs one topfarm_sgd_solve with seed = SLURM_ARRAY_TASK_ID.

#SBATCH --job-name=baseline
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=lumi/logs/baselines/%x_%A_%a.out
#SBATCH --error=/dev/null

cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs/baselines

PROBLEM=${PROBLEM:-results/problem_dei_n50.json}
SEED=${SLURM_ARRAY_TASK_ID}

pixi run python tools/run_single_baseline.py "$PROBLEM" --seed "$SEED"
