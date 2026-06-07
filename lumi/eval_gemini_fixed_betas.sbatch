#!/bin/bash
# Re-evaluate matrix with FIXED-beta Gemini schedule ablation.
#
# Runs tools/eval_matrix_schedules.py which is resume-safe: existing
# entries (Claude/Gemini scheduled/DeepSeek) are skipped; only the
# new label "Gemini schedule (fixed beta)" is computed (~48 cells
# at N<=80, plus 16 more at N=200/300 if time allows).
#
# Hypothesis: fixing beta1=0.3, beta2=0.5 (Claude's constants) does
# not degrade Gemini iter_192 AEP across the 48-cell matrix.
#
# Usage:
#   sbatch lumi/eval_gemini_fixed_betas.sbatch
#
#SBATCH --account=project_465002609
#SBATCH --job-name=fb-eval
#SBATCH --partition=small
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=lumi/logs/eval_fixed_betas_%j.out
#SBATCH --error=lumi/logs/eval_fixed_betas_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs

echo "=== Gemini fixed-beta ablation eval ==="
echo "Start: $(date)"
echo "Output appends to: results/matrix/schedules_matrix.json"

pixi run python tools/eval_matrix_schedules.py --max-n 80

echo "Done: $(date)"
