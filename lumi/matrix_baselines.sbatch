#!/bin/bash
# Run 100-multi-start baselines for the 48-cell farm × N × wind-rose
# matrix on LUMI. Shards across 8 GCDs on one standard-g node.
#
# Total work: 48 problems × 100 seeds = 4,800 SGD runs.
# (Original paper baseline uses 500 starts on DEI-n50 and ROWP only;
#  the matrix cells use 100 for tractable compute — still a robust
#  noise estimate for a per-cell reference line.)
#
# Grouped by problem so JAX JIT compile is amortized per problem
# rather than per run. Each GCD handles every 8th line of the work
# list.
#
# Usage:
#   sbatch lumi/matrix_baselines.sbatch

#SBATCH --job-name=mxbase
#SBATCH --account=project_465002609
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=lumi/logs/mxbase_%j.log
#SBATCH --error=lumi/logs/mxbase_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs/matrix_baselines

export JAX_ENABLE_X64=True
export PYTHONUNBUFFERED=1

# Build a work list: (problem_path, seed) pairs for all 48 cells × 500 seeds.
# Group by problem (problems contiguous) so JIT compile is amortized.
python3 << 'PYEOF' > /tmp/matrix_worklist.txt
import json, os
m = json.load(open("results/matrix/manifest.json"))
N_SEEDS = 100
for cell in m["cells"]:
    for seed in range(N_SEEDS):
        out = f"lumi/logs/matrix_baselines/{cell['farm']}_n{cell['n']}_rose{cell['rose']}_seed{seed}.out"
        # Skip if already done with non-empty output
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        print(f"{cell['path']} {seed} {out}")
PYEOF

TOTAL=$(wc -l < /tmp/matrix_worklist.txt)
echo "Matrix baselines: $TOTAL (problem, seed) pairs across 8 GCDs"
echo ""

# Launch 8 workers. Each worker handles every 8th line.
for gcd in 0 1 2 3 4 5 6 7; do
    (
        export ROCR_VISIBLE_DEVICES=$gcd
        export HIP_VISIBLE_DEVICES=$gcd
        awk -v g=$gcd 'NR % 8 == g' /tmp/matrix_worklist.txt | while read problem seed out; do
            if [ ! -s "$out" ]; then
                pixi run python tools/run_single_baseline.py "$problem" --seed "$seed" > "$out" 2>/dev/null
            fi
        done
        echo "GCD $gcd done"
    ) &
done
wait

echo ""
echo "All GCDs complete. Aggregate with:"
echo "  pixi run python tools/aggregate_matrix_baselines.py \\"
echo "      --matrix-dir results/matrix \\"
echo "      --out results/matrix/baselines_matrix.json"
