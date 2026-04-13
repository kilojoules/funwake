#!/bin/bash
# Submit 500 independent baseline jobs per farm variant.
# Each job runs ONE topfarm_sgd_solve start — true embarrassing parallelism.
#
# Usage:
#   bash lumi/submit_baselines.sh
#
# Results: lumi/logs/baseline_<farm>_<seed>.out (one JSON per job)
# Collect: cat lumi/logs/baseline_dei_n50_*.out | python3 -c '...'

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs/baselines

N_STARTS=500

for problem in results/problem_dei_n30.json results/problem_dei_n40.json results/problem_dei_n50.json results/problem_dei_n60.json results/problem_dei_n70.json results/problem_dei_n80.json results/problem_rowp.json; do
    farm=$(basename $problem .json)
    echo "Submitting $N_STARTS jobs for $farm..."

    for seed in $(seq 0 $((N_STARTS - 1))); do
        sbatch --account=project_465002609 \
            --partition=small-g \
            --gpus-per-node=1 \
            --time=00:15:00 \
            --mem=16G \
            --job-name="bl-${farm}-${seed}" \
            --output="lumi/logs/baselines/${farm}_${seed}.out" \
            --error="/dev/null" \
            --parsable \
            --wrap="pixi run python tools/run_single_baseline.py $problem --seed $seed" \
            > /dev/null
    done
    echo "  Submitted $N_STARTS jobs for $farm"
done

echo ""
echo "Total: $((N_STARTS * 7)) jobs submitted"
echo "Monitor: squeue -u \$(whoami) | grep bl- | wc -l"
echo "Collect results when done:"
echo "  python3 tools/collect_baselines.py"
