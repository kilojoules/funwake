#!/bin/bash
# Run all baselines in parallel across 8 GCDs on a standard-g node.
# Each GCD runs its own worker process that loops through assigned seeds.
#
# Usage:
#   sbatch lumi/baselines_parallel_node.sbatch

#SBATCH --job-name=bl-parallel
#SBATCH --account=project_465002609
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=lumi/logs/baseline_parallel_%j.log
#SBATCH --error=lumi/logs/baseline_parallel_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs/baselines

# 8 GCDs, each handles a slice of the work
# Total work: n=30 (seeds 206-499), n=60 (seeds 485-499), n=70/n=80/rowp (seeds 0-499 each)
# That's 294 + 15 + 1500 = 1809 starts

# Build a work list: (farm, seed) pairs that need computing
cat > /tmp/worklist.txt << 'EOF'
EOF

python3 << 'PYEOF'
import os, glob
todo = []
# n=30: 206-499
for s in range(206, 500):
    p = f"lumi/logs/baselines/problem_dei_n30_{s}.out"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        todo.append(("results/problem_dei_n30.json", s))
# n=60: 485-499
for s in range(485, 500):
    p = f"lumi/logs/baselines/problem_dei_n60_{s}.out"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        todo.append(("results/problem_dei_n60.json", s))
# n=70, n=80, rowp: 0-499
for problem in ["results/problem_dei_n70.json", "results/problem_dei_n80.json", "results/problem_rowp.json"]:
    farm = os.path.basename(problem).replace(".json", "")
    for s in range(0, 500):
        p = f"lumi/logs/baselines/{farm}_{s}.out"
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            todo.append((problem, s))

with open("/tmp/worklist.txt", "w") as f:
    for problem, seed in todo:
        f.write(f"{problem} {seed}\n")
print(f"Worklist: {len(todo)} starts")
PYEOF

TOTAL=$(wc -l < /tmp/worklist.txt)
echo "Total work: $TOTAL starts across 8 GCDs"

# Launch 8 workers, each handling every 8th line
for gcd in 0 1 2 3 4 5 6 7; do
    (
        export ROCR_VISIBLE_DEVICES=$gcd
        export HIP_VISIBLE_DEVICES=$gcd
        awk -v g=$gcd 'NR % 8 == g' /tmp/worklist.txt | while read problem seed; do
            farm=$(basename $problem .json)
            out="lumi/logs/baselines/${farm}_${seed}.out"
            if [ ! -s "$out" ]; then
                pixi run python tools/run_single_baseline.py "$problem" --seed "$seed" > "$out" 2>/dev/null
            fi
        done
        echo "GCD $gcd done"
    ) &
done
wait

echo "All GCDs complete"
