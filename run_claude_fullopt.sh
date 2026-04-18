#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="claude_fullopt"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 32
#SBATCH --nodes=1
#SBATCH --exclusive

source ~/.bash_profile
export DISABLE_AUTOUPDATER=1

# Fix for Sophia's older glibc (2.17) - pixi packages need 2.28+
export CONDA_OVERRIDE_GLIBC=2.28

cd /work/users/juqu/funwake

# Ensure pixwake is cloned
if [ ! -d "playground/pixwake" ]; then
    mkdir -p playground
    git clone https://github.com/kilojoules/cluster-tradeoffs.git playground/pixwake
    cd playground/pixwake && git checkout b8e905a && cd ../..
fi
cat fullopt_prompt.txt | claude -p --model sonnet --verbose \
    --dangerously-skip-permissions
