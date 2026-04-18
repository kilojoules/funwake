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

cd /work/users/juqu/funwake
cat fullopt_prompt.txt | claude -p --model sonnet --verbose \
    --dangerously-skip-permissions
