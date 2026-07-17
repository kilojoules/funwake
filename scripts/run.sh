#!/bin/bash
#SBATCH --partition=windq
#SBATCH --job-name="tag"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 32
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc
#conEnv
#eval "$(pixi shell-hook)"

pixi run python experiments/run_zoo_sweep.py --max-parallel 10
