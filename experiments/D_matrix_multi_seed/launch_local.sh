#!/usr/bin/env bash
# Local serial launch (one process). Suitable for verification or small
# n_seeds.
#
# Usage:
#   bash experiments/D_matrix_multi_seed/launch_local.sh           # default 3 seeds
#   N_SEEDS=5 bash experiments/D_matrix_multi_seed/launch_local.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

N_SEEDS=${N_SEEDS:-3}

pixi run python experiments/D_matrix_multi_seed/eval.py --n-seeds "$N_SEEDS"
