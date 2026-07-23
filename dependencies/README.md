# Dependencies

## pixwake (NOT vendored — private)

pixwake (the JAX wake-physics engine that computes AEP for every layout —
used by the skeleton, the benchmark, and all generated schedules) is
**intentionally not part of this repository** and must not be committed:
`dependencies/pixwake/` is gitignored. Its source lives in the private
cluster-tradeoffs repository.

- **Source:** https://github.com/kilojoules/cluster-tradeoffs.git (private)
- **Commit:** `b8e905a` ("Add flat multi-objective formulation for design regret")

To obtain it, run `setup.sh`, which clones the source repo and places the
package at `dependencies/pixwake/` when it is missing. It is used on
`PYTHONPATH` (`dependencies/pixwake/src`), not pip-installed; code does
`from pixwake import ...`.

The unit-test path (`playground/test_optimizer.py`, `tools/run_tests.py`)
does not require pixwake — it runs on the open-source `py_wake` package via
`playground/pywake_adapter.py` / `playground/skeleton_pywake.py`.

## energy_island_10y_daily_av_wind.csv

The training wind resource (10 years of daily-averaged Energy Island wind),
tracked in-repo at `dependencies/energy_island_10y_daily_av_wind.csv`.
