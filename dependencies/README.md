# Vendored dependencies

## pixwake

`dependencies/pixwake/` is a vendored copy of the **pixwake** package (the
JAX wake-physics engine that computes AEP for every layout — used by the
skeleton, the benchmark, and all generated schedules).

- **Source:** https://github.com/kilojoules/cluster-tradeoffs.git
- **Commit:** `b8e905a` ("Add flat multi-objective formulation for design regret")
- **What is included:** the `src/pixwake/` package (including its runtime
  `data/*.nc` lookup tables), the training wind resource
  `energy_island_10y_daily_av_wind.csv`, and `pyproject.toml` for reference.
  The rest of the upstream repo (precomputed layout caches, sweep scripts)
  is intentionally not vendored.

It is used on `PYTHONPATH` (`dependencies/pixwake/src`), not pip-installed;
code does `from pixwake import ...`.

### Updating

Re-copy `src/`, the wind CSV, and `pyproject.toml` from a fresh checkout of
the source repo at the desired commit, and update the commit hash above.
`setup.sh` will re-clone into this location only if the vendored copy is
missing.
