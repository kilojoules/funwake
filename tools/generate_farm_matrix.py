"""Generate the 48-cell farm × N × wind-rose matrix of problem JSONs.

Axes:
  polygon/turbine:   {DEI (farm 1, IEA 15 MW), ROWP (IEA 10 MW)}
  turbine count N:   {30, 40, 50, 60, 70, 80}
  wind rose:         {uniform (1 dir), omnidir (24 equal dirs),
                       DEI rose (observed 24 sectors),
                       ROWP rose (Weibull 12 sectors)}

Uniform and omnidirectional both reuse DEI's 24 speeds so the SPEED
distribution is shared across them (isolating the DIRECTION
distribution as the experimental variable). DEI rose and ROWP rose
are the originals from the respective base problem files.

Output: results/matrix/problem_<farm>_n<N>_rose<rose>.json for all 48
combinations, plus a results/matrix/manifest.json listing every cell.
"""
import json
import os
import sys


FARMS = ["dei", "rowp"]
NS = [30, 40, 50, 60, 70, 80, 200, 300]
ROSES = ["uniform", "omnidir", "dei", "rowp"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RESULTS, "matrix")


def load(path):
    with open(os.path.join(RESULTS, path)) as f:
        return json.load(f)


def build_rose(rose_name: str, dei_rose: dict, rowp_rose: dict) -> dict:
    """Construct the requested wind rose using DEI's speed distribution
    as the shared reference for uniform and omnidir."""
    dei_speeds = list(dei_rose["speeds_ms"])
    dei_weights = list(dei_rose["weights"])

    if rose_name == "dei":
        # Original DEI rose unchanged
        return {
            "directions_deg": list(dei_rose["directions_deg"]),
            "speeds_ms": dei_speeds,
            "weights": dei_weights,
        }
    if rose_name == "rowp":
        # Original ROWP rose unchanged
        return {
            "directions_deg": list(rowp_rose["directions_deg"]),
            "speeds_ms": list(rowp_rose["speeds_ms"]),
            "weights": list(rowp_rose["weights"]),
        }
    if rose_name == "uniform":
        # All wind from one direction (0 degrees). Keep the DEI speed
        # distribution — each "sector" is the same direction but one of
        # DEI's 24 speed bins, with the DEI weight at that speed bin.
        n = len(dei_speeds)
        return {
            "directions_deg": [0.0] * n,
            "speeds_ms": dei_speeds,
            "weights": dei_weights,
        }
    if rose_name == "omnidir":
        # 24 directions spaced evenly across 360 degrees, each direction
        # assigned one of DEI's 24 speeds with DEI's weight at that
        # speed bin — but direction is equally weighted (1/24 each
        # direction-speed pair contributes both speed-weight and
        # direction-weight = 1/n). Equivalent: rotate DEI's 24 sectors
        # to equally-spaced directions while preserving the speed array.
        n = len(dei_speeds)
        eq_dirs = [i * (360.0 / n) for i in range(n)]
        return {
            "directions_deg": eq_dirs,
            "speeds_ms": dei_speeds,
            "weights": dei_weights,
        }
    raise ValueError(f"unknown rose: {rose_name}")


def write_cell(farm: str, n: int, rose_name: str,
               dei_base: dict, rowp_base: dict,
               dei_rose: dict, rowp_rose: dict) -> str:
    """Write one problem JSON. Returns its relative path."""
    base = dei_base if farm == "dei" else rowp_base
    rose = build_rose(rose_name, dei_rose, rowp_rose)

    out = {
        "farm_id": f"{farm}_n{n}_rose_{rose_name}",
        "farm_name": f"{farm.upper()} polygon, N={n}, {rose_name} rose",
        "n_target": int(n),
        "rotor_diameter": float(base["rotor_diameter"]),
        "hub_height": float(base["hub_height"]),
        "min_spacing_m": float(base["min_spacing_m"]),
        "boundary_vertices": base["boundary_vertices"],
        "turbine": base["turbine"],
        "wind_rose": rose,
        "source_polygon": (
            base.get("source")
            or base.get("farm_name")
            or f"base:{farm}"
        ),
        "generated_by": "tools/generate_farm_matrix.py",
    }

    fname = f"problem_{farm}_n{n}_rose{rose_name}.json"
    path = os.path.join(OUT_DIR, fname)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return os.path.relpath(path, PROJECT_ROOT)


def main():
    # Load base problems
    dei_base = load("problem_dei_n50.json")  # polygon + turbine
    rowp_base = load("problem_rowp.json")
    dei_rose = dei_base["wind_rose"]
    rowp_rose = rowp_base["wind_rose"]

    manifest = {
        "axes": {"farms": FARMS, "ns": NS, "roses": ROSES},
        "cells": [],
    }

    for farm in FARMS:
        for n in NS:
            for rose in ROSES:
                rel = write_cell(farm, n, rose,
                                 dei_base, rowp_base,
                                 dei_rose, rowp_rose)
                manifest["cells"].append({
                    "farm": farm, "n": n, "rose": rose,
                    "path": rel,
                })

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest['cells'])} problem cells in {OUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
