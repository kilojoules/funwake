"""Build a synthetic L-shaped held-out polygon problem JSON.

Inherits turbine model + wind rose from ROWP (so the difference vs ROWP
is purely geometric). Polygon is non-convex to stress the boundary
penalty.

Usage:
    pixi run python experiments/F_extra_heldout/build_polygon.py \
        --output results/problem_lshape.json
"""
import argparse
import json
import os
import math


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def lshape_vertices(side_a=18000.0, side_b=12000.0, cut=6000.0):
    """L-shape: outer rectangle of side_a x side_b with a cut in upper-right.

    Coordinates centered around (0,0) for stability.
    """
    half_a = side_a / 2
    half_b = side_b / 2
    return [
        [-half_a, -half_b],
        [ half_a, -half_b],
        [ half_a,  half_b - cut],
        [ half_a - cut,  half_b - cut],
        [ half_a - cut,  half_b],
        [-half_a,  half_b],
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--source", default=os.path.join(PROJECT_ROOT, "results", "problem_rowp.json"),
                   help="problem JSON to inherit turbine + wind from")
    p.add_argument("--n-turbines", type=int, default=60)
    p.add_argument("--side-a", type=float, default=18000.0)
    p.add_argument("--side-b", type=float, default=12000.0)
    p.add_argument("--cut", type=float, default=6000.0)
    args = p.parse_args()

    src = json.load(open(args.source))
    out = dict(src)
    # Use the same schema keys as ROWP/DEI problem JSONs.
    out["boundary_vertices"] = lshape_vertices(args.side_a, args.side_b, args.cut)
    out["n_target"] = args.n_turbines
    # min_spacing_m is set on the source already; preserve it.
    out["farm_id"] = "lshape"
    out["farm_name"] = "L-shape (synthetic)"
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    # quick area sanity-check
    A = args.side_a * args.side_b - args.cut * args.cut
    print(f"[F] Wrote {args.output}")
    print(f"[F] L-shape area: {A/1e6:.1f} km^2, {args.n_turbines} turbines, "
          f"density {args.n_turbines / (A/1e6):.2f}/km^2")


if __name__ == "__main__":
    main()
