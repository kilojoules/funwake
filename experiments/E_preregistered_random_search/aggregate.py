"""Aggregate pre-registered random-search results.

Reports best feasible ROWP, distribution shape (median, top-10), and
side-by-side with LLM single-run numbers and the post-hoc
random-search number.

Usage:
    pixi run python experiments/E_preregistered_random_search/aggregate.py
"""
import json
import os
import statistics


HERE = os.path.dirname(os.path.abspath(__file__))


# Locked reference numbers from the paper at preregistration time.
LLM_REF = {
    "claude_iter_192_rowp": 4271.5,
    "gemini_iter_118_rowp": 4269.3,
    "post_hoc_random_search_320_rowp": 4268.6,
    "baseline_500_start_rowp": 4246.7,
}


def main():
    res = json.load(open(os.path.join(HERE, "results.json")))
    feas = [r for r in res.values()
            if r.get("rowp_feasible") and r.get("rowp_aep") is not None]
    rowp = sorted([r["rowp_aep"] for r in feas], reverse=True)

    summary = {
        "n_samples": len(res),
        "n_train_feasible": sum(1 for r in res.values() if r.get("train_feasible")),
        "n_rowp_feasible": len(feas),
        "rowp_best": rowp[0] if rowp else None,
        "rowp_top10_mean": (round(statistics.mean(rowp[:10]), 3) if len(rowp) >= 10 else None),
        "rowp_median": (round(statistics.median(rowp), 3) if rowp else None),
        "rowp_distribution_quantiles": (
            {q: round(rowp[int((1 - q) * len(rowp))], 3) for q in (0.99, 0.95, 0.9, 0.5)}
            if len(rowp) >= 10 else None
        ),
        "comparison": LLM_REF,
    }
    if rowp:
        # Pre-registered success criterion check
        gap_to_llm_best = LLM_REF["claude_iter_192_rowp"] - rowp[0]
        summary["gap_to_llm_best_gwh"] = round(gap_to_llm_best, 3)
        summary["preregistered_verdict"] = (
            "v2 matches LLM (parameterization is enough)"
            if gap_to_llm_best <= 5.0
            else "LLM structural prior matters"
            if gap_to_llm_best > 10.0
            else "ambiguous (5 < gap <= 10)"
        )

    out = os.path.join(HERE, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
