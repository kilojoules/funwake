"""Strategy taxonomy + classifier for the agent memory registry.

Two taxonomies — one per interface mode:
  - schedule-only: families of schedule_fn(step, total, lr0, alpha0)
  - full-optimizer: families of optimize(sim, n_target, ...) implementations

Classification is source-level regex matching. A single script can match
multiple families (e.g. "cosine + Gaussian bumps + coupled alpha"); the
classifier returns the full set it recognizes. Unknown scripts return
{"uncategorized"}.

The registry uses this taxonomy to compute UNEXPLORED = TAXONOMY \\ TRIED
and project a "mandatory next action" hint into agent_memory.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str
    description: str
    patterns: tuple[str, ...]  # regex patterns that, if ANY match, classify into this family

    def matches(self, source: str) -> bool:
        return any(re.search(p, source, re.IGNORECASE | re.MULTILINE)
                   for p in self.patterns)


# ── Schedule-only taxonomy ─────────────────────────────────────────────
#
# A schedule_fn returns (lr, alpha, beta1, beta2). Families classify the
# SHAPE of lr(t) and alpha(t), and Adam momentum style.

SCHEDULE_FAMILIES: tuple[Family, ...] = (
    Family(
        "lr_cosine",
        "Cosine LR decay (vanilla, no restarts, no perturbation)",
        (r"jnp\.cos\s*\(\s*jnp\.pi\s*\*",),
    ),
    Family(
        "lr_sgdr_warm_restarts",
        "Stochastic Gradient Descent with Warm Restarts (Loshchilov 2017)",
        (r"n_cycles?\s*=", r"warm[_ ]?restart", r"sgdr", r"t_?(?:i|cur)\s*%\s*", r"jnp\.mod\(.*t"),
    ),
    Family(
        "lr_cyclical_triangular",
        "Smith 2017 triangular cyclical LR",
        (r"triangular", r"cyclical", r"cyclic[_ ]?lr"),
    ),
    Family(
        "lr_one_cycle",
        "Smith/Howard one-cycle policy (ramp up then down)",
        (r"one[_ ]?cycle", r"onecycle"),
    ),
    Family(
        "lr_linear_decay",
        "Linear LR decay",
        (r"lr\s*=\s*lr0\s*\*\s*\(\s*1\.?0?\s*-\s*t", r"linear[_ ]?decay"),
    ),
    Family(
        "lr_exponential_decay",
        "Exponential LR decay",
        (r"jnp\.exp\s*\(\s*-[\d\.]+\s*\*\s*t", r"exp_?decay"),
    ),
    Family(
        "lr_polynomial_decay",
        "Polynomial LR decay (power > 1)",
        (r"\(\s*1\.?0?\s*-\s*t\s*\)\s*\*\*\s*[23456]", r"polynomial[_ ]?decay"),
    ),
    Family(
        "lr_gaussian_bumps",
        "Gaussian LR bumps — controlled escapes at specific t values (Claude iter_192 family)",
        (r"jnp\.exp\s*\(\s*-\s*0?\.5\s*\*\s*\(\s*\(?\s*t\s*-\s*[\d\.]+",
         r"bump_amp", r"gaussian_bump"),
    ),
    Family(
        "lr_sinusoidal_shake",
        "High-frequency decaying sinusoidal LR perturbation (Gemini iter_067 family)",
        (r"jnp\.sin\s*\(\s*[\d\.]+\s*\*\s*jnp\.pi\s*\*\s*t",
         r"sinusoidal", r"shake"),
    ),
    Family(
        "lr_noise_injection",
        "Stochastic LR noise (uniform or Gaussian) — Jastrzębski 2018 style",
        (r"jax\.random\.normal", r"jax\.random\.uniform.*lr",
         r"lr\s*\+=.*random"),
    ),
    Family(
        "lr_constant",
        "Constant LR (no decay)",
        (r"lr\s*=\s*lr0\s*\n", r"return\s*\(?\s*lr0\s*,"),
    ),
    Family(
        "alpha_coupled_inverse_lr",
        "Penalty weight coupled to 1/lr (TopFarm-style: alpha ∝ 1/eta)",
        (r"alpha\s*=.*alpha0\s*\*.*lr0\s*/\s*lr",),
    ),
    Family(
        "alpha_quadratic_ramp",
        "Quadratic alpha ramp (Gemini-style: alpha ∝ 1 + M*t²)",
        (r"alpha\s*=.*alpha0\s*\*.*\(\s*1\.?0?\s*\+\s*[\d\.]+\s*\*\s*t\s*\*\s*t",
         r"alpha.*\*\s*t\s*\*\*\s*2"),
    ),
    Family(
        "alpha_linear_ramp",
        "Linear alpha ramp",
        (r"alpha\s*=.*\*\s*\(\s*1\.?0?\s*\+\s*[\d\.]+\s*\*\s*t\s*\)",),
    ),
    Family(
        "alpha_cyclic",
        "Cyclic alpha (high → low → high) — anti-phase with lr cycles",
        (r"alpha.*cycle", r"alpha.*cos", r"alpha.*sin"),
    ),
    Family(
        "alpha_anti_phase_dip",
        "Penalty dip paired with LR bump (Claude's alpha-dip-at-bump mechanism)",
        (r"alpha.*\(.*1\.?0?\s*-\s*0?\.?\d+\s*\*\s*jnp\.exp\s*\(\s*-",),
    ),
    Family(
        "adam_standard",
        "Standard Adam (beta1≈0.9, beta2≈0.999)",
        (r"return\s*\(?\s*lr\s*,\s*alpha\s*,\s*0\.9\s*,\s*0\.999",
         r"beta1\s*=\s*0\.9\b", r"beta2\s*=\s*0\.999\b"),
    ),
    Family(
        "adam_topfarm_low",
        "TopFarm-style low-momentum Adam (beta1≈0.1, beta2≈0.2)",
        (r"beta1\s*=\s*0\.1\b.*beta2\s*=\s*0\.2\b",
         r"return\s*\(?\s*lr\s*,\s*alpha\s*,\s*0\.1\s*,\s*0\.2"),
    ),
    Family(
        "adam_zero_momentum",
        "Zero first-moment (beta1=0) — Gemini-style, pure RMSProp-like",
        (r"beta1\s*=\s*0\.?0?\b",
         r"return\s*\(?\s*lr\s*,\s*alpha\s*,\s*0\.?0?\s*,"),
    ),
    Family(
        "adam_high_beta2",
        "High beta2 (>0.99) — long-horizon second moment",
        (r"beta2\s*=\s*0\.99[5-9]", r"beta2\s*=\s*0\.9999"),
    ),
    Family(
        "schedule_two_phase",
        "Two-phase schedule (explore-then-refine with qualitative switch)",
        (r"jnp\.where\s*\(\s*t\s*<\s*0?\.[345]",
         r"if.*step.*<.*total_steps\s*//"),
    ),
    Family(
        "warmup",
        "Learning rate warmup from zero",
        (r"warmup", r"t\s*<\s*warmup_frac", r"lr\s*\*\s*jnp\.minimum\(1\.?0?,\s*t"),
    ),
)


# ── Full-optimizer taxonomy ────────────────────────────────────────────
#
# An optimize() function runs its own search. Families classify the TYPE
# of outer search, the initialization strategy, and the constraint handling.

FULLOPT_FAMILIES: tuple[Family, ...] = (
    Family(
        "topfarm_sgd_solve",
        "Use topfarm_sgd_solve as the optimizer (skeleton wrapper)",
        (r"topfarm_sgd_solve",),
    ),
    Family(
        "custom_adam",
        "Hand-rolled Adam / SGD loop with jax.grad",
        (r"jax\.grad", r"@jax\.jit.*adam", r"def\s+step\b.*mx\s*=.*b1"),
    ),
    Family(
        "scipy_slsqp",
        "scipy.optimize.minimize with SLSQP — explicit constraint Jacobians",
        (r"method\s*=\s*['\"]SLSQP['\"]", r"scipy.*SLSQP"),
    ),
    Family(
        "scipy_lbfgs",
        "scipy L-BFGS-B (penalty method for constraints)",
        (r"method\s*=\s*['\"]L-BFGS-B['\"]", r"L-BFGS", r"lbfgs"),
    ),
    Family(
        "scipy_trust_constr",
        "scipy trust-constr (interior-point constrained)",
        (r"trust[-_]constr",),
    ),
    Family(
        "scipy_differential_evolution",
        "scipy differential evolution — population-based global search",
        (r"differential_evolution",),
    ),
    Family(
        "scipy_basin_hopping",
        "scipy basin hopping — random restart over local minima",
        (r"basin[_]?hopping",),
    ),
    Family(
        "scipy_shgo",
        "scipy SHGO — simplicial homology global optimization",
        (r"\bshgo\b",),
    ),
    Family(
        "cmaes",
        "CMA-ES — covariance matrix adaptation evolution strategy",
        (r"\bcma\.", r"CMAEvolutionStrategy", r"cmaes"),
    ),
    Family(
        "pso",
        "Particle swarm optimization",
        (r"\bpso\b", r"particle[_ ]?swarm"),
    ),
    Family(
        "simulated_annealing",
        "Simulated annealing with constraint projection",
        (r"dual_annealing", r"simulated_annealing", r"\bSA\b.*temperature"),
    ),
    Family(
        "bayesian_optimization",
        "Bayesian optimization (GP surrogate) — skopt, optuna, bayesian-optimization",
        (r"skopt", r"optuna", r"BayesianOptimization", r"GPyOpt"),
    ),
    Family(
        "genetic_algorithm",
        "Explicit GA / evolutionary with crossover+mutation (not DE, not CMA)",
        (r"deap", r"crossover", r"mutation.*population", r"genetic"),
    ),
    Family(
        "nesterov_momentum",
        "Nesterov accelerated gradient",
        (r"nesterov",),
    ),
    Family(
        "init_hexagonal",
        "Hexagonal lattice initialization",
        (r"hexagonal", r"hex[_ ]?lattice", r"hex[_ ]?packing"),
    ),
    Family(
        "init_kmeans",
        "K-means clustering of wind sectors for init",
        (r"kmeans", r"KMeans"),
    ),
    Family(
        "init_wind_aware_grid",
        "Grid initialization rotated perpendicular to dominant wind direction",
        (r"wind[_ ]?aware", r"jnp\.arctan2.*weights.*sin",
         r"dominant.*wind"),
    ),
    Family(
        "init_random_restarts",
        "Random-restart multistart (N random inits, keep best)",
        (r"random.*restart", r"multistart", r"multi[_ ]?start"),
    ),
    Family(
        "init_latin_hypercube",
        "Latin hypercube / quasi-random initialization",
        (r"latin[_ ]?hypercube", r"qmc\.", r"sobol"),
    ),
    Family(
        "constraint_augmented_lagrangian",
        "Augmented Lagrangian for constraints (not pure penalty)",
        (r"augmented[_ ]?lagrang", r"lambda_k\s*\+="),
    ),
    Family(
        "constraint_projection",
        "Explicit feasibility projection step",
        (r"def\s+project", r"projection.*boundary"),
    ),
)


def classify(source: str, mode: str = "schedule") -> set[str]:
    """Return the set of families that match `source`.

    `mode` is 'schedule' or 'fullopt'. Unknown family returns
    {'uncategorized'} if no patterns matched.
    """
    families = SCHEDULE_FAMILIES if mode == "schedule" else FULLOPT_FAMILIES
    hits = {f.name for f in families if f.matches(source)}
    return hits or {"uncategorized"}


def family_by_name(name: str, mode: str = "schedule") -> Family | None:
    families = SCHEDULE_FAMILIES if mode == "schedule" else FULLOPT_FAMILIES
    for f in families:
        if f.name == name:
            return f
    return None


def all_family_names(mode: str = "schedule") -> list[str]:
    families = SCHEDULE_FAMILIES if mode == "schedule" else FULLOPT_FAMILIES
    return [f.name for f in families]


def describe_family(name: str, mode: str = "schedule") -> str:
    f = family_by_name(name, mode)
    return f"{name}: {f.description}" if f else name
