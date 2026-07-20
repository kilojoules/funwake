"""Gradient-based adversarial search using implicit differentiation.

This module enables finding neighbor configurations that maximize design regret
for a target wind farm using gradient-based optimization. The key insight is
that we can differentiate through the target farm's layout optimization using
the Implicit Function Theorem, avoiding the need to unroll thousands of SGD
iterations.

The bilevel optimization structure:
    - Outer loop: Gradient descent on neighbor positions to maximize regret
    - Inner loop: Target farm optimizes layout via SGD (differentiable via IFT)

Example:
    >>> searcher = GradientAdversarialSearch(sim, target_boundary, min_spacing, ws, wd)
    >>> result = searcher.search(init_target_x, init_target_y, init_neighbor_x, init_neighbor_y)
    >>> print(f"Max regret: {result.regret:.2f} GWh")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.lax import while_loop

from pixwake.optim.sgd import (
    SGDSettings,
    sgd_solve_implicit,
    topfarm_sgd_solve,
)


class AdversarialSearchResult(NamedTuple):
    """Result of gradient-based adversarial search.

    Attributes:
        neighbor_x: Optimized neighbor x positions.
        neighbor_y: Optimized neighbor y positions.
        target_x: Target farm layout after optimization with adversarial neighbors.
        target_y: Target farm layout after optimization with adversarial neighbors.
        liberal_x: Target farm layout without neighbors (liberal case).
        liberal_y: Target farm layout without neighbors (liberal case).
        regret: Design regret (conservative_aep - liberal_aep_present).
            The liberal layout is optimized in isolation, then evaluated with
            neighbors present. Positive regret = conservative design is better.
        liberal_aep: AEP of liberal layout without neighbors.
        conservative_aep: AEP of conservative layout with adversarial neighbors.
        history: List of (regret, neighbor_x, neighbor_y) tuples per iteration.
    """

    neighbor_x: jnp.ndarray
    neighbor_y: jnp.ndarray
    target_x: jnp.ndarray
    target_y: jnp.ndarray
    liberal_x: jnp.ndarray
    liberal_y: jnp.ndarray
    regret: float
    liberal_aep: float
    conservative_aep: float
    history: list


@dataclass
class AdversarialSearchSettings:
    """Settings for gradient-based adversarial search.

    Attributes:
        max_iter: Maximum outer loop iterations (default: 100).
        learning_rate: Learning rate for neighbor position updates (default: 10.0).
        tol: Convergence tolerance on regret change (default: 1e-4).
        neighbor_boundary: Boundary polygon for neighbor positions (optional).
        neighbor_min_spacing: Minimum spacing between neighbor turbines (optional).
        sgd_settings: Settings for inner SGD optimization.
        verbose: Print progress during optimization (default: True).
    """

    max_iter: int = 100
    learning_rate: float = 10.0
    tol: float = 1e-4
    neighbor_boundary: jnp.ndarray | None = None
    neighbor_min_spacing: float | None = None
    target_buffer: float | None = None
    sgd_settings: SGDSettings | None = None
    verbose: bool = True


class GradientAdversarialSearch:
    """Gradient-based adversarial neighbor search using implicit differentiation.

    This class finds neighbor turbine configurations that maximize regret for
    a target wind farm. It uses JAX automatic differentiation through the
    target farm's layout optimization via the Implicit Function Theorem.

    The regret is defined as:
        regret = AEP_conservative(w/ neighbors) - AEP_liberal(w/ neighbors)

    where:
        - AEP_liberal: Target farm AEP when optimized without neighbors
        - AEP_conservative: Target farm AEP when optimized with neighbors present

    Parameters
    ----------
    sim : WakeSimulation
        Wake simulation engine.
    target_boundary : jnp.ndarray
        Target farm boundary polygon, shape (n_vertices, 2).
    target_min_spacing : float
        Minimum spacing between target turbines.
    ws_amb : jnp.ndarray
        Ambient wind speeds.
    wd_amb : jnp.ndarray
        Wind directions.
    ti_amb : jnp.ndarray | None
        Ambient turbulence intensity (optional).
    weights : jnp.ndarray | None
        Probability weights for wind conditions (optional).

    Example
    -------
    >>> from pixwake import WakeSimulation, Turbine
    >>> from pixwake.deficit import NOJDeficit
    >>> from pixwake.optim.adversarial import GradientAdversarialSearch
    >>>
    >>> sim = WakeSimulation(turbine, NOJDeficit())
    >>> searcher = GradientAdversarialSearch(
    ...     sim, target_boundary, min_spacing=160.0,
    ...     ws_amb=jnp.array([10.0]), wd_amb=jnp.array([270.0])
    ... )
    >>> result = searcher.search(init_x, init_y, neighbor_x, neighbor_y)
    """

    def __init__(
        self,
        sim,
        target_boundary: jnp.ndarray,
        target_min_spacing: float,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ):
        self.sim = sim
        self.target_boundary = target_boundary
        self.target_min_spacing = target_min_spacing
        self.ws_amb = ws_amb
        self.wd_amb = wd_amb
        self.ti_amb = ti_amb
        self.weights = weights

    def _compute_aep(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_x: jnp.ndarray | None = None,
        neighbor_y: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute AEP for target turbines, optionally with neighbors."""
        n_target = target_x.shape[0]

        if neighbor_x is not None and neighbor_y is not None:
            x_all = jnp.concatenate([target_x, neighbor_x])
            y_all = jnp.concatenate([target_y, neighbor_y])
        else:
            x_all = target_x
            y_all = target_y

        result = self.sim(
            x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )

        # Only count target turbines' AEP
        power = result.power()[:, :n_target]  # (n_cases, n_target)

        if self.weights is not None:
            weighted_power = jnp.sum(power * self.weights[:, None])
            return weighted_power * 8760 / 1e6  # GWh
        return jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    def search(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        init_neighbor_x: jnp.ndarray,
        init_neighbor_y: jnp.ndarray,
        settings: AdversarialSearchSettings | None = None,
    ) -> AdversarialSearchResult:
        """Run gradient-based adversarial search.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Initial target farm turbine positions.
        init_neighbor_x, init_neighbor_y : jnp.ndarray
            Initial neighbor turbine positions.
        settings : AdversarialSearchSettings | None
            Search settings. Uses defaults if None.

        Returns
        -------
        AdversarialSearchResult
            Search results including optimized positions and regret.
        """
        if settings is None:
            settings = AdversarialSearchSettings()

        sgd_settings = settings.sgd_settings or SGDSettings()

        # Define objective function for inner SGD (with neighbor params)
        def objective_with_neighbors(
            x: jnp.ndarray, y: jnp.ndarray, neighbor_params: jnp.ndarray
        ) -> jnp.ndarray:
            n_neighbors = neighbor_params.shape[0] // 2
            neighbor_x = neighbor_params[:n_neighbors]
            neighbor_y = neighbor_params[n_neighbors:]
            return -self._compute_aep(x, y, neighbor_x, neighbor_y)

        # Compute liberal layout (no neighbors) - only once
        def liberal_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return -self._compute_aep(x, y)

        from pixwake.optim.sgd import topfarm_sgd_solve

        liberal_x, liberal_y = topfarm_sgd_solve(
            liberal_objective,
            init_target_x,
            init_target_y,
            self.target_boundary,
            self.target_min_spacing,
            sgd_settings,
        )
        liberal_aep = float(self._compute_aep(liberal_x, liberal_y))

        if settings.verbose:
            print(f"Liberal AEP (no neighbors): {liberal_aep:.2f} GWh")

        # Define regret function (to maximize)
        def compute_regret(neighbor_params: jnp.ndarray) -> jnp.ndarray:
            """Compute regret = AEP_conservative(w/ neighbors) - AEP_liberal(w/ neighbors).

            The liberal layout was optimized in isolation (maximizing AEP without
            neighbors), but is evaluated here WITH neighbors present to measure
            the design regret — the AEP benefit of having designed for neighbors.
            """
            n_neighbors = neighbor_params.shape[0] // 2
            neighbor_x = neighbor_params[:n_neighbors]
            neighbor_y = neighbor_params[n_neighbors:]

            # Liberal layout (optimized in isolation) evaluated WITH neighbors
            # This is differentiable w.r.t. neighbor_params
            liberal_aep_present = self._compute_aep(liberal_x, liberal_y, neighbor_x, neighbor_y)

            # Optimize target layout given neighbors (differentiable via IFT)
            opt_x, opt_y = sgd_solve_implicit(
                objective_with_neighbors,
                init_target_x,
                init_target_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
                neighbor_params,
            )

            # Conservative AEP: layout optimized WITH knowledge of neighbors
            conservative_aep = self._compute_aep(opt_x, opt_y, neighbor_x, neighbor_y)

            # Regret = benefit of designing with neighbor knowledge (always >= 0)
            return conservative_aep - liberal_aep_present

        # Gradient ascent on neighbor positions to maximize regret (ADAM optimizer)
        neighbor_params = jnp.concatenate([init_neighbor_x, init_neighbor_y])
        history = []

        regret_and_grad = value_and_grad(compute_regret)

        # ADAM state
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        m = jnp.zeros_like(neighbor_params)
        v = jnp.zeros_like(neighbor_params)
        best_regret = -jnp.inf
        best_neighbor_params = neighbor_params

        for i in range(settings.max_iter):
            regret, grad = regret_and_grad(neighbor_params)

            # NaN guard: break early if gradient is NaN
            if not jnp.all(jnp.isfinite(grad)):
                if settings.verbose:
                    print(f"Iter {i}: NaN gradient detected, stopping early")
                break

            history.append(
                (
                    float(regret),
                    neighbor_params[: len(init_neighbor_x)].copy(),
                    neighbor_params[len(init_neighbor_x) :].copy(),
                )
            )

            # Track best result
            if float(regret) > float(best_regret):
                best_regret = regret
                best_neighbor_params = neighbor_params

            grad_norm = float(jnp.linalg.norm(grad))
            if settings.verbose and i % 10 == 0:
                print(
                    f"Iter {i}: regret = {regret:.4f} GWh, "
                    f"|grad| = {grad_norm:.6f}"
                )

            # Check convergence
            if i > 0 and abs(history[-1][0] - history[-2][0]) < settings.tol:
                if settings.verbose:
                    print(f"Converged at iteration {i}")
                break

            # ADAM update (gradient ascent: maximize regret)
            t = i + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            neighbor_params = neighbor_params + settings.learning_rate * m_hat / (jnp.sqrt(v_hat) + adam_eps)

            # Apply neighbor boundary constraints if specified
            if settings.neighbor_boundary is not None:
                # Project back into boundary (simple clipping for now)
                n_neighbors = len(init_neighbor_x)
                neighbor_x = neighbor_params[:n_neighbors]
                neighbor_y = neighbor_params[n_neighbors:]

                x_min = settings.neighbor_boundary[:, 0].min()
                x_max = settings.neighbor_boundary[:, 0].max()
                y_min = settings.neighbor_boundary[:, 1].min()
                y_max = settings.neighbor_boundary[:, 1].max()

                neighbor_x = jnp.clip(neighbor_x, x_min, x_max)
                neighbor_y = jnp.clip(neighbor_y, y_min, y_max)
                neighbor_params = jnp.concatenate([neighbor_x, neighbor_y])

            # Enforce minimum buffer from target boundary
            if settings.target_buffer is not None:
                n_neighbors = len(init_neighbor_x)
                nb_x = neighbor_params[:n_neighbors]
                nb_y = neighbor_params[n_neighbors:]
                tb = self.target_boundary
                # Nearest point on target boundary box to each neighbor
                cx = jnp.clip(nb_x, tb[:, 0].min(), tb[:, 0].max())
                cy = jnp.clip(nb_y, tb[:, 1].min(), tb[:, 1].max())
                dx = nb_x - cx
                dy = nb_y - cy
                dist = jnp.sqrt(dx**2 + dy**2)
                buf = settings.target_buffer
                scale = jnp.where(dist < 1e-6, 1.0, buf / dist)
                nb_x = jnp.where(dist < buf, cx + dx * scale, nb_x)
                nb_y = jnp.where(dist < buf, cy + dy * scale, nb_y)
                neighbor_params = jnp.concatenate([nb_x, nb_y])

        # Use best params found (in case NaN terminated early)
        neighbor_params = best_neighbor_params

        # Final evaluation
        n_neighbors = len(init_neighbor_x)
        final_neighbor_x = neighbor_params[:n_neighbors]
        final_neighbor_y = neighbor_params[n_neighbors:]

        # Get final optimized target layout
        final_target_x, final_target_y = sgd_solve_implicit(
            objective_with_neighbors,
            init_target_x,
            init_target_y,
            self.target_boundary,
            self.target_min_spacing,
            sgd_settings,
            neighbor_params,
        )

        final_conservative_aep = float(
            self._compute_aep(final_target_x, final_target_y, final_neighbor_x, final_neighbor_y)
        )
        final_liberal_aep_present = float(
            self._compute_aep(liberal_x, liberal_y, final_neighbor_x, final_neighbor_y)
        )
        final_regret = final_conservative_aep - final_liberal_aep_present

        if settings.verbose:
            print(f"\nFinal Results:")
            print(f"  Liberal AEP (isolated): {liberal_aep:.2f} GWh")
            print(f"  Liberal AEP (w/ neighbors): {final_liberal_aep_present:.2f} GWh")
            print(f"  Conservative AEP (w/ neighbors): {final_conservative_aep:.2f} GWh")
            print(f"  Regret: {final_regret:.4f} GWh")

        return AdversarialSearchResult(
            neighbor_x=final_neighbor_x,
            neighbor_y=final_neighbor_y,
            target_x=final_target_x,
            target_y=final_target_y,
            liberal_x=liberal_x,
            liberal_y=liberal_y,
            regret=final_regret,
            liberal_aep=liberal_aep,
            conservative_aep=final_conservative_aep,
            history=history,
        )

    def _build_regret_fn(
        self,
        liberal_x: jnp.ndarray,
        liberal_y: jnp.ndarray,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        sgd_settings: SGDSettings,
        use_multistart: bool = False,
        init_x_batch: jnp.ndarray | None = None,
        init_y_batch: jnp.ndarray | None = None,
    ):
        """Build a pure-JAX regret function for use in while_loop.

        Returns a function ``regret_and_grad(neighbor_params) -> (regret, grad)``.
        """
        if use_multistart and init_x_batch is not None:
            from pixwake.optim.sgd import sgd_solve_implicit_multistart

            def objective_with_neighbors(x, y, neighbor_params):
                n_nb = neighbor_params.shape[0] // 2
                nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
                return -self._compute_aep(x, y, nb_x, nb_y)

            def compute_regret(neighbor_params):
                n_nb = neighbor_params.shape[0] // 2
                nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
                liberal_aep_present = self._compute_aep(
                    liberal_x, liberal_y, nb_x, nb_y
                )
                opt_x, opt_y = sgd_solve_implicit_multistart(
                    objective_with_neighbors,
                    init_x_batch, init_y_batch, neighbor_params,
                    self.target_boundary, self.target_min_spacing, sgd_settings,
                )
                conservative_aep = self._compute_aep(opt_x, opt_y, nb_x, nb_y)
                return conservative_aep - liberal_aep_present
        else:
            def objective_with_neighbors(x, y, neighbor_params):
                n_nb = neighbor_params.shape[0] // 2
                nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
                return -self._compute_aep(x, y, nb_x, nb_y)

            def compute_regret(neighbor_params):
                n_nb = neighbor_params.shape[0] // 2
                nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
                liberal_aep_present = self._compute_aep(
                    liberal_x, liberal_y, nb_x, nb_y
                )
                opt_x, opt_y = sgd_solve_implicit(
                    objective_with_neighbors,
                    init_target_x, init_target_y,
                    self.target_boundary, self.target_min_spacing, sgd_settings,
                    neighbor_params,
                )
                conservative_aep = self._compute_aep(opt_x, opt_y, nb_x, nb_y)
                return conservative_aep - liberal_aep_present

        return value_and_grad(compute_regret)

    def _search_single_jax(
        self,
        neighbor_params: jnp.ndarray,
        liberal_x: jnp.ndarray,
        liberal_y: jnp.ndarray,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        regret_and_grad_fn,
        max_iter: int,
        learning_rate: float,
        neighbor_clip_min: jnp.ndarray | None = None,
        neighbor_clip_max: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Pure JAX outer ADAM loop using jax.lax.while_loop.

        Returns (best_params, best_regret).
        """
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        n = neighbor_params.shape[0]

        m = jnp.zeros(n)
        v = jnp.zeros(n)
        best_params = neighbor_params
        best_regret = jnp.array(-jnp.inf)

        # State: (params, m, v, best_params, best_regret, iteration, converged)
        init_carry = (neighbor_params, m, v, best_params, best_regret, 0, False)

        def cond_fn(carry):
            _, _, _, _, _, it, converged = carry
            return jnp.logical_and(~converged, it < max_iter)

        def body_fn(carry):
            params, m, v, best_params, best_regret, it, converged = carry
            regret, grad = regret_and_grad_fn(params)

            # NaN guard
            is_finite = jnp.all(jnp.isfinite(grad))
            converged = jnp.logical_or(converged, ~is_finite)
            grad = jnp.where(is_finite, grad, jnp.zeros_like(grad))

            # ADAM update (gradient ascent: maximize regret)
            t = it + 1
            m_new = beta1 * m + (1 - beta1) * grad
            v_new = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m_new / (1 - beta1 ** t)
            v_hat = v_new / (1 - beta2 ** t)
            params_new = params + learning_rate * m_hat / (jnp.sqrt(v_hat) + adam_eps)

            # Boundary clipping if provided
            if neighbor_clip_min is not None and neighbor_clip_max is not None:
                params_new = jnp.clip(params_new, neighbor_clip_min, neighbor_clip_max)

            # Only update if not converged
            params = jnp.where(converged, params, params_new)
            m = jnp.where(converged, m, m_new)
            v = jnp.where(converged, v, v_new)

            # Track best
            is_better = jnp.logical_and(regret > best_regret, ~converged)
            best_params = jnp.where(is_better, params, best_params)
            best_regret = jnp.where(is_better, regret, best_regret)

            return (params, m, v, best_params, best_regret, t, converged)

        final = while_loop(cond_fn, body_fn, init_carry)
        _, _, _, best_params, best_regret, _, _ = final
        return best_params, best_regret

    def search_multistart(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        init_neighbor_x_batch: jnp.ndarray,
        init_neighbor_y_batch: jnp.ndarray,
        settings: AdversarialSearchSettings | None = None,
        inner_k: int = 1,
        inner_init_x_batch: jnp.ndarray | None = None,
        inner_init_y_batch: jnp.ndarray | None = None,
    ) -> AdversarialSearchResult:
        """Run M parallel outer trajectories, each with optional K inner starts.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Initial target farm layout, shape (n_turbines,).
        init_neighbor_x_batch : jnp.ndarray
            M initial neighbor x configs, shape (M, n_neighbors).
        init_neighbor_y_batch : jnp.ndarray
            M initial neighbor y configs, shape (M, n_neighbors).
        settings : AdversarialSearchSettings | None
            Search settings.
        inner_k : int
            Number of inner multistarts (default 1 = single start).
        inner_init_x_batch : jnp.ndarray | None
            K inner initial x layouts, shape (K, n_turbines). Required if inner_k > 1.
        inner_init_y_batch : jnp.ndarray | None
            K inner initial y layouts, shape (K, n_turbines). Required if inner_k > 1.

        Returns
        -------
        AdversarialSearchResult
            Best result across all M outer starts.
        """
        from jax.lax import while_loop as _wl

        if settings is None:
            settings = AdversarialSearchSettings()

        sgd_settings = settings.sgd_settings or SGDSettings()

        # Compute liberal layout (no neighbors) — shared across all starts
        def liberal_objective(x, y):
            return -self._compute_aep(x, y)

        from pixwake.optim.sgd import topfarm_sgd_solve

        liberal_x, liberal_y = topfarm_sgd_solve(
            liberal_objective, init_target_x, init_target_y,
            self.target_boundary, self.target_min_spacing, sgd_settings,
        )
        liberal_aep = float(self._compute_aep(liberal_x, liberal_y))

        if settings.verbose:
            print(f"Liberal AEP (no neighbors): {liberal_aep:.2f} GWh")

        # Build regret function
        use_multistart = inner_k > 1 and inner_init_x_batch is not None
        regret_and_grad_fn = self._build_regret_fn(
            liberal_x, liberal_y, init_target_x, init_target_y, sgd_settings,
            use_multistart=use_multistart,
            init_x_batch=inner_init_x_batch,
            init_y_batch=inner_init_y_batch,
        )

        # Build clip bounds from neighbor_boundary
        n_neighbors = init_neighbor_x_batch.shape[1]
        clip_min = clip_max = None
        if settings.neighbor_boundary is not None:
            x_min = float(settings.neighbor_boundary[:, 0].min())
            x_max = float(settings.neighbor_boundary[:, 0].max())
            y_min = float(settings.neighbor_boundary[:, 1].min())
            y_max = float(settings.neighbor_boundary[:, 1].max())
            clip_min = jnp.concatenate([
                jnp.full(n_neighbors, x_min),
                jnp.full(n_neighbors, y_min),
            ])
            clip_max = jnp.concatenate([
                jnp.full(n_neighbors, x_max),
                jnp.full(n_neighbors, y_max),
            ])

        M = init_neighbor_x_batch.shape[0]

        if settings.verbose:
            print(f"Running {M} outer starts (inner_k={inner_k})...")

        # Run M starts sequentially (vmap over outer starts would require
        # the entire regret_and_grad_fn to be vmappable, which it is NOT
        # because the wake sim has Python-level validation at trace time).
        # Instead, we run each start via _search_single_jax (while_loop).
        all_best_params = []
        all_best_regrets = []

        for m_idx in range(M):
            nb_params = jnp.concatenate([
                init_neighbor_x_batch[m_idx],
                init_neighbor_y_batch[m_idx],
            ])

            best_params, best_regret = self._search_single_jax(
                nb_params, liberal_x, liberal_y,
                init_target_x, init_target_y,
                regret_and_grad_fn,
                max_iter=settings.max_iter,
                learning_rate=settings.learning_rate,
                neighbor_clip_min=clip_min,
                neighbor_clip_max=clip_max,
            )

            all_best_params.append(best_params)
            all_best_regrets.append(float(best_regret))

            if settings.verbose:
                print(f"  Start {m_idx}: best regret = {float(best_regret):.4f} GWh")

        # Pick overall best
        best_m = int(np.argmax(all_best_regrets))
        winner_params = all_best_params[best_m]
        winner_regret = all_best_regrets[best_m]

        if settings.verbose:
            print(f"Best start: {best_m} with regret = {winner_regret:.4f} GWh")

        # Final evaluation with the winning neighbor config
        final_nb_x = winner_params[:n_neighbors]
        final_nb_y = winner_params[n_neighbors:]

        def objective_with_neighbors(x, y, neighbor_params):
            n_nb = neighbor_params.shape[0] // 2
            nb_x, nb_y = neighbor_params[:n_nb], neighbor_params[n_nb:]
            return -self._compute_aep(x, y, nb_x, nb_y)

        final_target_x, final_target_y = sgd_solve_implicit(
            objective_with_neighbors,
            init_target_x, init_target_y,
            self.target_boundary, self.target_min_spacing, sgd_settings,
            winner_params,
        )

        final_conservative_aep = float(
            self._compute_aep(final_target_x, final_target_y, final_nb_x, final_nb_y)
        )
        final_liberal_aep_present = float(
            self._compute_aep(liberal_x, liberal_y, final_nb_x, final_nb_y)
        )
        final_regret = final_conservative_aep - final_liberal_aep_present

        if settings.verbose:
            print(f"\nFinal Results (M={M}, K={inner_k}):")
            print(f"  Liberal AEP (isolated): {liberal_aep:.2f} GWh")
            print(f"  Liberal AEP (w/ neighbors): {final_liberal_aep_present:.2f} GWh")
            print(f"  Conservative AEP (w/ neighbors): {final_conservative_aep:.2f} GWh")
            print(f"  Regret: {final_regret:.4f} GWh")

        # Build history from all starts
        history = [(r, all_best_params[i][:n_neighbors], all_best_params[i][n_neighbors:])
                   for i, r in enumerate(all_best_regrets)]

        return AdversarialSearchResult(
            neighbor_x=final_nb_x,
            neighbor_y=final_nb_y,
            target_x=final_target_x,
            target_y=final_target_y,
            liberal_x=liberal_x,
            liberal_y=liberal_y,
            regret=final_regret,
            liberal_aep=liberal_aep,
            conservative_aep=final_conservative_aep,
            history=history,
        )


# =============================================================================
# Blob-Based Adversarial Discovery
# =============================================================================


class BlobDiscoveryResult(NamedTuple):
    """Result of blob-based adversarial discovery.

    Attributes:
        control_points: Final B-spline control points defining neighbor boundary.
        liberal_x: Liberal layout (optimized without neighbors).
        liberal_y: Liberal layout (optimized without neighbors).
        conservative_x: Conservative layout (optimized with neighbors).
        conservative_y: Conservative layout (optimized with neighbors).
        aep_L_absent: Liberal design AEP without neighbors.
        aep_L_present: Liberal design AEP with neighbors.
        aep_C_absent: Conservative design AEP without neighbors.
        aep_C_present: Conservative design AEP with neighbors.
        regret_liberal: AEP_C_present - AEP_L_present (benefit of conservative when neighbors appear).
        regret_conservative: AEP_L_absent - AEP_C_absent (cost of conservative when neighbors don't appear).
        history: List of (regret_liberal, control_points) tuples per iteration.
    """

    control_points: jnp.ndarray
    liberal_x: jnp.ndarray
    liberal_y: jnp.ndarray
    conservative_x: jnp.ndarray
    conservative_y: jnp.ndarray
    aep_L_absent: float
    aep_L_present: float
    aep_C_absent: float
    aep_C_present: float
    regret_liberal: float
    regret_conservative: float
    history: list


@dataclass
class BlobDiscoverySettings:
    """Settings for blob-based adversarial discovery.

    Attributes:
        max_iter: Maximum outer loop iterations.
        learning_rate: Learning rate for control point updates.
        tol: Convergence tolerance on regret change.
        temperature: Sigmoid sharpness for soft containment (annealed).
        temperature_final: Final temperature after annealing.
        sgd_settings: Settings for inner SGD optimization.
        objective: Which regret to maximize ('liberal' or 'conservative').
        verbose: Print progress during optimization.
    """

    max_iter: int = 100
    learning_rate: float = 50.0
    tol: float = 1e-4
    temperature: float = 1.0
    temperature_final: float = 20.0
    sgd_settings: SGDSettings | None = None
    objective: str = "liberal"  # 'liberal' or 'conservative'
    verbose: bool = True


class BlobAdversarialDiscovery:
    """Discover critical neighbor configurations using morphable blob geometry.

    Uses gradient ascent on regret with respect to B-spline control points
    to find neighbor farm shapes that maximize design tradeoffs.

    The neighbor farm is represented as a soft-packed region within a
    B-spline boundary, allowing gradients to flow through the effective
    number of neighbor turbines.

    Parameters
    ----------
    sim : WakeSimulation
        Wake simulation engine.
    target_boundary : jnp.ndarray
        Target farm boundary polygon.
    target_min_spacing : float
        Minimum spacing between target turbines.
    neighbor_grid : jnp.ndarray
        Dense reference grid for potential neighbor positions.
    ws_amb : jnp.ndarray
        Ambient wind speeds.
    wd_amb : jnp.ndarray
        Wind directions.
    ti_amb : jnp.ndarray | None
        Ambient turbulence intensity.
    weights : jnp.ndarray | None
        Probability weights for wind conditions.
    """

    def __init__(
        self,
        sim,
        target_boundary: jnp.ndarray,
        target_min_spacing: float,
        neighbor_grid: jnp.ndarray,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ):
        self.sim = sim
        self.target_boundary = target_boundary
        self.target_min_spacing = target_min_spacing
        self.neighbor_grid = neighbor_grid
        self.ws_amb = ws_amb
        self.wd_amb = wd_amb
        self.ti_amb = ti_amb
        self.weights = weights

    def _compute_aep_absent(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute AEP for target turbines WITHOUT neighbors."""
        result = self.sim(
            target_x, target_y, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )
        power = result.power()

        if self.weights is not None:
            return jnp.sum(power * self.weights[:, None]) * 8760 / 1e6
        return jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    def _compute_aep_present(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute AEP for target turbines WITH soft-masked neighbors.

        Uses soft interpolation between isolated and full-neighbor cases:
            AEP = (1 - f) * AEP_absent + f * AEP_full
        where f = sum(mask) / n_neighbors.
        """
        n_target = target_x.shape[0]

        # AEP without neighbors
        aep_absent = self._compute_aep_absent(target_x, target_y)

        # AEP with all neighbors
        neighbor_x = self.neighbor_grid[:, 0]
        neighbor_y = self.neighbor_grid[:, 1]
        x_all = jnp.concatenate([target_x, neighbor_x])
        y_all = jnp.concatenate([target_y, neighbor_y])

        result_full = self.sim(
            x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )
        power_full = result_full.power()[:, :n_target]

        if self.weights is not None:
            aep_full = jnp.sum(power_full * self.weights[:, None]) * 8760 / 1e6
        else:
            aep_full = jnp.sum(power_full) * 8760 / 1e6 / power_full.shape[0]

        # Soft interpolation
        n_neighbors = self.neighbor_grid.shape[0]
        effective_fraction = jnp.sum(neighbor_mask) / n_neighbors

        return aep_absent * (1 - effective_fraction) + aep_full * effective_fraction

    def _compute_aep(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute AEP (dispatches to absent/present based on mask)."""
        if neighbor_mask is None:
            return self._compute_aep_absent(target_x, target_y)
        return self._compute_aep_present(target_x, target_y, neighbor_mask)

    def discover(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        init_control_points: jnp.ndarray,
        settings: BlobDiscoverySettings | None = None,
    ) -> BlobDiscoveryResult:
        """Run blob-based adversarial discovery.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Initial target farm turbine positions.
        init_control_points : jnp.ndarray
            Initial B-spline control points for neighbor boundary.
        settings : BlobDiscoverySettings | None
            Discovery settings.

        Returns
        -------
        BlobDiscoveryResult
            Discovery results including optimized blob and regret.
        """
        from pixwake.optim.geometry import BSplineBoundary
        from pixwake.optim.sgd import topfarm_sgd_solve

        if settings is None:
            settings = BlobDiscoverySettings()

        sgd_settings = settings.sgd_settings or SGDSettings()

        # Compute LIBERAL layout (optimized WITHOUT neighbors)
        def liberal_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return -self._compute_aep_absent(x, y)

        liberal_x, liberal_y = topfarm_sgd_solve(
            liberal_objective,
            init_target_x,
            init_target_y,
            self.target_boundary,
            self.target_min_spacing,
            sgd_settings,
        )

        # Compute AEP_L_absent (liberal design, no neighbors) - this is fixed
        aep_L_absent = float(self._compute_aep_absent(liberal_x, liberal_y))

        if settings.verbose:
            print(f"Liberal design AEP (no neighbors): {aep_L_absent:.2f} GWh", flush=True)

        # Define regret function
        def compute_regret(control_points: jnp.ndarray, temperature: float) -> tuple[jnp.ndarray, dict]:
            """Compute liberal regret for given blob configuration.

            Returns regret_liberal = AEP_C_present - AEP_L_present
            (positive when conservative design outperforms liberal under neighbor presence)
            """
            spline = BSplineBoundary(control_points)
            neighbor_mask = spline.contains(self.neighbor_grid, temperature)

            # AEP_L_present: Liberal design with neighbors
            aep_L_present = self._compute_aep_present(liberal_x, liberal_y, neighbor_mask)

            # Compute CONSERVATIVE layout (optimized WITH neighbors)
            # Start from liberal layout for better convergence
            def conservative_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                return -self._compute_aep_present(x, y, neighbor_mask)

            conservative_x, conservative_y = topfarm_sgd_solve(
                conservative_objective,
                liberal_x,  # Start from liberal layout
                liberal_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
            )

            # AEP_C_present: Conservative design with neighbors
            aep_C_present = self._compute_aep_present(conservative_x, conservative_y, neighbor_mask)

            # AEP_C_absent: Conservative design without neighbors
            aep_C_absent = self._compute_aep_absent(conservative_x, conservative_y)

            # Regrets
            regret_liberal = aep_C_present - aep_L_present  # Benefit of conservative when neighbors appear
            regret_conservative = aep_L_absent - aep_C_absent  # Cost of conservative when neighbors don't appear

            details = {
                'aep_L_present': aep_L_present,
                'aep_C_present': aep_C_present,
                'aep_C_absent': aep_C_absent,
                'conservative_x': conservative_x,
                'conservative_y': conservative_y,
                'regret_conservative': regret_conservative,
            }

            return regret_liberal, details

        # Gradient ascent on control points using finite differences
        # (to avoid custom_vjp closure issues with nested differentiation)
        control_points = init_control_points
        history = []

        def compute_gradient_fd(cp: jnp.ndarray, temp: float, eps: float = 100.0) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
            """Compute gradient of liberal regret using finite differences."""
            grad = jnp.zeros_like(cp)
            base_regret, base_details = compute_regret(cp, temp)

            # Finite differences for each control point coordinate
            for i in range(cp.shape[0]):
                for j in range(cp.shape[1]):
                    cp_plus = cp.at[i, j].set(cp[i, j] + eps)
                    cp_minus = cp.at[i, j].set(cp[i, j] - eps)
                    regret_plus, _ = compute_regret(cp_plus, temp)
                    regret_minus, _ = compute_regret(cp_minus, temp)
                    grad = grad.at[i, j].set((regret_plus - regret_minus) / (2 * eps))

            return base_regret, grad, base_details

        for i in range(settings.max_iter):
            # Anneal temperature
            progress = i / max(settings.max_iter - 1, 1)
            temperature = (
                settings.temperature
                + progress * (settings.temperature_final - settings.temperature)
            )

            # Compute regret and gradient using finite differences
            regret, grad, details = compute_gradient_fd(control_points, temperature)

            history.append((float(regret), control_points.copy()))

            if settings.verbose and (i % 10 == 0 or i < 5):
                n_effective = float(
                    BSplineBoundary(control_points).contains(
                        self.neighbor_grid, temperature
                    ).sum()
                )
                step_size = settings.learning_rate * jnp.linalg.norm(grad)
                print(
                    f"Iter {i}: R_lib = {regret:.4f} GWh, "
                    f"R_con = {details['regret_conservative']:.4f} GWh, "
                    f"|grad| = {jnp.linalg.norm(grad):.6f}, "
                    f"n_eff = {n_effective:.1f}",
                    flush=True,
                )

            # Check convergence
            if i > 0 and abs(history[-1][0] - history[-2][0]) < settings.tol:
                if settings.verbose:
                    print(f"Converged at iteration {i}", flush=True)
                break

            # Gradient ascent step (maximize liberal regret)
            control_points = control_points + settings.learning_rate * grad

        # Final evaluation
        final_temperature = settings.temperature_final
        final_regret, final_details = compute_regret(control_points, final_temperature)

        final_mask = BSplineBoundary(control_points).contains(self.neighbor_grid, final_temperature)
        aep_L_present = float(final_details['aep_L_present'])
        aep_C_present = float(final_details['aep_C_present'])
        aep_C_absent = float(final_details['aep_C_absent'])
        regret_liberal = float(final_regret)
        regret_conservative = float(final_details['regret_conservative'])

        if settings.verbose:
            print(f"\n{'='*60}", flush=True)
            print("Final AEP Matrix:", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"                      Neighbors Present | Neighbors Absent", flush=True)
            print(f"  Liberal Design:     {aep_L_present:>10.2f} GWh | {aep_L_absent:>10.2f} GWh", flush=True)
            print(f"  Conservative Design:{aep_C_present:>10.2f} GWh | {aep_C_absent:>10.2f} GWh", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"Liberal Regret (R_lib = AEP_C_pres - AEP_L_pres): {regret_liberal:.2f} GWh", flush=True)
            print(f"Conservative Regret (R_con = AEP_L_abs - AEP_C_abs): {regret_conservative:.2f} GWh", flush=True)
            print(f"Effective neighbors: {float(final_mask.sum()):.1f}", flush=True)

        return BlobDiscoveryResult(
            control_points=control_points,
            liberal_x=liberal_x,
            liberal_y=liberal_y,
            conservative_x=final_details['conservative_x'],
            conservative_y=final_details['conservative_y'],
            aep_L_absent=aep_L_absent,
            aep_L_present=aep_L_present,
            aep_C_absent=aep_C_absent,
            aep_C_present=aep_C_present,
            regret_liberal=regret_liberal,
            regret_conservative=regret_conservative,
            history=history,
        )


# =============================================================================
# Pooled Multi-Start Blob Discovery (Proper Methodology)
# =============================================================================


class PooledBlobDiscoveryResult(NamedTuple):
    """Result of pooled multi-start blob discovery.

    The pooled methodology runs N random starts for both liberal and conservative
    strategies, cross-evaluates all layouts under both scenarios, and computes
    regret against pooled global bests. This ensures regret values reflect true
    fundamental tradeoffs rather than local minima artifacts.

    Attributes:
        control_points: B-spline control points defining neighbor boundary.
        global_best_aep_absent: Best AEP across all layouts when neighbors absent.
        global_best_aep_present: Best AEP across all layouts when neighbors present.
        min_liberal_regret: Minimum liberal regret achievable from the pool.
        min_conservative_regret: Minimum conservative regret achievable from the pool.
        best_liberal_layout: Layout achieving minimum liberal regret (x, y).
        best_conservative_layout: Layout achieving minimum conservative regret (x, y).
        all_layouts: List of all evaluated layouts with their AEPs.
        n_liberal_starts: Number of liberal optimization starts.
        n_conservative_starts: Number of conservative optimization starts.
        same_best_layout: Whether the same layout achieves both global bests.
    """

    control_points: jnp.ndarray
    global_best_aep_absent: float
    global_best_aep_present: float
    min_liberal_regret: float
    min_conservative_regret: float
    best_liberal_layout: tuple[jnp.ndarray, jnp.ndarray]
    best_conservative_layout: tuple[jnp.ndarray, jnp.ndarray]
    all_layouts: list[dict]
    n_liberal_starts: int
    n_conservative_starts: int
    same_best_layout: bool


@dataclass
class PooledBlobDiscoverySettings:
    """Settings for pooled multi-start blob discovery.

    Attributes:
        n_starts: Number of random starts per strategy (liberal/conservative).
        sgd_settings: Settings for inner SGD optimization.
        verbose: Print progress during optimization.
    """

    n_starts: int = 10
    sgd_settings: SGDSettings | None = None
    verbose: bool = True


class PooledBlobDiscovery:
    """Pooled multi-start blob discovery for proper regret estimation.

    This class implements the correct methodology for estimating design regret:
    1. For a given blob configuration, run N multi-start optimizations with
       liberal assumptions (ignoring neighbors).
    2. Run N multi-start optimizations with conservative assumptions
       (accounting for neighbors).
    3. Cross-evaluate ALL layouts from both pools under BOTH scenarios
       (neighbors present and absent).
    4. Compute regret against pooled global bests, ensuring regret values
       reflect true fundamental tradeoffs rather than local minima artifacts.

    Parameters
    ----------
    sim : WakeSimulation
        Wake simulation engine.
    target_boundary : jnp.ndarray
        Target farm boundary polygon.
    target_min_spacing : float
        Minimum spacing between target turbines.
    neighbor_grid : jnp.ndarray
        Dense reference grid for potential neighbor positions.
    ws_amb : jnp.ndarray
        Ambient wind speeds.
    wd_amb : jnp.ndarray
        Wind directions.
    ti_amb : jnp.ndarray | None
        Ambient turbulence intensity.
    weights : jnp.ndarray | None
        Probability weights for wind conditions.
    """

    def __init__(
        self,
        sim,
        target_boundary: jnp.ndarray,
        target_min_spacing: float,
        neighbor_grid: jnp.ndarray,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ):
        self.sim = sim
        self.target_boundary = target_boundary
        self.target_min_spacing = target_min_spacing
        self.neighbor_grid = neighbor_grid
        self.ws_amb = ws_amb
        self.wd_amb = wd_amb
        self.ti_amb = ti_amb
        self.weights = weights

    def _compute_aep_absent(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
    ) -> float:
        """Compute AEP for target turbines WITHOUT neighbors."""
        result = self.sim(
            target_x, target_y, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )
        power = result.power()

        if self.weights is not None:
            aep = jnp.sum(power * self.weights[:, None]) * 8760 / 1e6
        else:
            aep = jnp.sum(power) * 8760 / 1e6 / power.shape[0]
        return float(aep)

    def _compute_aep_present(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
    ) -> float:
        """Compute AEP for target turbines WITH soft-masked neighbors."""
        n_target = target_x.shape[0]

        # AEP without neighbors
        aep_absent = self._compute_aep_absent(target_x, target_y)

        # AEP with all neighbors
        neighbor_x = self.neighbor_grid[:, 0]
        neighbor_y = self.neighbor_grid[:, 1]
        x_all = jnp.concatenate([target_x, neighbor_x])
        y_all = jnp.concatenate([target_y, neighbor_y])

        result_full = self.sim(
            x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )
        power_full = result_full.power()[:, :n_target]

        if self.weights is not None:
            aep_full = float(jnp.sum(power_full * self.weights[:, None]) * 8760 / 1e6)
        else:
            aep_full = float(jnp.sum(power_full) * 8760 / 1e6 / power_full.shape[0])

        # Soft interpolation
        n_neighbors = self.neighbor_grid.shape[0]
        effective_fraction = float(jnp.sum(neighbor_mask)) / n_neighbors

        return aep_absent * (1 - effective_fraction) + aep_full * effective_fraction

    def _generate_random_init(
        self,
        key: jnp.ndarray,
        n_turbines: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random initial positions within boundary."""
        x_min = float(self.target_boundary[:, 0].min())
        x_max = float(self.target_boundary[:, 0].max())
        y_min = float(self.target_boundary[:, 1].min())
        y_max = float(self.target_boundary[:, 1].max())

        margin = self.target_min_spacing / 2
        key1, key2 = jax.random.split(key)
        x = jax.random.uniform(key1, (n_turbines,), minval=x_min + margin, maxval=x_max - margin)
        y = jax.random.uniform(key2, (n_turbines,), minval=y_min + margin, maxval=y_max - margin)
        return x, y

    def discover(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        control_points: jnp.ndarray,
        settings: PooledBlobDiscoverySettings | None = None,
        seed: int = 42,
    ) -> PooledBlobDiscoveryResult:
        """Run pooled multi-start discovery for given blob configuration.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Base target farm turbine positions (used as one of the starts).
        control_points : jnp.ndarray
            B-spline control points for neighbor boundary.
        settings : PooledBlobDiscoverySettings | None
            Discovery settings.
        seed : int
            Random seed for generating random starts.

        Returns
        -------
        PooledBlobDiscoveryResult
            Discovery results with pooled regret values.
        """
        from pixwake.optim.geometry import BSplineBoundary
        from pixwake.optim.sgd import topfarm_sgd_solve

        if settings is None:
            settings = PooledBlobDiscoverySettings()

        sgd_settings = settings.sgd_settings or SGDSettings()

        # Compute neighbor mask for this blob
        spline = BSplineBoundary(control_points)
        neighbor_mask = spline.contains(self.neighbor_grid, temperature=10.0)

        n_turbines = len(init_target_x)
        key = jax.random.PRNGKey(seed)

        # Define objective functions
        def liberal_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Objective for liberal optimization (no neighbors)."""
            result = self.sim(x, y, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb)
            power = result.power()
            if self.weights is not None:
                aep = jnp.sum(power * self.weights[:, None]) * 8760 / 1e6
            else:
                aep = jnp.sum(power) * 8760 / 1e6 / power.shape[0]
            return -aep

        def conservative_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Objective for conservative optimization (with neighbors)."""
            n_target = x.shape[0]
            neighbor_x = self.neighbor_grid[:, 0]
            neighbor_y = self.neighbor_grid[:, 1]
            x_all = jnp.concatenate([x, neighbor_x])
            y_all = jnp.concatenate([y, neighbor_y])

            result = self.sim(x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb)
            power_full = result.power()[:, :n_target]

            # AEP with soft-interpolated neighbors
            result_isolated = self.sim(x, y, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb)
            power_isolated = result_isolated.power()

            if self.weights is not None:
                aep_full = jnp.sum(power_full * self.weights[:, None]) * 8760 / 1e6
                aep_isolated = jnp.sum(power_isolated * self.weights[:, None]) * 8760 / 1e6
            else:
                aep_full = jnp.sum(power_full) * 8760 / 1e6 / power_full.shape[0]
                aep_isolated = jnp.sum(power_isolated) * 8760 / 1e6 / power_isolated.shape[0]

            n_neighbors = self.neighbor_grid.shape[0]
            effective_fraction = jnp.sum(neighbor_mask) / n_neighbors
            aep = aep_isolated * (1 - effective_fraction) + aep_full * effective_fraction
            return -aep

        all_layouts = []

        if settings.verbose:
            print(f"Running pooled multi-start optimization ({settings.n_starts} starts per strategy)")
            print(f"  Effective neighbors in blob: {float(neighbor_mask.sum()):.1f}")

        # Run liberal optimizations
        if settings.verbose:
            print(f"\nLiberal optimizations (ignoring neighbors):")
        for i in range(settings.n_starts):
            if i == 0:
                start_x, start_y = init_target_x, init_target_y
            else:
                key, subkey = jax.random.split(key)
                start_x, start_y = self._generate_random_init(subkey, n_turbines)

            opt_x, opt_y = topfarm_sgd_solve(
                liberal_objective,
                start_x,
                start_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
            )

            # Cross-evaluate under both scenarios
            aep_absent = self._compute_aep_absent(opt_x, opt_y)
            aep_present = self._compute_aep_present(opt_x, opt_y, neighbor_mask)

            all_layouts.append({
                'strategy': 'liberal',
                'start_idx': i,
                'x': opt_x,
                'y': opt_y,
                'aep_absent': aep_absent,
                'aep_present': aep_present,
            })

            if settings.verbose:
                print(f"  Start {i}: AEP_absent={aep_absent:.2f}, AEP_present={aep_present:.2f} GWh")

        # Run conservative optimizations
        if settings.verbose:
            print(f"\nConservative optimizations (accounting for neighbors):")
        for i in range(settings.n_starts):
            if i == 0:
                start_x, start_y = init_target_x, init_target_y
            else:
                key, subkey = jax.random.split(key)
                start_x, start_y = self._generate_random_init(subkey, n_turbines)

            opt_x, opt_y = topfarm_sgd_solve(
                conservative_objective,
                start_x,
                start_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
            )

            # Cross-evaluate under both scenarios
            aep_absent = self._compute_aep_absent(opt_x, opt_y)
            aep_present = self._compute_aep_present(opt_x, opt_y, neighbor_mask)

            all_layouts.append({
                'strategy': 'conservative',
                'start_idx': i,
                'x': opt_x,
                'y': opt_y,
                'aep_absent': aep_absent,
                'aep_present': aep_present,
            })

            if settings.verbose:
                print(f"  Start {i}: AEP_absent={aep_absent:.2f}, AEP_present={aep_present:.2f} GWh")

        # Compute pooled global bests
        global_best_aep_absent = max(l['aep_absent'] for l in all_layouts)
        global_best_aep_present = max(l['aep_present'] for l in all_layouts)

        # Compute regrets for each layout
        for layout in all_layouts:
            # Liberal regret: how much worse than best when neighbors appear
            layout['liberal_regret'] = global_best_aep_present - layout['aep_present']
            # Conservative regret: how much worse than best when neighbors absent
            layout['conservative_regret'] = global_best_aep_absent - layout['aep_absent']

        # Find minimum regrets
        min_liberal_regret = min(l['liberal_regret'] for l in all_layouts)
        min_conservative_regret = min(l['conservative_regret'] for l in all_layouts)

        # Find layouts achieving minimum regrets
        best_liberal_layout = min(all_layouts, key=lambda l: l['liberal_regret'])
        best_conservative_layout = min(all_layouts, key=lambda l: l['conservative_regret'])

        # Check if same layout achieves both bests
        best_absent_layout = max(all_layouts, key=lambda l: l['aep_absent'])
        best_present_layout = max(all_layouts, key=lambda l: l['aep_present'])
        same_best = (
            best_absent_layout['strategy'] == best_present_layout['strategy'] and
            best_absent_layout['start_idx'] == best_present_layout['start_idx']
        )

        if settings.verbose:
            print(f"\n{'='*60}")
            print("Pooled Results:")
            print(f"{'='*60}")
            print(f"Global best AEP (absent):  {global_best_aep_absent:.2f} GWh")
            print(f"Global best AEP (present): {global_best_aep_present:.2f} GWh")
            print(f"Min liberal regret:        {min_liberal_regret:.2f} GWh")
            print(f"Min conservative regret:   {min_conservative_regret:.2f} GWh")
            print(f"Same layout achieves both: {same_best}")
            if not same_best:
                print("  -> TRUE TRADEOFF EXISTS")
            else:
                print("  -> No fundamental tradeoff (single layout dominates)")

        # Convert layouts for serialization
        serializable_layouts = []
        for l in all_layouts:
            serializable_layouts.append({
                'strategy': l['strategy'],
                'start_idx': l['start_idx'],
                'x': l['x'].tolist() if hasattr(l['x'], 'tolist') else list(l['x']),
                'y': l['y'].tolist() if hasattr(l['y'], 'tolist') else list(l['y']),
                'aep_absent': l['aep_absent'],
                'aep_present': l['aep_present'],
                'liberal_regret': l['liberal_regret'],
                'conservative_regret': l['conservative_regret'],
            })

        return PooledBlobDiscoveryResult(
            control_points=control_points,
            global_best_aep_absent=global_best_aep_absent,
            global_best_aep_present=global_best_aep_present,
            min_liberal_regret=min_liberal_regret,
            min_conservative_regret=min_conservative_regret,
            best_liberal_layout=(best_liberal_layout['x'], best_liberal_layout['y']),
            best_conservative_layout=(best_conservative_layout['x'], best_conservative_layout['y']),
            all_layouts=serializable_layouts,
            n_liberal_starts=settings.n_starts,
            n_conservative_starts=settings.n_starts,
            same_best_layout=same_best,
        )


# =============================================================================
# Greedy Grid Search
# =============================================================================


class GreedyGridResult(NamedTuple):
    """Result of greedy grid search for adversarial neighbor placement.

    Attributes:
        neighbor_x: Placed neighbor x positions, shape (n_placed,).
        neighbor_y: Placed neighbor y positions, shape (n_placed,).
        target_x: Target layout re-optimized with all placed neighbors.
        target_y: Target layout re-optimized with all placed neighbors.
        liberal_x: Liberal layout (optimized without neighbors).
        liberal_y: Liberal layout (optimized without neighbors).
        regret: Final regret with all placed neighbors.
        liberal_aep: AEP of liberal layout without neighbors.
        conservative_aep: AEP of conservative layout with all placed neighbors.
        placement_order: Grid indices chosen at each greedy step.
        regret_history: Regret after each turbine placement.
        regret_maps: Per-step regret at every grid point (list of arrays).
    """

    neighbor_x: jnp.ndarray
    neighbor_y: jnp.ndarray
    target_x: jnp.ndarray
    target_y: jnp.ndarray
    liberal_x: jnp.ndarray
    liberal_y: jnp.ndarray
    regret: float
    liberal_aep: float
    conservative_aep: float
    placement_order: list[int]
    regret_history: list[float]
    regret_maps: list[jnp.ndarray]


@dataclass
class GreedyGridSettings:
    """Settings for greedy grid search.

    Attributes:
        sgd_settings: Settings for inner SGD optimization.
        n_inner_starts: Number of random inner starts for conservative
            optimization (best-of-K). Default 1 (single start).
        screen_top_k: If > 0, screen all candidates by AEP loss first
            (cheap forward eval only), then run full inner optimization
            on only the top-K candidates. Drastically reduces runtime
            for large grids. Default 0 (no screening, evaluate all).
        screen_chunk_size: Number of candidates to screen in each
            vectorized batch (vmap). Larger = faster but more memory.
            Default 100 (~0.4 GB for 66 turbines × 24 dirs).
        eval_parallel: If True, run the top-K full evaluations in
            parallel via vmap over candidate positions. Default True.
        verbose: Print progress during search.
    """

    sgd_settings: SGDSettings | None = None
    n_inner_starts: int = 1
    screen_top_k: int = 0
    screen_chunk_size: int = 100
    eval_parallel: bool = True
    verbose: bool = True


class GreedyGridSearch:
    """Greedy sequential placement of neighbor turbines on a discrete grid.

    Places neighbors one at a time. At each step, evaluates the regret at
    every remaining grid point (by re-optimizing the target layout with
    the candidate neighbor configuration) and selects the point that
    maximizes regret. Repeats until the desired number of neighbors is
    placed.

    Parameters
    ----------
    sim : WakeSimulation
        Wake simulation engine.
    target_boundary : jnp.ndarray
        Target farm boundary polygon, shape (n_vertices, 2).
    target_min_spacing : float
        Minimum spacing between target turbines.
    ws_amb : jnp.ndarray
        Ambient wind speeds.
    wd_amb : jnp.ndarray
        Wind directions.
    ti_amb : jnp.ndarray | None
        Ambient turbulence intensity (optional).
    weights : jnp.ndarray | None
        Probability weights for wind conditions (optional).
    """

    def __init__(
        self,
        sim,
        target_boundary: jnp.ndarray,
        target_min_spacing: float,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ):
        self.sim = sim
        self.target_boundary = target_boundary
        self.target_min_spacing = target_min_spacing
        self.ws_amb = ws_amb
        self.wd_amb = wd_amb
        self.ti_amb = ti_amb
        self.weights = weights

    def _compute_aep(
        self,
        target_x: jnp.ndarray,
        target_y: jnp.ndarray,
        neighbor_x: jnp.ndarray | None = None,
        neighbor_y: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute AEP for target turbines, optionally with neighbors."""
        n_target = target_x.shape[0]

        if neighbor_x is not None and neighbor_y is not None and neighbor_x.shape[0] > 0:
            x_all = jnp.concatenate([target_x, neighbor_x])
            y_all = jnp.concatenate([target_y, neighbor_y])
        else:
            x_all = target_x
            y_all = target_y

        result = self.sim(
            x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb, ti_amb=self.ti_amb
        )

        power = result.power()[:, :n_target]

        if self.weights is not None:
            weighted_power = jnp.sum(power * self.weights[:, None])
            return weighted_power * 8760 / 1e6
        return jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    def _evaluate_candidate(
        self,
        candidate_x: float,
        candidate_y: float,
        placed_x: jnp.ndarray,
        placed_y: jnp.ndarray,
        liberal_x: jnp.ndarray,
        liberal_y: jnp.ndarray,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        sgd_settings: SGDSettings,
        n_inner_starts: int = 1,
    ) -> tuple[float, jnp.ndarray, jnp.ndarray]:
        """Evaluate regret for adding one candidate neighbor.

        Returns (regret, conservative_x, conservative_y).
        """
        # Assemble all neighbors: already placed + candidate
        nb_x = jnp.concatenate([placed_x, jnp.array([candidate_x])])
        nb_y = jnp.concatenate([placed_y, jnp.array([candidate_y])])

        # Liberal AEP with these neighbors (no re-optimization)
        liberal_aep_present = self._compute_aep(liberal_x, liberal_y, nb_x, nb_y)

        # Conservative: re-optimize target layout with these neighbors
        def conservative_objective(x, y):
            return -self._compute_aep(x, y, nb_x, nb_y)

        # Start with liberal layout as a candidate (guarantees regret >= 0)
        best_aep = float(liberal_aep_present)
        best_cx, best_cy = liberal_x, liberal_y

        for k in range(n_inner_starts):
            if k == 0:
                start_x, start_y = init_target_x, init_target_y
            elif k == 1:
                # Use liberal layout as a start — SGD may improve it
                start_x, start_y = liberal_x, liberal_y
            else:
                key = jax.random.PRNGKey(hash((float(candidate_x), float(candidate_y), k)) % (2**31))
                start_x, start_y = self._random_init(key, init_target_x.shape[0])

            opt_x, opt_y = topfarm_sgd_solve(
                conservative_objective,
                start_x,
                start_y,
                self.target_boundary,
                self.target_min_spacing,
                sgd_settings,
            )

            aep = self._compute_aep(opt_x, opt_y, nb_x, nb_y)
            if aep > best_aep:
                best_aep = aep
                best_cx, best_cy = opt_x, opt_y

        conservative_aep = best_aep
        regret = float(conservative_aep - liberal_aep_present)
        return regret, best_cx, best_cy

    def _random_init(self, key, n_turbines):
        """Generate random initial positions within target boundary polygon.

        Uses rejection sampling to ensure all turbines are inside the
        convex hull of the boundary, not just within the bounding box.
        """
        from matplotlib.path import Path as MplPath

        boundary_np = np.array(self.target_boundary)
        poly_path = MplPath(boundary_np)
        margin = self.target_min_spacing / 2

        x_min = float(boundary_np[:, 0].min()) + margin
        x_max = float(boundary_np[:, 0].max()) - margin
        y_min = float(boundary_np[:, 1].min()) + margin
        y_max = float(boundary_np[:, 1].max()) - margin

        # Rejection sampling: generate candidates, keep those inside polygon
        xs, ys = [], []
        max_attempts = 100
        for _ in range(max_attempts):
            key, k1, k2 = jax.random.split(key, 3)
            n_need = n_turbines - len(xs)
            # Oversample to reduce iterations
            n_try = max(n_need * 3, 100)
            cx = jax.random.uniform(k1, (n_try,), minval=x_min, maxval=x_max)
            cy = jax.random.uniform(k2, (n_try,), minval=y_min, maxval=y_max)
            pts = np.column_stack([np.array(cx), np.array(cy)])
            inside = poly_path.contains_points(pts)
            xs.extend(float(cx[i]) for i in range(n_try) if inside[i])
            ys.extend(float(cy[i]) for i in range(n_try) if inside[i])
            if len(xs) >= n_turbines:
                break

        x = jnp.array(xs[:n_turbines])
        y = jnp.array(ys[:n_turbines])
        return x, y

    def _screen_batch(
        self,
        candidate_xs: jnp.ndarray,
        candidate_ys: jnp.ndarray,
        placed_x: jnp.ndarray,
        placed_y: jnp.ndarray,
        liberal_x: jnp.ndarray,
        liberal_y: jnp.ndarray,
        liberal_aep: float,
        chunk_size: int = 100,
    ) -> jnp.ndarray:
        """Screen candidates by AEP loss using vectorized evaluation.

        Args:
            candidate_xs: x positions of candidates, shape (n_cands,).
            candidate_ys: y positions of candidates, shape (n_cands,).
            placed_x: Already-placed neighbor x positions.
            placed_y: Already-placed neighbor y positions.
            liberal_x: Liberal layout x positions.
            liberal_y: Liberal layout y positions.
            liberal_aep: AEP of liberal layout without neighbors.
            chunk_size: Batch size for vmap. Default 100.

        Returns:
            AEP loss for each candidate, shape (n_cands,).
        """
        n_target = liberal_x.shape[0]
        n_placed = placed_x.shape[0]

        def eval_one_candidate(cx, cy):
            """Compute AEP loss for one candidate neighbor."""
            nb_x = jnp.concatenate([placed_x, jnp.array([cx])])
            nb_y = jnp.concatenate([placed_y, jnp.array([cy])])
            x_all = jnp.concatenate([liberal_x, nb_x])
            y_all = jnp.concatenate([liberal_y, nb_y])
            result = self.sim(
                x_all, y_all, ws_amb=self.ws_amb, wd_amb=self.wd_amb,
                ti_amb=self.ti_amb,
            )
            power = result.power()[:, :n_target]
            if self.weights is not None:
                aep = jnp.sum(power * self.weights[:, None]) * 8760 / 1e6
            else:
                aep = jnp.sum(power) * 8760 / 1e6 / power.shape[0]
            return liberal_aep - aep

        eval_batch = jax.vmap(eval_one_candidate)

        n_cands = candidate_xs.shape[0]
        all_losses = []
        for start in range(0, n_cands, chunk_size):
            end = min(start + chunk_size, n_cands)
            chunk_losses = eval_batch(
                candidate_xs[start:end], candidate_ys[start:end]
            )
            all_losses.append(chunk_losses)

        return jnp.concatenate(all_losses)

    def _evaluate_candidates_batch(
        self,
        candidate_xs: jnp.ndarray,
        candidate_ys: jnp.ndarray,
        placed_x: jnp.ndarray,
        placed_y: jnp.ndarray,
        liberal_x: jnp.ndarray,
        liberal_y: jnp.ndarray,
        init_target_xs: jnp.ndarray,
        init_target_ys: jnp.ndarray,
        sgd_settings: SGDSettings,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate regret for a batch of candidates via vmap'd inner SGD.

        Supports multistart: init_target_xs/ys can be shape (K, n_turbines)
        for single-start or (n_candidates, K, n_turbines) for multistart.
        When multistart, each candidate runs K inner SGDs in parallel and
        the best (highest AEP) is kept.

        All candidates × starts are flattened into one vmap call for
        maximum GPU utilization.

        Args:
            candidate_xs: Candidate x positions, shape (n_candidates,).
            candidate_ys: Candidate y positions, shape (n_candidates,).
            placed_x: Already-placed neighbor x positions.
            placed_y: Already-placed neighbor y positions.
            liberal_x: Liberal layout x positions.
            liberal_y: Liberal layout y positions.
            init_target_xs: Initial target x positions.
                Shape (n_turbines,) for single start,
                or (n_candidates, K, n_turbines) for multistart.
            init_target_ys: Initial target y positions (same shape rules).
            sgd_settings: SGD settings (with mid pre-computed).

        Returns:
            Tuple of (regrets, opt_xs, opt_ys) where:
                regrets: shape (n_candidates,)
                opt_xs: shape (n_candidates, n_turbines)
                opt_ys: shape (n_candidates, n_turbines)
        """
        from dataclasses import replace as _dc_replace
        from pixwake.optim.sgd import _compute_mid_bisection

        # Pre-compute mid so it doesn't happen inside vmap
        if sgd_settings.mid is None:
            gamma_min = sgd_settings.gamma_min_factor
            computed_mid = _compute_mid_bisection(
                learning_rate=sgd_settings.learning_rate,
                gamma_min=gamma_min,
                max_iter=sgd_settings.max_iter,
                lower=sgd_settings.bisect_lower,
                upper=sgd_settings.bisect_upper,
            )
            sgd_settings = _dc_replace(sgd_settings, mid=computed_mid)

        n_candidates = candidate_xs.shape[0]
        n_target = liberal_x.shape[0]
        sim = self.sim
        ws_amb = self.ws_amb
        wd_amb = self.wd_amb
        ti_amb = self.ti_amb
        weights = self.weights
        bnd = self.target_boundary
        min_sp = self.target_min_spacing

        # Determine if multistart
        multistart = init_target_xs.ndim == 3  # (n_candidates, K, n_turbines)

        if multistart:
            n_starts = init_target_xs.shape[1]
            # Flatten candidates × starts: (n_candidates * K,)
            flat_cx = jnp.repeat(candidate_xs, n_starts)
            flat_cy = jnp.repeat(candidate_ys, n_starts)
            flat_init_x = init_target_xs.reshape(-1, n_target)  # (n_candidates*K, n_turbines)
            flat_init_y = init_target_ys.reshape(-1, n_target)
        else:
            flat_cx = candidate_xs
            flat_cy = candidate_ys
            flat_init_x = jnp.broadcast_to(init_target_xs, (n_candidates, n_target))
            flat_init_y = jnp.broadcast_to(init_target_ys, (n_candidates, n_target))

        def solve_one(cx, cy, start_x, start_y):
            """Run inner SGD for one (candidate, start), return (aep, regret, opt_x, opt_y)."""
            nb_x = jnp.concatenate([placed_x, jnp.array([cx])])
            nb_y = jnp.concatenate([placed_y, jnp.array([cy])])

            def conservative_objective(x, y):
                x_all = jnp.concatenate([x, nb_x])
                y_all = jnp.concatenate([y, nb_y])
                result = sim(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb,
                             ti_amb=ti_amb)
                power = result.power()[:, :n_target]
                if weights is not None:
                    aep = jnp.sum(power * weights[:, None]) * 8760 / 1e6
                else:
                    aep = jnp.sum(power) * 8760 / 1e6 / power.shape[0]
                return -aep

            opt_x, opt_y = topfarm_sgd_solve(
                conservative_objective, start_x, start_y,
                bnd, min_sp, sgd_settings,
            )

            conservative_aep = -conservative_objective(opt_x, opt_y)

            # Liberal AEP with these neighbors
            x_lib_all = jnp.concatenate([liberal_x, nb_x])
            y_lib_all = jnp.concatenate([liberal_y, nb_y])
            result_lib = sim(x_lib_all, y_lib_all, ws_amb=ws_amb,
                             wd_amb=wd_amb, ti_amb=ti_amb)
            power_lib = result_lib.power()[:, :n_target]
            if weights is not None:
                lib_aep_present = jnp.sum(power_lib * weights[:, None]) * 8760 / 1e6
            else:
                lib_aep_present = jnp.sum(power_lib) * 8760 / 1e6 / power_lib.shape[0]

            regret = conservative_aep - lib_aep_present
            return conservative_aep, regret, opt_x, opt_y

        all_aeps, all_regrets, all_opt_xs, all_opt_ys = jax.vmap(solve_one)(
            flat_cx, flat_cy, flat_init_x, flat_init_y
        )

        if multistart:
            # Reshape back to (n_candidates, K, ...)
            all_aeps = all_aeps.reshape(n_candidates, n_starts)
            all_regrets = all_regrets.reshape(n_candidates, n_starts)
            all_opt_xs = all_opt_xs.reshape(n_candidates, n_starts, n_target)
            all_opt_ys = all_opt_ys.reshape(n_candidates, n_starts, n_target)

            # Pick best start per candidate (highest conservative AEP)
            best_k = jnp.argmax(all_aeps, axis=1)  # (n_candidates,)
            regrets = all_regrets[jnp.arange(n_candidates), best_k]
            opt_xs = all_opt_xs[jnp.arange(n_candidates), best_k]
            opt_ys = all_opt_ys[jnp.arange(n_candidates), best_k]
        else:
            regrets = all_regrets
            opt_xs = all_opt_xs
            opt_ys = all_opt_ys

        return regrets, opt_xs, opt_ys

    def search(
        self,
        init_target_x: jnp.ndarray,
        init_target_y: jnp.ndarray,
        grid: jnp.ndarray,
        n_neighbors: int,
        settings: GreedyGridSettings | None = None,
    ) -> GreedyGridResult:
        """Run greedy grid search to place neighbors sequentially.

        Parameters
        ----------
        init_target_x, init_target_y : jnp.ndarray
            Initial target farm turbine positions.
        grid : jnp.ndarray
            Candidate neighbor positions, shape (n_grid, 2). Each row is
            an (x, y) coordinate where a neighbor turbine could be placed.
        n_neighbors : int
            Number of neighbor turbines to place.
        settings : GreedyGridSettings | None
            Search settings. Uses defaults if None.

        Returns
        -------
        GreedyGridResult
        """
        if settings is None:
            settings = GreedyGridSettings()

        sgd_settings = settings.sgd_settings or SGDSettings()

        # Step 1: Compute liberal layout (no neighbors) — done once
        def liberal_objective(x, y):
            return -self._compute_aep(x, y)

        liberal_x, liberal_y = topfarm_sgd_solve(
            liberal_objective,
            init_target_x,
            init_target_y,
            self.target_boundary,
            self.target_min_spacing,
            sgd_settings,
        )
        liberal_aep = float(self._compute_aep(liberal_x, liberal_y))

        if settings.verbose:
            print(f"Liberal AEP (no neighbors): {liberal_aep:.2f} GWh", flush=True)
            print(f"Grid size: {grid.shape[0]} candidates, placing {n_neighbors} neighbors", flush=True)

        # Track state
        remaining = set(range(grid.shape[0]))
        placed_x = jnp.zeros(0)
        placed_y = jnp.zeros(0)
        placement_order = []
        regret_history = []
        regret_maps = []

        # Step 2: Greedy placement loop
        for step in range(n_neighbors):
            n_remaining = len(remaining)
            use_screening = settings.screen_top_k > 0 and n_remaining > settings.screen_top_k

            if settings.verbose:
                print(f"\n--- Greedy step {step + 1}/{n_neighbors} "
                      f"({n_remaining} candidates remaining) ---", flush=True)
                if use_screening:
                    print(f"  Screening phase: evaluating AEP loss for all "
                          f"{n_remaining} candidates...", flush=True)

            regret_map = np.full(grid.shape[0], np.nan)

            remaining_list = sorted(remaining)

            if use_screening:
                # Phase 1: Vectorized screening by AEP loss
                import time as _time
                _t0 = _time.time()
                cand_xs = jnp.array([grid[i, 0] for i in remaining_list])
                cand_ys = jnp.array([grid[i, 1] for i in remaining_list])

                aep_losses = self._screen_batch(
                    cand_xs, cand_ys, placed_x, placed_y,
                    liberal_x, liberal_y, liberal_aep,
                    chunk_size=settings.screen_chunk_size,
                )

                # Map back to grid indices
                for j, idx in enumerate(remaining_list):
                    regret_map[idx] = float(aep_losses[j])

                # Pick top-K by AEP loss
                top_k_local = jnp.argsort(-aep_losses)[:settings.screen_top_k]
                top_k_indices = [remaining_list[int(j)] for j in top_k_local]

                if settings.verbose:
                    _elapsed = _time.time() - _t0
                    print(f"  Screening: {n_remaining} candidates in "
                          f"{_elapsed:.1f}s (chunks of {settings.screen_chunk_size})", flush=True)
                    print(f"  Full eval phase: top {len(top_k_indices)} candidates...", flush=True)
                eval_set = top_k_indices
            else:
                eval_set = list(remaining)

            # Phase 2: Full evaluation (inner optimization)
            import time as _time
            _t0 = _time.time()

            if settings.eval_parallel and len(eval_set) > 1:
                # Vectorized: run all candidates (× starts) in parallel via vmap
                eval_xs = jnp.array([grid[i, 0] for i in eval_set])
                eval_ys = jnp.array([grid[i, 1] for i in eval_set])
                n_eval = len(eval_set)
                K = settings.n_inner_starts

                if K > 1:
                    # Build (n_eval, K, n_turbines) initial positions
                    # Start 0: grid init, Start 1: liberal layout, rest: random
                    n_turb = init_target_x.shape[0]
                    init_xs = np.zeros((n_eval, K, n_turb))
                    init_ys = np.zeros((n_eval, K, n_turb))
                    for ci in range(n_eval):
                        init_xs[ci, 0] = np.array(init_target_x)
                        init_ys[ci, 0] = np.array(init_target_y)
                        if K > 1:
                            init_xs[ci, 1] = np.array(liberal_x)
                            init_ys[ci, 1] = np.array(liberal_y)
                        for ki in range(2, K):
                            seed = hash((float(eval_xs[ci]), float(eval_ys[ci]), ki)) % (2**31)
                            key = jax.random.PRNGKey(seed)
                            rx, ry = self._random_init(key, n_turb)
                            init_xs[ci, ki] = np.array(rx)
                            init_ys[ci, ki] = np.array(ry)
                    init_xs_jnp = jnp.array(init_xs)
                    init_ys_jnp = jnp.array(init_ys)
                else:
                    init_xs_jnp = init_target_x
                    init_ys_jnp = init_target_y

                regrets_batch, opt_xs_batch, opt_ys_batch = \
                    self._evaluate_candidates_batch(
                        eval_xs, eval_ys,
                        placed_x, placed_y,
                        liberal_x, liberal_y,
                        init_xs_jnp, init_ys_jnp,
                        sgd_settings,
                    )

                # Find best (but don't overwrite screening AEP loss
                # in regret_map — the heatmap should show a single
                # consistent metric, not a mix of screening AEP loss
                # and true regret which are on different scales)
                best_local = int(jnp.argmax(regrets_batch))
                best_idx = eval_set[best_local]
                best_regret = float(regrets_batch[best_local])
                best_cx = opt_xs_batch[best_local]
                best_cy = opt_ys_batch[best_local]

                if settings.verbose:
                    _elapsed = _time.time() - _t0
                    n_solves = n_eval * K if K > 1 else n_eval
                    print(f"  Evaluated {n_eval} candidates × {K} starts "
                          f"({n_solves} solves) in parallel: "
                          f"{_elapsed:.1f}s, best = {best_regret:.4f} GWh", flush=True)
            else:
                # Sequential fallback
                best_regret = -np.inf
                best_idx = -1
                best_cx, best_cy = None, None

                for count, idx in enumerate(eval_set):
                    cx, cy = float(grid[idx, 0]), float(grid[idx, 1])

                    regret, cons_x, cons_y = self._evaluate_candidate(
                        cx, cy,
                        placed_x, placed_y,
                        liberal_x, liberal_y,
                        init_target_x, init_target_y,
                        sgd_settings,
                        settings.n_inner_starts,
                    )

                    # Don't overwrite screening AEP loss in regret_map
                    # when screening is active (metrics are on different
                    # scales). Only write regret when no screening was used.
                    if not use_screening:
                        regret_map[idx] = regret

                    if regret > best_regret:
                        best_regret = regret
                        best_idx = idx
                        best_cx, best_cy = cons_x, cons_y

                    if settings.verbose and (count + 1) % 5 == 0:
                        print(f"    Evaluated {count + 1}/{len(eval_set)}, "
                              f"best so far = {best_regret:.4f} GWh", flush=True)

            # Place the winning turbine
            placed_x = jnp.concatenate([placed_x, jnp.array([grid[best_idx, 0]])])
            placed_y = jnp.concatenate([placed_y, jnp.array([grid[best_idx, 1]])])
            remaining.remove(best_idx)
            placement_order.append(best_idx)
            regret_history.append(best_regret)
            regret_maps.append(jnp.array(regret_map))

            if settings.verbose:
                print(f"  Placed turbine at ({grid[best_idx, 0]:.0f}, "
                      f"{grid[best_idx, 1]:.0f}), regret = {best_regret:.4f} GWh", flush=True)

        # Step 3: Final evaluation with all placed neighbors
        def final_objective(x, y):
            return -self._compute_aep(x, y, placed_x, placed_y)

        final_target_x, final_target_y = topfarm_sgd_solve(
            final_objective,
            init_target_x,
            init_target_y,
            self.target_boundary,
            self.target_min_spacing,
            sgd_settings,
        )

        conservative_aep = float(
            self._compute_aep(final_target_x, final_target_y, placed_x, placed_y)
        )
        liberal_aep_present = float(
            self._compute_aep(liberal_x, liberal_y, placed_x, placed_y)
        )
        final_regret = conservative_aep - liberal_aep_present

        if settings.verbose:
            print(f"\nFinal Results:", flush=True)
            print(f"  Liberal AEP (isolated): {liberal_aep:.2f} GWh", flush=True)
            print(f"  Liberal AEP (w/ neighbors): {liberal_aep_present:.2f} GWh", flush=True)
            print(f"  Conservative AEP (w/ neighbors): {conservative_aep:.2f} GWh", flush=True)
            print(f"  Regret: {final_regret:.4f} GWh", flush=True)
            print(f"  Placement order: {placement_order}", flush=True)

        return GreedyGridResult(
            neighbor_x=placed_x,
            neighbor_y=placed_y,
            target_x=final_target_x,
            target_y=final_target_y,
            liberal_x=liberal_x,
            liberal_y=liberal_y,
            regret=final_regret,
            liberal_aep=liberal_aep,
            conservative_aep=conservative_aep,
            placement_order=placement_order,
            regret_history=regret_history,
            regret_maps=regret_maps,
        )

