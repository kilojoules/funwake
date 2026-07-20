"""JAX-native constrained SGD optimizer mirroring TopFarm's EasySGDDriver.

This module provides a fully differentiable SGD solver that enables bilevel
optimization through the Implicit Function Theorem. An adversary can
differentiate through the entire layout optimization process to find
neighbor configurations that maximize regret.

The implementation mirrors TopFarm's SGD with ADAM momentum:
- First moment: m = beta1 * m + (1 - beta1) * grad
- Second moment: v = beta2 * v + (1 - beta2) * grad^2
- Update: x -= lr * m_hat / (sqrt(v_hat) + eps)

Constraints are handled via differentiable penalty functions:
- Boundary constraint: KS aggregation of distance violations
- Spacing constraint: KS aggregation of inter-turbine distance violations

References:
    TopFarm SGD: topfarm/drivers/stochastic_gradient_descent_driver.py
    TopFarm constraints: topfarm/constraint_components/constraint_aggregation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import custom_vjp, vjp
from jax.lax import while_loop

from pixwake.optim.boundary import (
    _signed_distance_to_edge_line,
    containment_penalty as _containment_penalty,
)

# =============================================================================
# SGD State (Optax-style stateless design)
# =============================================================================


class SGDState(NamedTuple):
    """State container for SGD optimizer.

    Follows Optax pattern for stateless optimization.

    Attributes:
        m: First moment estimate (momentum).
        v: Second moment estimate (adaptive scaling).
        iteration: Current iteration count.
        learning_rate: Current learning rate.
        alpha: Current constraint penalty coefficient.
    """

    m_x: jnp.ndarray
    m_y: jnp.ndarray
    v_x: jnp.ndarray
    v_y: jnp.ndarray
    iteration: int
    learning_rate: float
    alpha: float
    alpha0: float
    lr0: float


@dataclass(frozen=True)
class SGDSettings:
    """Configuration for TopFarm-style SGD optimizer.

    The learning rate decays from `learning_rate` to `gamma_min_factor`
    over `max_iter` iterations using the decay function:
        lr_t = lr_{t-1} * 1/(1 + mid * t)

    The `mid` parameter is computed via bisection search to achieve the target
    final learning rate, matching TopFarm's EasySGDDriver behavior.

    Attributes:
        learning_rate: Initial learning rate (default: 10.0).
        gamma_min_factor: Target final learning rate (default: 0.01).
        beta1: First moment decay rate (default: 0.1).
        beta2: Second moment decay rate (default: 0.2).
        max_iter: Maximum number of iterations (default: 3000).
        tol: Convergence tolerance on gradient norm (default: 1e-6).
        mid: Learning rate decay factor. If None, computed via bisection to achieve
            gamma_min_factor. Set explicitly to override bisection.
        bisect_upper: Upper bound for bisection search (default: 0.1).
        bisect_lower: Lower bound for bisection search (default: 0.0).
        ks_rho: KS aggregation smoothness parameter (default: 100.0).
        spacing_weight: Weight for spacing penalty (default: 1.0).
        boundary_weight: Weight for boundary penalty (default: 1.0).
        additional_constant_lr_iterations: Number of initial iterations to run
            at the constant initial learning rate before starting the decay
            schedule. Matches TopFarm's ``additional_constant_lr_iterations``
            option. During these steps ADAM moments accumulate but the
            learning rate and constraint multiplier alpha stay fixed
            (default: 0).
        ift_cg_damping: Tikhonov damping for the CG solver in the IFT
            backward pass (default: 0.01).
        ift_cg_max_iter: Maximum CG iterations in the IFT backward pass
            (default: 100).
    """

    learning_rate: float = 10.0
    gamma_min_factor: float = 0.01
    beta1: float = 0.1
    beta2: float = 0.2
    max_iter: int = 3000
    tol: float = 1e-6
    mid: float | None = None
    bisect_upper: float = 0.1
    bisect_lower: float = 0.0
    ks_rho: float = 100.0
    spacing_weight: float = 1.0
    boundary_weight: float = 1.0
    additional_constant_lr_iterations: int = 0
    ift_cg_damping: float = 1.0
    ift_cg_max_iter: int = 100
    ift_polish_max_iter: int = 20
    ift_polish_tol: float = 1e-8
    # Quick 2023 Algorithm 1 early-stopping.
    # When `early_stopping` is True AND lr_i/lr_0 <= `early_stop_threshold`,
    # the search direction switches to constraint-penalty gradient only
    # (drops the AEP term). When the constraint gradient itself is ~0
    # (sum-of-squares < 1e-20), the while_loop terminates.
    # Paper-reported thresholds: 0.01, 0.05, 0.1 (Sect 4.3, Fig 8).
    # `early_stopping=False` is bit-for-bit identical to the historical impl.
    early_stopping: bool = False
    early_stop_threshold: float = 0.1


def _compute_mid_bisection(
    learning_rate: float,
    gamma_min: float,
    max_iter: int,
    lower: float = 0.0,
    upper: float = 0.1,
    n_bisect_iter: int = 100,
) -> float:
    """Compute the learning rate decay parameter via bisection search.

    Finds `mid` such that after `max_iter` steps of decay:
        lr_t = lr_{t-1} * 1/(1 + mid * t)
    the final learning rate equals `gamma_min`.

    This matches TopFarm's SGDDriver initialization.

    Args:
        learning_rate: Initial learning rate.
        gamma_min: Target final learning rate.
        max_iter: Number of optimization iterations.
        lower: Lower bound for bisection (default: 0.0).
        upper: Upper bound for bisection (default: 0.1).
        n_bisect_iter: Number of bisection iterations (default: 100).

    Returns:
        The computed `mid` parameter.
    """

    def final_lr(mid: float) -> float:
        """Compute final learning rate for a given mid value."""
        lr = learning_rate
        for t in range(1, max_iter + 1):
            lr = lr * 1.0 / (1.0 + mid * t)
        return lr

    # Bisection search
    for _ in range(n_bisect_iter):
        mid = (lower + upper) / 2.0
        lr_final = final_lr(mid)
        if lr_final < gamma_min:
            upper = mid
        else:
            lower = mid

    return mid


# =============================================================================
# Constraint Penalty Functions (JAX-native, differentiable)
# =============================================================================


# Backward-compatible aliases — delegate to boundary.py
_signed_distance_to_edge = _signed_distance_to_edge_line


def boundary_penalty(
    x: jnp.ndarray,
    y: jnp.ndarray,
    boundary_vertices: jnp.ndarray,
    rho: float = 100.0,
) -> jnp.ndarray:
    """Compute boundary constraint penalty matching TopFarm's formulation.

    Delegates to :func:`pixwake.optim.boundary.containment_penalty` (convex path).

    Args:
        x: Turbine x positions, shape (n_turbines,).
        y: Turbine y positions, shape (n_turbines,).
        boundary_vertices: Polygon vertices (CCW order), shape (n_vertices, 2).
        rho: Unused, kept for API compatibility.

    Returns:
        Scalar penalty value (0 if all constraints satisfied).
    """
    return _containment_penalty(x, y, boundary_vertices, convex=True)


def spacing_penalty(
    x: jnp.ndarray,
    y: jnp.ndarray,
    min_spacing: float,
    rho: float = 100.0,
) -> jnp.ndarray:
    """Compute spacing constraint penalty matching TopFarm's formulation.

    Penalizes turbine pairs that are closer than min_spacing.
    Matches TopFarm's DistanceConstraintAggregation which uses:
        sum(-1 * (d² - min_spacing²)) for violated pairs

    This is equivalent to: sum(min_spacing² - d²) where d² < min_spacing²

    Args:
        x: Turbine x positions, shape (n_turbines,).
        y: Turbine y positions, shape (n_turbines,).
        min_spacing: Minimum allowed distance between turbines.
        rho: Unused, kept for API compatibility.

    Returns:
        Scalar penalty value (0 if all constraints satisfied).
    """
    n = x.shape[0]
    if n < 2:
        return jnp.array(0.0)

    # Compute pairwise squared distances
    dx = x[:, None] - x[None, :]  # shape (n, n)
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2

    # Extract upper triangle (unique pairs)
    i_upper, j_upper = jnp.triu_indices(n, k=1)
    pair_dist_sq = dist_sq[i_upper, j_upper]

    # TopFarm uses: sum(min_spacing² - d²) for pairs where d² < min_spacing²
    # Violation: max(0, min_spacing² - d²)
    min_spacing_sq = min_spacing**2
    violations = jnp.maximum(0.0, min_spacing_sq - pair_dist_sq)

    return jnp.sum(violations)


# =============================================================================
# SGD Solver Core
# =============================================================================


def _init_sgd_state(
    x: jnp.ndarray,
    y: jnp.ndarray,
    grad_x: jnp.ndarray,
    grad_y: jnp.ndarray,
    settings: SGDSettings,
) -> SGDState:
    """Initialize SGD state following TopFarm's initialization.

    The initial alpha0 is computed from the initial gradient magnitude
    to balance objective and constraint contributions.

    Args:
        x, y: Initial turbine positions.
        grad_x, grad_y: Initial gradients of the objective.
        settings: SGD configuration.

    Returns:
        Initialized SGDState.
    """
    # Initialize moments to zeros
    m_x = jnp.zeros_like(x)
    m_y = jnp.zeros_like(y)
    v_x = jnp.zeros_like(x)
    v_y = jnp.zeros_like(y)

    # Compute initial alpha0 from gradient magnitude
    # alpha0 = mean(|grad|) / learning_rate (TopFarm: line 135-138)
    grad_mag = jnp.concatenate([jnp.abs(grad_x), jnp.abs(grad_y)])
    alpha0 = jnp.mean(grad_mag) / settings.learning_rate

    return SGDState(
        m_x=m_x,
        m_y=m_y,
        v_x=v_x,
        v_y=v_y,
        iteration=0,
        learning_rate=settings.learning_rate,
        alpha=alpha0,
        alpha0=alpha0,
        lr0=settings.learning_rate,
    )


def _sgd_step(
    x: jnp.ndarray,
    y: jnp.ndarray,
    state: SGDState,
    grad_obj_x: jnp.ndarray,
    grad_obj_y: jnp.ndarray,
    grad_con_x: jnp.ndarray,
    grad_con_y: jnp.ndarray,
    settings: SGDSettings,
) -> tuple[jnp.ndarray, jnp.ndarray, SGDState]:
    """Perform one SGD step with ADAM momentum.

    Implements TopFarm's SGD update rule:
    1. Combine objective and constraint gradients
    2. Update momentum estimates (ADAM)
    3. Apply bias correction
    4. Update positions
    5. Decay learning rate and update alpha

    Args:
        x, y: Current turbine positions.
        state: Current optimizer state.
        grad_obj_x, grad_obj_y: Gradients of the objective function.
        grad_con_x, grad_con_y: Gradients of the constraint penalty.
        settings: SGD configuration.

    Returns:
        Updated (x, y, state).
    """
    beta1, beta2 = settings.beta1, settings.beta2
    it = state.iteration + 1

    # Combined gradient: grad = grad_obj + alpha * grad_con
    jacobian_x = grad_obj_x + state.alpha * grad_con_x
    jacobian_y = grad_obj_y + state.alpha * grad_con_y

    # ADAM first moment update: m = beta1 * m + (1 - beta1) * grad
    m_x = beta1 * state.m_x + (1 - beta1) * jacobian_x
    m_y = beta1 * state.m_y + (1 - beta1) * jacobian_y

    # ADAM second moment update: v = beta2 * v + (1 - beta2) * grad^2
    v_x = beta2 * state.v_x + (1 - beta2) * jacobian_x**2
    v_y = beta2 * state.v_y + (1 - beta2) * jacobian_y**2

    # Bias correction
    m_hat_x = m_x / (1 - beta1**it)
    m_hat_y = m_y / (1 - beta1**it)
    v_hat_x = v_x / (1 - beta2**it)
    v_hat_y = v_y / (1 - beta2**it)

    # Position update: x -= lr * m_hat / (sqrt(v_hat) + eps)
    eps = 1e-12
    x_new = x - state.learning_rate * m_hat_x / (jnp.sqrt(v_hat_x) + eps)
    y_new = y - state.learning_rate * m_hat_y / (jnp.sqrt(v_hat_y) + eps)

    # During the constant-LR phase, keep lr and alpha fixed (TopFarm behavior:
    # iter_count is not incremented, so decay factor is 1/(1+mid*0) = 1).
    n_const = settings.additional_constant_lr_iterations
    mid = settings.mid if settings.mid is not None else 1.0 / settings.max_iter
    decaying = it > n_const
    decay_it = jnp.where(decaying, it - n_const, 0)

    # Learning rate decay: lr *= 1 / (1 + mid * decay_iter)
    new_lr = state.learning_rate * 1.0 / (1 + mid * decay_it)

    # Alpha update: alpha = alpha0 * lr0 / lr
    new_alpha = jnp.where(decaying, state.alpha0 * state.lr0 / new_lr, state.alpha)

    new_state = SGDState(
        m_x=m_x,
        m_y=m_y,
        v_x=v_x,
        v_y=v_y,
        iteration=it,
        learning_rate=new_lr,
        alpha=new_alpha,
        alpha0=state.alpha0,
        lr0=state.lr0,
    )

    return x_new, y_new, new_state


# =============================================================================
# Main Solver with jax.lax.while_loop
# =============================================================================


def topfarm_sgd_solve(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x: jnp.ndarray,
    init_y: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve constrained layout optimization using TopFarm-style SGD.

    This solver uses jax.lax.while_loop for JIT compatibility and enables
    implicit differentiation through the convergence point.

    Args:
        objective_fn: Function (x, y) -> scalar to minimize (e.g., negative AEP).
        init_x: Initial turbine x positions, shape (n_turbines,).
        init_y: Initial turbine y positions, shape (n_turbines,).
        boundary: Polygon vertices (CCW), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance.
        settings: SGD configuration. Uses defaults if None.

    Returns:
        Tuple of (optimized_x, optimized_y).

    Example:
        >>> def neg_aep(x, y):
        ...     result = sim(x, y, ws_amb=ws, wd_amb=wd)
        ...     return -result.aep()
        >>> opt_x, opt_y = topfarm_sgd_solve(neg_aep, x0, y0, boundary, min_spacing)
    """
    if settings is None:
        settings = SGDSettings()

    # Compute mid via bisection if not explicitly provided
    if settings.mid is None:
        # TopFarm uses gamma_min = gamma_min_factor directly (not multiplied by learning_rate)
        gamma_min = settings.gamma_min_factor
        computed_mid = _compute_mid_bisection(
            learning_rate=settings.learning_rate,
            gamma_min=gamma_min,
            max_iter=settings.max_iter,
            lower=settings.bisect_lower,
            upper=settings.bisect_upper,
        )
        # Create new settings with computed mid (preserve all other fields)
        from dataclasses import replace as _dc_replace
        settings = _dc_replace(settings, mid=computed_mid)

    rho = settings.ks_rho

    def total_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Augmented objective including constraint penalties."""
        obj = objective_fn(x, y)
        pen_boundary = settings.boundary_weight * boundary_penalty(x, y, boundary, rho)
        pen_spacing = settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho)
        return obj + pen_boundary + pen_spacing

    def constraint_penalty(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Combined constraint penalty for gradient computation."""
        return settings.boundary_weight * boundary_penalty(
            x, y, boundary, rho
        ) + settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho)

    # Compute initial gradients for state initialization
    grad_obj_fn = jax.grad(objective_fn, argnums=(0, 1))
    grad_con_fn = jax.grad(constraint_penalty, argnums=(0, 1))

    init_grad_obj_x, init_grad_obj_y = grad_obj_fn(init_x, init_y)
    init_state = _init_sgd_state(
        init_x, init_y, init_grad_obj_x, init_grad_obj_y, settings
    )

    # State for while_loop: (x, y, sgd_state, prev_x, prev_y)
    # Total iterations = additional constant LR steps + max_iter decaying steps
    total_iter = settings.max_iter + settings.additional_constant_lr_iterations

    # Quick 2023 Algorithm 1 early-stopping configuration. The carry adds a
    # scalar boolean `es_terminated` so the while_loop can break when the
    # constraint gradient is ~0 in the ES regime. With `early_stopping=False`,
    # the ES branch is never selected and the carry field stays False — the
    # numerical path is bit-identical to the historical implementation.
    es_enabled = bool(settings.early_stopping)
    es_threshold = float(settings.early_stop_threshold)

    def cond_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, SGDState, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        x, y, state, prev_x, prev_y, es_terminated = carry
        # Continue if not converged and under total iterations
        change = jnp.max(jnp.abs(x - prev_x)) + jnp.max(jnp.abs(y - prev_y))
        not_converged = change > settings.tol
        under_max_iter = state.iteration < total_iter
        not_es_terminated = jnp.logical_not(es_terminated)
        return jnp.logical_and(
            jnp.logical_and(not_converged, under_max_iter), not_es_terminated
        )

    def body_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, SGDState, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, SGDState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x, y, state, _, _, es_terminated_prev = carry

        # Compute gradients
        grad_obj_x, grad_obj_y = grad_obj_fn(x, y)
        grad_con_x, grad_con_y = grad_con_fn(x, y)

        # Quick 2023 Alg. 1 — drop the AEP gradient when ES is active so the
        # update follows the constraint penalty only.
        if es_enabled:
            lr_ratio = state.learning_rate / state.lr0
            es_active = lr_ratio <= es_threshold
            zero_x = jnp.zeros_like(grad_obj_x)
            zero_y = jnp.zeros_like(grad_obj_y)
            eff_obj_x = jnp.where(es_active, zero_x, grad_obj_x)
            eff_obj_y = jnp.where(es_active, zero_y, grad_obj_y)
        else:
            es_active = jnp.array(False)
            eff_obj_x = grad_obj_x
            eff_obj_y = grad_obj_y

        # SGD step
        x_new, y_new, new_state = _sgd_step(
            x, y, state, eff_obj_x, eff_obj_y, grad_con_x, grad_con_y, settings
        )

        # |j| ~ 0 break: in the ES regime, j == alpha * grad_con; alpha > 0,
        # so equivalent to |grad_con| ~ 0. Use a tight threshold on the
        # raw constraint-gradient sum-of-squares.
        if es_enabled:
            con_norm_sq = jnp.sum(grad_con_x**2) + jnp.sum(grad_con_y**2)
            es_terminated = jnp.logical_and(es_active, con_norm_sq < 1e-20)
            es_terminated = jnp.logical_or(es_terminated_prev, es_terminated)
        else:
            es_terminated = es_terminated_prev

        return (x_new, y_new, new_state, x, y, es_terminated)

    # Run optimization loop
    init_es_terminated = jnp.array(False)
    init_carry = (
        init_x,
        init_y,
        init_state,
        init_x - 1.0,
        init_y - 1.0,
        init_es_terminated,
    )
    final_x, final_y, _, _, _, _ = while_loop(cond_fn, body_fn, init_carry)

    return final_x, final_y


# =============================================================================
# Newton-CG Polishing (drives gradient residual to zero for IFT)
# =============================================================================


def _newton_polish(
    total_obj_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    max_iter: int = 20,
    tol: float = 1e-8,
    cg_max_iter: int = 50,
    damping: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gradient descent polishing to drive gradient residual toward zero.

    After SGD converges (position change < tol), the gradient residual may
    still be large (~0.01) due to the decaying learning rate.  Constant-LR
    gradient descent steps can reduce |grad|, improving the IFT assumption
    grad L(x*, params) ≈ 0.

    Uses backtracking line search for step size selection.
    Python-level loop (runs at trace time inside custom_vjp forward pass).
    """
    grad_fn = jax.grad(total_obj_fn, argnums=(0, 1))
    best_norm = float('inf')
    best_x, best_y = x, y

    for it in range(max_iter):
        gx, gy = grad_fn(x, y)
        grad_norm = float(jnp.sqrt(jnp.sum(gx**2) + jnp.sum(gy**2)))

        if grad_norm < best_norm:
            best_norm = grad_norm
            best_x, best_y = x, y

        if it == 0 or it % 5 == 0 or grad_norm < tol:
            print(f"    Polish iter {it}: |grad| = {grad_norm:.6e}", flush=True)

        if grad_norm < tol:
            break

        # Backtracking line search along -gradient
        obj_current = float(total_obj_fn(x, y))
        gnorm_sq = float(jnp.sum(gx**2) + jnp.sum(gy**2))
        step = 1.0  # start with step=1, halve until Armijo satisfied
        for _ in range(30):
            obj_new = float(total_obj_fn(x - step * gx, y - step * gy))
            if obj_new <= obj_current - 1e-4 * step * gnorm_sq:
                break
            step *= 0.5
        else:
            break  # line search failed — stop polishing

        x = x - step * gx
        y = y - step * gy

    print(f"    Polish done: best |grad| = {best_norm:.6e}", flush=True)
    return best_x, best_y


# =============================================================================
# Implicit Differentiation Wrapper
# =============================================================================


@partial(custom_vjp, nondiff_argnums=(0, 3, 4, 5))
def sgd_solve_implicit(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x: jnp.ndarray,
    init_y: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """SGD solver with implicit differentiation support.

    This wrapper enables differentiating through the optimization process
    with respect to external parameters (e.g., neighbor turbine positions)
    using the Implicit Function Theorem. This is essential for bilevel
    optimization where an outer adversary wants to find parameters that
    maximize regret after inner layout optimization.

    The key insight is that at convergence, the optimality conditions hold:
        grad_x L(x*, y*, params) = 0
        grad_y L(x*, y*, params) = 0

    Differentiating these conditions with respect to params gives us the
    gradient of the optimal solution (x*, y*) with respect to params without
    unrolling the 3000 SGD iterations.

    Args:
        objective_fn: Function (x, y, params) -> scalar to minimize.
        init_x: Initial turbine x positions.
        init_y: Initial turbine y positions.
        boundary: Polygon vertices (CCW order).
        min_spacing: Minimum inter-turbine distance.
        settings: SGD configuration.
        params: External parameters to differentiate with respect to
            (e.g., neighbor positions).

    Returns:
        Tuple of (optimized_x, optimized_y).

    Example:
        >>> def objective_with_neighbors(x, y, neighbor_pos):
        ...     neighbor_x, neighbor_y = neighbor_pos[:n_neighbors], neighbor_pos[n_neighbors:]
        ...     x_all = jnp.concatenate([x, neighbor_x])
        ...     y_all = jnp.concatenate([y, neighbor_y])
        ...     result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        ...     target_aep = result.aep()[:n_target]
        ...     return -jnp.sum(target_aep)
        >>>
        >>> # Optimize target layout given neighbor positions
        >>> opt_x, opt_y = sgd_solve_implicit(
        ...     objective_with_neighbors, init_x, init_y, boundary, min_spacing, settings, neighbor_pos
        ... )
        >>>
        >>> # Differentiate final AEP with respect to neighbor positions
        >>> def final_aep(neighbor_pos):
        ...     opt_x, opt_y = sgd_solve_implicit(
        ...         objective_with_neighbors, init_x, init_y, boundary, min_spacing, settings, neighbor_pos
        ...     )
        ...     # ... compute final AEP ...
        >>> grad_neighbor = jax.grad(final_aep)(neighbor_pos)
    """
    if settings is None:
        settings = SGDSettings()

    opt_x, opt_y = _sgd_and_polish(
        objective_fn, init_x, init_y, boundary, min_spacing, settings, params,
    )
    return opt_x, opt_y


def _sgd_and_polish(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x: jnp.ndarray,
    init_y: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run SGD then Newton-CG polishing to satisfy IFT optimality condition."""
    rho = settings.ks_rho

    def obj_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return objective_fn(x, y, params)

    opt_x, opt_y = topfarm_sgd_solve(
        obj_fn, init_x, init_y, boundary, min_spacing, settings,
    )

    # Newton-CG polishing: drive grad(total_obj) → 0 for IFT
    if settings.ift_polish_max_iter > 0:
        def total_obj(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return (
                obj_fn(x, y)
                + settings.boundary_weight * boundary_penalty(x, y, boundary, rho)
                + settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho)
            )

        opt_x, opt_y = _newton_polish(
            total_obj, opt_x, opt_y,
            max_iter=settings.ift_polish_max_iter,
            tol=settings.ift_polish_tol,
            cg_max_iter=settings.ift_cg_max_iter,
            damping=settings.ift_cg_damping,
        )

    return opt_x, opt_y


def _sgd_solve_implicit_fwd(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x: jnp.ndarray,
    init_y: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None,
    params: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple]:
    """Forward pass for implicit differentiation."""
    if settings is None:
        settings = SGDSettings()

    opt_x, opt_y = _sgd_and_polish(
        objective_fn, init_x, init_y, boundary, min_spacing, settings, params,
    )
    return (opt_x, opt_y), (opt_x, opt_y, params)


def _sgd_solve_implicit_bwd(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None,
    res: tuple,
    g: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward pass using implicit function theorem.

    At optimality, we have:
        grad_x L(x*, y*, params) = 0
        grad_y L(x*, y*, params) = 0

    Differentiating with respect to params:
        H @ d(x*,y*)/d(params) + grad_params grad_(x,y) L = 0

    Where H is the Hessian of L with respect to (x, y).

    We solve for the gradient using the conjugate gradient method
    or a fixed-point iteration.
    """
    if settings is None:
        settings = SGDSettings()
    opt_x, opt_y, params = res
    g_x, g_y = g
    rho = settings.ks_rho

    # Total objective including penalties
    def total_obj(x: jnp.ndarray, y: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        obj = objective_fn(x, y, p)
        pen_b = settings.boundary_weight * boundary_penalty(x, y, boundary, rho)
        pen_s = settings.spacing_weight * spacing_penalty(x, y, min_spacing, rho)
        return obj + pen_b + pen_s

    # Compute Jacobian of optimality conditions with respect to params
    # Using the implicit function theorem:
    # d(x*,y*)/d(params) = -H^{-1} @ (d^2 L / d(x,y) d(params))
    #
    # Uses pure AD via jax.jacfwd(jax.grad(...)) for the cross-Jacobian and
    # jax.jvp(jax.grad(...)) for Hessian-vector products. This is possible
    # because both fixed_point_fwd and fixed_point_rev now use
    # _fixed_point_raw (no custom_vjp), allowing JVP to propagate through
    # the VJP trace of the wake simulation.

    def grad_xy_fn(p: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute gradient of total_obj w.r.t. (x, y) at given params."""
        return jax.grad(lambda x, y: total_obj(x, y, p), argnums=(0, 1))(opt_x, opt_y)

    # Cross-Jacobian d(grad_xy)/d(params) via forward-over-reverse AD (jacfwd)
    # This works because fixed_point_fwd now calls _fixed_point_raw (no custom_vjp),
    # so JVP can propagate through the VJP trace of the wake simulation.
    jac = jax.jacfwd(grad_xy_fn)(params)
    jac_x = jac[0]  # shape (n_turbines, n_params)
    jac_y = jac[1]  # shape (n_turbines, n_params)

    # Hessian-vector product via forward-over-reverse AD (jvp of grad)
    def hvp(vx: jnp.ndarray, vy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Exact Hessian-vector product via jax.jvp(jax.grad(...))."""
        def grad_obj(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return jax.grad(lambda xx, yy: total_obj(xx, yy, params), argnums=(0, 1))(x, y)

        _, (hvp_x, hvp_y) = jax.jvp(grad_obj, (opt_x, opt_y), (vx, vy))
        return hvp_x, hvp_y

    # Solve (H + damping*I) @ v = g using truncated Conjugate Gradient (CG)
    # Tikhonov damping regularizes near-singular Hessians.
    # Truncated CG: if negative curvature is detected (p^T A p <= 0),
    # we stop and return the current iterate. This handles indefinite
    # Hessians that arise when the inner solver hasn't fully converged.
    damping = settings.ift_cg_damping
    cg_max_iter = settings.ift_cg_max_iter

    def solve_linear_system(
        g_x: jnp.ndarray, g_y: jnp.ndarray, max_iter: int = cg_max_iter, tol: float = 1e-6
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Solve (H + damping*I) @ v = g using truncated Conjugate Gradient."""

        def damped_hvp(vx: jnp.ndarray, vy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            hx, hy = hvp(vx, vy)
            return hx + damping * vx, hy + damping * vy

        def dot_xy(ax: jnp.ndarray, ay: jnp.ndarray, bx: jnp.ndarray, by: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(ax * bx) + jnp.sum(ay * by)

        # CG state: (v_x, v_y, r_x, r_y, p_x, p_y, rs_old, iteration)
        def cond_fn(carry):
            _, _, _, _, _, _, rs_old, it = carry
            # Continue while residual is large, iteration count < max,
            # and residual is finite (stops on NaN/Inf from divergence)
            return jnp.logical_and(
                jnp.logical_and(rs_old > tol**2, it < max_iter),
                jnp.isfinite(rs_old),
            )

        def body_fn(carry):
            v_x, v_y, r_x, r_y, p_x, p_y, rs_old, it = carry
            # A @ p
            ap_x, ap_y = damped_hvp(p_x, p_y)
            # p^T A p — must be positive for CG to work
            pap = dot_xy(p_x, p_y, ap_x, ap_y)
            # Truncated CG: if pap <= 0 (negative curvature), stop by
            # setting residual to 0 (triggers cond_fn exit) and keeping
            # current v unchanged.
            neg_curv = pap <= 1e-30
            # When negative curvature detected, use alpha=0 to freeze v
            alpha = jnp.where(neg_curv, 0.0, rs_old / jnp.maximum(pap, 1e-30))
            # update solution and residual
            v_x_new = v_x + alpha * p_x
            v_y_new = v_y + alpha * p_y
            r_x_new = r_x - alpha * ap_x
            r_y_new = r_y - alpha * ap_y
            rs_new = dot_xy(r_x_new, r_y_new, r_x_new, r_y_new)
            # Force exit on negative curvature by setting rs_new = 0
            rs_new = jnp.where(neg_curv, 0.0, rs_new)
            # update search direction
            beta = rs_new / jnp.maximum(rs_old, 1e-30)
            p_x_new = r_x_new + beta * p_x
            p_y_new = r_y_new + beta * p_y
            return (v_x_new, v_y_new, r_x_new, r_y_new, p_x_new, p_y_new, rs_new, it + 1)

        # Initial guess: v = 0, r = g - A@0 = g, p = r
        v0_x = jnp.zeros_like(g_x)
        v0_y = jnp.zeros_like(g_y)
        r0_x, r0_y = g_x, g_y
        p0_x, p0_y = g_x, g_y
        rs0 = dot_xy(r0_x, r0_y, r0_x, r0_y)
        init_carry = (v0_x, v0_y, r0_x, r0_y, p0_x, p0_y, rs0, 0)
        v_x, v_y, _, _, _, _, _, _ = while_loop(cond_fn, body_fn, init_carry)
        return v_x, v_y

    # Solve for the adjoint vector
    adj_x, adj_y = solve_linear_system(g_x, g_y)

    # Compute gradient with respect to params using the Jacobian
    # grad_params = adj_x^T @ jac_x + adj_y^T @ jac_y
    grad_params = jnp.sum(adj_x[:, None] * jac_x, axis=0) + jnp.sum(
        adj_y[:, None] * jac_y, axis=0
    )

    # Gradients with respect to init_x, init_y are zero (fixed point doesn't depend on initial guess)
    return (jnp.zeros_like(opt_x), jnp.zeros_like(opt_y), -grad_params)


sgd_solve_implicit.defvjp(_sgd_solve_implicit_fwd, _sgd_solve_implicit_bwd)


# =============================================================================
# Multistart Helpers
# =============================================================================


def generate_random_starts(
    key: jnp.ndarray,
    k: int,
    n_turbines: int,
    boundary: jnp.ndarray,
    min_spacing: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate K random initial turbine layouts within a boundary polygon.

    Uses uniform sampling within the bounding box of the polygon with a
    margin of min_spacing/2 from the edges.

    Args:
        key: JAX PRNG key.
        k: Number of random starts to generate.
        n_turbines: Number of turbines per layout.
        boundary: Polygon vertices (CCW), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance (used for margin).

    Returns:
        Tuple of (init_x_batch, init_y_batch), each shape (k, n_turbines).
    """
    x_min = float(boundary[:, 0].min())
    x_max = float(boundary[:, 0].max())
    y_min = float(boundary[:, 1].min())
    y_max = float(boundary[:, 1].max())

    margin = min_spacing / 2
    keys = jax.random.split(key, 2 * k)

    xs = []
    ys = []
    for i in range(k):
        x = jax.random.uniform(
            keys[2 * i], (n_turbines,),
            minval=x_min + margin, maxval=x_max - margin,
        )
        y = jax.random.uniform(
            keys[2 * i + 1], (n_turbines,),
            minval=y_min + margin, maxval=y_max - margin,
        )
        xs.append(x)
        ys.append(y)

    return jnp.stack(xs), jnp.stack(ys)


# =============================================================================
# Multistart SGD Solver
# =============================================================================


def topfarm_sgd_solve_multistart(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x_batch: jnp.ndarray,
    init_y_batch: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve layout optimization with K parallel starts via vmap.

    Runs topfarm_sgd_solve for each of the K initial layouts in parallel.
    The ``mid`` parameter is precomputed once (pure Python bisection) and
    passed explicitly so no trace-time computation occurs inside vmap.

    Args:
        objective_fn: Function (x, y) -> scalar to minimize.
        init_x_batch: Initial x positions, shape (K, n_turbines).
        init_y_batch: Initial y positions, shape (K, n_turbines).
        boundary: Polygon vertices (CCW), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance.
        settings: SGD configuration.

    Returns:
        Tuple of (all_x, all_y, all_objs) where:
            all_x: shape (K, n_turbines) — optimized x for each start
            all_y: shape (K, n_turbines) — optimized y for each start
            all_objs: shape (K,) — objective value at each optimum
    """
    if settings is None:
        settings = SGDSettings()

    # Precompute mid ONCE (pure Python, not traceable)
    if settings.mid is None:
        gamma_min = settings.gamma_min_factor
        computed_mid = _compute_mid_bisection(
            learning_rate=settings.learning_rate,
            gamma_min=gamma_min,
            max_iter=settings.max_iter,
            lower=settings.bisect_lower,
            upper=settings.bisect_upper,
        )
        from dataclasses import replace as _dc_replace
        settings = _dc_replace(settings, mid=computed_mid)

    def solve_one(init_x: jnp.ndarray, init_y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return topfarm_sgd_solve(
            objective_fn, init_x, init_y, boundary, min_spacing, settings
        )

    all_x, all_y = jax.vmap(solve_one)(init_x_batch, init_y_batch)

    # Evaluate objective at each optimum
    all_objs = jax.vmap(objective_fn)(all_x, all_y)

    return all_x, all_y, all_objs


# =============================================================================
# Multistart Implicit Differentiation (Envelope Theorem)
# =============================================================================


@partial(custom_vjp, nondiff_argnums=(0, 4, 5, 6))
def sgd_solve_implicit_multistart(
    objective_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_x_batch: jnp.ndarray,
    init_y_batch: jnp.ndarray,
    params: jnp.ndarray,
    boundary: jnp.ndarray,
    min_spacing: float,
    settings: SGDSettings | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multistart SGD solver with implicit differentiation (envelope theorem).

    Forward pass: vmaps K inner solves, picks the winner (lowest objective).
    Backward pass: IFT through the winning start only. The envelope theorem
    guarantees correctness when the winning start is locally stable:
        d/dp min_k f_k(p) = d/dp f_{k*}(p)

    Args:
        objective_fn: Function (x, y, params) -> scalar to minimize.
        init_x_batch: Initial x positions, shape (K, n_turbines).
        init_y_batch: Initial y positions, shape (K, n_turbines).
        params: External parameters to differentiate w.r.t.
        boundary: Polygon vertices (CCW), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance.
        settings: SGD configuration.

    Returns:
        Tuple of (winner_x, winner_y) — optimized layout from best start.
    """
    if settings is None:
        settings = SGDSettings()

    # Precompute mid
    if settings.mid is None:
        gamma_min = settings.gamma_min_factor
        computed_mid = _compute_mid_bisection(
            learning_rate=settings.learning_rate,
            gamma_min=gamma_min,
            max_iter=settings.max_iter,
            lower=settings.bisect_lower,
            upper=settings.bisect_upper,
        )
        from dataclasses import replace as _dc_replace
        settings = _dc_replace(settings, mid=computed_mid)

    def obj_fn(x, y):
        return objective_fn(x, y, params)

    def solve_one(init_x, init_y):
        return topfarm_sgd_solve(obj_fn, init_x, init_y, boundary, min_spacing, settings)

    all_x, all_y = jax.vmap(solve_one)(init_x_batch, init_y_batch)
    all_objs = jax.vmap(obj_fn)(all_x, all_y)
    k_star = jnp.argmin(all_objs)

    return all_x[k_star], all_y[k_star]


def _sgd_solve_implicit_multistart_fwd(
    objective_fn, init_x_batch, init_y_batch, params,
    boundary, min_spacing, settings,
):
    """Forward pass for multistart implicit differentiation."""
    if settings is None:
        settings = SGDSettings()

    if settings.mid is None:
        gamma_min = settings.gamma_min_factor
        computed_mid = _compute_mid_bisection(
            learning_rate=settings.learning_rate,
            gamma_min=gamma_min,
            max_iter=settings.max_iter,
            lower=settings.bisect_lower,
            upper=settings.bisect_upper,
        )
        from dataclasses import replace as _dc_replace
        settings = _dc_replace(settings, mid=computed_mid)

    def obj_fn(x, y):
        return objective_fn(x, y, params)

    def solve_one(init_x, init_y):
        return topfarm_sgd_solve(obj_fn, init_x, init_y, boundary, min_spacing, settings)

    all_x, all_y = jax.vmap(solve_one)(init_x_batch, init_y_batch)
    all_objs = jax.vmap(obj_fn)(all_x, all_y)
    k_star = jnp.argmin(all_objs)

    winner_x, winner_y = all_x[k_star], all_y[k_star]
    # Store residuals: winner solution + params + batch shape for zero grads
    return (winner_x, winner_y), (winner_x, winner_y, params, init_x_batch, init_y_batch)


def _sgd_solve_implicit_multistart_bwd(
    objective_fn, boundary, min_spacing, settings, res, g,
):
    """Backward pass: IFT through winning start only (envelope theorem).

    Reuses the exact same IFT backward logic as single-start
    ``_sgd_solve_implicit_bwd``. The envelope theorem guarantees this is
    correct: d/dp min_k f_k(p) = d/dp f_{k*}(p).
    """
    winner_x, winner_y, params, init_x_batch, init_y_batch = res
    # Delegate to the single-start backward — same math applies at the winner
    single_res = (winner_x, winner_y, params)
    _, _, grad_params = _sgd_solve_implicit_bwd(
        objective_fn, boundary, min_spacing, settings, single_res, g,
    )
    # Gradients w.r.t. init_x_batch and init_y_batch are zero
    # (fixed point doesn't depend on initial guess)
    return (
        jnp.zeros_like(init_x_batch),
        jnp.zeros_like(init_y_batch),
        grad_params,
    )


sgd_solve_implicit_multistart.defvjp(
    _sgd_solve_implicit_multistart_fwd,
    _sgd_solve_implicit_multistart_bwd,
)


# =============================================================================
# Convenience Wrapper for Common Use Case
# =============================================================================


def create_layout_optimizer(
    sim_engine,
    boundary: jnp.ndarray,
    min_spacing: float,
    ws_amb: jnp.ndarray | float,
    wd_amb: jnp.ndarray | float,
    ti_amb: jnp.ndarray | float | None = None,
    weights: jnp.ndarray | None = None,
    settings: SGDSettings | None = None,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Create a layout optimization function using TopFarm-style SGD.

    This is a convenience wrapper that creates a solver configured for
    AEP maximization with boundary and spacing constraints.

    Args:
        sim_engine: WakeSimulation instance.
        boundary: Polygon vertices (CCW order), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance.
        ws_amb: Ambient wind speed(s).
        wd_amb: Wind direction(s).
        ti_amb: Ambient turbulence intensity (optional).
        weights: Probability weights for each wind condition (optional).
        settings: SGD configuration.

    Returns:
        Function (init_x, init_y) -> (opt_x, opt_y).

    Example:
        >>> optimizer = create_layout_optimizer(sim, boundary, 200.0, ws, wd)
        >>> opt_x, opt_y = optimizer(init_x, init_y)
    """
    if settings is None:
        settings = SGDSettings()

    def neg_aep(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        result = sim_engine(x, y, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)
        if weights is not None:
            # Weighted AEP
            power = result.power()  # (n_cases, n_turbines)
            weighted_power = jnp.sum(power * weights[:, None])
            return -weighted_power * 8760 / 1e6  # Convert to GWh
        return -result.aep()

    def optimize(
        init_x: jnp.ndarray, init_y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return topfarm_sgd_solve(
            neg_aep, init_x, init_y, boundary, min_spacing, settings
        )

    return optimize


def create_bilevel_optimizer(
    sim_engine,
    target_boundary: jnp.ndarray,
    min_spacing: float,
    ws_amb: jnp.ndarray | float,
    wd_amb: jnp.ndarray | float,
    ti_amb: jnp.ndarray | float | None = None,
    weights: jnp.ndarray | None = None,
    settings: SGDSettings | None = None,
    n_target: int | None = None,
) -> Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]:
    """Create a bilevel layout optimizer for adversarial discovery.

    This function creates a solver that optimizes target farm layout while
    accounting for neighbor turbine wake effects. The solver supports implicit
    differentiation, allowing an outer optimizer to compute gradients of the
    optimized target AEP with respect to neighbor positions.

    Args:
        sim_engine: WakeSimulation instance.
        target_boundary: Target farm polygon vertices (CCW), shape (n_vertices, 2).
        min_spacing: Minimum inter-turbine distance.
        ws_amb: Ambient wind speed(s).
        wd_amb: Wind direction(s).
        ti_amb: Ambient turbulence intensity (optional).
        weights: Probability weights for each wind condition (optional).
        settings: SGD configuration.
        n_target: Number of target turbines (inferred if None).

    Returns:
        Function (init_x, init_y, neighbor_x, neighbor_y) -> (opt_x, opt_y).

    Example:
        >>> bilevel_opt = create_bilevel_optimizer(sim, boundary, 200.0, ws, wd)
        >>>
        >>> def adversarial_objective(neighbor_x, neighbor_y):
        ...     opt_x, opt_y = bilevel_opt(init_x, init_y, neighbor_x, neighbor_y)
        ...     # Compute final target AEP with neighbors
        ...     x_all = jnp.concatenate([opt_x, neighbor_x])
        ...     y_all = jnp.concatenate([opt_y, neighbor_y])
        ...     result = sim(x_all, y_all, ws_amb=ws, wd_amb=wd)
        ...     return -result.aep()[:n_target].sum()  # Minimize target AEP
        >>>
        >>> # Gradient of optimized AEP w.r.t. neighbor positions
        >>> grad_x, grad_y = jax.grad(adversarial_objective, argnums=(0, 1))(
        ...     neighbor_x, neighbor_y
        ... )
    """
    if settings is None:
        settings = SGDSettings()

    def objective_with_neighbors(
        x: jnp.ndarray, y: jnp.ndarray, neighbor_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Objective function with neighbor parameters."""
        n_neighbors = neighbor_params.shape[0] // 2
        neighbor_x = neighbor_params[:n_neighbors]
        neighbor_y = neighbor_params[n_neighbors:]

        # Combine target and neighbor positions
        x_all = jnp.concatenate([x, neighbor_x])
        y_all = jnp.concatenate([y, neighbor_y])

        # Simulate
        result = sim_engine(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb, ti_amb=ti_amb)

        # Target AEP only (first n_target turbines)
        _n_target = n_target if n_target is not None else x.shape[0]
        power = result.power()[:, :_n_target]

        if weights is not None:
            weighted_power = jnp.sum(power * weights[:, None])
            return -weighted_power * 8760 / 1e6
        return -jnp.sum(power) * 8760 / 1e6 / power.shape[0]

    def optimize_with_neighbors(
        init_x: jnp.ndarray,
        init_y: jnp.ndarray,
        neighbor_x: jnp.ndarray,
        neighbor_y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Optimize target layout given neighbor positions."""
        neighbor_params = jnp.concatenate([neighbor_x, neighbor_y])
        return sgd_solve_implicit(
            objective_with_neighbors,
            init_x,
            init_y,
            target_boundary,
            min_spacing,
            settings,
            neighbor_params,
        )

    return optimize_with_neighbors
