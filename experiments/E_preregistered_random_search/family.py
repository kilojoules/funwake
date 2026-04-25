"""Pre-registered Fourier-basis schedule family (v2).

Authored before inspecting LLM-generated schedule outputs for v2-design
purposes. The post-hoc family lives at tools/random_search_ablation.py
and is intentionally NOT imported here.

family_render(coeffs) -> Python source string for a schedule_fn module.

Coefficients layout (per output, in this order: lr, alpha, beta1, beta2):
    c0, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4
i.e. 9 floats per output, 36 total per sample.

Locked: do not edit after first checked-in version. To revise the
family, create family_v3.py and a new PREREGISTRATION_v3.md.
"""
from typing import Sequence

K = 4  # Fourier components per output
OUTPUTS = ("lr", "alpha", "beta1", "beta2")
N_PER_OUTPUT = 1 + 2 * K   # c0 + (a_k, b_k for k=1..K)
N_TOTAL = N_PER_OUTPUT * len(OUTPUTS)


def _format_floats(xs):
    return ", ".join(f"{x:.6f}" for x in xs)


SCHEDULE_TEMPLATE = '''"""Pre-registered random-search sample {sample_id} (Fourier family v2).

Coefficients (per output: c0, a_1..a_K, b_1..b_K with K={K}):
{coeff_doc}
"""
import jax.numpy as jnp


def _envelope(c0, a, b, t):
    K = {K}
    s = c0
    for k in range(1, K + 1):
        s = s + a[k - 1] * jnp.cos(2.0 * jnp.pi * k * t) + b[k - 1] * jnp.sin(2.0 * jnp.pi * k * t)
    return jnp.exp(s)


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    a_lr     = jnp.array([{a_lr}])
    b_lr     = jnp.array([{b_lr}])
    a_alpha  = jnp.array([{a_alpha}])
    b_alpha  = jnp.array([{b_alpha}])
    a_b1     = jnp.array([{a_b1}])
    b_b1     = jnp.array([{b_b1}])
    a_b2     = jnp.array([{a_b2}])
    b_b2     = jnp.array([{b_b2}])

    lr     = lr0    * _envelope({c0_lr},    a_lr,    b_lr,    t)
    alpha  = alpha0 * _envelope({c0_alpha}, a_alpha, b_alpha, t)
    beta1  = 0.9    * _envelope({c0_b1},    a_b1,    b_b1,    t)
    beta2  = 0.999  * _envelope({c0_b2},    a_b2,    b_b2,    t)

    lr    = jnp.maximum(lr, 1e-10)
    alpha = jnp.maximum(alpha, 1e-12)
    beta1 = jnp.clip(beta1, 0.5,  0.999)
    beta2 = jnp.clip(beta2, 0.9,  0.9999)

    return lr, alpha, beta1, beta2
'''


def render(sample_id: int, coeffs: Sequence[float]) -> str:
    """Render coefficient vector to a schedule_fn source string.

    coeffs must have length N_TOTAL = 36, ordered as
    [c0_lr, a_lr_1..K, b_lr_1..K,  c0_alpha, ...,  c0_b1, ...,  c0_b2, ...].
    """
    if len(coeffs) != N_TOTAL:
        raise ValueError(f"Expected {N_TOTAL} coeffs, got {len(coeffs)}")

    # Slice
    offsets = {name: i * N_PER_OUTPUT for i, name in enumerate(OUTPUTS)}
    parts = {}
    for name, off in offsets.items():
        parts[f"c0_{_short(name)}"] = coeffs[off]
        parts[f"a_{_short(name)}"]  = _format_floats(coeffs[off + 1:off + 1 + K])
        parts[f"b_{_short(name)}"]  = _format_floats(coeffs[off + 1 + K:off + 1 + 2 * K])

    coeff_doc = ", ".join(f"{c:.4f}" for c in coeffs)
    return SCHEDULE_TEMPLATE.format(
        sample_id=sample_id, K=K, coeff_doc=coeff_doc, **parts
    )


def _short(name: str) -> str:
    return {"lr": "lr", "alpha": "alpha", "beta1": "b1", "beta2": "b2"}[name]
