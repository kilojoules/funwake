"""Shared loader for the incumbent-port thin wrappers.

CORRECTED port transform (BLOCKING, spec SIGN-OFF ADDENDUM): the incumbents'
lr profiles were designed against a driver lr0 = 50 and RAMP UP ~4x, so the
scale-aware internal lr0 is::

    lr0 = (50 / 240) * D        # = 0.2083... * D

NOT (0.833*D)/50 (which was 4x too hot, putting iter_192's peak at ~3.3 D on
DEI). At D=240 this is EXACTLY 50.0, so the port is bit-identical to the
archived schedule at D=240 given the same alpha0 (G3). On ROWP (D=198) the
operating point is lr0 = 41.25; the archived schedule's own algebra is otherwise
unchanged (alpha0 passed straight through).
"""
import importlib.util
import os

_C_INC = 50.0 / 240.0   # -> exactly 50.0 at D=240; = 0.2083... * D
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_archived(rel_path):
    """Load the archived incumbent module (read-only) and return its schedule_fn."""
    path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(
        "archived_" + os.path.basename(path).replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.schedule_fn


def make_port(rel_path):
    """Build a scale-aware schedule_fn wrapping the archived one with lr0=c*D."""
    archived = load_archived(rel_path)

    def schedule_fn(step, total_steps, D, min_spacing, n_turbines,
                    gamma_min, alpha0):
        lr0 = _C_INC * float(D)     # 0.2083*D; exactly 50.0 at D=240
        return archived(step, total_steps, lr0, alpha0)

    return schedule_fn, archived
