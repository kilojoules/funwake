"""py_wake-backed wake simulation adapter for the unit-test path.

Mirrors the small simulation interface the tests use (Curve, Turbine,
WakeSimulation, BastankhahGaussianDeficit) on top of the open-source
py_wake library (>= 2.6).

Model configuration (chosen for parity with the scoring stack's reference
model):
  - Bastankhah-Gaussian deficit, constant wake expansion k (tests pass
    k=0.04), ceps=0.2, ctlim=0.899, ct2a=ct2a_madsen — py_wake's defaults.
  - Deficit scaled by the AMBIENT wind speed (use_effective_ws=False),
    thrust coefficient evaluated at the source turbine's EFFECTIVE speed.
  - Deficit truncated at the 2-sigma wake radius (|cw| < 2*sigma); see
    _RadiusMaskedBastankhah below.
  - SquaredSum superposition: ws_eff = ws_amb - sqrt(sum(deficit^2)).
  - PropagateDownwind engine, no turbulence model, no blockage.
  - Wind cases are paired (wd_i, ws_i) time-series with explicit weights;
    AEP [GWh] = sum(power_kw * weight) * 8760 / 1e6.
"""

import numpy as np

from py_wake import np as pw_np
from py_wake.deficit_models.gaussian import (
    BastankhahGaussianDeficit as _PWBastankhahGaussianDeficit,
)
from py_wake.site import UniformSite
from py_wake.superposition_models import SquaredSum
from py_wake.utils import gradients
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


class Curve:
    """A performance curve: wind speeds and corresponding values."""

    def __init__(self, ws, values):
        self.ws = np.asarray(ws, dtype=float)
        self.values = np.asarray(values, dtype=float)


class Turbine:
    """Wind turbine spec: rotor diameter, hub height, power and Ct curves."""

    def __init__(self, rotor_diameter, hub_height, power_curve, ct_curve):
        self.rotor_diameter = float(rotor_diameter)
        self.hub_height = float(hub_height)
        self.power_curve = power_curve
        self.ct_curve = ct_curve


class BastankhahGaussianDeficit:
    """Config marker: Bastankhah-Gaussian deficit with wake expansion k."""

    def __init__(self, k=0.0324555):
        self.k = k


class _RadiusMaskedBastankhah(_PWBastankhahGaussianDeficit):
    """Bastankhah-Gaussian deficit truncated at the 2-sigma wake radius.

    py_wake's Gaussian deficit has unbounded lateral tails; the reference
    model only applies the deficit where |cw| < wake_radius = 2*sigma.
    The comparison (non-differentiable, zero gradient a.e.) is done on
    squared quantities so it stays valid under autograd tracing.
    """

    def calc_deficit(self, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):
        deficit = super().calc_deficit(D_src_il=D_src_il, dw_ijlk=dw_ijlk,
                                       cw_ijlk=cw_ijlk, ct_ilk=ct_ilk, **kwargs)
        wake_radius = self.wake_radius(D_src_il=D_src_il, dw_ijlk=dw_ijlk,
                                       ct_ilk=ct_ilk, **kwargs)
        return deficit * (cw_ijlk**2 < wake_radius**2)


class _SimResult:
    """Simulation result: turbine power per (wind case, turbine) in kW."""

    def __init__(self, power_kw):
        self._power_kw = power_kw

    def power(self):
        """Power array, shape (n_cases, n_turbines), in kW."""
        return self._power_kw


def _merge_curves(power_curve, ct_curve):
    """Interpolate power and ct onto the union ws grid.

    Piecewise-linear interpolation onto the union of the two grids is exact,
    so a single-grid tabular power/ct function reproduces both curves.
    """
    ws = np.union1d(power_curve.ws, ct_curve.ws)
    power = np.interp(ws, power_curve.ws, power_curve.values)
    ct = np.interp(ws, ct_curve.ws, ct_curve.values)
    return ws, power, ct


class WakeSimulation:
    """py_wake-backed wake simulation for a single turbine type.

    Callable as sim(x, y, ws_amb=..., wd_amb=..., ti_amb=None), returning a
    result whose .power() is a (n_cases, n_turbines) array in kW, where the
    wind cases are paired (wd_i, ws_i) conditions.
    """

    def __init__(self, turbine, deficit):
        ws, power_kw, ct = _merge_curves(turbine.power_curve, turbine.ct_curve)
        power_ct = PowerCtTabular(ws, power_kw, "kw", ct, method="linear",
                                  additional_models=[])
        wt = WindTurbine(name="turbine",
                         diameter=turbine.rotor_diameter,
                         hub_height=turbine.hub_height,
                         powerCtFunction=power_ct)
        # TI value is inert: constant-k deficit, no turbulence model.
        site = UniformSite(p_wd=[1.0], ti=0.1)
        self.wfm = PropagateDownwind(
            site, wt,
            wake_deficitModel=_RadiusMaskedBastankhah(k=deficit.k),
            superpositionModel=SquaredSum())

    def _power_ilk(self, x, y, wd, ws):
        """Turbine power in W, shape (n_turbines, n_cases, 1). Autograd-safe."""
        return self.wfm._run(x, y, wd=wd, ws=ws, time=True)[2]

    def __call__(self, x, y, ws_amb, wd_amb, ti_amb=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        wd = np.asarray(wd_amb, dtype=float)
        ws = np.asarray(ws_amb, dtype=float)
        power_w = np.asarray(self._power_ilk(x, y, wd, ws))
        return _SimResult(power_w[:, :, 0].T / 1e3)

    def make_neg_aep(self, wd, ws, weights):
        """Build the negative-AEP objective [GWh] and its (x, y) gradient.

        Returns (neg_aep, neg_aep_grad); neg_aep_grad(x, y) -> (gx, gy).
        Gradients via py_wake's autograd backend.
        """
        wd = np.asarray(wd, dtype=float)
        ws = np.asarray(ws, dtype=float)
        weights = np.asarray(weights, dtype=float)

        def neg_aep(x, y):
            p_kw = self._power_ilk(x, y, wd, ws)[:, :, 0] / 1e3  # (n_wt, n_cases)
            return -pw_np.sum(p_kw * weights[None, :]) * 8760.0 / 1e6

        grad_fn = gradients.autograd(neg_aep, vector_interdependence=True,
                                     argnum=[0, 1])

        def neg_aep_grad(x, y):
            gx, gy = grad_fn(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            return np.asarray(gx), np.asarray(gy)

        return neg_aep, neg_aep_grad
