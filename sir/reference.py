"""High-accuracy reference trajectories via SciPy's adaptive RK45."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from sir.model import SIRModel, SIRParameters, SIRState, SimulationResult
from sir.time_grid import IntegrationGrid

DerivativeFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]


class ScipyReferenceIntegrator:
    """Dense-output RK45 solution evaluated on a prescribed time grid.

    Used as a reference when measuring the truncation error of fixed-step
    schemes (forward Euler and leap-frog).
    """

    def __init__(
        self,
        grid: IntegrationGrid,
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        self.grid = grid
        self.rtol = rtol
        self.atol = atol

    def integrate(
        self,
        derivative: DerivativeFn,
        y0: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        time = self.grid.build()
        y0_arr = np.asarray(y0, dtype=float)

        if len(time) == 1:
            return time, y0_arr.reshape(1, -1)

        solution = solve_ivp(
            fun=lambda _t, y: derivative(y),
            t_span=(float(time[0]), float(time[-1])),
            y0=y0_arr,
            t_eval=time,
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
            dense_output=False,
        )
        if not solution.success:
            raise RuntimeError(f"RK45 failed: {solution.message}")

        return time, solution.y.T

    def integrate_sir(
        self,
        model: SIRModel,
        initial: SIRState,
    ) -> SimulationResult:
        params = model.parameters
        initial.validate(params.population)
        time, trajectory = self.integrate(model.derivative, initial.as_array())
        return SimulationResult(
            time=time,
            susceptible=trajectory[:, 0],
            infected=trajectory[:, 1],
            recovered=trajectory[:, 2],
            parameters=params,
        )


def simulate_sir_reference(
    parameters: SIRParameters,
    initial: SIRState,
    *,
    t_min: float = 0.0,
    t_max: float = 17.0,
    step: float = 1e-3,
) -> SimulationResult:
    grid = IntegrationGrid(t_min=t_min, t_max=t_max, step=step)
    return ScipyReferenceIntegrator(grid).integrate_sir(SIRModel(parameters), initial)
