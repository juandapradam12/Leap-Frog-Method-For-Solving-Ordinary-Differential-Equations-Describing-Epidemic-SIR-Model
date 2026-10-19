"""Forward-Euler integrator for autonomous ODEs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from sir.model import SIRModel, SIRParameters, SIRState, SimulationResult
from sir.time_grid import IntegrationGrid

DerivativeFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]


class ForwardEulerIntegrator:
    """Integrate y' = f(y) with the explicit forward-Euler method.

    $$
    y_{n+1} = y_n + h\\, f(y_n).
    $$

    The scheme is first-order accurate in $h$ and is included as a baseline
    against the second-order leap-frog method.
    """

    def __init__(self, grid: IntegrationGrid) -> None:
        self.grid = grid

    def integrate(
        self,
        derivative: DerivativeFn,
        y0: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        h = self.grid.step
        time = self.grid.build()
        n = len(time)
        y = np.empty((n, y0.size), dtype=float)
        y[0] = np.asarray(y0, dtype=float)

        for i in range(1, n):
            y[i] = y[i - 1] + h * derivative(y[i - 1])

        return time, y

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


def simulate_sir_euler(
    parameters: SIRParameters,
    initial: SIRState,
    *,
    t_min: float = 0.0,
    t_max: float = 17.0,
    step: float = 1e-3,
) -> SimulationResult:
    grid = IntegrationGrid(t_min=t_min, t_max=t_max, step=step)
    return ForwardEulerIntegrator(grid).integrate_sir(SIRModel(parameters), initial)
