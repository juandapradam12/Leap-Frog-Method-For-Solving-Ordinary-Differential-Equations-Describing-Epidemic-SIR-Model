"""Two-step leap-frog (centered difference) integrator for autonomous ODEs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from sir.model import SIRModel, SIRParameters, SIRState, SimulationResult
from sir.time_grid import IntegrationGrid

DerivativeFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]

__all__ = ["IntegrationGrid", "LeapFrogIntegrator", "simulate_sir"]


class LeapFrogIntegrator:
    """Integrate y' = f(y) with the explicit leap-frog method.

    After an Euler starter step,

    $$
    y_{n+1} = y_{n-1} + 2 h\\, f(y_n).
    $$

    The scheme is second-order accurate and symplectic for separable
    Hamiltonian systems; here it is applied as a centered two-step method
    on the first-order SIR vector field.
    """

    def __init__(self, grid: IntegrationGrid) -> None:
        self.grid = grid

    def integrate(
        self,
        derivative: DerivativeFn,
        y0: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return (time, trajectory) with shape (n,) and (n, d)."""
        h = self.grid.step
        time = self.grid.build()
        n = len(time)
        y = np.empty((n, y0.size), dtype=float)
        y[0] = np.asarray(y0, dtype=float)

        if n == 1:
            return time, y

        y[1] = y[0] + h * derivative(y[0])

        for i in range(2, n):
            y[i] = y[i - 2] + 2.0 * h * derivative(y[i - 1])

        return time, y

    def integrate_sir(
        self,
        model: SIRModel,
        initial: SIRState,
    ) -> SimulationResult:
        """Integrate an SIR model and return a structured result."""
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


def simulate_sir(
    parameters: SIRParameters,
    initial: SIRState,
    *,
    t_min: float = 0.0,
    t_max: float = 17.0,
    step: float = 1e-3,
) -> SimulationResult:
    """Convenience wrapper: build model + leap-frog and integrate."""
    grid = IntegrationGrid(t_min=t_min, t_max=t_max, step=step)
    integrator = LeapFrogIntegrator(grid)
    return integrator.integrate_sir(SIRModel(parameters), initial)
