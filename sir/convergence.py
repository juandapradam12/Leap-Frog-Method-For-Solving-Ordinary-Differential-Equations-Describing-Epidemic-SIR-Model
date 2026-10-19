"""Fixed-step error study against an adaptive RK45 reference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from sir.euler import ForwardEulerIntegrator
from sir.leapfrog import LeapFrogIntegrator
from sir.model import SIRModel, SimulationResult
from sir.reference import ScipyReferenceIntegrator
from sir.scenarios import Scenario
from sir.time_grid import IntegrationGrid

DEFAULT_STEPS: tuple[float, ...] = (2.0e-1, 1.0e-1, 5.0e-2, 2.5e-2, 1.25e-2)


@dataclass(frozen=True)
class ErrorSample:
    """Global error of one method at one step size."""

    method: str
    step: float
    max_abs_error: float
    infected_max_abs_error: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _state_matrix(result: SimulationResult) -> NDArray[np.floating]:
    return np.column_stack(
        [result.susceptible, result.infected, result.recovered]
    )


def _error_against_reference(
    approximate: SimulationResult,
    reference: SimulationResult,
) -> tuple[float, float]:
    diff = _state_matrix(approximate) - _state_matrix(reference)
    max_abs = float(np.max(np.abs(diff)))
    infected_max = float(np.max(np.abs(approximate.infected - reference.infected)))
    return max_abs, infected_max


def observed_order(steps: NDArray[np.floating], errors: NDArray[np.floating]) -> float:
    """Estimate convergence order from a log-log least-squares fit.

    Returns NaN when fewer than two positive samples are available.
    """
    mask = (steps > 0) & (errors > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(steps[mask]), np.log(errors[mask]), deg=1)
    return float(coeffs[0])


def run_convergence_study(
    scenario: Scenario,
    *,
    steps: tuple[float, ...] = DEFAULT_STEPS,
) -> list[ErrorSample]:
    """Compare forward Euler and leap-frog to RK45 on identical grids."""
    model = SIRModel(scenario.parameters)
    samples: list[ErrorSample] = []

    for step in steps:
        grid = IntegrationGrid(
            t_min=scenario.t_min,
            t_max=scenario.t_max,
            step=step,
        )
        reference = ScipyReferenceIntegrator(grid).integrate_sir(
            model, scenario.initial
        )
        for method_name, integrator in (
            ("forward_euler", ForwardEulerIntegrator(grid)),
            ("leapfrog", LeapFrogIntegrator(grid)),
        ):
            approx = integrator.integrate_sir(model, scenario.initial)
            max_abs, infected_max = _error_against_reference(approx, reference)
            samples.append(
                ErrorSample(
                    method=method_name,
                    step=step,
                    max_abs_error=max_abs,
                    infected_max_abs_error=infected_max,
                )
            )

    return samples


def summarize_orders(samples: list[ErrorSample]) -> dict[str, float]:
    """Observed order for each method based on max_abs_error vs step."""
    orders: dict[str, float] = {}
    methods = sorted({sample.method for sample in samples})
    for method in methods:
        subset = [s for s in samples if s.method == method]
        steps = np.array([s.step for s in subset], dtype=float)
        errors = np.array([s.max_abs_error for s in subset], dtype=float)
        orders[method] = observed_order(steps, errors)
    return orders
