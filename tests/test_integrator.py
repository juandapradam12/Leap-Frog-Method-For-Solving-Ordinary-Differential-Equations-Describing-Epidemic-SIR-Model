"""Unit tests for the leap-frog integrator and end-to-end SIR runs."""

import numpy as np
import pytest

from sir.analysis import infection_peak, population_drift, reproduction_number
from sir.leapfrog import IntegrationGrid, LeapFrogIntegrator, simulate_sir
from sir.model import SIRModel, SIRParameters, SIRState
from sir.scenarios import SCENARIOS


def test_grid_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        IntegrationGrid(t_min=1.0, t_max=0.5, step=0.1)


def test_leapfrog_integrates_exponential_decay() -> None:
    """For y' = -y, the leap-frog solution should track e^{-t} reasonably."""
    grid = IntegrationGrid(t_min=0.0, t_max=2.0, step=1e-3)
    integrator = LeapFrogIntegrator(grid)
    time, traj = integrator.integrate(lambda y: -y, np.array([1.0]))
    exact = np.exp(-time)
    assert np.max(np.abs(traj[:, 0] - exact)) < 5e-3


def test_sir_population_nearly_conserved() -> None:
    scenario = SCENARIOS["high_transmission"]
    result = simulate_sir(
        scenario.parameters,
        scenario.initial,
        t_min=scenario.t_min,
        t_max=scenario.t_max,
        step=scenario.step,
    )
    assert population_drift(result) < 1e-6


def test_epidemic_occurs_when_r0_above_one() -> None:
    scenario = SCENARIOS["high_transmission"]
    assert reproduction_number(scenario.parameters) > 1.0
    result = simulate_sir(scenario.parameters, scenario.initial)
    peak = infection_peak(result)
    assert peak.infected > scenario.initial.infected
    assert peak.time > 0.0


def test_no_epidemic_growth_when_r0_below_one() -> None:
    params = SIRParameters(beta=1e-4, gamma=0.5, population=771.0)
    assert reproduction_number(params) < 1.0
    result = simulate_sir(
        params,
        SIRState(susceptible=770.0, infected=1.0, recovered=0.0),
        t_max=20.0,
        step=1e-3,
    )
    assert infection_peak(result).infected < 2.0


def test_moderate_peak_later_than_high_transmission() -> None:
    high = simulate_sir(
        SCENARIOS["high_transmission"].parameters,
        SCENARIOS["high_transmission"].initial,
    )
    moderate = simulate_sir(
        SCENARIOS["moderate_transmission"].parameters,
        SCENARIOS["moderate_transmission"].initial,
    )
    assert infection_peak(moderate).time > infection_peak(high).time


def test_integrate_sir_matches_manual_model_call() -> None:
    params = SIRParameters(beta=0.0022, gamma=0.45, population=771.0)
    initial = SIRState(susceptible=770.0, infected=1.0, recovered=0.0)
    grid = IntegrationGrid(t_min=0.0, t_max=5.0, step=1e-3)
    via_helper = LeapFrogIntegrator(grid).integrate_sir(SIRModel(params), initial)
    via_simulate = simulate_sir(params, initial, t_min=0.0, t_max=5.0, step=1e-3)
    assert np.allclose(via_helper.infected, via_simulate.infected)
