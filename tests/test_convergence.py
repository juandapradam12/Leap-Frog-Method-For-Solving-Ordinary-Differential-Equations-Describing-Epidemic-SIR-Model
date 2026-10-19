"""Tests for fixed-step accuracy against the RK45 reference."""

import numpy as np
import pytest

from sir.convergence import (
    observed_order,
    run_convergence_study,
    summarize_orders,
)
from sir.euler import simulate_sir_euler
from sir.leapfrog import simulate_sir
from sir.reference import simulate_sir_reference
from sir.scenarios import SCENARIOS


def test_observed_order_recovers_known_slope() -> None:
    steps = np.array([0.2, 0.1, 0.05, 0.025])
    errors = 3.0 * steps**2
    assert observed_order(steps, errors) == pytest.approx(2.0, abs=1e-9)


def test_leapfrog_more_accurate_than_euler_at_same_step() -> None:
    scenario = SCENARIOS["high_transmission"]
    step = 5e-2
    kwargs = dict(
        t_min=scenario.t_min,
        t_max=scenario.t_max,
        step=step,
    )
    reference = simulate_sir_reference(
        scenario.parameters, scenario.initial, **kwargs
    )
    euler = simulate_sir_euler(scenario.parameters, scenario.initial, **kwargs)
    leap = simulate_sir(scenario.parameters, scenario.initial, **kwargs)
    euler_err = float(np.max(np.abs(euler.infected - reference.infected)))
    leap_err = float(np.max(np.abs(leap.infected - reference.infected)))
    assert leap_err < euler_err


def test_convergence_study_orders() -> None:
    scenario = SCENARIOS["high_transmission"]
    samples = run_convergence_study(
        scenario,
        steps=(1.0e-1, 5.0e-2, 2.5e-2, 1.25e-2),
    )
    orders = summarize_orders(samples)

    assert orders["forward_euler"] == pytest.approx(1.0, abs=0.35)
    assert orders["leapfrog"] == pytest.approx(2.0, abs=0.45)
    assert orders["leapfrog"] > orders["forward_euler"]
