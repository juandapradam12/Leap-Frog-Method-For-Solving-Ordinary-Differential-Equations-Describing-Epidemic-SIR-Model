"""Derived epidemiological quantities from SIR solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sir.model import SIRParameters, SimulationResult


def reproduction_number(parameters: SIRParameters) -> float:
    """Basic reproduction number R0 = beta * N / gamma."""
    return parameters.beta * parameters.population / parameters.gamma


@dataclass(frozen=True)
class InfectionPeak:
    """Time and size of the infection peak."""

    time: float
    infected: float
    index: int


def infection_peak(result: SimulationResult) -> InfectionPeak:
    """Locate the maximum of the infected compartment."""
    index = int(np.argmax(result.infected))
    return InfectionPeak(
        time=float(result.time[index]),
        infected=float(result.infected[index]),
        index=index,
    )


def population_drift(result: SimulationResult) -> float:
    """Maximum absolute deviation of S+I+R from the initial population."""
    return float(np.max(np.abs(result.population_trajectory - result.parameters.population)))


def final_attack_size(result: SimulationResult) -> float:
    """Fraction of the population that eventually leaves the susceptible class."""
    s_inf = float(result.susceptible[-1])
    return 1.0 - s_inf / result.parameters.population
