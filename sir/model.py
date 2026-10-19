"""SIR compartmental model definition and simulation result types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SIRParameters:
    """Epidemiological parameters for a closed SIR population.

    Attributes:
        beta: Transmission rate (contacts that transmit infection per unit time).
        gamma: Recovery rate (inverse of the mean infectious period).
        population: Total population size N = S + I + R (conserved).
    """

    beta: float
    gamma: float
    population: float

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.population <= 0:
            raise ValueError("population must be positive")


@dataclass(frozen=True)
class SIRState:
    """Initial compartment sizes."""

    susceptible: float
    infected: float
    recovered: float = 0.0

    def as_array(self) -> NDArray[np.floating]:
        return np.array(
            [self.susceptible, self.infected, self.recovered],
            dtype=float,
        )

    def validate(self, population: float, *, tol: float = 1e-9) -> None:
        total = self.susceptible + self.infected + self.recovered
        if any(v < 0 for v in (self.susceptible, self.infected, self.recovered)):
            raise ValueError("compartment sizes must be non-negative")
        if abs(total - population) > tol:
            raise ValueError(
                f"S + I + R = {total} does not match population {population}"
            )


@dataclass(frozen=True)
class SimulationResult:
    """Time series produced by integrating the SIR system."""

    time: NDArray[np.floating]
    susceptible: NDArray[np.floating]
    infected: NDArray[np.floating]
    recovered: NDArray[np.floating]
    parameters: SIRParameters

    @property
    def population_trajectory(self) -> NDArray[np.floating]:
        return self.susceptible + self.infected + self.recovered


class SIRModel:
    """Right-hand side of the classical SIR ODE system.

    $$
    \\frac{dS}{dt} = -\\beta S I,\\quad
    \\frac{dI}{dt} = \\beta S I - \\gamma I,\\quad
    \\frac{dR}{dt} = \\gamma I
    $$
    """

    def __init__(self, parameters: SIRParameters) -> None:
        self.parameters = parameters

    def derivative(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate f(y) for y = (S, I, R)."""
        s, i, _r = state
        beta = self.parameters.beta
        gamma = self.parameters.gamma
        return np.array(
            [
                -beta * s * i,
                beta * s * i - gamma * i,
                gamma * i,
            ],
            dtype=float,
        )
