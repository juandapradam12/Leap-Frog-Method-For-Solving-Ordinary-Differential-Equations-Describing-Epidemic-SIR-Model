"""Leap-frog integration of the classical SIR epidemic model."""

from sir.analysis import infection_peak, reproduction_number
from sir.convergence import ErrorSample, run_convergence_study, summarize_orders
from sir.leapfrog import LeapFrogIntegrator, simulate_sir
from sir.model import SIRModel, SIRParameters, SIRState, SimulationResult
from sir.scenarios import SCENARIOS, Scenario
from sir.time_grid import IntegrationGrid

__all__ = [
    "ErrorSample",
    "IntegrationGrid",
    "LeapFrogIntegrator",
    "SIRModel",
    "SIRParameters",
    "SIRState",
    "SimulationResult",
    "Scenario",
    "SCENARIOS",
    "infection_peak",
    "reproduction_number",
    "run_convergence_study",
    "simulate_sir",
    "summarize_orders",
]

__version__ = "1.1.0"
