"""Leap-frog integration of the classical SIR epidemic model."""

from sir.analysis import infection_peak, reproduction_number
from sir.leapfrog import LeapFrogIntegrator
from sir.model import SIRModel, SIRParameters, SIRState, SimulationResult
from sir.scenarios import SCENARIOS, Scenario

__all__ = [
    "LeapFrogIntegrator",
    "SIRModel",
    "SIRParameters",
    "SIRState",
    "SimulationResult",
    "Scenario",
    "SCENARIOS",
    "infection_peak",
    "reproduction_number",
]

__version__ = "1.0.0"
