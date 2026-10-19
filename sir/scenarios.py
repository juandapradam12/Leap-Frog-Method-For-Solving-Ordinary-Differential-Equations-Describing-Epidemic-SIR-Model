"""Named simulation scenarios used in this case study."""

from __future__ import annotations

from dataclasses import dataclass

from sir.model import SIRParameters, SIRState


@dataclass(frozen=True)
class Scenario:
    """A complete, reproducible SIR experiment definition."""

    name: str
    description: str
    parameters: SIRParameters
    initial: SIRState
    t_min: float = 0.0
    t_max: float = 17.0
    step: float = 1e-3


POPULATION = 771.0
INITIAL = SIRState(susceptible=770.0, infected=1.0, recovered=0.0)

SCENARIOS: dict[str, Scenario] = {
    "high_transmission": Scenario(
        name="High transmission",
        description=(
            "Higher contact transmission rate and faster recovery; "
            "a sharp epidemic wave with an early infection peak."
        ),
        parameters=SIRParameters(beta=0.0022, gamma=0.45, population=POPULATION),
        initial=INITIAL,
    ),
    "moderate_transmission": Scenario(
        name="Moderate transmission",
        description=(
            "Lower transmission and slower recovery; the outbreak "
            "spreads over a longer horizon with a delayed peak."
        ),
        parameters=SIRParameters(beta=0.001, gamma=0.2, population=POPULATION),
        initial=INITIAL,
    ),
}
