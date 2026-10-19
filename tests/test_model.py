"""Unit tests for SIR parameter and state validation."""

import numpy as np
import pytest

from sir.model import SIRModel, SIRParameters, SIRState


def test_parameters_reject_non_positive_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        SIRParameters(beta=0.1, gamma=0.0, population=100)


def test_state_must_match_population() -> None:
    state = SIRState(susceptible=90, infected=5, recovered=0)
    with pytest.raises(ValueError, match="population"):
        state.validate(100)


def test_derivative_zero_when_no_infected() -> None:
    model = SIRModel(SIRParameters(beta=0.01, gamma=0.2, population=100))
    dydt = model.derivative(np.array([100.0, 0.0, 0.0]))
    assert np.allclose(dydt, 0.0)


def test_derivative_conserves_population_locally() -> None:
    model = SIRModel(SIRParameters(beta=0.002, gamma=0.3, population=500))
    state = np.array([400.0, 50.0, 50.0])
    dydt = model.derivative(state)
    assert dydt.sum() == pytest.approx(0.0)
