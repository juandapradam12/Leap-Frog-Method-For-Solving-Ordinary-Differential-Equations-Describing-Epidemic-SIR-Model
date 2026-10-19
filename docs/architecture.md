# Architecture

The repository is organized as a small, testable Python package with a thin CLI entry point.

```text
.
├── main.py                 # CLI: run scenarios, write figure + JSON summary
├── sir/
│   ├── model.py            # Parameters, state, SIR right-hand side
│   ├── leapfrog.py         # Time grid + leap-frog integrator
│   ├── scenarios.py        # Named experiment definitions
│   ├── analysis.py         # R0, peak, conservation, attack size
│   └── plotting.py         # Comparison figures
├── tests/                  # Unit and integration tests
├── docs/                   # Case-study documentation
└── results/                # Generated outputs (ignored by git)
```

## Layering

1. **Model** — pure domain types (`SIRParameters`, `SIRState`, `SIRModel`). No I/O.
2. **Integrator** — generic leap-frog over any autonomous $y' = f(y)$, plus an SIR-specific façade that returns a `SimulationResult`.
3. **Scenarios** — frozen experiment configs so runs are reproducible and comparable.
4. **Analysis / plotting** — derived metrics and presentation, kept out of the solver.
5. **CLI (`main.py`)** — orchestration only: select scenarios, persist artifacts.

This separation keeps the numerical core reusable (for example, swapping leap-frog for another scheme) without touching epidemic definitions or reporting.
