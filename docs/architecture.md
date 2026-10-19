# Architecture

The repository is organized as a small, testable Python package with a thin CLI entry point.

```text
.
├── main.py                 # CLI: scenarios, figures, convergence study
├── pyproject.toml          # Package metadata and installable deps
├── sir/
│   ├── model.py            # Parameters, state, SIR right-hand side
│   ├── time_grid.py        # Shared uniform mesh
│   ├── leapfrog.py         # Second-order leap-frog integrator
│   ├── euler.py            # First-order forward-Euler baseline
│   ├── reference.py        # Adaptive RK45 reference (SciPy)
│   ├── convergence.py      # Global-error study and observed orders
│   ├── scenarios.py        # Named experiment definitions
│   ├── analysis.py         # R0, peak, conservation, attack size
│   └── plotting.py         # Trajectory and convergence figures
├── tests/                  # Unit and integration tests
├── docs/                   # Case-study documentation
├── .github/workflows/ci.yml
└── results/                # Generated outputs (ignored by git)
```

## Layering

1. **Model** — pure domain types (`SIRParameters`, `SIRState`, `SIRModel`). No I/O.
2. **Integrators** — interchangeable fixed-step and reference solvers over the same `IntegrationGrid`.
3. **Convergence** — measures $E(h)$ against RK45 without coupling to plotting.
4. **Scenarios** — frozen experiment configs so runs are reproducible and comparable.
5. **Analysis / plotting** — derived metrics and presentation, kept out of the solver.
6. **CLI (`main.py`)** — orchestration only: select scenarios, persist artifacts.

This separation keeps the numerical core reusable (for example, swapping leap-frog for another scheme) without touching epidemic definitions or reporting.
