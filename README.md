# Leap-Frog Integration of the SIR Epidemic Model

Numerical case study of the classical **Susceptible–Infected–Recovered (SIR)** compartmental model, integrated with the explicit **leap-frog** (centered two-step) method. The project is structured as a small Python package suitable for a data-science / scientific-computing portfolio: typed domain models, a reusable ODE integrator, reproducible scenarios, tests, and concise technical documentation.

## Mathematical summary

Closed population $N = S + I + R$ with transmission rate $\beta$ and recovery rate $\gamma$:

$$
\frac{dS}{dt} = -\beta S I, \qquad
\frac{dI}{dt} = \beta S I - \gamma I, \qquad
\frac{dR}{dt} = \gamma I.
$$

On a uniform step $h$, leap-frog advances the state $y = (S, I, R)$ by

$$
y_{n+1} = y_{n-1} + 2h\, f(y_n),
$$

after a forward-Euler starter $y_1 = y_0 + h\, f(y_0)$. Epidemic potential is summarized by

$$
R_0 = \frac{\beta N}{\gamma}.
$$

Full derivation, discretization details, and scenario rationale: [`docs/methodology.md`](docs/methodology.md). Package layout: [`docs/architecture.md`](docs/architecture.md).

## What this case study shows

Two calibrated outbreaks on the same population ($N = 771$, one index case) are compared:

| Scenario | $\beta$ | $\gamma$ | $R_0$ | Qualitative outcome |
|----------|---------|----------|-------|---------------------|
| High transmission | $0.0022$ | $0.45$ | $\approx 3.77$ | Early, sharp infection peak |
| Moderate transmission | $0.001$ | $0.2$ | $\approx 3.86$ | Later peak, longer epidemic wave |

Even with similar $R_0$, the $(\beta, \gamma)$ pair reshapes peak timing—an illustration of why reproduction numbers alone do not determine epidemic shape.

![SIR leap-frog comparison](docs/figures/sir_leapfrog_comparison.png)

## Project layout

```text
sir/           Core package (model, integrator, scenarios, analysis, plots)
main.py        CLI entry point
tests/         pytest suite
docs/          Methodology and architecture notes
results/       Generated figures and JSON summaries (local only)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Run

```bash
python main.py
```

Writes:

- `results/sir_leapfrog_comparison.png` — stacked $S,I,R$ trajectories
- `results/summary.json` — $R_0$, peak day/height, attack size, population drift

Optional flags:

```bash
python main.py --scenarios high_transmission
python main.py --output-dir results
```

## Tests

```bash
pytest -q
```

Coverage includes parameter validation, population conservation under leap-frog, $R_0$-threshold behaviour, and relative peak timing between scenarios.

## Design highlights

- **Typed domain layer** — immutable dataclasses for parameters, state, and results
- **Reusable integrator** — leap-frog over any autonomous $f(y)$, not hard-wired to plotting
- **Reproducible scenarios** — experiment configs isolated from the solver
- **Measurable numerics** — population drift and peak statistics reported alongside the figure

## License

MIT
