# Methodology

## Compartmental model

A closed population of size $N$ is partitioned into three compartments:

| Symbol | Meaning |
|--------|---------|
| $S(t)$ | Susceptible individuals |
| $I(t)$ | Infectious individuals |
| $R(t)$ | Recovered (and immune) individuals |

with the conservation constraint

$$
S(t) + I(t) + R(t) = N.
$$

Transmission and recovery are parameterized by $\beta > 0$ (effective contact rate) and $\gamma > 0$ (recovery rate). The mean infectious period is $1/\gamma$.

## Governing equations

The classical mass-action SIR system is the autonomous ODE

$$
\begin{aligned}
\frac{dS}{dt} &= -\beta S I, \\
\frac{dI}{dt} &= \beta S I - \gamma I, \\
\frac{dR}{dt} &= \gamma I.
\end{aligned}
$$

The basic reproduction number,

$$
R_0 = \frac{\beta N}{\gamma},
$$

controls the early growth of an outbreak seeded by a small infected fraction. When $R_0 > 1$, the disease-free equilibrium is unstable and an epidemic wave develops; when $R_0 < 1$, infections decay monotonically.

## Leap-frog discretization

Write the state as $y = (S, I, R)^\top$ and $y' = f(y)$. On a uniform mesh with step $h$, the explicit leap-frog (centered two-step) update is

$$
y_{n+1} = y_{n-1} + 2h\, f(y_n), \qquad n \ge 1,
$$

started with a single forward-Euler step

$$
y_1 = y_0 + h\, f(y_0).
$$

The method is second-order accurate in $h$. Because $f$ is divergence-free for the SIR vector field ($\partial_S f_S + \partial_I f_I + \partial_R f_R = 0$), the continuous flow preserves $N$; the discrete trajectory inherits this conservation up to truncation error, which is monitored in the analysis layer.

## Baseline and reference schemes

For the same autonomous system $y' = f(y)$, two additional integrators are used only for accuracy assessment:

| Method | Update | Formal order |
|--------|--------|--------------|
| Forward Euler | $y_{n+1} = y_n + h\, f(y_n)$ | $\mathcal{O}(h)$ |
| Leap-frog | $y_{n+1} = y_{n-1} + 2h\, f(y_n)$ | $\mathcal{O}(h^2)$ |
| RK45 (`scipy.integrate.solve_ivp`) | adaptive | reference |

On each mesh $h$, the adaptive RK45 solution evaluated at the same nodes defines $y_{\mathrm{ref}}$. The global error of a fixed-step method is

$$
E(h) = \max_t \bigl\| y_h(t) - y_{\mathrm{ref}}(t) \bigr\|_\infty.
$$

Observed orders are recovered from a log-log least-squares fit of $E(h)$ versus $h$. Empirically, Euler tracks $\mathcal{O}(h)$ while leap-frog tracks $\mathcal{O}(h^2)$, which is the numerical justification for preferring leap-frog in this case study.

## Scenario design

Both scenarios share $N = 771$ and the seed $(S_0, I_0, R_0) = (770, 1, 0)$, and are integrated on $t \in [0, 17]$ with $h = 10^{-3}$.

| Scenario | $\beta$ | $\gamma$ | $R_0$ | Expected dynamics |
|----------|---------|----------|-------|-------------------|
| High transmission | $0.0022$ | $0.45$ | $\approx 3.77$ | Early, sharp infection peak |
| Moderate transmission | $0.001$ | $0.2$ | $\approx 3.86$ | Delayed peak, longer epidemic |

Comparing the two trajectories isolates how the balance between transmission and recovery shapes peak timing, even when $R_0$ is of similar magnitude.

## Quantities of interest

For each integrated trajectory the code reports:

- peak infection time $t^\star = \arg\max_t I(t)$ and height $I(t^\star)$;
- final attack size $1 - S(t_{\mathrm{end}})/N$;
- maximum population drift $\max_t |S+I+R - N|$ as a numerical sanity check;
- convergence table $E(h)$ and observed orders for Euler versus leap-frog.
