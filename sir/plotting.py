"""Publication-style figures for SIR trajectories and convergence studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sir.convergence import ErrorSample
from sir.model import SimulationResult
from sir.scenarios import Scenario

COMPARTMENT_STYLES = (
    ("Susceptible", "susceptible", "#1f4e79"),
    ("Infected", "infected", "#c0392b"),
    ("Recovered", "recovered", "#1e8449"),
)

METHOD_STYLES = {
    "forward_euler": ("Forward Euler", "#c0392b", "o"),
    "leapfrog": ("Leap-frog", "#1f4e79", "s"),
}


def _style_axes(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Population")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")


def plot_scenario(ax: Axes, result: SimulationResult, scenario: Scenario) -> None:
    """Draw S(t), I(t), R(t) for a single scenario on the given axes."""
    for label, attr, color in COMPARTMENT_STYLES:
        ax.plot(
            result.time,
            getattr(result, attr),
            label=label,
            color=color,
            linewidth=1.6,
        )
    _style_axes(ax, scenario.name)


def build_comparison_figure(
    results: list[tuple[Scenario, SimulationResult]],
) -> Figure:
    """Stacked comparison of one or more SIR scenarios."""
    n = len(results)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(9.5, 3.4 * n),
        sharex=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    for ax, (scenario, result) in zip(axes, results):
        plot_scenario(ax, result, scenario)

    fig.suptitle(
        "SIR epidemic dynamics via leap-frog integration",
        fontsize=13,
        fontweight="bold",
    )
    return fig


def save_comparison_figure(
    results: list[tuple[Scenario, SimulationResult]],
    output_path: str | Path,
    *,
    dpi: int = 160,
) -> Path:
    """Render and persist a comparison figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_comparison_figure(results)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


INFECTION_CURVE_COLORS = ("#c0392b", "#1f4e79", "#1e8449", "#8e44ad")


def build_infection_overlay_figure(
    results: list[tuple[Scenario, SimulationResult]],
) -> Figure:
    """Overlay I(t) across scenarios and mark each infection peak."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)

    for color, (scenario, result) in zip(INFECTION_CURVE_COLORS, results):
        peak_idx = int(np.argmax(result.infected))
        peak_t = float(result.time[peak_idx])
        peak_i = float(result.infected[peak_idx])
        ax.plot(
            result.time,
            result.infected,
            color=color,
            linewidth=2.0,
            label=scenario.name,
        )
        ax.plot(peak_t, peak_i, marker="o", color=color, markersize=7)
        ax.annotate(
            f"peak @ {peak_t:.1f} d",
            xy=(peak_t, peak_i),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

    ax.set_title("Infection peaks under similar $R_0$", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected population")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    return fig


def save_infection_overlay_figure(
    results: list[tuple[Scenario, SimulationResult]],
    output_path: str | Path,
    *,
    dpi: int = 160,
) -> Path:
    """Render and persist the infection-peak overlay figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_infection_overlay_figure(results)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def build_convergence_figure(samples: list[ErrorSample]) -> Figure:
    """Log-log global error versus step size for each fixed-step method."""
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)

    for method, (label, color, marker) in METHOD_STYLES.items():
        subset = [s for s in samples if s.method == method]
        if not subset:
            continue
        steps = np.array([s.step for s in subset], dtype=float)
        errors = np.array([s.max_abs_error for s in subset], dtype=float)
        order = steps.argsort()
        ax.loglog(
            steps[order],
            errors[order],
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=7,
            label=label,
        )

    # Reference slopes anchored at the coarsest leap-frog sample when available.
    leap = [s for s in samples if s.method == "leapfrog"]
    if leap:
        anchor = max(leap, key=lambda s: s.step)
        h_ref = np.array([anchor.step, min(s.step for s in leap)], dtype=float)
        e0 = anchor.max_abs_error
        ax.loglog(
            h_ref,
            e0 * (h_ref / h_ref[0]),
            linestyle="--",
            color="#7f8c8d",
            linewidth=1.2,
            label=r"$\mathcal{O}(h)$",
        )
        ax.loglog(
            h_ref,
            e0 * (h_ref / h_ref[0]) ** 2,
            linestyle=":",
            color="#7f8c8d",
            linewidth=1.2,
            label=r"$\mathcal{O}(h^2)$",
        )

    ax.set_xlabel("Step size $h$")
    ax.set_ylabel(r"Global error $\max_t\|y_h(t) - y_{\mathrm{ref}}(t)\|_\infty$")
    ax.set_title("Fixed-step accuracy versus RK45 reference")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    return fig


def save_convergence_figure(
    samples: list[ErrorSample],
    output_path: str | Path,
    *,
    dpi: int = 160,
) -> Path:
    """Render and persist the convergence comparison figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_convergence_figure(samples)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
