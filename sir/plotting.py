"""Publication-style figures for SIR trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sir.model import SimulationResult
from sir.scenarios import Scenario

COMPARTMENT_STYLES = (
    ("Susceptible", "susceptible", "#1f4e79"),
    ("Infected", "infected", "#c0392b"),
    ("Recovered", "recovered", "#1e8449"),
)


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
