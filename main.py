#!/usr/bin/env python3
"""Run the SIR leap-frog case study and write summary outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sir.analysis import (
    final_attack_size,
    infection_peak,
    population_drift,
    reproduction_number,
)
from sir.convergence import run_convergence_study, summarize_orders
from sir.leapfrog import simulate_sir
from sir.plotting import (
    save_comparison_figure,
    save_convergence_figure,
    save_infection_overlay_figure,
)
from sir.scenarios import SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate classical SIR scenarios with leap-frog and compare "
            "fixed-step accuracy against an RK45 reference."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for figures and summary JSON (default: results/).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=list(SCENARIOS),
        help="Scenario keys to integrate.",
    )
    parser.add_argument(
        "--convergence-scenario",
        choices=sorted(SCENARIOS),
        default="high_transmission",
        help="Scenario used for the Euler vs leap-frog convergence study.",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="Skip the numerical method comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired = []
    summaries = []

    for key in args.scenarios:
        scenario = SCENARIOS[key]
        result = simulate_sir(
            scenario.parameters,
            scenario.initial,
            t_min=scenario.t_min,
            t_max=scenario.t_max,
            step=scenario.step,
        )
        peak = infection_peak(result)
        r0 = reproduction_number(scenario.parameters)
        summary = {
            "key": key,
            "name": scenario.name,
            "description": scenario.description,
            "beta": scenario.parameters.beta,
            "gamma": scenario.parameters.gamma,
            "population": scenario.parameters.population,
            "R0": r0,
            "peak_infection_day": peak.time,
            "peak_infected": peak.infected,
            "final_attack_size": final_attack_size(result),
            "max_population_drift": population_drift(result),
        }
        summaries.append(summary)
        paired.append((scenario, result))

        print(
            f"{scenario.name}: R0={r0:.3f}, "
            f"I_max={peak.infected:.1f} at t={peak.time:.3f} days"
        )

    figure_path = save_comparison_figure(
        paired,
        args.output_dir / "sir_leapfrog_comparison.png",
    )
    overlay_path = save_infection_overlay_figure(
        paired,
        args.output_dir / "infection_peaks.png",
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Figure written to {figure_path}")
    print(f"Infection overlay written to {overlay_path}")
    print(f"Summary written to {summary_path}")

    if not args.skip_convergence:
        conv_scenario = SCENARIOS[args.convergence_scenario]
        samples = run_convergence_study(conv_scenario)
        orders = summarize_orders(samples)
        conv_figure = save_convergence_figure(
            samples,
            args.output_dir / "method_convergence.png",
        )
        conv_path = args.output_dir / "convergence.json"
        conv_path.write_text(
            json.dumps(
                {
                    "scenario": args.convergence_scenario,
                    "observed_orders": orders,
                    "samples": [sample.to_dict() for sample in samples],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "Observed orders: "
            + ", ".join(f"{name}≈{order:.2f}" for name, order in orders.items())
        )
        print(f"Convergence figure written to {conv_figure}")
        print(f"Convergence summary written to {conv_path}")


if __name__ == "__main__":
    main()
