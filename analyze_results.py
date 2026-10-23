"""Summarise and plot a results file written by run_benchmarks.py.

Test cases are grouped by the number of states of their target DFA (``randmNN`` in the Oliveira
benchmarks, ``resNN-k`` in the DFAMiner benchmarks). This prints how many test cases were solved and,
per group, statistics over the solved ones. It saves plots of the solved test cases and the running
time per group as PNG
files in ``benchmarking/plots/<results file name>/``.

Usage:
    python3 analyze_results.py                                            # newest results file
    python3 analyze_results.py benchmarking/results/benchmark_t200_all.csv
"""

import argparse
import csv
import glob
import os
import re
import statistics
from dataclasses import dataclass

RESULTS_DIRECTORY = "benchmarking/results"
PLOTS_DIRECTORY = "benchmarking/plots"

# Unsolved test cases listed in the summary before the rest are only counted.
MAX_LISTED_UNSOLVED = 20

COLOR = "#2a78d6"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# Plotted metrics: file name, Result attribute, title and whether the axis is logarithmic.
METRIC_PLOTS = [
    ("time", "total_time", "Running time (s)", True),
]


@dataclass
class Result:
    test_case: str
    states: int
    solved: bool
    total_time: float | None
    automaton_size: int | None


def _number(value: str, kind=float):
    return kind(value) if value not in ("", None) else None


def load_results(path: str) -> list[Result]:
    results = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            match = re.search(r"randm(\d+)|/res(\d+)-", row["file name"])
            if match is None:
                raise ValueError(f"Cannot determine the number of states of {row['file name']}")
            automaton_size = _number(row["automaton_size"], int)
            results.append(Result(test_case=row["file name"], states=int(match.group(1) or match.group(2)),
                                  solved=row["succeeded"] == "True" and bool(automaton_size),
                                  total_time=_number(row["total_time"]),
                                  automaton_size=automaton_size))
    return results


def group_by_states(results: list[Result]) -> dict[int, list[Result]]:
    groups: dict[int, list[Result]] = {}
    for result in results:
        groups.setdefault(result.states, []).append(result)
    return dict(sorted(groups.items()))


def _mean(values) -> str:
    values = list(values)
    return f"{statistics.mean(values):.2f}" if values else "-"


def _median(values) -> str:
    values = list(values)
    return f"{statistics.median(values):.2f}" if values else "-"


def _max(values) -> str:
    values = list(values)
    return f"{max(values):.2f}" if values else "-"


def print_summary(path: str, results: list[Result]) -> None:
    solved = [result for result in results if result.solved]
    unsolved = [result for result in results if not result.solved]
    print(f"Results: {path}")
    print(f"Solved: {len(solved)}/{len(results)} test cases")
    if solved:
        times = [result.total_time for result in solved]
        print(f"Running time of solved test cases: {sum(times):.2f}s in total, {statistics.mean(times):.2f}s mean, "
              f"{statistics.median(times):.2f}s median, {max(times):.2f}s max")
    print()

    header = ["states", "solved", "mean time (s)", "median time (s)", "max time (s)", "mean size"]
    rows = []
    for states, group in [*group_by_states(results).items(), ("all", results)]:
        group_solved = [result for result in group if result.solved]
        rows.append([str(states), f"{len(group_solved)}/{len(group)}",
                     _mean(result.total_time for result in group_solved),
                     _median(result.total_time for result in group_solved),
                     _max(result.total_time for result in group_solved),
                     _mean(result.automaton_size for result in group_solved)])
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))
    print("(statistics over the solved test cases)")

    if unsolved:
        print()
        print(f"Unsolved test cases ({len(unsolved)}):")
        for result in unsolved[:MAX_LISTED_UNSOLVED]:
            print(f"  {result.test_case}")
        if len(unsolved) > MAX_LISTED_UNSOLVED:
            print(f"  ... and {len(unsolved) - MAX_LISTED_UNSOLVED} more")


def _style_axes(axes, title: str, states: list[int]) -> None:
    axes.set_title(title, loc="left", color=INK, fontsize=11)
    axes.set_xlabel("Number of states of the target DFA", color=MUTED)
    axes.set_xticks(states)
    axes.grid(True, axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(which="both", colors=MUTED, labelcolor=MUTED, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)


def plot_results(results: list[Result], directory: str) -> None:
    """Save the plots of results as PNG files in directory."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    os.makedirs(directory, exist_ok=True)
    groups = group_by_states(results)
    states = list(groups)
    saved = []

    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.bar(states, [100 * sum(result.solved for result in group) / len(group) for group in groups.values()],
             width=0.7, color=COLOR)
    axes.set_ylim(0, 100)
    _style_axes(axes, "Solved test cases (%)", states)
    saved.append(os.path.join(directory, "solved.png"))
    figure.savefig(saved[-1], dpi=200)
    plt.close(figure)

    for name, attribute, title, logarithmic in METRIC_PLOTS:
        plotted_states, lower, median, upper = [], [], [], []
        for group_states, group in groups.items():
            values = sorted(getattr(result, attribute) for result in group if result.solved)
            if not values:
                continue
            quartiles = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
            plotted_states.append(group_states)
            lower.append(quartiles[0])
            median.append(quartiles[1])
            upper.append(quartiles[2])

        figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
        axes.fill_between(plotted_states, lower, upper, color=COLOR, alpha=0.15, linewidth=0)
        axes.plot(plotted_states, median, color=COLOR, marker="o", markersize=6, linewidth=1.5,
                  markeredgecolor="white", markeredgewidth=1)
        if logarithmic:
            axes.set_yscale("log")
            axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        else:
            axes.set_ylim(bottom=0)
            axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        _style_axes(axes, f"{title} of solved test cases (median with interquartile range)", states)
        saved.append(os.path.join(directory, f"{name}.png"))
        figure.savefig(saved[-1], dpi=200)
        plt.close(figure)

    print()
    print("Saved plots:")
    for path in saved:
        print(f"  {path}")


def newest_results_file() -> str:
    paths = glob.glob(os.path.join(RESULTS_DIRECTORY, "*.csv"))
    if not paths:
        raise SystemExit(f"No results files in {RESULTS_DIRECTORY}; run run_benchmarks.py first")
    return max(paths, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise and plot a results file of run_benchmarks.py")
    parser.add_argument("results", nargs="?",
                        help=f"Results CSV file. Default: the newest file in {RESULTS_DIRECTORY}")
    args = parser.parse_args()

    path = args.results or newest_results_file()
    results = load_results(path)
    print_summary(path, results)
    plot_results(results, os.path.join(PLOTS_DIRECTORY, os.path.splitext(os.path.basename(path))[0]))


if __name__ == "__main__":
    main()
