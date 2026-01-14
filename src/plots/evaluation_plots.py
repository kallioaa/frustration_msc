"""Plot helpers for evaluation metrics."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt


def plot_sweep_win_rate_bar(
    labels: Iterable[str],
    values: Iterable[float],
) -> None:
    """Plot win rates for sweep runs as a bar chart."""
    labels_list = list(labels)
    values_list = list(values)
    if not values_list:
        raise ValueError("No win_rate values provided.")

    plt.figure(figsize=(12, 6))
    plt.bar(labels_list, values_list)
    plt.xlabel("Sweep Setting")
    plt.ylabel("Win Rate")
    plt.title("Win Rate by Sweep Setting")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
