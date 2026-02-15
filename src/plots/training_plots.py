"""Shared plot helpers for training/evaluation series."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_moving_average_multi(
    series: dict[str, list[float]],
    window: int = 100,
    ylabel: str = "Value",
    title: str = "Moving Average",
    xlabel: str = "Episode",
) -> None:
    """Plot moving average for multiple parameter settings."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if not series:
        raise ValueError("series must not be empty")

    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        if len(values) < window:
            raise ValueError(f"window is larger than the number of points for {label}")
        values_array = np.asarray(values, dtype=float)
        moving_avg = np.convolve(values_array, np.ones(window) / window, mode="valid")
        x_values = np.arange(window - 1, len(values))
        plt.plot(x_values, moving_avg, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
