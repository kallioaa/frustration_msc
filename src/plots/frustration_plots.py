"""Plot helpers for frustration metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from metrics.frustration_metrics import negative_td_errors


def plot_moving_average_td_errors(td_errors: list[float], window: int = 100) -> None:
    """Plot the moving average of negative and positive TD errors."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(td_errors) < window:
        raise ValueError("window is larger than the number of td_errors")

    negative_errors = negative_td_errors(td_errors)
    positive_errors = [float(e) if e > 0.0 else 0.0 for e in td_errors]

    negative_array = np.asarray(negative_errors, dtype=float)
    positive_array = np.asarray(positive_errors, dtype=float)
    negative_avg = np.convolve(negative_array, np.ones(window) / window, mode="valid")
    positive_avg = np.convolve(positive_array, np.ones(window) / window, mode="valid")
    x_values = np.arange(window - 1, len(td_errors))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values,
        negative_avg,
        label=f"{window}-Step Moving Average (Negative TD Error)",
        color="red",
    )
    plt.plot(
        x_values,
        positive_avg,
        label=f"{window}-Step Moving Average (Positive TD Error)",
        color="green",
    )
    plt.xlabel("Step")
    plt.ylabel("TD Error")
    plt.title("Moving Average of TD Error Signs")
    plt.legend()
    plt.grid(True)
    plt.show()
