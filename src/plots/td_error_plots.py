"""Plot helpers for frustration metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_moving_average_td_errors(td_errors: list[float], window: int = 100) -> None:
    """Plot the moving average of TD errors."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(td_errors) < window:
        raise ValueError("window is larger than the number of td_errors")

    errors_array = np.asarray(td_errors, dtype=float)
    # errors should all be positive for the average
    errors_array = np.abs(errors_array)

    moving_avg = np.convolve(errors_array, np.ones(window) / window, mode="valid")
    x_values = np.arange(window - 1, len(td_errors))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values, moving_avg, label=f"{window}-Episode Moving Average", color="orange"
    )
    plt.xlabel("Episode")
    plt.ylabel("TD Error")
    plt.title("Moving Average of TD Errors")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_moving_average_td_errors_multi(
    series: dict[str, list[float]], window: int = 100
) -> None:
    """Plot moving average of TD errors for multiple parameter settings."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if not series:
        raise ValueError("series must not be empty")

    plt.figure(figsize=(10, 6))
    for label, td_errors in series.items():
        if len(td_errors) < window:
            raise ValueError(
                f"window is larger than the number of td_errors for {label}"
            )
        errors_array = np.asarray(td_errors, dtype=float)
        errors_array = np.abs(errors_array)
        moving_avg = np.convolve(errors_array, np.ones(window) / window, mode="valid")
        x_values = np.arange(window - 1, len(td_errors))
        plt.plot(x_values, moving_avg, label=label)

    plt.xlabel("Episode")
    plt.ylabel("TD Error")
    plt.title("Moving Average of TD Errors")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_moving_average_td_errors_neg_and_pos(
    td_errors: list[float], window: int = 100
) -> None:
    """Plot the moving average of negative and positive TD errors."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(td_errors) < window:
        raise ValueError("window is larger than the number of td_errors")

    negative_errors = [float(e) if e < 0.0 else 0.0 for e in td_errors]
    positive_errors = [float(e) if e >= 0.0 else 0.0 for e in td_errors]

    negative_array = np.asarray(negative_errors, dtype=float)
    positive_array = np.asarray(positive_errors, dtype=float)
    negative_avg = np.convolve(negative_array, np.ones(window) / window, mode="valid")
    positive_avg = np.convolve(positive_array, np.ones(window) / window, mode="valid")
    x_values = np.arange(window - 1, len(td_errors))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values,
        negative_avg,
        label=f"{window}-Episode Moving Average (Negative TD Error)",
        color="red",
    )
    plt.plot(
        x_values,
        positive_avg,
        label=f"{window}-Episode Moving Average (Positive TD Error)",
        color="green",
    )
    plt.xlabel("Episode")
    plt.ylabel("TD Error")
    plt.title("Moving Average of TD Error Signs")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_moving_average_td_errors_neg_multi(
    series: dict[str, list[float]], window: int = 100
) -> None:
    """Plot moving average of negative TD errors for multiple runs."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if not series:
        raise ValueError("series must not be empty")

    plt.figure(figsize=(10, 6))
    for label, td_errors in series.items():
        if len(td_errors) < window:
            raise ValueError(
                f"window is larger than the number of td_errors for {label}"
            )
        negative_errors = [float(e) if e < 0.0 else 0.0 for e in td_errors]

        negative_array = np.asarray(negative_errors, dtype=float)
        negative_avg = np.convolve(
            negative_array, np.ones(window) / window, mode="valid"
        )
        x_values = np.arange(window - 1, len(td_errors))

        plt.plot(
            x_values,
            negative_avg,
            label=label,
        )

    plt.xlabel("Episode")
    plt.ylabel("TD Error")
    plt.title("Moving Average of Negative TD Errors")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_frustration_quantiles(
    td_errors: list[float],
    window: int = 100,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> None:
    """Plot rolling quantiles for negative TD errors."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(td_errors) < window:
        raise ValueError("window is larger than the number of td_errors")
    if not quantiles:
        raise ValueError("quantiles must be non-empty")

    negative_errors = [float(e) if e < 0.0 else 0.0 for e in td_errors]
    x_values = np.arange(window - 1, len(td_errors))

    plt.figure(figsize=(10, 6))
    for q in quantiles:
        if q < 0.0 or q > 1.0:
            raise ValueError("quantiles must be between 0 and 1")
        series = [
            float(np.quantile(negative_errors[i - window + 1 : i + 1], q))
            for i in range(window - 1, len(negative_errors))
        ]
        plt.plot(x_values, series, label=f"Quantile {q:.2f}")

    plt.xlabel("Episode")
    plt.ylabel("Negative TD Error")
    plt.title("Rolling Quantiles of Negative TD Errors")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_frustration_rate(
    frustration_rate_per_episode: list[float], window: int = 100
) -> None:
    """Plot moving average of per-episode frustration rate."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(frustration_rate_per_episode) < window:
        raise ValueError("window is larger than the number of episodes")

    values = np.asarray(frustration_rate_per_episode, dtype=float)
    moving_avg = np.convolve(values, np.ones(window) / window, mode="valid")
    x_values = np.arange(window - 1, len(values))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values,
        moving_avg,
        label=f"{window}-Episode Moving Average (Frustration Rate)",
        color="purple",
    )
    plt.xlabel("Episode")
    plt.ylabel("Frustration Rate")
    plt.title("Frustration Rate per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_frustration_rate_multi(
    series: dict[str, list[float]], window: int = 100
) -> None:
    """Plot moving average of frustration rate for multiple parameter settings."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if not series:
        raise ValueError("series must not be empty")

    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        if len(values) < window:
            raise ValueError(
                f"window is larger than the number of episodes for {label}"
            )
        values_array = np.asarray(values, dtype=float)
        moving_avg = np.convolve(values_array, np.ones(window) / window, mode="valid")
        x_values = np.arange(window - 1, len(values_array))
        plt.plot(x_values, moving_avg, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Frustration Rate")
    plt.title("Frustration Rate per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_tail_frustration(
    tail_frustration_per_episode: list[float], window: int = 100
) -> None:
    """Plot moving average of per-episode tail frustration."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(tail_frustration_per_episode) < window:
        raise ValueError("window is larger than the number of episodes")

    values = np.asarray(tail_frustration_per_episode, dtype=float)
    moving_avg = np.convolve(values, np.ones(window) / window, mode="valid")
    x_values = np.arange(window - 1, len(values))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values,
        moving_avg,
        label=f"{window}-Episode Moving Average (Tail Frustration)",
        color="orange",
    )
    plt.xlabel("Episode")
    plt.ylabel("Tail Frustration")
    plt.title("Tail Frustration per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cvar_tail_frustration(
    cvar_tail_frustration_per_episode: list[float], window: int = 100
) -> None:
    """Plot moving average of per-episode CVaR tail frustration."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(cvar_tail_frustration_per_episode) < window:
        raise ValueError("window is larger than the number of episodes")

    values = np.asarray(cvar_tail_frustration_per_episode, dtype=float)
    moving_avg = np.convolve(values, np.ones(window) / window, mode="valid")
    x_values = np.arange(window - 1, len(values))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values,
        moving_avg,
        label=f"{window}-Episode Moving Average (CVaR Tail Frustration)",
        color="brown",
    )
    plt.xlabel("Episode")
    plt.ylabel("CVaR Tail Frustration")
    plt.title("CVaR Tail Frustration per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()
