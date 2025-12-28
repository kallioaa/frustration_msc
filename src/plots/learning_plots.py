import matplotlib.pyplot as plt
import numpy as np


def plot_moving_average_returns(returns: list[float], window: int = 100) -> None:
    """Plot the moving average of episode returns."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(returns) < window:
        raise ValueError("window is larger than the number of returns")

    returns_array = np.asarray(returns, dtype=float)
    moving_avg = np.convolve(returns_array, np.ones(window) / window, mode="valid")
    x_values = np.arange(window - 1, len(returns))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values, moving_avg, label=f"{window}-Episode Moving Average", color="green"
    )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Moving Average of Episode Returns")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_moving_average_td_errors(td_errors: list[float], window: int = 100) -> None:
    """Plot the moving average of TD errors."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(td_errors) < window:
        raise ValueError("window is larger than the number of td_errors")

    errors_array = np.asarray(td_errors, dtype=float)
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
