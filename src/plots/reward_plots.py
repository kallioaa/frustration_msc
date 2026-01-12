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


def plot_moving_average_returns_multi(
    series: dict[str, list[float]], window: int = 100
) -> None:
    """Plot moving average returns for multiple parameter settings."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if not series:
        raise ValueError("series must not be empty")

    plt.figure(figsize=(10, 6))
    for label, returns in series.items():
        if len(returns) < window:
            raise ValueError(f"window is larger than the number of returns for {label}")
        returns_array = np.asarray(returns, dtype=float)
        moving_avg = np.convolve(returns_array, np.ones(window) / window, mode="valid")
        x_values = np.arange(window - 1, len(returns))
        plt.plot(x_values, moving_avg, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Moving Average of Episode Returns")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_moving_average_episode_won(
    episode_won: list[float], window: int = 100
) -> None:
    """Plot the moving average of per-episode wins."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if len(episode_won) < window:
        raise ValueError("window is larger than the number of episodes")

    won_array = np.asarray(episode_won, dtype=float)
    moving_avg = np.convolve(won_array, np.ones(window) / window, mode="valid")
    x_values = np.arange(window - 1, len(episode_won))

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values,
        moving_avg,
        label=f"{window}-Episode Moving Average (Win Rate)",
        color="blue",
    )
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Moving Average of Episode Wins")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_moving_average_episode_won_multi(
    series: dict[str, list[float]], window: int = 100
) -> None:
    """Plot moving average episode wins for multiple parameter settings."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if not series:
        raise ValueError("series must not be empty")

    plt.figure(figsize=(10, 6))
    for label, episode_won in series.items():
        if len(episode_won) < window:
            raise ValueError(
                f"window is larger than the number of episodes for {label}"
            )
        won_array = np.asarray(episode_won, dtype=float)
        moving_avg = np.convolve(won_array, np.ones(window) / window, mode="valid")
        x_values = np.arange(window - 1, len(episode_won))
        plt.plot(x_values, moving_avg, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Moving Average of Episode Wins")
    plt.legend()
    plt.grid(True)
    plt.show()
