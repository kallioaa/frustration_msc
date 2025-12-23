import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Optional, Sequence, Tuple


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


def plot_state_values(
    state_values: Iterable[float],
    grid_shape: Optional[Tuple[int, int]] = None,
    annotate: bool = False,
) -> None:
    """Plot state values as a line plot or 2D heatmap.

    Args:
        state_values: Flat list/array of values per discrete state.
        grid_shape: Optional (rows, cols) to render a 2D heatmap.
        annotate: If True and grid_shape is set, draw values in each cell.
    """
    values = np.asarray(list(state_values), dtype=float)
    if values.size == 0:
        raise ValueError("state_values must be non-empty")

    if grid_shape is None:
        plt.figure(figsize=(10, 6))
        plt.plot(values, label="State Value", color="blue")
        plt.xlabel("State")
        plt.ylabel("Value")
        plt.title("State Values")
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    rows, cols = grid_shape
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape must contain positive integers")
    if values.size != rows * cols:
        raise ValueError("grid_shape does not match number of state values")

    grid = values.reshape(rows, cols)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(grid, cmap="viridis", origin="upper")
    plt.title("State Values")
    plt.colorbar(im, shrink=0.8)
    plt.xticks(range(cols))
    plt.yticks(range(rows))

    if annotate:
        for r in range(rows):
            for c in range(cols):
                plt.text(
                    c,
                    r,
                    f"{grid[r, c]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )

    plt.tight_layout()
    plt.show()
