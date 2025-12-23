import matplotlib.pyplot as plt


def plot_episode_returns(returns: list[float]) -> None:
    """Plot returns for each episode."""
    plt.figure(figsize=(10, 6))
    plt.plot(returns, label="Return", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns")
    plt.legend()
    plt.grid(True)
    plt.show()
