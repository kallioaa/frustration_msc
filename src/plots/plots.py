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
    start_episode: int = 0,
    end_episode: int | None = None,
) -> None:
    """Plot moving average for multiple parameter settings."""
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if not series:
        raise ValueError("series must not be empty")
    if start_episode < 0:
        raise ValueError("start_episode must be >= 0")
    if end_episode is not None and end_episode <= start_episode:
        raise ValueError("end_episode must be greater than start_episode")

    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        episode_offset = start_episode
        if start_episode or end_episode is not None:
            values = values[start_episode:end_episode]
        if len(values) < window:
            raise ValueError(f"window is larger than the number of points for {label}")
        values_array = np.asarray(values, dtype=float)
        moving_avg = np.convolve(values_array, np.ones(window) / window, mode="valid")
        x_values = np.arange(
            episode_offset + window - 1,
            episode_offset + len(values),
        )
        plt.plot(x_values, moving_avg, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_bar_mean_multi(
    series: dict[str, list[float]],
    window: int = 100,
    ylabel: str = "Value",
    title: str = "Mean Value",
    xlabel: str = "Setting",
    start_episode: int = 0,
    end_episode: int | None = None,
) -> None:
    """Plot one bar per setting using the mean of each series."""
    del window  # kept for compatibility with generic plot function call sites
    if not series:
        raise ValueError("series must not be empty")
    if start_episode < 0:
        raise ValueError("start_episode must be >= 0")
    if end_episode is not None and end_episode <= start_episode:
        raise ValueError("end_episode must be greater than start_episode")

    labels: list[str] = []
    means: list[float] = []
    for label, values in series.items():
        sliced = values[start_episode:end_episode]
        if not sliced:
            continue
        labels.append(label)
        means.append(float(np.mean(np.asarray(sliced, dtype=float))))

    if not labels:
        raise ValueError("no values available after slicing for any series")

    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    plt.bar(x, means)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mid_eval_metric(
    results,
    metric_key="total_reward_per_episode",
    title=None,
    max_points_from_start: int | None = None,
):
    grouped = {}

    for r in results:
        params = r.get("params", {})
        key = str(_strip_seed_from_params(params))
        checkpoints = r.get("training", {}).get("mid_eval", [])
        if not checkpoints:
            continue

        episodes = [int(c["episode"]) for c in checkpoints if metric_key in c]
        values = [float(c[metric_key]) for c in checkpoints if metric_key in c]
        if max_points_from_start is not None:
            episodes = episodes[:max_points_from_start]
            values = values[:max_points_from_start]
        if not episodes:
            continue

        if key not in grouped:
            grouped[key] = {"label": _label_from_params(params), "runs": []}
        grouped[key]["runs"].append((episodes, values))

    plt.figure(figsize=(8, 5))
    for g in grouped.values():
        min_len = min(len(v) for _, v in g["runs"])
        xs = g["runs"][0][0][:min_len]  # same eval schedule across seeds
        ys = np.array([v[:min_len] for _, v in g["runs"]], dtype=float)
        mean_y = ys.mean(axis=0)
        plt.plot(xs, mean_y, label=g["label"])

    plt.xlabel("Training episode")
    plt.ylabel(metric_key)
    plt.title(title or f"Mid-training evaluation: {metric_key}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _strip_seed_from_params(params):
    p = {"agent_kwargs": dict((params.get("agent_kwargs") or {}))}
    p["agent_kwargs"].pop("seed", None)
    return p


def _label_from_params(params):
    ak = params.get("agent_kwargs", {})
    if "alpha_conf" in ak:
        return f"ac={ak['alpha_conf']}, ad={ak['alpha_disconf']}"
    if "alpha_positive" in ak:
        return f"ap={ak['alpha_positive']}, an={ak['alpha_negative']}"
    return str(ak)
