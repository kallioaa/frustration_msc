"""Shared plot helpers for training/evaluation series."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import t as _scipy_student_t
except Exception:  # pragma: no cover - optional dependency
    _scipy_student_t = None


def _critical_value(level: float, n: int, method: str) -> float:
    """Return a critical value for a two-sided confidence interval."""
    alpha = 1.0 - level
    if method == "normal":
        return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
    if method != "t":
        raise ValueError(f"Unsupported ci_method: {method}")
    if n < 2:
        return 0.0
    if _scipy_student_t is not None:
        return float(_scipy_student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
    # Fallback when SciPy is unavailable.
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _mean_confidence_interval(
    values: list[float],
    level: float = 0.95,
    method: str = "t",
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) CI bounds for a list of run-level values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return (math.nan, math.nan, math.nan)

    mean = float(np.mean(x))
    if n == 1:
        return (mean, mean, mean)

    std = float(np.std(x, ddof=1))
    sem = std / math.sqrt(n)
    crit = _critical_value(level=level, n=n, method=method)
    half_width = crit * sem
    return (mean, mean - half_width, mean + half_width)


def plot_moving_average_multi(
    series: dict[str, list[float]],
    window: int = 100,
    ylabel: str = "Value",
    title: str = "Moving Average",
    xlabel: str = "Episode",
    start_episode: int = 0,
    end_episode: int | None = None,
    figsize: tuple[float, float] = (10, 6),
    legend_loc: str = "best",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    show_legend: bool = True,
    line_alpha: float = 1.0,
    grid_alpha: float = 1.0,
    tight_layout: bool = False,
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    save_bbox_inches: str | None = "tight",
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

    plt.figure(figsize=figsize)
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
        plt.plot(x_values, moving_avg, label=label, alpha=line_alpha)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show_legend:
        plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
    plt.grid(True, alpha=grid_alpha)
    if ylim is not None:
        plt.ylim(*ylim)
    if tight_layout:
        plt.tight_layout()
    if save_path is not None:
        savefig_kwargs = {}
        if save_bbox_inches is not None:
            savefig_kwargs["bbox_inches"] = save_bbox_inches
        plt.savefig(save_path, **savefig_kwargs)
    plt.show()


def plot_bar_mean_multi(
    series: dict[str, list[float]],
    window: int = 100,
    ylabel: str = "Value",
    title: str = "Mean Value",
    xlabel: str = "Setting",
    start_episode: int = 0,
    end_episode: int | None = None,
    figsize: tuple[float, float] = (10, 6),
    xtick_rotation: float = 25,
    xtick_ha: str = "right",
    bar_alpha: float = 1.0,
    grid_alpha: float = 0.3,
    tight_layout: bool = True,
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    save_bbox_inches: str | None = "tight",
    runs_by_label: dict[str, list[list[float]]] | None = None,
    ci_level: float | None = 0.95,
    ci_method: str = "t",
    ci_capsize: float = 4.0,
) -> None:
    """Plot one bar per setting using the mean of each series.

    When ``runs_by_label`` is provided, bars are computed from per-run means and
    confidence intervals are drawn across runs (typically seeds).
    """
    del window  # kept for compatibility with generic plot function call sites
    if not series:
        raise ValueError("series must not be empty")
    if start_episode < 0:
        raise ValueError("start_episode must be >= 0")
    if end_episode is not None and end_episode <= start_episode:
        raise ValueError("end_episode must be greater than start_episode")
    if ci_level is not None and not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be in (0, 1) or None")

    labels: list[str] = []
    means: list[float] = []
    lower_errs: list[float] = []
    upper_errs: list[float] = []
    for label, values in series.items():
        sliced = values[start_episode:end_episode]
        if not sliced:
            continue
        mean_value = float(np.mean(np.asarray(sliced, dtype=float)))
        lower = mean_value
        upper = mean_value

        if runs_by_label is not None and ci_level is not None:
            run_means: list[float] = []
            for run_values in runs_by_label.get(label, []):
                run_sliced = run_values[start_episode:end_episode]
                if not run_sliced:
                    continue
                run_means.append(float(np.mean(np.asarray(run_sliced, dtype=float))))
            if run_means:
                mean_value, lower, upper = _mean_confidence_interval(
                    run_means,
                    level=ci_level,
                    method=ci_method,
                )

        labels.append(label)
        means.append(mean_value)
        lower_errs.append(max(0.0, mean_value - lower))
        upper_errs.append(max(0.0, upper - mean_value))

    if not labels:
        raise ValueError("no values available after slicing for any series")

    plt.figure(figsize=figsize)
    x = np.arange(len(labels))
    bar_kwargs = {"alpha": bar_alpha}
    if ci_level is not None and any(err > 0.0 for err in lower_errs + upper_errs):
        bar_kwargs["yerr"] = np.vstack([lower_errs, upper_errs])
        bar_kwargs["capsize"] = ci_capsize
    plt.bar(x, means, **bar_kwargs)
    plt.xticks(x, labels, rotation=xtick_rotation, ha=xtick_ha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=grid_alpha)
    if ylim is not None:
        plt.ylim(*ylim)
    if tight_layout:
        plt.tight_layout()
    if save_path is not None:
        savefig_kwargs = {}
        if save_bbox_inches is not None:
            savefig_kwargs["bbox_inches"] = save_bbox_inches
        plt.savefig(save_path, **savefig_kwargs)
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
