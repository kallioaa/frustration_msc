"""Plot utilities for sweep results."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


def plot_sweep_training(
    results: list[Dict[str, Any]],
    window_size: int = 100,
    label_fn: Callable[[Dict[str, Any]], str] | None = None,
    plot_specs: list[dict[str, Any]] | None = None,
    start_episode: int = 0,
    end_episode: int | None = None,
    use_td_error_v: bool = False,
    save_dir: str | Path | None = None,
    save_format: str = "pdf",
    title_suffix: str | None = None,
) -> None:
    """Plot training curves from sweep results using metric plot specs.

    When ``use_td_error_v`` is True, TD-error metrics are read from ``{key}_v``
    when available and fall back to ``key`` if the ``_v`` variant is missing.
    Optional ``plot_kwargs`` in each plot spec are forwarded to that metric's
    ``plot_fn``. Training plot specs may additionally define
    ``metric_transform="abs"`` to transform each run series before aggregation.
    When ``title_suffix`` is provided, it is appended to each plot title as
    ``"{title} ({title_suffix})"``.
    """
    if not plot_specs:
        raise ValueError("plot_specs must be provided and non-empty")
    save_dir_path = Path(save_dir) if save_dir is not None else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)
    metric_specs: list[
        tuple[
            str,
            str,
            str,
            str,
            str,
            Callable[..., Any],
            dict[str, Any],
            str | None,
        ]
    ] = []
    for spec in plot_specs:
        source = spec.get("source")
        key = spec.get("key")
        ylabel = spec.get("ylabel")
        title = spec.get("title")
        xlabel = spec.get("xlabel", "Episode")
        plot_fn = spec.get("plot_fn")
        plot_kwargs = spec.get("plot_kwargs", {})
        metric_transform = spec.get("metric_transform")
        if source not in {"reward", "td_error"}:
            raise ValueError(f"Unsupported metric source: {source}")
        if not isinstance(key, str) or not key:
            raise ValueError("Each plot spec must include a non-empty string key")
        if not isinstance(ylabel, str) or not ylabel:
            raise ValueError(
                f"Each plot spec must include a non-empty ylabel for {key}"
            )
        if not isinstance(title, str) or not title:
            raise ValueError(f"Each plot spec must include a non-empty title for {key}")
        if not isinstance(xlabel, str) or not xlabel:
            raise ValueError(f"xlabel must be a non-empty string for {key}")
        if not callable(plot_fn):
            raise ValueError(f"Each plot spec must include callable plot_fn for {key}")
        if not isinstance(plot_kwargs, dict):
            raise ValueError(f"plot_kwargs must be a dict for {key}")
        if metric_transform is not None and metric_transform != "abs":
            raise ValueError(
                f"Unsupported metric_transform for {key}: {metric_transform!r}"
            )
        metric_specs.append(
            (source, key, ylabel, title, xlabel, plot_fn, plot_kwargs, metric_transform)
        )

    grouped: Dict[tuple, Dict[str, Any]] = {}

    for result in results:
        params = result.get("params", {})
        grouped_params = _strip_seed(params)
        key = _freeze(grouped_params)

        training = result.get("training", {})
        reward_metrics = training.get("reward", {})
        td_error_metrics = training.get("td_error", {})

        entry = grouped.setdefault(
            key,
            {
                "params": grouped_params,
                "metric_runs": {},
            },
        )

        for source, metric_key, _, _, _, _, _, metric_transform in metric_specs:
            source_metrics = reward_metrics if source == "reward" else td_error_metrics
            resolved_metric_key = metric_key
            if source == "td_error" and use_td_error_v:
                metric_key_v = f"{metric_key}_v"
                resolved_metric_key = (
                    metric_key_v if metric_key_v in source_metrics else metric_key
                )
            values = source_metrics.get(resolved_metric_key)
            if values is None:
                values = _derive_training_metric_series(
                    metric_key=metric_key,
                    params=params,
                    td_error_metrics=td_error_metrics,
                )
            if values is None:
                continue
            values = _transform_metric_series(values, metric_transform)
            metric_id = f"{source}:{metric_key}:{metric_transform or 'none'}"
            metric_runs = entry["metric_runs"].setdefault(metric_id, [])
            metric_runs.append(values)

    used_filenames: set[str] = set()
    for (
        source,
        metric_key,
        ylabel,
        title,
        xlabel,
        plot_fn,
        plot_kwargs,
        metric_transform,
    ) in metric_specs:
        metric_id = f"{source}:{metric_key}:{metric_transform or 'none'}"
        series_by_label: Dict[str, list[float]] = {}
        runs_by_label: Dict[str, list[list[float]]] = {}
        for entry in grouped.values():
            runs = entry["metric_runs"].get(metric_id, [])
            if not runs:
                continue
            params = entry["params"]
            label = label_fn(params) if label_fn else _format_sweep_label(params)
            series_by_label[label] = _mean_series(runs)
            runs_by_label[label] = runs
        if series_by_label:
            call_plot_kwargs = dict(plot_kwargs)
            if _plot_fn_supports_arg(plot_fn, "runs_by_label"):
                call_plot_kwargs["runs_by_label"] = runs_by_label
            if save_dir_path is not None and _plot_fn_supports_arg(
                plot_fn, "save_path"
            ):
                call_plot_kwargs["save_path"] = _build_plot_save_path(
                    save_dir_path=save_dir_path,
                    title=title,
                    metric_key=metric_key,
                    extension=save_format,
                    used_names=used_filenames,
                )
            resolved_title = f"{title} ({title_suffix})" if title_suffix else title
            plot_fn(
                series_by_label,
                window=window_size,
                ylabel=ylabel,
                title=resolved_title,
                xlabel=xlabel,
                start_episode=start_episode,
                end_episode=end_episode,
                **call_plot_kwargs,
            )


def plot_sweep_evaluation(
    results: list[Dict[str, Any]],
    window_size: int = 100,
    label_fn: Callable[[Dict[str, Any]], str] | None = None,
    plot_specs: list[dict[str, Any]] | None = None,
    start_episode: int = 0,
    end_episode: int | None = None,
    save_dir: str | Path | None = None,
    save_format: str = "pdf",
    title_suffix: str | None = None,
) -> None:
    """Plot evaluation curves from sweep results using metric plot specs.

    Optional ``plot_kwargs`` in each plot spec are forwarded to that metric's
    ``plot_fn``. Evaluation plot specs may additionally define
    ``metric_transform="abs"`` to transform each run series before aggregation.
    When ``title_suffix`` is provided, it is appended to each plot title as
    ``"{title} ({title_suffix})"``.
    """
    if not plot_specs:
        raise ValueError("plot_specs must be provided and non-empty")
    save_dir_path = Path(save_dir) if save_dir is not None else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)
    metric_specs: list[
        tuple[str, str, str, str, Callable[..., Any], dict[str, Any], str | None]
    ] = []
    for spec in plot_specs:
        key = spec.get("key")
        ylabel = spec.get("ylabel")
        title = spec.get("title")
        xlabel = spec.get("xlabel", "Episode")
        plot_fn = spec.get("plot_fn")
        plot_kwargs = spec.get("plot_kwargs", {})
        metric_transform = spec.get("metric_transform")
        if not isinstance(key, str) or not key:
            raise ValueError("Each plot spec must include a non-empty string key")
        if not isinstance(ylabel, str) or not ylabel:
            raise ValueError(
                f"Each plot spec must include a non-empty ylabel for {key}"
            )
        if not isinstance(title, str) or not title:
            raise ValueError(f"Each plot spec must include a non-empty title for {key}")
        if not isinstance(xlabel, str) or not xlabel:
            raise ValueError(f"xlabel must be a non-empty string for {key}")
        if not callable(plot_fn):
            raise ValueError(f"Each plot spec must include callable plot_fn for {key}")
        if not isinstance(plot_kwargs, dict):
            raise ValueError(f"plot_kwargs must be a dict for {key}")
        if metric_transform is not None and metric_transform != "abs":
            raise ValueError(
                f"Unsupported metric_transform for {key}: {metric_transform!r}"
            )
        metric_specs.append(
            (key, ylabel, title, xlabel, plot_fn, plot_kwargs, metric_transform)
        )

    grouped: Dict[tuple, Dict[str, Any]] = {}

    for result in results:
        params = result.get("params", {})
        grouped_params = _strip_seed(params)
        key = _freeze(grouped_params)

        evaluation = result.get("evaluation", {}).get("eval", {})
        entry = grouped.setdefault(
            key,
            {
                "params": grouped_params,
                "metric_runs": {},
            },
        )

        for metric_key, _, _, _, _, _, metric_transform in metric_specs:
            values = evaluation.get(metric_key)
            if values is None:
                continue
            metric_id = f"{metric_key}:{metric_transform or 'none'}"
            values = _transform_metric_series(values, metric_transform)
            metric_runs = entry["metric_runs"].setdefault(metric_id, [])
            metric_runs.append(values)

    used_filenames: set[str] = set()
    for (
        metric_key,
        ylabel,
        title,
        xlabel,
        plot_fn,
        plot_kwargs,
        metric_transform,
    ) in metric_specs:
        metric_id = f"{metric_key}:{metric_transform or 'none'}"
        series_by_label: Dict[str, list[float]] = {}
        runs_by_label: Dict[str, list[list[float]]] = {}
        for entry in grouped.values():
            runs = entry["metric_runs"].get(metric_id, [])
            if not runs:
                continue
            params = entry["params"]
            label = label_fn(params) if label_fn else _format_sweep_label(params)
            series_by_label[label] = _mean_series(runs)
            runs_by_label[label] = runs
        if series_by_label:
            call_plot_kwargs = dict(plot_kwargs)
            if _plot_fn_supports_arg(plot_fn, "runs_by_label"):
                call_plot_kwargs["runs_by_label"] = runs_by_label
            if save_dir_path is not None and _plot_fn_supports_arg(
                plot_fn, "save_path"
            ):
                call_plot_kwargs["save_path"] = _build_plot_save_path(
                    save_dir_path=save_dir_path,
                    title=title,
                    metric_key=metric_key,
                    extension=save_format,
                    used_names=used_filenames,
                )
            resolved_title = f"{title} ({title_suffix})" if title_suffix else title
            plot_fn(
                series_by_label,
                window=window_size,
                ylabel=ylabel,
                title=resolved_title,
                xlabel=xlabel,
                start_episode=start_episode,
                end_episode=end_episode,
                **call_plot_kwargs,
            )


def _format_sweep_label(params: Dict[str, Any]) -> str:
    """Create a compact label from agent/env sweep parameters."""
    agent_kwargs = params.get("agent_kwargs") or {}
    env_kwargs = params.get("env_kwargs") or {}
    parts = []

    if agent_kwargs:
        q_table_label = agent_kwargs.get("initial_q_table_label")
        if q_table_label is None and "initial_q_table" in agent_kwargs:
            q_table_label = "zeros"
        agent_items_list = []
        for key, value in agent_kwargs.items():
            if key == "initial_q_table":
                continue
            if key == "initial_q_table_label":
                agent_items_list.append(f"q_table={value}")
                continue
            agent_items_list.append(f"{key}={value}")
        if q_table_label is not None and "initial_q_table_label" not in agent_kwargs:
            agent_items_list.append(f"q_table={q_table_label}")
        agent_items = ", ".join(agent_items_list)
        parts.append(f"agent: {agent_items}")
    if env_kwargs:
        env_items = ", ".join(f"{k}={v}" for k, v in env_kwargs.items())
        parts.append(f"env: {env_items}")

    return " | ".join(parts) if parts else "run"


def _strip_seed(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return params without seed keys for grouping/labeling."""
    agent_kwargs = dict(params.get("agent_kwargs") or {})
    env_kwargs = dict(params.get("env_kwargs") or {})

    agent_kwargs.pop("seed", None)
    env_kwargs.pop("seed", None)

    stripped: Dict[str, Any] = {}
    if agent_kwargs:
        stripped["agent_kwargs"] = agent_kwargs
    if env_kwargs:
        stripped["env_kwargs"] = env_kwargs
    return stripped


def _freeze(value: Any) -> Any:
    """Make a nested structure hashable for dict keys."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, np.ndarray):
        return ("__ndarray__", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mean_series(series_list: list[list[float]]) -> list[float]:
    """Element-wise mean of multiple series, truncating to shortest."""
    min_len = min(len(series) for series in series_list)
    if min_len == 0:
        return []
    stacked = np.array([series[:min_len] for series in series_list], dtype=float)
    return list(np.mean(stacked, axis=0))


def _derive_training_metric_series(
    metric_key: str,
    params: Dict[str, Any],
    td_error_metrics: Dict[str, list[float]],
) -> list[float] | None:
    """Derive synthetic training metrics from saved params and episode logs."""
    if metric_key == "effective_learning_rate_per_episode":
        return _effective_learning_rate_per_episode(
            params=params,
            td_error_metrics=td_error_metrics,
        )

    return None


def _effective_learning_rate_per_episode(
    params: Dict[str, Any],
    td_error_metrics: Dict[str, list[float]],
) -> list[float] | None:
    """Return per-episode effective learning rate for biased SARSA agents."""
    agent_kwargs = dict(params.get("agent_kwargs") or {})

    # Positivity-bias SARSA switches between alpha_positive/alpha_negative.
    if "alpha_positive" in agent_kwargs and "alpha_negative" in agent_kwargs:
        ratio = td_error_metrics.get("positive_lr_update_ratio_per_episode")
        if not ratio:
            return None
        try:
            alpha_pos = float(agent_kwargs["alpha_positive"])
            alpha_neg = float(agent_kwargs["alpha_negative"])
        except (TypeError, ValueError):
            return None
        r = np.asarray(ratio, dtype=float)
        return list(alpha_neg + r * (alpha_pos - alpha_neg))

    # Confirmation-bias SARSA switches between alpha_conf/alpha_disconf.
    if "alpha_conf" in agent_kwargs and "alpha_disconf" in agent_kwargs:
        ratio = td_error_metrics.get("confirmatory_update_ratio_per_episode")
        if not ratio:
            return None
        try:
            alpha_conf = float(agent_kwargs["alpha_conf"])
            alpha_disconf = float(agent_kwargs["alpha_disconf"])
        except (TypeError, ValueError):
            return None
        r = np.asarray(ratio, dtype=float)
        return list(alpha_disconf + r * (alpha_conf - alpha_disconf))

    return None


def _transform_metric_series(
    values: list[float], metric_transform: str | None
) -> list[float]:
    """Apply an optional transform to a metric series."""
    if metric_transform is None:
        return list(values)
    if metric_transform == "abs":
        return list(np.abs(np.asarray(values, dtype=float)))
    raise ValueError(f"Unsupported metric_transform: {metric_transform}")


def _plot_fn_supports_arg(plot_fn: Callable[..., Any], arg_name: str) -> bool:
    """Return True if a callable accepts ``arg_name`` or arbitrary kwargs."""
    try:
        signature = inspect.signature(plot_fn)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return arg_name in signature.parameters


def _build_plot_save_path(
    save_dir_path: Path,
    title: str,
    metric_key: str,
    extension: str,
    used_names: set[str],
) -> Path:
    """Create a unique save path using the plot title as the filename stem."""
    ext = extension.lstrip(".") or "pdf"
    stem = _slugify_filename(title) or _slugify_filename(metric_key) or "plot"
    candidate = stem
    suffix = 2
    while candidate in used_names:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return save_dir_path / f"{candidate}.{ext}"


def _slugify_filename(value: str) -> str:
    """Convert a human-readable title into a filesystem-friendly filename."""
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return text.strip("_")
