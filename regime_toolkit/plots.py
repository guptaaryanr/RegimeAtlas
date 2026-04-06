from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Sequence, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_publication_style() -> None:
    """
    A conservative, journal-friendly style that avoids LaTeX dependencies.

    If you already have house styles, replace this function and keep the rest of the plotting API.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metric_vs_param(
    results: Dict[str, Any],
    metric: str,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    savepath: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot a single metric curve vs the swept control parameter.
    """
    p = results["param_values"]
    y = results["metrics"][metric]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p, y, marker="o", markersize=3.5)

    ax.set_xlabel(xlabel or results["control_param"])
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)

    if savepath is not None:
        _ensure_dir(savepath.parent)
        fig.savefig(savepath)
    if show:
        plt.show()
    return fig


def plot_lyapunov_curves(
    results: Dict[str, Any],
    exp_keys: Sequence[str],
    *,
    xlabel: Optional[str] = None,
    ylabel: str = "Lyapunov exponent",
    title: Optional[str] = None,
    savepath: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot one or more Lyapunov exponent curves vs parameter.
    """
    p = results["param_values"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for k in exp_keys:
        ax.plot(p, results["metrics"][k], marker="o", markersize=3.5, label=k)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(xlabel or results["control_param"])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)

    if savepath is not None:
        _ensure_dir(savepath.parent)
        fig.savefig(savepath)
    if show:
        plt.show()
    return fig


def plot_attractor_projection(
    t: np.ndarray,
    x: np.ndarray,
    dims: Tuple[int, int] = (0, 1),
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    stride: int = 1,
    savepath: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Phase portrait / attractor projection in 2D.
    """
    i, j = dims
    xs = x[::stride, i]
    ys = x[::stride, j]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, linewidth=1.0)
    ax.set_xlabel(xlabel or f"x[{i}]")
    ax.set_ylabel(ylabel or f"x[{j}]")
    if title:
        ax.set_title(title)

    if savepath is not None:
        _ensure_dir(savepath.parent)
        fig.savefig(savepath)
    if show:
        plt.show()
    return fig


def plot_regime_atlas(
    results: Dict[str, Any],
    metric_keys: Sequence[str],
    *,
    xlabel: Optional[str] = None,
    titles: Optional[Sequence[str]] = None,
    savepath: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Multi-panel summary plot: each metric on its own axis, shared parameter axis.

    This is the “regime atlas” view: you can visually align where different diagnostics change.
    """
    p = results["param_values"]
    n = len(metric_keys)
    if titles is None:
        titles = metric_keys
    if len(titles) != n:
        raise ValueError("titles must match metric_keys length")

    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, key, ttl in zip(axes, metric_keys, titles):
        ax.plot(p, results["metrics"][key], marker="o", markersize=3.5)
        ax.set_ylabel(key)
        ax.set_title(ttl)

    axes[-1].set_xlabel(xlabel or results["control_param"])

    if savepath is not None:
        _ensure_dir(savepath.parent)
        fig.savefig(savepath)
    if show:
        plt.show()
    return fig


def plot_curve_with_boundaries(
    param_values: np.ndarray,
    values: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    boundaries: Optional[Sequence[tuple[float, str]]] = None,
    savepath: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot a single curve with optional vertical boundary markers.
    """
    x = np.asarray(param_values, dtype=float)
    y = np.asarray(values, dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, marker="o", markersize=3.5)
    if boundaries is not None:
        for val, label in boundaries:
            ax.axvline(float(val), linewidth=1.2, linestyle="--", label=label)
        if len(boundaries) > 0:
            ax.legend(frameon=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if savepath is not None:
        _ensure_dir(savepath.parent)
        fig.savefig(savepath)
    if show:
        plt.show()
    return fig


def plot_boundary_overlay(
    param_values: np.ndarray,
    structural_score: np.ndarray,
    oracle_curve: np.ndarray,
    *,
    xlabel: str,
    structural_ylabel: str,
    oracle_ylabel: str,
    structural_boundary: Optional[float] = None,
    qualitative_boundary: Optional[float] = None,
    qualitative_label: str = "qualitative boundary",
    structural_label: str = "structural boundary",
    savepath: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Two-panel overlay comparing structural and qualitative boundary curves.
    """
    x = np.asarray(param_values, dtype=float)
    score = np.asarray(structural_score, dtype=float)
    oracle = np.asarray(oracle_curve, dtype=float)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0, ax1 = axes
    ax0.plot(x, score, marker="o", markersize=3.5)
    ax1.plot(x, oracle, marker="o", markersize=3.5)

    if structural_boundary is not None:
        ax0.axvline(
            float(structural_boundary),
            linestyle="--",
            linewidth=1.2,
            label=structural_label,
        )
        ax1.axvline(
            float(structural_boundary),
            linestyle="--",
            linewidth=1.2,
            label=structural_label,
        )
    if qualitative_boundary is not None:
        ax0.axvline(
            float(qualitative_boundary),
            linestyle=":",
            linewidth=1.2,
            label=qualitative_label,
        )
        ax1.axvline(
            float(qualitative_boundary),
            linestyle=":",
            linewidth=1.2,
            label=qualitative_label,
        )

    ax0.set_ylabel(structural_ylabel)
    ax1.set_ylabel(oracle_ylabel)
    ax1.set_xlabel(xlabel)
    ax0.set_title("Boundary overlay")
    ax0.legend(frameon=False)
    ax1.legend(frameon=False)

    if savepath is not None:
        _ensure_dir(savepath.parent)
        fig.savefig(savepath)
    if show:
        plt.show()
    return fig


def plot_robustness_boundary_summary(
    summary_rows: Sequence[Dict[str, Any]],
    *,
    savepath: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize per-case boundary shifts from the robustness harness.

    If boundary CI widths are available, show them as symmetric error bars around the
    boundary shift point. This makes replicate-based uncertainty visible without changing
    the basic plotting contract used by earlier phases.
    """
    names = [row["case"] for row in summary_rows]
    shifts = np.asarray([row["boundary_shift"] for row in summary_rows], dtype=float)
    ci_widths = np.asarray(
        [
            (
                np.nan
                if row.get("boundary_ci_width") is None
                else float(row.get("boundary_ci_width")) / 2.0
            )
            for row in summary_rows
        ],
        dtype=float,
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(names))
    if np.any(np.isfinite(ci_widths)):
        yerr = np.where(np.isfinite(ci_widths), ci_widths, 0.0)
        ax.errorbar(x, shifts, yerr=yerr, fmt="o")
    else:
        ax.plot(x, shifts, marker="o", linestyle="none")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Boundary shift from baseline")
    ax.set_title("Robustness boundary shifts")

    if savepath is not None:
        _ensure_dir(savepath.parent)
        fig.savefig(savepath)
    if show:
        plt.show()
    return fig
