from __future__ import annotations

import sys
from pathlib import Path as _PathBootstrap

REPO_ROOT = _PathBootstrap(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from regime_toolkit.systems import AutonomousForcedVanDerPol
from regime_toolkit.simulate import SimulationConfig, simulate
from regime_toolkit.oracles import vdp_complexity_oracle
from regime_toolkit.io import save_json
from regime_toolkit.plots import set_publication_style
from regime_toolkit.calibration import resolve_sweep_values



def run_scout(
    *,
    outdir: Path,
    A_values: list[float] | None = None,
    omega_values: list[float] | None = None,
    n_A: int | None = 7,
    A_min: float | None = 0.2,
    A_max: float | None = 1.6,
    n_omega: int | None = 7,
    omega_min: float | None = 0.8,
    omega_max: float | None = 1.2,
    mu_fixed: float = 8.0,
    seed: int = 0,
    sim_config: SimulationConfig | None = None,
    cluster_threshold: int = 4,
    period_cv_threshold: float = 0.03,
    norm_spread_threshold: float = 0.75,
) -> dict:
    set_publication_style()
    outdir.mkdir(parents=True, exist_ok=True)

    A_grid = resolve_sweep_values(param_values=A_values, n=n_A, min_value=A_min, max_value=A_max, scale="linear")
    omega_grid = resolve_sweep_values(param_values=omega_values, n=n_omega, min_value=omega_min, max_value=omega_max, scale="linear")

    system = AutonomousForcedVanDerPol()
    base_params = system.default_params()
    base_params["mu"] = float(mu_fixed)

    sim_cfg = sim_config or SimulationConfig(
        t_final=120.0,
        dt=0.05,
        transient=80.0,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
        max_step=0.05,
        reset_time_after_transient=True,
    )

    rows = []
    cluster_map = np.empty((omega_grid.size, A_grid.size), dtype=float)
    label_map = np.empty_like(cluster_map)
    period_cv_map = np.empty_like(cluster_map)

    for i, omega in enumerate(omega_grid):
        for j, A in enumerate(A_grid):
            params = dict(base_params)
            params["omega"] = float(omega)
            params["A"] = float(A)
            t, x, _ = simulate(system, params, sim_cfg, seed=seed)
            oracle = vdp_complexity_oracle(
                t,
                x,
                tail_fraction=0.7,
                phase_index=2,
                state_dims=(0, 1),
                cluster_tol_fraction=0.05,
                cluster_threshold=cluster_threshold,
                period_cv_threshold=period_cv_threshold,
                norm_spread_threshold=norm_spread_threshold,
            )
            row = {
                "A": float(A),
                "omega": float(omega),
                "cluster_count": int(oracle.cluster_count),
                "cluster_ratio": float(oracle.cluster_ratio),
                "norm_spread": float(oracle.norm_spread),
                "period_cv": None if oracle.period_cv is None else float(oracle.period_cv),
                "amplitude": float(oracle.amplitude),
                "is_complex": int(oracle.is_complex),
            }
            rows.append(row)
            cluster_map[i, j] = oracle.cluster_count
            label_map[i, j] = 1.0 if oracle.is_complex else 0.0
            period_cv_map[i, j] = np.nan if oracle.period_cv is None else float(oracle.period_cv)

    csv_path = outdir / "vdp_scout.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



    omega_summaries = []
    for i, omega in enumerate(omega_grid):
        mask = [row for row in rows if float(row["omega"]) == float(omega)]
        labels = np.asarray([row["is_complex"] for row in mask], dtype=float)
        clusters = np.asarray([row["cluster_count"] for row in mask], dtype=float)
        unique_labels = int(np.unique(labels).size)
        transition_exists = unique_labels > 1
        boundary_A = None
        for j in range(len(mask) - 1):
            if clusters[j] > float(cluster_threshold) and clusters[j + 1] <= float(cluster_threshold):
                A0, A1 = float(mask[j]["A"]), float(mask[j + 1]["A"])
                c0, c1 = float(clusters[j]), float(clusters[j + 1])
                alpha = 1.0 if c1 == c0 else (float(cluster_threshold) - c0) / (c1 - c0)
                boundary_A = float(A0 + alpha * (A1 - A0))
                break
        cluster_drop = float(clusters[0] - clusters[-1])
        score = float((1.0 if transition_exists else 0.0) * max(cluster_drop, 0.0))
        omega_summaries.append({
            "omega": float(omega),
            "unique_labels": unique_labels,
            "transition_exists": bool(transition_exists),
            "complexity_boundary_A": boundary_A,
            "cluster_drop": cluster_drop,
            "selection_score": score,
        })
    best = max(omega_summaries, key=lambda row: row["selection_score"]) if omega_summaries else None

    payload = {
        "A_values": A_grid,
        "omega_values": omega_grid,
        "cluster_threshold": int(cluster_threshold),
        "period_cv_threshold": float(period_cv_threshold),
        "norm_spread_threshold": float(norm_spread_threshold),
        "rows": rows,
        "omega_summaries": omega_summaries,
        "recommended_omega": None if best is None else best.get("omega"),
    }
    save_json(payload, outdir / "vdp_scout.json")

    def _heatmap(array: np.ndarray, title: str, savepath: Path) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(array, origin="lower", aspect="auto", extent=[A_grid[0], A_grid[-1], omega_grid[0], omega_grid[-1]])
        ax.set_xlabel("A")
        ax.set_ylabel("omega")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.savefig(savepath)
        plt.close(fig)

    figs = outdir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    _heatmap(cluster_map, "Strobe cluster count", figs / "cluster_count_heatmap.png")
    _heatmap(label_map, "Complex-response label", figs / "complex_label_heatmap.png")
    _heatmap(period_cv_map, "Tail-peak period CV", figs / "period_cv_heatmap.png")

    return payload



def main() -> None:
    parser = argparse.ArgumentParser(description="Scout A-omega cuts for the autonomous forced VdP final oracle.")
    parser.add_argument("--out", type=str, default="outputs/vdp_scout")
    parser.add_argument("--n_A", type=int, default=7)
    parser.add_argument("--A_min", type=float, default=0.2)
    parser.add_argument("--A_max", type=float, default=1.6)
    parser.add_argument("--n_omega", type=int, default=7)
    parser.add_argument("--omega_min", type=float, default=0.8)
    parser.add_argument("--omega_max", type=float, default=1.2)
    parser.add_argument("--mu_fixed", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_scout(
        outdir=Path(args.out),
        n_A=args.n_A,
        A_min=args.A_min,
        A_max=args.A_max,
        n_omega=args.n_omega,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        mu_fixed=args.mu_fixed,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
