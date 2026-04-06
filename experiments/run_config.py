from __future__ import annotations

import sys
from pathlib import Path as _PathBootstrap

REPO_ROOT = _PathBootstrap(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from pathlib import Path

from regime_toolkit.simulate import SimulationConfig
from experiments.fhn_pipeline import run_pipeline as run_fhn
from experiments.fhn_nuisance_pipeline import run_pipeline as run_fhn_nuisance
from experiments.vdp_pipeline import run_pipeline as run_vdp
from experiments.vdp_nuisance_pipeline import run_pipeline as run_vdp_nuisance
from experiments.vdp_scout import run_scout as run_vdp_scout

SUPPORTED_EXPERIMENTS = {
    "fhn_pipeline": run_fhn,
    "fhn_nuisance_pipeline": run_fhn_nuisance,
    "vdp_pipeline": run_vdp,
    "vdp_nuisance_pipeline": run_vdp_nuisance,
    "vdp_scout": run_vdp_scout,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a supported experiment from a JSON config file.")
    parser.add_argument("config", type=str, help="Path to a JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    experiment = config.get("experiment")
    if experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            f"unsupported experiment '{experiment}'. Supported: {sorted(SUPPORTED_EXPERIMENTS.keys())}"
        )

    runner = SUPPORTED_EXPERIMENTS[experiment]
    kwargs = dict(config)
    kwargs.pop("experiment", None)
    if "outdir" in kwargs:
        kwargs["outdir"] = Path(kwargs["outdir"])
    if isinstance(kwargs.get("sim_config"), dict):
        kwargs["sim_config"] = SimulationConfig(**kwargs["sim_config"])
    runner(**kwargs)


if __name__ == "__main__":
    main()
