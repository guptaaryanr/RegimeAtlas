from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ._version import __version__
from .integrity import validate_manifest_file
from .schema import normalize_study_config
from .simulate import SimulationConfig


def _run_study(args: argparse.Namespace) -> int:
    from experiments.run_study import RUNNER_MAP
    from regime_toolkit.contracts import default_paper_study_contract
    from regime_toolkit.study import run_study_suite

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    config = normalize_study_config(raw)
    run_study_suite(
        study_name=str(config["study_name"]),
        outdir=Path(config["outdir"]),
        runs=list(config["runs"]),
        runner_map=RUNNER_MAP,
        contract=default_paper_study_contract(),
    )
    return 0


def _run_config(args: argparse.Namespace) -> int:
    from experiments.run_config import SUPPORTED_EXPERIMENTS

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
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    config = normalize_study_config(raw)
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


def _validate_manifest(args: argparse.Namespace) -> int:
    report = validate_manifest_file(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regime-toolkit", description="Structural regime toolkit CLI.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate_config = sub.add_parser("validate-config", help="Normalize and validate a study config.")
    p_validate_config.add_argument("config", type=str)
    p_validate_config.set_defaults(func=_validate_config)

    p_validate_manifest = sub.add_parser("validate-manifest", help="Validate a saved artifact manifest.")
    p_validate_manifest.add_argument("manifest", type=str)
    p_validate_manifest.set_defaults(func=_validate_manifest)

    p_run_config = sub.add_parser("run-config", help="Run a single experiment from a JSON config.")
    p_run_config.add_argument("config", type=str)
    p_run_config.set_defaults(func=_run_config)

    p_run_study = sub.add_parser("run-study", help="Run a full study suite from a JSON config.")
    p_run_study.add_argument("config", type=str)
    p_run_study.set_defaults(func=_run_study)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
