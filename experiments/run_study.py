from __future__ import annotations

import sys
from pathlib import Path as _PathBootstrap

REPO_ROOT = _PathBootstrap(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from pathlib import Path

from regime_toolkit.contracts import default_paper_study_contract
from regime_toolkit.study import run_study_suite
from regime_toolkit.schema import normalize_study_config
from experiments.fhn_pipeline import run_pipeline as run_fhn
from experiments.fhn_nuisance_pipeline import run_pipeline as run_fhn_nuisance
from experiments.vdp_pipeline import run_pipeline as run_vdp
from experiments.vdp_nuisance_pipeline import run_pipeline as run_vdp_nuisance

RUNNER_MAP = {
    "fhn_pipeline": run_fhn,
    "fhn_nuisance_pipeline": run_fhn_nuisance,
    "vdp_pipeline": run_vdp,
    "vdp_nuisance_pipeline": run_vdp_nuisance,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a supported study suite from a JSON config.")
    parser.add_argument("config", type=str, help="Path to a JSON study or benchmark-suite config")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)
    config = normalize_study_config(raw_config)

    contract = default_paper_study_contract()
    result = run_study_suite(
        study_name=str(config["study_name"]),
        outdir=Path(config["outdir"]),
        runs=list(config["runs"]),
        runner_map=RUNNER_MAP,
        contract=contract,
    )
    outdir = Path(config["outdir"])
    print(f"\nSaved study outputs to: {outdir.resolve()}")
    print(f"Specificity passed: {result['specificity_report']['passed']}")
    print(f"Ablation passed: {result['ablation_report']['passed']}")
    print(f"Integrity passed: {result['integrity_report']['passed']}")
    print(f"Contract passed: {result['contract_report']['passed']}")


if __name__ == "__main__":
    main()
