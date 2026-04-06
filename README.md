# Structural Regimes Toolkit

A minimal, non-ML Python toolkit for detecting **structural regime changes in deterministic dynamical systems from trajectories**.

This repository is the paper-freeze implementation for the structural-regimes study. The codebase emphasizes:

- canonical physical ODE benchmarks
- trajectory-only structural indicators
- oracle diagnostics separated from the core claim
- reproducible sweep studies with saved artifacts, manifests, and contract checks

## Scope

The current paper scope is:

- **primary claim**: structural-regime detection from **multivariate/full-state trajectories**
- **supplemental claim**: scalar delay-embedding robustness
- **benchmarks**: FitzHugh–Nagumo and autonomous forced van der Pol
- **method style**: no machine learning, no system identification, no black-box fitting

## Repository layout

```text
regime_toolkit/      core library
experiments/         runnable pipelines and study runners
configs/              supported paper and benchmark study configs
tests/               unit and smoke tests
.docs/               optional local docs cache (ignored)
docs/                reproduction and release notes
```

## Requirements

- Python 3.10+
- numpy
- scipy
- matplotlib

Install locally from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

Run the full test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Validate a study config:

```bash
regime-toolkit validate-config configs/paper_suite.json
```

Run the paper smoke study:

```bash
regime-toolkit run-study configs/paper_suite_smoke.json
```

Run the final paper study:

```bash
regime-toolkit run-study configs/paper_suite.json
```

Run the legacy benchmark-suite format, which is still supported:

```bash
regime-toolkit run-study configs/benchmark_suite.json
```

## Main paper artifacts

A successful paper study writes a directory like:

```text
outputs/paper_suite/
  study_index.json
  study_summary.csv
  study_manifest.json
  specificity_report.json
  ablation_report.json
  integrity_report.json
  contract_report.json
  fhn_main/
  fhn_nuisance/
  vdp_main/
  vdp_nuisance/
```

## Reproducibility contract

The paper-ready run is considered valid when all of the following are true:

- unit tests pass
- the paper study completes from `configs/paper_suite.json`
- `contract_report.json` passes
- `specificity_report.json` passes
- `ablation_report.json` passes
- `integrity_report.json` passes

See [docs/PAPER_REPRODUCTION.md](docs/PAPER_REPRODUCTION.md) for the exact freeze procedure.

## Packaging / repository notes

- `configs/` contains the supported study configs for the paper and benchmark suite
- generated outputs are ignored by default via `.gitignore`
- this repository does **not** assign a public license yet; choose one before public release
