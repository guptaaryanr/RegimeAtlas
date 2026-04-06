# Structural Regimes Toolkit v1.0.0

This is the first clean repository-ready package for the structural-regimes paper.

## What changed from the internal phase-labelled tree

- removed phase-labelled experiment module names
- removed phase-labelled config directories and replaced them with top-level `configs/`
- removed phase-labelled summary filenames in the current code paths (`summary.json` is the standard)
- updated package and contract versioning to `v1`
- removed cached build artifacts and generated outputs from the repository package
- kept the supported paper and benchmark study entry points:
  - `configs/paper_suite.json`
  - `configs/paper_suite_smoke.json`
  - `configs/benchmark_suite.json`

## Packaging notes

This repository package is intentionally code-only.
Generated study outputs should be archived separately from the repository.

## Supported commands

```bash
python -m unittest discover -s tests -p "test_*.py" -v
pip install -e .
regime-toolkit run-study configs/paper_suite_smoke.json
regime-toolkit run-study configs/paper_suite.json
regime-toolkit run-study configs/benchmark_suite.json
```
