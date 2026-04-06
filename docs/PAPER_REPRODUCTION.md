# Paper Reproduction

This is the supported sequence for regenerating the paper evidence from a clean checkout.

## 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Run the tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## 3. Run the paper study

```bash
regime-toolkit run-study configs/paper_suite.json
```

## 4. Verify the study contract

Expected top-level outputs:

- `outputs/paper_suite/study_index.json`
- `outputs/paper_suite/study_summary.csv`
- `outputs/paper_suite/study_manifest.json`
- `outputs/paper_suite/specificity_report.json`
- `outputs/paper_suite/ablation_report.json`
- `outputs/paper_suite/integrity_report.json`
- `outputs/paper_suite/contract_report.json`

Quick validation commands:

```bash
regime-toolkit validate-manifest outputs/paper_suite/study_manifest.json
python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/paper_suite')
for name in ['specificity_report.json', 'ablation_report.json', 'integrity_report.json', 'contract_report.json']:
    payload = json.loads((root / name).read_text())
    print(name, payload.get('passed'))
PY
```

## 5. Archive the evidence

Archive the whole `outputs/paper_suite/` directory together with the exact commit hash and test log used for the paper.
