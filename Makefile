.PHONY: test paper-smoke paper-suite benchmark-suite

test:
	python -m unittest discover -s tests -p "test_*.py" -v

paper-smoke:
	regime-toolkit run-study configs/paper_suite_smoke.json

paper-suite:
	regime-toolkit run-study configs/paper_suite.json

benchmark-suite:
	regime-toolkit run-study configs/benchmark_suite.json
