PY ?= .venv/bin/python

.PHONY: install test lint all demo notebook clean

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[data,dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

# Full back-test on Yahoo Finance data (downloads once, then uses the cache)
all:
	$(PY) scripts/run_backtest.py --source yahoo

# Same engine on a simulated portfolio: no network needed
demo:
	$(PY) scripts/run_backtest.py --source synthetic

notebook:
	$(PY) scripts/build_notebook.py

clean:
	rm -rf reports/figures/*.png reports/*.md reports/*.json
