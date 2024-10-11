VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
SHELL := /bin/bash

all: install
	@$(PYTHON) -m logreg_predict ../datasets/dataset_test.csv evaluation_logs.json

train: install
	@$(PYTHON) -m logreg_train ../datasets/dataset_train.csv

describe: install
	@$(PYTHON) -m data_analysis.main ../datasets/dataset_train.csv

histogram: install
	@$(PYTHON) -m data_visualization.histogram

scatter_plot: install
	@$(PYTHON) -m data_visualization.scatter_plot

pair_plot: install
	@$(PYTHON) -m data_visualization.pair_plot

install: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

clean-venv:
	rm -rf $(VENV)

re: clean clean-venv all

.PHONY: install describe histogram scatter_plot pair_plot clean clean-venv re

# .PHONY: test
# test:
# 	$(PYTHON) -m unittest discover tests

