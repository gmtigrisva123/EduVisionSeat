# Convenience targets. Everything runs from the repository root.
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install install-dev test lint format check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install runtime + development dependencies
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Check style and common defects
	$(PYTHON) -m ruff check .

format: ## Reformat the code
	$(PYTHON) -m ruff format .

check: lint test ## Run everything CI would run

clean: ## Remove caches and bytecode
	find . -type d -name __pycache__ -not -path "./venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
