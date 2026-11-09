SHELL := /bin/bash
.PHONY: all env-check fast check lint format test

all: fast

env-check:
	@python -c "import os, sys; \
cond_env = os.environ.get('CONDA_DEFAULT_ENV', ''); \
ver = f'{sys.version_info.major}.{sys.version_info.minor}'; \
errors = []; \
ver != '3.11' and errors.append(f'Python version must be 3.11, found {ver} ({sys.version})'); \
cond_env != 'gnani' and errors.append(f'CONDA_DEFAULT_ENV must be \"gnani\", found \"{cond_env}\"'); \
errors and (print('ENVIRONMENT ERROR: ' + ' | '.join(errors), file=sys.stderr), sys.exit(1))"

fast: env-check
	python tools/timed.py 15 pytest -m "not slow and not live" --disable-socket --timeout=5

check: env-check
	ruff check .
	ruff format --check .
	lint-imports
	mypy --strict overlap tests tools
	pytest -m "not live" --disable-socket --timeout=5
	python tools/dag_lint.py validate

lint: env-check
	ruff check .
	ruff format --check .
	lint-imports
	mypy --strict overlap tests tools

format: env-check
	ruff check --fix .
	ruff format .

test: fast
