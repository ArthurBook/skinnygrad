.PHONY: all

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort| while read -r line; do \
		printf "\033[1;32m$$(echo $$line | cut -f 1 -d':')\033[00m:$$(echo $$line | cut -f 2- -d'#')\n"; \
	done

.PHONY: setup
setup: # Installs project dependencies including all extras with Poetry.
	@brew_prefix_graphviz=$$(brew --prefix graphviz); \
	poetry run pip install --config-settings="--global-option=build_ext" \
		--config-settings="--global-option=-I$${brew_prefix_graphviz}/include/" \
		--config-settings="--global-option=-L$${brew_prefix_graphviz}/lib/" \
		pygraphviz && \
	poetry install --all-extras

.PHONY: fmt
fmt: # Formats the code in the src/ and tests/ directories using black.
	poetry run isort src/ tests/ 
	poetry run black src/ tests/ 

.PHONY: lint
lint: # Lints the code in the src/ and tests/ directories using pylint.
	poetry run pylint src/ tests/ --rcfile=pyproject.toml src/skinnygrad/

.PHONY: typecheck
typecheck: # Performs type checking in the src/ and tests/ directories using mypy.
	poetry run mypy src/ tests/

.PHONY: test
test: # Runs automated tests using pytest.
	SKINNYGRAD_LOGLEVEL=DEBUG PYTHONPATH=./src/:${PYTHONPATH} poetry run pytest

.PHONY: linecount
linecount: # Counts the number of lines of Python code in the src/ directory.
	find src/ -name '*.py' -exec wc -l {} +
