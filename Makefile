.PHONY: format lint test clean install pre-commit build

# Install dependencies
install:
	poetry install

# Format code
format:
	poetry run black .
	poetry run ruff . --fix

# Run linting
lint:
	poetry run black . --check
	poetry run ruff .
	poetry run mypy .

# Run tests
test:
	poetry run pytest

# Clean build artifacts
clean:
	rm -rf dist/
	rm -rf build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +

# Run pre-commit on all files
pre-commit:
	poetry run pre-commit run --all-files

# Build package
build:
	poetry build

# Export requirements
requirements:
	poetry export -f requirements.txt --output requirements.txt
	poetry export -f requirements.txt --with dev --output requirements-dev.txt
