# CI/CD Pipeline

This project uses GitHub Actions for Continuous Integration and Continuous Deployment.

## Pipeline Overview

The CI/CD pipeline consists of the following jobs:

1. **Lint** - Runs code quality checks
2. **Test** - Runs unit tests
3. **Build and Publish Package** - Builds and publishes the Python package to GitHub Registry
4. **Build and Publish Docker** - Builds and publishes the Docker image to GitHub Container Registry
5. **Deploy GitHub Pages** - Deploys documentation to GitHub Pages

## Pipeline Triggers

The pipeline is triggered by:

- **Push to main** - Runs linting, testing, and GitHub Pages deployment
- **Pull Requests to main** - Runs linting and testing
- **Tags starting with v*** - Runs the complete pipeline, including package and Docker image publishing

## GitHub Actions Workflow

The workflow is defined in `.github/workflows/ci.yml`.

### Linting and Testing

```yaml
lint:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install Poetry
      uses: snok/install-poetry@v1
    - name: Install dependencies
      run: poetry install --no-interaction --no-root
    - name: Run Black
      run: poetry run black . --check
    - name: Run Ruff
      run: poetry run ruff .
    - name: Run MyPy
      run: poetry run mypy .
```

### Docker Image Publishing

The workflow builds and publishes the Docker image to GitHub Container Registry when tags are pushed:

```bash
# Tag a new version
git tag v0.1.0
git push origin v0.1.0
```
